#!/usr/bin/env python3
"""Fetch: latest CMS MCD datasets (LCD, Article, NCD) -> ./data, ready for mcd_etl.py.

Typical use
    python mcd_fetch.py --accept-licenses              # current LCDs, Articles and NCDs
    python mcd_fetch.py --accept-licenses --retired    # "Current and Retired" LCD/Article variants
    python mcd_fetch.py --accept-licenses --only ncd   # just one dataset
    python mcd_fetch.py --list                         # show what the downloads page offers

The download links are scraped from the CMS downloads page, so the "Data as of" dates it shows
are reported too. Each ZIP holds an Access .mdb plus a nested ZIP of CSVs; only the one the ETL
will read is unpacked:
    data/current_lcd.mdb            --format mdb (default when mdbtools is installed)
    data/current_lcd_csv/*.csv      --format csv (default otherwise)

Re-running is cheap and designed for the weekly refresh: the ETag of every download is kept in
data/.mcd_fetch.json and compared with a HEAD request, so unchanged datasets are skipped.

robots.txt is read from every host before anything else is requested from it (cms.gov for the
page, downloads.cms.gov for the ZIPs) and its Allow/Disallow rules and Crawl-delay are obeyed,
redirects included. A URL it disallows is reported as an error and never requested.

The datasets embed CPT (AMA), CDT (ADA) and UB-04 (AHA/NUBC) content. CMS makes you accept those
license agreements on the downloads page before downloading; --accept-licenses is that click.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from mcd_common import DATA_DIR

PAGE_URL = "https://www.cms.gov/medicare-coverage-database/downloads/downloads.aspx"
EXPORTS_URL = "https://downloads.cms.gov/medicare-coverage-database/downloads/exports/"
# kind -> ZIP file name on the downloads page: (current only, current and retired)
DATASETS = {
    "lcd": ("current_lcd.zip", "all_lcd.zip"),
    "article": ("current_article.zip", "all_article.zip"),
    "ncd": ("ncd.zip", "ncd.zip"),
}
MANIFEST = ".mcd_fetch.json"
USER_AGENT = "mcd-rag-fetch/1.0 (python-requests)"
ROBOTS_TOKEN = USER_AGENT.split("/")[0]  # the name robots.txt can address this script by
TIMEOUT = (15, 120)  # connect, read
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------------- robots.txt
class RobotsDisallowed(requests.RequestException):
    """robots.txt does not allow this URL to be fetched."""


def parse_robots(text: str, token: str) -> tuple[list[tuple[str, bool]], Optional[float]]:
    """([(path pattern, allowed)], crawl delay) from the robots.txt group for token, else for "*".

    Parsed here (per RFC 9309) because urllib.robotparser ignores the * and $ wildcards that
    cms.gov's rules rely on, and takes the first matching rule rather than the most specific.
    """
    groups: dict[str, dict] = {}
    agents: list[str] = []
    in_rules = False
    for line in text.splitlines():
        key, sep, value = line.split("#", 1)[0].partition(":")
        key, value = key.strip().lower(), value.strip()
        if not sep:
            continue
        if key == "user-agent":
            if in_rules:  # a User-agent line after rules starts the next group
                agents, in_rules = [], False
            agents.append(value.lower())
            groups.setdefault(value.lower(), {"rules": [], "delay": None})
        elif key in ("allow", "disallow", "crawl-delay"):
            in_rules = True
            for agent in agents:
                if key == "crawl-delay":
                    try:
                        groups[agent]["delay"] = float(value)
                    except ValueError:
                        pass
                elif value:  # an empty Disallow restricts nothing
                    groups[agent]["rules"].append((value, key == "allow"))
    group = groups.get(token.lower()) or groups.get("*") or {"rules": [], "delay": None}
    return group["rules"], group["delay"]


def robots_allows(rules: list[tuple[str, bool]], path: str) -> bool:
    """Apply the longest matching rule to path (with its query string); Allow wins a tie."""
    best, verdict = -1, True
    for pattern, allow in rules:
        rx = ".*".join(re.escape(p) for p in pattern.rstrip("$").split("*"))
        if pattern.endswith("$"):
            rx += "$"
        if re.match(rx, path) and (len(pattern) > best or (len(pattern) == best and allow)):
            best, verdict = len(pattern), allow
    return verdict


class RobotsSession(requests.Session):
    """A Session that obeys each host's robots.txt.

    Enforced in send() rather than get()/head() so that every hop of a redirect is checked too.
    """

    def __init__(self) -> None:
        super().__init__()
        self.headers["User-Agent"] = USER_AGENT
        self._robots: dict[str, tuple[list[tuple[str, bool]], Optional[float]]] = {}
        self._last: dict[str, float] = {}  # origin -> when it was last sent a request

    def _policy(self, origin: str) -> tuple[list[tuple[str, bool]], Optional[float]]:
        if origin not in self._robots:
            url = origin + "/robots.txt"
            try:
                status = (r := self.get(url, timeout=TIMEOUT)).status_code
            except requests.RequestException as e:
                status = e
            if status == 200:
                policy = parse_robots(r.text, ROBOTS_TOKEN)
            elif isinstance(status, int) and 400 <= status < 500 and status != 429:
                policy = ([], None)  # no robots.txt: nothing is restricted
            else:  # unreachable: RFC 9309 says to assume everything is disallowed
                print(f"warning: could not read {url} ({status}); treating {origin} as off limits",
                      file=sys.stderr)
                policy = ([("/", False)], None)
            self._robots[origin] = policy
        return self._robots[origin]

    def send(self, request, **kwargs):
        u = urlparse(request.url)
        if u.path == "/robots.txt":
            return super().send(request, **kwargs)
        origin = f"{u.scheme}://{u.netloc}"
        rules, delay = self._policy(origin)
        if not robots_allows(rules, (u.path or "/") + (f"?{u.query}" if u.query else "")):
            raise RobotsDisallowed(f"{origin}/robots.txt disallows {request.url}")
        if delay and origin in self._last:
            time.sleep(max(0.0, self._last[origin] + delay - time.monotonic()))
        try:
            return super().send(request, **kwargs)
        finally:
            self._last[origin] = time.monotonic()


# ------------------------------------------------------------------- downloads page
def scrape_page(session: requests.Session) -> dict[str, dict]:
    """{zip file name: {url, label, data_as_of, released}} for every ZIP on the downloads page."""
    r = session.get(PAGE_URL, timeout=TIMEOUT)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    out: dict[str, dict] = {}
    for a in soup.find_all("a", href=True):
        url = urljoin(PAGE_URL, a["href"])
        name = Path(urlparse(url).path).name.lower()
        if not name.endswith(".zip") or name in out:
            continue
        info = {"url": url, "label": a.get("download") or a.get_text(" ", strip=True),
                "data_as_of": None, "released": None}
        tr = a.find_parent("tr")
        if tr:  # Dataset | Description | Data as of | Released
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if len(cells) >= 4:
                info["data_as_of"], info["released"] = cells[2] or None, cells[3] or None
        out[name] = info
    return out


def resolve(session: requests.Session, kinds: list[str], retired: bool) -> dict[str, dict]:
    """Pick the download for each kind; fall back to the well-known URL if the page changed."""
    try:
        page = scrape_page(session)
    except requests.RequestException as e:
        print(f"warning: could not read the downloads page ({e}); using default URLs", file=sys.stderr)
        page = {}
    out = {}
    for kind in kinds:
        name = DATASETS[kind][1 if retired else 0]
        if name not in page and page:
            print(f"warning: {name} is not linked on the downloads page; trying {EXPORTS_URL}{name}",
                  file=sys.stderr)
        out[kind] = page.get(name) or {"url": EXPORTS_URL + name, "label": name,
                                       "data_as_of": None, "released": None}
    return out


# ------------------------------------------------------------------------- manifest
def load_manifest(dest: Path) -> dict:
    try:
        return json.loads((dest / MANIFEST).read_text())
    except (OSError, ValueError):
        return {}


def save_manifest(dest: Path, manifest: dict) -> None:
    tmp = dest / (MANIFEST + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, dest / MANIFEST)


def remove_output(dest: Path, name: Optional[str]) -> None:
    """Delete something an earlier run unpacked (only ever a direct child of dest)."""
    if not name or Path(name).name != name:
        return
    p = dest / name
    if p.is_dir():
        shutil.rmtree(p)
    elif p.exists():
        p.unlink()


# ------------------------------------------------------------------ download + unpack
def unchanged(session: requests.Session, url: str, prev: dict) -> bool:
    """True if the server still has the file an earlier run downloaded.

    Compared here with a HEAD because downloads.cms.gov ignores If-None-Match.
    """
    try:
        r = session.head(url, allow_redirects=True, timeout=TIMEOUT)
        r.raise_for_status()
    except requests.RequestException:
        return False  # let the GET report the real problem
    for key, header in (("etag", "ETag"), ("last_modified", "Last-Modified")):
        if prev.get(key) and r.headers.get(header):
            return prev[key] == r.headers[header]
    return False


def download(session: requests.Session, url: str, part: Path) -> dict:
    """Stream url into part. Returns the response's cache headers."""
    with session.get(url, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        done, next_report = 0, 0.25
        with open(part, "wb") as f:
            for block in r.iter_content(chunk_size=1 << 20):
                f.write(block)
                done += len(block)
                if total and done / total >= next_report and done < total:
                    log(f"    {done / 1e6:6.1f} / {total / 1e6:.1f} MB")
                    next_report += 0.25
        if total and done != total:
            raise IOError(f"incomplete download: got {done} of {total} bytes")
        return {"etag": r.headers.get("ETag"), "last_modified": r.headers.get("Last-Modified"),
                "bytes": done}


def unpack(zip_path: Path, dest: Path, fmt: str) -> str:
    """Unpack the .mdb, or the nested CSV ZIP, from a downloaded dataset. Returns the output name.

    Members are written under their base name only, so a hostile archive can't escape dest.
    """
    with zipfile.ZipFile(zip_path) as z:
        files = [i for i in z.infolist() if not i.is_dir()]
        if fmt == "mdb":
            mdbs = [i for i in files if i.filename.lower().endswith(".mdb")]
            if len(mdbs) != 1:
                raise ValueError(f"expected one .mdb in the download, found {len(mdbs)}")
            name = Path(mdbs[0].filename).name
            tmp = dest / (name + ".part")
            with z.open(mdbs[0]) as src, open(tmp, "wb") as out:
                shutil.copyfileobj(src, out, 1 << 20)
            os.replace(tmp, dest / name)
            return name

        inner = [i for i in files if i.filename.lower().endswith(".zip")]
        if len(inner) > 1:
            raise ValueError(f"expected one CSV .zip in the download, found {len(inner)}")
        name = zip_path.name.split(".")[0] + "_csv"
        tmp = dest / (name + ".part")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        # CSVs are normally in a nested ZIP; accept them at the top level too
        csv_zip = zipfile.ZipFile(io.BytesIO(z.read(inner[0]))) if inner else z
        n = 0
        for i in csv_zip.infolist():
            if i.is_dir() or not i.filename.lower().endswith(".csv"):
                continue
            with csv_zip.open(i) as src, open(tmp / Path(i.filename).name, "wb") as out:
                shutil.copyfileobj(src, out, 1 << 20)
            n += 1
        if not n:
            shutil.rmtree(tmp)
            raise ValueError("no CSV files found in the download")
        if (dest / name).exists():
            shutil.rmtree(dest / name)
        os.replace(tmp, dest / name)
        return name


def fetch_one(session: requests.Session, kind: str, info: dict, dest: Path, fmt: str,
              manifest: dict, force: bool) -> bool:
    """Download and unpack one dataset. Returns True if it changed."""
    url = info["url"]
    prev = manifest.get(kind) or {}
    # the ETag only counts if it describes what is on disk right now
    reusable = (not force and prev.get("url") == url and prev.get("format") == fmt
                and prev.get("output") and (dest / prev["output"]).exists())
    as_of = f" (data as of {info['data_as_of']})" if info.get("data_as_of") else ""
    log(f"{kind:8s} <- {url}{as_of}")

    if reusable and unchanged(session, url, prev):
        log(f"{kind:8s}    unchanged, keeping {dest / prev['output']}")
        return False
    part = dest / (Path(urlparse(url).path).name + ".part")
    try:
        meta = download(session, url, part)
        output = unpack(part, dest, fmt)
    finally:
        part.unlink(missing_ok=True)
    # a leftover from another variant/format would otherwise compete in discover_sources()
    if prev.get("output") != output:
        remove_output(dest, prev.get("output"))
    manifest[kind] = {**meta, "url": url, "format": fmt, "output": output,
                      "data_as_of": info.get("data_as_of"), "released": info.get("released"),
                      "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    save_manifest(dest, manifest)
    log(f"{kind:8s} -> {dest / output} ({meta['bytes'] / 1e6:.1f} MB download)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accept-licenses", action="store_true",
                    help="you have read and accept the AMA (CPT), ADA (CDT) and AHA/NUBC (UB-04) "
                         f"license agreements shown at {PAGE_URL}")
    ap.add_argument("--dest", type=Path, default=DATA_DIR,
                    help="where to put the datasets (default: $MCD_DATA_DIR or ./data)")
    ap.add_argument("--only", nargs="+", choices=list(DATASETS), help="fetch only these datasets")
    ap.add_argument("--retired", action="store_true",
                    help='fetch the "Current and Retired" LCD and Article datasets instead of current only')
    ap.add_argument("--format", choices=["mdb", "csv"], default=None,
                    help="which copy of the data to unpack (default: mdb if mdbtools is installed, else csv)")
    ap.add_argument("--force", action="store_true", help="download even if the ETag is unchanged")
    ap.add_argument("--list", action="store_true", help="list the datasets on the downloads page and exit")
    args = ap.parse_args()

    session = RobotsSession()

    if args.list:
        for name, info in scrape_page(session).items():
            print(f"{name:22s} as of {info['data_as_of'] or '?':10s}  released {info['released'] or '?':10s}  "
                  f"{info['label']}")
        return 0

    if not args.accept_licenses:
        print("The MCD datasets embed CPT (AMA), CDT (ADA) and UB-04 (AHA/NUBC) content.\n"
              f"Read the license agreements at {PAGE_URL}\n"
              "and re-run with --accept-licenses if you accept them.", file=sys.stderr)
        return 2

    fmt = args.format or ("mdb" if shutil.which("mdb-export") else "csv")
    if not args.format and fmt == "csv":
        log("mdbtools not found; unpacking the CSV copies instead of the .mdb files")
    args.dest.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.dest)
    kinds = [k for k in DATASETS if not args.only or k in args.only]
    targets = resolve(session, kinds, args.retired)

    changed, failed = 0, []
    for kind in kinds:
        try:
            changed += fetch_one(session, kind, targets[kind], args.dest, fmt, manifest, args.force)
        except (requests.RequestException, zipfile.BadZipFile, OSError, ValueError) as e:
            print(f"error: {kind}: {e}", file=sys.stderr)
            failed.append(kind)
    log(f"done: {changed} updated, {len(kinds) - changed - len(failed)} unchanged, {len(failed)} failed")
    if failed:
        return 1
    if changed:
        log("next: python mcd_etl.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
