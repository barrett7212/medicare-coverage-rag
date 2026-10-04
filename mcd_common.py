"""Shared helpers for the MCD pipeline.

* connect()        - psycopg connection with search_path = mcd, public
* html_to_text()   - turns the HTML stored in LCD/Article fields into readable text
                     (tables -> "a | b | c" rows, lists -> "- item", bold-only short
                     paragraphs -> "### Heading" so the chunker can split on them)
* to_date/to_ts/to_int/to_bool/ch1 - tolerant value parsers (source nulls are '')
* Source / discover_sources() - read tables from an .mdb (via mdbtools) or a CSV directory
"""
from __future__ import annotations

import csv
import io
import os
import re
import shutil
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

import psycopg
from bs4 import BeautifulSoup, NavigableString

try:  # lxml is faster; fall back to the stdlib parser
    import lxml  # noqa: F401
    _PARSER = "lxml"
except ImportError:  # pragma: no cover
    _PARSER = "html.parser"

DSN = os.environ.get("MCD_DSN", "postgresql://localhost/mcd")
DATA_DIR = Path(os.environ.get("MCD_DATA_DIR", "data"))
SCHEMA_FILE = Path(__file__).with_name("mcd_schema.sql")

csv.field_size_limit(sys.maxsize)


# --------------------------------------------------------------------------- DB
def connect(dsn: Optional[str] = None) -> psycopg.Connection:
    conn = psycopg.connect(dsn or DSN)
    conn.execute("SET search_path = mcd, public")
    return conn


def schema_exists(conn: psycopg.Connection) -> bool:
    return conn.execute("SELECT to_regclass('mcd.policy_doc') IS NOT NULL").fetchone()[0]


def init_schema(conn: psycopg.Connection, reset: bool = False) -> None:
    if reset:
        conn.execute("DROP SCHEMA IF EXISTS mcd CASCADE")
    conn.execute(SCHEMA_FILE.read_text())
    conn.execute("SET search_path = mcd, public")
    conn.commit()


# ------------------------------------------------------------------ value parsing
_DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y",
    "%m/%d/%y %H:%M:%S", "%m/%d/%y", "%Y-%m-%dT%H:%M:%S",
)


def _parse_dt(v: Optional[str]) -> Optional[datetime]:
    if not v:
        return None
    v = v.strip()
    if not v:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(v, fmt)
        except ValueError:
            continue
    return None


def to_date(v: Optional[str]) -> Optional[date]:
    d = _parse_dt(v)
    return d.date() if d else None


def to_ts(v: Optional[str]) -> Optional[datetime]:
    d = _parse_dt(v)
    return d.replace(tzinfo=timezone.utc) if d else None  # MCD datetimes are UTC


def to_int(v: Optional[str]) -> Optional[int]:
    if v is None:
        return None
    v = v.strip()
    if not v:
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def to_bool(v: Optional[str]) -> Optional[bool]:
    if v is None:
        return None
    s = v.strip().lower()
    if s in ("1", "-1", "true", "t", "y", "yes"):
        return True
    if s in ("0", "false", "f", "n", "no"):
        return False
    return None


def ch1(v: Optional[str]) -> Optional[str]:
    """Single-character flag column; blank/space -> NULL."""
    if v is None:
        return None
    v = v.strip()
    return v[:1] if v else None


def clean_str(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    v = v.replace("\x00", "").strip()
    return v or None


def clean_url(v: Optional[str]) -> Optional[str]:
    v = clean_str(v)
    return None if v in (None, "http://", "https://") else v


# ---------------------------------------------------------------------- HTML -> text
_WS_RE = re.compile(r"[ \t\r\f\v]+")


def _compact(s: str) -> str:
    return _WS_RE.sub(" ", s.replace("\xa0", " ").replace("\u200b", "")).strip()


def _is_pseudo_heading(p) -> bool:
    """Short paragraph that is (almost) entirely bold - LCDs use these as headings."""
    txt = p.get_text(" ", strip=True)
    if not txt or len(txt) > 100 or txt.endswith((".", ";", ",")):
        return False
    bold = "".join(t.get_text("") for t in p.find_all(["strong", "b"]))
    n_all = len(re.sub(r"\s+", "", txt))
    n_bold = len(re.sub(r"\s+", "", bold))
    return n_all > 0 and n_bold >= 0.9 * n_all


def html_to_text(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    raw = raw.replace("\x00", "").strip()
    if not raw:
        return None
    if "<" not in raw and "&" not in raw:
        return _normalize_lines(raw) or None

    soup = BeautifulSoup(raw, _PARSER)
    for t in soup(["script", "style"]):
        t.decompose()

    # 1. tables -> pipe-delimited rows (deepest first so nested tables flatten)
    for table in reversed(soup.find_all("table")):
        rows = []
        for tr in table.find_all("tr"):
            cells = [_compact(c.get_text(" ")) for c in tr.find_all(["td", "th"])]
            if any(cells):
                rows.append(" | ".join(cells))
        table.replace_with(NavigableString("\n\n" + "\n".join(rows) + "\n\n"))

    # 2. bold-only short paragraphs and real headings -> "### Heading"
    for p in soup.find_all("p"):
        if _is_pseudo_heading(p):
            p.replace_with(NavigableString("\n\n### " + _compact(p.get_text(" ")) + "\n\n"))
    for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        h.replace_with(NavigableString("\n\n### " + _compact(h.get_text(" ")) + "\n\n"))

    # 3. list items (deepest first so nested lists keep their text)
    for li in reversed(soup.find_all("li")):
        li.replace_with(NavigableString("\n- " + _compact(li.get_text(" ")) + "\n"))

    # 4. line breaks and block containers
    for br in soup.find_all("br"):
        br.replace_with(NavigableString("\n"))
    for blk in soup.find_all(["p", "div", "blockquote", "pre", "ul", "ol", "tr", "section"]):
        blk.insert_before(NavigableString("\n\n"))
        blk.insert_after(NavigableString("\n\n"))
        blk.unwrap()

    return _normalize_lines(soup.get_text()) or None


def _normalize_lines(text: str) -> str:
    lines = [_compact(l) for l in text.replace("\r", "").split("\n")]
    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"(?m)^(- .*)\n\n(?=- )", r"\1\n", out)   # keep consecutive list items together
    return out.strip()


# ------------------------------------------------------------------- source tables
class Source:
    """A dataset on disk: one .mdb file (via mdbtools) or a directory of CSV files."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.kind = "mdb" if self.path.suffix.lower() == ".mdb" else "csv"
        self._tables: Optional[list[str]] = None
        if self.kind == "mdb" and not shutil.which("mdb-export"):
            raise RuntimeError(
                "mdbtools not found. Install it (macOS: `brew install mdbtools`, "
                "Ubuntu: `apt install mdbtools`) or point the pipeline at the CSV directory."
            )

    def __repr__(self) -> str:
        return f"Source({self.path})"

    def tables(self) -> list[str]:
        if self._tables is None:
            if self.kind == "mdb":
                out = subprocess.run(["mdb-tables", "-1", str(self.path)],
                                     capture_output=True, text=True, check=True).stdout
                self._tables = [t.strip().lower() for t in out.splitlines() if t.strip()]
            else:
                self._tables = sorted(p.stem.lower() for p in self.path.rglob("*.csv"))
        return self._tables

    def has(self, table: str) -> bool:
        return table.lower() in self.tables()

    def rows(self, table: str) -> Iterator[dict[str, Optional[str]]]:
        """Yield rows as {lower_case_column: str | None}. Empty strings become None."""
        table = table.lower()
        if self.kind == "mdb":
            # -D/-T make dates unambiguous (default is 2-digit years)
            proc = subprocess.Popen(
                ["mdb-export", "-D", "%Y-%m-%d", "-T", "%Y-%m-%d %H:%M:%S", str(self.path), table],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stream = io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace", newline="")
            try:
                yield from self._dict_rows(stream)
            finally:
                stream.close()
                err = proc.stderr.read().decode("utf-8", "replace").strip()
                if proc.wait() != 0:
                    raise RuntimeError(f"mdb-export {table} failed: {err}")
        else:
            match = [p for p in self.path.rglob("*.csv") if p.stem.lower() == table]
            if not match:
                return
            try:
                with open(match[0], newline="", encoding="utf-8-sig") as f:
                    yield from self._dict_rows(f)
            except UnicodeDecodeError:
                with open(match[0], newline="", encoding="cp1252") as f:
                    yield from self._dict_rows(f)

    @staticmethod
    def _dict_rows(stream) -> Iterator[dict[str, Optional[str]]]:
        reader = csv.reader(stream)
        try:
            header = [h.strip().lower() for h in next(reader)]
        except StopIteration:
            return
        n = len(header)
        for rec in reader:
            if not rec:
                continue
            if len(rec) < n:
                rec = rec + [""] * (n - len(rec))
            yield {header[i]: (rec[i] if rec[i] != "" else None) for i in range(n)}


def discover_sources(paths: Iterable[Path]) -> dict[str, Source]:
    """Find the LCD / Article / NCD datasets under the given files or directories.

    Classification is by which tables a dataset contains, not by file name.
    If several datasets of one kind are found the newest file wins (with a warning).
    """
    candidates: list[Source] = []
    for p in map(Path, paths):
        if p.is_file() and p.suffix.lower() == ".mdb":
            candidates.append(Source(p))
        elif p.is_dir():
            mdbs = sorted(p.rglob("*.mdb")) + sorted(p.rglob("*.MDB"))
            if mdbs:
                candidates.extend(Source(m) for m in dict.fromkeys(mdbs))
            else:  # directories holding CSV exports
                for d in sorted({c.parent for c in p.rglob("*.csv")}):
                    candidates.append(Source(d))
    found: dict[str, list[Source]] = {}
    for s in candidates:
        t = set(s.tables())
        if "lcd" in t:
            found.setdefault("lcd", []).append(s)
        if "article" in t:
            found.setdefault("article", []).append(s)
        if "ncd_trkg" in t:
            found.setdefault("ncd", []).append(s)
    out: dict[str, Source] = {}
    for kind, lst in found.items():
        lst.sort(key=lambda s: s.path.stat().st_mtime)
        if len(lst) > 1:
            print(f"warning: {len(lst)} {kind} datasets found; using newest: {lst[-1].path}", file=sys.stderr)
        out[kind] = lst[-1]
    return out
