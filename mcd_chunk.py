#!/usr/bin/env python3
"""Chunk MCD policy text into doc_chunk rows (embedding happens in mcd_embed.py).

    python mcd_chunk.py                       # active, non-draft LCDs / Articles / NCDs
    python mcd_chunk.py --show L33252         # print the chunks for one document, write nothing
    python mcd_chunk.py --types LCD --limit 20 --dry-run

What becomes a chunk
  overview      one per document: id, title, type, status, dates, MACs, states, related documents
  text sections each narrative field (indication, coding guidelines, article text, NCD
                indications, ...) split on headings, then packed to ~--target-chars with a
                one-paragraph overlap
  code sections the introductory paragraph of every code group, plus a capped, compact
                "CODE - description" listing (exact code lookups belong in SQL - policy_code -
                not in the vector index, so listings are capped at --max-code-chunks per group)
  other         sticky notes, revision history, "other coding" blocks (+ comments on request)

The run is incremental: each chunk carries a sha256 of (heading_path + content). Unchanged
chunks keep their row and embedding; changed chunks lose their embedding; vanished chunks are
deleted. heading_path is stored separately from content and is prepended at embedding time.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

from psycopg.rows import dict_row

from mcd_common import connect

DOC_LABEL = {"LCD": "Local Coverage Determination (LCD)",
             "ART": "Local Coverage Article",
             "NCD": "National Coverage Determination (NCD)"}

# (column, heading label, section name == column). Order = order of appearance in the document.
TEXT_SECTIONS: dict[str, list[tuple[str, str]]] = {
    "LCD": [
        ("issue", "Issue"),
        ("cms_cov_policy", "CMS National Coverage Policy"),     # lives on policy_doc
        ("indication", "Coverage Indications, Limitations and/or Medical Necessity"),
        ("diagnoses_support", "Diagnoses that Support Medical Necessity"),
        ("diagnoses_dont_support", "Diagnoses that Do Not Support Medical Necessity"),
        ("coding_guidelines", "Coding Guidelines"),
        ("doc_reqs", "Documentation Requirements"),
        ("util_guide", "Utilization Guidelines"),
        ("appendices", "Appendices"),
        ("associated_info", "Associated Information"),
        ("summary_of_evidence", "Summary of Evidence"),
        ("analysis_of_evidence", "Analysis of Evidence"),
        ("issue_change", "Summary of Changes to the Issue"),
        ("synopsis_changes", "Synopsis of Changes"),
        ("add_icd10_info", "Additional ICD-10 Information"),
        ("revenue_para", "Revenue Codes"),
        ("history_exp", "Revision Explanation"),
        ("source_info", "Source Information"),          # skipped unless --include-bibliography
        ("bibliography", "Bibliography"),                # skipped unless --include-bibliography
    ],
    "ART": [
        ("description", "Article Text"),
        ("cms_cov_policy", "CMS National Coverage Policy"),
        ("revenue_para", "Revenue Codes"),
        ("add_icd10_info", "Additional ICD-10 Information"),
        ("other_comments", "Other Comments"),
        ("history_exp", "Revision Explanation"),
    ],
    "NCD": [
        ("itm_srvc_desc", "Item/Service Description"),
        ("indctn_lmtn", "Indications and Limitations of Coverage"),
        ("xref_txt", "Cross Reference"),
        ("othr_txt", "Other"),
        ("rev_hstry", "Revision History"),
    ],
}
LOW_VALUE = {"bibliography", "source_info"}

CODE_LABEL = {
    ("HCPCS", "listed"): "CPT/HCPCS Codes",
    ("HCPCS_MOD", "listed"): "CPT/HCPCS Modifiers",
    ("ICD10CM", "covered"): "ICD-10-CM Codes that Support Medical Necessity",
    ("ICD10CM", "noncovered"): "ICD-10-CM Codes that DO NOT Support Medical Necessity",
    ("ICD10PCS", "covered"): "ICD-10-PCS Codes",
    ("REV", "listed"): "Revenue Codes",
    ("BILL", "listed"): "Bill Types",
}
REL_PREFIX = {"LCD": "L", "ART": "A", "NCD": "NCD id "}
BOILERPLATE = {"n/a", "na", "none", "not applicable", "none.", "n/a."}


TYPE_TABLE = {"LCD": "lcd", "ART": "article", "NCD": "ncd"}


def source_table_for(section: str, doc_type: str) -> str:
    """Which table the chunk's text came from (for citations / traceability)."""
    if section == "overview" or section == "cms_cov_policy":
        return "policy_doc"
    if section.endswith(("_intro", "_note")):
        return "policy_code_group"
    if section.endswith("_codes"):
        return "policy_code"
    return {"other_coding": "article_other_coding_group", "sticky_note": "policy_sticky_note",
            "revision_history": "policy_revision_history",
            "response_to_comment": "article_response_to_comment"}.get(section, TYPE_TABLE[doc_type])


@dataclass
class Chunk:
    section: str
    source_key: str
    chunk_index: int
    heading_path: str
    content: str

    @property
    def key(self) -> tuple:
        return (self.section, self.source_key, self.chunk_index)

    @property
    def hash(self) -> str:
        return hashlib.sha256(f"{self.heading_path}\n{self.content}".encode()).hexdigest()


@dataclass
class Cfg:
    target: int = 1400
    hard: int = 2000
    overlap: int = 200
    min_chars: int = 300
    max_code_chunks: int = 8
    code_chunks: bool = True
    bibliography: bool = False
    comments: bool = False


# ------------------------------------------------------------------ text splitting
def substantive(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.strip()
    return len(t) >= 25 and t.lower().strip(" .") not in BOILERPLATE


def _hard_cut(s: str, size: int) -> list[str]:
    out = []
    while len(s) > size:
        cut = s.rfind(" ", 0, size)
        cut = cut if cut > size // 2 else size
        out.append(s[:cut].strip())
        s = s[cut:].strip()
    if s:
        out.append(s)
    return out


def split_long(p: str, cfg: Cfg) -> list[str]:
    """Split an over-long paragraph on lines (tables/lists) or sentences."""
    if len(p) <= cfg.hard:
        return [p]
    sep = "\n" if "\n" in p else " "
    units = p.split("\n") if sep == "\n" else re.split(r"(?<=[.!?;])\s+", p)
    pieces, buf = [], ""
    for u in units:
        if len(u) > cfg.hard:
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.extend(_hard_cut(u, cfg.target))
        elif buf and len(buf) + len(sep) + len(u) > cfg.target:
            pieces.append(buf)
            buf = u
        else:
            buf = f"{buf}{sep}{u}" if buf else u
    if buf:
        pieces.append(buf)
    return pieces


def chunk_text(text: str, cfg: Cfg) -> list[tuple[Optional[str], str]]:
    """-> [(heading in effect at chunk start, chunk text)]. Headings are '### ' paragraphs."""
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    out: list[tuple[Optional[str], str]] = []
    cur: list[str] = []
    cur_len = 0
    cur_head: Optional[str] = None      # heading currently in effect
    start_head: Optional[str] = None    # heading in effect when this chunk began
    fresh = False                       # chunk holds something beyond the carried overlap

    def flush(carry: bool) -> None:
        nonlocal cur, cur_len, fresh, start_head
        if cur and fresh:
            out.append((start_head, "\n\n".join(cur)))
        keep = [cur[-1]] if (carry and cur and not cur[-1].startswith("### ")
                             and len(cur[-1]) <= cfg.overlap) else []
        cur, cur_len, fresh = keep, sum(len(k) + 2 for k in keep), False
        start_head = cur_head

    for p in paras:
        is_head = p.startswith("### ") and "\n" not in p
        if is_head:
            cur_head = p[4:].strip()
            if cur and fresh and cur_len >= cfg.min_chars:
                flush(carry=False)
                cur_head = p[4:].strip()
                start_head = cur_head
        for piece in ([p] if is_head else split_long(p, cfg)):
            if cur and fresh and cur_len + len(piece) + 2 > cfg.target:
                flush(carry=True)
            if not cur:
                start_head = cur_head
            cur.append(piece)
            cur_len += len(piece) + 2
            fresh = True
    flush(carry=False)

    # fold a tiny trailing chunk into its predecessor when it fits
    if len(out) >= 2 and len(out[-1][1]) < 200 and len(out[-2][1]) + len(out[-1][1]) + 2 <= cfg.hard:
        h, t = out[-2]
        out[-2:] = [(h, t + "\n\n" + out[-1][1])]
    return out


def text_to_chunks(base: str, label: str, section: str, source_key: str, text: Optional[str],
                   cfg: Cfg) -> list[Chunk]:
    if not substantive(text):
        return []
    res = []
    for i, (head, body) in enumerate(chunk_text(text, cfg)):
        lead = f"### {head}\n\n" if head else None
        if lead and body.startswith(lead) and len(body) > len(lead) + 40:
            body = body[len(lead):]          # heading already lives in heading_path
        path = f"{base} > {label}" + (f" > {head}" if head and head != label else "")
        res.append(Chunk(section, source_key, i, path, body))
    return res


# --------------------------------------------------------------- per-document assembly
def fmt_code(c: dict) -> str:
    desc = c["description"] or c["short_description"]
    line = f"{c['code']} - {desc}" if desc else c["code"]
    if c["range_pos"] == "B":
        line += " (start of range)"
    elif c["range_pos"] == "E":
        line += " (end of range)"
    return line


def code_chunks(base: str, system: str, role: str, group_no: int, codes: list[dict], cfg: Cfg) -> list[Chunk]:
    label = CODE_LABEL.get((system, role), f"{system} Codes")
    label += f" > Group {group_no}" if group_no else ""
    lines = [fmt_code(c) for c in codes]
    chunks_txt, buf = [], ""
    for ln in lines:
        if buf and len(buf) + 1 + len(ln) > cfg.target:
            chunks_txt.append(buf)
            buf = ln
        else:
            buf = f"{buf}\n{ln}" if buf else ln
    if buf:
        chunks_txt.append(buf)
    if len(chunks_txt) > cfg.max_code_chunks:
        kept = chunks_txt[:cfg.max_code_chunks]
        shown = sum(t.count("\n") + 1 for t in kept)
        kept[-1] += f"\n... plus {len(lines) - shown} more codes (full list in policy_code)"
        chunks_txt = kept
    sec = f"{system.lower()}_{role}_codes"
    key = f"group={group_no}"
    return [Chunk(sec, key, i, f"{base} > {label}", t) for i, t in enumerate(chunks_txt)]


def build_chunks(doc: dict, ext: dict, groups: list[dict], codes: list[dict], sticky: list[dict],
                 revs: list[dict], rtc: list[dict], other: list[dict], contractors: list[dict],
                 related: list[dict], cfg: Cfg) -> list[Chunk]:
    base = f"{doc['public_id']}: {doc['title']}"
    out: list[Chunk] = []

    # --- overview
    bits = [f"{doc['public_id']}: {doc['title']}", f"Document type: {DOC_LABEL[doc['doc_type']]}"]
    status = {"A": "Active", "R": "Retired", "P": "Proposed"}.get(doc["status"] or "", None)
    if doc["is_draft"]:
        status = "Draft/Proposed"
    if status:
        bits.append(f"Status: {status}")
    if doc["effective_date"]:
        bits.append(f"Effective: {doc['effective_date']}")
    if doc["end_date"]:
        bits.append(f"Ends: {doc['end_date']}")
    if contractors:
        by_name: dict[str, list[str]] = {}
        for c in contractors:
            by_name.setdefault(c["contractor_bus_name"] or "?", []).append(c["contractor_number"] or "?")
        items = [f"{n} ({', '.join(nums[:4])}{'...' if len(nums) > 4 else ''})" for n, nums in by_name.items()]
        bits.append("Medicare Administrative Contractor(s): " + "; ".join(items[:8])
                    + (f"; +{len(items) - 8} more" if len(items) > 8 else ""))
    if doc["state_abbrevs"]:
        bits.append("Jurisdiction (states/territories): " + ", ".join(doc["state_abbrevs"]))
    elif doc["doc_type"] == "NCD":
        bits.append("Jurisdiction: national (applies in all states)")
    if related:
        bits.append("Related documents: " + "; ".join(
            f"{r['public_id'] or REL_PREFIX[r['related_doc_type']] + str(r['related_doc_id'])}"
            + (f" ({r['title']})" if r["title"] else "") for r in related[:15]))
    out.append(Chunk("overview", "", 0, base + " > Overview", "\n".join(bits)))

    # --- narrative sections
    for col, label in TEXT_SECTIONS[doc["doc_type"]]:
        if col in LOW_VALUE and not cfg.bibliography:
            continue
        text = doc.get(col) if col == "cms_cov_policy" else ext.get(col)
        out.extend(text_to_chunks(base, label, col, "", text, cfg))

    # --- code groups: intro paragraph + capped listing
    by_group = defaultdict(list)
    for c in codes:
        by_group[(c["code_system"], c["role"], c["group_no"])].append(c)
    gmeta = {(g["code_system"], g["role"], g["group_no"]): g for g in groups}
    for gkey in sorted(set(by_group) | set(gmeta), key=lambda k: (k[0], k[1], k[2])):
        system, role, gno = gkey
        label = CODE_LABEL.get((system, role), f"{system} Codes") + (f" > Group {gno}" if gno else "")
        g = gmeta.get(gkey)
        if g:
            sec = f"{system.lower()}_{role}"
            out.extend(text_to_chunks(base, label, sec + "_intro", f"group={gno}", g["paragraph"], cfg))
            out.extend(text_to_chunks(base, label + " > Note", sec + "_note", f"group={gno}",
                                      g["asterisk_text"], cfg))
        if cfg.code_chunks and gkey in by_group:
            out.extend(code_chunks(base, system, role, gno, by_group[gkey], cfg))

    # --- other coding information (articles)
    for o in other:
        txt = "\n\n".join(x for x in (o["paragraph"], o["codes"]) if x)
        out.extend(text_to_chunks(base, "Other Coding Information", "other_coding",
                                  f"group={o['other_coding_group']}", txt, cfg))

    # --- sticky notes, revision history, comments
    for s in sticky:
        out.extend(text_to_chunks(base, "Sticky Note", "sticky_note", f"v{s['sticky_note_version']}",
                                  s["sticky_note"], cfg))
    if revs:
        txt = "\n\n".join(f"{r['rev_hist_date'] or ''}: {r['rev_hist_exp']}".strip(": ")
                          for r in sorted(revs, key=lambda r: r["rev_hist_date"] or r["last_updated"], reverse=True)
                          if substantive(r["rev_hist_exp"]))
        out.extend(text_to_chunks(base, "Revision History", "revision_history", "", txt, cfg))
    if cfg.comments and rtc:
        txt = "\n\n".join(f"Comment: {r['comment'] or ''}\nResponse: {r['response'] or ''}" for r in rtc
                          if substantive(r["comment"]) or substantive(r["response"]))
        out.extend(text_to_chunks(base, "Comments and Responses", "response_to_comment", "", txt, cfg))

    # a (section, source_key, index) key must be unique
    seen, uniq = set(), []
    for c in out:
        if c.key not in seen:
            seen.add(c.key)
            uniq.append(c)
    return uniq


# -------------------------------------------------------------------------- DB I/O
def select_docs(conn, args) -> list[int]:
    statuses = ["A"] + (["P"] if args.include_drafts else []) + (["R"] if args.include_retired else [])
    where = ["d.doc_type = ANY(%(types)s)", "d.status = ANY(%(st)s)"]
    params = {"types": args.types, "st": statuses}
    if not args.include_drafts:
        where.append("NOT d.is_draft")
    if not args.all_versions:
        where.append("d.is_latest")
    if args.public_id:
        where.append("d.public_id = ANY(%(pids)s)")
        params["pids"] = args.public_id
    sql = f"SELECT d.doc_pk FROM policy_doc d WHERE {' AND '.join(where)} ORDER BY d.doc_pk"
    if args.limit:
        sql += f" LIMIT {int(args.limit)}"
    return [r[0] for r in conn.execute(sql, params)]


def fetch_batch(conn, pks: list[int]) -> dict:
    cur = conn.cursor(row_factory=dict_row)
    q = lambda sql: cur.execute(sql, (pks,)).fetchall()  # noqa: E731
    data = {"docs": {r["doc_pk"]: r for r in q("SELECT * FROM policy_doc WHERE doc_pk = ANY(%s)")}}
    ext = {}
    for t in ("lcd", "article", "ncd"):
        for r in q(f"SELECT * FROM {t} WHERE doc_pk = ANY(%s)"):
            ext[r["doc_pk"]] = r
    data["ext"] = ext
    def grouped(sql):
        d = defaultdict(list)
        for r in q(sql):
            d[r["doc_pk"]].append(r)
        return d
    data["groups"] = grouped("SELECT * FROM policy_code_group WHERE doc_pk = ANY(%s)")
    data["codes"] = grouped("""SELECT doc_pk, code_system, role, group_no, code, range_pos, description,
                                      short_description FROM policy_code WHERE doc_pk = ANY(%s)
                               ORDER BY doc_pk, code_system, role, group_no, sort_order NULLS LAST, code""")
    data["sticky"] = grouped("SELECT * FROM policy_sticky_note WHERE doc_pk = ANY(%s) ORDER BY sticky_note_version")
    data["revs"] = grouped("SELECT * FROM policy_revision_history WHERE doc_pk = ANY(%s)")
    data["rtc"] = grouped("SELECT * FROM article_response_to_comment WHERE doc_pk = ANY(%s) ORDER BY rtc_num")
    data["other"] = grouped("SELECT * FROM article_other_coding_group WHERE doc_pk = ANY(%s) ORDER BY other_coding_group")
    data["contractors"] = grouped("""SELECT pc.doc_pk, c.contractor_bus_name, c.contractor_number
                                     FROM policy_contractor pc JOIN contractor c USING
                                       (contractor_id, contractor_type_id, contractor_version)
                                     WHERE pc.doc_pk = ANY(%s) ORDER BY c.contractor_bus_name""")
    data["related"] = grouped("""SELECT r.doc_pk, r.related_doc_type, r.related_doc_id, t.public_id, t.title
                                 FROM policy_related r LEFT JOIN policy_doc t
                                   ON t.doc_type = r.related_doc_type AND t.doc_id = r.related_doc_id AND t.is_latest
                                 WHERE r.doc_pk = ANY(%s) ORDER BY r.related_doc_type, r.related_doc_id""")
    ex = defaultdict(dict)
    for r in q("SELECT chunk_id, doc_pk, section, source_key, chunk_index, content_hash FROM doc_chunk WHERE doc_pk = ANY(%s)"):
        ex[r["doc_pk"]][(r["section"], r["source_key"], r["chunk_index"])] = (r["chunk_id"], r["content_hash"])
    data["existing"] = ex
    return data


def chunks_for(data: dict, pk: int, cfg: Cfg) -> list[Chunk]:
    return build_chunks(data["docs"][pk], data["ext"].get(pk, {}), data["groups"].get(pk, []),
                        data["codes"].get(pk, []), data["sticky"].get(pk, []), data["revs"].get(pk, []),
                        data["rtc"].get(pk, []), data["other"].get(pk, []), data["contractors"].get(pk, []),
                        data["related"].get(pk, []), cfg)


def apply_batch(conn, plan: dict) -> None:
    cur = conn.cursor()
    if plan["delete"]:
        cur.execute("DELETE FROM doc_chunk WHERE chunk_id = ANY(%s)", (plan["delete"],))
    if plan["update"]:
        cur.execute("DELETE FROM chunk_embedding WHERE chunk_id = ANY(%s)", ([u[0] for u in plan["update"]],))
        cur.executemany("""UPDATE doc_chunk SET heading_path = %s, content = %s, token_count = %s,
                             content_hash = %s WHERE chunk_id = %s""",
                        [(h, c, t, ch, cid) for cid, h, c, t, ch in plan["update"]])
    if plan["insert"]:
        cur.executemany("""INSERT INTO doc_chunk (doc_pk, section, source_table, source_key, chunk_index,
                             heading_path, content, token_count, content_hash)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        plan["insert"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn")
    ap.add_argument("--types", nargs="+", choices=["LCD", "ART", "NCD"], default=["LCD", "ART", "NCD"])
    ap.add_argument("--include-drafts", action="store_true", help="also chunk Proposed LCDs / Draft Articles")
    ap.add_argument("--include-retired", action="store_true")
    ap.add_argument("--all-versions", action="store_true", help="chunk superseded NCD versions too")
    ap.add_argument("--public-id", nargs="+", help="only these documents, e.g. L33252 A52464 'NCD 220.6.17'")
    ap.add_argument("--limit", type=int, help="only the first N documents (development)")
    ap.add_argument("--show", nargs="+", metavar="PUBLIC_ID", help="print chunks for these documents; write nothing")
    ap.add_argument("--target-chars", type=int, default=1400)
    ap.add_argument("--max-chars", type=int, default=2000)
    ap.add_argument("--overlap-chars", type=int, default=200)
    ap.add_argument("--max-code-chunks", type=int, default=8, help="cap per code group (default 8)")
    ap.add_argument("--no-code-chunks", action="store_true", help="skip code listings entirely")
    ap.add_argument("--include-bibliography", action="store_true", help="include bibliography + source info")
    ap.add_argument("--include-comments", action="store_true", help="include comments & responses (articles)")
    ap.add_argument("--no-prune", action="store_true", help="keep chunks of documents no longer selected")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch", type=int, default=100, help="documents per batch")
    args = ap.parse_args()

    cfg = Cfg(target=args.target_chars, hard=args.max_chars, overlap=args.overlap_chars,
              max_code_chunks=args.max_code_chunks, code_chunks=not args.no_code_chunks,
              bibliography=args.include_bibliography, comments=args.include_comments)
    conn = connect(args.dsn)

    if args.show:
        args.public_id, args.limit = args.show, None
        args.include_drafts = args.include_retired = args.all_versions = True
        pks = select_docs(conn, args)
        if not pks:
            print("no matching documents", file=sys.stderr)
            return 1
        data = fetch_batch(conn, pks)
        for pk in pks:
            for c in chunks_for(data, pk, cfg):
                print(f"\n=== [{c.section} | {c.source_key or '-'} | #{c.chunk_index}] {c.heading_path}  ({len(c.content)} chars)")
                print(c.content)
        return 0

    t0 = time.time()
    pks = select_docs(conn, args)
    print(f"{len(pks)} documents selected", flush=True)
    stats, per_section, sizes = Counter(), Counter(), []
    for i in range(0, len(pks), args.batch):
        batch = pks[i:i + args.batch]
        data = fetch_batch(conn, batch)
        plan = {"insert": [], "update": [], "delete": []}
        for pk in batch:
            desired = {c.key: c for c in chunks_for(data, pk, cfg)}
            existing = data["existing"].get(pk, {})
            for key, c in desired.items():
                per_section[c.section] += 1
                sizes.append(len(c.content))
                tokens = max(1, (len(c.heading_path) + len(c.content)) // 4)
                if key not in existing:
                    plan["insert"].append((pk, c.section, source_table_for(c.section, data["docs"][pk]["doc_type"]),
                                           c.source_key, c.chunk_index, c.heading_path, c.content, tokens, c.hash))
                    stats["inserted"] += 1
                elif existing[key][1] != c.hash:
                    plan["update"].append((existing[key][0], c.heading_path, c.content, tokens, c.hash))
                    stats["updated"] += 1
                else:
                    stats["unchanged"] += 1
            for key, (cid, _) in existing.items():
                if key not in desired:
                    plan["delete"].append(cid)
                    stats["deleted"] += 1
        apply_batch(conn, plan)
        conn.commit()
        print(f"  {min(i + args.batch, len(pks))}/{len(pks)} documents  ({time.time() - t0:.0f}s)", flush=True)

    if not args.limit and not args.public_id and not args.no_prune and not args.dry_run:
        cur = conn.execute("DELETE FROM doc_chunk WHERE doc_pk <> ALL(%s)", (pks,))
        stats["pruned (unselected docs)"] = cur.rowcount
        conn.commit()
    if args.dry_run:
        conn.rollback()

    total = sum(per_section.values())
    sizes.sort()
    print(f"\n{total} chunks"
          + (f"  (median {sizes[len(sizes) // 2]} chars, p95 {sizes[int(len(sizes) * .95)]}, max {sizes[-1]})" if sizes else ""))
    print("  " + ", ".join(f"{k}={v}" for k, v in stats.items()))
    for sec, n in per_section.most_common(14):
        print(f"  {sec:32s}{n:8d}")
    if args.dry_run:
        print("dry run: rolled back")
    return 0


if __name__ == "__main__":
    sys.exit(main())
