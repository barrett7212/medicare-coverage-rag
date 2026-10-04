#!/usr/bin/env python3
"""ETL: CMS MCD downloads (.mdb or CSV) -> PostgreSQL (schema in server/mcd_schema.sql).

Typical use
    python mcd_etl.py --init-schema                    # reads ./data (see mcd_fetch.py)
    python mcd_etl.py --source ~/Downloads/*.mdb       # or point at files you already have
    python mcd_etl.py --dry-run                        # load, report counts, roll back

Re-running is safe and designed for the weekly refresh:
  * lookup tables are upserted
  * policy_doc is upserted on (doc_type, doc_id, doc_version) so doc_pk - and therefore the
    chunks/embeddings attached to it - stays stable for unchanged documents; document
    versions that disappeared from the download are deleted (cascading to their chunks)
  * every other table is cleared for the document types being loaded and bulk-reloaded
Everything runs in one transaction.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import psycopg

from mcd_common import (DATA_DIR, Source, ch1, clean_str, clean_url, connect, discover_sources,
                        html_to_text, init_schema, schema_exists, to_bool, to_date, to_int, to_ts)

T0 = time.time()
SKIPPED: Counter = Counter()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ------------------------------------------------------------------ generic loaders
def copy_rows(conn: psycopg.Connection, table: str, cols: list[str], rows: Iterable[tuple]) -> int:
    n = 0
    with conn.cursor() as cur, cur.copy(f"COPY {table} ({', '.join(cols)}) FROM STDIN") as cp:
        for r in rows:
            cp.write_row(r)
            n += 1
    return n


def upsert_rows(conn: psycopg.Connection, table: str, cols: list[str], pk: list[str],
                rows: Iterable[tuple]) -> int:
    """Stage via COPY, then INSERT .. ON CONFLICT DO UPDATE. Last duplicate wins."""
    idx = [cols.index(c) for c in pk]
    dedup = {tuple(r[i] for i in idx): r for r in rows}
    stg = f"stg_{table}"
    conn.execute(f"CREATE TEMP TABLE {stg} AS SELECT {', '.join(cols)} FROM {table} WITH NO DATA")
    copy_rows(conn, stg, cols, dedup.values())
    sets = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in pk)
    action = f"DO UPDATE SET {sets}" if sets else "DO NOTHING"
    conn.execute(f"INSERT INTO {table} ({', '.join(cols)}) SELECT {', '.join(cols)} FROM {stg} "
                 f"ON CONFLICT ({', '.join(pk)}) {action}")
    conn.execute(f"DROP TABLE {stg}")
    return len(dedup)


Spec = list  # list of (target_col, converter[, source_col])


def convert(rows: Iterable[dict], spec: Spec) -> Iterator[tuple]:
    for r in rows:
        yield tuple(s[1](r.get(s[2] if len(s) > 2 else s[0])) for s in spec)


def cols_of(spec: Spec) -> list[str]:
    return [s[0] for s in spec]


def all_rows(sources: dict[str, Source], table: str, kinds=("lcd", "article", "ncd")) -> Iterator[dict]:
    for k in kinds:
        s = sources.get(k)
        if s and s.has(table):
            yield from s.rows(table)


I, S, D, T, B, C = to_int, clean_str, to_date, to_ts, to_bool, ch1
H = html_to_text

# --------------------------------------------------------------------- lookup specs
LOOKUPS: list[tuple[str, Spec, list[str], tuple[str, ...]]] = [
    # table, spec, pk, source kinds that may carry it
    ("state_lookup", [("state_id", I), ("state_abbrev", S), ("description", S)], ["state_id"], ("lcd", "article")),
    ("region_lookup", [("region_id", I), ("description", S)], ["region_id"], ("lcd", "article")),
    ("state_x_region", [("state_id", I), ("region_id", I)], ["state_id"], ("lcd", "article")),
    ("dmerc_region_lookup", [("region_id", I), ("description", S), ("psc_description", S),
                             ("mac_description", S), ("super_mac_description", S)], ["region_id"], ("lcd", "article")),
    ("contractor_type_lookup", [("contractor_type_id", I), ("description", S)], ["contractor_type_id"], ("lcd", "article")),
    ("contractor_subtype_lookup", [("contractor_subtype_id", I), ("description", S)], ["contractor_subtype_id"], ("lcd", "article")),
    ("update_period", [("period_id", I), ("begin_date", D), ("end_date", D)], ["period_id"], ("lcd", "article")),
    ("article_type_lookup", [("article_type_id", I), ("description", S), ("last_updated", T)], ["article_type_id"], ("article",)),
    ("reason_change_lookup", [("reason_change_id", I), ("reason_change_version", I), ("description", S),
                              ("sort_order", I), ("last_updated", T)],
     ["reason_change_id", "reason_change_version"], ("lcd",)),
    ("synopsis_changes_fields_lookup", [("synopsis_changes_field_id", I), ("field_name", S), ("field_anchor", S),
                                        ("mcd_field_name", S), ("mcd_field_anchor", S)],
     ["synopsis_changes_field_id"], ("lcd",)),
    ("ncd_bnft_ctgry_ref", [("bnft_ctgry_cd", I), ("bnft_ctgry_desc", S)], ["bnft_ctgry_cd"], ("ncd",)),
    ("ncd_pblctn_ref", [("pblctn_cd", I), ("pblctn_num", S), ("pblctn_title", S)], ["pblctn_cd"], ("ncd",)),
]

CONTRACTOR_SPEC: Spec = [
    ("contractor_id", I), ("contractor_type_id", I), ("contractor_version", I),
    ("contractor_bus_name", S), ("contractor_number", S), ("dmerc_rgn", I),
    ("address1", S), ("address2", S), ("address3", S), ("city", S), ("state_id", I), ("zipcode", S),
    ("phone", S), ("fax", S), ("url", S), ("email", S), ("status", C), ("status_flag", C),
    ("ignore_flag", C, "ignore"), ("cmd_name", S), ("cmd_title", S), ("contractor_subtype_id", I),
    ("last_updated", T),
]
CONTRACTOR_PK = ["contractor_id", "contractor_type_id", "contractor_version"]


def load_lookups(conn, sources) -> None:
    for table, spec, pk, kinds in LOOKUPS:
        n = upsert_rows(conn, table, cols_of(spec), pk, convert(all_rows(sources, table, kinds), spec))
        log(f"  {table}: {n}")

    # URL types: LCD and Article have separate lookups; IDs may overlap -> key on doc_type
    def url_types() -> Iterator[tuple]:
        for kind, tbl, dt in (("lcd", "lcd_url_type_lookup", "LCD"), ("article", "article_url_type_lookup", "ART")):
            s = sources.get(kind)
            if s and s.has(tbl):
                for r in s.rows(tbl):
                    yield (dt, I(r.get("url_type_id")), S(r.get("description")) or "", I(r.get("sort_order")),
                           T(r.get("last_updated")))
    n = upsert_rows(conn, "url_type_lookup",
                    ["doc_type", "url_type_id", "description", "sort_order", "last_updated"],
                    ["doc_type", "url_type_id"], url_types())
    log(f"  url_type_lookup: {n}")

    # draft contacts: state_id must exist
    states = {r[0] for r in conn.execute("SELECT state_id FROM state_lookup")}
    spec = [("contact_id", I), ("email_address", S), ("first_name", S), ("middle_initial", S), ("last_name", S),
            ("phone", S), ("p_ext", S), ("address1", S), ("address2", S), ("address3", S), ("city", S),
            ("state_id", I), ("zipcode", S), ("last_updated", T)]
    si = cols_of(spec).index("state_id")
    rows = (tuple(None if (i == si and v not in states) else v for i, v in enumerate(r))
            for r in convert(all_rows(sources, "draft_contact_lookup", ("lcd",)), spec))
    n = upsert_rows(conn, "draft_contact_lookup", cols_of(spec), ["contact_id"], rows)
    log(f"  draft_contact_lookup: {n}")

    ctypes = {r[0] for r in conn.execute("SELECT contractor_type_id FROM contractor_type_lookup")}
    ci = cols_of(CONTRACTOR_SPEC).index("contractor_type_id")
    si = cols_of(CONTRACTOR_SPEC).index("state_id")

    def contractors() -> Iterator[tuple]:
        for r in convert(all_rows(sources, "contractor", ("lcd", "article")), CONTRACTOR_SPEC):
            if r[ci] not in ctypes:
                SKIPPED["contractor (unknown type)"] += 1
                continue
            yield tuple(None if (i == si and v not in states) else v for i, v in enumerate(r))
    n = upsert_rows(conn, "contractor", cols_of(CONTRACTOR_SPEC), CONTRACTOR_PK, contractors())
    log(f"  contractor: {n}")

    ckeys = {tuple(r) for r in conn.execute(
        "SELECT contractor_id, contractor_type_id, contractor_version FROM contractor")}
    regions = {r[0] for r in conn.execute("SELECT region_id FROM region_lookup")}

    spec = [("contractor_id", I), ("contractor_type_id", I), ("contractor_version", I), ("state_id", I),
            ("active_date", D), ("term_date", D), ("last_updated", T)]
    rows = (r for r in convert(all_rows(sources, "contractor_jurisdiction", ("lcd", "article")), spec)
            if r[:3] in ckeys and r[3] in states)
    n = upsert_rows(conn, "contractor_jurisdiction", cols_of(spec), CONTRACTOR_PK + ["state_id"], rows)
    log(f"  contractor_jurisdiction: {n}")

    spec = [("contractor_id", I), ("contractor_type_id", I), ("contractor_version", I), ("region_id", I),
            ("last_updated", T)]
    rows = (r for r in convert(all_rows(sources, "contractor_oversight", ("lcd", "article")), spec)
            if r[:3] in ckeys and r[3] in regions)
    n = upsert_rows(conn, "contractor_oversight", cols_of(spec), CONTRACTOR_PK + ["region_id"], rows)
    log(f"  contractor_oversight: {n}")


# ------------------------------------------------------------------- document builders
PD_COLS = ["doc_type", "doc_id", "doc_version", "display_id", "public_id", "title", "status", "is_draft",
           "is_latest", "effective_date", "end_date", "retired_date", "published_date", "last_updated",
           "icd10_doc", "cms_cov_policy", "keywords"]

LCD_COLS: Spec = [
    ("determination_number", S), ("orig_det_eff_date", D), ("ent_det_end_date", D),
    ("issue", H), ("issue_change", H), ("indication", H), ("diagnoses_support", H), ("diagnoses_dont_support", H),
    ("icd9_dont_support_para", H), ("icd9_dont_support_ast", H), ("coding_guidelines", H), ("doc_reqs", H),
    ("appendices", H), ("util_guide", H), ("source_info", H), ("summary_of_evidence", H),
    ("analysis_of_evidence", H), ("bibliography", H), ("associated_info", H), ("add_icd10_info", H),
    ("revenue_para", H), ("synopsis_changes", H), ("history_exp", H), ("rev_hist_num", I),
    ("last_reviewed_on", D), ("adv_meeting", H), ("comment_start_dt", D), ("comment_end_dt", D),
    ("notice_start_dt", D), ("notice_end_dt", D), ("mcd_publish_date", D), ("draft_released_date", D),
    ("source_lcd_id", I), ("draft_contact", I), ("mac_initiated", C), ("thirty_percent", C),
]
ART_COLS: Spec = [
    ("article_type_id", I, "article_type"), ("description", H), ("other_comments", H), ("history_exp", H),
    ("add_icd10_info", H), ("icd9_covered_para", H), ("icd9_noncovered_para", H), ("revenue_para", H),
    ("sad_url", clean_url), ("key_article", C), ("reference_article", C), ("source_article_id", I),
    ("article_rev_end_date", D), ("thirty_percent", C),
]
NCD_COLS: Spec = [
    ("is_ncd", B, "natl_cvrg_type"), ("cvrg_lvl_cd", I), ("mnl_sect", S, "ncd_mnl_sect"),
    ("mnl_sect_title", S, "ncd_mnl_sect_title"), ("implementation_date", D, "ncd_impltn_dt"),
    ("itm_srvc_desc", H), ("indctn_lmtn", H), ("xref_txt", H), ("othr_txt", H), ("rev_hstry", H),
    ("trnsmtl_num", S), ("trnsmtl_issnc_dt", D), ("trnsmtl_url", clean_url), ("chg_rqst_num", S),
    ("pblctn_cd", I), ("under_review", B, "under_rvw"), ("is_lab_ncd", B, "ncd_lab"),
    ("ama_notice", B, "ncd_ama"), ("created_ts", T, "creatd_tmstmp"), ("last_updated_ts", T, "last_updt_tmstmp"),
    ("last_cleared_ts", T, "last_clrnc_tmstmp"),
]


def _pd(**kw) -> dict:
    kw.setdefault("display_id", None)
    return kw


def build_lcds(src: Source) -> Iterator[tuple[dict, tuple]]:
    for r in src.rows("lcd"):
        lid, ver = I(r.get("lcd_id")), I(r.get("lcd_version"))
        if lid is None or ver is None:
            continue
        disp = S(r.get("display_id"))
        pd = _pd(doc_type="LCD", doc_id=lid, doc_version=ver, display_id=disp,
                 public_id=f"DL{disp}" if disp else f"L{lid}",
                 title=S(r.get("title")) or "(untitled)", status=C(r.get("status")), is_draft=bool(disp),
                 effective_date=D(r.get("rev_eff_date")) or D(r.get("orig_det_eff_date")),
                 end_date=D(r.get("rev_end_date")) or D(r.get("ent_det_end_date")),
                 retired_date=D(r.get("date_retired")),
                 published_date=D(r.get("mcd_publish_date")) or D(r.get("orig_det_eff_date")),
                 last_updated=T(r.get("last_updated")), icd10_doc=B(r.get("icd10_doc")),
                 cms_cov_policy=H(r.get("cms_cov_policy")), keywords=S(r.get("keywords")))
        yield pd, tuple(s[1](r.get(s[2] if len(s) > 2 else s[0])) for s in LCD_COLS)


def build_articles(src: Source) -> Iterator[tuple[dict, tuple]]:
    for r in src.rows("article"):
        aid, ver = I(r.get("article_id")), I(r.get("article_version"))
        if aid is None or ver is None:
            continue
        disp = S(r.get("display_id"))
        pd = _pd(doc_type="ART", doc_id=aid, doc_version=ver, display_id=disp,
                 public_id=f"DA{disp}" if disp else f"A{aid}",
                 title=S(r.get("title")) or "(untitled)", status=C(r.get("status")), is_draft=bool(disp),
                 effective_date=D(r.get("article_eff_date")), end_date=D(r.get("article_end_date")),
                 retired_date=D(r.get("date_retired")), published_date=D(r.get("article_pub_date")),
                 last_updated=T(r.get("last_updated")), icd10_doc=B(r.get("icd10_doc")),
                 cms_cov_policy=H(r.get("cms_cov_policy")), keywords=S(r.get("keywords")))
        yield pd, tuple(s[1](r.get(s[2] if len(s) > 2 else s[0])) for s in ART_COLS)


def build_ncds(src: Source) -> Iterator[tuple[dict, tuple]]:
    for r in src.rows("ncd_trkg"):
        nid, ver = I(r.get("ncd_id")), I(r.get("ncd_vrsn_num"))
        if nid is None or ver is None:
            continue
        sect = S(r.get("ncd_mnl_sect")) or str(nid)
        upd, clr = T(r.get("last_updt_tmstmp")), T(r.get("last_clrnc_tmstmp"))
        pd = _pd(doc_type="NCD", doc_id=nid, doc_version=ver, public_id=f"NCD {sect}",
                 title=S(r.get("ncd_mnl_sect_title")) or f"NCD {sect}", status=None, is_draft=False,
                 effective_date=D(r.get("ncd_efctv_dt")), end_date=D(r.get("ncd_trmntn_dt")),
                 retired_date=None, published_date=D(r.get("trnsmtl_issnc_dt")),
                 last_updated=max([x for x in (upd, clr) if x], default=None), icd10_doc=None,
                 cms_cov_policy=None, keywords=S(r.get("ncd_keyword")))
        yield pd, tuple(s[1](r.get(s[2] if len(s) > 2 else s[0])) for s in NCD_COLS)


def load_documents(conn, sources, kinds: list[str]) -> dict[tuple, int]:
    """Build + upsert policy_doc, load lcd/article/ncd. Returns {(doc_type, id, ver): doc_pk}."""
    builders = {"lcd": ("LCD", build_lcds, LCD_COLS, "lcd"),
                "article": ("ART", build_articles, ART_COLS, "article"),
                "ncd": ("NCD", build_ncds, NCD_COLS, "ncd")}
    built: dict[str, list[tuple[dict, tuple]]] = {}
    for k in kinds:
        dt, fn, _, _ = builders[k]
        built[dt] = list(fn(sources[k]))
        log(f"  parsed {len(built[dt])} {dt} documents")

    today = date.today()
    for dt, lst in built.items():
        latest: dict[int, int] = {}
        for pd, _ in lst:
            latest[pd["doc_id"]] = max(latest.get(pd["doc_id"], -1), pd["doc_version"])
        for pd, _ in lst:
            pd["is_latest"] = pd["doc_version"] == latest[pd["doc_id"]]
            if dt == "NCD":  # NCDs have no status column
                live = pd["is_latest"] and (pd["end_date"] is None or pd["end_date"] >= today)
                pd["status"] = "A" if live else "R"

    # --- policy_doc upsert + stale delete
    conn.execute(f"CREATE TEMP TABLE stg_policy_doc AS SELECT {', '.join(PD_COLS)} FROM policy_doc WITH NO DATA")
    n = copy_rows(conn, "stg_policy_doc", PD_COLS,
                  (tuple(pd[c] for c in PD_COLS) for lst in built.values() for pd, _ in lst))
    sets = ", ".join(f"{c} = EXCLUDED.{c}" for c in PD_COLS if c not in ("doc_type", "doc_id", "doc_version"))
    conn.execute(f"INSERT INTO policy_doc ({', '.join(PD_COLS)}) SELECT {', '.join(PD_COLS)} FROM stg_policy_doc "
                 f"ON CONFLICT (doc_type, doc_id, doc_version) DO UPDATE SET {sets}")
    cur = conn.execute("""DELETE FROM policy_doc d WHERE d.doc_type = ANY(%s) AND NOT EXISTS (
                            SELECT 1 FROM stg_policy_doc s WHERE s.doc_type = d.doc_type
                              AND s.doc_id = d.doc_id AND s.doc_version = d.doc_version)""", (list(built),))
    log(f"  policy_doc: {n} upserted, {cur.rowcount} stale versions removed")
    conn.execute("DROP TABLE stg_policy_doc")

    pkmap = {(t, i, v): pk for pk, t, i, v in conn.execute(
        "SELECT doc_pk, doc_type, doc_id, doc_version FROM policy_doc WHERE doc_type = ANY(%s)", (list(built),))}

    # --- clear everything hanging off these doc types, then reload
    clear_children(conn, list(built))

    states = {r[0] for r in conn.execute("SELECT state_id FROM state_lookup")}  # noqa: F841 (parity w/ lookups)
    contacts = {r[0] for r in conn.execute("SELECT contact_id FROM draft_contact_lookup")}
    atypes = {r[0] for r in conn.execute("SELECT article_type_id FROM article_type_lookup")}
    pubs = {r[0] for r in conn.execute("SELECT pblctn_cd FROM ncd_pblctn_ref")}

    for dt, table, spec, fk_col, fk_set in (("LCD", "lcd", LCD_COLS, "draft_contact", contacts),
                                            ("ART", "article", ART_COLS, "article_type_id", atypes),
                                            ("NCD", "ncd", NCD_COLS, "pblctn_cd", pubs)):
        if dt not in built:
            continue
        cols = ["doc_pk"] + cols_of(spec)
        fi = cols.index(fk_col)
        rows = []
        for pd, vals in built[dt]:
            row = [pkmap[(dt, pd["doc_id"], pd["doc_version"])], *vals]
            if row[fi] is not None and row[fi] not in fk_set:
                row[fi] = None
            rows.append(tuple(row))
        log(f"  {table}: {copy_rows(conn, table, cols, rows)}")
    return pkmap


CHILD_TABLES = ["policy_contractor", "policy_state", "policy_code", "policy_code_group", "policy_related",
                "policy_source_icd9", "policy_revision_history", "policy_sticky_note", "policy_url"]
TYPE_TABLES = {"LCD": "lcd", "ART": "article", "NCD": "ncd"}  # their children cascade on delete


def clear_children(conn, doc_types: list[str]) -> None:
    for dt in doc_types:
        conn.execute(f"DELETE FROM {TYPE_TABLES[dt]} t USING policy_doc d "
                     f"WHERE d.doc_pk = t.doc_pk AND d.doc_type = %s", (dt,))
    for t in CHILD_TABLES:
        conn.execute(f"DELETE FROM {t} c USING policy_doc d WHERE d.doc_pk = c.doc_pk AND d.doc_type = ANY(%s)",
                     (doc_types,))
    if "LCD" in doc_types or "ART" in doc_types:
        conn.execute("DELETE FROM policy_future_retire WHERE doc_type = ANY(%s)", (doc_types,))


# ------------------------------------------------------------------------ crosswalks
def keyed(sources, kind, table, id_col, ver_col, pkmap) -> Iterator[tuple[int, dict]]:
    """Yield (doc_pk, row) for a crosswalk table, skipping orphans."""
    s = sources.get(kind)
    if not s or not s.has(table):
        return
    dt = {"lcd": "LCD", "article": "ART", "ncd": "NCD"}[kind]
    for r in s.rows(table):
        pk = pkmap.get((dt, I(r.get(id_col)), I(r.get(ver_col))))
        if pk is None:
            SKIPPED[f"{table} (orphan)"] += 1
            continue
        yield pk, r


KIND_COLS = {"lcd": ("lcd_id", "lcd_version"), "article": ("article_id", "article_version"),
             "ncd": ("ncd_id", "ncd_vrsn_num")}

CODE_SPECS = [
    # kind, table, system, role, code, ver, group, range, sort, desc, short, asterisk
    ("lcd", "lcd_x_hcpc_code", "HCPCS", "listed", "hcpc_code_id", "hcpc_code_version", "hcpc_code_group", "range", None, "long_description", "short_description", None),
    ("article", "article_x_hcpc_code", "HCPCS", "listed", "hcpc_code_id", "hcpc_code_version", "hcpc_code_group", "range", None, "long_description", "short_description", None),
    ("article", "article_x_hcpc_modifier", "HCPCS_MOD", "listed", "hcpc_modifier_code_id", "hcpc_modifier_code_version", "hcpc_modifier_group", None, None, "description", None, None),
    ("article", "article_x_icd10_covered", "ICD10CM", "covered", "icd10_code_id", "icd10_code_version", "icd10_covered_group", "range", "sort_order", "description", None, "asterisk"),
    ("article", "article_x_icd10_noncovered", "ICD10CM", "noncovered", "icd10_code_id", "icd10_code_version", "icd10_noncovered_group", "range", "sort_order", "description", None, None),
    ("article", "article_x_icd10_pcs_code", "ICD10PCS", "covered", "icd10_pcs_code_id", "icd10_pcs_code_version", "icd10_pcs_code_group", "range", "sort_order", "description", None, None),
    ("article", "article_x_revenue_code", "REV", "listed", "revenue_code_id", "revenue_code_version", None, "range", None, "description", None, None),
    ("article", "article_x_bill_code", "BILL", "listed", "bill_code_id", "bill_code_version", None, None, None, "description", None, None),
]
GROUP_SPECS = [
    # kind, table, system, role, group col, asterisk-text col
    ("lcd", "lcd_x_hcpc_code_group", "HCPCS", "listed", "hcpc_code_group", None),
    ("article", "article_x_hcpc_code_group", "HCPCS", "listed", "hcpc_code_group", None),
    ("article", "article_x_hcpc_modifier_group", "HCPCS_MOD", "listed", "hcpc_modifier_group", None),
    ("article", "article_x_icd10_covered_group", "ICD10CM", "covered", "icd10_covered_group", "icd10_covered_ast"),
    ("article", "article_x_icd10_noncovered_group", "ICD10CM", "noncovered", "icd10_noncovered_group", None),
    ("article", "article_x_icd10_pcs_code_group", "ICD10PCS", "covered", "icd10_pcs_code_group", None),
]


def load_crosswalks(conn, sources, pkmap, kinds: list[str]) -> None:
    def k(kind):  # (id col, version col) for a source kind
        return KIND_COLS[kind]

    active = [x for x in kinds if x in ("lcd", "article", "ncd")]
    states = {r[0] for r in conn.execute("SELECT state_id FROM state_lookup")}
    ckeys = {tuple(r) for r in conn.execute(
        "SELECT contractor_id, contractor_type_id, contractor_version FROM contractor")}

    # ---- contractors
    def contractors():
        for kind, tbl in (("lcd", "lcd_x_contractor"), ("article", "article_x_contractor")):
            if kind not in active:
                continue
            for pk, r in keyed(sources, kind, tbl, *k(kind), pkmap):
                key = (I(r.get("contractor_id")), I(r.get("contractor_type_id")), I(r.get("contractor_version")))
                if key in ckeys:
                    yield (pk, *key)
                else:
                    SKIPPED[f"{tbl} (unknown contractor)"] += 1
    n = upsert_rows(conn, "policy_contractor",
                    ["doc_pk", "contractor_id", "contractor_type_id", "contractor_version"],
                    ["doc_pk", "contractor_id", "contractor_type_id", "contractor_version"], contractors())
    log(f"  policy_contractor: {n}")

    # ---- primary jurisdiction -> staging (resolved into policy_state below)
    conn.execute("CREATE TEMP TABLE stg_primary (doc_pk bigint, state_id integer) ON COMMIT DROP")

    def primary():
        for kind, tbl in (("lcd", "lcd_x_primary_jurisdiction"), ("article", "article_x_primary_jurisdiction")):
            if kind not in active:
                continue
            for pk, r in keyed(sources, kind, tbl, *k(kind), pkmap):
                sid = I(r.get("state_id"))
                if sid in states:
                    yield (pk, sid)
    copy_rows(conn, "stg_primary", ["doc_pk", "state_id"], primary())

    # ---- code lists (one unified table)
    seen: set = set()

    def codes():
        for kind, tbl, system, role, ccol, vcol, gcol, rcol, scol, dcol, shcol, acol in CODE_SPECS:
            if kind not in active:
                continue
            for pk, r in keyed(sources, kind, tbl, *k(kind), pkmap):
                code = S(r.get(ccol))
                if not code:
                    continue
                ver = I(r.get(vcol))
                grp = (I(r.get(gcol)) if gcol else None) or 0
                rng = C(r.get(rcol)) if rcol else None
                key = (pk, system, role, grp, code, ver or 0, rng or "-")
                if key in seen:
                    continue
                seen.add(key)
                yield (pk, system, role, code, ver, grp, rng, I(r.get(scol)) if scol else None,
                       S(r.get(dcol)) if dcol else None, S(r.get(shcol)) if shcol else None,
                       (C(r.get(acol)) == "Y") if acol else None, T(r.get("last_updated")))
    ccols = ["doc_pk", "code_system", "role", "code", "code_version", "group_no", "range_pos", "sort_order",
             "description", "short_description", "has_asterisk", "last_updated"]
    log(f"  policy_code: {copy_rows(conn, 'policy_code', ccols, codes())}")
    seen.clear()

    def groups():
        done = set()
        for kind, tbl, system, role, gcol, acol in GROUP_SPECS:
            if kind not in active:
                continue
            for pk, r in keyed(sources, kind, tbl, *k(kind), pkmap):
                grp = I(r.get(gcol))
                if grp is None or (pk, system, role, grp) in done:
                    continue
                done.add((pk, system, role, grp))
                yield (pk, system, role, grp, H(r.get("paragraph")), H(r.get(acol)) if acol else None,
                       T(r.get("last_updated")))
    log(f"  policy_code_group: " + str(copy_rows(
        conn, "policy_code_group",
        ["doc_pk", "code_system", "role", "group_no", "paragraph", "asterisk_text", "last_updated"], groups())))

    # ---- related documents (links always point at the latest version of the target)
    def related():
        done = set()
        for kind, tbl in (("lcd", "lcd_related_documents"), ("article", "article_related_documents")):
            if kind not in active:
                continue
            for pk, r in keyed(sources, kind, tbl, *k(kind), pkmap):
                for dt, col in (("ART", "r_article_id"), ("LCD", "r_lcd_id")):
                    rid = I(r.get(col))
                    if rid and (pk, dt, rid) not in done:
                        done.add((pk, dt, rid))
                        yield (pk, dt, rid, I(r.get("related_num")), I(r.get("r_contractor_id")),
                               T(r.get("last_updated")))
        for kind, tbl in (("lcd", "lcd_related_ncd_documents"), ("article", "article_related_ncd_documents")):
            if kind not in active:
                continue
            for pk, r in keyed(sources, kind, tbl, *k(kind), pkmap):
                rid = I(r.get("r_ncd_id"))
                if rid and (pk, "NCD", rid) not in done:  # r_ncd_id 0 == "N/A"
                    done.add((pk, "NCD", rid))
                    yield (pk, "NCD", rid, I(r.get("related_num")), None, T(r.get("last_updated")))
    log("  policy_related: " + str(copy_rows(
        conn, "policy_related",
        ["doc_pk", "related_doc_type", "related_doc_id", "related_num", "related_contractor_id", "last_updated"],
        related())))

    def icd9():
        for kind, tbl, col in (("lcd", "lcd_related_source_icd9", "source_lcd_id"),
                               ("article", "article_related_source_icd9", "source_article_id")):
            if kind not in active:
                continue
            for pk, r in keyed(sources, kind, tbl, *k(kind), pkmap):
                yield (pk, I(r.get("related_num")), I(r.get(col)), T(r.get("last_updated")))
    n = upsert_rows(conn, "policy_source_icd9", ["doc_pk", "related_num", "source_doc_id", "last_updated"],
                    ["doc_pk", "related_num"], (r for r in icd9() if r[1] is not None and r[2] is not None))
    log(f"  policy_source_icd9: {n}")

    # ---- revision history / sticky notes / URLs
    def revhist():
        for kind, tbl in (("lcd", "lcd_x_revision_history"), ("article", "article_x_revision_history")):
            if kind in active:
                for pk, r in keyed(sources, kind, tbl, *k(kind), pkmap):
                    yield (pk, I(r.get("rev_hist_num")), D(r.get("rev_hist_date")), H(r.get("rev_hist_exp")),
                           T(r.get("last_updated")))
    n = upsert_rows(conn, "policy_revision_history",
                    ["doc_pk", "rev_hist_num", "rev_hist_date", "rev_hist_exp", "last_updated"],
                    ["doc_pk", "rev_hist_num"], (r for r in revhist() if r[1] is not None))
    log(f"  policy_revision_history: {n}")

    def sticky():
        for kind, tbl in (("lcd", "lcd_x_sticky_note"), ("article", "article_x_sticky_note")):
            if kind in active:
                for pk, r in keyed(sources, kind, tbl, *k(kind), pkmap):
                    yield (pk, I(r.get("sticky_note_version")), H(r.get("sticky_note")),
                           T(r.get("sticky_note_dt")), T(r.get("sticky_note_posting_dt")))
    n = upsert_rows(conn, "policy_sticky_note",
                    ["doc_pk", "sticky_note_version", "sticky_note", "sticky_note_dt", "sticky_note_posting_dt"],
                    ["doc_pk", "sticky_note_version"], (r for r in sticky() if r[1] is not None))
    log(f"  policy_sticky_note: {n}")

    utypes = {tuple(r) for r in conn.execute("SELECT doc_type, url_type_id FROM url_type_lookup")}

    def urls():
        for kind, tbl, dt in (("lcd", "lcd_x_urls", "LCD"), ("article", "article_x_urls", "ART")):
            if kind not in active:
                continue
            for pk, r in keyed(sources, kind, tbl, *k(kind), pkmap):
                tid, uid = I(r.get("url_type_id")), I(r.get("url_id"))
                if tid is None or uid is None or not S(r.get("url")):
                    continue
                if (dt, tid) not in utypes:
                    SKIPPED[f"{tbl} (unknown url type)"] += 1
                    continue
                yield (pk, dt, tid, uid, S(r.get("url")), S(r.get("url_name")), S(r.get("url_description")),
                       I(r.get("sort_order")), T(r.get("last_updated")))
    n = upsert_rows(conn, "policy_url",
                    ["doc_pk", "doc_type", "url_type_id", "url_id", "url", "url_name", "url_description",
                     "sort_order", "last_updated"], ["doc_pk", "url_type_id", "url_id"], urls())
    log(f"  policy_url: {n}")

    # ---- article-only tables
    if "article" in active:
        def code_table():
            for pk, r in keyed(sources, "article", "article_x_code_table", *k("article"), pkmap):
                yield (pk, I(r.get("code_table_row")), S(r.get("hcpc_code_id")), I(r.get("hcpc_code_version")),
                       S(r.get("brand_name")), D(r.get("eff_date")), D(r.get("end_date")), H(r.get("comments")),
                       S(r.get("long_description")), S(r.get("short_description")), T(r.get("last_updated")))
        n = upsert_rows(conn, "article_code_table",
                        ["doc_pk", "code_table_row", "hcpc_code_id", "hcpc_code_version", "brand_name", "eff_date",
                         "end_date", "comments", "long_description", "short_description", "last_updated"],
                        ["doc_pk", "code_table_row"], (r for r in code_table() if r[1] is not None))
        log(f"  article_code_table: {n}")

        def other():
            for pk, r in keyed(sources, "article", "article_x_other_coding_group", *k("article"), pkmap):
                yield (pk, I(r.get("other_coding_group")), H(r.get("paragraph")), H(r.get("codes")),
                       T(r.get("last_updated")))
        n = upsert_rows(conn, "article_other_coding_group",
                        ["doc_pk", "other_coding_group", "paragraph", "codes", "last_updated"],
                        ["doc_pk", "other_coding_group"], (r for r in other() if r[1] is not None))
        log(f"  article_other_coding_group: {n}")

        def rtc():
            for pk, r in keyed(sources, "article", "article_x_response_to_comment", *k("article"), pkmap):
                yield (pk, I(r.get("rtc_num")), H(r.get("comment")), H(r.get("response")), T(r.get("last_updated")))
        n = upsert_rows(conn, "article_response_to_comment",
                        ["doc_pk", "rtc_num", "comment", "response", "last_updated"], ["doc_pk", "rtc_num"],
                        (r for r in rtc() if r[1] is not None))
        log(f"  article_response_to_comment: {n}")

    # ---- LCD-only tables
    if "lcd" in active:
        rc = {tuple(r) for r in conn.execute("SELECT reason_change_id, reason_change_version FROM reason_change_lookup")}

        def reason():
            for pk, r in keyed(sources, "lcd", "lcd_x_reason_change", *k("lcd"), pkmap):
                key = (I(r.get("reason_change_id")), I(r.get("reason_change_version")))
                if key in rc:
                    yield (pk, *key, S(r.get("reason_change_other")), T(r.get("last_updated")))
        n = upsert_rows(conn, "lcd_reason_change",
                        ["doc_pk", "reason_change_id", "reason_change_version", "reason_change_other", "last_updated"],
                        ["doc_pk", "reason_change_id", "reason_change_version"], reason())
        log(f"  lcd_reason_change: {n}")

        def advisory():
            for pk, r in keyed(sources, "lcd", "lcd_x_advisory_committee", *k("lcd"), pkmap):
                yield (pk, I(r.get("meeting_id")), D(r.get("meeting_date")), H(r.get("meeting_info")),
                       I(r.get("sort_order")), T(r.get("last_updated")))
        n = upsert_rows(conn, "lcd_advisory_committee",
                        ["doc_pk", "meeting_id", "meeting_date", "meeting_info", "sort_order", "last_updated"],
                        ["doc_pk", "meeting_id"], (r for r in advisory() if r[1] is not None))
        log(f"  lcd_advisory_committee: {n}")

        sf = {r[0] for r in conn.execute("SELECT synopsis_changes_field_id FROM synopsis_changes_fields_lookup")}

        def synopsis():
            for pk, r in keyed(sources, "lcd", "lcd_x_synopsis_changes_fields", *k("lcd"), pkmap):
                fid = I(r.get("synopsis_changes_field_id"))
                if fid in sf:
                    yield (pk, fid, T(r.get("last_updated")))
        n = upsert_rows(conn, "lcd_synopsis_changes_field", ["doc_pk", "synopsis_changes_field_id", "last_updated"],
                        ["doc_pk", "synopsis_changes_field_id"], synopsis())
        log(f"  lcd_synopsis_changes_field: {n}")

        def letters():
            for pk, r in keyed(sources, "lcd", "lcd_x_requestor_letters", *k("lcd"), pkmap):
                yield (pk, I(r.get("letter_id")), S(r.get("requestor_name")), S(r.get("letter_path")),
                       S(r.get("size")), I(r.get("sort_order")), T(r.get("last_updated")))
        n = upsert_rows(conn, "lcd_requestor_letter",
                        ["doc_pk", "letter_id", "requestor_name", "letter_path", "size", "sort_order", "last_updated"],
                        ["doc_pk", "letter_id"], (r for r in letters() if r[1] is not None))
        log(f"  lcd_requestor_letter: {n}")

    # ---- future retirements (keyed by id, not version)
    def retire():
        for kind, tbl, idc, dt in (("lcd", "lcd_future_retire", "lcd_id", "LCD"),
                                   ("article", "article_future_retire", "article_id", "ART")):
            if kind in active and sources[kind].has(tbl):
                for r in sources[kind].rows(tbl):
                    if I(r.get(idc)) is not None and D(r.get("retire_dt")):
                        yield (dt, I(r.get(idc)), D(r.get("retire_dt")))
    n = upsert_rows(conn, "policy_future_retire", ["doc_type", "doc_id", "retire_dt"], ["doc_type", "doc_id"], retire())
    log(f"  policy_future_retire: {n}")

    # ---- NCD benefit categories
    if "ncd" in active:
        cats = {r[0] for r in conn.execute("SELECT bnft_ctgry_cd FROM ncd_bnft_ctgry_ref")}

        def benefits():
            for pk, r in keyed(sources, "ncd", "ncd_trkg_bnft_xref", *k("ncd"), pkmap):
                cd = I(r.get("bnft_ctgry_cd"))
                if cd in cats:
                    yield (pk, cd)
        n = upsert_rows(conn, "ncd_benefit_category", ["doc_pk", "bnft_ctgry_cd"], ["doc_pk", "bnft_ctgry_cd"], benefits())
        log(f"  ncd_benefit_category: {n}")

    # ---- jurisdiction: 1 contractor -> primary jurisdiction, >1 -> contractor_jurisdiction
    types = [{"lcd": "LCD", "article": "ART", "ncd": "NCD"}[x] for x in active]
    conn.execute("""
        INSERT INTO policy_state (doc_pk, state_id)
        SELECT DISTINCT p.doc_pk, p.state_id FROM stg_primary p
        WHERE (SELECT count(*) FROM policy_contractor c WHERE c.doc_pk = p.doc_pk) <= 1
        ON CONFLICT DO NOTHING""")
    conn.execute("""
        INSERT INTO policy_state (doc_pk, state_id)
        SELECT DISTINCT c.doc_pk, j.state_id
        FROM policy_contractor c
        JOIN contractor_jurisdiction j USING (contractor_id, contractor_type_id, contractor_version)
        WHERE (SELECT count(*) FROM policy_contractor c2 WHERE c2.doc_pk = c.doc_pk) > 1
          AND (j.term_date IS NULL OR j.term_date >= CURRENT_DATE)
        ON CONFLICT DO NOTHING""")
    conn.execute("""   -- fallback: single contractor with no primary-jurisdiction row
        INSERT INTO policy_state (doc_pk, state_id)
        SELECT DISTINCT c.doc_pk, j.state_id
        FROM policy_contractor c
        JOIN contractor_jurisdiction j USING (contractor_id, contractor_type_id, contractor_version)
        WHERE NOT EXISTS (SELECT 1 FROM policy_state s WHERE s.doc_pk = c.doc_pk)
          AND (j.term_date IS NULL OR j.term_date >= CURRENT_DATE)
        ON CONFLICT DO NOTHING""")
    log("  policy_state: " + str(conn.execute(
        "SELECT count(*) FROM policy_state s JOIN policy_doc d USING (doc_pk) WHERE d.doc_type = ANY(%s)",
        (types,)).fetchone()[0]))

    # ---- denormalized filter columns on policy_doc
    conn.execute("""
        UPDATE policy_doc d SET
          state_abbrevs  = COALESCE((SELECT array_agg(DISTINCT s.state_abbrev ORDER BY s.state_abbrev)
                                     FROM policy_state ps JOIN state_lookup s USING (state_id)
                                     WHERE ps.doc_pk = d.doc_pk), '{}'),
          contractor_ids = COALESCE((SELECT array_agg(DISTINCT contractor_id) FROM policy_contractor pc
                                     WHERE pc.doc_pk = d.doc_pk), '{}'),
          hcpcs_codes    = COALESCE((SELECT array_agg(DISTINCT code) FROM policy_code pc
                                     WHERE pc.doc_pk = d.doc_pk AND pc.code_system = 'HCPCS'), '{}'),
          icd10_codes    = COALESCE((SELECT array_agg(DISTINCT code) FROM policy_code pc
                                     WHERE pc.doc_pk = d.doc_pk AND pc.code_system = 'ICD10CM'), '{}')
        WHERE d.doc_type = ANY(%s)""", (types,))
    log("  policy_doc filter arrays populated")


# ----------------------------------------------------------------------------- main
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", nargs="*", type=Path, default=[DATA_DIR],
                    help="directories and/or .mdb files (default: $MCD_DATA_DIR or ./data)")
    ap.add_argument("--dsn", default=None, help="Postgres DSN (default: $MCD_DSN or postgresql://localhost/mcd)")
    ap.add_argument("--only", nargs="+", choices=["lcd", "article", "ncd"], help="load only these datasets")
    ap.add_argument("--init-schema", action="store_true", help="create the schema if it does not exist")
    ap.add_argument("--reset-schema", action="store_true", help="DROP and recreate schema mcd (destroys chunks!)")
    ap.add_argument("--dry-run", action="store_true", help="do everything, then roll back")
    args = ap.parse_args(argv)

    sources = discover_sources(args.source)
    if not sources:
        print(f"No MCD datasets found in {[str(p) for p in args.source]}. Run mcd_fetch.py first "
              f"or pass --source.", file=sys.stderr)
        return 2
    kinds = [k for k in ("lcd", "article", "ncd") if k in sources and (not args.only or k in args.only)]
    for k in kinds:
        log(f"{k:8s} <- {sources[k].path}")

    conn = connect(args.dsn)
    if args.reset_schema or (args.init_schema and not schema_exists(conn)):
        log("creating schema")
        init_schema(conn, reset=args.reset_schema)
    elif not schema_exists(conn):
        print("Schema 'mcd' not found. Re-run with --init-schema.", file=sys.stderr)
        return 2

    try:
        log("lookups")
        load_lookups(conn, sources)
        log("documents")
        doc_types = [{"lcd": "LCD", "article": "ART", "ncd": "NCD"}[k] for k in kinds]
        pkmap = load_documents(conn, sources, kinds)
        log("crosswalks")
        load_crosswalks(conn, sources, pkmap, kinds)
        if args.dry_run:
            conn.rollback()
            log("dry run: rolled back")
        else:
            conn.commit()
            conn.autocommit = True
            conn.execute("ANALYZE")
            log("committed")
    except Exception:
        conn.rollback()
        raise

    if SKIPPED:
        print("\nrows skipped (referential gaps in the source data):")
        for k, v in SKIPPED.most_common():
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
