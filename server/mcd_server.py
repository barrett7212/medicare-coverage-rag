#!/usr/bin/env python3
"""MCP server exposing the Medicare Coverage Database (MCD) in the local Postgres instance.

    python server/mcd_server.py               # stdio transport (what MCP clients launch)

Tools
  lookup_code         exact code -> active policies that list it, filtered by state   (structured)
  list_policy_codes   the code lists of one policy                                    (structured)
  search_policy_text  hybrid vector + keyword search over the chunked policy text     (chunks)
  get_policy          header, jurisdiction, MACs, related documents, section index    (structured)
  get_policy_text     the full text of one section of one policy                      (chunks)

Every query runs in a read-only transaction. Only currently effective, non-draft documents
(v_active_policy) are searched; get_policy / get_policy_text can read any loaded document.

Configuration (same variables as ingest/): MCD_DSN, MCD_EMBED_MODEL, OLLAMA_HOST.
If Ollama cannot be reached, search_policy_text falls back to keyword search alone.
"""
from __future__ import annotations

import os
import re
from typing import Optional

import psycopg
import requests
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from psycopg.rows import dict_row

DSN = os.environ.get("MCD_DSN", "postgresql://localhost/mcd")
MODEL = os.environ.get("MCD_EMBED_MODEL", "nomic-embed-text")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# model family -> query prefix (must match the document prefix used by ingest/mcd_embed.py)
QUERY_PREFIXES = {"nomic-embed-text": "search_query: "}

DOC_TYPES = {"LCD": "Local Coverage Determination", "ART": "Article", "NCD": "National Coverage Determination"}
STATUS = {"A": "active", "R": "retired", "P": "proposed"}
NCD_COVERAGE = {1: "full coverage", 2: "restricted coverage", 3: "no coverage"}
CODE_SYSTEMS = ("HCPCS", "HCPCS_MOD", "ICD10CM", "ICD10PCS", "REV", "BILL")
RRF_K = 60          # reciprocal rank fusion constant
CANDIDATES = 50     # per-ranker candidates fed into the fusion

mcp = MCPServer("mcd", instructions="""\
Medicare coverage policy from the CMS Medicare Coverage Database: National Coverage Determinations
(NCD, apply in every state), Local Coverage Determinations (LCD, one MAC's jurisdiction) and
Articles (ART, the billing/coding companion of an LCD, where most code lists live).

To answer "is procedure X covered in state Y":
1. If you have a CPT/HCPCS or ICD-10 code, call lookup_code with the state. Otherwise start with
   search_policy_text.
2. Read the criteria with search_policy_text (restrict it with public_ids) or get_policy_text.
   LCD sections 'indication' and NCD section 'indctn_lmtn' hold the coverage rules.
3. Follow related documents (get_policy): an Article's rules are in its LCD, an LCD's codes are in
   its Articles, and an NCD overrides both.
A code being listed in a policy does not by itself mean it is covered, and no matching policy does
not mean it is not covered: without an NCD or LCD the MAC decides claim by claim. Always cite the
public_id (e.g. L33252, A57520, NCD 220.6.17) and the states it applies to.""")


# --------------------------------------------------------------------------- DB
def query(sql: str, params=None) -> list[dict]:
    try:
        conn = psycopg.connect(DSN, row_factory=dict_row)
    except psycopg.OperationalError as e:
        raise ToolError(f"cannot connect to the MCD database ($MCD_DSN): {e}") from e
    conn.read_only = True  # must precede the first statement: psycopg opens the transaction READ ONLY
    with conn:
        conn.execute("SET search_path = mcd, public")
        conn.execute("SET statement_timeout = '30s'")
        conn.execute("SET hnsw.ef_search = 100")
        # pgvector >= 0.8: keep scanning the HNSW index until the jurisdiction filter is satisfied
        conn.execute("""SELECT set_config('hnsw.iterative_scan', 'relaxed_order', false) FROM pg_extension
                        WHERE extname = 'vector' AND string_to_array(extversion, '.')::int[] >= ARRAY[0, 8]""")
        return conn.execute(sql, params).fetchall()


def resolve_states(state: str) -> list[str]:
    """'CA' / 'California' -> the MCD jurisdiction codes that cover it.

    CMS splits a few states into sub-jurisdictions (California: CA entire state, NF northern,
    SF southern; also New York and Missouri). A whole state matches all of its parts; a part
    matches itself and the entire-state code.
    """
    s = state.strip()
    rows = query("""SELECT state_abbrev, description, split_part(description, ' - ', 1) AS base
                    FROM state_lookup""")
    hit = [r for r in rows if r["state_abbrev"].lower() == s.lower()] \
        or [r for r in rows if r["description"].lower() == s.lower()] \
        or [r for r in rows if r["base"].lower() == s.lower()]
    if not hit:
        raise ToolError(f"unknown state {state!r}; use a two-letter abbreviation such as 'MN' or a full state name")
    if len(hit) == 1 and " - " in hit[0]["description"] and "Entire State" not in hit[0]["description"]:
        base = hit[0]["base"]  # a sub-jurisdiction: itself + the entire-state row
        return sorted({hit[0]["state_abbrev"]} |
                      {r["state_abbrev"] for r in rows if r["base"] == base and "Entire State" in r["description"]})
    bases = {r["base"] for r in hit}
    return sorted(r["state_abbrev"] for r in rows if r["base"] in bases)


def norm_public_id(public_id: str) -> str:
    """'l33252' -> 'L33252', '220.6.17' / 'ncd220.6.17' -> 'NCD 220.6.17'."""
    s = public_id.strip().upper()
    m = re.fullmatch(r"(?:NCD)?\s*(\d+(?:\.\d+)*)", s)
    return f"NCD {m.group(1)}" if m else s


def find_doc(public_id: str) -> dict:
    """The current row for a public id (drafts can share an id across versions)."""
    pid = norm_public_id(public_id)
    rows = query("""SELECT * FROM policy_doc WHERE public_id = %s
                    ORDER BY is_latest DESC, (status = 'A') DESC, doc_version DESC LIMIT 1""", (pid,))
    if not rows:
        raise ToolError(f"no document with public_id {pid!r}; find ids with lookup_code or search_policy_text")
    return rows[0]


def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


# Shared SELECT list: what an agent needs to cite a document and judge whether it applies.
DOC_COLS = """d.public_id, d.doc_type, d.title, d.effective_date, d.state_abbrevs,
    (SELECT array_agg(DISTINCT c.contractor_bus_name || ' (' || t.description || ')')
       FROM policy_contractor pc
       JOIN contractor c USING (contractor_id, contractor_type_id, contractor_version)
       JOIN contractor_type_lookup t USING (contractor_type_id)
      WHERE pc.doc_pk = d.doc_pk) AS contractors"""


def doc_summary(r: dict, states: Optional[list[str]] = None) -> dict:
    """states: the jurisdiction filter of the call; the (often 50+) other states are then only counted."""
    out = {"public_id": r["public_id"], "doc_type": r["doc_type"], "title": r["title"],
           "effective_date": r["effective_date"] and r["effective_date"].isoformat()}
    if r["doc_type"] == "NCD":
        out["jurisdiction"] = "national (all states)"
    else:
        all_states = sorted(r["state_abbrevs"])
        if states:
            out["applies_in"] = [s for s in all_states if s in states]
            out["other_states"] = len(all_states) - len(out["applies_in"])
        else:
            out["states"] = all_states
        out["contractors"] = sorted(r.get("contractors") or [])
    return out


def related_docs(doc_pks: list[int]) -> dict[int, list[dict]]:
    rows = query("""SELECT r.doc_pk, t.public_id, t.doc_type, t.title, t.status
                    FROM policy_related r
                    JOIN policy_doc t ON t.doc_type = r.related_doc_type AND t.doc_id = r.related_doc_id
                                     AND t.is_latest AND NOT t.is_draft
                    WHERE r.doc_pk = ANY(%s)
                    ORDER BY r.doc_pk, t.doc_type, t.public_id""", (doc_pks,))
    out: dict[int, list[dict]] = {}
    for r in rows:
        out.setdefault(r["doc_pk"], []).append(
            {"public_id": r["public_id"], "doc_type": r["doc_type"], "title": r["title"],
             "status": STATUS.get(r["status"], r["status"])})
    return out


# ------------------------------------------------------------------- embeddings
def embed_query(text: str) -> Optional[str]:
    """Query vector as a pgvector literal, or None when Ollama is unavailable."""
    host = OLLAMA_HOST.strip().rstrip("/")
    url = host if "://" in host else f"http://{host}"
    prefix = QUERY_PREFIXES.get(MODEL.split(":")[0], "")
    try:
        r = requests.post(f"{url}/api/embed", json={"model": MODEL, "input": [prefix + text]}, timeout=60)
        r.raise_for_status()
        return "[" + ",".join(map(repr, r.json()["embeddings"][0])) + "]"
    except (requests.RequestException, KeyError, IndexError, ValueError):
        return None


# ------------------------------------------------------------------------ tools
@mcp.tool()
def lookup_code(code: str, state: Optional[str] = None, code_system: Optional[str] = None,
                limit: int = 25) -> dict:
    """Find the active policies that list an exact billing or diagnosis code, optionally in one state.

    Use this first when the question names a code. NCDs carry no code lists, so this returns
    LCDs and Articles only; check NCDs with search_policy_text.

    Args:
        code: CPT/HCPCS code ('95810', 'E0601'), ICD-10-CM/PCS code ('G47.33'), HCPCS modifier,
            revenue code or bill type.
        state: Two-letter abbreviation or state name ('MN', 'Minnesota'). Omit for all states.
        code_system: One of HCPCS (includes CPT), HCPCS_MOD, ICD10CM, ICD10PCS, REV, BILL.
            Omit to search all systems.
        limit: Maximum number of policies to return (1-100).

    Returns policies with, per code group: the role of the code ('covered' / 'noncovered' for
    ICD-10 lists, 'listed' for CPT/HCPCS, which is neutral: read group_note and the policy text
    to learn whether the group is a covered or a non-covered list), the group's introductory
    note, and related documents (for an Article, the LCD holding the coverage criteria).
    """
    c = code.strip().upper()
    system = code_system.strip().upper() if code_system else None
    if system and system not in CODE_SYSTEMS:
        raise ToolError(f"code_system must be one of {', '.join(CODE_SYSTEMS)}")
    codes = [c]
    if "." not in c and len(c) > 3 and (system or "ICD10CM") == "ICD10CM":
        codes.append(f"{c[:3]}.{c[3:]}")  # ICD-10-CM typed without the dot
    states = resolve_states(state) if state else None

    rows = query(f"""
        SELECT d.doc_pk, {DOC_COLS}, pc.code_system, pc.code, pc.role, pc.group_no,
               COALESCE(pc.description, pc.short_description) AS description,
               (SELECT left(string_agg(k.content, E'\\n' ORDER BY k.chunk_index), 700)
                  FROM doc_chunk k
                 WHERE k.doc_pk = d.doc_pk AND k.source_key = 'group=' || pc.group_no
                   AND k.section = lower(pc.code_system) || '_' || pc.role || '_intro') AS group_note
        FROM policy_code pc
        JOIN v_active_policy d USING (doc_pk)
        WHERE pc.code = ANY(%(codes)s)
          AND (%(system)s::text IS NULL OR pc.code_system = %(system)s)
          AND (%(states)s::text[] IS NULL OR d.state_abbrevs && %(states)s)
        ORDER BY d.doc_type DESC, d.public_id, pc.group_no""",
                 {"codes": codes, "system": system, "states": states})

    related = related_docs(list({r["doc_pk"] for r in rows}))
    policies: dict[int, dict] = {}
    for r in rows:
        p = policies.setdefault(r["doc_pk"], {**doc_summary(r, states), "matches": [],
                                              "related": related.get(r["doc_pk"], [])})
        p["matches"].append({"code_system": r["code_system"], "code": r["code"], "role": r["role"],
                             "description": r["description"], "group": r["group_no"],
                             "group_note": r["group_note"]})
    limit = clamp(limit, 1, 100)
    out = {"code": c, "states_searched": states or "all", "policy_count": len(policies),
           "policies": list(policies.values())[:limit]}
    if len(policies) > limit:
        out["note"] = f"{len(policies) - limit} more policies not shown; narrow by state or code_system"
    if not policies:
        out["note"] = ("No active LCD or Article lists this code here. That is not a non-coverage finding: "
                       "check NCDs and policy text with search_policy_text; without a policy the MAC "
                       "decides on medical necessity claim by claim.")
    return out


@mcp.tool()
def list_policy_codes(public_id: str, code_system: Optional[str] = None, role: Optional[str] = None,
                      starts_with: Optional[str] = None, limit: int = 100) -> dict:
    """List the codes attached to one LCD or Article, e.g. to check which diagnoses support a procedure.

    Args:
        public_id: Document id such as 'A57520' or 'L33252'.
        code_system: HCPCS (includes CPT), HCPCS_MOD, ICD10CM, ICD10PCS, REV or BILL.
        role: 'covered' or 'noncovered' (ICD-10 lists) or 'listed' (everything else).
        starts_with: Code prefix filter, e.g. 'E11' for all type 2 diabetes diagnoses.
        limit: Maximum number of codes to return (1-500).

    Returns a count per code_system / role / group (always complete) and the matching codes.
    """
    doc = find_doc(public_id)
    system = code_system.strip().upper() if code_system else None
    if system and system not in CODE_SYSTEMS:
        raise ToolError(f"code_system must be one of {', '.join(CODE_SYSTEMS)}")
    role = role.strip().lower() if role else None
    if role and role not in ("listed", "covered", "noncovered"):
        raise ToolError("role must be 'listed', 'covered' or 'noncovered'")
    limit = clamp(limit, 1, 500)
    params = {"pk": doc["doc_pk"], "system": system, "role": role,
              "prefix": starts_with.strip().upper() + "%" if starts_with else None, "limit": limit + 1}
    where = """doc_pk = %(pk)s
          AND (%(system)s::text IS NULL OR code_system = %(system)s)
          AND (%(role)s::text IS NULL OR role = %(role)s)
          AND (%(prefix)s::text IS NULL OR code LIKE %(prefix)s)"""
    summary = query("""SELECT code_system, role, group_no AS "group", count(*) AS codes
                       FROM policy_code WHERE doc_pk = %(pk)s
                       GROUP BY 1, 2, 3 ORDER BY 1, 2, 3""", params)
    codes = query(f"""SELECT code_system, role, group_no AS "group", code,
                             COALESCE(description, short_description) AS description
                      FROM policy_code WHERE {where}
                      ORDER BY code_system, role, group_no, code LIMIT %(limit)s""", params)
    out = {"public_id": doc["public_id"], "title": doc["title"], "code_lists": summary,
           "codes": codes[:limit], "truncated": len(codes) > limit}
    if not summary:
        rel = related_docs([doc["doc_pk"]]).get(doc["doc_pk"], [])
        out["note"] = "This document has no code lists; its codes are usually in a related Article."
        out["related"] = rel
    return out


@mcp.tool()
def search_policy_text(query_text: str, state: Optional[str] = None, doc_type: Optional[str] = None,
                       public_ids: Optional[list[str]] = None, include_code_lists: bool = False,
                       limit: int = 8) -> dict:
    """Search the text of active coverage policies (semantic + keyword), optionally within one state.

    Use this to find policies when you have no code, and to find the passages that state coverage
    criteria, limitations and documentation requirements.

    Args:
        query_text: A natural-language question or phrase, e.g. 'sleep study criteria for
            obstructive sleep apnea'. Describe the procedure; do not put the state in the text.
        state: Two-letter abbreviation or state name. Keeps national NCDs plus the LCDs and
            Articles that apply in that state. Omit to search every jurisdiction.
        doc_type: 'NCD', 'LCD' or 'ART' to search one document family.
        public_ids: Restrict the search to these documents, e.g. ['L33252', 'A57520'].
        include_code_lists: Also search the chunks that are plain code lists (off by default;
            lookup_code answers code questions exactly).
        limit: Number of passages to return (1-25).

    Returns passages ranked by relevance, each with its document, states, section and text.
    Use get_policy_text(public_id, section) to read a whole section around a passage.
    """
    q = query_text.strip()
    if not q:
        raise ToolError("query_text is empty")
    dt = doc_type.strip().upper() if doc_type else None
    if dt == "ARTICLE":
        dt = "ART"
    if dt and dt not in DOC_TYPES:
        raise ToolError("doc_type must be 'NCD', 'LCD' or 'ART'")
    states = resolve_states(state) if state else None
    limit = clamp(limit, 1, 25)
    vec = embed_query(q)
    params = {"q": q, "vec": vec, "model": MODEL, "states": states, "doc_type": dt,
              "ids": [norm_public_id(p) for p in public_ids] if public_ids else None,
              "codes": include_code_lists, "n": CANDIDATES, "k": RRF_K, "limit": limit}

    scope = """c.doc_pk IN (SELECT doc_pk FROM docs)
               AND (%(codes)s OR c.section NOT LIKE '%%\\_codes')"""
    vector_cte = f"""
        vec AS (SELECT chunk_id, row_number() OVER (ORDER BY dist) AS rank FROM (
                    SELECT c.chunk_id, e.embedding <=> %(vec)s::vector AS dist
                    FROM chunk_embedding e JOIN doc_chunk c USING (chunk_id)
                    WHERE e.model = %(model)s AND {scope}
                    ORDER BY dist LIMIT %(n)s) v),""" if vec else """
        vec AS (SELECT NULL::bigint AS chunk_id, NULL::bigint AS rank WHERE false),"""
    rows = query(f"""
        WITH docs AS (
            SELECT doc_pk FROM v_active_policy
            WHERE (%(states)s::text[] IS NULL OR doc_type = 'NCD' OR state_abbrevs && %(states)s)
              AND (%(doc_type)s::text IS NULL OR doc_type = %(doc_type)s)
              AND (%(ids)s::text[] IS NULL OR public_id = ANY(%(ids)s))),
        {vector_cte}
        -- any query word may match (plainto_tsquery would require all of them)
        tsq AS (SELECT NULLIF(replace(plainto_tsquery('english', %(q)s)::text, '&', '|'), '')::tsquery AS q),
        kw AS (SELECT chunk_id, row_number() OVER (ORDER BY score DESC) AS rank FROM (
                    SELECT c.chunk_id, ts_rank_cd(c.content_tsv, tsq.q, 32) AS score
                    FROM doc_chunk c, tsq
                    WHERE c.content_tsv @@ tsq.q AND {scope}
                    ORDER BY score DESC LIMIT %(n)s) s),
        fused AS (SELECT chunk_id, sum(1.0 / (%(k)s + rank)) AS score
                  FROM (SELECT * FROM vec UNION ALL SELECT * FROM kw) r
                  GROUP BY chunk_id ORDER BY score DESC LIMIT %(limit)s)
        SELECT f.score, c.section, c.heading_path, c.content, {DOC_COLS}
        FROM fused f JOIN doc_chunk c USING (chunk_id) JOIN policy_doc d USING (doc_pk)
        ORDER BY f.score DESC""", params)

    out = {"states_searched": states or "all",
           "results": [{**doc_summary(r, states), "section": r["section"], "heading": r["heading_path"],
                        "text": r["content"], "score": round(float(r["score"]), 4)} for r in rows]}
    if vec is None:
        out["note"] = f"Embedding model unavailable at {OLLAMA_HOST}; these are keyword-only results."
    if not rows:
        out["note"] = "No passages matched. Try broader wording, or drop the state / doc_type filter."
    return out


@mcp.tool()
def get_policy(public_id: str) -> dict:
    """Describe one policy document: status, dates, jurisdiction, MACs, related documents, sections.

    Call this to see where a policy applies, to follow LCD <-> Article <-> NCD links, and to
    learn which sections get_policy_text can read.

    Args:
        public_id: 'L33252' (LCD), 'A57520' (Article) or 'NCD 220.6.17' (also accepts '220.6.17').
    """
    doc = find_doc(public_id)
    pk = doc["doc_pk"]
    head = query(f"SELECT {DOC_COLS} FROM policy_doc d WHERE d.doc_pk = %s", (pk,))[0]
    out = doc_summary(head)
    out.update({"doc_type_name": DOC_TYPES[doc["doc_type"]],
                "status": STATUS.get(doc["status"], doc["status"]), "is_draft": doc["is_draft"],
                "end_date": doc["end_date"] and doc["end_date"].isoformat(),
                "retired_date": doc["retired_date"] and doc["retired_date"].isoformat()})
    if doc["doc_type"] == "NCD":
        ncd = query("""SELECT n.cvrg_lvl_cd, n.mnl_sect_title, n.under_review,
                              (SELECT array_agg(b.bnft_ctgry_desc ORDER BY b.bnft_ctgry_desc)
                                 FROM ncd_benefit_category x JOIN ncd_bnft_ctgry_ref b USING (bnft_ctgry_cd)
                                WHERE x.doc_pk = n.doc_pk) AS benefit_categories
                       FROM ncd n WHERE n.doc_pk = %s""", (pk,))
        if ncd:
            n = ncd[0]
            out.update({"coverage_level": NCD_COVERAGE.get(n["cvrg_lvl_cd"]),
                        "under_review": n["under_review"], "benefit_categories": n["benefit_categories"] or []})
    out["related"] = related_docs([pk]).get(pk, [])
    # documents that point at this one (an NCD or LCD does not always link back)
    out["referenced_by"] = query("""SELECT s.public_id, s.doc_type, s.title
                                    FROM policy_related r JOIN v_active_policy s USING (doc_pk)
                                    WHERE r.related_doc_type = %s AND r.related_doc_id = %s
                                    ORDER BY s.doc_type, s.public_id LIMIT 50""", (doc["doc_type"], doc["doc_id"]))
    out["code_lists"] = query("""SELECT code_system, role, count(*) AS codes FROM policy_code
                                 WHERE doc_pk = %s GROUP BY 1, 2 ORDER BY 1, 2""", (pk,))
    out["sections"] = query("""SELECT section, count(*) AS chunks, sum(length(content)) AS chars
                               FROM doc_chunk WHERE doc_pk = %s
                               GROUP BY section ORDER BY min(chunk_id)""", (pk,))
    if not out["sections"]:
        out["note"] = "No text is indexed for this document (only active, non-draft documents are chunked)."
    return out


@mcp.tool()
def get_policy_text(public_id: str, section: str, offset: int = 0, max_chars: int = 6000) -> dict:
    """Read one section of a policy in full, in document order.

    Args:
        public_id: Document id, e.g. 'L33252' or 'NCD 220.6.17'.
        section: A section name from get_policy or a search result. The coverage rules are in
            'indication' (LCD), 'indctn_lmtn' (NCD) and 'description' (Article); 'overview'
            is a short summary of any document.
        offset: Character offset to continue from when a previous call was truncated.
        max_chars: Maximum characters to return (500-20000).
    """
    doc = find_doc(public_id)
    rows = query("""SELECT heading_path, content FROM doc_chunk
                    WHERE doc_pk = %s AND section = %s
                    ORDER BY source_key, chunk_index""", (doc["doc_pk"], section.strip().lower()))
    if not rows:
        have = [r["section"] for r in query(
            "SELECT section FROM doc_chunk WHERE doc_pk = %s GROUP BY 1 ORDER BY min(chunk_id)", (doc["doc_pk"],))]
        raise ToolError(f"{doc['public_id']} has no section {section!r}; available sections: {', '.join(have) or 'none'}")
    parts, last = [], None
    for r in rows:  # print a heading only when it changes
        if r["heading_path"] != last:
            parts.append(f"## {r['heading_path']}")
            last = r["heading_path"]
        parts.append(r["content"])
    text = "\n\n".join(parts)
    offset, max_chars = max(0, offset), clamp(max_chars, 500, 20000)
    end = offset + max_chars
    out = {"public_id": doc["public_id"], "title": doc["title"], "section": section.strip().lower(),
           "total_chars": len(text), "text": text[offset:end]}
    if end < len(text):
        out["next_offset"] = end
    return out


if __name__ == "__main__":
    mcp.run()
