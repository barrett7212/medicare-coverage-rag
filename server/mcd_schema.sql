-- =====================================================================
-- CMS Medicare Coverage Database (MCD) -> PostgreSQL + pgvector schema
-- Sources: LCD, Article, and NCD download databases (.mdb)
--
-- Design:
--   1. Lookup / contractor tables   (mirrors CMS, mostly 1:1)
--   2. policy_doc                   (one row per LCD / Article / NCD version;
--                                    unifies the three document families)
--   3. lcd / article / ncd          (document-type-specific columns)
--   4. Unified crosswalks           (codes, URLs, revisions, sticky notes, ...)
--   5. Vector layer                 (doc_chunk + chunk_embedding)
--
-- NOTE: NOT NULL constraints are deliberately loosened vs. the CMS data
-- dictionaries (the LCD and Article dictionaries disagree with each other
-- on nullability for the same fields). Tighten after profiling real data.
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS vector;     -- pgvector >= 0.7 recommended
CREATE EXTENSION IF NOT EXISTS pg_trgm;    -- fuzzy title search

CREATE SCHEMA IF NOT EXISTS mcd;
SET search_path = mcd, public;

-- =====================================================================
-- 1. LOOKUPS
-- =====================================================================
CREATE TABLE state_lookup (
    state_id      integer PRIMARY KEY,
    state_abbrev  text NOT NULL,
    description   text NOT NULL
);

CREATE TABLE region_lookup (
    region_id     integer PRIMARY KEY,
    description   text NOT NULL
);

CREATE TABLE state_x_region (
    state_id      integer PRIMARY KEY REFERENCES state_lookup,
    region_id     integer NOT NULL REFERENCES region_lookup
);

CREATE TABLE dmerc_region_lookup (
    region_id            integer PRIMARY KEY,
    description          text,
    psc_description      text,
    mac_description      text,
    super_mac_description text
);

CREATE TABLE contractor_type_lookup (
    contractor_type_id integer PRIMARY KEY,
    description        text NOT NULL
);

CREATE TABLE contractor_subtype_lookup (
    contractor_subtype_id integer PRIMARY KEY,
    description           text NOT NULL
);

CREATE TABLE article_type_lookup (
    article_type_id integer PRIMARY KEY,
    description     text NOT NULL,
    last_updated    timestamptz
);

CREATE TABLE reason_change_lookup (
    reason_change_id      integer NOT NULL,
    reason_change_version integer NOT NULL,
    description           text NOT NULL,
    sort_order            integer,
    last_updated          timestamptz,
    PRIMARY KEY (reason_change_id, reason_change_version)
);

CREATE TABLE draft_contact_lookup (
    contact_id     integer PRIMARY KEY,
    email_address  text, first_name text, middle_initial text, last_name text,
    phone text, p_ext text,
    address1 text, address2 text, address3 text, city text,
    state_id integer REFERENCES state_lookup, zipcode text,
    last_updated timestamptz
);

CREATE TABLE synopsis_changes_fields_lookup (
    synopsis_changes_field_id integer PRIMARY KEY,
    field_name       text, field_anchor text,
    mcd_field_name   text, mcd_field_anchor text
);

-- LCD and Article have separate URL-type lookups; IDs may overlap, so key on doc_type too.
CREATE TABLE url_type_lookup (
    doc_type     char(3) NOT NULL CHECK (doc_type IN ('LCD','ART')),
    url_type_id  integer NOT NULL,
    description  text NOT NULL,
    sort_order   integer,
    last_updated timestamptz,
    PRIMARY KEY (doc_type, url_type_id)
);

CREATE TABLE ncd_bnft_ctgry_ref (
    bnft_ctgry_cd   integer PRIMARY KEY,
    bnft_ctgry_desc text NOT NULL
);

CREATE TABLE ncd_pblctn_ref (
    pblctn_cd    integer PRIMARY KEY,
    pblctn_num   text NOT NULL,
    pblctn_title text NOT NULL
);

-- =====================================================================
-- 1b. CONTRACTORS (MACs)
-- =====================================================================
CREATE TABLE contractor (
    contractor_id       integer NOT NULL,
    contractor_type_id  integer NOT NULL REFERENCES contractor_type_lookup,
    contractor_version  integer NOT NULL,
    contractor_bus_name text,
    contractor_number   text,
    dmerc_rgn           integer,            -- DMERC contractors only
    address1 text, address2 text, address3 text, city text,
    state_id            integer REFERENCES state_lookup,
    zipcode text, phone text, fax text, url text, email text,
    status              char(1),            -- A=approved, D=deleted
    status_flag         char(1),            -- per dictionary: Y=not retired, N=retired
    ignore_flag         char(1),            -- source column "ignore" (reserved word-ish)
    cmd_name text, cmd_title text,
    contractor_subtype_id integer,          -- present in real files, missing from dictionary
    last_updated        timestamptz,
    PRIMARY KEY (contractor_id, contractor_type_id, contractor_version)
);

CREATE TABLE contractor_jurisdiction (
    contractor_id      integer NOT NULL,
    contractor_type_id integer NOT NULL,
    contractor_version integer NOT NULL,
    state_id           integer NOT NULL REFERENCES state_lookup,
    active_date        date,
    term_date          date,
    last_updated       timestamptz,
    PRIMARY KEY (contractor_id, contractor_type_id, contractor_version, state_id),
    FOREIGN KEY (contractor_id, contractor_type_id, contractor_version)
        REFERENCES contractor
);

CREATE TABLE contractor_oversight (
    contractor_id      integer NOT NULL,
    contractor_type_id integer NOT NULL,
    contractor_version integer NOT NULL,
    region_id          integer NOT NULL REFERENCES region_lookup,
    last_updated       timestamptz,
    PRIMARY KEY (contractor_id, contractor_type_id, contractor_version, region_id),
    FOREIGN KEY (contractor_id, contractor_type_id, contractor_version)
        REFERENCES contractor
);

-- =====================================================================
-- 2. POLICY_DOC: unified header for LCD / Article / NCD versions
-- =====================================================================
CREATE TABLE policy_doc (
    doc_pk          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    doc_type        char(3) NOT NULL CHECK (doc_type IN ('LCD','ART','NCD')),
    doc_id          integer NOT NULL,      -- lcd_id / article_id / NCD_id
    doc_version     integer NOT NULL,      -- lcd_version / article_version / NCD_vrsn_num
    display_id      text,                  -- non-null => Proposed LCD (DL...) or Draft Article (DA...)
    public_id       text NOT NULL,         -- 'L33999', 'A56789', 'DL12345', 'DA12345', 'NCD 220.6.17'
    title           text NOT NULL,
    status          char(1),               -- A approved, R retired, P proposed (LCD)
    is_draft        boolean NOT NULL DEFAULT false,
    is_latest       boolean NOT NULL DEFAULT true,   -- LCD/Article downloads only contain max version
    effective_date  date,                  -- lcd.rev_eff_date | article_eff_date | NCD_efctv_dt
    end_date        date,                  -- lcd.rev_end_date | article_end_date | NCD_trmntn_dt
    retired_date    date,
    published_date  date,
    last_updated    timestamptz,
    icd10_doc       boolean,
    cms_cov_policy  text,                  -- LCD/Article: related national policy text
    keywords        text,
    -- denormalized filter columns (populate after load; see section 6)
    state_abbrevs   text[]  NOT NULL DEFAULT '{}',
    contractor_ids  integer[] NOT NULL DEFAULT '{}',
    hcpcs_codes     text[]  NOT NULL DEFAULT '{}',
    icd10_codes     text[]  NOT NULL DEFAULT '{}',
    UNIQUE (doc_type, doc_id, doc_version)
);
CREATE INDEX policy_doc_public_id_idx ON policy_doc (public_id);
CREATE INDEX policy_doc_title_trgm    ON policy_doc USING gin (title gin_trgm_ops);
CREATE INDEX policy_doc_states_gin    ON policy_doc USING gin (state_abbrevs);
CREATE INDEX policy_doc_contr_gin     ON policy_doc USING gin (contractor_ids);
CREATE INDEX policy_doc_hcpcs_gin     ON policy_doc USING gin (hcpcs_codes);
CREATE INDEX policy_doc_icd10_gin     ON policy_doc USING gin (icd10_codes);
CREATE INDEX policy_doc_active_idx    ON policy_doc (doc_type, status, is_latest);

-- =====================================================================
-- 3. TYPE-SPECIFIC TABLES
-- =====================================================================
CREATE TABLE lcd (
    doc_pk                  bigint PRIMARY KEY REFERENCES policy_doc ON DELETE CASCADE,
    determination_number    text,
    orig_det_eff_date       date,
    ent_det_end_date        date,
    -- narrative sections (RAG text sources)
    issue                   text,
    issue_change            text,
    indication              text,
    diagnoses_support       text,
    diagnoses_dont_support  text,
    icd9_dont_support_para  text,
    icd9_dont_support_ast   text,
    coding_guidelines       text,
    doc_reqs                text,
    appendices              text,
    util_guide              text,
    source_info             text,
    summary_of_evidence     text,
    analysis_of_evidence    text,
    bibliography            text,
    associated_info         text,
    add_icd10_info          text,
    revenue_para            text,
    synopsis_changes        text,           -- Proposed LCDs only
    history_exp             text,
    rev_hist_num            integer,
    last_reviewed_on        date,           -- present in real files, missing from dictionary
    adv_meeting             text,
    -- process dates / flags
    comment_start_dt        date,
    comment_end_dt          date,
    notice_start_dt         date,
    notice_end_dt           date,
    mcd_publish_date        date,
    draft_released_date     date,
    source_lcd_id           integer,        -- original ICD-9 LCD
    draft_contact           integer REFERENCES draft_contact_lookup,
    mac_initiated           char(1),
    thirty_percent          char(1)
);

CREATE TABLE article (
    doc_pk                  bigint PRIMARY KEY REFERENCES policy_doc ON DELETE CASCADE,
    article_type_id         integer REFERENCES article_type_lookup,
    description             text,           -- main body
    other_comments          text,
    history_exp             text,
    add_icd10_info          text,
    icd9_covered_para       text,
    icd9_noncovered_para    text,
    revenue_para            text,
    sad_url                 text,           -- SAD Exclusion List articles only
    key_article             char(1),
    reference_article       char(1),        -- present in real files, missing from dictionary
    source_article_id       integer,
    article_rev_end_date    date,
    thirty_percent          char(1)
);

CREATE TABLE ncd (
    doc_pk              bigint PRIMARY KEY REFERENCES policy_doc ON DELETE CASCADE,
    is_ncd              boolean,            -- natl_cvrg_type: true=NCD, false=coverage provision
    cvrg_lvl_cd         smallint,           -- 1 full, 2 restricted, 3 none
    mnl_sect            text,               -- manual section number, e.g. '220.6.17'
    mnl_sect_title      text,
    implementation_date date,
    itm_srvc_desc       text,
    indctn_lmtn         text,
    xref_txt            text,
    othr_txt            text,
    rev_hstry           text,
    trnsmtl_num         text,
    trnsmtl_issnc_dt    date,
    trnsmtl_url         text,
    chg_rqst_num        text,
    pblctn_cd           integer REFERENCES ncd_pblctn_ref,
    under_review        boolean,
    is_lab_ncd          boolean,
    ama_notice          boolean,            -- show AMA CPT copyright notice
    created_ts          timestamptz,
    last_updated_ts     timestamptz,
    last_cleared_ts     timestamptz         -- greatest(updated, cleared) = latest change
);

CREATE TABLE ncd_benefit_category (
    doc_pk          bigint  NOT NULL REFERENCES ncd ON DELETE CASCADE,
    bnft_ctgry_cd   integer NOT NULL REFERENCES ncd_bnft_ctgry_ref,
    PRIMARY KEY (doc_pk, bnft_ctgry_cd)
);

-- =====================================================================
-- 4. UNIFIED CROSSWALKS (shared by LCD + Article via doc_pk)
-- =====================================================================

-- Jurisdiction resolved at load time:
--   1 contractor on the doc  -> *_X_PRIMARY_JURISDICTION
--   >1 contractors           -> CONTRACTOR_JURISDICTION
CREATE TABLE policy_contractor (
    doc_pk             bigint  NOT NULL REFERENCES policy_doc ON DELETE CASCADE,
    contractor_id      integer NOT NULL,
    contractor_type_id integer NOT NULL,
    contractor_version integer NOT NULL,
    PRIMARY KEY (doc_pk, contractor_id, contractor_type_id, contractor_version),
    FOREIGN KEY (contractor_id, contractor_type_id, contractor_version) REFERENCES contractor
);

CREATE TABLE policy_state (
    doc_pk     bigint  NOT NULL REFERENCES policy_doc ON DELETE CASCADE,
    state_id   integer NOT NULL REFERENCES state_lookup,
    PRIMARY KEY (doc_pk, state_id)
);

-- One table for ALL code lists.
--   code_system : HCPCS | HCPCS_MOD | ICD10CM | ICD10PCS | REV | BILL
--   role        : listed (HCPCS/mods/rev/bill) | covered | noncovered
--   range_pos   : B/E/M/N (and X/Y/Z for revenue "X" ranges)
-- NOTE: LCDs carry only HCPCS codes. ICD-10 covered/noncovered codes live on
--       Articles, which are linked to LCDs via policy_related.
CREATE TABLE policy_code (
    code_pk           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    doc_pk            bigint NOT NULL REFERENCES policy_doc ON DELETE CASCADE,
    code_system       text   NOT NULL CHECK (code_system IN
                        ('HCPCS','HCPCS_MOD','ICD10CM','ICD10PCS','REV','BILL')),
    role              text   NOT NULL DEFAULT 'listed'
                        CHECK (role IN ('listed','covered','noncovered')),
    code              text   NOT NULL,
    code_version      integer,
    group_no          integer NOT NULL DEFAULT 0,
    range_pos         char(1),
    sort_order        integer,
    description       text,           -- long description
    short_description text,
    has_asterisk      boolean,        -- ICD10 covered: group has an explanation
    last_updated      timestamptz
);
CREATE UNIQUE INDEX policy_code_uq
    ON policy_code (doc_pk, code_system, role, group_no, code, COALESCE(code_version,0), COALESCE(range_pos,'-'));
CREATE INDEX policy_code_lookup_idx ON policy_code (code_system, code);   -- "which policies mention E11.9?"
CREATE INDEX policy_code_prefix_idx ON policy_code (code_system, code text_pattern_ops);

CREATE TABLE policy_code_group (
    doc_pk       bigint NOT NULL REFERENCES policy_doc ON DELETE CASCADE,
    code_system  text   NOT NULL,
    role         text   NOT NULL DEFAULT 'listed',
    group_no     integer NOT NULL,
    paragraph    text,               -- intro text for the code section (RAG text source)
    asterisk_text text,              -- icd10_covered_ast
    last_updated timestamptz,
    PRIMARY KEY (doc_pk, code_system, role, group_no)
);

-- SAD Exclusion List articles
CREATE TABLE article_code_table (
    doc_pk            bigint  NOT NULL REFERENCES article ON DELETE CASCADE,
    code_table_row    integer NOT NULL,
    hcpc_code_id      text,
    hcpc_code_version integer,
    brand_name        text,
    eff_date          date,
    end_date          date,
    comments          text,
    long_description  text,
    short_description text,
    last_updated      timestamptz,
    PRIMARY KEY (doc_pk, code_table_row)
);

CREATE TABLE article_other_coding_group (
    doc_pk       bigint  NOT NULL REFERENCES article ON DELETE CASCADE,
    other_coding_group integer NOT NULL,
    paragraph    text,
    codes        text,
    last_updated timestamptz,
    PRIMARY KEY (doc_pk, other_coding_group)
);

-- Related documents. Links always point at the LATEST version of the target,
-- and the target may not be in the download -> no FK on target.
CREATE TABLE policy_related (
    doc_pk           bigint  NOT NULL REFERENCES policy_doc ON DELETE CASCADE,
    related_doc_type char(3) NOT NULL CHECK (related_doc_type IN ('LCD','ART','NCD')),
    related_doc_id   integer NOT NULL,       -- NCD id 0 = "N/A": skipped on load
    related_num      integer,                -- counters restart per source table, so not part of the key
    related_contractor_id integer,
    last_updated     timestamptz,
    PRIMARY KEY (doc_pk, related_doc_type, related_doc_id)
);
CREATE INDEX policy_related_target_idx ON policy_related (related_doc_type, related_doc_id);

CREATE TABLE policy_source_icd9 (
    doc_pk        bigint  NOT NULL REFERENCES policy_doc ON DELETE CASCADE,
    related_num   integer NOT NULL,
    source_doc_id integer NOT NULL,
    last_updated  timestamptz,
    PRIMARY KEY (doc_pk, related_num)
);

CREATE TABLE policy_revision_history (
    doc_pk        bigint  NOT NULL REFERENCES policy_doc ON DELETE CASCADE,
    rev_hist_num  integer NOT NULL,
    rev_hist_date date,
    rev_hist_exp  text,
    last_updated  timestamptz,
    PRIMARY KEY (doc_pk, rev_hist_num)
);

CREATE TABLE policy_sticky_note (
    doc_pk              bigint  NOT NULL REFERENCES policy_doc ON DELETE CASCADE,
    sticky_note_version integer NOT NULL,
    sticky_note         text,
    sticky_note_dt      timestamptz,
    sticky_note_posting_dt timestamptz,
    PRIMARY KEY (doc_pk, sticky_note_version)
);

CREATE TABLE policy_url (
    doc_pk          bigint  NOT NULL REFERENCES policy_doc ON DELETE CASCADE,
    doc_type        char(3) NOT NULL,
    url_type_id     integer NOT NULL,
    url_id          integer NOT NULL,
    url             text,
    url_name        text,
    url_description text,
    sort_order      integer,
    last_updated    timestamptz,
    PRIMARY KEY (doc_pk, url_type_id, url_id),
    FOREIGN KEY (doc_type, url_type_id) REFERENCES url_type_lookup
);

CREATE TABLE article_response_to_comment (
    doc_pk    bigint  NOT NULL REFERENCES article ON DELETE CASCADE,
    rtc_num   integer NOT NULL,
    comment   text,
    response  text,
    last_updated timestamptz,
    PRIMARY KEY (doc_pk, rtc_num)
);

-- LCD-only crosswalks
CREATE TABLE lcd_reason_change (
    doc_pk                bigint  NOT NULL REFERENCES lcd ON DELETE CASCADE,
    reason_change_id      integer NOT NULL,
    reason_change_version integer NOT NULL,
    reason_change_other   text,
    last_updated          timestamptz,
    PRIMARY KEY (doc_pk, reason_change_id, reason_change_version),
    FOREIGN KEY (reason_change_id, reason_change_version) REFERENCES reason_change_lookup
);

CREATE TABLE lcd_advisory_committee (        -- Proposed LCDs only
    doc_pk       bigint  NOT NULL REFERENCES lcd ON DELETE CASCADE,
    meeting_id   integer NOT NULL,
    meeting_date date,
    meeting_info text,
    sort_order   integer,
    last_updated timestamptz,
    PRIMARY KEY (doc_pk, meeting_id)
);

CREATE TABLE lcd_synopsis_changes_field (    -- Proposed LCDs only
    doc_pk bigint  NOT NULL REFERENCES lcd ON DELETE CASCADE,
    synopsis_changes_field_id integer NOT NULL REFERENCES synopsis_changes_fields_lookup,
    last_updated timestamptz,
    PRIMARY KEY (doc_pk, synopsis_changes_field_id)
);

CREATE TABLE lcd_requestor_letter (          -- Proposed LCDs only
    doc_pk         bigint  NOT NULL REFERENCES lcd ON DELETE CASCADE,
    letter_id      integer NOT NULL,
    requestor_name text,
    letter_path    text,
    size           text,
    sort_order     integer,
    last_updated   timestamptz,
    PRIMARY KEY (doc_pk, letter_id)
);

-- Future retirements (keyed on id, not version)
CREATE TABLE policy_future_retire (
    doc_type char(3) NOT NULL CHECK (doc_type IN ('LCD','ART')),
    doc_id   integer NOT NULL,
    retire_dt date NOT NULL,
    PRIMARY KEY (doc_type, doc_id)
);

CREATE TABLE update_period (
    period_id  integer PRIMARY KEY,
    begin_date date NOT NULL,
    end_date   date NOT NULL
);

-- Optional reference table for code descriptions that are NOT in these downloads.
-- ICD-10-CM / ICD-10-PCS / HCPCS Level II are public; CPT descriptions are AMA-licensed.
CREATE TABLE code_ref (
    code_system text NOT NULL,
    code        text NOT NULL,
    description text,
    valid_from  date,
    valid_to    date,
    PRIMARY KEY (code_system, code)
);

-- =====================================================================
-- 5. VECTOR LAYER
-- =====================================================================
-- One row per retrievable text chunk. Each chunk traces back to a doc
-- (doc_pk) and a specific section/source row for citations.
CREATE TABLE doc_chunk (
    chunk_id     bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    doc_pk       bigint NOT NULL REFERENCES policy_doc ON DELETE CASCADE,
    section      text   NOT NULL,   -- 'indication','coding_guidelines','description','icd10_covered_paragraph',
                                    -- 'analysis_of_evidence','sticky_note','itm_srvc_desc','indctn_lmtn', ...
    source_table text   NOT NULL,   -- e.g. 'lcd','article','policy_code_group','policy_sticky_note'
    source_key   text NOT NULL DEFAULT '',  -- e.g. 'group=2' for traceability ('' when n/a)
    chunk_index  integer NOT NULL DEFAULT 0,
    heading_path text,              -- 'L33999 > Coding Guidelines' (prepend to chunk text before embedding)
    content      text   NOT NULL,
    token_count  integer,
    content_hash char(64) NOT NULL, -- sha256; skip re-embedding unchanged chunks
    content_tsv  tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (doc_pk, section, source_key, chunk_index)
);
CREATE INDEX doc_chunk_doc_idx ON doc_chunk (doc_pk);
CREATE INDEX doc_chunk_hash_idx ON doc_chunk (content_hash);          -- embedding reuse across versions
CREATE INDEX doc_chunk_tsv_idx ON doc_chunk USING gin (content_tsv);   -- keyword half of hybrid search

-- Embeddings kept separate so you can swap/compare models without touching chunks.
-- Dimension is fixed per column: 768 fits nomic-embed-text (Ollama). Use 1024 for bge-m3 / mxbai-embed-large.
-- For a second model with a different dimension, create a second table (chunk_embedding_1024, ...).
CREATE TABLE chunk_embedding (
    chunk_id   bigint NOT NULL REFERENCES doc_chunk ON DELETE CASCADE,
    model      text   NOT NULL,                 -- 'nomic-embed-text:v1.5'
    embedding  vector(768) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (chunk_id, model)
);
CREATE INDEX chunk_embedding_hnsw
    ON chunk_embedding USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- =====================================================================
-- 6. POST-LOAD: populate denormalized filter columns on policy_doc
-- =====================================================================
-- UPDATE policy_doc d SET state_abbrevs = COALESCE((
--     SELECT array_agg(DISTINCT s.state_abbrev)
--     FROM policy_state ps JOIN state_lookup s USING (state_id)
--     WHERE ps.doc_pk = d.doc_pk), '{}');
--
-- UPDATE policy_doc d SET contractor_ids = COALESCE((
--     SELECT array_agg(DISTINCT contractor_id) FROM policy_contractor pc WHERE pc.doc_pk = d.doc_pk), '{}');
--
-- UPDATE policy_doc d SET hcpcs_codes = COALESCE((
--     SELECT array_agg(DISTINCT code) FROM policy_code pc
--     WHERE pc.doc_pk = d.doc_pk AND pc.code_system = 'HCPCS'), '{}');
--
-- UPDATE policy_doc d SET icd10_codes = COALESCE((
--     SELECT array_agg(DISTINCT code) FROM policy_code pc
--     WHERE pc.doc_pk = d.doc_pk AND pc.code_system = 'ICD10CM'), '{}');

-- =====================================================================
-- 7. VIEWS
-- =====================================================================
-- Currently effective, non-draft policies
CREATE VIEW v_active_policy AS
SELECT *
FROM policy_doc
WHERE status = 'A'
  AND NOT is_draft
  AND is_latest
  AND (effective_date IS NULL OR effective_date <= CURRENT_DATE)
  AND (end_date IS NULL OR end_date >= CURRENT_DATE);

-- Code -> policy (exact match; this is where code lookups should happen, not in the vector index)
CREATE VIEW v_code_to_policy AS
SELECT pc.code_system, pc.code, pc.role, pc.group_no, pc.description,
       d.doc_pk, d.doc_type, d.public_id, d.title, d.state_abbrevs, d.effective_date
FROM policy_code pc
JOIN v_active_policy d USING (doc_pk);

-- LCD -> related Articles (where the ICD-10 coverage lists live)
CREATE VIEW v_lcd_articles AS
SELECT l.doc_pk AS lcd_doc_pk, l.public_id AS lcd_public_id,
       a.doc_pk AS article_doc_pk, a.public_id AS article_public_id, a.title AS article_title
FROM policy_doc l
JOIN policy_related r ON r.doc_pk = l.doc_pk AND r.related_doc_type = 'ART'
JOIN policy_doc a     ON a.doc_type = 'ART' AND a.doc_id = r.related_doc_id AND a.is_latest
WHERE l.doc_type = 'LCD';

-- =====================================================================
-- 8. EXAMPLE RETRIEVAL PATTERN (hybrid: structured filter + vector)
-- =====================================================================
-- Step 1 (SQL, exact): policies in the user's state that list CPT 95810
--   SELECT DISTINCT doc_pk FROM v_code_to_policy
--   WHERE code_system='HCPCS' AND code='95810' AND 'MN' = ANY(state_abbrevs);
--
-- Step 2 (vector, restricted to those docs):
--   SELECT c.chunk_id, c.doc_pk, c.section, c.content,
--          e.embedding <=> :query_vec AS distance
--   FROM chunk_embedding e
--   JOIN doc_chunk c USING (chunk_id)
--   WHERE e.model = :model
--     AND c.doc_pk = ANY(:doc_pks)
--   ORDER BY e.embedding <=> :query_vec
--   LIMIT 20;
--
-- Step 3: merge with keyword hits (c.content_tsv @@ websearch_to_tsquery(...))
--         using reciprocal rank fusion, then rerank / pass to the LLM.
