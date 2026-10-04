#!/usr/bin/env python3
"""Pipeline: fetch -> ETL -> chunk -> embed in one command, for the first load and the weekly refresh.

Typical use
    python mcd_pipeline.py --accept-licenses              # download, load, chunk and embed everything
    python mcd_pipeline.py --accept-licenses --retired    # "Current and Retired" LCD/Article variants
    python mcd_pipeline.py --accept-licenses --only ncd   # just one dataset
    python mcd_pipeline.py --skip-fetch                   # datasets are already in ./data
    python mcd_pipeline.py --accept-licenses --skip-embed # stop after chunking (Ollama not needed)

Stages, each of which is also a script of its own with more options (see its --help):
    mcd_fetch.py    CMS downloads page -> ./data          (skipped per dataset when the ETag is unchanged)
    mcd_etl.py      datasets -> Postgres                  (creates the schema if it does not exist)
    mcd_chunk.py    policy text -> doc_chunk              (only touches chunks whose text changed)
    mcd_embed.py    doc_chunk -> chunk_embedding          (only chunks without an embedding; needs Ollama)

Every stage is safe to re-run, so the same command serves the first load and the refresh. The
pipeline stops at the first stage that fails and exits with that stage's exit code. The stages
before it are already committed, so if the embedding fails (Ollama not running, model not pulled)
fix that and run mcd_embed.py, or the pipeline again.

There is no --dry-run: the ETL's dry run rolls back, so the chunker and the embedder would run
against the old data. Use the --dry-run of the individual scripts instead.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import mcd_chunk
import mcd_embed
import mcd_etl
import mcd_fetch

# dataset kind (mcd_fetch.py / mcd_etl.py --only) -> document type (mcd_chunk.py --types)
DOC_TYPES = {"lcd": "LCD", "article": "ART", "ncd": "NCD"}
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accept-licenses", action="store_true",
                    help="you have read and accept the AMA (CPT), ADA (CDT) and AHA/NUBC (UB-04) "
                         f"license agreements shown at {mcd_fetch.PAGE_URL}")
    ap.add_argument("--retired", action="store_true",
                    help='fetch the "Current and Retired" LCD and Article datasets instead of current only')
    ap.add_argument("--only", nargs="+", choices=list(DOC_TYPES), help="run the pipeline for these datasets only")
    ap.add_argument("--dsn", default=None, help="Postgres DSN (default: $MCD_DSN or postgresql://localhost/mcd)")
    ap.add_argument("--skip-fetch", action="store_true",
                    help="do not download; load the datasets already in $MCD_DATA_DIR or ./data")
    ap.add_argument("--skip-embed", action="store_true",
                    help="stop after chunking; run mcd_embed.py later (it needs a running Ollama)")
    args = ap.parse_args(argv)

    only = ["--only", *args.only] if args.only else []
    dsn = ["--dsn", args.dsn] if args.dsn else []

    fetch = [*only]
    if args.accept_licenses:
        fetch.append("--accept-licenses")
    if args.retired:
        fetch.append("--retired")

    chunk = [*dsn]
    if args.only:
        # a partial run must not prune the chunks of the document types it leaves out
        chunk += ["--types", *(DOC_TYPES[k] for k in args.only), "--no-prune"]

    stages = [("etl", mcd_etl.main, ["--init-schema", *only, *dsn]),
              ("chunk", mcd_chunk.main, chunk)]
    if not args.skip_fetch:
        stages.insert(0, ("fetch", mcd_fetch.main, fetch))
    if not args.skip_embed:
        # not narrowed by --only: whatever has no embedding yet is embedded
        stages.append(("embed", mcd_embed.main, [*dsn]))

    for name, stage, stage_argv in stages:
        log(f"=== {name}")
        rc = stage(stage_argv)
        if rc:
            print(f"error: {name} failed (exit code {rc}); pipeline stopped", file=sys.stderr)
            return rc
    log("pipeline done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
