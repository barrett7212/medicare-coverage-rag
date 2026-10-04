#!/usr/bin/env python3
"""Embed doc_chunk rows into chunk_embedding with a local Ollama model.

    python mcd_embed.py                       # every chunk that has no embedding for the model yet
    python mcd_embed.py --dry-run             # count what would be embedded, write nothing
    python mcd_embed.py --model nomic-embed-text --batch 32
    python mcd_embed.py --force               # drop this model's embeddings and embed everything again

What is embedded
  heading_path + blank line + content, with the model's document prefix in front (nomic-embed-text
  is trained with "search_document: " / "search_query: " task prefixes; queries must use the
  matching query prefix - see PREFIXES).

The run is incremental and resumable: mcd_chunk.py deletes the embedding of a chunk whose text
changed, so "no embedding for this model" is exactly the work that is left. Chunks are grouped
by content_hash (sha256 of heading_path + content), so identical text is embedded once, and a
chunk whose hash already has an embedding elsewhere copies it instead of calling the model.
Every batch is committed, so an interrupted run picks up where it stopped.

The embedding column has a fixed dimension (vector(768) in mcd_schema.sql). A model with another
dimension is refused before anything is written.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import requests

from mcd_common import connect

MODEL = os.environ.get("MCD_EMBED_MODEL", "nomic-embed-text")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# model family -> (document prefix, query prefix). Models not listed here get no prefix.
PREFIXES = {"nomic-embed-text": ("search_document: ", "search_query: ")}

MISSING = "NOT EXISTS (SELECT 1 FROM chunk_embedding x WHERE x.chunk_id = c.chunk_id AND x.model = %(model)s)"


def doc_prefix(model: str) -> str:
    return PREFIXES.get(model.split(":")[0], ("", ""))[0]


def ollama_url(host: str) -> str:
    """OLLAMA_HOST may be a bare host:port (that is how Ollama itself reads it)."""
    host = host.strip().rstrip("/")
    return host if "://" in host else f"http://{host}"


def embed(session: requests.Session, url: str, model: str, texts: list[str]) -> list[list[float]]:
    try:
        r = session.post(f"{url}/api/embed", json={"model": model, "input": texts}, timeout=600)
    except requests.ConnectionError as e:
        raise RuntimeError(f"cannot reach Ollama at {url} (start it with `ollama serve`): {e}") from e
    if r.status_code != 200:
        try:
            detail = r.json().get("error", r.text)
        except ValueError:
            detail = r.text
        hint = f" (pull it with `ollama pull {model}`)" if r.status_code == 404 else ""
        raise RuntimeError(f"Ollama returned {r.status_code}: {detail}{hint}")
    vectors = r.json().get("embeddings") or []
    if len(vectors) != len(texts):
        raise RuntimeError(f"Ollama returned {len(vectors)} embeddings for {len(texts)} texts")
    return vectors


def column_dim(conn) -> int:
    """Dimension of chunk_embedding.embedding (pgvector keeps it in the type modifier)."""
    return conn.execute("""SELECT atttypmod FROM pg_attribute
                           WHERE attrelid = 'chunk_embedding'::regclass AND attname = 'embedding'""").fetchone()[0]


def reuse_by_hash(conn, model: str) -> int:
    """Copy an existing embedding to chunks with the same content_hash that lack one."""
    cur = conn.execute(f"""INSERT INTO chunk_embedding (chunk_id, model, embedding)
                           SELECT DISTINCT ON (c.chunk_id) c.chunk_id, %(model)s, e.embedding
                           FROM doc_chunk c
                           JOIN doc_chunk s ON s.content_hash = c.content_hash AND s.chunk_id <> c.chunk_id
                           JOIN chunk_embedding e ON e.chunk_id = s.chunk_id AND e.model = %(model)s
                           WHERE {MISSING}
                           ORDER BY c.chunk_id, e.created_at DESC
                           ON CONFLICT DO NOTHING""", {"model": model})
    return cur.rowcount


def pending(conn, model: str, limit: Optional[int]) -> list[tuple[str, list[int]]]:
    """-> [(content_hash, [chunk_id, ...])] for chunks without an embedding, one entry per distinct text."""
    sql = f"""SELECT c.content_hash, array_agg(c.chunk_id ORDER BY c.chunk_id)
              FROM doc_chunk c WHERE {MISSING}
              GROUP BY c.content_hash ORDER BY min(c.chunk_id)"""
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql, {"model": model}).fetchall()


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn")
    ap.add_argument("--model", default=MODEL,
                    help=f"Ollama embedding model; also the chunk_embedding.model value (default: $MCD_EMBED_MODEL or {MODEL})")
    ap.add_argument("--host", default=OLLAMA_HOST, help="Ollama base URL (default: $OLLAMA_HOST or http://localhost:11434)")
    ap.add_argument("--prefix", default=None,
                    help='text put in front of every chunk (default: the model\'s document prefix, "" if it has none)')
    ap.add_argument("--batch", type=int, default=64, help="texts per Ollama request and per commit (default 64)")
    ap.add_argument("--limit", type=int, help="only the first N distinct texts (development)")
    ap.add_argument("--force", action="store_true", help="delete this model's embeddings first and embed everything again")
    ap.add_argument("--dry-run", action="store_true", help="report what would be embedded; do not call Ollama, write nothing")
    args = ap.parse_args(argv)

    url = ollama_url(args.host)
    prefix = doc_prefix(args.model) if args.prefix is None else args.prefix
    conn = connect(args.dsn)
    t0 = time.time()

    if args.force:
        cur = conn.execute("DELETE FROM chunk_embedding WHERE model = %s", (args.model,))
        print(f"--force: {cur.rowcount} existing {args.model} embeddings deleted", flush=True)
    reused = reuse_by_hash(conn, args.model)
    todo = pending(conn, args.model, args.limit)
    n_chunks = sum(len(ids) for _, ids in todo)
    print(f"{n_chunks} chunks to embed ({len(todo)} distinct texts), {reused} reused by content hash", flush=True)
    if args.dry_run:
        conn.rollback()
        print("dry run: rolled back")
        return 0
    conn.commit()
    if not todo:
        return 0

    dim = column_dim(conn)
    session = requests.Session()
    done = 0
    for i in range(0, len(todo), args.batch):
        batch = todo[i:i + args.batch]
        text = dict(conn.execute("""SELECT chunk_id, concat_ws(E'\\n\\n', heading_path, content)
                                    FROM doc_chunk WHERE chunk_id = ANY(%s)""", ([ids[0] for _, ids in batch],)))
        # a chunk can vanish if mcd_chunk.py runs at the same time; skip it
        batch = [(h, ids) for h, ids in batch if ids[0] in text]
        if not batch:
            continue
        try:
            vectors = embed(session, url, args.model, [prefix + text[ids[0]] for _, ids in batch])
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr)
            print(f"{done} chunks embedded before the failure; re-run to continue", file=sys.stderr)
            return 1
        if len(vectors[0]) != dim:
            print(f"error: {args.model} returns {len(vectors[0])}-dimensional vectors but chunk_embedding.embedding "
                  f"is vector({dim}); use a {dim}-dimensional model or change the schema", file=sys.stderr)
            return 1
        rows = [(cid, args.model, "[" + ",".join(map(repr, v)) + "]", cid)
                for (_, ids), v in zip(batch, vectors) for cid in ids]
        conn.cursor().executemany("""INSERT INTO chunk_embedding (chunk_id, model, embedding)
                                     SELECT %s, %s, %s::vector
                                     WHERE EXISTS (SELECT 1 FROM doc_chunk WHERE chunk_id = %s)
                                     ON CONFLICT DO NOTHING""", rows)
        conn.commit()
        done += len(rows)
        if (i // args.batch) % 10 == 9 or i + args.batch >= len(todo):
            el = time.time() - t0
            print(f"  {done}/{n_chunks} chunks  ({el:.0f}s, {done / el:.0f}/s)", flush=True)

    print(f"\n{done} chunks embedded with {args.model} ({dim} dimensions) in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
