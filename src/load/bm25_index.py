"""Writes chunks to a local BM25 sparse index (spec.md §6, TASKS.md SEARCH-2), the lexical
sibling of `load/vector_store.py`'s dense index. `bm25s` has no partial-index update, so
every write rebuilds the whole index from a manifest of chunk rows kept alongside it —
delete-then-insert idempotency (CLAUDE.md "Idempotency") implemented as a full rebuild
rather than an in-place edit. Purely local (CLAUDE.md "Retrieval is local-first"): no
network call, no model download.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import bm25s

from models.schemas import Chunk

_MANIFEST_NAME = "manifest.jsonl"


def bm25_dir(store_dir: Path) -> Path:
    return store_dir / "bm25"


def _manifest_path(store_dir: Path) -> Path:
    return bm25_dir(store_dir) / _MANIFEST_NAME


def _row(chunk: Chunk) -> dict[str, object]:
    return {
        "doc_id": chunk.doc_id,
        "chunk_id": chunk.chunk_id,
        "start": chunk.start,
        "end": chunk.end,
        "segment_ids": chunk.segment_ids,
        "display_text": chunk.display_text,
    }


def read_rows(store_dir: Path, doc_id: str | None = None) -> list[dict[str, object]]:
    """Every row in the BM25 manifest, optionally restricted to one `doc_id`. Empty list
    if nothing has been indexed yet."""
    path = _manifest_path(store_dir)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if doc_id is not None:
        rows = [row for row in rows if row["doc_id"] == doc_id]
    return rows


def _rebuild(store_dir: Path, rows: list[dict[str, object]]) -> None:
    directory = bm25_dir(store_dir)
    if not rows:
        drop_index(store_dir)
        return

    tokens = bm25s.tokenize(
        [str(row["display_text"]) for row in rows], stopwords=None, show_progress=False
    )
    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=False)

    directory.mkdir(parents=True, exist_ok=True)
    retriever.save(directory, show_progress=False)
    with _manifest_path(store_dir).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_chunks(chunks: list[Chunk], store_dir: Path) -> None:
    """Replaces the BM25 rows for every doc_id present in `chunks` — the same
    delete-then-insert idempotency `load/vector_store.write_chunks` gives the dense index,
    so re-ingesting a file, or the same file after a chunk-size config change, leaves no
    orphaned rows here either."""
    if not chunks:
        return
    doc_ids = {chunk.doc_id for chunk in chunks}
    kept = [row for row in read_rows(store_dir) if row["doc_id"] not in doc_ids]
    kept.extend(_row(chunk) for chunk in chunks)
    _rebuild(store_dir, kept)


def delete_doc(store_dir: Path, doc_id: str) -> None:
    """Removes every row for `doc_id` from the BM25 index, e.g. for the `rm` command
    (TASKS.md SEARCH-2: "deleted alongside vectors in INDEX-3's path"). A no-op if the
    index does not exist yet."""
    rows = read_rows(store_dir)
    kept = [row for row in rows if row["doc_id"] != doc_id]
    if len(kept) == len(rows):
        return
    _rebuild(store_dir, kept)


def drop_index(store_dir: Path) -> None:
    """Drops the BM25 index entirely, e.g. before a full `reindex`. A no-op if it does not
    exist yet."""
    directory = bm25_dir(store_dir)
    if directory.exists():
        shutil.rmtree(directory)
