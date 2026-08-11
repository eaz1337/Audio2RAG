"""Writes chunks to the local LanceDB store (spec.md §5, TASKS.md INDEX-2). `lancedb.connect`
just opens a directory on disk — no server, no network call (CLAUDE.md "Retrieval is
local-first"). Collection metadata records the embedding model name and dim (CLAUDE.md "Model
pinning") so a later mismatch can be caught before querying a stale index (INDEX-4)."""

from __future__ import annotations

from pathlib import Path

import lancedb
import pyarrow as pa

from load.manage import list_transcripts
from load.write_canonical import read_segments
from models.schemas import Chunk
from transform.chunk import chunk_segments
from transform.embed import Embedder

_TABLE_NAME = "chunks"
_MODEL_KEY = b"embedding_model"
_DIM_KEY = b"embedding_dim"


def _table_exists(db: lancedb.DBConnection) -> bool:
    return _TABLE_NAME in db.list_tables().tables


def lance_dir(store_dir: Path) -> Path:
    return store_dir / "lance"


def _schema(embedder: Embedder) -> pa.Schema:
    return pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), embedder.dim)),
            pa.field("doc_id", pa.string()),
            pa.field("chunk_id", pa.int64()),
            pa.field("start", pa.float64()),
            pa.field("end", pa.float64()),
            pa.field("segment_ids", pa.list_(pa.int64())),
            pa.field("display_text", pa.string()),
            pa.field("embed_text", pa.string()),
        ],
        metadata={_MODEL_KEY: embedder.name.encode(), _DIM_KEY: str(embedder.dim).encode()},
    )


def _rows(chunks: list[Chunk], vectors: pa.Array) -> list[dict[str, object]]:
    return [
        {
            "vector": vector.tolist(),
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "start": chunk.start,
            "end": chunk.end,
            "segment_ids": chunk.segment_ids,
            "display_text": chunk.display_text,
            "embed_text": chunk.embed_text,
        }
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]


def _doc_ids_filter(doc_ids: set[str]) -> str:
    quoted = ", ".join(f"'{doc_id}'" for doc_id in sorted(doc_ids))
    return f"doc_id IN ({quoted})"


def write_chunks(chunks: list[Chunk], embedder: Embedder, store_dir: Path) -> None:
    """Embeds `chunk.embed_text` for every chunk and replaces the rows for every doc_id
    present in `chunks` (CLAUDE.md "Idempotency": DELETE WHERE doc_id then INSERT, never
    per-chunk upsert), so re-ingesting the same file — or the same file after a chunk-size
    config change — leaves no orphaned rows."""
    if not chunks:
        return

    vectors = embedder.embed([chunk.embed_text for chunk in chunks])
    schema = _schema(embedder)
    data = pa.Table.from_pylist(_rows(chunks, vectors), schema=schema)

    db = lancedb.connect(lance_dir(store_dir))
    if _table_exists(db):
        table = db.open_table(_TABLE_NAME)
        table.delete(_doc_ids_filter({chunk.doc_id for chunk in chunks}))
        table.add(data)
    else:
        db.create_table(_TABLE_NAME, data=data)


def delete_doc(store_dir: Path, doc_id: str) -> None:
    """Removes every row for `doc_id`, e.g. for the `rm` command. A no-op if the table
    does not exist yet."""
    db = lancedb.connect(lance_dir(store_dir))
    if not _table_exists(db):
        return
    db.open_table(_TABLE_NAME).delete(f"doc_id = '{doc_id}'")


def read_chunks(store_dir: Path, doc_id: str | None = None) -> list[dict[str, object]]:
    """Every row in the `chunks` table, optionally restricted to one `doc_id`."""
    db = lancedb.connect(lance_dir(store_dir))
    if not _table_exists(db):
        return []
    table = db.open_table(_TABLE_NAME)
    query = table.search()
    if doc_id is not None:
        query = query.where(f"doc_id = '{doc_id}'")
    return query.to_list()


def collection_metadata(store_dir: Path) -> tuple[str, int] | None:
    """`(embedding_model_name, dim)` recorded on the `chunks` table, or `None` if the
    table does not exist yet."""
    db = lancedb.connect(lance_dir(store_dir))
    if not _table_exists(db):
        return None
    metadata = db.open_table(_TABLE_NAME).schema.metadata or {}
    return metadata[_MODEL_KEY].decode(), int(metadata[_DIM_KEY].decode())


class EmbeddingModelMismatch(RuntimeError):
    """Raised when a query is about to run against a store built with a different
    embedder (CLAUDE.md "Model pinning") — never silently query a stale index."""

    def __init__(self, stored_name: str, stored_dim: int, current_name: str, current_dim: int) -> None:
        super().__init__(
            f"store was built with embedder {stored_name!r} (dim={stored_dim}), but the "
            f"configured embedder is {current_name!r} (dim={current_dim}). Run "
            "`audio2rag reindex` to rebuild the store with the current embedder."
        )


def assert_embedder_matches(store_dir: Path, embedder: Embedder) -> None:
    """Guards a query against running on a stale index: raises `EmbeddingModelMismatch`
    if the store's recorded embedding model or dim differs from `embedder`. A no-op if
    the store does not exist yet."""
    stored = collection_metadata(store_dir)
    if stored is None:
        return
    stored_name, stored_dim = stored
    if stored_name != embedder.name or stored_dim != embedder.dim:
        raise EmbeddingModelMismatch(stored_name, stored_dim, embedder.name, embedder.dim)


def drop_table(store_dir: Path) -> None:
    """Drops the `chunks` table entirely. LanceDB tables have a fixed vector width, so a
    `reindex` with a different embedding dimension needs a fresh table rather than the
    per-doc-id delete-then-insert `write_chunks` does. A no-op if the table does not exist
    yet."""
    db = lancedb.connect(lance_dir(store_dir))
    if _table_exists(db):
        db.drop_table(_TABLE_NAME)


def reindex_all(
    output_dir: Path,
    store_dir: Path,
    embedder: Embedder,
    target_tokens: int = 500,
    overlap_ratio: float = 0.15,
    pause_threshold_s: float = 1.5,
) -> list[str]:
    """Rebuilds the vector store from scratch, from every `<doc_id>.segments.jsonl` under
    `output_dir` — never from an `ASRBackend`, so switching the embedding model or the
    chunking config never re-transcribes. Returns the doc_ids rebuilt."""
    drop_table(store_dir)
    doc_ids: list[str] = []
    for meta in list_transcripts(output_dir):
        segments = read_segments(meta.doc_id, output_dir)
        chunks = chunk_segments(
            segments,
            meta.title,
            meta.date,
            target_tokens=target_tokens,
            overlap_ratio=overlap_ratio,
            pause_threshold_s=pause_threshold_s,
        )
        write_chunks(chunks, embedder, store_dir)
        doc_ids.append(meta.doc_id)
    return doc_ids
