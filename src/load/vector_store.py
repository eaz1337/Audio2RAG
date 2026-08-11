"""Writes chunks to the local LanceDB store (spec.md §5, TASKS.md INDEX-2). `lancedb.connect`
just opens a directory on disk — no server, no network call (CLAUDE.md "Retrieval is
local-first"). Collection metadata records the embedding model name and dim (CLAUDE.md "Model
pinning") so a later mismatch can be caught before querying a stale index (INDEX-4)."""

from __future__ import annotations

from pathlib import Path

import lancedb
import pyarrow as pa

from models.schemas import Chunk
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


def write_chunks(chunks: list[Chunk], embedder: Embedder, store_dir: Path) -> None:
    """Embeds `chunk.embed_text` for every chunk and appends the rows to the `chunks`
    table, creating it (with collection metadata) on first write. Re-running this for the
    same doc_id duplicates rows — delete-then-insert idempotency is INDEX-3's job."""
    if not chunks:
        return

    vectors = embedder.embed([chunk.embed_text for chunk in chunks])
    schema = _schema(embedder)
    data = pa.Table.from_pylist(_rows(chunks, vectors), schema=schema)

    db = lancedb.connect(lance_dir(store_dir))
    if _table_exists(db):
        db.open_table(_TABLE_NAME).add(data)
    else:
        db.create_table(_TABLE_NAME, data=data)


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
