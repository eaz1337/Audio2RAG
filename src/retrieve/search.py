"""Dense and BM25 sparse search over the local stores (spec.md §6, TASKS.md SEARCH-1,
SEARCH-2). `dense_search` alone is the Standard-tier deliverable: smart search, no LLM
call. `sparse_search` is the Advanced-tier lexical counterpart — exact-term matches (e.g.
rare technical terms) that a dense embedder can miss — fused with it in a later task
(CLAUDE.md: `retrieve/` is a read path and must never write to the store)."""

from __future__ import annotations

from pathlib import Path

import bm25s

from load import bm25_index
from load.vector_store import assert_embedder_matches, open_table
from models.schemas import RetrievalHit
from transform.embed import Embedder


def dense_search(query: str, k: int, embedder: Embedder, store_dir: Path) -> list[RetrievalHit]:
    """Embeds `query` and returns the `k` nearest chunks by cosine similarity, best match
    first. Raises `EmbeddingModelMismatch` if the store was built with a different embedder
    (CLAUDE.md "Model pinning"). Empty list if the store has no chunks yet."""
    assert_embedder_matches(store_dir, embedder)
    table = open_table(store_dir)
    if table is None:
        return []

    [vector] = embedder.embed([query])
    rows = table.search(vector.tolist()).metric("cosine").limit(k).to_list()

    return [
        RetrievalHit(
            doc_id=row["doc_id"],
            chunk_id=row["chunk_id"],
            score=1.0 - row["_distance"],
            start=row["start"],
            end=row["end"],
            segment_ids=row["segment_ids"],
            display_text=row["display_text"],
        )
        for row in rows
    ]


def sparse_search(query: str, k: int, store_dir: Path) -> list[RetrievalHit]:
    """BM25 lexical search over the same chunks the dense index holds. Empty list if the
    BM25 index has no chunks yet (SEARCH-2). No embedder needed — no network call, no
    model download, purely local."""
    rows = bm25_index.read_rows(store_dir)
    if not rows:
        return []

    directory = bm25_index.bm25_dir(store_dir)
    retriever = bm25s.BM25.load(directory, load_corpus=False, show_progress=False)
    query_tokens = bm25s.tokenize(query, stopwords=None, show_progress=False)
    [hit_rows], [scores] = retriever.retrieve(
        query_tokens, corpus=rows, k=min(k, len(rows)), show_progress=False
    )

    return [
        RetrievalHit(
            doc_id=row["doc_id"],
            chunk_id=row["chunk_id"],
            score=float(score),
            start=row["start"],
            end=row["end"],
            segment_ids=row["segment_ids"],
            display_text=row["display_text"],
        )
        for row, score in zip(hit_rows, scores, strict=True)
    ]
