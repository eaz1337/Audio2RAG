"""Reciprocal Rank Fusion of dense and sparse hit lists (spec.md §6, TASKS.md SEARCH-3).
Pure function over two already-fetched `RetrievalHit` lists — no store/index access, so it
holds CLAUDE.md's "retrieve/ never writes, never calls out" rule trivially."""

from __future__ import annotations

from models.schemas import RetrievalHit


def rrf(
    dense_hits: list[RetrievalHit], sparse_hits: list[RetrievalHit], k: int = 60
) -> list[RetrievalHit]:
    """Combines `dense_hits` and `sparse_hits`, ranked highest first, using Reciprocal Rank
    Fusion: a hit's fused score is the sum of `1 / (k + rank)` over every list it appears in
    (rank counted from 1), which lets two rankings with incomparable score scales (cosine
    similarity vs. BM25) be merged without normalization. A hit present in only one list is
    still returned, scored by that list alone. Identity is `(doc_id, chunk_id)`; the returned
    hit's non-score fields come from its first occurrence, `dense_hits` before `sparse_hits`."""
    scores: dict[tuple[str, int], float] = {}
    first_seen: dict[tuple[str, int], RetrievalHit] = {}

    for hits in (dense_hits, sparse_hits):
        for rank, hit in enumerate(hits, start=1):
            key = (hit.doc_id, hit.chunk_id)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(key, hit)

    fused = [
        first_seen[key].model_copy(update={"score": score}) for key, score in scores.items()
    ]
    fused.sort(key=lambda hit: hit.score, reverse=True)
    return fused
