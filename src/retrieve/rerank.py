"""Reranker protocol and the bge-reranker-v2-m3 default implementation (spec.md §6,
TASKS.md SEARCH-4) — the cross-encoder pass after RRF fusion that a bi-encoder's dense
score and BM25's lexical score can't capture on their own. Reranking is skippable
(`--no-rerank`, wired in ANSWER-1): pass `reranker=None` to `rerank()` and the fused order
is returned unchanged."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from models.schemas import RetrievalHit


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]: ...


def rerank(query: str, hits: list[RetrievalHit], reranker: Reranker | None) -> list[RetrievalHit]:
    """Reorders `hits` by relevance to `query` using `reranker`. `reranker=None` is the
    `--no-rerank` path: `hits` is returned as-is, in fused order."""
    if reranker is None:
        return hits
    return reranker.rerank(query, hits)


class BgeReranker:
    """Default reranker (spec.md §6) — BAAI/bge-reranker-v2-m3, cross-encoder, handles
    Polish well. Behind the `[rag]` extra; `sentence_transformers.CrossEncoder` is imported
    lazily so installs without that extra never pay for it."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, device=device)
        self.name = model_name

    def rerank(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        if not hits:
            return []
        pairs = [(query, hit.display_text) for hit in hits]
        scores = self._model.predict(pairs)
        scored = [
            hit.model_copy(update={"score": float(score)})
            for hit, score in zip(hits, scores, strict=True)
        ]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored
