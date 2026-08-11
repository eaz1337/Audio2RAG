"""Embedder protocol and the bge-m3 default implementation (spec.md §4). Dense output only —
bge-m3's learned-sparse output is deliberately ignored: BM25 (SEARCH-2) covers the lexical path
instead, so it stays identical across an embedding-model swap. Runs locally regardless of the
configured ASR backend (ADR 0002, CLAUDE.md "Retrieval is local-first")."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> NDArray[np.float32]: ...


class BgeM3Embedder:
    """Default embedder (spec.md §4) — multilingual, 8192 ctx, solid Polish. Behind the `[rag]`
    extra; `sentence-transformers` is imported lazily so installs without that extra never pay
    for it."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self.name = model_name
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(embeddings, dtype=np.float32)
