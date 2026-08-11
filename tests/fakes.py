"""Fake implementations of protocol boundaries, injected in tests in place of real backends
(see CLAUDE.md "Testing")."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from models.schemas import Segment


class FakeASRBackend:
    """Returns a fixed, injectable segment list instead of calling a real ASR API/model."""

    def __init__(self, segments: list[Segment]) -> None:
        self._segments = segments
        self.calls: list[tuple[Path, str]] = []

    def transcribe(self, path: Path, language: str) -> list[Segment]:
        self.calls.append((path, language))
        return self._segments


class FakeEmbedder:
    """Deterministic vectors derived from a text hash — no model, no download (CLAUDE.md
    "Testing": non-slow tests use a fake embedder instead of the real one)."""

    name = "fake-embedder"

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        vectors = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
            vectors[i] = np.random.default_rng(seed).standard_normal(self.dim)
        return vectors
