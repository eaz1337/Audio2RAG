"""Fake implementations of protocol boundaries, injected in tests in place of real backends
(see CLAUDE.md "Testing")."""

from __future__ import annotations

from pathlib import Path

from models.schemas import Segment


class FakeASRBackend:
    """Returns a fixed, injectable segment list instead of calling a real ASR API/model."""

    def __init__(self, segments: list[Segment]) -> None:
        self._segments = segments
        self.calls: list[tuple[Path, str]] = []

    def transcribe(self, path: Path, language: str) -> list[Segment]:
        self.calls.append((path, language))
        return self._segments
