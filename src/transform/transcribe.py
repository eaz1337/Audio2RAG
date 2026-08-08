"""Thin wrapper calling whichever ASRBackend is configured (see ADR 0002, CLAUDE.md
"ASR is pluggable, not assumed"). Never imports a concrete backend at module level — the
backend is injected by the caller (CLI wiring, in INGEST-1)."""

from __future__ import annotations

from pathlib import Path

from models.schemas import Segment
from transform.asr.base import ASRBackend


class Transcriber:
    def __init__(self, backend: ASRBackend) -> None:
        self._backend = backend

    def transcribe(self, path: Path, language: str) -> list[Segment]:
        return self._backend.transcribe(path, language)
