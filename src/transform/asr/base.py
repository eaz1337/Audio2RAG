"""ASRBackend protocol — the enforced boundary between backends and everything downstream
of `list[Segment]` (see ADR 0002 and CLAUDE.md "ASR is pluggable, not assumed")."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from models.schemas import Segment


@runtime_checkable
class ASRBackend(Protocol):
    def transcribe(self, path: Path, language: str) -> list[Segment]: ...
