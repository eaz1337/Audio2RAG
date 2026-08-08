"""Thin wrapper calling whichever ASRBackend is configured (see ADR 0002, CLAUDE.md
"ASR is pluggable, not assumed"). Never imports a concrete backend at module level — the
backend is injected by the caller (CLI wiring, in INGEST-1)."""

from __future__ import annotations

from pathlib import Path

from extract.hashing import compute_sha256
from load.write_canonical import meta_path, read_meta, read_segments, segments_path
from models.schemas import Segment
from transform.asr.base import ASRBackend


class Transcriber:
    def __init__(self, backend: ASRBackend) -> None:
        self._backend = backend

    def transcribe(self, path: Path, language: str) -> list[Segment]:
        return self._backend.transcribe(path, language)

    def transcribe_cached(
        self, path: Path, language: str, output_dir: Path, *, force: bool = False
    ) -> list[Segment]:
        """Skips the ASR backend entirely when a cached transcript for this exact audio
        content already exists (see CLAUDE.md "Transcription cache") — this matters for
        hosted backends, where a skipped call also saves money, not just time.
        `--force` (the `force` flag) always re-transcribes and overwrites the cache."""
        if not force and self._is_cached(path, output_dir):
            doc_id = compute_sha256(path)[:16]
            return read_segments(doc_id, output_dir)
        return self._backend.transcribe(path, language)

    @staticmethod
    def _is_cached(path: Path, output_dir: Path) -> bool:
        sha256 = compute_sha256(path)
        doc_id = sha256[:16]
        if not segments_path(doc_id, output_dir).exists():
            return False
        if not meta_path(doc_id, output_dir).exists():
            return False
        return read_meta(doc_id, output_dir).sha256 == sha256
