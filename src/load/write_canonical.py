"""Writes the canonical transcript artifacts (see CLAUDE.md "Canonical source" and ADR 0001):
`output/<doc_id>.segments.jsonl` and `output/<doc_id>.meta.json`. Everything else — PDF, MD,
SRT — is a renderer that reads these, never the other way round."""

from __future__ import annotations

from pathlib import Path

from models.schemas import Segment, TranscriptMeta


def segments_path(doc_id: str, output_dir: Path) -> Path:
    return output_dir / f"{doc_id}.segments.jsonl"


def meta_path(doc_id: str, output_dir: Path) -> Path:
    return output_dir / f"{doc_id}.meta.json"


def write_canonical(segments: list[Segment], meta: TranscriptMeta, output_dir: Path) -> None:
    """Writes both artifacts for `meta.doc_id`. `Path.write_text` truncates on each call, so
    re-running overwrites the previous artifacts rather than appending to them."""
    output_dir.mkdir(parents=True, exist_ok=True)

    body = "".join(segment.model_dump_json() + "\n" for segment in segments)
    segments_path(meta.doc_id, output_dir).write_text(body)

    meta_path(meta.doc_id, output_dir).write_text(meta.model_dump_json(indent=2))


def read_segments(doc_id: str, output_dir: Path) -> list[Segment]:
    lines = segments_path(doc_id, output_dir).read_text().splitlines()
    return [Segment.model_validate_json(line) for line in lines if line]


def read_meta(doc_id: str, output_dir: Path) -> TranscriptMeta:
    return TranscriptMeta.model_validate_json(meta_path(doc_id, output_dir).read_text())
