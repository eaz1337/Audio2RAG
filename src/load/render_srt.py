"""Renders an SRT subtitle file over the canonical transcript (see CLAUDE.md "Canonical source"
and ADR 0001). This is the only module allowed to produce this SRT rendering, and it only ever
reads `list[Segment]` — never a raw string. Generated on request only, never as a side effect
of ingest; nothing anywhere parses SRT back into data."""

from __future__ import annotations

from pathlib import Path

from load.write_canonical import read_segments
from models.schemas import Segment


def _format_timestamp(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    hours, remainder_ms = divmod(total_ms, 3_600_000)
    minutes, remainder_ms = divmod(remainder_ms, 60_000)
    secs, millis = divmod(remainder_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def render_srt(segments: list[Segment], output_path: Path) -> None:
    """Writes an SRT file at `output_path` from already-loaded segments."""
    blocks = []
    for index, segment in enumerate(segments, start=1):
        speaker = f"{segment.speaker}: " if segment.speaker else ""
        start = _format_timestamp(segment.start)
        end = _format_timestamp(segment.end)
        blocks.append(f"{index}\n{start} --> {end}\n{speaker}{segment.text}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(blocks) + "\n")


def render_srt_from_jsonl(doc_id: str, output_dir: Path) -> Path:
    """Reads `<doc_id>.segments.jsonl` from `output_dir` and writes `<doc_id>.srt`
    alongside it, returning the written path."""
    segments = read_segments(doc_id, output_dir)
    output_path = output_dir / f"{doc_id}.srt"
    render_srt(segments, output_path)
    return output_path
