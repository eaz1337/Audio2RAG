"""Renders Markdown over the canonical transcript (see CLAUDE.md "Canonical source" and ADR
0001). This is the only module allowed to produce this Markdown rendering, and it only ever
reads `list[Segment]` — never a raw string. Generated on request only, never as a side effect
of ingest; nothing anywhere parses Markdown back into data."""

from __future__ import annotations

from pathlib import Path

from load.write_canonical import read_meta, read_segments
from models.schemas import Segment, TranscriptMeta


def _format_timestamp(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def render_md(segments: list[Segment], meta: TranscriptMeta, output_path: Path) -> None:
    """Writes a Markdown file at `output_path` from already-loaded segments and metadata."""
    lines = [f"# {meta.title}", ""]
    for segment in segments:
        timestamp = f"[{_format_timestamp(segment.start)}-{_format_timestamp(segment.end)}]"
        speaker = f" **{segment.speaker}:**" if segment.speaker else ""
        lines.append(f"{timestamp}{speaker} {segment.text}")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def render_md_from_jsonl(doc_id: str, output_dir: Path) -> Path:
    """Reads `<doc_id>.segments.jsonl`/`.meta.json` from `output_dir` and writes
    `<doc_id>.md` alongside them, returning the written path."""
    segments = read_segments(doc_id, output_dir)
    meta = read_meta(doc_id, output_dir)
    output_path = output_dir / f"{doc_id}.md"
    render_md(segments, meta, output_path)
    return output_path
