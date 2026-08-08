"""Renders a PDF over the canonical transcript (see CLAUDE.md "Canonical source" and ADR 0001).
This is the only module allowed to produce a PDF, and it only ever reads `list[Segment]` — never
a raw string. PDF is generated on request, never as a side effect of ingest; nothing here (or
anywhere else) parses a PDF back into data."""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

from load.write_canonical import read_meta, read_segments
from models.schemas import Segment, TranscriptMeta

# Vendored for correct Polish glyphs (spec.md §10): core PDF fonts lack ą/ć/ę/ł/ń/ó/ś/ź/ż.
_FONT_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"
_FONT_REGULAR = _FONT_DIR / "DejaVuSans.ttf"
_FONT_BOLD = _FONT_DIR / "DejaVuSans-Bold.ttf"


def _format_timestamp(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def render_pdf(segments: list[Segment], meta: TranscriptMeta, output_path: Path) -> None:
    """Writes a PDF at `output_path` from already-loaded segments and metadata."""
    pdf = FPDF()
    pdf.add_font("DejaVuSans", "", str(_FONT_REGULAR))
    pdf.add_font("DejaVuSans", "B", str(_FONT_BOLD))
    pdf.add_page()

    pdf.set_font("DejaVuSans", "B", 16)
    pdf.multi_cell(0, 10, meta.title)
    pdf.ln(4)

    pdf.set_font("DejaVuSans", "", 11)
    for segment in segments:
        timestamp = f"[{_format_timestamp(segment.start)}-{_format_timestamp(segment.end)}]"
        speaker = f" {segment.speaker}:" if segment.speaker else ""
        pdf.multi_cell(0, 8, f"{timestamp}{speaker} {segment.text}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))


def render_pdf_from_jsonl(doc_id: str, output_dir: Path) -> Path:
    """Reads `<doc_id>.segments.jsonl`/`.meta.json` from `output_dir` and writes
    `<doc_id>.pdf` alongside them, returning the written path."""
    segments = read_segments(doc_id, output_dir)
    meta = read_meta(doc_id, output_dir)
    output_path = output_dir / f"{doc_id}.pdf"
    render_pdf(segments, meta, output_path)
    return output_path
