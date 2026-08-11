"""Groups ASR segments into retrieval-sized chunks (spec.md §3, CLAUDE.md "Pipeline layout").

Pure function over data — no embedding model, no tokenizer model. Token counts are
approximated by whitespace-delimited word count, which is enough to hit the 400-600
target band; exact tokenization happens downstream, at embed time.
"""

from __future__ import annotations

import re
from datetime import date as date_

from models.schemas import Chunk, Segment

_WORD_RE = re.compile(r"\S+")
_SENTENCE_END_RE = re.compile(r"[.!?…]+[\"'’”)\]]*$")

_MAX_OVERLAP_SEGMENTS = 3


def _count_tokens(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _is_preferred_boundary(current: Segment, following: Segment, pause_threshold_s: float) -> bool:
    """True if the gap after `current` is a good place to end a chunk (spec.md §3)."""
    if following.start - current.end > pause_threshold_s:
        return True
    if current.speaker != following.speaker:
        return True
    return _SENTENCE_END_RE.search(current.text.strip()) is not None


def _overlap_count(buffer: list[Segment], overlap_ratio: float) -> int:
    """How many trailing segments of `buffer` should repeat at the start of the next chunk."""
    if len(buffer) <= 1:
        return 0
    total_tokens = sum(_count_tokens(segment.text) for segment in buffer)
    target_overlap_tokens = total_tokens * overlap_ratio
    accumulated = 0
    count = 0
    for segment in reversed(buffer):
        accumulated += _count_tokens(segment.text)
        count += 1
        if accumulated >= target_overlap_tokens:
            break
    return min(count, len(buffer) - 1, _MAX_OVERLAP_SEGMENTS)


def _format_timestamp(seconds: float) -> str:
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _resolve_speaker(buffer: list[Segment]) -> str | None:
    """Speakers present in `buffer`, in order of first appearance, joined for the header."""
    speakers = list(dict.fromkeys(segment.speaker for segment in buffer if segment.speaker))
    return "/".join(speakers) if speakers else None


def _format_header(title: str, date: date_ | None, speaker: str | None, start: float, end: float) -> str:
    """Contextual header prepended to `embed_text` (spec.md §3, canonical form from §7)."""
    parts = [title]
    if date is not None:
        parts.append(date.isoformat())
    if speaker is not None:
        parts.append(speaker)
    parts.append(f"{_format_timestamp(start)}–{_format_timestamp(end)}")
    return "[" + ", ".join(parts) + "]"


def _build_chunk(
    doc_id: str, chunk_id: int, buffer: list[Segment], title: str, date: date_ | None
) -> Chunk:
    text = " ".join(segment.text for segment in buffer)
    start, end = buffer[0].start, buffer[-1].end
    header = _format_header(title, date, _resolve_speaker(buffer), start, end)
    return Chunk(
        doc_id=doc_id,
        chunk_id=chunk_id,
        start=start,
        end=end,
        segment_ids=[segment.seg for segment in buffer],
        display_text=text,
        embed_text=f"{header}\n{text}",
    )


def chunk_segments(
    segments: list[Segment],
    title: str,
    date: date_ | None = None,
    target_tokens: int = 500,
    overlap_ratio: float = 0.15,
    pause_threshold_s: float = 1.5,
) -> list[Chunk]:
    """Groups consecutive segments into ~`target_tokens`-sized chunks with overlap.

    Splits preferentially at pauses > `pause_threshold_s`, sentence ends, or speaker
    changes once a chunk's token count is within the target band (target_tokens ±20%).
    A chunk never splits a segment, so a single segment longer than the upper bound
    becomes a chunk on its own.

    `title` and `date` come from the document's `TranscriptMeta` and are prepended,
    together with the chunk's speaker(s) and timestamp range, as a contextual header
    on `embed_text` (spec.md §3) — `display_text` stays the clean body.
    """
    if not segments:
        raise ValueError("segments must not be empty")

    doc_id = segments[0].doc_id
    if any(segment.doc_id != doc_id for segment in segments):
        raise ValueError("all segments must share the same doc_id")

    lower = round(target_tokens * 0.8)
    upper = round(target_tokens * 1.2)

    chunks: list[Chunk] = []
    start = 0
    n = len(segments)
    while start < n:
        idx = start
        total = _count_tokens(segments[idx].text)
        while idx < n - 1:
            if total >= upper:
                break
            if total >= lower and _is_preferred_boundary(
                segments[idx], segments[idx + 1], pause_threshold_s
            ):
                break
            next_tokens = _count_tokens(segments[idx + 1].text)
            if total >= lower and total + next_tokens > upper:
                break
            total += next_tokens
            idx += 1

        buffer = segments[start : idx + 1]
        chunks.append(_build_chunk(doc_id, len(chunks), buffer, title, date))

        if idx == n - 1:
            break
        overlap = _overlap_count(buffer, overlap_ratio)
        start = max(idx + 1 - overlap, start + 1)

    return chunks
