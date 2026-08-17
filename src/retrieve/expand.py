"""Context expansion for the LLM answer path (spec.md §6 "Context expansion", TASKS.md
SEARCH-6): an answer often starts one sentence before the retrieved chunk, so each hit's
text is padded with the segment immediately before its first segment and immediately
after its last, read straight from the canonical `<doc_id>.segments.jsonl` (CLAUDE.md
"Canonical source") — never from the vector store. Expansion clamps at the document's
first/last segment and never pulls in a segment already owned by another hit in the same
batch, so two adjacent hits never repeat text between them. Citation fields
(doc_id/chunk_id/start/end) are untouched — only the text handed to the LLM changes.

Out of scope: answering (ANSWER-1 consumes this)."""

from __future__ import annotations

from pathlib import Path

from load.write_canonical import read_segments
from models.schemas import RetrievalHit


def expand_hits(hits: list[RetrievalHit], output_dir: Path) -> list[str]:
    """Returns expanded text for each hit in `hits`, in the same order. For a hit whose
    segment ids span `lo..hi`, prepends segment `lo - 1` and appends segment `hi + 1`
    when that segment exists in the document and isn't already claimed by another hit's
    own segment range."""
    if not hits:
        return []

    texts_by_doc = {
        doc_id: {segment.seg: segment.text for segment in read_segments(doc_id, output_dir)}
        for doc_id in {hit.doc_id for hit in hits}
    }

    ranges_by_doc: dict[str, list[tuple[int, int]]] = {}
    for hit in hits:
        ranges_by_doc.setdefault(hit.doc_id, []).append((min(hit.segment_ids), max(hit.segment_ids)))

    expanded = []
    for hit in hits:
        texts = texts_by_doc[hit.doc_id]
        ranges = ranges_by_doc[hit.doc_id]
        lo, hi = min(hit.segment_ids), max(hit.segment_ids)

        parts = [hit.display_text]

        prev_id = lo - 1
        if prev_id in texts and not _claimed(prev_id, ranges):
            parts.insert(0, texts[prev_id])

        next_id = hi + 1
        if next_id in texts and not _claimed(next_id, ranges):
            parts.append(texts[next_id])

        expanded.append(" ".join(parts))
    return expanded


def _claimed(segment_id: int, ranges: list[tuple[int, int]]) -> bool:
    return any(lo <= segment_id <= hi for lo, hi in ranges)
