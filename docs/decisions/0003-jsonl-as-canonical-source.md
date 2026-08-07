# 0003 — JSONL as the canonical transcript source, not PDF

**Status:** Accepted
**Date:** 2026-08-04
**Supersedes:** part of 0002 (PDF as the primary deliverable)

## Context

The original scope (0002) treated PDF as the pipeline's output. Extending the project toward
question-answering over recordings (the actual goal — see `spec.md`) exposed that PDF is a poor
foundation to build retrieval on top of:

- PDF is a presentation format, not a data format. Any later stage that needs the transcript's
  content would have to parse the PDF back out (e.g. with `pdfplumber`) — a lossy round trip for
  data that was structured a moment before it was rendered.
- Rendering to PDF discards per-segment timestamps. Without them, an answer can never point back
  to "this was said at 14:22" — it can only point to "this was said somewhere in this document."
- It discards segment and speaker boundaries. A chunker operating on flattened PDF text has no
  signal for where one thought ends and another begins, and no idea who was speaking.
- It has no natural place for metadata (course, date, presenter, tags) that later filtering
  depends on.
- Line wrapping and hyphenation introduced by PDF rendering pollute any text extracted back out of
  it, which would in turn pollute embeddings computed on that text.

## Decision

The transcript is written once, in canonical form, before any rendering happens:

```
output/<doc_id>.segments.jsonl   — one JSON object per segment: start, end, text, speaker, conf
output/<doc_id>.meta.json        — recording-level metadata: title, course, date, speakers, tags, type
```

PDF, Markdown, and SRT are *renderers* that read this canonical form. They are never the source of
anything downstream, and nothing in the codebase is permitted to parse a PDF back into structured
data. If a later stage needs transcript content, it reads the JSONL.

Two runs on the same input file must overwrite the canonical artifacts, not append to them
(idempotency, extending the requirement from 0002 to the new artifact).

## Consequences

- Every renderer (PDF, MD, SRT) becomes a small, independently testable function over
  `list[Segment]`, rather than the endpoint of the pipeline. Adding a new output format never
  touches the ASR or chunking code.
- Chunking, embedding, and retrieval — the actual point of the project per `spec.md` — get
  timestamps, speaker labels, and metadata for free, because they were never thrown away.
- This is what makes the project's tiered delivery (`TASKS.md`, `spec.md` Tiers) work as a clean
  upsell: a client who buys only transcription (Starter) already has the canonical JSONL sitting
  on disk. Adding search (Standard) or grounded answering (Advanced) later means building *on*
  that file, not regenerating or reprocessing anything already delivered.
- PDF generation moves from "always produced" to "rendered on request" (`--render pdf`), since it
  is no longer load-bearing for anything else in the pipeline.
