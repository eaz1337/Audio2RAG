# 0001 — JSONL as the canonical transcript source, PDF as a renderer

**Status:** Accepted
**Date:** 2026-08-08

## Context

Audio2RAG's goal (see `spec.md`) is a queryable knowledge base over recordings, not a
transcription tool that stops at a document. Whatever format the transcript is written in first
becomes the thing every later stage — rendering, chunking, embedding, retrieval — has to build on
top of. That format needs to survive being read back into typed data losslessly, carry
per-segment timestamps and speaker boundaries, and hold metadata (course, date, tags) without
contortion.

## Alternatives considered

- **Index PDFs directly.** Rejected: PDF is a presentation format. Extracting text back out of it
  (e.g. with `pdfplumber`) is a lossy round trip — line wrapping and hyphenation introduced by
  rendering pollute the extracted text, which would in turn pollute embeddings computed on it.
  PDF also has no natural place for per-segment timestamps or speaker labels once rendered, so
  "this was said at 14:22" becomes unrecoverable.
- **Index plain `.txt`.** Rejected: flattening segments into plain text discards segment
  boundaries, timestamps, speaker turns, and confidence scores in one step. A chunker operating on
  flattened text has no signal for where one thought ends and another begins, and answers could
  never cite a timestamp.
- **JSONL as canonical source (chosen).** One JSON object per segment
  (`start`, `end`, `text`, `speaker`, `conf`) preserves everything the above two options discard,
  is trivially appended/streamed, and round-trips losslessly into the `Segment` Pydantic model.

## Decision

The transcript is written once, in canonical form, before any rendering happens:

```
output/<doc_id>.segments.jsonl   — one JSON object per segment: start, end, text, speaker, conf
output/<doc_id>.meta.json        — recording-level metadata: title, course, date, speakers, tags, type
```

PDF, Markdown, and SRT are *renderers* that read this canonical form. They are never the source of
anything downstream, and nothing in the codebase is permitted to parse a PDF back into structured
data — if a later stage needs transcript content, it reads the JSONL. Rendering is opt-in
(`--render pdf,md,srt`), never a default side effect of ingest.

Two runs on the same input file overwrite the canonical artifacts, not append to them
(idempotency, keyed on `doc_id = sha256(audio bytes)[:16]`).

## Consequences

- Every renderer (PDF, MD, SRT) becomes a small, independently testable function over
  `list[Segment]`, rather than the endpoint of the pipeline. Adding a new output format never
  touches the ASR or chunking code.
- Chunking, embedding, and retrieval get timestamps, speaker labels, and metadata for free,
  because they were never thrown away.
- This is what makes tiered delivery (`TASKS.md`, `spec.md` Tiers) work as a clean upsell: a
  client who buys only transcription (Starter) already has the canonical JSONL sitting on disk.
  Adding search (Standard) or grounded answering (Advanced) later means building *on* that file,
  not regenerating or reprocessing anything already delivered.
- PDF generation moves from "always produced" to "rendered on request," since it is no longer
  load-bearing for anything else in the pipeline.
