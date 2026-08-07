# 0002 — Pivot to audio-to-PDF as initial scope

**Status:** Accepted (superseded in part by 0003 and 0004 — see below)
**Date:** 2026-08-04

## Context

The project's first working scope was defined as: take an audio file (lecture, training,
meeting), transcribe it with Whisper (`faster-whisper`), and produce a single PDF transcript.
Polish was the default spoken language. The pipeline followed a simple
`extract -> transform -> load` shape:

- **Extract:** validate the input audio file path and format
- **Transform:** run Whisper transcription
- **Load:** render the transcript into a PDF using a DejaVuSans font (needed for correct Polish
  glyph rendering — the default fonts in most PDF libraries don't cover Polish diacritics well)

This scope was deliberately narrow: a single input, a single output, no search, no database, no
UI. The goal was a small, testable, idempotent CLI tool rather than a platform.

## Decision

Ship audio-to-PDF as the first complete slice of the project, with:

- Idempotency as a hard requirement from day one (`re-running overwrites, never duplicates`)
- Platform-aware ASR device selection (CUDA on Windows, CPU fallback on Linux/macOS)
- A hard performance target (60-minute audio in under 5 minutes on Windows/CUDA, under 15 minutes
  on CPU) to keep the tool usable for real recordings, not just short clips

## Consequences

- This scope proved the ASR + rendering pipeline works end to end and is fast enough to be useful.
- It also surfaced the tool's real limitation: a PDF transcript is something a person has to read
  start to finish to find anything. There was no way to ask "where was X discussed" without
  opening the file and searching manually.
- That limitation motivated the pivot recorded in **0003** (stop treating PDF as the primary
  artifact; treat the transcript as structured, queryable data) and, downstream of that, the
  broader reframing in **0004** and `spec.md` toward a meeting-intelligence / RAG product where
  PDF is one of several optional renderers rather than the deliverable.
- The idempotency and cross-platform design decisions made here were not overturned — they carry
  forward as the foundation for every later phase.
