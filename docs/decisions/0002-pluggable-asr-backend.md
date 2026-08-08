# 0002 — Pluggable ASR backend; offline is not a hard requirement

**Status:** Accepted
**Date:** 2026-08-08

## Context

Audio2RAG's value (see `spec.md`'s Goal section) lives in chunking, retrieval, and grounded
answering — not in transcription itself. Speech-to-text with word-level timestamps and speaker
diarization is a commodity problem that `faster-whisper`, `WhisperX`, and hosted APIs (AssemblyAI,
Deepgram) all solve well. Before committing to an ASR approach, it's worth being explicit that
offline / on-premise processing is **not** a hard constraint for this project — recordings may be
sent to a third-party API — since that assumption would otherwise silently steer the whole
pipeline toward self-hosting.

Given that offline is not required, self-hosting Whisper is not the obvious default:

- Hosted APIs (AssemblyAI, Deepgram nova-3) return word-level timestamps *and* diarization in a
  single call, with no GPU/model management and no Hugging Face token plumbing.
- Self-hosting only earns back its complexity at high enough volume, or when a specific engagement
  has a genuine on-premise requirement — a case that can still arise per-client and shouldn't
  require redesigning the pipeline when it does.

## Alternatives considered

- **Self-hosted `faster-whisper` only.** Rejected as the sole path: it forces GPU/CPU device
  selection, model download/management, and (for diarization) a `pyannote` dependency plus an HF
  token onto every deployment, even ones with no on-premise requirement at all.
- **Hosted API only, no fallback.** Rejected: some engagements have a real on-premise or
  compliance requirement, and per-request cost matters at high volume. The pipeline needs an escape
  hatch that doesn't require an architecture change to use.
- **Pluggable backend behind a shared protocol (chosen).** One interface, two implementations
  (hosted default, self-hosted fallback), selected by config — see Decision below.

## Decision

ASR sits behind a small interface, defined in `src/transform/asr/base.py`:

```python
class ASRBackend(Protocol):
    def transcribe(self, path: Path, language: str) -> list[Segment]: ...
```

- **Default backend:** a hosted API (AssemblyAI or Deepgram nova-3), selected via
  `asr.backend: assemblyai | deepgram` in `config.yaml`. API keys come from the environment
  (`ASSEMBLYAI_API_KEY` / `DEEPGRAM_API_KEY`) and are never logged or written into `meta.json`.
- **Fallback backend:** self-hosted `faster-whisper` (`asr.backend: whisper-local`), behind the
  same interface, for cases where the hosted path doesn't fit — cost at scale, or a client with a
  genuine on-premise requirement.
- `ASRBackend` is the enforced boundary: nothing outside `src/transform/asr/` may import a
  specific backend's SDK, and nothing downstream of `list[Segment]` — chunking, embedding,
  indexing, retrieval, answering — may branch on which backend produced it. Adding a new backend
  means writing one new class, not touching `chunk.py`, `embed.py`, `load/`, or `retrieve/`.
- What stays local and offline regardless of ASR backend choice: chunking, embedding, the vector
  store, reranking, and the retrieval/answer path. This decision concerns only the transcription
  step; it does not make the rest of the system dependent on the network.
- Before sending any client's recordings to a third-party API, confirm nothing in them is
  something that shouldn't leave the building (salaries, personal data) — the absence of a formal
  compliance requirement does not remove the need for that judgment call.

## Consequences

- Diarization for the default (hosted) path comes for free — no `pyannote`, no HF token, no
  `[diarize]` extra needed for the common case. That extra exists solely to support the
  self-hosted fallback (`whisper-local`).
- The transcription cache (re-running does not re-transcribe if the input hash is unchanged)
  matters for cost as well as time on the hosted path — a skipped API call is a skipped charge,
  not just a skipped wait.
- This decision can be revisited per engagement: if a specific client has a hard on-premise
  requirement, the fallback backend already exists behind the same protocol and requires no
  architectural change to select.
