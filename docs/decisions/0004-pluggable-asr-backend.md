# 0004 — Pluggable ASR backend; offline is not a hard requirement

**Status:** Accepted
**Date:** 2026-08-04
**Related:** 0002 (originally assumed local `faster-whisper` as the only ASR path)

## Context

The project originally assumed self-hosted `faster-whisper` as the sole transcription engine, with
a platform-specific compute table (CUDA on Windows, CPU int8 on Linux/macOS) and optional
diarization via `pyannote/speaker-diarization-3.1`, which requires a Hugging Face token and adds a
non-trivial dependency (`[diarize]` extra).

Revisiting the requirement: offline / on-premise processing is **not** a hard constraint for this
project. Recordings may be sent to a third-party API. This was an assumption carried over from an
earlier framing of the project, not a stated client or compliance requirement.

Given that, self-hosting Whisper stops being the obvious default:

- Audio-to-transcription is a commodity problem (`faster-whisper`, `WhisperX`, hosted APIs like
  AssemblyAI and Deepgram all solve it well); it is explicitly **not** where this project's value
  lives (see `spec.md` Goal — the value is in chunking, retrieval, and grounded answering).
- Hosted APIs (AssemblyAI, Deepgram nova-3) return word-level timestamps *and* diarization in a
  single call, with no GPU/model management and no HF token plumbing.
- Self-hosting only earns its complexity back at high enough volume, or when a specific
  engagement genuinely needs on-premise processing (a case that can still arise per-client, and
  should not require redesigning the pipeline when it does).

## Decision

ASR sits behind a small interface:

```python
class ASRBackend(Protocol):
    def transcribe(self, path: Path, language: str) -> list[Segment]: ...
```

- **Default backend:** a hosted API (AssemblyAI or Deepgram nova-3), selected via
  `asr.backend: assemblyai | deepgram` in `config.yaml`. API keys come from the environment
  (`ASSEMBLYAI_API_KEY` / `DEEPGRAM_API_KEY`) and are never logged or written into `meta.json`.
- **Fallback backend:** self-hosted `faster-whisper` (`asr.backend: whisper-local`), kept behind
  the same interface, for cases where the hosted path doesn't fit — cost at scale, or a client
  with a genuine on-premise requirement.
- Nothing outside `src/transform/asr/` may import a specific backend's SDK. Nothing downstream of
  `list[Segment]` — chunking, embedding, indexing, retrieval, answering — may branch on which
  backend produced it.
- What stays local and offline regardless of ASR backend choice: chunking, embedding, the vector
  store, reranking, and the retrieval/answer path. The ASR pivot only concerns the transcription
  step; it does not make the rest of the system dependent on the network.
- Before sending any client's recordings to a third-party API, confirm nothing in them is
  something that shouldn't leave the building (salaries, personal data) — the absence of a formal
  compliance requirement does not remove the need for that judgment call.

## Consequences

- Diarization for the default (hosted) path comes for free — no `pyannote`, no HF token, no
  `[diarize]` extra needed for the common case. That extra now exists solely to support the
  self-hosted fallback.
- The platform-support table (Windows/CUDA, Linux/macOS CPU) that used to describe the whole ASR
  step now only applies when `whisper-local` is selected. It still applies unconditionally to
  local embedding and reranking inference, which are unaffected by this decision.
- Adding a new ASR backend (another hosted vendor, a different local model) means writing one new
  class behind `ASRBackend` — it does not require touching `chunk.py`, `embed.py`, `load/`, or
  `retrieve/`.
- The transcription cache (re-running does not re-transcribe if the input hash is unchanged) now
  matters for cost as well as time — a skipped hosted-API call is a skipped charge, not just a
  skipped wait.
- This decision can be revisited per engagement: if a specific client has a hard on-premise
  requirement, the fallback backend already exists and requires no architectural change to select.
