# Spec — Audio2RAG: organizational memory for recorded meetings and training

## Goal

This is a **meeting intelligence / organizational memory** product, not a transcription tool.
The category: companies record lectures, training sessions, and meetings; that knowledge normally
dies in a folder of mp3 files. Audio2RAG turns those recordings into a queryable knowledge base —
"what did we decide about the Q3 budget?" → an answer with a link to the exact minute it was said.
(Same category as Otter, Fireflies, Fathom's meeting-intelligence features.)

**Audio-to-PDF transcription is not the differentiator.** It's a commodity solved by
faster-whisper/WhisperX/Buzz/MacWhisper, or by hosted APIs (AssemblyAI, Deepgram, OpenAI).
What isn't commodity — and is most of this spec — is everything from chunking onward: retrieval
tuned for Polish, grounded answers, citations anchored to a timestamp, and refusal instead of
confabulation when the answer isn't in the recordings.

---

## Tiers

The spec is written so each tier below is a **complete, shippable, demoable product** — not a
partial build of the next one. A client who only buys Starter gets something genuinely useful,
not a stub. This mirrors how the project is actually built: each tier is a stopping point where
tests are green and the tool works end to end.

| | **Starter** | **Standard** | **Advanced** |
|---|---|---|---|
| **Positioning** | Reliable batch transcription tool | + searchable knowledge base | + grounded Q&A, hallucination-resistant |
| **Client can now...** | turn a folder of recordings into clean, timestamped transcripts (PDF/MD/SRT) | ask "find where X was discussed" and jump to the moment | ask a question in plain language and get a sourced answer |
| **Maps to** (`TASKS.md`) | INIT, CANON, INGEST | + CHUNK, INDEX, SEARCH (dense only) | + SEARCH (hybrid+rerank+filters), ANSWER, EVAL |
| **Spec sections below** | §1–2, §8, §9 (partial), §10 | + §3–6 (dense-only), §9 (search) | + §6 (full), §7, §9 (ask), §11 (eval) |

Sections are tagged `[STARTER]` / `[STANDARD]` / `[ADVANCED]` so you can hand a client-facing
excerpt of just their tier without editing the document.

---

## `[STARTER]` ASR is a pluggable backend, not a design commitment

Offline/on-premise is **not a hard requirement** by default (confirmed for this project —
recordings may go through a cloud API; revisit if a specific client needs on-prem). That changes
what's worth building in-house:

| Backend | When to use | Notes |
|---|---|---|
| **Hosted API** (AssemblyAI, Deepgram nova-3) | **default** | word-level timestamps + diarization in one call, no GPU/model management, scales without local compute planning |
| OpenAI Whisper API | fallback | no built-in diarization — pair with a separate diarization step if needed |
| Self-hosted `faster-whisper` | only if: cost at high volume justifies it, or a specific client needs on-prem | keep behind an `ASRBackend` interface so it's swappable, not load-bearing |

**Design consequence:** ASR sits behind a small interface (`transcribe(audio) -> list[Segment]`).
Whichever backend is selected, everything downstream — canonical JSONL, chunking, indexing,
retrieval — is unaffected. Do not couple the rest of the pipeline to any one backend's output shape.
This is also what makes Starter → Standard → Advanced a clean upsell instead of a rewrite: nothing
built at Starter has to change when Standard adds the index.

Practical note: verify before sending a client's recordings to a third party that no meeting on
them touches something that shouldn't leave the building (salaries, personal data) — "not a hard
compliance requirement" doesn't mean "no judgment needed."

---

## `[STARTER]` Key decision: the transcript, not PDF, is the source of truth

The canonical transcript (`.segments.jsonl`) is the single source of truth from Starter onward.
PDF is **one of several renderers** over it, generated on request.

Why PDF is a poor foundation to build on (even at Starter, before RAG exists):

| Problem | Consequence |
|---|---|
| PDF is a presentation format, not a data format | you'd have to parse it back out (`pdfplumber`) — round-tripping for nothing |
| Temporal structure is lost | no timestamps → can't jump to a moment, can't do Standard/Advanced later without redoing this |
| Segment and speaker boundaries are lost | chunking later would cut mid-sentence |
| No place for metadata | course, date, presenter, tags have nowhere to live |

**Rule:** the transcript is written once, in canonical form; PDF/MD/SRT are *rendered* from it,
on request, never automatically. Never the reverse. This single rule is why Standard and Advanced
can be sold later as upsells on an existing Starter delivery, instead of a rebuild.

---

## Architecture

```
                    ┌─────────────────────────────────────┐
  audio ──────────► │ 1. ASR  (pluggable: hosted API /     │  [STARTER]
  (mp3/wav/mp4/...) │    self-hosted) + optional diarize   │
                    └──────────────┬──────────────────────┘
                                   │  segments {start,end,text,speaker}
                                   ▼
                    ┌─────────────────────────────────────┐
                    │ 2. CANONICAL WRITE                  │  [STARTER]
                    │    output/<id>.segments.jsonl       │◄── source of truth,
                    └───────┬───────────────────┬─────────┘    always produced
                            │                   │
        on request ─────────┤                   ├──── Standard+ only
                            ▼                   ▼
              ┌──────────────────────┐  ┌─────────────────────────┐
              │ .pdf  (for reading)  │  │ 3. Chunking (on pauses) │  [STANDARD]
              │ .md   (for notes)    │  │ 4. Embedding (PL-aware) │  [STANDARD]
              │ .srt  (for video)    │  │ 5. Vector store + BM25* │  [STANDARD, *BM25 = ADVANCED]
              └──────────────────────┘  └───────────┬─────────────┘
                                                    ▼
                                        ┌─────────────────────────┐
                                        │ 6. Retrieval + reranker │  [STANDARD: dense only]
                                        │ 7. Answer with citations│  [ADVANCED: hybrid+rerank+answer]
                                        │    [Lecture 3, 14:22]   │
                                        └─────────────────────────┘
```

Starter stops after step 2 (plus optional renderers). Standard adds 3–5 and a dense-only version
of 6 (find the right chunk; client reads it themselves). Advanced completes 6 with hybrid
search + reranking, and adds 7 (an actual generated answer with citations and refusal).

---

## `[STARTER]` 1. ASR — transcription

Behind an `ASRBackend` interface: `transcribe(audio_path, language) -> list[Segment]`.
All backends must return segment-level timestamps and, where available, speaker labels — later
tiers depend on both being present, so get this right even if the client only buys Starter.

**Default: hosted API** (AssemblyAI or Deepgram nova-3).
- Diarization included in the same call — no separate step, no HF token, no `pyannote` dependency
- No GPU/model management, no platform-specific compute table to maintain
- Cost scales with usage (per-minute), not with owning hardware

**Fallback / alternative: self-hosted `faster-whisper`** — kept behind the same interface:
- `vad_filter=True` — strips silence, speedup and fewer hallucinations (meetings have a lot of dead air)
- `word_timestamps=True` — needed for precise citation anchoring later
- `condition_on_previous_text=False` for recordings > 30 min — limits drift and looping
- **Diarization (optional, `--diarize`)**: `pyannote/speaker-diarization-3.1` (requires an HF token) —
  this is the extra plumbing the hosted path avoids.

Backend is chosen in `config.yaml` (`asr.backend: assemblyai | deepgram | whisper-local`).
Nothing downstream of `list[Segment]` knows or cares which backend produced it.

**Supported input extensions** (`config.yaml` `ingest.supported_extensions`): `mp3, wav, m4a,
flac, ogg, mp4, mov, webm`. This is the list the Starter acceptance criterion below ("all listed
audio formats") refers to. `extract/discover.py` rejects anything else with a warning rather than
attempting to re-encode it.

---

## `[STARTER]` 2. Canonical format — `output/<doc_id>.segments.jsonl`

One line per segment:

```json
{"doc_id":"a3f9c1b2","seg":142,"start":852.4,"end":867.1,"speaker":"SPEAKER_00","text":"Deadlock occurs when all four Coffman conditions hold...","conf":-0.21}
```

Plus a sidecar `output/<doc_id>.meta.json`:

```json
{
  "doc_id": "a3f9c1b2",
  "source_path": "recordings/os-lecture-03.mp3",
  "sha256": "a3f9c1b2...",
  "type": "lecture",
  "title": "Operating Systems — Lecture 3",
  "course": "Operating Systems",
  "speakers": {"SPEAKER_00": "Dr. Kowalski"},
  "date": "2026-03-04",
  "duration_s": 5412,
  "language": "pl",
  "model": "large-v3",
  "tags": ["synchronization", "deadlock"],
  "ingested_at": "2026-03-04T18:22:11Z"
}
```

`type` ∈ `lecture | training | meeting | interview | other` — used later to filter queries
(Standard+ feature; the field is cheap to capture now, expensive to backfill later).

Note: `language` refers to the **spoken content**, which stays Polish by default. Code, config
keys, CLI, and documentation are English throughout, at every tier.

---

## `[STANDARD]` 3. Chunking

Naive fixed-size splitting hurts quality. Rules:

- **Target 400–600 tokens** per chunk, **~15% overlap** (2–3 segments)
- **Preferred boundaries**: pause > 1.5 s between segments, sentence end, speaker change
- A chunk never splits a segment in half
- Every chunk carries `start`/`end` in seconds → a citation points straight to the minute of audio

**Contextual header** (contextual retrieval) — prepended to the text *before* embedding. Cheap
accuracy win, because an isolated transcript fragment is often incomprehensible on its own:

```
[Operating Systems — Lecture 3, 2026-03-04, Dr. Kowalski, 14:12–14:27]
...chunk text...
```

---

## `[STANDARD]` 4. Embeddings — models that handle Polish

This is where anglocentric defaults hurt the most, and where "Standard" earns its price over a
generic RAG template.

| Model | Notes |
|---|---|
| `BAAI/bge-m3` | **default** — multilingual, 8192 ctx, emits dense + sparse in one pass, solid Polish |
| `sdadas/mmlw-retrieval-roberta-large` | Polish-specific, very strong on PIRB; needs `zapytanie: ` / `pasaż: ` prefixes |
| `intfloat/multilingual-e5-large` | safe baseline, `query: ` / `passage: ` prefixes |
| ~~`all-MiniLM-L6-v2`~~ | **do not use** — effectively no Polish capability |

Model is configurable in `config.yaml`; **changing it forces a full reindex**. The collection
metadata stores model name and vector dimension so mismatches are detected at query time.

---

## `[STANDARD]` 5. Vector store

| Option | When |
|---|---|
| **LanceDB** | default — on-disk file, no server, good metadata filtering |
| Chroma | simpler API, fine below roughly 100 h of recordings |
| Qdrant (docker) | when you want native dense+sparse hybrid and multi-user access — needed at Advanced |

A BM25 index (`bm25s`, or built into Qdrant) is added at **Advanced** — see §6.

---

## `[STANDARD → ADVANCED]` 6. Retrieval

**Standard scope:** dense search only. `dense_search(query, k)` returns ranked chunks; the client
reads the chunk and jumps to its timestamp themselves. No LLM call, no synthesized answer — this
is "smart search," not "chat with your recordings."

**Advanced scope** adds everything a plain dense search gets wrong on transcripts (speech is
verbose and full of filler, while users query with precise terminology):

```
query
  ├─► dense  (top 30)  ─┐
  ├─► BM25   (top 30)  ─┼─► RRF (fusion) ─► reranker ─► top 5 → LLM
  └─► metadata filters ─┘
```

- **Fusion**: Reciprocal Rank Fusion, `k=60`
- **Reranker**: `BAAI/bge-reranker-v2-m3` — cross-encoder, handles Polish well; usually the single
  largest quality jump in the whole pipeline — this is the main thing separating a "real RAG" from
  a demo that falls apart on a hard question
- **Filters**: `--course`, `--type`, `--after`, `--speaker`, `--tag`
- **Context expansion**: the LLM receives each chunk ± 1 adjacent segment (the answer to a question
  often starts one sentence earlier)

---

## `[ADVANCED]` 7. Answering

- Every answer carries citations formatted as `[title, HH:MM:SS]`
- If retrieval returns nothing above the score threshold, the model must say
  "not present in the recordings" rather than improvise — this is the concrete, testable claim
  behind "hallucination-resistant" in the gig description, backed by the eval harness in §11
- Citations are clickable → open the audio file at that second (UI), or print path + timestamp (CLI)

---

## `[STARTER]` 8. Idempotency

Built once at Starter, holds at every tier without modification:

- `doc_id = sha256(audio bytes)[:16]` — the same file under a different name won't duplicate
- Re-ingest: `DELETE WHERE doc_id = ?` → `INSERT` (delete-then-insert, not per-chunk upsert — the
  chunk count can change with a different model or parameters, relevant once Standard adds an index)
- Metadata-only changes (e.g. fixing a presenter's name) → `audio2rag relabel`, no re-transcription
- Transcription is cached: if `.segments.jsonl` exists and the hash matches, ASR does not run
  again (it is by far the most expensive stage, hosted or self-hosted)

---

## 9. CLI (grows per tier)

```bash
# [STARTER] transcribe + render requested artifacts
audio2rag ingest recordings/os-lecture-03.mp3 \
    --type lecture --course "Operating Systems" \
    --date 2026-03-04 --speaker "Dr. Kowalski" \
    --render pdf,md          # renderers are opt-in, nothing generated by default

audio2rag ingest recordings/ --recursive          # batch, whole directory
audio2rag list                                     # what's in the store
audio2rag rm <doc_id>

# [STANDARD] adds indexing + search (dense only)
audio2rag ingest recordings/os-lecture-03.mp3 --index   # also chunk + embed + store
audio2rag search "deadlock and Coffman conditions" --course "Operating Systems"
audio2rag reindex          # rebuild after an embedding-model change, no re-transcription

# [ADVANCED] adds hybrid search + answering
audio2rag ask "what was said about deadlocks and the Coffman conditions?" \
    --course "Operating Systems"
audio2rag serve --port 8080     # local chat UI, player seeks to cited timestamp
```

---

## `[STARTER]` 10. Outputs

| File | Role | Tier |
|---|---|---|
| `output/<doc_id>.segments.jsonl` | **canonical** — source for everything else | Starter |
| `output/<doc_id>.meta.json` | recording metadata | Starter |
| `output/<doc_id>.pdf` | reading / printing (DejaVuSans, correct Polish glyphs) | Starter |
| `output/<doc_id>.md` | notes, git, Obsidian — headings with timestamps | Starter |
| `output/<doc_id>.srt` | subtitles, if the recording has a video track | Starter |
| `store/` | LanceDB + BM25 index | Standard (BM25 part: Advanced) |

---

## Platform support

Applies at Starter and holds at every later tier. The embedding/reranker columns only become
relevant at Standard/Advanced. The "ASR device" column only matters if `asr.backend: whisper-local`
is selected; with a hosted API backend, no local GPU/CPU work happens for transcription at all.

| OS | ASR device (self-hosted only) | Compute | Embeddings (Standard+) | Reranker (Advanced) |
|---|---|---|---|---|
| Windows 11+ | CUDA | float16 | CUDA | CUDA |
| Linux | CPU (CUDA if present) | int8 | CPU/CUDA | CPU/CUDA |
| macOS (Apple Silicon) | CPU | int8 | MPS | MPS |

A CPU reranker adds roughly 1–3 s per query at 60 candidates — acceptable, though on a large
corpus it's worth dropping to top 20 candidates.

---

## Acceptance criteria (per tier — a tier ships when its own list is green)

**Starter:**
- Hosted API: 60 min of audio transcribes well under the recording's own duration
  (API-bound — track p50/p95 turnaround, not a fixed local benchmark)
- Self-hosted fallback: < 5 min on Windows/CUDA, < 15 min on CPU, for 60 min of audio
- All listed audio formats supported without re-encoding
- PDF (when rendered) shows Polish characters correctly; nothing rendered unless requested
- Idempotent: 2× ingest = 1 canonical transcript, no duplication, no data loss

**Standard (adds):**
- Re-ingesting the same file does **not** increase the chunk count in the store
- Metadata filters (`--course`) genuinely narrow results to a single course
- An embedding-model change is detected and blocks querying a stale index, with a
  "run `reindex`" message
- Dense search returns a relevant chunk in the top 5 for an in-corpus query, in a quick
  spot-check set (~10 questions) — full recall@5 measurement is an Advanced deliverable

**Advanced (adds):**
- Every answer contains at least one citation with `doc_id` + timestamp, accurate to ± 5 s
- A Polish-language query for a term spoken in a recording returns the right fragment in the top 5
  (~30 test questions over a reference corpus; target recall@5 ≥ 0.85, measured by the eval harness)
- A query about something absent from the corpus produces an explicit refusal, not a confabulation
- Query latency: < 3 s on CUDA, < 8 s on CPU (excluding LLM generation time and ASR API latency)

---

## `[ADVANCED]` 11. Evaluation

This is what turns "hallucination-resistant" from a marketing claim into a number a client can
be shown: ~30 reference questions with known correct doc_id + timestamp window, scored for
recall@5 and MRR, with a committed baseline. Selling point: you can hand a client a before/after
table when tuning chunk size, candidate counts, or the refusal threshold — a generic RAG gig
rarely offers this.

---

## Open decisions

1. **Answering LLM** (Advanced) — local (Bielik / Llama 3.1 via Ollama) or API? With ASR already
   going through a cloud API, "keep everything local" is no longer a constraint forcing this
   choice — it's now purely a quality-vs-cost call. API models currently do noticeably better
   Polish.
2. ~~Diarization on by default?~~ **Resolved by the ASR pivot** — the default hosted backend
   returns diarization in the same call at no extra integration cost, so it can be on by default
   for every `type`, not just `meeting`.
3. **Collect user transcript corrections?** (domain terminology, surnames) — a simple
   `corrections.yaml` with post-ASR replacement rules is a large win for very little cost, doable
   at any tier.
4. **Audio retention** — keep original files after ingest (needed for timestamp playback in
   Advanced's `serve`), or store paths only? Worth deciding deliberately once audio has already
   left the building for transcription once.