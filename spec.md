# Spec — Audio2RAG: organizational memory for recorded meetings and training

## Goal

A **meeting intelligence / organizational memory** product, not a transcription tool. Companies
record lectures, training sessions and meetings; that knowledge dies in a folder of mp3 files.
Audio2RAG turns those recordings into a queryable knowledge base — "what did we decide about the
Q3 budget?" → an answer with a link to the exact minute it was said.

Audio-to-PDF transcription is a commodity (faster-whisper, AssemblyAI, Deepgram). The product is
everything from chunking onward: retrieval tuned for Polish, grounded answers, citations anchored
to a timestamp, and refusal instead of confabulation when the answer isn't in the recordings.

Design rationale is not repeated here — see
[ADR 0001](docs/decisions/0001-jsonl-as-canonical-source.md) (why JSONL, not PDF, is canonical) and
[ADR 0002](docs/decisions/0002-pluggable-asr-backend.md) (why ASR isn't assumed local).

---

## Tiers

Each tier is a **complete, shippable product**, not a partial build of the next one, and a stopping
point where tests are green and the tool works end to end.

| | **Starter** | **Standard** | **Advanced** |
|---|---|---|---|
| **Positioning** | Reliable batch transcription tool | + searchable knowledge base | + grounded Q&A, hallucination-resistant |
| **Client can now...** | turn a folder of recordings into clean, timestamped transcripts (PDF/MD/SRT) | ask "find where X was discussed" and jump to the moment | ask a question in plain language and get a sourced answer |
| **Maps to** (`TASKS.md`) | INIT, CANON, INGEST | + CHUNK, INDEX, SEARCH (dense only) | + SEARCH (hybrid+rerank+filters), ANSWER, EVAL |
| **Spec sections** | §1–2, §8, §9 (partial), §10 | + §3–6 (dense-only), §9 (search) | + §6 (full), §7, §9 (ask), §11 |

Sections are tagged `[STARTER]` / `[STANDARD]` / `[ADVANCED]` so a client-facing excerpt of one
tier can be cut without editing the document.

---

## Architecture

```
                    ┌─────────────────────────────────────┐
  audio ──────────► │ 1. ASR  (pluggable: hosted API /     │  [STARTER]
  (mp3/wav/mp4/...) │    whisper-local) + optional diarize │
                    └──────────────┬──────────────────────┘
                                   │  Segment {start,end,text,speaker,conf}
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
                                        │ 7. Answer with citations│  [ADVANCED: hybrid+rerank]
                                        │    [OS — Lecture 3,     │
                                        │     00:14:22]           │
                                        └─────────────────────────┘
```

Starter stops after step 2 (plus optional renderers). Standard adds 3–5 and a dense-only version
of 6. Advanced completes 6 with hybrid search + reranking and adds 7.

---

## `[STARTER]` 1. ASR — transcription

Every backend sits behind one interface:

```python
class ASRBackend(Protocol):
    def transcribe(self, path: Path, language: str) -> list[Segment]: ...
```

All backends return segment-level timestamps and, where available, speaker labels — later tiers
depend on both, so this holds even if the client only buys Starter. Nothing downstream of
`list[Segment]` knows which backend produced it.

| Backend | When | Notes |
|---|---|---|
| **AssemblyAI / Deepgram nova-3** | **default** | word-level timestamps + diarization in one call, no GPU or model management |
| OpenAI Whisper API | alternative | no built-in diarization — needs a separate diarization step |
| `faster-whisper` (`whisper-local`) | on-prem requirement, or cost at high volume | the only path that needs CUDA, `pyannote` and an HF token |

Backend is selected in `config.yaml` (`asr.backend: assemblyai | deepgram | whisper-local`).

**`whisper-local` settings:** `vad_filter=True` (meetings have a lot of dead air — speedup and
fewer hallucinations), `word_timestamps=True` (precise citation anchoring),
`condition_on_previous_text=False` for recordings > 30 min (limits drift and looping).
Diarization is optional (`--diarize`, `pyannote/speaker-diarization-3.1`, requires `HF_TOKEN`);
when absent, ingest proceeds without speaker labels and warns.

**`conf` is normalised at the interface**: `Segment.conf` is `float | None` in `0.0–1.0`, higher is
better. Hosted backends pass their confidence through; `whisper-local` converts `avg_logprob` via
`exp()`. Downstream code must never assume a log-probability scale.

**Privacy:** with a hosted backend, recordings leave the machine. `asr.backend: whisper-local` is
the on-prem path. Confirm per client that no recording touches something that shouldn't leave the
building (salaries, personal data) before selecting a hosted backend.

**Supported input extensions** (`config.yaml` `ingest.supported_extensions`): `mp3, wav, m4a, flac,
ogg, mp4, mov, webm`. `extract/discover.py` rejects anything else with a warning rather than
re-encoding it.

---

## `[STARTER]` 2. Canonical format — `output/<doc_id>.segments.jsonl`

One line per segment:

```json
{"doc_id":"a3f9c1b2","seg":142,"start":852.4,"end":867.1,"speaker":"SPEAKER_00","text":"Deadlock occurs when all four Coffman conditions hold...","conf":0.81}
```

Sidecar `output/<doc_id>.meta.json`:

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
  "asr_backend": "assemblyai",
  "asr_model": "best",
  "diarized": true,
  "tags": ["synchronization", "deadlock"],
  "ingested_at": "2026-03-04T18:22:11Z"
}
```

`asr_backend` + `asr_model` record what actually produced the transcript — needed to explain
quality differences and to decide what to re-transcribe after a backend change. Never contains
API keys.

`type` ∈ `lecture | training | meeting | interview | other` — used for query filters at Standard+;
cheap to capture now, expensive to backfill.

`language` refers to the **spoken content** (Polish by default). Code, config keys, CLI and
documentation are English at every tier.

---

## `[STANDARD]` 3. Chunking

- **Target 400–600 tokens** per chunk, **~15% overlap** (2–3 segments)
- **Preferred boundaries**: pause > 1.5 s between segments, sentence end, speaker change
- A chunk never splits a segment in half
- Every chunk carries `start`/`end` in seconds → a citation points to the minute of audio

**Contextual header** — prepended to the text *before* embedding, because an isolated transcript
fragment is often incomprehensible on its own:

```
[Operating Systems — Lecture 3, 2026-03-04, Dr. Kowalski, 00:14:12–00:14:27]
...chunk text...
```

---

## `[STANDARD]` 4. Embeddings — models that handle Polish

| Model | Notes |
|---|---|
| `BAAI/bge-m3` | **default** — multilingual, 8192 ctx, solid Polish |
| `sdadas/mmlw-retrieval-roberta-large` | Polish-specific, very strong on PIRB; needs `zapytanie: ` / `pasaż: ` prefixes |
| `intfloat/multilingual-e5-large` | safe baseline, `query: ` / `passage: ` prefixes |
| ~~`all-MiniLM-L6-v2`~~ | **do not use** — effectively no Polish capability |

Only bge-m3's **dense** output is used. Its learned-sparse output is deliberately ignored — lexical
matching comes from BM25 (§6), so the lexical path stays identical across embedding models and
survives a model swap without re-tuning fusion.

Model is configurable; **changing it forces a full reindex**. Collection metadata stores model name
and vector dimension so mismatches are detected at query time.

---

## `[STANDARD]` 5. Vector store

**LanceDB** — on-disk file, no server, good metadata filtering. Chosen for the whole product;
BM25 comes from a separate `bm25s` index at Advanced (§6), so no server is needed at any tier.

Qdrant (docker) is the documented migration path **only** if multi-user concurrent access becomes
a requirement. It is not needed for hybrid search — `bm25s` + RRF covers that locally.

---

## `[STANDARD → ADVANCED]` 6. Retrieval

**Standard scope:** dense search only. `dense_search(query, k)` returns ranked chunks; the client
reads the chunk and jumps to its timestamp. No LLM call — this is "smart search", not "chat with
your recordings".

**Advanced scope** adds what plain dense search gets wrong on transcripts (speech is verbose and
full of filler; users query with precise terminology):

```
query
  ├─► dense  (top 30)  ─┐
  ├─► BM25   (top 30)  ─┼─► RRF (fusion) ─► reranker ─► top 5 → LLM
  └─► metadata filters ─┘
```

- **Fusion**: Reciprocal Rank Fusion, `k=60`
- **Reranker**: `BAAI/bge-reranker-v2-m3` — cross-encoder, handles Polish well; usually the single
  largest quality jump in the pipeline
- **Filters**: `--course`, `--type`, `--after`, `--speaker`, `--tag`
- **Context expansion**: the LLM receives each chunk ± 1 adjacent segment (an answer often starts
  one sentence earlier)

---

## `[ADVANCED]` 7. Answering

- **Canonical citation format: `[title, HH:MM:SS]`** — e.g. `[Operating Systems — Lecture 3,
  00:14:22]`. This exact form is used in answers, in chunk contextual headers (§3) and in the eval
  harness (§11); nothing renders timestamps differently.
- If retrieval returns nothing above the score threshold, the answer is an explicit refusal
  ("not present in the recordings"), never an improvisation. This is the testable claim behind
  "hallucination-resistant", backed by §11.
- Citations are clickable → open the audio at that second (UI), or print path + timestamp (CLI)

---

## `[STARTER]` 8. Idempotency

Built at Starter, unchanged at every tier:

- `doc_id = sha256(audio bytes)[:16]` — the same file under a different name won't duplicate
- Re-ingest: `DELETE WHERE doc_id = ?` → `INSERT` (not per-chunk upsert — chunk count changes with
  a different model or parameters)
- Metadata-only changes (e.g. fixing a presenter's name) → `audio2rag relabel`, no re-transcription
- Transcription is cached: if `.segments.jsonl` exists and the hash matches, ASR does not run again
  (by far the most expensive stage, hosted or local)

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
| `store/lance/` | LanceDB vector index | Standard |
| `store/bm25/` | BM25 index (`bm25s`) | Advanced |

---

## Platform support

The ASR-device column applies **only** to `asr.backend: whisper-local`; with a hosted backend no
local compute is used for transcription. Embedding/reranker columns apply from Standard/Advanced.

| OS | ASR device (whisper-local only) | Compute | Embeddings (Standard+) | Reranker (Advanced) |
|---|---|---|---|---|
| Windows 11+ | CUDA | float16 | CUDA | CUDA |
| Linux | CPU (CUDA if present) | int8 | CPU/CUDA | CPU/CUDA |
| macOS (Apple Silicon) | CPU | int8 | MPS | MPS |

A CPU reranker adds roughly 1–3 s per query at 60 candidates — acceptable; on a large corpus drop
to top 20 candidates.

---

## Acceptance criteria (a tier ships when its own list is green)

**Starter:**
- Hosted backend: 60 min of audio completes in **≤ 15 min wall-clock, p95 over ≥ 10 runs**
  (API-bound; the number is a regression guard, not a compute benchmark)
- Every listed audio format ingests without re-encoding
- PDF (when rendered) shows Polish characters correctly; nothing is rendered unless requested
- Idempotent: 2× ingest = 1 canonical transcript, no duplication, no data loss
- `meta.json` records `asr_backend` / `asr_model` and contains no secrets
- *Gates `whisper-local` only:* 60 min of audio in < 5 min on CUDA, < 15 min on CPU

**Standard (adds):**
- Re-ingesting the same file does **not** increase the chunk count in the store
- Metadata filters (`--course`) genuinely narrow results to a single course
- An embedding-model change is detected and blocks querying a stale index, with a "run `reindex`"
  message
- Dense search returns a relevant chunk in the top 5 for an in-corpus query, on a ~10-question
  spot-check set (full recall@5 measurement is an Advanced deliverable)

**Advanced (adds):**
- Every answer carries ≥ 1 citation with `doc_id` + timestamp, accurate to ± 5 s, in the §7 format
- A Polish-language query for a term spoken in a recording returns the right fragment in the top 5
  (~30 test questions, target recall@5 ≥ 0.85, measured by §11)
- A query about something absent from the corpus produces an explicit refusal
- Query latency < 3 s on CUDA, < 8 s on CPU (excluding LLM generation and ASR API latency)

---

## `[ADVANCED]` 11. Evaluation

~30 reference questions with known correct `doc_id` + timestamp window, scored for **recall@5** and
**MRR**, with a committed baseline. Turns "hallucination-resistant" into a number, and gives a
before/after table when tuning chunk size, candidate counts or the refusal threshold.

---

## Open decisions

1. **Answering LLM** (Advanced) — local (Bielik / Llama 3.1 via Ollama) or API? Purely a
   quality-vs-cost call; API models currently do noticeably better Polish.
2. **Collect user transcript corrections?** A `corrections.yaml` with post-ASR replacement rules
   (domain terminology, surnames) — large win for little cost, doable at any tier.
3. **Audio retention** — keep original files after ingest (needed for timestamp playback in
   `serve`), or store paths only?