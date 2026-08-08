# TASKS.md — Audio2RAG build plan

Incremental plan, one task per Claude Code session. Each phase ends at a point where
`pytest -m "not slow"` is green and the tool works end to end — no half-built state is ever
committed. This mirrors spec.md's tiers directly, so a task list doubles as a sellable
scope-of-work: **Starter = phases 1–3, Standard = phases 4–6 (dense-only), Advanced = phases
6 (full) –8.**

## Phases at a glance

| # | Phase | Prefix | Tasks | Tier | What you have when it's done |
|---|---|---|---|---|---|
| 1 | Bootstrap | `INIT` | 3 | — (prerequisite) | Repo scaffold + forward-looking ADRs the rest of the plan implements |
| 2 | Canonical transcript | `CANON` | 7 | **Starter** | Timestamped JSONL as source of truth; PDF/MD/SRT rendered from it |
| 3 | Ingest & CLI | `INGEST` | 6 | **Starter** | Real CLI, metadata, batch ingest, pluggable ASR (hosted default) |
| 4 | Chunking | `CHUNK` | 2 | **Standard** | Speech split on pauses and speaker turns, not blindly |
| 5 | Index | `INDEX` | 4 | **Standard** | Embeddings in a local vector store, idempotent |
| 6 | Search | `SEARCH` | 6 | **Standard** (dense+filters) / **Advanced** (hybrid+rerank+expansion) | Retrieval, from "find the chunk" to "grounded answer" |
| 7 | Answer | `ANSWER` | 4 | **Advanced** | Answers with citations, refusal when unsupported |
| 8 | Eval | `EVAL` | 3 | **Advanced** | Measured quality, tuning, optional UI |

Task IDs are `PREFIX-n` (e.g. `CANON-5`). Each task is also tagged `[STARTER]` / `[STANDARD]` /
`[ADVANCED]` so a scope-of-work for a given tier can be extracted by grepping the tag.

## How to work through this

- **One task per session.** Start a fresh Claude Code session per task; paste the task block
  as the prompt. Do not batch tasks — the point of the split is a small diff and a small context.
- **Green tests before and after.** `pytest -m "not slow"` must pass at the end of every task.
- **One commit per task**, message `INIT-1: <title>`. Tick the checkbox here in the same commit.
- **Respect "Out of scope".** It exists to stop scope creep mid-task, which is the main way an
  agent gets lost in a migration this size, and the main way a fixed-price tier quietly balloons.
- If a task turns out to need more than ~300 changed lines, stop and split it.

Phases 1–3 leave you with a strictly better transcription tool (timestamps, MD/SRT, caching, real
CLI, pluggable ASR) and no RAG — this is the Starter deliverable, and a valid place to pause and
ship or invoice.

---

## Phase 1 — INIT: repo bootstrap

Starting from zero: no existing pipeline, no legacy behaviour to protect. This phase produces a
scaffold and the two forward-looking ADRs that the rest of the plan implements — it is not a
safety net for a refactor, since there is nothing yet to refactor. (If you're picking this plan up
against a repo that already has working code, do SAFE-2-style characterization tests instead of
INIT-1 — the two are mutually exclusive, not sequential.)

### [x] INIT-1 — Repo scaffold and tooling
**Touch:** `pyproject.toml`, `config.yaml`, `.env.example`, `.gitignore`, empty `src/` package
tree matching CLAUDE.md's pipeline layout (`models/`, `extract/`, `transform/asr/`, `load/`,
`retrieve/`, `cli.py`), empty `tests/` tree with `pytest` configured (`slow` marker registered,
`pythonpath = ["src"]`).
**Done when:** `pytest -m "not slow"` exits 0 with zero tests collected, `ruff check .` is clean
on the empty tree, `pip install -e ".[dev]"` succeeds.
**Out of scope:** any pipeline logic, any ADR content.

### [x] INIT-2 — ADR: JSONL as canonical source
**Touch:** `docs/decisions/0001-jsonl-as-canonical-source.md`
**Done when:** ADR records the decision "JSONL is the source of truth, PDF is a renderer",
the alternatives considered (index PDFs directly / index plain .txt), and the consequences.
**Out of scope:** any code.

### [x] INIT-3 — ADR: pluggable ASR backend, not offline-first
**Touch:** `docs/decisions/0002-pluggable-asr-backend.md`
**Done when:** ADR records that offline/on-prem is not a hard requirement for this project,
names the default (hosted API) and fallback (self-hosted `faster-whisper`) backends, states the
`ASRBackend` protocol as the enforced boundary, and notes what still stays local regardless
(embedding, vector store, reranking, retrieval).
**Out of scope:** any code — but this ADR is what CANON-2 and INGEST-1 implement, read it first.

---

## Phase 2 — `[STARTER]` CANON: canonical transcript format

### [x] CANON-1 — Segment and metadata schemas
**Touch:** `src/models/schemas.py`, `tests/models/test_schemas.py`
**Done when:** `Segment` (doc_id, seg, start, end, text, speaker|None, conf) and
`TranscriptMeta` (per spec §2) exist as Pydantic models with validation
(`end > start`, `seg >= 0`, non-empty text) and round-trip tests.
**Out of scope:** using them anywhere yet.

### [x] CANON-2 — `ASRBackend` protocol and fake
**Why:** this is where the ASR pivot (ADR 0004) becomes real code — get the interface right
before any concrete backend exists, so hosted/self-hosted are equally easy to add.
**Touch:** `src/transform/asr/base.py` (protocol), `tests/fakes.py` (`FakeASRBackend`),
`src/transform/transcribe.py` (thin wrapper calling whichever backend is configured)
**Done when:** `ASRBackend.transcribe(path, language) -> list[Segment]` is a `Protocol`;
`FakeASRBackend` returns a fixed, injectable segment list. The wrapper takes a backend as a
constructor argument (dependency injection), never imports a concrete backend at module level.
**Out of scope:** any real backend implementation — that's INGEST-1.

### [x] CANON-3 — doc_id and file hashing
**Touch:** `src/extract/`, tests
**Done when:** `compute_doc_id(path) -> str` returns `sha256(bytes)[:16]`, is stable across runs,
and differs for different content. Same content under a different filename yields the same id.
**Out of scope:** using doc_id for output naming yet.

### [x] CANON-4 — Write the canonical artifacts
**Touch:** `src/load/write_canonical.py`, tests
**Done when:** `output/<doc_id>.segments.jsonl` (one JSON object per line) and
`output/<doc_id>.meta.json` are written and can be read back into the schemas losslessly.
Re-running overwrites, never appends — assert line count is unchanged after two runs.
**Out of scope:** touching the PDF renderer.

### [x] CANON-5 — PDF becomes a renderer over JSONL
**Why:** this is the load-bearing task of the whole migration.
**Touch:** `src/load/render_pdf.py`, tests
**Done when:** the PDF renderer takes `list[Segment]` (or reads the JSONL) instead of a raw
string and emits timestamps in the output. No code path outside
`load/` produces a PDF. PDF is only generated when explicitly requested (`--render pdf`), never
as a default side effect of ingest.
**Out of scope:** MD/SRT renderers.

### [x] CANON-6 — Markdown and SRT renderers
**Touch:** `src/load/render_md.py`, `src/load/render_srt.py`, tests
**Done when:** both read the same `list[Segment]` and produce valid output, both opt-in via
`--render`; SRT timecodes are `HH:MM:SS,mmm` and validated by a parser in the test.
**Out of scope:** deciding which renderers run by default — none do, by design.

### [x] CANON-7 — Transcription cache
**Touch:** `src/transform/`, CLI entry point, tests
**Done when:** if `<doc_id>.segments.jsonl` exists and `meta.json` records a matching hash, the
ASR backend is not invoked — assert the fake backend's call count is zero, regardless of which
backend is configured (this matters more for hosted backends, where a skipped call also saves
money, not just time). `--force` bypasses the cache.
**Out of scope:** cache invalidation on model change — covered by INDEX-4's guard.

---

## Phase 3 — `[STARTER]` INGEST: ingest, metadata, CLI

### [x] INGEST-1 — Hosted ASR backend (default)
**Touch:** `src/transform/asr/hosted.py`, `config.yaml`, tests (against `FakeASRBackend`/mocked HTTP)
**Done when:** a real hosted backend (AssemblyAI or Deepgram nova-3) implements `ASRBackend`,
returns segments with speaker labels from the API's diarization, and is selected via
`asr.backend: assemblyai | deepgram` in config. API key read from environment
(`ASSEMBLYAI_API_KEY` / `DEEPGRAM_API_KEY`), never logged, never written to `meta.json`.
No non-`slow` test makes a real HTTP call.
**Out of scope:** the self-hosted fallback — that's INGEST-5.

### [ ] INGEST-2 — `ingest` subcommand with metadata flags
**Touch:** `src/cli.py`, tests
**Done when:** `audio2rag ingest <path> --type --title --course --date --speaker --tag --render`
works, values land in `meta.json`, `type` is validated against the enum
(`lecture|training|meeting|interview|other`), and `--render pdf,md,srt` controls which renderers
run (none by default).
**Out of scope:** `ask`, `serve` — those are Advanced (Phase 7).

### [ ] INGEST-3 — Directory ingest
**Touch:** `src/extract/discover.py`, CLI, tests
**Done when:** `ingest <dir> --recursive` finds all supported extensions, skips unsupported files
with a warning, and continues past a single file's failure instead of aborting the batch.
**Out of scope:** parallelism.

### [ ] INGEST-4 — `list`, `rm`, `relabel`
**Touch:** CLI, `src/load/`, tests
**Done when:** `list` prints doc_id/title/type/duration from meta files; `rm <doc_id>` removes
all artifacts for that id; `relabel <doc_id> --speaker ...` edits `meta.json` only and never
re-runs transcription.
**Out of scope:** vector store cleanup in `rm` — add it in INDEX-3.

### [ ] INGEST-5 — Self-hosted `faster-whisper` fallback backend
**Touch:** `src/transform/asr/whisper_local.py`, `pyproject.toml` (`[cuda]` extra), tests
**Done when:** implements `ASRBackend` with `vad_filter=True`, `word_timestamps=True`, and
`condition_on_previous_text=False` for inputs over 30 minutes, selectable via
`asr.backend: whisper-local`. CPU fallback works without CUDA installed. Never imported unless
this backend is selected (lazy import).
**Out of scope:** diarization for this backend — that's INGEST-6. Note this backend is now a
fallback, not the default — build it after the hosted path works, not before.

### [ ] INGEST-6 — Diarization for the self-hosted fallback
**Why:** the hosted default backend (INGEST-1) already returns speaker labels for free — this
task only matters if `whisper-local` is in use.
**Touch:** `src/transform/asr/whisper_local.py` (diarization hook), `pyproject.toml`
(`[diarize]` extra), tests
**Done when:** with `--diarize` and `pyannote` present, `whisper-local` segments carry speaker
labels; with the extra or `HF_TOKEN` missing, ingest completes without labels and logs a warning
— it does not crash. Import of `pyannote` is lazy and inside the function.
**Out of scope:** mapping `SPEAKER_00` to real names (that is `relabel`).

---

## Phase 4 — `[STANDARD]` CHUNK: chunking

### [ ] CHUNK-1 — Chunk schema and chunker
**Why:** pure function over data, no models needed — fully testable and a good place to be strict.
**Touch:** `src/models/schemas.py` (`Chunk`), `src/transform/chunk.py`, tests
**Done when:** chunker groups segments into 400–600 token chunks with ~15% overlap, splits
preferentially at pauses > 1.5 s / sentence ends / speaker changes, never splits a segment, and
each chunk carries `start`, `end`, `doc_id`, `segment_ids`. Tests cover: single short segment,
one long monologue, rapid speaker alternation, and a segment longer than the target size.
**Out of scope:** embedding.

### [ ] CHUNK-2 — Contextual header
**Touch:** `src/transform/chunk.py`, tests
**Done when:** `Chunk.embed_text` is the header (`[title, date, speaker, HH:MM–HH:MM]`) plus the
body, while `Chunk.display_text` stays clean for citation display. Both asserted separately.
**Out of scope:** proving it improves retrieval — EVAL-1.

---

## Phase 5 — `[STANDARD]` INDEX: embeddings and vector store

### [ ] INDEX-1 — Embedder protocol and fake
**Touch:** `src/transform/embed.py`, `tests/fakes.py`
**Done when:** an `Embedder` Protocol (`embed(texts) -> ndarray`, `name`, `dim`) exists, plus a
`FakeEmbedder` producing deterministic vectors from a text hash. Real `bge-m3` implementation is
behind the `[rag]` extra with a lazy import and is marked `slow` in tests. This runs locally
regardless of ASR backend — see ADR 0004.
**Out of scope:** the store.

### [ ] INDEX-2 — LanceDB writer
**Touch:** `src/load/vector_store.py`, tests (using `FakeEmbedder`, temp dir)
**Done when:** chunks are written with all metadata fields queryable; collection metadata records
embedding model name and dim. Tests do not download anything.
**Out of scope:** search.

### [ ] INDEX-3 — Delete-then-insert idempotency
**Touch:** `src/load/vector_store.py`, `rm` command, tests
**Done when:** ingesting the same file twice leaves the chunk count unchanged; ingesting after a
chunk-size config change replaces cleanly with no orphans; `rm <doc_id>` removes chunks too.
**Out of scope:** BM25 index (added in SEARCH-2, an Advanced task — then wire its deletion here too).

### [ ] INDEX-4 — Model mismatch guard and `reindex`
**Touch:** `src/load/vector_store.py`, CLI, tests
**Done when:** querying a store built with a different embedder raises a clear error naming both
models and telling the user to run `reindex`. `reindex` rebuilds from existing JSONL without
re-transcribing (assert the ASR fake is never called).
**Out of scope:** retrieval quality.

---

## Phase 6 — SEARCH: retrieval (`[STANDARD]` dense+filters, `[ADVANCED]` hybrid+rerank+expansion)

### [ ] SEARCH-1 — `[STANDARD]` Dense search
**Touch:** `src/retrieve/search.py`, tests with `FakeEmbedder` and a toy corpus
**Done when:** `dense_search(query, k)` returns `list[RetrievalHit]` with scores, ordered. This
alone is the Standard-tier deliverable: "smart search," no LLM call yet.
**Out of scope:** fusion, reranking, LLM.

### [ ] SEARCH-2 — `[ADVANCED]` BM25 index and search
**Touch:** `src/load/bm25_index.py`, `src/retrieve/search.py`, tests
**Done when:** BM25 index is built during ingest, deleted alongside vectors in INDEX-3's path, and
`sparse_search(query, k)` returns comparable hits. A test shows BM25 finding an exact rare term
that the fake dense path misses.
**Out of scope:** fusion.

### [ ] SEARCH-3 — `[ADVANCED]` RRF fusion
**Touch:** `src/retrieve/fusion.py`, tests
**Done when:** `rrf(dense_hits, sparse_hits, k=60)` is a pure function with hand-computed
expected rankings in the tests, including disjoint result sets and full overlap.
**Out of scope:** reranking.

### [ ] SEARCH-4 — `[ADVANCED]` Reranker behind a protocol
**Touch:** `src/retrieve/rerank.py`, `tests/fakes.py`
**Done when:** `Reranker` Protocol plus `FakeReranker` (deterministic, e.g. reverses order) so
wiring is testable; real `bge-reranker-v2-m3` behind the `[rag]` extra, marked `slow`. Reranking
is skippable via `--no-rerank`. This is usually the single biggest quality jump — the main thing
that justifies the Advanced price over Standard.
**Out of scope:** tuning candidate counts.

### [ ] SEARCH-5 — `[STANDARD]` Metadata filters
**Touch:** `src/retrieve/search.py`, CLI, tests
**Done when:** `--course --type --after --before --speaker --tag` restrict results; a test with
two courses in the corpus proves no cross-contamination. Filters apply *before* reranking, and
work identically whether reranking (Advanced) is enabled or not (Standard).
**Out of scope:** query parsing / natural-language filters.

### [ ] SEARCH-6 — `[ADVANCED]` Context expansion
**Touch:** `src/retrieve/expand.py`, tests
**Done when:** each hit is expanded with ±1 adjacent segment read from the JSONL, correctly
clamped at document start/end, with no duplicated text when two hits are adjacent.
**Out of scope:** answering.

---

## Phase 7 — `[ADVANCED]` ANSWER: answering

### [ ] ANSWER-1 — Answer and Citation schemas
**Why:** encoding "no answer without a source" in the type system is more reliable than prompting
— this is the concrete mechanism behind any "hallucination-resistant" claim in the gig description.
**Touch:** `src/models/schemas.py`, tests
**Done when:** `Citation` (doc_id, title, start, end) and `Answer` with a non-empty citations list;
`Refusal` is a separate type. Constructing an `Answer` with zero citations raises.
**Out of scope:** the LLM call.

### [ ] ANSWER-2 — Prompts module and LLM protocol
**Touch:** `src/retrieve/prompts.py`, `src/retrieve/llm.py`, `tests/fakes.py`
**Done when:** all prompt strings live in `prompts.py` and are versioned; an `LLMClient` Protocol
plus `FakeLLM` allow full answer-path tests with no model. No inline prompt strings elsewhere
(add a test that greps `src/` for them if useful).
**Out of scope:** choosing the production LLM — still an open decision in the spec.

### [ ] ANSWER-3 — Threshold and refusal path
**Touch:** `src/retrieve/answer.py`, tests
**Done when:** when the top reranked score falls below the configured threshold, the pipeline
returns `Refusal` without calling the LLM at all (assert `FakeLLM` was not invoked).
**Out of scope:** tuning the threshold — EVAL-2.

### [ ] ANSWER-4 — `ask` command
**Touch:** `src/cli.py`, tests
**Done when:** `audio2rag ask "..." [filters]` prints the answer with `[title, HH:MM:SS]`
citations and the source audio path. `--json` emits the `Answer` model.
**Out of scope:** the web UI.

---

## Phase 8 — `[ADVANCED]` EVAL: evaluation and UI

### [ ] EVAL-1 — Eval harness
**Why:** without this, every later tuning decision is guesswork — and it's the deliverable that
turns "hallucination-resistant" into a number you can show a client.
**Touch:** `eval/questions.yaml`, `eval/run.py`, docs
**Done when:** ~30 Polish questions with expected doc_id + timestamp windows; the harness reports
recall@5 and MRR; a documented baseline number is committed. Marked `slow`.
**Out of scope:** hitting the ≥ 0.85 target — that is the tuning that follows.

### [ ] EVAL-2 — Tuning pass
**Done when:** chunk size, overlap, candidate counts and threshold are tuned against EVAL-1, with
before/after numbers recorded in the ADR or a `docs/eval-log.md`.

### [ ] EVAL-3 — `serve` (optional, last)
**Done when:** local chat UI with an audio player that seeks to the cited timestamp.
Deliberately last: it is the only part with no effect on retrieval quality — a natural
"nice-to-have" upsell on top of a completed Advanced delivery.

---

## Dependency order

```
INIT-1 ─ INIT-2 ─ INIT-3 ─ CANON-1 ─ CANON-2 ─ CANON-3 ─ CANON-4 ─ CANON-5 ─┬─ CANON-6
                                                                           │
                                                                           └─ CANON-7 ─┐
                                                                                       │
                              INGEST-1 ──────────────────────────────────────────────┤
                              INGEST-2 ─ INGEST-3 ─ INGEST-4                          │
                              INGEST-5 ─ INGEST-6 (fallback, can trail behind) ────────┘
                                                                                       │
                                                          CHUNK-1 ─ CHUNK-2 ───────────┘
                                                             │
                     INDEX-1 ─ INDEX-2 ─ INDEX-3 ─ INDEX-4 ──┘
                        │
                        └─ SEARCH-1 ─ SEARCH-5 ──────────────────────────┐  ◄── Standard ends here
                              │                                          │
                              └─ SEARCH-2 ─┬─ SEARCH-3 ─ SEARCH-4 ─ SEARCH-6 ─┐
                                 (BM25)    ┘                                  │
                                                                               ▼
                                      ANSWER-1..4 ─ EVAL-1 ─ EVAL-2 ─ EVAL-3  ◄── Advanced
```

**CANON-5** is the riskiest task (it rewires the existing output path) and **CHUNK-1** is the
highest-leverage one (chunk quality caps everything downstream). Give those two the most
review attention regardless of which tier a client has bought.
