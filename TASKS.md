# TASKS.md — Audio2RAG build plan

Incremental plan, one task per Claude Code session. Each phase ends where `pytest -m "not slow"` is
green and the tool works end to end. Mirrors spec.md's tiers: **Starter = phases 1–3, Standard =
phases 4–6 (dense-only), Advanced = phases 6 (full)–8.**

## Phases at a glance

| # | Phase | Prefix | Tasks | Tier | What you have when it's done |
|---|---|---|---|---|---|
| 1 | Bootstrap | `INIT` | 3 | — (prerequisite) | Repo scaffold + the two ADRs the rest of the plan implements |
| 2 | Canonical transcript | `CANON` | 7 | **Starter** | Timestamped JSONL as source of truth; PDF/MD/SRT rendered from it |
| 3 | Ingest & CLI | `INGEST` | 6 | **Starter** | Real CLI, metadata, batch ingest, pluggable ASR (hosted default) |
| 4 | Chunking | `CHUNK` | 2 | **Standard** | Speech split on pauses and speaker turns, not blindly |
| 5 | Index | `INDEX` | 4 | **Standard** | Embeddings in a local vector store, idempotent |
| 6 | Search | `SEARCH` | 6 | **Standard** (dense+filters) / **Advanced** (hybrid+rerank+expansion) | Retrieval, from "find the chunk" to "grounded answer" |
| 6.5 | Smoke test | `SMOKE` | 5 | — (internal validation) | A real `ask` on real audio, dense-only, before building the hybrid stack |
| 7 | Answer | `ANSWER` | 1 | **Advanced** | Hybrid retrieval behind the same `ask` command |
| 8 | Eval | `EVAL` | 3 | **Advanced** | Measured quality, tuned threshold, optional UI |

Task IDs are `PREFIX-n` (e.g. `CANON-5`). Each task is tagged `[STARTER]` / `[STANDARD]` /
`[ADVANCED]` so a scope-of-work for a tier can be extracted by grepping the tag. `SMOKE-*` tasks
are tagged `[VALIDATION]` and deliberately excluded from that grep.

## How to work through this

- **One task per session.** Fresh Claude Code session per task; paste the task block as the prompt.
  Do not batch tasks — the point of the split is a small diff and a small context.
- **Tests:** run the tests for the module you touched at the end of a task
  (`pytest -q --tb=short -x tests/<module>`). Run the full `pytest -q -m "not slow"` at the end of a
  phase, not after every task.
- **No commits.** Claude Code does not run git. Tick the checkbox in this file; the user commits.
- **Respect "Out of scope".** It exists to stop scope creep mid-task, which is the main way an agent
  gets lost in a build this size.
- If a task needs more than ~300 changed lines, stop and split it.

Phases 1–3 leave a strictly better transcription tool (timestamps, MD/SRT, caching, real CLI,
pluggable ASR) and no RAG — the Starter deliverable, and a valid place to pause and ship.

---

## Phase 1 — INIT: repo bootstrap

Starting from an empty repo: no existing pipeline, no legacy behaviour to protect. This phase
produces a scaffold and the two ADRs the rest of the plan implements.

### [x] INIT-1 — Repo scaffold and tooling
**Touch:** `pyproject.toml`, `config.yaml`, `.env.example`, `.gitignore`, empty `src/` package tree
matching CLAUDE.md's pipeline layout (`models/`, `extract/`, `transform/asr/`, `load/`, `retrieve/`,
`cli.py`), empty `tests/` tree with `pytest` configured (`slow` marker registered,
`pythonpath = ["src"]`).
**Done when:** `pytest --collect-only -q` exits 0 (note: `pytest` itself exits 5 on an empty suite —
that is expected here and not a failure), `ruff check --select F .` is clean, `pip install -e
".[dev]"` succeeds.
**Out of scope:** any pipeline logic, any ADR content.

### [x] INIT-2 — ADR: JSONL as canonical source
**Touch:** `docs/decisions/0001-jsonl-as-canonical-source.md`
**Done when:** ADR records the decision "JSONL is the source of truth, PDF is a renderer", the
alternatives considered (index PDFs directly / index plain `.txt`), and the consequences.
**Out of scope:** any code.

### [x] INIT-3 — ADR: pluggable ASR backend, not offline-first
**Touch:** `docs/decisions/0002-pluggable-asr-backend.md`
**Done when:** ADR records that offline/on-prem is not a hard requirement, names the default
(hosted API) and fallback (`whisper-local`) backends, states the `ASRBackend` protocol as the
enforced boundary, and notes what stays local regardless (chunking, embedding, vector store,
reranking, retrieval).
**Out of scope:** any code — but CANON-2 and INGEST-1 implement this ADR, read it first.

---

## Phase 2 — `[STARTER]` CANON: canonical transcript format

### [x] CANON-1 — Segment and metadata schemas
**Touch:** `src/models/schemas.py`, `tests/models/test_schemas.py`
**Done when:** `Segment` (doc_id, seg, start, end, text, speaker|None, conf) and `TranscriptMeta`
(per spec §2, including `asr_backend`, `asr_model`, `diarized`) exist as Pydantic models with
validation (`end > start`, `seg >= 0`, non-empty text, `conf` is `None` or within `0.0–1.0`) and
round-trip tests.
**Out of scope:** using them anywhere yet.

### [x] CANON-2 — `ASRBackend` protocol and fake
**Why:** get the interface right before any concrete backend exists, so hosted and local are
equally easy to add (ADR 0002).
**Touch:** `src/transform/asr/base.py` (protocol), `tests/fakes.py` (`FakeASRBackend`),
`src/transform/transcribe.py` (thin wrapper calling whichever backend is configured)
**Done when:** `ASRBackend.transcribe(path: Path, language: str) -> list[Segment]` is a `Protocol`;
`FakeASRBackend` returns a fixed, injectable segment list. The wrapper takes a backend as a
constructor argument, never imports a concrete backend at module level.
**Out of scope:** any real backend implementation — that's INGEST-1.

### [x] CANON-3 — doc_id and file hashing
**Touch:** `src/extract/`, tests
**Done when:** `compute_doc_id(path) -> str` returns `sha256(bytes)[:16]`, is stable across runs and
differs for different content. Same content under a different filename yields the same id.
**Out of scope:** using doc_id for output naming yet.

### [x] CANON-4 — Write the canonical artifacts
**Touch:** `src/load/write_canonical.py`, tests
**Done when:** `output/<doc_id>.segments.jsonl` (one JSON object per line) and
`output/<doc_id>.meta.json` are written and read back into the schemas losslessly. Re-running
overwrites, never appends — assert line count is unchanged after two runs.
**Out of scope:** the PDF renderer.

### [x] CANON-5 — PDF renderer over JSONL
**Why:** highest-risk task in the phase — it fixes the output path everything downstream assumes.
**Touch:** `src/load/render_pdf.py`, tests
**Done when:** the renderer takes `list[Segment]` (or reads the JSONL) and emits timestamps. No code
path outside `load/` produces a PDF. PDF is generated only when explicitly requested
(`--render pdf`), never as a side effect of ingest.
**Out of scope:** MD/SRT renderers.

### [x] CANON-6 — Markdown and SRT renderers
**Touch:** `src/load/render_md.py`, `src/load/render_srt.py`, tests
**Done when:** both read the same `list[Segment]` and produce valid output, both opt-in via
`--render`; SRT timecodes are `HH:MM:SS,mmm` and validated by a parser in the test.
**Out of scope:** deciding which renderers run by default — none do, by design.

### [x] CANON-7 — Transcription cache
**Touch:** `src/transform/`, CLI entry point, tests
**Done when:** the ASR backend is skipped only if `<doc_id>.segments.jsonl` exists **and**
`meta.json` records a matching `sha256` **and** a matching `asr_backend` + `asr_model`. Switching
backend or model invalidates the cache and re-transcribes. Assert the fake backend's call count is
zero on a cache hit and one after a backend switch. `--force` bypasses the cache.
**Out of scope:** embedding-model mismatch — that's INDEX-4's guard, a different concern.

---

## Phase 3 — `[STARTER]` INGEST: ingest, metadata, CLI

### [x] INGEST-1 — Hosted ASR backend (default)
**Touch:** `src/transform/asr/hosted.py`, `config.yaml`, tests (`FakeASRBackend` / mocked HTTP)
**Done when:** a hosted backend (AssemblyAI or Deepgram nova-3) implements `ASRBackend`, returns
segments with speaker labels from the API's diarization and `conf` normalised to `0.0–1.0`, and is
selected via `asr.backend: assemblyai | deepgram`. API key read from the environment
(`ASSEMBLYAI_API_KEY` / `DEEPGRAM_API_KEY`), never logged, never written to `meta.json`.
`meta.json` records `asr_backend` and `asr_model`. No non-`slow` test makes a real HTTP call.
**Out of scope:** the local fallback — that's INGEST-5.

### [x] INGEST-2 — `ingest` subcommand with metadata flags
**Touch:** `src/cli.py`, tests
**Done when:** `audio2rag ingest <path> --type --title --course --date --speaker --tag --render`
works, values land in `meta.json`, `type` is validated against the enum
(`lecture|training|meeting|interview|other`), and `--render pdf,md,srt` controls which renderers run
(none by default).
**Out of scope:** `ask`, `serve`.

### [ ] INGEST-3 — Directory ingest
**Touch:** `src/extract/discover.py`, CLI, tests
**Done when:** `ingest <dir> --recursive` finds all supported extensions, skips unsupported files
with a warning, and continues past a single file's failure instead of aborting the batch.
**Out of scope:** parallelism.

### [ ] INGEST-4 — `list`, `rm`, `relabel`
**Touch:** CLI, `src/load/`, tests
**Done when:** `list` prints doc_id/title/type/duration from meta files; `rm <doc_id>` removes all
artifacts for that id; `relabel <doc_id> --speaker ...` edits `meta.json` only and never re-runs
transcription.
**Out of scope:** vector store cleanup in `rm` — added in INDEX-3.

### [ ] INGEST-5 — `whisper-local` fallback backend
**Touch:** `src/transform/asr/whisper_local.py`, `pyproject.toml` (`[cuda]` extra), tests
**Done when:** implements `ASRBackend` with `vad_filter=True`, `word_timestamps=True` and
`condition_on_previous_text=False` for inputs over 30 minutes, selectable via
`asr.backend: whisper-local`, converting `avg_logprob` to a `0.0–1.0` `conf`. CPU fallback works
without CUDA. Never imported unless selected (lazy import).
**Out of scope:** diarization — that's INGEST-6.

### [ ] INGEST-6 — Diarization for the local fallback
**Why:** the hosted default already returns speaker labels; this only matters for `whisper-local`.
**Touch:** `src/transform/diarize.py` (pyannote wrapper — the file CLAUDE.md's layout reserves for
this), a hook in `src/transform/asr/whisper_local.py`, `pyproject.toml` (`[diarize]` extra), tests
**Done when:** with `--diarize` and `pyannote` present, `whisper-local` segments carry speaker
labels and `meta.json` records `diarized: true`; with the extra or `HF_TOKEN` missing, ingest
completes without labels, sets `diarized: false` and logs a warning — it does not crash. Import of
`pyannote` is lazy and inside the function.
**Out of scope:** mapping `SPEAKER_00` to real names (that is `relabel`).

---

## Phase 4 — `[STANDARD]` CHUNK: chunking

### [ ] CHUNK-1 — Chunk schema and chunker
**Why:** highest-leverage task in the plan — chunk quality caps everything downstream. Pure function
over data, no models needed, so be strict here.
**Touch:** `src/models/schemas.py` (`Chunk`), `src/transform/chunk.py`, tests
**Done when:** the chunker groups segments into 400–600 token chunks with ~15% overlap, splits
preferentially at pauses > 1.5 s / sentence ends / speaker changes, never splits a segment, and each
chunk carries `start`, `end`, `doc_id`, `segment_ids`. Tests cover: a single short segment, one long
monologue, rapid speaker alternation, and a segment longer than the target size.
**Out of scope:** embedding.

### [ ] CHUNK-2 — Contextual header
**Touch:** `src/transform/chunk.py`, tests
**Done when:** `Chunk.embed_text` is the header (`[title, date, speaker, HH:MM:SS–HH:MM:SS]`, the
spec §7 citation format) plus the body, while `Chunk.display_text` stays clean for citation display.
Both asserted separately.
**Out of scope:** proving it improves retrieval — EVAL-1.

---

## Phase 5 — `[STANDARD]` INDEX: embeddings and vector store

### [ ] INDEX-1 — Embedder protocol and fake
**Touch:** `src/transform/embed.py`, `tests/fakes.py`
**Done when:** an `Embedder` Protocol (`embed(texts) -> ndarray`, `name`, `dim`) exists, plus a
`FakeEmbedder` producing deterministic vectors from a text hash. The real `bge-m3` implementation is
behind the `[rag]` extra with a lazy import and marked `slow`. Dense output only (spec §4). This
runs locally regardless of ASR backend — see ADR 0002.
**Out of scope:** the store.

### [ ] INDEX-2 — LanceDB writer
**Touch:** `src/load/vector_store.py`, tests (`FakeEmbedder`, temp dir)
**Done when:** chunks are written to `store/lance/` with all metadata fields queryable; collection
metadata records embedding model name and dim. Tests download nothing and make no network call.
**Out of scope:** search.

### [ ] INDEX-3 — Delete-then-insert idempotency
**Touch:** `src/load/vector_store.py`, `rm` command, tests
**Done when:** ingesting the same file twice leaves the chunk count unchanged; ingesting after a
chunk-size config change replaces cleanly with no orphans; `rm <doc_id>` removes chunks too.
**Out of scope:** the BM25 index (SEARCH-2) — wire its deletion into this path there.

### [ ] INDEX-4 — Model mismatch guard and `reindex`
**Touch:** `src/load/vector_store.py`, CLI, tests
**Done when:** querying a store built with a different embedder raises a clear error naming both
models and telling the user to run `reindex`. `reindex` rebuilds from existing JSONL without
re-transcribing (assert the ASR fake is never called), and rebuilds **every** index that exists for
the corpus — after SEARCH-2 lands, that includes BM25, so an upgraded Standard install gets a BM25
index without re-ingesting.
**Out of scope:** retrieval quality.

---

## Phase 6 — SEARCH: retrieval (`[STANDARD]` dense+filters, `[ADVANCED]` hybrid+rerank+expansion)

### [ ] SEARCH-1 — `[STANDARD]` Dense search
**Touch:** `src/retrieve/search.py`, tests with `FakeEmbedder` and a toy corpus
**Done when:** `dense_search(query, k)` returns an ordered `list[RetrievalHit]` with scores. This
alone is the Standard deliverable: smart search, no LLM call.
**Out of scope:** fusion, reranking, LLM.

### [ ] SEARCH-2 — `[ADVANCED]` BM25 index and search
**Touch:** `src/load/bm25_index.py`, `src/retrieve/search.py`, tests
**Done when:** a `bm25s` index is built at `store/bm25/` during ingest **and** during `reindex`
(INDEX-4), deleted alongside vectors in INDEX-3's path, and `sparse_search(query, k)` returns
comparable hits. A test shows BM25 finding an exact rare term the fake dense path misses.
**Out of scope:** fusion.

### [ ] SEARCH-3 — `[ADVANCED]` RRF fusion
**Touch:** `src/retrieve/fusion.py`, tests
**Done when:** `rrf(dense_hits, sparse_hits, k=60)` is a pure function with hand-computed expected
rankings in the tests, including disjoint result sets and full overlap.
**Out of scope:** reranking.

### [ ] SEARCH-4 — `[ADVANCED]` Reranker behind a protocol
**Touch:** `src/retrieve/rerank.py`, `tests/fakes.py`
**Done when:** a `Reranker` Protocol plus `FakeReranker` (deterministic, e.g. reverses order) make
the wiring testable; the real `bge-reranker-v2-m3` sits behind the `[rag]` extra, marked `slow`.
Reranking is skippable via `--no-rerank`.
**Out of scope:** tuning candidate counts.

### [ ] SEARCH-5 — `[STANDARD]` Metadata filters
**Touch:** `src/retrieve/search.py`, CLI, tests
**Done when:** `--course --type --after --before --speaker --tag` restrict results; a test with two
courses in the corpus proves no cross-contamination. Filters apply *before* reranking and work
identically whether reranking is enabled or not.
**Out of scope:** natural-language filter parsing.

### [ ] SEARCH-6 — `[ADVANCED]` Context expansion
**Touch:** `src/retrieve/expand.py`, tests
**Done when:** each hit is expanded with ±1 adjacent segment read from the JSONL, correctly clamped
at document start/end, with no duplicated text when two hits are adjacent.
**Out of scope:** answering.

---

## Phase 6.5 — `[VALIDATION]` SMOKE: early answer path

**Why this phase exists:** SEARCH-2..4,6 and EVAL-1 are expensive to build and easy to over-invest
in before you know the basic idea works on your own recordings. This phase pulls a minimal `ask`
path forward, wired to **dense-only search (SEARCH-1 + SEARCH-5)** — no BM25, no fusion, no
reranker, no expansion. What's built here is superseded by ANSWER-1, not shipped to an Advanced
client.

**Depends only on** SEARCH-1 and SEARCH-5. Run it before SEARCH-2/3/4/6 to get a working `ask`
sooner.

### [ ] SMOKE-1 — Answer and Citation schemas
**Why:** encoding "no answer without a source" in the type system is more reliable than prompting.
**Touch:** `src/models/schemas.py`, tests
**Done when:** `Citation` (doc_id, title, start, end) and `Answer` with a non-empty citations list;
`Refusal` is a separate type. Constructing an `Answer` with zero citations raises.
**Out of scope:** the LLM call.

### [ ] SMOKE-2 — Prompts module and LLM protocol
**Touch:** `src/retrieve/prompts.py`, `src/retrieve/llm.py`, `tests/fakes.py`
**Done when:** all prompt strings live in `prompts.py`, are versioned and unit-tested for refusal
behaviour; an `LLMClient` Protocol plus `FakeLLM` allow full answer-path tests with no model. No
inline prompt strings elsewhere.
**Out of scope:** choosing the production LLM — pick something that runs today to unblock this
phase. The real choice (spec.md "Open decisions" #1) is revisited once EVAL-1 gives data.

### [ ] SMOKE-3 — Naive threshold and refusal path, dense-only
**Touch:** `src/retrieve/answer.py`, tests
**Done when:** given only `dense_search` results, a top score below the configured threshold returns
`Refusal` without calling the LLM (assert `FakeLLM` was not invoked). The threshold is a throwaway
starting guess — it exists so the refusal *mechanism* is real from day one.
**Out of scope:** tuning the threshold — EVAL-2.

### [ ] SMOKE-4 — `ask` command, dense-only
**Touch:** `src/cli.py`, tests
**Done when:** `audio2rag ask "..." [filters]` works end to end against a fixture store, printing an
answer or a refusal with `[title, HH:MM:SS]` citations and the source audio path. `--json` emits the
`Answer` model.
**Out of scope:** hybrid retrieval, the web UI.

### [ ] SMOKE-5 — Your own reference questions
**Why:** a handful of real questions about recordings you care about is worth more right now than a
30-question harness you can't act on for another five tasks. This set grows into EVAL-1's, it isn't
thrown away.
**Touch:** `eval/smoke_questions.yaml` (5–10 entries: question, doc_id, expected timestamp window),
plus a one-off script (exploration only, not `src/`) that runs each through `ask` and prints the
result next to the expected answer.
**Done when:** you've run these on real audio and formed an opinion — "close enough to harden" or
"chunking/retrieval is off, Advanced work is premature" — before spending time on SEARCH-2..4,6.
**Out of scope:** recall@5 / MRR scoring — that's EVAL-1.

---

## Phase 7 — `[ADVANCED]` ANSWER: harden the answer path

Schemas, prompts, the `LLMClient` protocol and CLI wiring from Phase 6.5 carry over unchanged. This
phase's only job is to swap dense-only retrieval for the full pipeline. The refusal threshold stays
at SMOKE-3's placeholder until EVAL-2 sets it from measured data — that ordering is deliberate, so
this task does not depend on an eval harness that does not exist yet.

### [ ] ANSWER-1 — Wire hybrid retrieval into `ask`
**Touch:** `src/retrieve/answer.py`, `src/cli.py`, tests
**Done when:** `ask` calls the full SEARCH-2..4,6 pipeline (BM25 + RRF + reranker + context
expansion) instead of SMOKE-3's dense-only path; `--no-rerank` still works and falls back toward
SMOKE-3's behaviour so before/after is directly comparable. Threshold value unchanged from SMOKE-3.
**Out of scope:** setting the threshold from data (EVAL-2), the web UI (EVAL-3).

---

## Phase 8 — `[ADVANCED]` EVAL: evaluation and UI

### [ ] EVAL-1 — Eval harness
**Why:** without this, every tuning decision is guesswork — and it's what turns
"hallucination-resistant" into a number.
**Touch:** `eval/questions.yaml`, `eval/run.py`, docs
**Done when:** ~30 Polish questions (grown from SMOKE-5's set) with expected doc_id + timestamp
windows; the harness reports recall@5 and MRR against the ANSWER-1 pipeline; a baseline number is
committed. Marked `slow`.
**Out of scope:** hitting the ≥ 0.85 target — that is EVAL-2.

### [ ] EVAL-2 — Tuning pass and refusal threshold
**Touch:** `config.yaml`, `docs/eval-log.md`
**Done when:** chunk size, overlap, candidate counts and the refusal threshold are tuned against
EVAL-1, with before/after numbers recorded. This is where SMOKE-3's placeholder threshold is
replaced by a measured one; a test asserts an out-of-corpus question still refuses at the new value.

### [ ] EVAL-3 — `serve` (optional, last)
**Done when:** a local chat UI with an audio player that seeks to the cited timestamp. Last on
purpose: the only part with no effect on retrieval quality.

---

## Dependency order

```
INIT-1 → INIT-2 → INIT-3
INIT-3 → CANON-1 → CANON-2 → CANON-3 → CANON-4 → CANON-5 → CANON-6
CANON-4 → CANON-7
CANON-2 + CANON-7 → INGEST-1 → INGEST-2 → INGEST-3 → INGEST-4
INGEST-1 → INGEST-5 → INGEST-6          (local fallback, can trail behind)
CANON-4 → CHUNK-1 → CHUNK-2
CHUNK-2 → INDEX-1 → INDEX-2 → INDEX-3 → INDEX-4
INDEX-2 → SEARCH-1 → SEARCH-5           ◄── Standard tier ends here
SEARCH-5 → SMOKE-1 → SMOKE-2 → SMOKE-3 → SMOKE-4 → SMOKE-5   [VALIDATION]
INDEX-3 → SEARCH-2 → SEARCH-3 → SEARCH-4 → SEARCH-6
SMOKE-4 + SEARCH-6 → ANSWER-1 → EVAL-1 → EVAL-2 → EVAL-3     ◄── Advanced
```

The `SMOKE` chain and the `SEARCH-2..6` chain are independent — run SMOKE first for early signal.
`ANSWER-1` is the only task needing both.

**CHUNK-1** is the highest-leverage task (chunk quality caps everything downstream) and **CANON-5**
the riskiest early one. **SMOKE-5** is where you decide, with evidence, whether SEARCH-2..4,6 is
worth building yet.