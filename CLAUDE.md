# CLAUDE.md — project: Audio2RAG

CLI that turns recorded meetings/lectures/training into a queryable knowledge base: audio →
transcription (pluggable ASR backend, spoken content defaults to PL) → canonical timestamped
transcript → optional human-readable renderers (PDF/MD/SRT) → local vector index → question
answering with citations and timestamp-anchored refusal.

This is a meeting-intelligence product, not a transcription tool — see spec.md's Goal section
before assuming a change belongs in `transform/asr/` rather than `retrieve/`.

Built from scratch, from an empty repo — TASKS.md's `INIT` phase is the bootstrap, not a
safety net for a pre-existing codebase. See [spec.md](spec.md) and the ADRs in
`docs/decisions/` (written before the code they govern, not retrospectively):
- [0001-jsonl-as-canonical-source.md](docs/decisions/0001-jsonl-as-canonical-source.md)
- [0002-pluggable-asr-backend.md](docs/decisions/0002-pluggable-asr-backend.md) — why ASR is not
  assumed local/offline, and why that's safe given what stays local

## Commands

```bash
pip install -e ".[dev]"      # core + test/lint tooling
pip install -e ".[dev,rag]"  # add local embedding/reranker for Standard+ work
pytest -m "not slow"         # must be green before and after every task (see TASKS.md)
pytest -m slow                # real ASR API / real model / GPU tests — run deliberately, costs money or time
ruff check .
mypy src
```

## Pipeline layout

```
src/
  models/schemas.py   # Pydantic contracts for every stage boundary
  extract/            # file discovery, format + path validation, hashing
  transform/
    asr/              # ASRBackend protocol + implementations (hosted API default, whisper-local fallback)
    diarize.py         # only used by the whisper-local backend; hosted backends return speakers inline
    chunk.py
    embed.py
  load/               # canonical JSONL write, renderers (pdf/md/srt), vector store write
  retrieve/           # hybrid search, reranking, answer assembly  <- read path, not ETL
```

`extract/ -> transform/ -> load/` is the write path. `retrieve/` is a separate read path and must
never write to the store or mutate artifacts.

## Rules

- **Language:** All code, variable names, docstrings, comments, config keys, CLI flags and metadata
  enum values MUST ALWAYS be written in English. The *spoken* language of recordings is data, not
  code — it lives in `config.yaml` defaults and metadata, never hardcoded outside it.

- **ASR is pluggable, not assumed:** every ASR implementation sits behind the `ASRBackend` protocol
  (`transcribe(path, language) -> list[Segment]`). The default backend is a hosted API
  (AssemblyAI/Deepgram); `whisper-local` is a fallback behind the same interface. Nothing outside
  `transform/asr/` may import a specific backend's SDK, and nothing downstream of `list[Segment]`
  may branch on which backend produced it. Adding a backend means writing one new class, not
  touching `chunk.py`, `embed.py`, `load/`, or `retrieve/`.

- **Not offline-first — retrieval is local-first.** ASR may call an external API; that's a
  deliberate choice (see ADR 0002), not an oversight. What *does* stay local and must never make
  a network call: chunking, embedding, the vector store, reranking, and the retrieval/answer path.
  If you're adding a network call, check first whether it belongs in `transform/asr/` (fine) or in
  `retrieve/`/`load/vector_store.py` (not fine).

- **Canonical source:** `output/<doc_id>.segments.jsonl` is the single source of truth. PDF, MD and
  SRT are renderers that READ it, generated on request, never automatically as a side effect of
  ingest. Nothing in the codebase may parse a PDF back into data — if you need transcript content,
  read the JSONL.

- **Typed boundaries:** Each function has clear, typed (Pydantic) input/output and is testable via
  dependency injection (ASR backend, embedder, reranker and vector store passed as arguments,
  never global imports) — see `src/transform/asr/base.py`.

- **Validation:** Data validation (Pydantic) happens at the pipeline input
  (`src/models/schemas.py` `AudioInput`), not after the fact. `Segment`, `Chunk`, `RetrievalHit`
  and `Answer` are likewise validated at each stage boundary.

- **Citations are structural, not stylistic:** `Answer.citations` is a non-empty list at the type
  level. An answer with no supporting chunk must be an explicit `Refusal` object, never free text
  claiming knowledge. This is the concrete mechanism behind any "hallucination-resistant" claim
  made about the product — it must hold in code, not just in a prompt.

- **Idempotency:** `doc_id = sha256(audio bytes)[:16]`. Re-running with the same input overwrites
  the output artifacts and does delete-then-insert in the vector store (`DELETE WHERE doc_id`, then
  `INSERT`) — never per-chunk upsert, since chunk count changes with model/parameters. Two runs =
  one set of artifacts and an unchanged chunk count.

- **Transcription cache:** If `.segments.jsonl` exists and its recorded hash matches the input, the
  ASR backend does not run again — regardless of which backend is configured, since hosted-API
  calls cost money per run just as GPU time does locally. Re-rendering and re-indexing must be
  possible without re-transcribing.

- **Model pinning:** Embedding model name and vector dimension are written into the collection
  metadata. On mismatch, querying raises and instructs the user to run `reindex`— it must never
  silently query a stale index.

- **Notebooks:** Notebooks are for exploration only — production logic ALWAYS belongs in `src/`.

- **Test fixtures:** Small, real files in `tests/fixtures/` (e.g. a 3-second `.wav`, a 20-line
  `segments.jsonl`, a 5-chunk toy corpus), not path mocks.

- **Testing:** ASR (any backend)/GPU/embedding/reranker tests are marked `slow` and skipped by
  default (`pytest.ini_options` in `pyproject.toml`). Transformation tests use a `FakeASRBackend`,
  never call a real API or download/run real Whisper. Retrieval tests use a deterministic fake
  embedder (stable vectors from a text hash) so ranking assertions are reproducible without a real
  model. No non-`slow` test touches the network — this includes hosted ASR APIs, not just local
  GPU models.

- **Prompts:** All LLM prompts live in `src/retrieve/prompts.py`, versioned and unit-tested for
  their refusal behaviour. No inline prompt strings scattered across modules.

- **Dependencies:** `torch`/CUDA deps are optional (`pip install -e ".[cuda]"`, only relevant to the
  `whisper-local` backend and to local embedding/reranker inference) and the code must work with a
  CPU fallback without them. Diarization for the `whisper-local` backend is likewise optional
  (`pip install -e ".[diarize]"`); when `pyannote` or its HF token is absent, ingest proceeds
  without speaker labels and warns, rather than failing. Hosted backends never need this extra —
  they return speaker labels inline.

- **Secrets:** ASR API keys (`ASSEMBLYAI_API_KEY`, `DEEPGRAM_API_KEY`) and `HF_TOKEN` (for
  `whisper-local` diarization only) come from the environment (see `.env.example`). Never
  committed, never logged, never written into `meta.json`.
