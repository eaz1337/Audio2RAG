# CLAUDE.md — project: Audio2RAG

CLI that turns recorded meetings/lectures into a queryable knowledge base: audio → ASR (pluggable
backend, spoken content defaults to PL) → canonical timestamped JSONL → optional renderers
(PDF/MD/SRT) → local vector index → question answering with citations and timestamp-anchored
refusal. This is a meeting-intelligence product, not a transcription tool.

Built from scratch from an empty repo — TASKS.md `INIT` is the bootstrap, not a safety net for
existing code. Rationale lives in [spec.md](spec.md) and the ADRs, not here:
[0001-jsonl-as-canonical-source](docs/decisions/0001-jsonl-as-canonical-source.md),
[0002-pluggable-asr-backend](docs/decisions/0002-pluggable-asr-backend.md).

## Commands

```bash
pip install -e ".[dev]"                    # core + test/lint tooling
pip install -e ".[dev,rag]"                # + local embedding/reranker
pytest -q --tb=short -x tests/<module>     # DEFAULT after a task: only what you touched
pytest -q --tb=short -m "not slow"         # end of a TASKS.md phase only
pytest -m slow                             # real ASR API / GPU — deliberate, costs money
ruff check --select F .                    # dead code / undefined names — once per phase
ruff check . / ruff format / mypy src      # ON REQUEST ONLY — never on your own initiative
```

## Workflow

- **Git:** never run `git add`, `git commit`, `git push`, never create branches or tags.
  Committing is the user's job; don't propose commit messages unless asked.
- **Formatting:** the user does one formatting pass at the end of the project. Don't run
  `ruff format` or full `ruff check`. `ruff check --select F .` once per phase is fine — that's
  error detection, not style.
- **Output:** no end-of-task summaries, no progress reports, no new `.md` files unless asked.
  Keep prose short; effort goes into the code.
- **Reading:** targeted line ranges over whole files; never re-read a file already in context.

## Pipeline layout

```
src/
  models/schemas.py   # Pydantic contracts for every stage boundary
  extract/            # file discovery, format + path validation, hashing
  transform/
    asr/              # ASRBackend protocol + implementations (hosted API default, whisper-local fallback)
    diarize.py        # whisper-local backend only; hosted backends return speakers inline
    chunk.py
    embed.py
  load/               # canonical JSONL write, renderers (pdf/md/srt), vector store write
  retrieve/           # hybrid search, reranking, answer assembly  <- read path, not ETL
```

`extract/ -> transform/ -> load/` is the write path. `retrieve/` is a separate read path and must
never write to the store or mutate artifacts.

## Rules

- **Language:** all code, names, docstrings, comments, config keys, CLI flags and enum values are
  English. The *spoken* language of recordings is data — it lives in `config.yaml` and metadata.

- **ASR is pluggable:** every backend sits behind the `ASRBackend` protocol
  (`transcribe(path, language) -> list[Segment]`, see `src/transform/asr/base.py`). Nothing outside
  `transform/asr/` imports a backend SDK; nothing downstream of `list[Segment]` branches on which
  backend produced it. A new backend = one new class, no changes to `chunk.py`, `embed.py`,
  `load/`, `retrieve/`.

- **Retrieval is local-first.** ASR may call an external API (ADR 0002, deliberate). Chunking,
  embedding, the vector store, reranking and the retrieval/answer path must never make a network
  call. New network call → belongs in `transform/asr/`, nowhere else.

- **Canonical source:** `output/<doc_id>.segments.jsonl` is the single source of truth. PDF/MD/SRT
  are renderers that read it, generated on request, never as a side effect of ingest. Nothing may
  parse a PDF back into data.

- **Typed boundaries:** typed Pydantic in/out per function; ASR backend, embedder, reranker and
  vector store are passed as arguments, never global imports.

- **Validation:** at the pipeline input (`AudioInput` in `src/models/schemas.py`), and at each
  stage boundary (`Segment`, `Chunk`, `RetrievalHit`, `Answer`).

- **Citations are structural:** `Answer.citations` is non-empty at the type level. No supporting
  chunk → an explicit `Refusal` object, never free text claiming knowledge. This must hold in
  code, not in a prompt.

- **Idempotency:** `doc_id = sha256(audio bytes)[:16]`. Re-running overwrites artifacts and does
  `DELETE WHERE doc_id` then `INSERT` — never per-chunk upsert. Two runs = one set of artifacts,
  unchanged chunk count.

- **Transcription cache:** if `.segments.jsonl` exists and its recorded hash matches the input, no
  backend runs again. Re-rendering and re-indexing must work without re-transcribing.

- **Model pinning:** embedding model name and vector dimension live in collection metadata. On
  mismatch, querying raises and tells the user to run `reindex` — never silently query a stale index.

- **Notebooks:** exploration only. Production logic always in `src/`.

- **Testing:** small real files in `tests/fixtures/` (3-second `.wav`, 20-line `segments.jsonl`,
  5-chunk corpus), not path mocks. ASR/GPU/embedding/reranker tests are marked `slow` and skipped
  by default. Transformation tests use `FakeASRBackend`; retrieval tests use a deterministic fake
  embedder (stable vectors from a text hash). No non-`slow` test touches the network — hosted ASR
  APIs included.

- **Prompts:** all in `src/retrieve/prompts.py`, versioned and unit-tested for refusal behaviour.
  No inline prompt strings elsewhere.

- **Dependencies:** `torch`/CUDA is optional (`.[cuda]`, only for `whisper-local` and local
  inference); code must work on CPU without it. Diarization is optional (`.[diarize]`) — when
  `pyannote` or the HF token is missing, ingest proceeds without speaker labels and warns.

- **Secrets:** `ASSEMBLYAI_API_KEY`, `DEEPGRAM_API_KEY`, `HF_TOKEN` come from the environment (see
  `.env.example`). Never committed, never logged, never written into `meta.json`.