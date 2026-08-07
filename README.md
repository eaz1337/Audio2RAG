# Audio2RAG

Meeting-intelligence CLI: recordings → timestamped transcript → local knowledge base →
grounded Q&A with citations.

Start here, in this order:
- **[CLAUDE.md](CLAUDE.md)** — rules and pipeline layout for anyone (human or agent) working in this repo.
- **[spec.md](spec.md)** — full product spec, tiered Starter / Standard / Advanced.
- **[TASKS.md](TASKS.md)** — the build plan, one task per Claude Code session.

## Status

Bootstrap only (`INIT-1`) — package skeleton and tooling exist, no pipeline logic yet.
Next task: `INIT-2` (ADR: JSONL as canonical source).

## Dev setup

```bash
pip install -e ".[dev]"
cp .env.example .env   # fill in only the keys your config.yaml asr.backend needs
pytest -m "not slow"
ruff check .
```
