"""Naive dense-only answer path (spec.md §7, TASKS.md SMOKE-3). The
refusal-over-hallucination guarantee has two independent layers: a below-threshold
top score refuses before the LLM is ever called (this module's job), and the LLM
itself can still refuse via `prompts.REFUSAL_MARKER` when retrieval clears the
threshold but the excerpts don't actually answer the question — either path
returns the same `Refusal` type, never free text claiming knowledge. Dense-only
until ANSWER-1 swaps `dense_search` for the full hybrid pipeline (BM25 + RRF +
reranker + context expansion); schemas, prompts and `LLMClient` carry over
unchanged."""

from __future__ import annotations

from pathlib import Path

from models.schemas import Answer, Citation, Refusal
from retrieve.llm import LLMClient
from retrieve.prompts import SYSTEM_PROMPT, build_user_prompt, is_refusal
from retrieve.search import dense_search
from transform.embed import Embedder


def _format_timestamp(seconds: float) -> str:
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def answer_question(
    query: str,
    embedder: Embedder,
    store_dir: Path,
    llm: LLMClient,
    titles: dict[str, str],
    *,
    k: int = 30,
    threshold: float = 0.35,
    doc_ids: set[str] | None = None,
) -> Answer | Refusal:
    """Runs `dense_search`; a top score below `threshold` (or no hits at all)
    refuses without ever calling `llm` (assert `FakeLLM.calls == []` for this
    path). Above threshold, `llm` sees every retrieved chunk tagged
    `[title, HH:MM:SS]` and either answers or emits `REFUSAL_MARKER`, which
    becomes a `Refusal` the same way. `titles` maps doc_id to `TranscriptMeta.title`
    for the citation tags — the caller's job to build (e.g. from `list_transcripts`),
    keeping this module free of filesystem access beyond `store_dir`."""
    hits = dense_search(query, k, embedder, store_dir, doc_ids)
    if not hits or hits[0].score < threshold:
        return Refusal(query=query)

    context_blocks = [
        f"[{titles[hit.doc_id]}, {_format_timestamp(hit.start)}]\n{hit.display_text}"
        for hit in hits
    ]
    output = llm.complete(SYSTEM_PROMPT, build_user_prompt(query, context_blocks))
    if is_refusal(output):
        return Refusal(query=query)

    citations = [
        Citation(doc_id=hit.doc_id, title=titles[hit.doc_id], start=hit.start, end=hit.end)
        for hit in hits
    ]
    return Answer(text=output, citations=citations)
