"""All prompt strings for the answer path (spec.md §7, TASKS.md SMOKE-2) —
CLAUDE.md "Prompts": versioned and unit-tested for refusal behaviour, no inline
prompt strings anywhere else. `SMOKE-3`'s threshold check keeps the LLM from being
called at all when retrieval is weak; this module covers the other refusal path —
the LLM itself declaring the excerpts insufficient, signalled by `REFUSAL_MARKER`
so `answer.py` can turn it into a `Refusal` without parsing free text."""

from __future__ import annotations

PROMPT_VERSION = "v1"

REFUSAL_MARKER = "NOT_IN_RECORDINGS"

SYSTEM_PROMPT = f"""You answer questions about recorded meetings and lectures using only the \
excerpts given to you. Never use outside knowledge and never guess.

Every claim in your answer must be followed by the excerpt's citation tag, exactly as given, in \
the form [title, HH:MM:SS].

If the excerpts do not contain the answer to the question, respond with exactly this and nothing \
else: {REFUSAL_MARKER}"""


def build_user_prompt(query: str, context_blocks: list[str]) -> str:
    """`context_blocks` are pre-formatted excerpts, each already carrying its
    `[title, HH:MM:SS]` citation tag (spec.md §7) — assembling those tags from
    `RetrievalHit`s and transcript metadata is the answer path's job, not this
    module's."""
    excerpts = "\n\n".join(context_blocks)
    return f"Excerpts:\n{excerpts}\n\nQuestion: {query}"


def is_refusal(llm_output: str) -> bool:
    """True when the LLM judged the excerpts insufficient (see `SYSTEM_PROMPT`)."""
    return llm_output.strip() == REFUSAL_MARKER
