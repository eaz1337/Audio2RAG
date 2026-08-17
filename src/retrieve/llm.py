"""LLMClient protocol and the Ollama-backed default implementation (spec.md §7,
TASKS.md SMOKE-2). CLAUDE.md "Retrieval is local-first": unlike ASR, the answer
path must never call a hosted API, so the default here talks to a local Ollama
server instead — the production model choice is still open (spec.md "Open
decisions" #1), revisited once EVAL-1 gives data."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import requests


@runtime_checkable
class LLMClient(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class OllamaLLM:
    """Calls a local Ollama server's chat endpoint. Never imported unless
    selected — construct it only when `answer.llm` names it."""

    def __init__(
        self,
        model: str = "bielik-11b-v2.3-instruct",
        base_url: str = "http://localhost:11434",
        session: requests.Session | None = None,
    ) -> None:
        self.name = model
        self._base_url = base_url.rstrip("/")
        self._session = session or requests.Session()

    def complete(self, system: str, user: str) -> str:
        response = self._session.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self.name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
            },
        )
        response.raise_for_status()
        content: str = response.json()["message"]["content"]
        return content
