from fakes import FakeLLM

from retrieve.llm import LLMClient, OllamaLLM


class _NotAnLLM:
    pass


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    """Stands in for `requests.Session` — records calls, never touches the network."""

    def __init__(self, reply: str) -> None:
        self.posts: list[tuple[str, dict]] = []
        self._reply = reply

    def post(self, url: str, json: dict) -> _FakeResponse:
        self.posts.append((url, json))
        return _FakeResponse({"message": {"content": self._reply}})


def test_fake_llm_satisfies_the_protocol():
    assert isinstance(FakeLLM(), LLMClient)


def test_unrelated_object_does_not_satisfy_the_protocol():
    assert not isinstance(_NotAnLLM(), LLMClient)


def test_ollama_llm_satisfies_the_protocol():
    assert isinstance(OllamaLLM(session=_FakeSession("hi")), LLMClient)


def test_ollama_llm_posts_system_and_user_messages():
    session = _FakeSession("The answer is 42.")
    llm = OllamaLLM(model="bielik-11b-v2.3-instruct", session=session)

    result = llm.complete("system text", "user text")

    assert result == "The answer is 42."
    [(url, payload)] = session.posts
    assert url == "http://localhost:11434/api/chat"
    assert payload["model"] == "bielik-11b-v2.3-instruct"
    assert payload["messages"] == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "user text"},
    ]


def test_ollama_llm_strips_trailing_slash_from_base_url():
    session = _FakeSession("ok")
    llm = OllamaLLM(base_url="http://localhost:11434/", session=session)

    llm.complete("s", "u")

    [(url, _)] = session.posts
    assert url == "http://localhost:11434/api/chat"
