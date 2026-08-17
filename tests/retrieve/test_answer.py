from fakes import FakeEmbedder, FakeLLM

from load.vector_store import write_chunks
from models.schemas import Answer, Chunk, Refusal
from retrieve.answer import answer_question
from retrieve.prompts import REFUSAL_MARKER, SYSTEM_PROMPT

DOC_ID = "a3f9c1b2d4e6f801"


def make_chunk(**overrides):
    fields = {
        "doc_id": DOC_ID,
        "chunk_id": 0,
        "start": 12.0,
        "end": 16.2,
        "segment_ids": [0, 1],
        "display_text": "Deadlock occurs when all four Coffman conditions hold.",
        "embed_text": "[Lecture 3, 00:00:12-00:00:16]\nDeadlock occurs when all four Coffman conditions hold.",
    }
    fields.update(overrides)
    return Chunk(**fields)


TITLES = {DOC_ID: "Lecture 3"}


def test_refuses_without_calling_llm_when_top_score_is_below_threshold(tmp_path):
    embedder = FakeEmbedder(dim=8)
    chunk = make_chunk()
    write_chunks([chunk], embedder, tmp_path)
    llm = FakeLLM(response="should never be seen")

    result = answer_question(
        chunk.embed_text, embedder, tmp_path, llm, TITLES, threshold=2.0
    )

    assert isinstance(result, Refusal)
    assert result.query == chunk.embed_text
    assert llm.calls == []


def test_refuses_without_calling_llm_when_store_has_no_chunks(tmp_path):
    llm = FakeLLM()

    result = answer_question("deadlock", FakeEmbedder(dim=8), tmp_path, llm, TITLES)

    assert isinstance(result, Refusal)
    assert llm.calls == []


def test_returns_an_answer_when_top_score_clears_the_threshold(tmp_path):
    embedder = FakeEmbedder(dim=8)
    chunk = make_chunk()
    write_chunks([chunk], embedder, tmp_path)
    llm = FakeLLM(response="Deadlock needs all four Coffman conditions [Lecture 3, 00:00:12].")

    result = answer_question(
        chunk.embed_text, embedder, tmp_path, llm, TITLES, threshold=0.35
    )

    assert isinstance(result, Answer)
    assert result.text == llm._response
    assert len(llm.calls) == 1


def test_answer_citations_carry_the_retrieved_chunk_fields(tmp_path):
    embedder = FakeEmbedder(dim=8)
    chunk = make_chunk()
    write_chunks([chunk], embedder, tmp_path)
    llm = FakeLLM(response="answer text")

    result = answer_question(chunk.embed_text, embedder, tmp_path, llm, TITLES, threshold=0.35)

    assert isinstance(result, Answer)
    [citation] = result.citations
    assert citation.doc_id == DOC_ID
    assert citation.title == "Lecture 3"
    assert citation.start == chunk.start
    assert citation.end == chunk.end


def test_llm_prompt_carries_the_title_and_timestamp_tag(tmp_path):
    embedder = FakeEmbedder(dim=8)
    chunk = make_chunk()
    write_chunks([chunk], embedder, tmp_path)
    llm = FakeLLM(response="answer text")

    answer_question(chunk.embed_text, embedder, tmp_path, llm, TITLES, threshold=0.35)

    [(system, user)] = llm.calls
    assert system == SYSTEM_PROMPT
    assert "[Lecture 3, 00:00:12]" in user
    assert chunk.display_text in user


def test_refuses_when_llm_declares_the_excerpts_insufficient(tmp_path):
    embedder = FakeEmbedder(dim=8)
    chunk = make_chunk()
    write_chunks([chunk], embedder, tmp_path)
    llm = FakeLLM(response=REFUSAL_MARKER)

    result = answer_question(chunk.embed_text, embedder, tmp_path, llm, TITLES, threshold=0.35)

    assert isinstance(result, Refusal)
    assert result.query == chunk.embed_text
    assert len(llm.calls) == 1
