from fakes import FakeReranker
from models.schemas import RetrievalHit
from retrieve.rerank import rerank

DOC_ID = "a3f9c1b2d4e6f801"


def hit(chunk_id: int, score: float, display_text: str = "text") -> RetrievalHit:
    return RetrievalHit(
        doc_id=DOC_ID,
        chunk_id=chunk_id,
        score=score,
        start=float(chunk_id),
        end=float(chunk_id) + 1.0,
        segment_ids=[chunk_id],
        display_text=display_text,
    )


def test_rerank_none_returns_hits_unchanged():
    hits = [hit(0, score=0.9), hit(1, score=0.8)]

    result = rerank("query", hits, reranker=None)

    assert result == hits


def test_rerank_with_reranker_reorders_hits():
    hits = [hit(0, score=0.9), hit(1, score=0.8), hit(2, score=0.7)]

    result = rerank("query", hits, reranker=FakeReranker())

    assert [h.chunk_id for h in result] == [2, 1, 0]


def test_rerank_empty_hits_returns_empty():
    assert rerank("query", [], reranker=FakeReranker()) == []


def test_rerank_delegates_query_to_reranker():
    seen: list[str] = []

    class RecordingReranker:
        name = "recording"

        def rerank(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
            seen.append(query)
            return hits

    rerank("what time is it", [hit(0, score=0.5)], reranker=RecordingReranker())

    assert seen == ["what time is it"]
