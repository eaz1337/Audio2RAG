import pytest
from fakes import FakeEmbedder

from load.vector_store import EmbeddingModelMismatch, write_chunks
from models.schemas import Chunk, RetrievalHit
from retrieve.search import dense_search, sparse_search

DOC_ID = "a3f9c1b2d4e6f801"


def make_chunk(**overrides):
    fields = {
        "doc_id": DOC_ID,
        "chunk_id": 0,
        "start": 0.0,
        "end": 4.2,
        "segment_ids": [0, 1],
        "display_text": "Deadlock occurs when all four Coffman conditions hold.",
        "embed_text": "[Lecture 3, 00:00:00–00:00:04]\nDeadlock occurs when all four Coffman conditions hold.",
    }
    fields.update(overrides)
    return Chunk(**fields)


def toy_corpus():
    return [
        make_chunk(
            chunk_id=0,
            segment_ids=[0, 1],
            display_text="Deadlock occurs when all four Coffman conditions hold.",
            embed_text="[Lecture 3, 00:00:00-00:00:04]\nDeadlock occurs when all four Coffman conditions hold.",
        ),
        make_chunk(
            chunk_id=1,
            segment_ids=[2, 3],
            display_text="A semaphore is an integer variable guarded by atomic operations.",
            embed_text="[Lecture 3, 00:00:04-00:00:09]\nA semaphore is an integer variable guarded by atomic operations.",
        ),
        make_chunk(
            chunk_id=2,
            segment_ids=[4, 5],
            display_text="Round-robin scheduling assigns each process a fixed time quantum.",
            embed_text="[Lecture 3, 00:00:09-00:00:14]\nRound-robin scheduling assigns each process a fixed time quantum.",
        ),
    ]


def test_dense_search_returns_empty_list_before_any_write(tmp_path):
    assert dense_search("deadlock", k=5, embedder=FakeEmbedder(dim=8), store_dir=tmp_path) == []


def test_dense_search_finds_exact_match_first(tmp_path):
    embedder = FakeEmbedder(dim=8)
    chunks = toy_corpus()
    write_chunks(chunks, embedder, tmp_path)

    hits = dense_search(chunks[1].embed_text, k=3, embedder=embedder, store_dir=tmp_path)

    assert hits[0].chunk_id == 1
    assert hits[0].display_text == chunks[1].display_text
    assert hits[0].score == max(hit.score for hit in hits)


def test_dense_search_returns_retrieval_hits_with_chunk_fields(tmp_path):
    embedder = FakeEmbedder(dim=8)
    chunk = make_chunk()
    write_chunks([chunk], embedder, tmp_path)

    [hit] = dense_search(chunk.embed_text, k=1, embedder=embedder, store_dir=tmp_path)

    assert isinstance(hit, RetrievalHit)
    assert hit.doc_id == chunk.doc_id
    assert hit.chunk_id == chunk.chunk_id
    assert hit.start == chunk.start
    assert hit.end == chunk.end
    assert hit.segment_ids == chunk.segment_ids
    assert hit.display_text == chunk.display_text


def test_dense_search_respects_k(tmp_path):
    embedder = FakeEmbedder(dim=8)
    write_chunks(toy_corpus(), embedder, tmp_path)

    hits = dense_search("scheduling", k=2, embedder=embedder, store_dir=tmp_path)

    assert len(hits) == 2


def test_dense_search_orders_by_score_descending(tmp_path):
    embedder = FakeEmbedder(dim=8)
    write_chunks(toy_corpus(), embedder, tmp_path)

    hits = dense_search("semaphore", k=3, embedder=embedder, store_dir=tmp_path)

    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_dense_search_raises_on_embedder_mismatch(tmp_path):
    write_chunks(toy_corpus(), FakeEmbedder(dim=8), tmp_path)

    with pytest.raises(EmbeddingModelMismatch):
        dense_search("deadlock", k=3, embedder=FakeEmbedder(dim=16), store_dir=tmp_path)


def test_sparse_search_returns_empty_list_before_any_write(tmp_path):
    assert sparse_search("deadlock", k=5, store_dir=tmp_path) == []


def test_sparse_search_returns_retrieval_hits_with_chunk_fields(tmp_path):
    embedder = FakeEmbedder(dim=8)
    chunk = make_chunk()
    write_chunks([chunk], embedder, tmp_path)

    [hit] = sparse_search("Coffman", k=1, store_dir=tmp_path)

    assert isinstance(hit, RetrievalHit)
    assert hit.doc_id == chunk.doc_id
    assert hit.chunk_id == chunk.chunk_id
    assert hit.start == chunk.start
    assert hit.end == chunk.end
    assert hit.segment_ids == chunk.segment_ids
    assert hit.display_text == chunk.display_text


def test_sparse_search_respects_k(tmp_path):
    embedder = FakeEmbedder(dim=8)
    write_chunks(toy_corpus(), embedder, tmp_path)

    hits = sparse_search("scheduling", k=2, store_dir=tmp_path)

    assert len(hits) == 2


def test_sparse_search_orders_by_score_descending(tmp_path):
    embedder = FakeEmbedder(dim=8)
    write_chunks(toy_corpus(), embedder, tmp_path)

    hits = sparse_search("semaphore atomic", k=3, store_dir=tmp_path)

    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_sparse_search_finds_rare_term_the_fake_dense_path_misses(tmp_path):
    """The fake embedder's vectors are unrelated to text content, so a query on an exact,
    rare technical term can rank a chunk that doesn't contain it above the one that does
    (TASKS.md SEARCH-2). BM25 has no such blind spot: it ranks the one chunk containing
    the term first, with every other candidate scoring zero."""
    embedder = FakeEmbedder(dim=8)
    chunks = toy_corpus()
    chunks[2] = make_chunk(
        chunk_id=2,
        segment_ids=[4, 5],
        display_text=(
            "Round-robin scheduling assigns each process a fixed time quantum, using "
            "cgroups for isolation."
        ),
        embed_text="[Lecture 3, 00:00:09-00:00:14]\ncgroups",
    )
    write_chunks(chunks, embedder, tmp_path)

    dense_hits = dense_search("cgroups", k=3, embedder=embedder, store_dir=tmp_path)
    sparse_hits = sparse_search("cgroups", k=3, store_dir=tmp_path)

    assert dense_hits[0].chunk_id != 2
    assert sparse_hits[0].chunk_id == 2
    assert sparse_hits[0].score > 0
    assert sparse_hits[1].score == 0
    assert sparse_hits[2].score == 0
