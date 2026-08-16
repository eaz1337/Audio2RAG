import pytest

from models.schemas import RetrievalHit
from retrieve.fusion import rrf

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


def test_rrf_empty_lists_returns_empty():
    assert rrf([], []) == []


def test_rrf_dense_only_scores_by_dense_rank_alone():
    dense_hits = [hit(0, score=0.9), hit(1, score=0.8)]

    fused = rrf(dense_hits, [], k=60)

    assert [f.chunk_id for f in fused] == [0, 1]
    assert fused[0].score == pytest.approx(1 / 61)
    assert fused[1].score == pytest.approx(1 / 62)


def test_rrf_full_overlap_reorders_by_combined_rank():
    """Every chunk appears in both lists but at different ranks. Hand-computed with
    k=60: chunk_id=1 (dense rank 2, sparse rank 1) has the highest combined score even
    though it isn't first in either individual list."""
    dense_hits = [hit(0, score=0.9), hit(1, score=0.8), hit(2, score=0.7)]
    sparse_hits = [hit(1, score=5.0), hit(2, score=4.0), hit(0, score=3.0)]

    fused = rrf(dense_hits, sparse_hits, k=60)

    expected = {
        0: 1 / 61 + 1 / 63,
        1: 1 / 62 + 1 / 61,
        2: 1 / 63 + 1 / 62,
    }
    assert [f.chunk_id for f in fused] == [1, 0, 2]
    for f in fused:
        assert f.score == pytest.approx(expected[f.chunk_id])


def test_rrf_disjoint_result_sets_keeps_all_hits():
    """Dense and sparse share no chunks. Both rank-1 hits tie (and both rank-2 hits tie);
    ties keep dense-before-sparse, then original-list order."""
    dense_hits = [hit(0, score=0.9), hit(1, score=0.8)]
    sparse_hits = [hit(2, score=5.0), hit(3, score=4.0)]

    fused = rrf(dense_hits, sparse_hits, k=60)

    assert [f.chunk_id for f in fused] == [0, 2, 1, 3]
    assert fused[0].score == pytest.approx(1 / 61)
    assert fused[1].score == pytest.approx(1 / 61)
    assert fused[2].score == pytest.approx(1 / 62)
    assert fused[3].score == pytest.approx(1 / 62)


def test_rrf_replaces_score_with_fused_value_not_original():
    dense_hits = [hit(0, score=0.9)]
    sparse_hits = [hit(0, score=123.0)]

    [fused] = rrf(dense_hits, sparse_hits, k=60)

    assert fused.score not in (0.9, 123.0)
    assert fused.score == pytest.approx(1 / 61 + 1 / 61)


def test_rrf_preserves_dense_hit_fields_on_overlap():
    dense_hits = [hit(0, score=0.9, display_text="from dense")]
    sparse_hits = [hit(0, score=5.0, display_text="from sparse")]

    [fused] = rrf(dense_hits, sparse_hits, k=60)

    assert fused.display_text == "from dense"
    assert fused.start == dense_hits[0].start
    assert fused.segment_ids == dense_hits[0].segment_ids


def test_rrf_default_k_is_60():
    dense_hits = [hit(0, score=0.9), hit(1, score=0.8)]

    assert rrf(dense_hits, []) == rrf(dense_hits, [], k=60)
