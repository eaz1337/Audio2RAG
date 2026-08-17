from pathlib import Path

from load.write_canonical import segments_path
from models.schemas import RetrievalHit, Segment
from retrieve.expand import expand_hits

DOC_ID = "a3f9c1b2d4e6f801"
OTHER_DOC_ID = "b7c2e9a1f3d5e802"


def make_segment(**overrides) -> Segment:
    fields = {
        "doc_id": DOC_ID,
        "seg": 0,
        "start": 0.0,
        "end": 1.0,
        "text": "segment text",
        "conf": 0.9,
    }
    fields.update(overrides)
    return Segment(**fields)


def write_segments(output_dir: Path, segments: list[Segment]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    body = "".join(segment.model_dump_json() + "\n" for segment in segments)
    segments_path(segments[0].doc_id, output_dir).write_text(body)


def make_hit(**overrides) -> RetrievalHit:
    fields = {
        "doc_id": DOC_ID,
        "chunk_id": 0,
        "score": 1.0,
        "start": 2.0,
        "end": 3.0,
        "segment_ids": [2],
        "display_text": "chunk two",
    }
    fields.update(overrides)
    return RetrievalHit(**fields)


def five_segments(doc_id: str = DOC_ID) -> list[Segment]:
    return [
        make_segment(doc_id=doc_id, seg=i, start=float(i), end=float(i + 1), text=f"segment {i}")
        for i in range(5)
    ]


def test_expands_with_previous_and_next_segment(tmp_path):
    write_segments(tmp_path, five_segments())
    hit = make_hit(segment_ids=[2], display_text="segment 2")

    [expanded] = expand_hits([hit], tmp_path)

    assert expanded == "segment 1 segment 2 segment 3"


def test_clamps_at_document_start(tmp_path):
    write_segments(tmp_path, five_segments())
    hit = make_hit(segment_ids=[0], display_text="segment 0")

    [expanded] = expand_hits([hit], tmp_path)

    assert expanded == "segment 0 segment 1"


def test_clamps_at_document_end(tmp_path):
    write_segments(tmp_path, five_segments())
    hit = make_hit(segment_ids=[4], display_text="segment 4")

    [expanded] = expand_hits([hit], tmp_path)

    assert expanded == "segment 3 segment 4"


def test_multi_segment_hit_expands_from_its_span(tmp_path):
    write_segments(tmp_path, five_segments())
    hit = make_hit(segment_ids=[1, 2], display_text="segment 1 segment 2")

    [expanded] = expand_hits([hit], tmp_path)

    assert expanded == "segment 0 segment 1 segment 2 segment 3"


def test_adjacent_hits_do_not_duplicate_text(tmp_path):
    write_segments(tmp_path, five_segments())
    first = make_hit(chunk_id=0, segment_ids=[1], display_text="segment 1")
    second = make_hit(chunk_id=1, segment_ids=[2], display_text="segment 2")

    first_expanded, second_expanded = expand_hits([first, second], tmp_path)

    assert first_expanded == "segment 0 segment 1"
    assert second_expanded == "segment 2 segment 3"


def test_empty_hits_returns_empty_list(tmp_path):
    assert expand_hits([], tmp_path) == []


def test_expands_multiple_docs_independently(tmp_path):
    write_segments(tmp_path, five_segments(DOC_ID))
    write_segments(tmp_path, five_segments(OTHER_DOC_ID))
    hit_a = make_hit(doc_id=DOC_ID, segment_ids=[2], display_text="segment 2")
    hit_b = make_hit(doc_id=OTHER_DOC_ID, segment_ids=[2], display_text="segment 2")

    expanded_a, expanded_b = expand_hits([hit_a, hit_b], tmp_path)

    assert expanded_a == "segment 1 segment 2 segment 3"
    assert expanded_b == "segment 1 segment 2 segment 3"
