from datetime import date

import pytest

from models.schemas import Segment
from transform.chunk import chunk_segments

DOC_ID = "a3f9c1b2d4e6f801"
TITLE = "Operating Systems — Lecture 3"


def make_segment(**overrides):
    fields = {
        "doc_id": DOC_ID,
        "seg": 0,
        "start": 0.0,
        "end": 1.0,
        "text": "Hello world.",
        "speaker": None,
        "conf": -0.1,
    }
    fields.update(overrides)
    return Segment(**fields)


def words(n: int, prefix: str = "word") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n)) + "."


class TestChunkSegments:
    def test_single_short_segment(self):
        segment = make_segment(seg=0, start=0.0, end=2.0, text="Hello world.")

        chunks = chunk_segments([segment], TITLE)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.doc_id == DOC_ID
        assert chunk.chunk_id == 0
        assert chunk.start == 0.0
        assert chunk.end == 2.0
        assert chunk.segment_ids == [0]
        assert chunk.display_text == "Hello world."
        assert chunk.embed_text == f"[{TITLE}, 00:00:00–00:00:02]\nHello world."

    def test_long_monologue_splits_by_token_target_and_overlaps(self):
        # 10 segments of ~80 tokens each, back-to-back, same speaker, no pauses:
        # nothing but the token target drives the split.
        segments = [
            make_segment(seg=i, start=float(i * 10), end=float(i * 10 + 9), text=words(80))
            for i in range(10)
        ]

        chunks = chunk_segments(segments, TITLE, target_tokens=500, overlap_ratio=0.15)

        assert len(chunks) > 1
        for chunk in chunks:
            # never splits a segment
            assert chunk.segment_ids == sorted(chunk.segment_ids)
            assert all(0 <= s < len(segments) for s in chunk.segment_ids)
        # consecutive chunks overlap by at least one shared segment id
        for a, b in zip(chunks, chunks[1:]):
            assert set(a.segment_ids) & set(b.segment_ids)
        # every segment is covered by some chunk
        covered = {s for chunk in chunks for s in chunk.segment_ids}
        assert covered == set(range(len(segments)))

    def test_rapid_speaker_alternation_splits_on_speaker_change(self):
        segments = [
            make_segment(
                seg=i,
                start=float(i * 2),
                end=float(i * 2 + 1),
                text=words(60),
                speaker="SPEAKER_00" if i % 2 == 0 else "SPEAKER_01",
            )
            for i in range(20)
        ]

        chunks = chunk_segments(segments, TITLE, target_tokens=500)

        assert len(chunks) > 1
        covered = {s for chunk in chunks for s in chunk.segment_ids}
        assert covered == set(range(len(segments)))

    def test_never_splits_an_oversized_segment(self):
        oversized = make_segment(seg=0, start=0.0, end=120.0, text=words(700))
        trailing = make_segment(seg=1, start=121.0, end=125.0, text=words(50))

        chunks = chunk_segments([oversized, trailing], TITLE, target_tokens=500)

        assert chunks[0].segment_ids == [0]
        assert 1 in {s for chunk in chunks for s in chunk.segment_ids}

    def test_chunk_never_splits_a_single_segment_in_half(self):
        segments = [
            make_segment(seg=i, start=float(i * 10), end=float(i * 10 + 9), text=words(50))
            for i in range(6)
        ]

        chunks = chunk_segments(segments, TITLE, target_tokens=500)

        seen_ids = [s for chunk in chunks for s in chunk.segment_ids]
        for seg_id in range(len(segments)):
            assert seen_ids.count(seg_id) >= 1

    def test_rejects_empty_segments(self):
        with pytest.raises(ValueError):
            chunk_segments([], TITLE)

    def test_rejects_mixed_doc_ids(self):
        a = make_segment(seg=0, doc_id=DOC_ID)
        b = make_segment(seg=1, doc_id="b" * 16)

        with pytest.raises(ValueError):
            chunk_segments([a, b], TITLE)


class TestContextualHeader:
    def test_header_with_date_and_speaker(self):
        segment = make_segment(
            seg=0, start=852.0, end=867.0, text="Deadlock text.", speaker="SPEAKER_00"
        )

        chunks = chunk_segments([segment], TITLE, date=date(2026, 3, 4))

        assert chunks[0].embed_text == (
            f"[{TITLE}, 2026-03-04, SPEAKER_00, 00:14:12–00:14:27]\nDeadlock text."
        )
        assert chunks[0].display_text == "Deadlock text."

    def test_header_omits_absent_date_and_speaker(self):
        segment = make_segment(seg=0, start=0.0, end=5.0, text="No metadata.", speaker=None)

        chunks = chunk_segments([segment], TITLE)

        assert chunks[0].embed_text == f"[{TITLE}, 00:00:00–00:00:05]\nNo metadata."

    def test_header_joins_multiple_speakers_in_one_chunk(self):
        segments = [
            make_segment(seg=0, start=0.0, end=1.0, text="Hi.", speaker="SPEAKER_00"),
            make_segment(seg=1, start=1.0, end=2.0, text="there.", speaker="SPEAKER_01"),
        ]

        chunks = chunk_segments(segments, TITLE, target_tokens=500)

        assert len(chunks) == 1
        assert "SPEAKER_00/SPEAKER_01" in chunks[0].embed_text

    def test_display_text_never_carries_the_header(self):
        segments = [
            make_segment(seg=i, start=float(i * 10), end=float(i * 10 + 9), text=words(80))
            for i in range(10)
        ]

        chunks = chunk_segments(segments, TITLE, date=date(2026, 3, 4), target_tokens=500)

        for chunk in chunks:
            assert not chunk.display_text.startswith("[")
            assert chunk.embed_text.endswith(chunk.display_text)
