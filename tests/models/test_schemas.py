from datetime import date, datetime

import pytest
from pydantic import ValidationError

from models.schemas import Segment, TranscriptMeta, TranscriptType

DOC_ID = "a3f9c1b2d4e6f801"


def make_segment(**overrides):
    fields = {
        "doc_id": DOC_ID,
        "seg": 142,
        "start": 852.4,
        "end": 867.1,
        "speaker": "SPEAKER_00",
        "text": "Deadlock occurs when all four Coffman conditions hold.",
        "conf": -0.21,
    }
    fields.update(overrides)
    return Segment(**fields)


def make_meta(**overrides):
    fields = {
        "doc_id": DOC_ID,
        "source_path": "recordings/os-lecture-03.mp3",
        "sha256": DOC_ID + "0" * 48,
        "type": "lecture",
        "title": "Operating Systems — Lecture 3",
        "course": "Operating Systems",
        "speakers": {"SPEAKER_00": "Dr. Kowalski"},
        "date": "2026-03-04",
        "duration_s": 5412,
        "language": "pl",
        "model": "large-v3",
        "tags": ["synchronization", "deadlock"],
        "ingested_at": "2026-03-04T18:22:11Z",
    }
    fields.update(overrides)
    return TranscriptMeta(**fields)


class TestSegment:
    def test_round_trip(self):
        segment = make_segment()
        restored = Segment.model_validate_json(segment.model_dump_json())
        assert restored == segment

    def test_speaker_optional(self):
        segment = make_segment(speaker=None)
        assert segment.speaker is None

    def test_rejects_end_not_after_start(self):
        with pytest.raises(ValidationError):
            make_segment(start=10.0, end=10.0)
        with pytest.raises(ValidationError):
            make_segment(start=10.0, end=5.0)

    def test_rejects_negative_seg(self):
        with pytest.raises(ValidationError):
            make_segment(seg=-1)

    def test_rejects_empty_text(self):
        with pytest.raises(ValidationError):
            make_segment(text="")
        with pytest.raises(ValidationError):
            make_segment(text="   ")

    def test_rejects_malformed_doc_id(self):
        with pytest.raises(ValidationError):
            make_segment(doc_id="not-hex!!")


class TestTranscriptMeta:
    def test_round_trip(self):
        meta = make_meta()
        restored = TranscriptMeta.model_validate_json(meta.model_dump_json())
        assert restored == meta
        assert restored.type is TranscriptType.LECTURE
        assert restored.date == date(2026, 3, 4)
        assert restored.ingested_at == datetime.fromisoformat("2026-03-04T18:22:11+00:00")

    def test_optional_fields_default(self):
        meta = make_meta(course=None, speakers={}, date=None, tags=[])
        assert meta.course is None
        assert meta.speakers == {}
        assert meta.date is None
        assert meta.tags == []

    def test_rejects_invalid_type(self):
        with pytest.raises(ValidationError):
            make_meta(type="podcast")

    def test_rejects_empty_title(self):
        with pytest.raises(ValidationError):
            make_meta(title="")

    def test_rejects_negative_duration(self):
        with pytest.raises(ValidationError):
            make_meta(duration_s=-1)

    def test_rejects_malformed_doc_id(self):
        with pytest.raises(ValidationError):
            make_meta(doc_id="short")
