import json

from load.write_canonical import (
    meta_path,
    read_meta,
    read_segments,
    segments_path,
    write_canonical,
)
from models.schemas import Segment, TranscriptMeta

DOC_ID = "a3f9c1b2d4e6f801"


def make_segment(**overrides):
    fields = {
        "doc_id": DOC_ID,
        "seg": 0,
        "start": 0.0,
        "end": 4.2,
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


def test_writes_one_json_line_per_segment(tmp_path):
    segments = [make_segment(seg=i, start=float(i), end=float(i) + 1.0) for i in range(3)]
    write_canonical(segments, make_meta(), tmp_path)

    lines = segments_path(DOC_ID, tmp_path).read_text().splitlines()
    assert len(lines) == 3
    for line in lines:
        json.loads(line)  # each line is a standalone JSON object


def test_round_trip_is_lossless(tmp_path):
    segments = [make_segment(seg=i, start=float(i), end=float(i) + 1.0) for i in range(5)]
    meta = make_meta()
    write_canonical(segments, meta, tmp_path)

    assert read_segments(DOC_ID, tmp_path) == segments
    assert read_meta(DOC_ID, tmp_path) == meta


def test_rerun_overwrites_not_appends(tmp_path):
    segments = [make_segment(seg=i, start=float(i), end=float(i) + 1.0) for i in range(4)]
    meta = make_meta()

    write_canonical(segments, meta, tmp_path)
    write_canonical(segments, meta, tmp_path)

    lines = segments_path(DOC_ID, tmp_path).read_text().splitlines()
    assert len(lines) == 4


def test_creates_output_dir_if_missing(tmp_path):
    output_dir = tmp_path / "nested" / "output"
    write_canonical([make_segment()], make_meta(), output_dir)

    assert meta_path(DOC_ID, output_dir).exists()
