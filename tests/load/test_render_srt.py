import re
from datetime import timedelta

import srt

from load.render_srt import render_srt, render_srt_from_jsonl
from load.write_canonical import write_canonical
from models.schemas import Segment, TranscriptMeta

DOC_ID = "a3f9c1b2d4e6f801"

_TIMECODE_RE = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3}$")


def make_segment(**overrides):
    fields = {
        "doc_id": DOC_ID,
        "seg": 0,
        "start": 65.0,
        "end": 70.5,
        "speaker": "SPEAKER_00",
        "text": "Zakleszczenie wymaga łącznie czterech warunków Coffmana.",
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
        "title": "Systemy Operacyjne — Wykład 3",
        "duration_s": 5412,
        "language": "pl",
        "model": "large-v3",
        "ingested_at": "2026-03-04T18:22:11Z",
    }
    fields.update(overrides)
    return TranscriptMeta(**fields)


def test_renders_valid_srt_parseable_with_correct_timecode_format(tmp_path):
    segments = [
        make_segment(seg=0, start=65.0, end=70.5, text="Pierwsza wypowiedź."),
        make_segment(seg=1, start=70.5, end=72.0, speaker="SPEAKER_01", text="Druga wypowiedź."),
    ]
    output_path = tmp_path / "out.srt"
    render_srt(segments, output_path)

    raw = output_path.read_text()
    timecode_lines = [line for line in raw.splitlines() if "-->" in line]
    assert len(timecode_lines) == 2
    for line in timecode_lines:
        start_str, end_str = (part.strip() for part in line.split("-->"))
        assert _TIMECODE_RE.match(start_str)
        assert _TIMECODE_RE.match(end_str)

    subtitles = list(srt.parse(raw))
    assert len(subtitles) == 2
    assert subtitles[0].index == 1
    assert subtitles[0].start == timedelta(seconds=65.0)
    assert subtitles[0].end == timedelta(seconds=70.5)
    assert "SPEAKER_00" in subtitles[0].content
    assert "Pierwsza wypowiedź" in subtitles[0].content
    assert subtitles[1].index == 2
    assert "SPEAKER_01" in subtitles[1].content


def test_renders_from_jsonl_via_doc_id(tmp_path):
    segments = [make_segment()]
    write_canonical(segments, make_meta(), tmp_path)

    output_path = render_srt_from_jsonl(DOC_ID, tmp_path)

    assert output_path == tmp_path / f"{DOC_ID}.srt"
    subtitles = list(srt.parse(output_path.read_text()))
    assert len(subtitles) == 1


def test_creates_output_dir_if_missing(tmp_path):
    output_path = tmp_path / "nested" / "output" / "out.srt"
    render_srt([make_segment()], output_path)

    assert output_path.exists()
