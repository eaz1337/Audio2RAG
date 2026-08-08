from load.render_md import render_md, render_md_from_jsonl
from load.write_canonical import write_canonical
from models.schemas import Segment, TranscriptMeta

DOC_ID = "a3f9c1b2d4e6f801"


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


def test_renders_markdown_with_timestamps_and_polish_glyphs(tmp_path):
    output_path = tmp_path / "out.md"
    render_md([make_segment()], make_meta(), output_path)

    text = output_path.read_text()
    assert "[00:01:05-00:01:10]" in text
    assert "SPEAKER_00" in text
    assert "łącznie czterech warunków" in text
    assert "Systemy Operacyjne" in text


def test_renders_from_jsonl_via_doc_id(tmp_path):
    segments = [make_segment()]
    meta = make_meta()
    write_canonical(segments, meta, tmp_path)

    output_path = render_md_from_jsonl(DOC_ID, tmp_path)

    assert output_path == tmp_path / f"{DOC_ID}.md"
    assert output_path.exists()


def test_creates_output_dir_if_missing(tmp_path):
    output_path = tmp_path / "nested" / "output" / "out.md"
    render_md([make_segment()], make_meta(), output_path)

    assert output_path.exists()
