from datetime import UTC, datetime
from pathlib import Path

from load.manage import list_transcripts, relabel_speakers, remove_transcript
from load.write_canonical import write_canonical
from models.schemas import Segment, TranscriptMeta, TranscriptType


def make_meta(doc_id: str, **overrides) -> TranscriptMeta:
    fields = {
        "doc_id": doc_id,
        "source_path": "/tmp/audio.wav",
        "sha256": "a" * 64,
        "type": TranscriptType.LECTURE,
        "title": "Lecture 1",
        "course": None,
        "speakers": {},
        "date": None,
        "duration_s": 12.0,
        "language": "pl",
        "model": "fake",
        "tags": [],
        "ingested_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    fields.update(overrides)
    return TranscriptMeta(**fields)


def make_segment(doc_id: str, **overrides) -> Segment:
    fields = {
        "doc_id": doc_id,
        "seg": 0,
        "start": 0.0,
        "end": 1.5,
        "text": "Hello world.",
        "speaker": "SPEAKER_00",
        "conf": 0.9,
    }
    fields.update(overrides)
    return Segment(**fields)


def write_fixture(output_dir: Path, doc_id: str, **meta_overrides) -> None:
    write_canonical([make_segment(doc_id)], make_meta(doc_id, **meta_overrides), output_dir)


class TestListTranscripts:
    def test_empty_directory_returns_no_transcripts(self, tmp_path):
        assert list_transcripts(tmp_path) == []

    def test_lists_every_ingested_transcript(self, tmp_path):
        write_fixture(tmp_path, "0000000000000001", title="Lecture 1")
        write_fixture(tmp_path, "0000000000000002", title="Lecture 2")

        metas = list_transcripts(tmp_path)

        assert {meta.doc_id for meta in metas} == {"0000000000000001", "0000000000000002"}
        assert {meta.title for meta in metas} == {"Lecture 1", "Lecture 2"}


class TestRemoveTranscript:
    def test_removes_segments_and_meta(self, tmp_path):
        write_fixture(tmp_path, "0000000000000001")

        remove_transcript("0000000000000001", tmp_path)

        assert list(tmp_path.glob("0000000000000001.*")) == []

    def test_removes_rendered_files_too(self, tmp_path):
        write_fixture(tmp_path, "0000000000000001")
        (tmp_path / "0000000000000001.pdf").write_bytes(b"fake-pdf")
        (tmp_path / "0000000000000001.md").write_text("fake-md")
        (tmp_path / "0000000000000001.srt").write_text("fake-srt")

        remove_transcript("0000000000000001", tmp_path)

        assert list(tmp_path.glob("0000000000000001.*")) == []

    def test_missing_doc_id_does_not_raise(self, tmp_path):
        remove_transcript("0000000000000099", tmp_path)

    def test_does_not_affect_other_doc_ids(self, tmp_path):
        write_fixture(tmp_path, "0000000000000001")
        write_fixture(tmp_path, "0000000000000002")

        remove_transcript("0000000000000001", tmp_path)

        assert list_transcripts(tmp_path) == [
            TranscriptMeta.model_validate_json(
                (tmp_path / "0000000000000002.meta.json").read_text()
            )
        ]


class TestRelabelSpeakers:
    def test_updates_speakers_dict_only(self, tmp_path):
        write_fixture(tmp_path, "0000000000000001", speakers={"SPEAKER_00": "Unknown"})

        updated = relabel_speakers(
            "0000000000000001", tmp_path, {"SPEAKER_00": "Dr. Kowalski"}
        )

        assert updated.speakers == {"SPEAKER_00": "Dr. Kowalski"}

    def test_merges_with_existing_labels(self, tmp_path):
        write_fixture(
            tmp_path,
            "0000000000000001",
            speakers={"SPEAKER_00": "Dr. Kowalski"},
        )

        updated = relabel_speakers(
            "0000000000000001", tmp_path, {"SPEAKER_01": "Ms. Nowak"}
        )

        assert updated.speakers == {
            "SPEAKER_00": "Dr. Kowalski",
            "SPEAKER_01": "Ms. Nowak",
        }

    def test_does_not_touch_segments_jsonl(self, tmp_path):
        write_fixture(tmp_path, "0000000000000001")
        segments_before = (tmp_path / "0000000000000001.segments.jsonl").read_text()

        relabel_speakers("0000000000000001", tmp_path, {"SPEAKER_00": "Dr. Kowalski"})

        segments_after = (tmp_path / "0000000000000001.segments.jsonl").read_text()
        assert segments_before == segments_after
