from pathlib import Path

import pytest
from typer.testing import CliRunner

from fakes import FakeASRBackend

import cli
from load.write_canonical import read_meta
from models.schemas import Segment, TranscriptType

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_A = FIXTURES / "sample_a.wav"

runner = CliRunner()


def make_segment(**overrides) -> Segment:
    fields = {
        "doc_id": "0000000000000000",
        "seg": 0,
        "start": 0.0,
        "end": 1.5,
        "text": "Hello world.",
        "speaker": None,
        "conf": 0.9,
    }
    fields.update(overrides)
    return Segment(**fields)


FAKE_CONFIG = {
    "asr": {"backend": "fake", "language": "pl"},
    "ingest": {"render_default": []},
}


class TestIngestFile:
    def test_metadata_lands_in_meta_json(self, tmp_path):
        backend = FakeASRBackend([make_segment()])

        result_path = cli.ingest_file(
            SAMPLE_A,
            backend=backend,
            language="pl",
            output_dir=tmp_path,
            doc_type=TranscriptType.LECTURE,
            title="Lecture 3",
            course="Operating Systems",
            date=None,
            speakers=[],
            tags=["deadlock", "sync"],
            render_targets=[],
            asr_model_name="fake",
        )

        assert result_path.exists()
        doc_id = result_path.stem.removesuffix(".segments")
        meta = read_meta(doc_id, tmp_path)
        assert meta.type == TranscriptType.LECTURE
        assert meta.title == "Lecture 3"
        assert meta.course == "Operating Systems"
        assert meta.tags == ["deadlock", "sync"]

    def test_title_defaults_to_filename_stem(self, tmp_path):
        backend = FakeASRBackend([make_segment()])

        result_path = cli.ingest_file(
            SAMPLE_A,
            backend=backend,
            language="pl",
            output_dir=tmp_path,
            doc_type=TranscriptType.OTHER,
            title=None,
            course=None,
            date=None,
            speakers=[],
            tags=[],
            render_targets=[],
            asr_model_name="fake",
        )

        doc_id = result_path.stem.removesuffix(".segments")
        meta = read_meta(doc_id, tmp_path)
        assert meta.title == SAMPLE_A.stem

    def test_speaker_names_map_onto_diarized_labels_in_order(self, tmp_path):
        backend = FakeASRBackend(
            [
                make_segment(seg=0, speaker="A"),
                make_segment(seg=1, start=1.5, end=3.0, speaker="B"),
            ]
        )

        result_path = cli.ingest_file(
            SAMPLE_A,
            backend=backend,
            language="pl",
            output_dir=tmp_path,
            doc_type=TranscriptType.MEETING,
            title=None,
            course=None,
            date=None,
            speakers=["Dr. Kowalski", "Ms. Nowak"],
            tags=[],
            render_targets=[],
            asr_model_name="fake",
        )

        doc_id = result_path.stem.removesuffix(".segments")
        meta = read_meta(doc_id, tmp_path)
        assert meta.speakers == {"A": "Dr. Kowalski", "B": "Ms. Nowak"}

    def test_diarized_true_when_segments_carry_speaker_labels(self, tmp_path):
        backend = FakeASRBackend([make_segment(speaker="SPEAKER_00")])

        result_path = cli.ingest_file(
            SAMPLE_A,
            backend=backend,
            language="pl",
            output_dir=tmp_path,
            doc_type=TranscriptType.OTHER,
            title=None,
            course=None,
            date=None,
            speakers=[],
            tags=[],
            render_targets=[],
            asr_model_name="fake",
        )

        doc_id = result_path.stem.removesuffix(".segments")
        assert read_meta(doc_id, tmp_path).diarized is True

    def test_diarized_false_when_no_segment_has_a_speaker(self, tmp_path):
        backend = FakeASRBackend([make_segment(speaker=None)])

        result_path = cli.ingest_file(
            SAMPLE_A,
            backend=backend,
            language="pl",
            output_dir=tmp_path,
            doc_type=TranscriptType.OTHER,
            title=None,
            course=None,
            date=None,
            speakers=[],
            tags=[],
            render_targets=[],
            asr_model_name="fake",
        )

        doc_id = result_path.stem.removesuffix(".segments")
        assert read_meta(doc_id, tmp_path).diarized is False

    def test_duration_is_last_segment_end(self, tmp_path):
        backend = FakeASRBackend(
            [
                make_segment(seg=0, start=0.0, end=1.5),
                make_segment(seg=1, start=1.5, end=4.2),
            ]
        )

        result_path = cli.ingest_file(
            SAMPLE_A,
            backend=backend,
            language="pl",
            output_dir=tmp_path,
            doc_type=TranscriptType.OTHER,
            title=None,
            course=None,
            date=None,
            speakers=[],
            tags=[],
            render_targets=[],
            asr_model_name="fake",
        )

        doc_id = result_path.stem.removesuffix(".segments")
        meta = read_meta(doc_id, tmp_path)
        assert meta.duration_s == 4.2

    def test_no_render_targets_produces_no_renderer_files(self, tmp_path):
        backend = FakeASRBackend([make_segment()])

        cli.ingest_file(
            SAMPLE_A,
            backend=backend,
            language="pl",
            output_dir=tmp_path,
            doc_type=TranscriptType.OTHER,
            title=None,
            course=None,
            date=None,
            speakers=[],
            tags=[],
            render_targets=[],
            asr_model_name="fake",
        )

        assert list(tmp_path.glob("*.pdf")) == []
        assert list(tmp_path.glob("*.md")) == []
        assert list(tmp_path.glob("*.srt")) == []

    def test_render_targets_produce_requested_files_only(self, tmp_path):
        backend = FakeASRBackend([make_segment()])

        cli.ingest_file(
            SAMPLE_A,
            backend=backend,
            language="pl",
            output_dir=tmp_path,
            doc_type=TranscriptType.OTHER,
            title=None,
            course=None,
            date=None,
            speakers=[],
            tags=[],
            render_targets=["pdf", "md"],
            asr_model_name="fake",
        )

        assert len(list(tmp_path.glob("*.pdf"))) == 1
        assert len(list(tmp_path.glob("*.md"))) == 1
        assert list(tmp_path.glob("*.srt")) == []

    def test_second_ingest_hits_cache_unless_forced(self, tmp_path):
        backend = FakeASRBackend([make_segment()])

        for _ in range(2):
            cli.ingest_file(
                SAMPLE_A,
                backend=backend,
                language="pl",
                output_dir=tmp_path,
                doc_type=TranscriptType.OTHER,
                title=None,
                course=None,
                date=None,
                speakers=[],
                tags=[],
                render_targets=[],
                asr_model_name="fake",
            )
        assert len(backend.calls) == 1

        cli.ingest_file(
            SAMPLE_A,
            backend=backend,
            language="pl",
            output_dir=tmp_path,
            doc_type=TranscriptType.OTHER,
            title=None,
            course=None,
            date=None,
            speakers=[],
            tags=[],
            render_targets=[],
            asr_model_name="fake",
            force=True,
        )
        assert len(backend.calls) == 2


class TestParseRenderTargets:
    def test_empty_string_uses_default(self):
        assert cli._parse_render_targets("", ["pdf"]) == ["pdf"]

    def test_comma_separated_list_is_split_and_stripped(self):
        assert cli._parse_render_targets("pdf, md , srt", []) == ["pdf", "md", "srt"]

    def test_unknown_renderer_raises(self):
        with pytest.raises(Exception, match="unknown renderer"):
            cli._parse_render_targets("pdf,docx", [])


class TestIngestCommand:
    def test_full_flag_set_lands_in_meta_json(self, tmp_path, monkeypatch):
        backend = FakeASRBackend([make_segment(speaker="A")])
        monkeypatch.setattr(cli, "load_config", lambda: FAKE_CONFIG)
        monkeypatch.setattr(cli, "build_asr_backend", lambda config, **kwargs: backend)
        monkeypatch.setattr(cli, "OUTPUT_DIR", tmp_path)

        result = runner.invoke(
            cli.app,
            [
                "ingest",
                str(SAMPLE_A),
                "--type",
                "meeting",
                "--title",
                "Weekly sync",
                "--course",
                "Ops",
                "--date",
                "2026-03-04",
                "--speaker",
                "Dr. Kowalski",
                "--tag",
                "budget",
                "--tag",
                "q3",
                "--render",
                "pdf,md",
            ],
        )

        assert result.exit_code == 0, result.output
        meta_files = list(tmp_path.glob("*.meta.json"))
        assert len(meta_files) == 1
        doc_id = meta_files[0].stem.removesuffix(".meta")
        meta = read_meta(doc_id, tmp_path)
        assert meta.type == TranscriptType.MEETING
        assert meta.title == "Weekly sync"
        assert meta.course == "Ops"
        assert meta.date.isoformat() == "2026-03-04"
        assert meta.speakers == {"A": "Dr. Kowalski"}
        assert meta.tags == ["budget", "q3"]
        assert (tmp_path / f"{doc_id}.pdf").exists()
        assert (tmp_path / f"{doc_id}.md").exists()
        assert not (tmp_path / f"{doc_id}.srt").exists()

    def test_diarize_flag_is_forwarded_to_build_asr_backend(self, tmp_path, monkeypatch):
        backend = FakeASRBackend([make_segment()])
        received: dict = {}
        monkeypatch.setattr(cli, "load_config", lambda: FAKE_CONFIG)
        monkeypatch.setattr(
            cli,
            "build_asr_backend",
            lambda config, **kwargs: (received.update(kwargs), backend)[1],
        )
        monkeypatch.setattr(cli, "OUTPUT_DIR", tmp_path)

        result = runner.invoke(cli.app, ["ingest", str(SAMPLE_A), "--diarize"])

        assert result.exit_code == 0, result.output
        assert received == {"diarize": True}

    def test_no_render_flag_generates_nothing(self, tmp_path, monkeypatch):
        backend = FakeASRBackend([make_segment()])
        monkeypatch.setattr(cli, "load_config", lambda: FAKE_CONFIG)
        monkeypatch.setattr(cli, "build_asr_backend", lambda config, **kwargs: backend)
        monkeypatch.setattr(cli, "OUTPUT_DIR", tmp_path)

        result = runner.invoke(cli.app, ["ingest", str(SAMPLE_A)])

        assert result.exit_code == 0, result.output
        assert list(tmp_path.glob("*.pdf")) == []
        assert list(tmp_path.glob("*.md")) == []
        assert list(tmp_path.glob("*.srt")) == []

    def test_invalid_type_rejected(self, tmp_path, monkeypatch):
        backend = FakeASRBackend([make_segment()])
        monkeypatch.setattr(cli, "load_config", lambda: FAKE_CONFIG)
        monkeypatch.setattr(cli, "build_asr_backend", lambda config, **kwargs: backend)
        monkeypatch.setattr(cli, "OUTPUT_DIR", tmp_path)

        result = runner.invoke(cli.app, ["ingest", str(SAMPLE_A), "--type", "podcast"])

        assert result.exit_code != 0
        assert backend.calls == []

    def test_invalid_render_target_rejected(self, tmp_path, monkeypatch):
        backend = FakeASRBackend([make_segment()])
        monkeypatch.setattr(cli, "load_config", lambda: FAKE_CONFIG)
        monkeypatch.setattr(cli, "build_asr_backend", lambda config, **kwargs: backend)
        monkeypatch.setattr(cli, "OUTPUT_DIR", tmp_path)

        result = runner.invoke(cli.app, ["ingest", str(SAMPLE_A), "--render", "docx"])

        assert result.exit_code != 0

    def test_missing_path_rejected(self, tmp_path, monkeypatch):
        backend = FakeASRBackend([make_segment()])
        monkeypatch.setattr(cli, "load_config", lambda: FAKE_CONFIG)
        monkeypatch.setattr(cli, "build_asr_backend", lambda config, **kwargs: backend)
        monkeypatch.setattr(cli, "OUTPUT_DIR", tmp_path)

        result = runner.invoke(cli.app, ["ingest", str(tmp_path / "missing.wav")])

        assert result.exit_code != 0
        assert backend.calls == []


class TestBuildAsrBackend:
    def test_unimplemented_backend_raises(self):
        with pytest.raises(ValueError, match="not implemented"):
            cli.build_asr_backend({"asr": {"backend": "deepgram"}})

    def test_whisper_local_backend_built_from_config(self):
        from transform.asr.whisper_local import WhisperLocalBackend

        backend = cli.build_asr_backend(
            {
                "asr": {
                    "backend": "whisper-local",
                    "whisper_local": {
                        "model": "large-v3",
                        "vad_filter": True,
                        "word_timestamps": True,
                        "condition_on_previous_text_max_minutes": 30,
                    },
                }
            }
        )

        assert isinstance(backend, WhisperLocalBackend)
        assert backend._model_name == "large-v3"
        assert backend._diarize is False

    def test_diarize_flag_overrides_config_default(self):
        from transform.asr.whisper_local import WhisperLocalBackend

        backend = cli.build_asr_backend(
            {
                "asr": {
                    "backend": "whisper-local",
                    "whisper_local": {
                        "model": "large-v3",
                        "vad_filter": True,
                        "word_timestamps": True,
                        "condition_on_previous_text_max_minutes": 30,
                        "diarize": False,
                    },
                }
            },
            diarize=True,
        )

        assert isinstance(backend, WhisperLocalBackend)
        assert backend._diarize is True

    def test_config_diarize_default_used_when_flag_not_passed(self):
        from transform.asr.whisper_local import WhisperLocalBackend

        backend = cli.build_asr_backend(
            {
                "asr": {
                    "backend": "whisper-local",
                    "whisper_local": {
                        "model": "large-v3",
                        "vad_filter": True,
                        "word_timestamps": True,
                        "condition_on_previous_text_max_minutes": 30,
                        "diarize": True,
                    },
                }
            }
        )

        assert isinstance(backend, WhisperLocalBackend)
        assert backend._diarize is True


class FailingBackend:
    """Raises on a chosen path, so batch-ingest failure handling can be exercised without
    a real backend."""

    def __init__(self, segments, fail_on: str):
        self._segments = segments
        self._fail_on = fail_on
        self.calls: list = []

    def transcribe(self, path, language):
        self.calls.append((path, language))
        if path.name == self._fail_on:
            raise RuntimeError("boom")
        return self._segments


class TestIngestDirectory:
    def test_ingests_every_supported_file_and_skips_unsupported(self, tmp_path):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "a.wav").write_bytes(SAMPLE_A.read_bytes())
        (source_dir / "b.wav").write_bytes(b"different-bytes")
        (source_dir / "notes.txt").write_text("not audio")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        backend = FakeASRBackend([make_segment()])

        results = cli.ingest_directory(
            source_dir,
            recursive=False,
            supported_extensions=["wav"],
            backend=backend,
            language="pl",
            output_dir=output_dir,
            doc_type=TranscriptType.OTHER,
            title=None,
            course=None,
            date=None,
            speakers=[],
            tags=[],
            render_targets=[],
            asr_model_name="fake",
        )

        assert len(results) == 2
        assert len(backend.calls) == 2
        assert len(list(output_dir.glob("*.meta.json"))) == 2

    def test_recursive_flag_descends_into_subdirectories(self, tmp_path):
        source_dir = tmp_path / "source"
        (source_dir / "nested").mkdir(parents=True)
        (source_dir / "top.wav").write_bytes(SAMPLE_A.read_bytes())
        (source_dir / "nested" / "deep.wav").write_bytes(b"different-bytes")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        backend = FakeASRBackend([make_segment()])

        results = cli.ingest_directory(
            source_dir,
            recursive=True,
            supported_extensions=["wav"],
            backend=backend,
            language="pl",
            output_dir=output_dir,
            doc_type=TranscriptType.OTHER,
            title=None,
            course=None,
            date=None,
            speakers=[],
            tags=[],
            render_targets=[],
            asr_model_name="fake",
        )

        assert len(results) == 2

    def test_non_recursive_ignores_subdirectories(self, tmp_path):
        source_dir = tmp_path / "source"
        (source_dir / "nested").mkdir(parents=True)
        (source_dir / "top.wav").write_bytes(SAMPLE_A.read_bytes())
        (source_dir / "nested" / "deep.wav").write_bytes(b"different-bytes")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        backend = FakeASRBackend([make_segment()])

        results = cli.ingest_directory(
            source_dir,
            recursive=False,
            supported_extensions=["wav"],
            backend=backend,
            language="pl",
            output_dir=output_dir,
            doc_type=TranscriptType.OTHER,
            title=None,
            course=None,
            date=None,
            speakers=[],
            tags=[],
            render_targets=[],
            asr_model_name="fake",
        )

        assert len(results) == 1

    def test_one_file_failing_does_not_abort_the_batch(self, tmp_path, capsys):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "bad.wav").write_bytes(SAMPLE_A.read_bytes())
        (source_dir / "good.wav").write_bytes(b"different-bytes")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        backend = FailingBackend([make_segment()], fail_on="bad.wav")

        results = cli.ingest_directory(
            source_dir,
            recursive=False,
            supported_extensions=["wav"],
            backend=backend,
            language="pl",
            output_dir=output_dir,
            doc_type=TranscriptType.OTHER,
            title=None,
            course=None,
            date=None,
            speakers=[],
            tags=[],
            render_targets=[],
            asr_model_name="fake",
        )

        assert len(results) == 1
        assert len(backend.calls) == 2
        assert "bad.wav" in capsys.readouterr().err

    def test_directory_ingest_via_ingest_command(self, tmp_path, monkeypatch):
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "a.wav").write_bytes(SAMPLE_A.read_bytes())
        (source_dir / "b.wav").write_bytes(b"different-bytes")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        backend = FakeASRBackend([make_segment()])
        config = {
            "asr": {"backend": "fake", "language": "pl"},
            "ingest": {"render_default": [], "supported_extensions": ["wav"]},
        }
        monkeypatch.setattr(cli, "load_config", lambda: config)
        monkeypatch.setattr(cli, "build_asr_backend", lambda config, **kwargs: backend)
        monkeypatch.setattr(cli, "OUTPUT_DIR", output_dir)

        result = runner.invoke(cli.app, ["ingest", str(source_dir), "--recursive"])

        assert result.exit_code == 0, result.output
        assert len(list(output_dir.glob("*.meta.json"))) == 2


class TestListCommand:
    def test_lists_ingested_transcripts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "OUTPUT_DIR", tmp_path)
        backend = FakeASRBackend([make_segment()])
        cli.ingest_file(
            SAMPLE_A,
            backend=backend,
            language="pl",
            output_dir=tmp_path,
            doc_type=TranscriptType.LECTURE,
            title="Lecture 1",
            course=None,
            date=None,
            speakers=[],
            tags=[],
            render_targets=[],
            asr_model_name="fake",
        )

        result = runner.invoke(cli.app, ["list"])

        assert result.exit_code == 0, result.output
        assert "Lecture 1" in result.output
        assert "lecture" in result.output

    def test_empty_output_dir_prints_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "OUTPUT_DIR", tmp_path)

        result = runner.invoke(cli.app, ["list"])

        assert result.exit_code == 0, result.output
        assert result.output.strip() == ""


class TestRmCommand:
    def test_removes_all_artifacts_for_doc_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "OUTPUT_DIR", tmp_path)
        backend = FakeASRBackend([make_segment()])
        result_path = cli.ingest_file(
            SAMPLE_A,
            backend=backend,
            language="pl",
            output_dir=tmp_path,
            doc_type=TranscriptType.OTHER,
            title=None,
            course=None,
            date=None,
            speakers=[],
            tags=[],
            render_targets=["pdf", "md", "srt"],
            asr_model_name="fake",
        )
        doc_id = result_path.stem.removesuffix(".segments")

        result = runner.invoke(cli.app, ["rm", doc_id])

        assert result.exit_code == 0, result.output
        assert list(tmp_path.glob(f"{doc_id}.*")) == []

    def test_unknown_doc_id_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "OUTPUT_DIR", tmp_path)

        result = runner.invoke(cli.app, ["rm", "0000000000000099"])

        assert result.exit_code != 0


class TestRelabelCommand:
    def test_updates_meta_speakers_without_retranscribing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "OUTPUT_DIR", tmp_path)
        backend = FakeASRBackend([make_segment(speaker="SPEAKER_00")])
        result_path = cli.ingest_file(
            SAMPLE_A,
            backend=backend,
            language="pl",
            output_dir=tmp_path,
            doc_type=TranscriptType.MEETING,
            title=None,
            course=None,
            date=None,
            speakers=[],
            tags=[],
            render_targets=[],
            asr_model_name="fake",
        )
        doc_id = result_path.stem.removesuffix(".segments")
        assert len(backend.calls) == 1

        result = runner.invoke(
            cli.app, ["relabel", doc_id, "--speaker", "SPEAKER_00=Dr. Kowalski"]
        )

        assert result.exit_code == 0, result.output
        assert read_meta(doc_id, tmp_path).speakers == {"SPEAKER_00": "Dr. Kowalski"}
        assert len(backend.calls) == 1

    def test_malformed_pair_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "OUTPUT_DIR", tmp_path)
        backend = FakeASRBackend([make_segment()])
        result_path = cli.ingest_file(
            SAMPLE_A,
            backend=backend,
            language="pl",
            output_dir=tmp_path,
            doc_type=TranscriptType.OTHER,
            title=None,
            course=None,
            date=None,
            speakers=[],
            tags=[],
            render_targets=[],
            asr_model_name="fake",
        )
        doc_id = result_path.stem.removesuffix(".segments")

        result = runner.invoke(cli.app, ["relabel", doc_id, "--speaker", "no-equals-sign"])

        assert result.exit_code != 0

    def test_unknown_doc_id_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "OUTPUT_DIR", tmp_path)

        result = runner.invoke(
            cli.app, ["relabel", "0000000000000099", "--speaker", "A=B"]
        )

        assert result.exit_code != 0
