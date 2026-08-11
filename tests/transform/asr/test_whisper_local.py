import logging
import math
from pathlib import Path

from extract.hashing import compute_sha256
from transform.asr.whisper_local import WhisperLocalBackend
from transform.diarize import SpeakerTurn

FIXTURES = Path(__file__).parent.parent.parent / "fixtures"
SAMPLE_A = FIXTURES / "sample_a.wav"


class _FakeWhisperSegment:
    def __init__(self, start: float, end: float, text: str, avg_logprob: float) -> None:
        self.start = start
        self.end = end
        self.text = text
        self.avg_logprob = avg_logprob


class _FakeModel:
    def __init__(self, segments: list[_FakeWhisperSegment]) -> None:
        self._segments = segments
        self.calls: list[dict] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append({"audio": audio, **kwargs})
        return self._segments, object()


def _make_backend(
    *,
    segments: list[_FakeWhisperSegment] | None = None,
    duration_s: float = 60.0,
    condition_on_previous_text_max_minutes: float = 30.0,
):
    model = _FakeModel(
        segments
        if segments is not None
        else [
            _FakeWhisperSegment(0.0, 1.5, "  Hello world.  ", -0.1),
            _FakeWhisperSegment(1.5, 3.0, "Second line.", -0.5),
        ]
    )
    factories: list[tuple] = []

    def model_factory(model_name: str, device: str, compute_type: str):
        factories.append((model_name, device, compute_type))
        return model

    def audio_decoder(path: Path):
        return ("decoded-audio", duration_s)

    backend = WhisperLocalBackend(
        model="large-v3",
        condition_on_previous_text_max_minutes=condition_on_previous_text_max_minutes,
        model_factory=model_factory,
        audio_decoder=audio_decoder,
    )
    return backend, model, factories


class TestWhisperLocalBackendTranscribe:
    def test_returns_segments_with_conf_from_logprob(self):
        backend, _model, _factories = _make_backend()

        segments = backend.transcribe(SAMPLE_A, "pl")

        assert [s.text for s in segments] == ["Hello world.", "Second line."]
        assert [s.seg for s in segments] == [0, 1]
        assert segments[0].start == 0.0
        assert segments[0].end == 1.5
        assert all(s.speaker is None for s in segments)
        assert segments[0].conf == math.exp(-0.1)
        assert 0.0 <= segments[0].conf <= 1.0
        assert 0.0 <= segments[1].conf <= 1.0

    def test_doc_id_computed_from_audio_content(self):
        backend, _model, _factories = _make_backend()

        segments = backend.transcribe(SAMPLE_A, "pl")

        assert segments[0].doc_id == compute_sha256(SAMPLE_A)[:16]

    def test_conf_clamped_to_unit_interval(self):
        backend, _model, _ = _make_backend(
            segments=[_FakeWhisperSegment(0.0, 1.0, "Loud and clear.", 5.0)]
        )

        segments = backend.transcribe(SAMPLE_A, "pl")

        assert segments[0].conf == 1.0

    def test_model_loaded_lazily_and_cached(self):
        backend, _model, factories = _make_backend()

        assert factories == []

        backend.transcribe(SAMPLE_A, "pl")
        backend.transcribe(SAMPLE_A, "pl")

        assert len(factories) == 1

    def test_short_audio_conditions_on_previous_text(self):
        backend, model, _ = _make_backend(duration_s=60.0, condition_on_previous_text_max_minutes=30.0)

        backend.transcribe(SAMPLE_A, "pl")

        assert model.calls[0]["condition_on_previous_text"] is True

    def test_long_audio_disables_condition_on_previous_text(self):
        backend, model, _ = _make_backend(
            duration_s=31 * 60.0, condition_on_previous_text_max_minutes=30.0
        )

        backend.transcribe(SAMPLE_A, "pl")

        assert model.calls[0]["condition_on_previous_text"] is False

    def test_vad_and_word_timestamps_passed_through(self):
        model = _FakeModel([_FakeWhisperSegment(0.0, 1.0, "Hi.", -0.2)])

        def model_factory(model_name, device, compute_type):
            return model

        def audio_decoder(path):
            return ("decoded-audio", 10.0)

        backend = WhisperLocalBackend(
            vad_filter=True,
            word_timestamps=True,
            model_factory=model_factory,
            audio_decoder=audio_decoder,
        )

        backend.transcribe(SAMPLE_A, "pl")

        assert model.calls[0]["vad_filter"] is True
        assert model.calls[0]["word_timestamps"] is True
        assert model.calls[0]["language"] == "pl"


class TestWhisperLocalBackendLazyImport:
    def test_module_import_does_not_require_faster_whisper(self):
        """The `faster-whisper` import must be lazy (CLAUDE.md "Dependencies") — merely
        constructing a backend with an injected factory/decoder must never import it."""
        backend = WhisperLocalBackend(
            model_factory=lambda *a: _FakeModel([]),
            audio_decoder=lambda p: ("audio", 1.0),
        )

        backend.transcribe(SAMPLE_A, "pl")


class TestWhisperLocalBackendDiarization:
    def test_diarize_disabled_leaves_speaker_none(self):
        backend, _model, _factories = _make_backend()

        segments = backend.transcribe(SAMPLE_A, "pl")

        assert all(s.speaker is None for s in segments)

    def test_diarize_enabled_labels_segments_from_turns(self):
        model = _FakeModel(
            [
                _FakeWhisperSegment(0.0, 1.5, "Hello world.", -0.1),
                _FakeWhisperSegment(1.5, 3.0, "Second line.", -0.5),
            ]
        )
        turns = [
            SpeakerTurn(start=0.0, end=1.5, speaker="SPEAKER_00"),
            SpeakerTurn(start=1.5, end=3.0, speaker="SPEAKER_01"),
        ]
        backend = WhisperLocalBackend(
            diarize=True,
            hf_token="hf-token",
            model_factory=lambda *a: model,
            audio_decoder=lambda p: ("decoded-audio", 60.0),
            diarizer=lambda path, hf_token: turns,
        )

        segments = backend.transcribe(SAMPLE_A, "pl")

        assert [s.speaker for s in segments] == ["SPEAKER_00", "SPEAKER_01"]

    def test_diarize_enabled_but_hf_token_missing_warns_and_continues(self, caplog):
        model = _FakeModel([_FakeWhisperSegment(0.0, 1.5, "Hello world.", -0.1)])
        diarizer_calls = []

        def diarizer(path, hf_token):
            diarizer_calls.append((path, hf_token))
            raise AssertionError("diarizer must not be called without an HF token")

        backend = WhisperLocalBackend(
            diarize=True,
            hf_token=None,
            model_factory=lambda *a: model,
            audio_decoder=lambda p: ("decoded-audio", 60.0),
            diarizer=diarizer,
        )

        with caplog.at_level(logging.WARNING):
            segments = backend.transcribe(SAMPLE_A, "pl")

        assert diarizer_calls == []
        assert all(s.speaker is None for s in segments)
        assert any("HF_TOKEN" in record.message for record in caplog.records)

    def test_diarize_enabled_but_pyannote_missing_warns_and_continues(self, caplog):
        model = _FakeModel([_FakeWhisperSegment(0.0, 1.5, "Hello world.", -0.1)])

        def diarizer(path, hf_token):
            raise ImportError("pyannote.audio is not installed")

        backend = WhisperLocalBackend(
            diarize=True,
            hf_token="hf-token",
            model_factory=lambda *a: model,
            audio_decoder=lambda p: ("decoded-audio", 60.0),
            diarizer=diarizer,
        )

        with caplog.at_level(logging.WARNING):
            segments = backend.transcribe(SAMPLE_A, "pl")

        assert all(s.speaker is None for s in segments)
        assert any("pyannote" in record.message for record in caplog.records)

    def test_hf_token_read_from_environment(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "env-token")
        seen_tokens = []

        def diarizer(path, hf_token):
            seen_tokens.append(hf_token)
            return []

        model = _FakeModel([_FakeWhisperSegment(0.0, 1.5, "Hello world.", -0.1)])
        backend = WhisperLocalBackend(
            diarize=True,
            model_factory=lambda *a: model,
            audio_decoder=lambda p: ("decoded-audio", 60.0),
            diarizer=diarizer,
        )

        backend.transcribe(SAMPLE_A, "pl")

        assert seen_tokens == ["env-token"]
