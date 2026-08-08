from pathlib import Path

from fakes import FakeASRBackend

from models.schemas import Segment
from transform.transcribe import Transcriber

DOC_ID = "a3f9c1b2d4e6f801"


def make_segment(**overrides):
    fields = {
        "doc_id": DOC_ID,
        "seg": 0,
        "start": 0.0,
        "end": 1.5,
        "text": "Hello world.",
        "speaker": None,
        "conf": -0.1,
    }
    fields.update(overrides)
    return Segment(**fields)


class TestFakeASRBackend:
    def test_returns_fixed_injected_segments(self):
        segments = [make_segment(seg=0), make_segment(seg=1, start=1.5, end=3.0)]
        backend = FakeASRBackend(segments)

        result = backend.transcribe(Path("audio.mp3"), "pl")

        assert result == segments

    def test_records_calls(self):
        backend = FakeASRBackend([make_segment()])

        backend.transcribe(Path("a.mp3"), "pl")
        backend.transcribe(Path("b.mp3"), "en")

        assert backend.calls == [(Path("a.mp3"), "pl"), (Path("b.mp3"), "en")]


class TestTranscriber:
    def test_delegates_to_injected_backend(self):
        segments = [make_segment()]
        backend = FakeASRBackend(segments)
        transcriber = Transcriber(backend)

        result = transcriber.transcribe(Path("audio.mp3"), "pl")

        assert result == segments
        assert backend.calls == [(Path("audio.mp3"), "pl")]

    def test_does_not_call_backend_until_asked(self):
        backend = FakeASRBackend([make_segment()])
        Transcriber(backend)

        assert backend.calls == []
