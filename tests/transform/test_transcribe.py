from pathlib import Path

from fakes import FakeASRBackend

from extract.hashing import compute_sha256
from load.write_canonical import write_canonical
from models.schemas import Segment, TranscriptMeta
from transform.transcribe import Transcriber

FIXTURES = Path(__file__).parent.parent / "fixtures"
SAMPLE_A = FIXTURES / "sample_a.wav"
SAMPLE_B = FIXTURES / "sample_b.wav"

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


def make_meta(**overrides):
    fields = {
        "doc_id": DOC_ID,
        "source_path": str(SAMPLE_A),
        "sha256": compute_sha256(SAMPLE_A),
        "type": "lecture",
        "title": "Sample",
        "duration_s": 3.0,
        "language": "pl",
        "model": "fake",
        "ingested_at": "2026-03-04T18:22:11Z",
    }
    fields.update(overrides)
    return TranscriptMeta(**fields)


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


class TestTranscribeCached:
    def _cached_doc_id(self) -> str:
        return compute_sha256(SAMPLE_A)[:16]

    def test_cache_hit_does_not_call_backend(self, tmp_path):
        doc_id = self._cached_doc_id()
        cached_segments = [make_segment(doc_id=doc_id)]
        write_canonical(cached_segments, make_meta(doc_id=doc_id), tmp_path)
        backend = FakeASRBackend([make_segment(doc_id=doc_id, text="fresh transcription")])
        transcriber = Transcriber(backend)

        result = transcriber.transcribe_cached(SAMPLE_A, "pl", tmp_path)

        assert result == cached_segments
        assert backend.calls == []

    def test_cache_miss_when_no_prior_artifacts(self, tmp_path):
        segments = [make_segment()]
        backend = FakeASRBackend(segments)
        transcriber = Transcriber(backend)

        result = transcriber.transcribe_cached(SAMPLE_A, "pl", tmp_path)

        assert backend.calls == [(SAMPLE_A, "pl")]
        assert result == segments

    def test_cache_miss_when_recorded_hash_does_not_match(self, tmp_path):
        doc_id = self._cached_doc_id()
        write_canonical(
            [make_segment(doc_id=doc_id)],
            make_meta(doc_id=doc_id, sha256=compute_sha256(SAMPLE_B)),
            tmp_path,
        )
        backend = FakeASRBackend([make_segment(doc_id=doc_id)])
        transcriber = Transcriber(backend)

        transcriber.transcribe_cached(SAMPLE_A, "pl", tmp_path)

        assert backend.calls == [(SAMPLE_A, "pl")]

    def test_force_bypasses_cache(self, tmp_path):
        doc_id = self._cached_doc_id()
        write_canonical(
            [make_segment(doc_id=doc_id)], make_meta(doc_id=doc_id), tmp_path
        )
        backend = FakeASRBackend([make_segment(doc_id=doc_id)])
        transcriber = Transcriber(backend)

        transcriber.transcribe_cached(SAMPLE_A, "pl", tmp_path, force=True)

        assert backend.calls == [(SAMPLE_A, "pl")]
