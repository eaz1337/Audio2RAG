from pathlib import Path

from fakes import FakeASRBackend

from models.schemas import Segment
from transform.asr.base import ASRBackend

DOC_ID = "a3f9c1b2d4e6f801"


class _NotABackend:
    pass


def test_fake_backend_satisfies_the_protocol():
    backend = FakeASRBackend([])
    assert isinstance(backend, ASRBackend)


def test_unrelated_object_does_not_satisfy_the_protocol():
    assert not isinstance(_NotABackend(), ASRBackend)


def test_protocol_signature_matches_asr_backend_transcribe():
    segment = Segment(
        doc_id=DOC_ID,
        seg=0,
        start=0.0,
        end=1.0,
        text="Hello.",
        speaker=None,
        conf=0.0,
    )
    backend: ASRBackend = FakeASRBackend([segment])

    result = backend.transcribe(Path("audio.mp3"), "pl")

    assert result == [segment]
