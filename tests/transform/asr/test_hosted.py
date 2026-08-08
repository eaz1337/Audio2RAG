from pathlib import Path

import pytest

from extract.hashing import compute_sha256
from transform.asr.hosted import AssemblyAIBackend

FIXTURES = Path(__file__).parent.parent.parent / "fixtures"
SAMPLE_A = FIXTURES / "sample_a.wav"


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    """Stands in for `requests.Session` — records calls, never touches the network."""

    def __init__(self, poll_statuses: list[dict] | None = None) -> None:
        self.posts: list[tuple[str, dict]] = []
        self.gets: list[str] = []
        self._poll_statuses = poll_statuses or [
            {
                "status": "completed",
                "utterances": [
                    {
                        "start": 0,
                        "end": 1500,
                        "text": "Hello world.",
                        "speaker": "A",
                        "confidence": 0.97,
                    },
                    {
                        "start": 1500,
                        "end": 3000,
                        "text": "Second line.",
                        "speaker": "B",
                        "confidence": 0.91,
                    },
                ],
            }
        ]

    def post(self, url: str, headers: dict, **kwargs) -> _FakeResponse:
        self.posts.append((url, headers))
        if url.endswith("/upload"):
            return _FakeResponse({"upload_url": "https://cdn.assemblyai.com/upload/fake"})
        return _FakeResponse({"id": "transcript-123"})

    def get(self, url: str, headers: dict, **kwargs) -> _FakeResponse:
        self.gets.append(url)
        payload = self._poll_statuses[min(len(self.gets) - 1, len(self._poll_statuses) - 1)]
        return _FakeResponse(payload)


class TestAssemblyAIBackendInit:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)

        with pytest.raises(ValueError, match="ASSEMBLYAI_API_KEY"):
            AssemblyAIBackend(session=_FakeSession())

    def test_reads_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("ASSEMBLYAI_API_KEY", "env-key")

        backend = AssemblyAIBackend(session=_FakeSession())

        assert backend._api_key == "env-key"

    def test_explicit_api_key_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("ASSEMBLYAI_API_KEY", "env-key")

        backend = AssemblyAIBackend(api_key="explicit-key", session=_FakeSession())

        assert backend._api_key == "explicit-key"


class TestAssemblyAIBackendTranscribe:
    def test_returns_segments_with_speaker_labels(self):
        session = _FakeSession()
        backend = AssemblyAIBackend(api_key="k", session=session)

        segments = backend.transcribe(SAMPLE_A, "pl")

        assert [s.text for s in segments] == ["Hello world.", "Second line."]
        assert [s.speaker for s in segments] == ["A", "B"]
        assert [s.seg for s in segments] == [0, 1]
        assert segments[0].start == 0.0
        assert segments[0].end == 1.5
        assert segments[1].start == 1.5
        assert segments[1].end == 3.0

    def test_doc_id_computed_from_audio_content(self):
        session = _FakeSession()
        backend = AssemblyAIBackend(api_key="k", session=session)

        segments = backend.transcribe(SAMPLE_A, "pl")

        expected_doc_id = compute_sha256(SAMPLE_A)[:16]
        assert all(s.doc_id == expected_doc_id for s in segments)

    def test_uploads_then_submits_with_diarization_requested(self):
        session = _FakeSession()
        backend = AssemblyAIBackend(api_key="k", session=session)

        backend.transcribe(SAMPLE_A, "pl")

        upload_call, submit_call = session.posts
        assert upload_call[0] == "https://api.assemblyai.com/v2/upload"
        assert submit_call[0] == "https://api.assemblyai.com/v2/transcript"

    def test_language_is_passed_through(self):
        session = _FakeSession()
        backend = AssemblyAIBackend(api_key="k", session=session)
        captured = {}
        original_post = session.post

        def spy_post(url, headers, **kwargs):
            if url.endswith("/transcript"):
                captured.update(kwargs.get("json", {}))
            return original_post(url, headers, **kwargs)

        session.post = spy_post

        backend.transcribe(SAMPLE_A, "pl")

        assert captured["language_code"] == "pl"
        assert captured["speaker_labels"] is True

    def test_api_key_sent_only_in_header_not_body_or_url(self):
        session = _FakeSession()
        backend = AssemblyAIBackend(api_key="secret-key", session=session)

        backend.transcribe(SAMPLE_A, "pl")

        for url, headers in session.posts:
            assert "secret-key" not in url
            assert headers["authorization"] == "secret-key"

    def test_polls_until_completed(self):
        session = _FakeSession(
            poll_statuses=[
                {"status": "processing"},
                {"status": "processing"},
                {"status": "completed", "utterances": []},
            ]
        )
        backend = AssemblyAIBackend(api_key="k", poll_interval_s=0, session=session)

        segments = backend.transcribe(SAMPLE_A, "pl")

        assert len(session.gets) == 3
        assert segments == []

    def test_error_status_raises_runtime_error(self):
        session = _FakeSession(poll_statuses=[{"status": "error", "error": "bad audio"}])
        backend = AssemblyAIBackend(api_key="k", session=session)

        with pytest.raises(RuntimeError, match="bad audio"):
            backend.transcribe(SAMPLE_A, "pl")

    def test_timeout_raises(self):
        session = _FakeSession(poll_statuses=[{"status": "processing"}])
        backend = AssemblyAIBackend(
            api_key="k", poll_interval_s=0, poll_timeout_s=0, session=session
        )

        with pytest.raises(TimeoutError):
            backend.transcribe(SAMPLE_A, "pl")

    def test_no_utterances_returns_empty_list(self):
        session = _FakeSession(poll_statuses=[{"status": "completed"}])
        backend = AssemblyAIBackend(api_key="k", session=session)

        assert backend.transcribe(SAMPLE_A, "pl") == []


class TestAssemblyAIBackendSatisfiesProtocol:
    def test_isinstance_of_asr_backend(self):
        from transform.asr.base import ASRBackend

        backend = AssemblyAIBackend(api_key="k", session=_FakeSession())

        assert isinstance(backend, ASRBackend)
