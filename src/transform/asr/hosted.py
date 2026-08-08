"""Hosted ASR backend (default, see ADR 0002) — calls a third-party API instead of running a
local model. Nothing outside `transform/asr/` may import this module's HTTP client directly."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import requests

from extract.hashing import compute_sha256
from models.schemas import Segment

_UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
_TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"


class AssemblyAIBackend:
    """`ASRBackend` implementation calling AssemblyAI's hosted transcription API.
    Diarization (`speaker_labels=True`) is requested in the same call — the hosted default
    gets speaker labels for free, see ADR 0002 and CLAUDE.md "ASR is pluggable, not assumed"."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        poll_interval_s: float = 3.0,
        poll_timeout_s: float = 1800.0,
        session: requests.Session | None = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get("ASSEMBLYAI_API_KEY")
        if not key:
            raise ValueError(
                "ASSEMBLYAI_API_KEY is not set (see .env.example) — required when "
                "asr.backend is 'assemblyai'"
            )
        self._api_key = key
        self._poll_interval_s = poll_interval_s
        self._poll_timeout_s = poll_timeout_s
        self._session = session or requests.Session()

    def transcribe(self, path: Path, language: str) -> list[Segment]:
        doc_id = compute_sha256(path)[:16]
        audio_url = self._upload(path)
        transcript_id = self._submit(audio_url, language)
        transcript = self._poll(transcript_id)
        return self._to_segments(transcript, doc_id)

    def _headers(self) -> dict[str, str]:
        return {"authorization": self._api_key}

    def _upload(self, path: Path) -> str:
        with path.open("rb") as f:
            response = self._session.post(_UPLOAD_URL, headers=self._headers(), data=f)
        response.raise_for_status()
        upload_url: str = response.json()["upload_url"]
        return upload_url

    def _submit(self, audio_url: str, language: str) -> str:
        response = self._session.post(
            _TRANSCRIPT_URL,
            headers=self._headers(),
            json={
                "audio_url": audio_url,
                "language_code": language,
                "speaker_labels": True,
            },
        )
        response.raise_for_status()
        transcript_id: str = response.json()["id"]
        return transcript_id

    def _poll(self, transcript_id: str) -> dict[str, Any]:
        elapsed_s = 0.0
        while True:
            response = self._session.get(
                f"{_TRANSCRIPT_URL}/{transcript_id}", headers=self._headers()
            )
            response.raise_for_status()
            transcript: dict[str, Any] = response.json()
            status = transcript["status"]
            if status == "completed":
                return transcript
            if status == "error":
                raise RuntimeError(f"AssemblyAI transcription failed: {transcript.get('error')}")
            if elapsed_s >= self._poll_timeout_s:
                raise TimeoutError(
                    f"AssemblyAI transcript {transcript_id} did not complete within "
                    f"{self._poll_timeout_s}s"
                )
            time.sleep(self._poll_interval_s)
            elapsed_s += self._poll_interval_s

    @staticmethod
    def _to_segments(transcript: dict[str, Any], doc_id: str) -> list[Segment]:
        utterances = transcript.get("utterances") or []
        return [
            Segment(
                doc_id=doc_id,
                seg=i,
                start=utterance["start"] / 1000.0,
                end=utterance["end"] / 1000.0,
                text=utterance["text"],
                speaker=utterance.get("speaker"),
                conf=utterance.get("confidence", 0.0),
            )
            for i, utterance in enumerate(utterances)
        ]
