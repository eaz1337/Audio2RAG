"""Local `whisper-local` fallback backend (see ADR 0002) — runs entirely offline via
faster-whisper. `faster-whisper` and `torch` live behind the `[cuda]` extra and are imported
lazily so nothing pays for them unless `asr.backend: whisper-local` is actually selected."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Callable

from extract.hashing import compute_sha256
from models.schemas import Segment
from transform.diarize import SpeakerTurn, assign_speaker
from transform.diarize import diarize as _pyannote_diarize

_SAMPLE_RATE = 16000

logger = logging.getLogger(__name__)


def _decode_audio(path: Path) -> tuple[Any, float]:
    from faster_whisper.audio import decode_audio

    audio = decode_audio(str(path))
    duration_s = len(audio) / _SAMPLE_RATE
    return audio, duration_s


def _load_model(model_name: str, device: str, compute_type: str) -> Any:
    from faster_whisper import WhisperModel

    return WhisperModel(model_name, device=device, compute_type=compute_type)


def _logprob_to_conf(avg_logprob: float) -> float:
    """faster-whisper's `avg_logprob` is a mean log-probability (<= 0, less negative is
    better); `exp` maps it onto the same `0.0-1.0` scale every other backend's `conf` uses."""
    return max(0.0, min(1.0, math.exp(avg_logprob)))


class WhisperLocalBackend:
    """`ASRBackend` implementation running `faster-whisper` locally — the offline fallback
    named in ADR 0002. Never imported unless `asr.backend: whisper-local` is selected."""

    def __init__(
        self,
        model: str = "large-v3",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        vad_filter: bool = True,
        word_timestamps: bool = True,
        condition_on_previous_text_max_minutes: float = 30.0,
        diarize: bool = False,
        hf_token: str | None = None,
        model_factory: Callable[[str, str, str], Any] = _load_model,
        audio_decoder: Callable[[Path], tuple[Any, float]] = _decode_audio,
        diarizer: Callable[[Path, str | None], list[SpeakerTurn]] = _pyannote_diarize,
    ) -> None:
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._vad_filter = vad_filter
        self._word_timestamps = word_timestamps
        self._condition_on_previous_text_max_minutes = condition_on_previous_text_max_minutes
        self._diarize = diarize
        self._hf_token = hf_token if hf_token is not None else os.environ.get("HF_TOKEN")
        self._model_factory = model_factory
        self._audio_decoder = audio_decoder
        self._diarizer = diarizer
        self._model: Any | None = None

    def transcribe(self, path: Path, language: str) -> list[Segment]:
        doc_id = compute_sha256(path)[:16]
        audio, duration_s = self._audio_decoder(path)
        condition_on_previous_text = (
            duration_s / 60.0 <= self._condition_on_previous_text_max_minutes
        )

        if self._model is None:
            self._model = self._model_factory(self._model_name, self._device, self._compute_type)

        segments, _info = self._model.transcribe(
            audio,
            language=language,
            vad_filter=self._vad_filter,
            word_timestamps=self._word_timestamps,
            condition_on_previous_text=condition_on_previous_text,
        )
        result = [
            Segment(
                doc_id=doc_id,
                seg=i,
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
                speaker=None,
                conf=_logprob_to_conf(segment.avg_logprob),
            )
            for i, segment in enumerate(segments)
        ]

        if self._diarize:
            turns = self._run_diarization(path)
            if turns is not None:
                result = [
                    segment.model_copy(
                        update={"speaker": assign_speaker(segment.start, segment.end, turns)}
                    )
                    for segment in result
                ]
        return result

    def _run_diarization(self, path: Path) -> list[SpeakerTurn] | None:
        """Returns `None` (never raises) when diarization can't run — missing `HF_TOKEN`
        or the `[diarize]` extra not installed — so ingest completes without speaker
        labels rather than crashing (TASKS.md INGEST-6)."""
        if not self._hf_token:
            logger.warning(
                "whisper-local diarization requested but HF_TOKEN is not set; "
                "continuing without speaker labels"
            )
            return None
        try:
            return self._diarizer(path, self._hf_token)
        except ImportError:
            logger.warning(
                "whisper-local diarization requested but pyannote.audio is not installed "
                "(pip install -e '.[diarize]'); continuing without speaker labels"
            )
            return None
