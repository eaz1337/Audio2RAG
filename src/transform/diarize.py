"""`pyannote.audio` wrapper for the `whisper-local` fallback's diarization (TASKS.md
INGEST-6). Hosted backends return speaker labels inline and never need this (ADR 0002).
`pyannote.audio` lives behind the `[diarize]` extra and is imported lazily, inside
`diarize()`, so nothing pays for it unless diarization is actually requested."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


def diarize(path: Path, hf_token: str | None) -> list[SpeakerTurn]:
    """Runs `pyannote`'s speaker-diarization pipeline over `path`. Raises `ImportError` if
    the `[diarize]` extra is not installed."""
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )
    annotation = pipeline(str(path))
    return [
        SpeakerTurn(start=turn.start, end=turn.end, speaker=speaker)
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


def assign_speaker(start: float, end: float, turns: list[SpeakerTurn]) -> str | None:
    """Labels a `[start, end)` span with the speaker of greatest time overlap among
    `turns`, or `None` if it overlaps none of them."""
    best_speaker: str | None = None
    best_overlap = 0.0
    for turn in turns:
        overlap = min(end, turn.end) - max(start, turn.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn.speaker
    return best_speaker
