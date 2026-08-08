"""Pydantic contracts for every pipeline stage boundary (see CLAUDE.md "Typed boundaries")."""

from __future__ import annotations

import re
from datetime import date as date_
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator

_DOC_ID_RE = re.compile(r"^[0-9a-f]{16}$")


class TranscriptType(StrEnum):
    LECTURE = "lecture"
    TRAINING = "training"
    MEETING = "meeting"
    INTERVIEW = "interview"
    OTHER = "other"


class Segment(BaseModel):
    """One line of `output/<doc_id>.segments.jsonl` (spec.md §2)."""

    doc_id: str
    seg: int = Field(ge=0)
    start: float
    end: float
    text: str
    speaker: str | None = None
    conf: float

    @field_validator("doc_id")
    @classmethod
    def _doc_id_format(cls, v: str) -> str:
        if not _DOC_ID_RE.match(v):
            raise ValueError("doc_id must be 16 lowercase hex characters (sha256(bytes)[:16])")
        return v

    @field_validator("text")
    @classmethod
    def _text_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v

    @model_validator(mode="after")
    def _end_after_start(self) -> Segment:
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class TranscriptMeta(BaseModel):
    """`output/<doc_id>.meta.json` sidecar (spec.md §2)."""

    doc_id: str
    source_path: str
    sha256: str
    type: TranscriptType
    title: str
    course: str | None = None
    speakers: dict[str, str] = Field(default_factory=dict)
    date: date_ | None = None
    duration_s: float
    language: str
    model: str
    tags: list[str] = Field(default_factory=list)
    ingested_at: datetime

    @field_validator("doc_id")
    @classmethod
    def _doc_id_format(cls, v: str) -> str:
        if not _DOC_ID_RE.match(v):
            raise ValueError("doc_id must be 16 lowercase hex characters (sha256(bytes)[:16])")
        return v

    @field_validator("title")
    @classmethod
    def _title_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("title must not be empty")
        return v

    @field_validator("duration_s")
    @classmethod
    def _duration_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("duration_s must not be negative")
        return v
