from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from forensicaudio.config import Mode, StageKind


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Word(BaseModel):
    text: str
    start: float
    end: float
    confidence: float


class Segment(BaseModel):
    start: float
    end: float
    speaker_id: str | None = None
    words: list[Word] = Field(default_factory=list)
    # Provenance of the audio this segment was transcribed from. Forensic
    # invariant: never collapse or drop this field downstream.
    audio_provenance: StageKind


class Transcript(BaseModel):
    case_id: str
    segments: list[Segment] = Field(default_factory=list)
    asr_model: str
    created_at: datetime = Field(default_factory=_now)


class StageLog(BaseModel):
    name: str
    kind: StageKind
    mode: Mode
    params: dict[str, Any] = Field(default_factory=dict)
    input_sha256: str | None = None
    output_sha256: str | None = None
    model_version: str | None = None
    started_at: datetime
    finished_at: datetime


class ProcessingLog(BaseModel):
    case_id: str
    mode: Mode
    stages: list[StageLog] = Field(default_factory=list)
    input_files: dict[str, str]
    output_files: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_now)
