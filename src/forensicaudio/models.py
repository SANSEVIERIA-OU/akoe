"""Data models. The forensic invariants live here, in the types.

- Every transcribed segment declares the provenance of its audio
  (`audio_provenance`), and machine transcripts are always AI hypotheses.
- Every derived artifact (profiles, adapters, lexicon, assessments) is an
  `ArtifactRef` with a SHA-256, so the chain of custody covers more than audio.
"""
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from forensicaudio.config import (
    Channel,
    Mode,
    RecordingSource,
    RunProfile,
    StageKind,
    Tier,
    VocalEffort,
)
from forensicaudio.utils import sha256_file


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# Legacy transcript (May 2026). Kept for compatibility with the CLI pipeline.
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Artifacts and recording conditions
# --------------------------------------------------------------------------


class ArtifactRef(BaseModel):
    """A file under the chain of custody. `path` is relative to the case root
    when the artifact belongs to a case, absolute otherwise."""

    path: str
    sha256: str
    kind: str
    created_at: datetime = Field(default_factory=_now)

    @classmethod
    def from_file(cls, path: Path, kind: str, root: Path | None = None) -> "ArtifactRef":
        rel = str(path.relative_to(root)) if root is not None else str(path)
        return cls(path=rel, sha256=sha256_file(path), kind=kind)

    def resolve(self, root: Path | None = None) -> Path:
        p = Path(self.path)
        return p if p.is_absolute() or root is None else root / p

    def verify(self, root: Path | None = None) -> bool:
        p = self.resolve(root)
        return p.exists() and sha256_file(p) == self.sha256


class RecordingCondition(BaseModel):
    """Where and how a recording was made. Profiles and self-assessment are
    reported per condition (MASTER_PLAN §4.2, §4.7)."""

    channel: Channel = Channel.UNKNOWN
    vocal_effort: VocalEffort = VocalEffort.UNKNOWN
    source: RecordingSource = RecordingSource.OTHER
    codec: str | None = None
    sample_rate: int | None = None
    notes: str | None = None

    def key(self) -> str:
        return "/".join(
            [self.channel.value, self.source.value, self.codec or "unknown-codec", self.vocal_effort.value]
        )


class EvidenceRole(str, Enum):
    UNDER_ANALYSIS = "under_analysis"  # recordings to be transcribed
    REFERENCE = "reference"  # the subject's known speech, used to build his model


class EvidenceItem(BaseModel):
    evidence_id: str
    original_name: str
    artifact: ArtifactRef
    condition: RecordingCondition = Field(default_factory=RecordingCondition)
    # Local and transmitted recordings of the same event share a pair_id.
    pair_id: str | None = None
    role: EvidenceRole = EvidenceRole.UNDER_ANALYSIS
    imported_at: datetime = Field(default_factory=_now)


# --------------------------------------------------------------------------
# Hypotheses: multiple interpretations instead of a single transcript
# --------------------------------------------------------------------------


class ConfidenceBand(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Alternative(BaseModel):
    """One possible reading of a word or a segment, with the source that
    produced it (e.g. "whisper-large-v3/generic", "parakeet-tdt-0.6b-v3/adapted:ab12")."""

    text: str
    score: float | None = None
    origin: str
    audio_variant: str = "original"


class WordHypothesis(BaseModel):
    text: str
    start: float
    end: float
    confidence: float | None = None
    confidence_band: ConfidenceBand | None = None
    origin: str
    alternatives: list[Alternative] = Field(default_factory=list)


class SegmentFlag(str, Enum):
    UNINTELLIGIBLE = "unintelligible"
    POSSIBLE_HALLUCINATION = "possible_hallucination"
    MODEL_DISAGREEMENT = "model_disagreement"
    POSSIBLE_OTHER_SPEAKER = "possible_other_speaker"
    LEXICON_DEPENDENT = "lexicon_dependent"


class SegmentHypotheses(BaseModel):
    start: float
    end: float
    # Forensic invariant: provenance is mandatory (no default).
    audio_provenance: StageKind
    hypotheses: list[Alternative] = Field(default_factory=list)  # ranked, best first
    words: list[WordHypothesis] = Field(default_factory=list)
    flags: list[SegmentFlag] = Field(default_factory=list)
    speaker_score: float | None = None
    speaker_condition: str | None = None


class HypothesisTranscript(BaseModel):
    """Machine output. Always an AI hypothesis, never the transcription itself:
    the transcription belongs to the human transcriber (MASTER_PLAN §4.8)."""

    case_id: str
    evidence_id: str | None = None
    segments: list[SegmentHypotheses] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    is_ai_hypothesis: Literal[True] = True
    created_at: datetime = Field(default_factory=_now)


# --------------------------------------------------------------------------
# Subject, vehicle, adapters, datasets, self-assessment
# --------------------------------------------------------------------------


class DatasetSplit(str, Enum):
    TRAIN = "train"
    HOLDOUT = "holdout"


class DatasetItem(BaseModel):
    """A text+audio segment of the subject. Only verified items may be used
    for adaptation (verification loop, MASTER_PLAN §4.2)."""

    item_id: str
    recording_id: str
    audio: ArtifactRef
    start: float
    end: float
    text: str
    condition: RecordingCondition = Field(default_factory=RecordingCondition)
    verified: bool = False
    split: DatasetSplit | None = None


class ConditionProfile(BaseModel):
    condition: RecordingCondition
    seconds: float = 0.0
    n_segments: int = 0
    voiceprint: ArtifactRef | None = None
    timbre: dict[str, Any] = Field(default_factory=dict)


class SpeakerProfile(BaseModel):
    """Subject model across contexts: one entry per recording condition."""

    subject_id: str
    version: int = 1
    conditions: list[ConditionProfile] = Field(default_factory=list)
    dataset_manifest: ArtifactRef | None = None
    holdout_recordings: list[str] = Field(default_factory=list)
    calibration: ArtifactRef | None = None  # score calibration on case material
    created_at: datetime = Field(default_factory=_now)


class NoiseRecording(BaseModel):
    label: str  # e.g. "idle", "city", "highway-windows-open"
    artifact: ArtifactRef
    speed_kmh: float | None = None
    windows_open: bool | None = None
    ventilation: bool | None = None


class VehicleProfile(BaseModel):
    vehicle_id: str
    impulse_responses: list[ArtifactRef] = Field(default_factory=list)
    noise_recordings: list[NoiseRecording] = Field(default_factory=list)
    # True when estimated from the recording itself instead of measured.
    estimated: bool = False
    observed_codecs: list[str] = Field(default_factory=list)
    notes: str | None = None
    created_at: datetime = Field(default_factory=_now)


class AdapterTechnique(str, Enum):
    LORA = "lora"
    FULL = "full"


class AdapterArtifact(BaseModel):
    """A trained subject model. Trained where there is compute, usable anywhere."""

    adapter_id: str
    base_model: str
    base_model_sha256: str | None = None
    technique: AdapterTechnique
    hyperparams: dict[str, Any] = Field(default_factory=dict)
    seed: int
    train_manifest_sha256: str
    conditions: list[str] = Field(default_factory=list)  # RecordingCondition.key() values
    weights: ArtifactRef
    created_at: datetime = Field(default_factory=_now)


class ConditionAssessment(BaseModel):
    condition_key: str
    n_items: int
    metrics: dict[str, float] = Field(default_factory=dict)


class SelfAssessment(BaseModel):
    """Per-case measurement of the system on held-out subject material."""

    case_id: str
    models: list[str] = Field(default_factory=list)
    snr_db: float | None = None
    overall: dict[str, float] = Field(default_factory=dict)
    by_condition: list[ConditionAssessment] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)


# --------------------------------------------------------------------------
# Processing log (chain of custody)
# --------------------------------------------------------------------------


class StageLog(BaseModel):
    name: str
    kind: StageKind
    mode: Mode
    params: dict[str, Any] = Field(default_factory=dict)
    input_sha256: str | None = None
    output_sha256: str | None = None
    model_version: str | None = None
    model_sha256: str | None = None
    tier: Tier | None = None
    device: str | None = None
    started_at: datetime
    finished_at: datetime


class ProcessingLog(BaseModel):
    case_id: str
    mode: Mode
    tier: Tier | None = None
    profile: RunProfile | None = None
    stages: list[StageLog] = Field(default_factory=list)
    input_files: dict[str, str]
    output_files: dict[str, str] = Field(default_factory=dict)
    # Derived artifacts beyond audio: profiles, adapters, lexicon, assessments.
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)

    def register(self, artifact: ArtifactRef) -> None:
        self.artifacts.append(artifact)
