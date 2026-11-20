# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
from enum import Enum
from pathlib import Path

from pydantic import BaseModel


class Mode(str, Enum):
    # forensic: deterministic transforms only — admissible chain of custody.
    # investigative: includes neural generative stages — internal use only.
    FORENSIC = "forensic"
    INVESTIGATIVE = "investigative"
    BOTH = "both"


class StageKind(str, Enum):
    DETERMINISTIC = "deterministic"
    NEURAL = "neural"


class Tier(str, Enum):
    """Where the engine runs. Same interface on every tier (MASTER_PLAN §5.2)."""

    EDGE = "edge"  # laptop without GPU
    WORKSTATION = "workstation"  # single NVIDIA GPU
    CLOUD = "cloud"  # GPU VM under the user's control, never a SaaS


class RunProfile(str, Enum):
    """Which pipeline shape to run.

    vehicle-single-speaker: primary scenario (subject alone in a car). Target
    speaker extraction is parked; the transcriber receives the original audio
    by default (MASTER_PLAN §4.4).
    generic: the May 2026 pipeline, kept for comparison.
    """

    VEHICLE_SINGLE_SPEAKER = "vehicle-single-speaker"
    GENERIC = "generic"


class Channel(str, Enum):
    CAR = "car"
    PHONE = "phone"
    ROOM = "room"
    OTHER = "other"
    UNKNOWN = "unknown"


class VocalEffort(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    RAISED = "raised"
    WHISPERED = "whispered"
    UNKNOWN = "unknown"


class RecordingSource(str, Enum):
    """Covert devices may both record locally and transmit (MASTER_PLAN §4.3)."""

    LOCAL = "local"
    TRANSMITTED = "transmitted"
    OTHER = "other"  # phone intercepts, interviews, public statements...


class CaseConfig(BaseModel):
    case_id: str
    evidence_path: Path
    speaker_samples_dir: Path
    output_dir: Path
    mode: Mode = Mode.BOTH
    tier: Tier = Tier.EDGE
    profile: RunProfile = RunProfile.VEHICLE_SINGLE_SPEAKER
