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


class CaseConfig(BaseModel):
    case_id: str
    evidence_path: Path
    speaker_samples_dir: Path
    output_dir: Path
    mode: Mode = Mode.BOTH
