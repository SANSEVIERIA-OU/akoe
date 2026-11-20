# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
from pathlib import Path

from akoe.config import StageKind
from akoe.models import Transcript

# ASR output is always an AI hypothesis, labeled with per-word confidence.
KIND = StageKind.NEURAL


def run(
    input_path: Path,
    case_id: str,
    segments: list[tuple[float, float]] | None = None,
) -> Transcript:
    """Whisper-based ASR with per-word confidence. Marked as AI hypothesis."""
    raise NotImplementedError("Install [ml] extras and integrate faster-whisper here.")
