from pathlib import Path

import numpy as np

from forensicaudio.config import StageKind

# ECAPA inference is deterministic given fixed weights; we still treat the
# embedding as a derived artifact, never as audio.
KIND = StageKind.DETERMINISTIC


def run(samples_dir: Path) -> np.ndarray:
    """Compute a single speaker embedding (e.g. ECAPA-TDNN) from reference samples.

    Adapts to material volume: short clips → embedding only; longer material
    enables optional ASR fine-tuning downstream (not handled here).
    """
    raise NotImplementedError
