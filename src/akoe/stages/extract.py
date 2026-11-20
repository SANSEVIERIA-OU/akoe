# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
from pathlib import Path

import numpy as np

from akoe.config import StageKind

# Generative: this stage can hallucinate content under low SNR. Never used
# in forensic mode.
KIND = StageKind.NEURAL


def run(input_path: Path, output_path: Path, speaker_embedding: np.ndarray) -> None:
    """Target speaker extraction (WeSep / SpeakerBeam).

    Isolates the target speaker's voice in a mix using the embedding.
    """
    raise NotImplementedError
