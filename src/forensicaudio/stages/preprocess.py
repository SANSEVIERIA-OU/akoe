from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from forensicaudio.config import StageKind

KIND = StageKind.DETERMINISTIC


def run(input_path: Path, output_path: Path, *, target_sr: int = 16000) -> None:
    """Load audio, resample to mono at target_sr, peak-normalize, write PCM_16."""
    audio, _ = librosa.load(str(input_path), sr=target_sr, mono=True)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak * 0.95
    sf.write(str(output_path), audio.astype(np.float32), target_sr, subtype="PCM_16")
