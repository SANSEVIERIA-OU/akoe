from pathlib import Path

import noisereduce as nr
import numpy as np
import soundfile as sf


def run_deterministic(input_path: Path, output_path: Path, *, stationary: bool = True) -> None:
    """Spectral-gating noise reduction. Reproducible from parameters + library version."""
    audio, sr = sf.read(str(input_path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    reduced = nr.reduce_noise(y=audio.astype(np.float32), sr=sr, stationary=stationary)
    sf.write(str(output_path), reduced.astype(np.float32), sr, subtype="PCM_16")


def run_neural(input_path: Path, output_path: Path) -> None:
    """DeepFilterNet. Neural enhancement — investigative mode only."""
    raise NotImplementedError("Install [ml] extras and integrate DeepFilterNet here.")
