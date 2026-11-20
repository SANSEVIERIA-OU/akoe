from pathlib import Path

import numpy as np
import soundfile as sf

from forensicaudio.config import StageKind

KIND = StageKind.DETERMINISTIC


def run(
    input_path: Path,
    *,
    frame_ms: int = 30,
    threshold_db: float = -40.0,
    min_speech_s: float = 0.3,
) -> list[tuple[float, float]]:
    """Energy-based VAD. Returns (start_s, end_s) tuples of speech regions.

    Simple deterministic detector — replace with pyannote VAD for production.
    """
    audio, sr = sf.read(str(input_path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    frame_len = int(sr * frame_ms / 1000)
    if frame_len == 0:
        return []
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return []

    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)
    active = db > threshold_db

    segments: list[tuple[float, float]] = []
    in_speech = False
    start_frame = 0
    for i, a in enumerate(active):
        if a and not in_speech:
            in_speech = True
            start_frame = i
        elif not a and in_speech:
            in_speech = False
            t_start = start_frame * frame_ms / 1000.0
            t_end = i * frame_ms / 1000.0
            if t_end - t_start >= min_speech_s:
                segments.append((t_start, t_end))
    if in_speech:
        t_start = start_frame * frame_ms / 1000.0
        t_end = n_frames * frame_ms / 1000.0
        if t_end - t_start >= min_speech_s:
            segments.append((t_start, t_end))
    return segments
