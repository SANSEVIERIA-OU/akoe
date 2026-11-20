"""Audio helpers shared by the vehicle modules: I/O, resampling, levels, lags.

Audio handed between modules is mono float32 in [-1, 1] (interfaces.py
convention); computations run in float64. Every helper is deterministic.
"""
from __future__ import annotations

import platform
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

from forensicaudio.utils import sha256_file
from forensicaudio.vehicle import ffmpeg

EPS = 1e-12


def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x if x.ndim == 1 else x.mean(axis=1)


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Polyphase resampling with an exact rational ratio (deterministic)."""
    x = np.asarray(x, dtype=np.float64)
    if int(sr_from) == int(sr_to):
        return x.copy()
    ratio = Fraction(int(sr_to), int(sr_from))
    return resample_poly(x, ratio.numerator, ratio.denominator)


def pad_or_trim(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.concatenate([x, np.zeros(n - len(x), dtype=x.dtype)])


def shift(x: np.ndarray, lag: int) -> np.ndarray:
    """Advance `x` by `lag` samples (lag > 0 removes leading samples), same length."""
    n = len(x)
    if lag > 0:
        return pad_or_trim(x[lag:], n)
    if lag < 0:
        return np.concatenate([np.zeros(-lag, dtype=x.dtype), x])[:n]
    return x.copy()


def load_audio(path: Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load a file as mono float32, optionally resampled to `sr`.

    soundfile first; ffmpeg as fallback for containers libsndfile cannot read
    (AMR, AAC/M4A, 3GP...).
    """
    path = Path(path)
    try:
        data, native = sf.read(str(path), dtype="float64", always_2d=False)
        audio = to_mono(data)
    except (sf.LibsndfileError, RuntimeError, TypeError):
        decoded, native = ffmpeg.decode_file(path)
        audio = decoded.astype(np.float64)
    if sr is not None and sr != native:
        audio = resample(audio, native, sr)
        native = sr
    return audio.astype(np.float32), int(native)


def write_audio(path: Path, audio: np.ndarray, sr: int, subtype: str = "PCM_16") -> str:
    """Write mono audio (clipped to [-1, 1] unless FLOAT/DOUBLE) and return its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(audio, dtype=np.float64)
    if subtype not in ("FLOAT", "DOUBLE"):
        data = np.clip(data, -1.0, 1.0)
    sf.write(str(path), data.astype(np.float32), int(sr), subtype=subtype)
    return sha256_file(path)


def power(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean(x * x)) if x.size else 0.0


def db(p: float) -> float:
    return float(10.0 * np.log10(max(p, EPS)))


def frame_powers(x: np.ndarray, frame: int, hop: int | None = None) -> np.ndarray:
    """Mean power of consecutive frames (hop defaults to frame)."""
    x = np.asarray(x, dtype=np.float64)
    hop = hop or frame
    if len(x) < frame:
        return np.array([power(x)]) if len(x) else np.zeros(0)
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return np.mean(x[idx] ** 2, axis=1)


def active_power(
    x: np.ndarray, sr: int, *, frame_s: float = 0.02, dynamic_range_db: float = 40.0
) -> float:
    """Approximate active level: mean power of frames within `dynamic_range_db`
    of the 99th-percentile frame (a simplified stand-in for ITU-T P.56)."""
    p = frame_powers(x, max(1, int(sr * frame_s)))
    if p.size == 0 or float(p.max()) <= EPS:
        return 0.0
    ref = float(np.percentile(p, 99))
    active = p[p >= ref * 10 ** (-dynamic_range_db / 10)]
    return float(active.mean()) if active.size else power(x)


def estimate_lag(ref: np.ndarray, sig: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Lag (samples) such that sig[n] ~ ref[n - lag], searched in [-max_lag, max_lag].

    Returns (lag, normalized correlation at the peak). Correlation is 0 when
    either signal is silent.
    """
    ref = np.asarray(ref, dtype=np.float64)
    sig = np.asarray(sig, dtype=np.float64)
    e = float(np.sqrt(np.sum(ref**2) * np.sum(sig**2)))
    if e <= EPS or not len(ref) or not len(sig):
        return 0, 0.0
    c = fftconvolve(sig, ref[::-1], mode="full")
    zero = len(ref) - 1
    lo, hi = max(0, zero - max_lag), min(len(c), zero + max_lag + 1)
    k = lo + int(np.argmax(c[lo:hi]))
    return k - zero, float(c[k] / e)


def environment_info() -> dict[str, Any]:
    """Library versions, logged with every deterministic output (SWGDE 20-A-001)."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "soundfile": sf.__version__,
        "libsndfile": sf.__libsndfile_version__,
        "ffmpeg": ffmpeg.version(),
    }
