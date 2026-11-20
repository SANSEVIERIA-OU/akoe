# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Shared helpers for the car-specific cleaning stages (M6).

Conventions used by every M6 stage:

- Array functions take mono float audio and a sample rate and return
  ``(audio, params)``; ``params`` is a JSON-serialisable dict with every
  setting and derived quantity needed to reproduce the result (SWGDE
  20-A-001-2.0). File wrappers (``run``) read, process, write and return the
  same ``params`` plus the output write details.
- Deterministic stages are bit-reproducible: same input, same parameters and
  same library versions give the same output samples. No timestamps or random
  state without an explicit seed go into ``params``.
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import soundfile as sf
from scipy import signal

Segments = list[tuple[float, float]]


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    """Read an audio file as mono float64 (channels averaged)."""
    audio, sr = sf.read(str(path), always_2d=False, dtype="float64")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float64), int(sr)


def write_mono(path: Path, audio: np.ndarray, sr: int, *,
               subtype: str = "PCM_16") -> dict[str, Any]:
    """Write mono audio, clipping explicitly to [-1, 1] and reporting how many
    samples were clipped (never silently wrapped)."""
    a = np.asarray(audio, dtype=np.float64)
    n_clipped = int(np.count_nonzero(np.abs(a) > 1.0))
    sf.write(str(path), np.clip(a, -1.0, 1.0).astype(np.float32), sr, subtype=subtype)
    return {"output_subtype": subtype, "n_clipped_samples": n_clipped}


def as_float(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError("expected mono audio (1-D array)")
    return a


def stft(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Hann STFT, shape (n_fft // 2 + 1, n_frames). Frame ``t`` is centred on
    sample ``t * hop``."""
    _, _, z = signal.stft(x, nperseg=n_fft, noverlap=n_fft - hop, window="hann",
                          boundary="zeros", padded=True)
    return z


def istft(z: np.ndarray, n_fft: int, hop: int, length: int) -> np.ndarray:
    _, y = signal.istft(z, nperseg=n_fft, noverlap=n_fft - hop, window="hann", boundary=True)
    y = np.asarray(y, dtype=np.float64)
    if len(y) < length:
        y = np.pad(y, (0, length - len(y)))
    return y[:length]


def frame_mask(segments: Segments | None, frame_times_s: np.ndarray,
               margin_s: float = 0.0) -> np.ndarray:
    """True for frames whose centre lies inside a segment widened by ``margin_s``."""
    mask = np.zeros(len(frame_times_s), dtype=bool)
    for start, end in segments or []:
        mask |= (frame_times_s >= start - margin_s) & (frame_times_s <= end + margin_s)
    return mask


def sample_mask(segments: Segments | None, n: int, sr: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for start, end in segments or []:
        a = max(0, int(round(start * sr)))
        b = min(n, int(round(end * sr)))
        if b > a:
            mask[a:b] = True
    return mask


def array_sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def db(power: float | np.ndarray, floor: float = 1e-20) -> float | np.ndarray:
    return 10.0 * np.log10(np.maximum(power, floor))


def single_thread_blas():
    """Context manager running BLAS on one thread: fixed summation order, so
    results do not depend on the machine's core count (multi-threaded
    OpenBLAS changes the last bits), and no thread contention on the small
    matrices used here. ``threadpoolctl`` ships with scikit-learn."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover
        return contextlib.nullcontext()
    return threadpool_limits(limits=1, user_api="blas")


def with_single_thread_blas(fn):
    """Decorator form of :func:`single_thread_blas`."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with single_thread_blas():
            return fn(*args, **kwargs)

    return wrapper


def versions() -> dict[str, str]:
    """Library versions that affect numerical results."""
    return {"numpy": np.__version__, "scipy": scipy.__version__}


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays and non-string keys so that the
    params dict can go straight into ``StageLog.params``."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "value") and isinstance(getattr(obj, "value"), str):  # Enum
        return obj.value
    return obj


def segments_to_list(segments: Segments | None) -> list[list[float]]:
    return [[round(float(s), 4), round(float(e), 4)] for s, e in segments or []]
