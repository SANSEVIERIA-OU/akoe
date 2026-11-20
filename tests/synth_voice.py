"""Synthetic voice-like signals for speaker tests (no real speech, no downloads).

A "voice" is a glottal impulse train with slow F0 modulation, filtered by a
cascade of formant resonators, gated by a syllable-rate envelope (so that
there are pauses) and mixed with low-level noise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.signal as ss
import soundfile as sf

SR = 16000

SPEAKER_A = {"f0": 120.0, "formants": [(700.0, 80.0), (1220.0, 90.0), (2600.0, 120.0)]}
SPEAKER_B = {"f0": 210.0, "formants": [(400.0, 70.0), (2000.0, 100.0), (3000.0, 150.0)]}


def _resonator(f: float, bw: float, sr: int):
    r = np.exp(-np.pi * bw / sr)
    th = 2 * np.pi * f / sr
    return [1 - r], [1.0, -2 * r * np.cos(th), r * r]


def voice(
    f0: float,
    formants: list[tuple[float, float]],
    *,
    seconds: float = 3.0,
    sr: int = SR,
    seed: int = 0,
    noise: float = 0.003,
    level: float = 0.5,
    syllable_hz: float = 3.0,
    phase: float = np.pi / 2,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    t = np.arange(n) / sr
    f0t = f0 * (1 + 0.05 * np.sin(2 * np.pi * 0.7 * t + seed))
    src = np.diff(np.floor(np.cumsum(f0t / sr)), prepend=0.0)
    y = src
    for f, bw in formants:
        b, a = _resonator(f, bw, sr)
        y = ss.lfilter(b, a, y)
    # Starts voiced (phase pi/2), ~35% pauses.
    env = (np.sin(2 * np.pi * syllable_hz * t + phase) > -0.45).astype(float)
    k = np.hanning(161)
    env = np.convolve(env, k / k.sum(), mode="same")
    y = y * env
    y = y / (np.max(np.abs(y)) + 1e-12) * level
    return (y + rng.normal(0.0, noise, n)).astype(np.float32)


def speaker_voice(spk: dict, **kw) -> np.ndarray:
    return voice(spk["f0"], spk["formants"], **kw)


def bandlimit(y: np.ndarray, lo: float, hi: float, sr: int = SR) -> np.ndarray:
    sos = ss.butter(6, [lo, hi], btype="bandpass", fs=sr, output="sos")
    return ss.sosfilt(sos, y).astype(np.float32)


def write_wav(path: Path, y: np.ndarray, sr: int = SR, subtype: str = "PCM_16") -> Path:
    sf.write(str(path), y, sr, subtype=subtype)
    return path
