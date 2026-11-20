"""Synthetic signals for the vehicle tests (no real audio, no downloads)."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt


def speech_like(
    sr: int, duration: float, *, seed: int = 0, level: float = 0.1, pause_prob: float = 0.25
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Syllable-like bursts: formant-shaped harmonic complexes plus some frication,
    separated by short gaps and longer pauses. Returns (audio, syllable intervals in s)."""
    rng = np.random.default_rng(seed)
    n = int(sr * duration)
    out = np.zeros(n)
    intervals: list[tuple[float, float]] = []
    t = 0.3
    hp = butter(4, 2000, btype="high", fs=sr, output="sos")
    while True:
        dur = rng.uniform(0.08, 0.25)
        if t + dur > duration - 0.2:
            break
        a, m = int(t * sr), int(dur * sr)
        tt = np.arange(m) / sr
        f0 = rng.uniform(100, 160) * (1 + rng.uniform(-0.1, 0.1) * tt / dur)
        phase = 2 * np.pi * np.cumsum(f0) / sr
        f1, f2 = rng.uniform(300, 800), rng.uniform(900, 2500)
        syl = np.zeros(m)
        for k in range(1, int(3500 / f0.max()) + 1):
            fk = k * f0.mean()
            amp = (1 / (1 + ((fk - f1) / 150) ** 2) + 0.5 / (1 + ((fk - f2) / 250) ** 2)) / np.sqrt(k)
            syl += amp * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
        if rng.random() < 0.3:
            syl += 0.3 * np.std(syl) * sosfilt(hp, rng.standard_normal(m))
        out[a:a + m] += syl * np.hanning(m)
        intervals.append((t, t + dur))
        t += dur + rng.uniform(0.03, 0.1)
        if rng.random() < pause_prob:
            t += rng.uniform(0.3, 0.8)
    act = out[np.abs(out) > 0]
    out *= level / np.sqrt(np.mean(act**2))
    return out, intervals


def car_noise(
    sr: int, duration: float, *, seed: int = 1, level: float = 0.01,
    engine_f0: float | tuple[float, float] | None = None, engine_level: float = 0.0,
) -> np.ndarray:
    """Low-frequency-heavy broadband noise, optionally with engine harmonics
    (constant f0 or a linear glide (f_start, f_end))."""
    rng = np.random.default_rng(seed)
    n = int(sr * duration)
    w = rng.standard_normal(n)
    x = sosfilt(butter(1, 200, btype="low", fs=sr, output="sos"), w) * 3 + 0.3 * w
    x *= level / np.std(x)
    if engine_f0 is not None and engine_level > 0:
        f_a, f_b = engine_f0 if isinstance(engine_f0, tuple) else (engine_f0, engine_f0)
        f = np.linspace(f_a, f_b, n)
        phase = 2 * np.pi * np.cumsum(f) / sr
        for k in range(1, 6):
            x += engine_level / k * np.sin(k * phase + k)
    return x


def synthetic_ir(sr: int, *, length_s: float = 0.2, direct_index: int = 40, rt_s: float = 0.04,
                 seed: int = 2) -> np.ndarray:
    """Delay + direct path (1.0) + a few strong early reflections + exponential diffuse tail."""
    rng = np.random.default_rng(seed)
    h = np.zeros(int(length_s * sr))
    h[direct_index] = 1.0
    for d, g in ((int(0.003 * sr), 0.5), (int(0.0045 * sr), -0.35), (int(0.007 * sr), 0.25)):
        h[direct_index + d] += g
    start = direct_index + int(0.008 * sr)
    m = np.arange(len(h) - start)
    h[start:] += 0.15 * rng.standard_normal(len(m)) * np.exp(-6.9 * m / (rt_s * sr))
    return h
