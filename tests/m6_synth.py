# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Synthetic signals for the M6 cleaning tests: speech-like voice, engine
harmonics, coloured car noise, cabin impulse responses, music-like reference.

Everything is seeded; nothing is downloaded. These are not realistic speech,
only signals with the properties the stages rely on (harmonic voiced sounds
with formants and syllabic rhythm, low-frequency engine orders, broadband
noise tilted towards low frequencies, strong cabin reverberation).
"""
from __future__ import annotations

import numpy as np
from scipy import signal

SR = 16000

# (F1, F2, F3) in Hz for a few Italian-like vowels.
VOWELS = [(750, 1300, 2500), (500, 1900, 2600), (300, 2300, 3000),
          (500, 900, 2400), (320, 800, 2300)]


def speech_like(
    duration: float,
    segments: list[tuple[float, float]],
    *,
    seed: int,
    sr: int = SR,
    f0_range: tuple[float, float] = (105.0, 140.0),
    level: float = 0.1,
) -> np.ndarray:
    """Voiced syllables (harmonic source shaped by vowel formants) inside the
    given segments, silence elsewhere."""
    rng = np.random.default_rng(seed)
    out = np.zeros(int(duration * sr))
    for seg_start, seg_end in segments:
        t0 = seg_start
        while t0 < seg_end - 0.08:
            syl = min(rng.uniform(0.15, 0.28), seg_end - t0)
            n0, n = int(t0 * sr), int(syl * sr)
            f_start = rng.uniform(*f0_range)
            f0 = np.linspace(f_start, f_start * rng.uniform(0.92, 1.08), n)
            phase = 2 * np.pi * np.cumsum(f0) / sr
            formants = VOWELS[rng.integers(len(VOWELS))]
            x = np.zeros(n)
            for k in range(1, int(3800 / f_start) + 1):
                fk = k * f_start
                amp = sum(np.exp(-0.5 * ((fk - fm) / (60 + 0.06 * fm)) ** 2) for fm in formants)
                amp = (amp + 0.03) / np.sqrt(k)
                x += amp * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
            edge = max(1, int(0.03 * sr))
            env = np.ones(n)
            ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, edge))
            env[:edge] = ramp
            env[-edge:] = ramp[::-1]
            out[n0:n0 + n] += x * env
            t0 += syl + rng.uniform(0.0, 0.04)
    peak_rms = np.sqrt(np.mean(out[np.abs(out) > 0] ** 2)) if np.any(out) else 1.0
    return out / peak_rms * level


def engine_noise(
    duration: float,
    *,
    seed: int,
    sr: int = SR,
    f0_start: float = 30.0,
    f0_end: float = 40.0,
    n_harmonics: int = 16,
    level: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic engine noise with a slowly varying fundamental (smooth sweep)
    and slow amplitude modulation. Returns (signal, f0 per sample)."""
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    t = np.arange(n) / sr
    f0 = f0_start + (f0_end - f0_start) * (0.5 - 0.5 * np.cos(np.pi * t / duration))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    x = np.zeros(n)
    for k in range(1, n_harmonics + 1):
        amp = k ** -0.6 * 10 ** (rng.uniform(-3, 3) / 20)
        am = 1 + 0.15 * np.sin(2 * np.pi * rng.uniform(0.2, 0.8) * t + rng.uniform(0, 2 * np.pi))
        x += amp * am * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
    return x / np.sqrt(np.mean(x ** 2)) * level, f0


def colored_noise(duration: float, *, seed: int, sr: int = SR, slope_db_per_octave: float = -4.5,
                  level: float = 0.05) -> np.ndarray:
    """Broadband noise whose power falls by ``slope_db_per_octave`` (car-like
    road/wind noise when negative)."""
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / sr)
    f[0] = f[1]
    spec *= (f / 100.0) ** (slope_db_per_octave / (20 * np.log10(2)))
    x = np.fft.irfft(spec, n)
    return x / np.sqrt(np.mean(x ** 2)) * level


def cabin_rir(*, seed: int, sr: int = SR, length_s: float = 0.1,
              rt60_s: float = 0.08) -> np.ndarray:
    """Direct path plus an exponentially decaying random tail whose energy is
    close to the direct sound (strong cabin reverberation)."""
    rng = np.random.default_rng(seed)
    n = int(length_s * sr)
    t = np.arange(n) / sr
    tail = rng.standard_normal(n) * np.exp(-6.9 * t / rt60_s)
    tail[: int(0.001 * sr)] = 0.0
    tail *= np.sqrt(0.8 / np.sum(tail ** 2))
    h = tail
    h[0] = 1.0
    return h


def music_like(duration: float, *, seed: int, sr: int = SR, level: float = 0.1) -> np.ndarray:
    """Broadband music-like reference: chords of harmonic notes changing every
    ~0.4 s plus a filtered noise bed (percussion/strings stand-in)."""
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    t = np.arange(n) / sr
    x = np.zeros(n)
    note_len = int(0.4 * sr)
    for start in range(0, n, note_len):
        seg = slice(start, min(n, start + note_len))
        for _ in range(3):
            f = 110 * 2 ** (rng.integers(0, 36) / 12)
            for k in range(1, 8):
                if k * f < sr / 2 - 500:
                    x[seg] += np.sin(2 * np.pi * k * f * t[seg] + rng.uniform(0, 6.3)) / k
    sos = signal.butter(2, [200, 6000], btype="band", fs=sr, output="sos")
    x = x / np.sqrt(np.mean(x ** 2)) + 0.7 * signal.sosfilt(sos, rng.standard_normal(n))
    return x / np.sqrt(np.mean(x ** 2)) * level


def assert_loggable(params: dict, name: str = "stage") -> None:
    """Params must be plain JSON and fit in the processing log as-is."""
    import json
    from datetime import datetime, timezone

    from akoe.config import Mode, StageKind
    from akoe.models import StageLog

    json.dumps(params, allow_nan=True)
    now = datetime.now(timezone.utc)
    log = StageLog(name=name, kind=StageKind.DETERMINISTIC, mode=Mode.FORENSIC, params=params,
                   started_at=now, finished_at=now)
    assert json.loads(log.model_dump_json())["params"].keys() == params.keys()


def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    err = estimate - reference
    return float(10 * np.log10(np.sum(reference ** 2) / max(np.sum(err ** 2), 1e-30)))


def segments_to_frame_truth(segments, n_frames: int, hop_s: float) -> np.ndarray:
    t = (np.arange(n_frames) + 0.5) * hop_s
    truth = np.zeros(n_frames, dtype=bool)
    for a, b in segments:
        truth |= (t >= a) & (t < b)
    return truth
