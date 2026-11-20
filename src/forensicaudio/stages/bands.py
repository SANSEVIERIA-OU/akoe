"""Band shaping motivated by the Speech Intelligibility Index (M6, issue #29).

Deterministic, zero-phase octave-band equalisation. The gain of each octave
band is the sum of up to three terms, all in dB and all logged:

1. **SII importance** (ANSI S3.5-1997, octave-band procedure): bands that
   carry little intelligibility are attenuated relative to the most important
   one, ``sii_strength * 10*log10(w_b / max(w))``. The 1-4 kHz octaves carry
   about 72% of the importance; cutting below 500 Hz costs little
   (MASTER_PLAN §3.3).
2. **Subject long-term spectrum** (optional): ``subject_strength * (L_b -
   max(L))`` where ``L`` is the subject's long-term average spectrum in dB
   (from ``speaker/timbre.py``, M1). Bands where the subject puts little
   energy carry mostly noise for this speaker.
3. **SII band audibility** (optional): with a per-band SNR measured on the
   case recording, ``A_b = clip((SNR_b + 15) / 30, 0, 1)`` as in the SII;
   ``audibility_strength * 20*log10(max(A_b, floor))``. Bands buried in noise
   (A_b ~ 0) are attenuated, since they add noise without adding speech.

The summed gain is clamped to ``[-max_cut_db, +max_boost_db]`` and, by
default, offset so that the largest gain is 0 dB (no boost, no new
clipping). The curve is interpolated on a log-frequency axis, realised as a
linear-phase FIR (window method) applied centred (zero phase), plus an
optional zero-phase Butterworth high-pass (``sosfiltfilt``) for engine
rumble. Both filters are fully described in the returned parameters.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

from forensicaudio.config import StageKind
from forensicaudio.stages.common import (
    Segments,
    as_float,
    load_mono,
    sample_mask,
    to_jsonable,
    versions,
    write_mono,
)
from forensicaudio.utils import sha256_file

KIND = StageKind.DETERMINISTIC
ALGORITHM_VERSION = "1"

# ANSI S3.5-1997, octave-band procedure: band-importance function for average speech.
SII_OCTAVE_WEIGHTS: dict[int, float] = {
    250: 0.0617, 500: 0.1671, 1000: 0.2373, 2000: 0.2648, 4000: 0.2142, 8000: 0.0549,
}


def _centers() -> np.ndarray:
    return np.array(sorted(SII_OCTAVE_WEIGHTS), dtype=np.float64)


def _interp_log_f(points: dict[float, float], f: np.ndarray) -> np.ndarray:
    """Interpolate dB values given at arbitrary frequencies on a log2 axis
    (values held constant outside the given range)."""
    keys = sorted(points)
    vals = np.array([points[k] for k in keys], dtype=np.float64)
    return np.interp(np.log2(np.maximum(f, 1.0)), np.log2(np.array(keys)), vals)


def _norm_dict(d: dict) -> dict[float, float]:
    return {float(k): float(v) for k, v in d.items()}


def octave_band_levels(
    audio: np.ndarray,
    sr: int,
    *,
    segments: Segments | None = None,
    exclude_segments: Segments | None = None,
    nperseg: int = 4096,
) -> dict[str, float]:
    """Octave-band levels in dB (power, arbitrary reference) at the SII centre
    frequencies, from a Welch PSD. ``segments`` restricts the measure to those
    portions (e.g. speech), ``exclude_segments`` removes them (e.g. noise
    measured outside speech). Bands above Nyquist are omitted."""
    x = as_float(audio)
    keep = np.ones(len(x), dtype=bool) if segments is None else sample_mask(segments, len(x), sr)
    if exclude_segments:
        keep &= ~sample_mask(exclude_segments, len(x), sr)
    x = x[keep]
    if len(x) < 16:
        raise ValueError("not enough samples to measure band levels")
    f, psd = signal.welch(x, fs=sr, window="hann", nperseg=min(nperseg, len(x)))
    df = f[1] - f[0]
    out: dict[str, float] = {}
    for fc in _centers():
        lo, hi = fc / np.sqrt(2), min(fc * np.sqrt(2), sr / 2.0)
        if lo >= sr / 2.0:
            continue
        sel = (f >= lo) & (f < hi)
        out[str(int(fc))] = float(10 * np.log10(max(np.sum(psd[sel]) * df, 1e-20)))
    return out


def estimate_band_snr(audio: np.ndarray, sr: int, speech_segments: Segments) -> dict[str, float]:
    """Per-octave SNR on the case recording: speech power estimated as
    (speech-portion power - noise power), noise from the non-speech portions."""
    noisy = octave_band_levels(audio, sr, segments=speech_segments)
    noise = octave_band_levels(audio, sr, exclude_segments=speech_segments)
    snr: dict[str, float] = {}
    for k in noisy:
        p_s = 10 ** (noisy[k] / 10) - 10 ** (noise[k] / 10)
        snr[k] = float(10 * np.log10(max(p_s, 1e-20)) - noise[k])
    return snr


def design_band_gains(
    *,
    sii_strength: float = 0.5,
    subject_ltas: dict | None = None,
    subject_strength: float = 0.5,
    band_snr_db: dict | None = None,
    audibility_strength: float = 0.5,
    max_cut_db: float = 18.0,
    max_boost_db: float = 6.0,
    normalize_max_gain: bool = True,
) -> dict[str, Any]:
    """Per-octave target gains (dB) with each contributing term, for the log."""
    centers = _centers()
    w = np.array([SII_OCTAVE_WEIGHTS[int(c)] for c in centers])
    terms: dict[str, np.ndarray] = {"sii": sii_strength * 10 * np.log10(w / w.max())}
    if subject_ltas:
        lvl = _interp_log_f(_norm_dict(subject_ltas), centers)
        terms["subject"] = subject_strength * np.maximum(lvl - lvl.max(), -max_cut_db)
    if band_snr_db:
        snr = _interp_log_f(_norm_dict(band_snr_db), centers)
        audibility = np.clip((snr + 15.0) / 30.0, 0.0, 1.0)
        floor = 10 ** (-max_cut_db / 20)
        terms["audibility"] = audibility_strength * 20 * np.log10(np.maximum(audibility, floor))
    total = np.clip(sum(terms.values()), -max_cut_db, max_boost_db)
    offset = -float(total.max()) if normalize_max_gain else 0.0
    total = np.clip(total + offset, -max_cut_db, max_boost_db)
    key = [str(int(c)) for c in centers]
    return {
        "centers_hz": [int(c) for c in centers],
        "terms_db": {name: dict(zip(key, vals.round(4).tolist())) for name, vals in terms.items()},
        "normalization_offset_db": round(offset, 4),
        "gain_db": dict(zip(key, total.round(4).tolist())),
    }


def shape(
    audio: np.ndarray,
    sr: int,
    *,
    sii_strength: float = 0.5,
    subject_ltas: dict | None = None,
    subject_strength: float = 0.5,
    band_snr_db: dict | None = None,
    audibility_strength: float = 0.5,
    max_cut_db: float = 18.0,
    max_boost_db: float = 6.0,
    normalize_max_gain: bool = True,
    fir_numtaps: int = 1025,
    fir_window: str = "hann",
    highpass_hz: float | None = 80.0,
    highpass_order: int = 4,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply SII-motivated band shaping. Returns ``(audio, params)``."""
    x = as_float(audio)
    if fir_numtaps % 2 == 0:
        raise ValueError("fir_numtaps must be odd (type I, exactly centred)")
    design = design_band_gains(
        sii_strength=sii_strength, subject_ltas=subject_ltas, subject_strength=subject_strength,
        band_snr_db=band_snr_db, audibility_strength=audibility_strength,
        max_cut_db=max_cut_db, max_boost_db=max_boost_db, normalize_max_gain=normalize_max_gain,
    )
    nyq = sr / 2.0
    grid = np.linspace(0.0, nyq, 1025)
    gains_db = _interp_log_f({float(k): v for k, v in design["gain_db"].items()}, grid)
    taps = signal.firwin2(fir_numtaps, grid, 10 ** (gains_db / 20), fs=sr, window=fir_window)
    # Symmetric odd-length FIR, output centred on the input: zero phase.
    y = signal.oaconvolve(x, taps, mode="same") if len(x) else x.copy()

    sos = None
    if highpass_hz:
        sos = signal.butter(highpass_order, highpass_hz, btype="highpass", fs=sr, output="sos")
        padlen = min(3 * (2 * len(sos) + 1), max(len(y) - 1, 0))
        if len(y) > 1:
            y = signal.sosfiltfilt(sos, y, padlen=padlen)

    eval_f = np.array([float(c) for c in design["centers_hz"] if c < nyq] + [63.0, 125.0])
    _, h_fir = signal.freqz(taps, worN=eval_f, fs=sr)
    resp = np.abs(h_fir) ** 2
    if sos is not None:
        _, h_hp = signal.sosfreqz(sos, worN=eval_f, fs=sr)
        resp = resp * np.abs(h_hp) ** 4  # sosfiltfilt squares the magnitude
    realised = {str(int(f)): round(float(10 * np.log10(max(r, 1e-30))), 3)
                for f, r in zip(eval_f, resp)}

    params = {
        "algorithm": "sii_octave_band_shaping",
        "algorithm_version": ALGORITHM_VERSION,
        "kind": KIND.value,
        "sr": sr,
        "sii_weights": {str(k): v for k, v in SII_OCTAVE_WEIGHTS.items()},
        "sii_reference": "ANSI S3.5-1997 octave-band importance (average speech)",
        "sii_strength": sii_strength,
        "subject_ltas_db": subject_ltas,
        "subject_strength": subject_strength,
        "band_snr_db": band_snr_db,
        "audibility_strength": audibility_strength,
        "max_cut_db": max_cut_db,
        "max_boost_db": max_boost_db,
        "normalize_max_gain": normalize_max_gain,
        **design,
        "interpolation": "linear in dB on log2(frequency), held outside 250-8000 Hz",
        "fir": {"design": "scipy.signal.firwin2", "numtaps": fir_numtaps, "window": fir_window,
                "application": "centred convolution (linear phase, zero delay)"},
        "highpass": None if sos is None else {
            "type": "butterworth", "cutoff_hz": highpass_hz, "order": highpass_order,
            "application": "scipy.signal.sosfiltfilt (zero phase, effective order doubled)"},
        "zero_phase": True,
        "realised_gain_db": realised,
        "versions": versions(),
    }
    return y, to_jsonable(params)


def run(input_path: Path, output_path: Path, *, subtype: str = "PCM_16",
        **kwargs) -> dict[str, Any]:
    """File wrapper for :func:`shape`. Returns the params for the log."""
    audio, sr = load_mono(input_path)
    y, params = shape(audio, sr, **kwargs)
    params.update(write_mono(output_path, y, sr, subtype=subtype))
    params["input_sha256"] = sha256_file(input_path)
    params["output_sha256"] = sha256_file(output_path)
    return params
