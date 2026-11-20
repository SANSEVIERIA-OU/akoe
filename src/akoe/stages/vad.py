# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
from pathlib import Path

import numpy as np
import soundfile as sf

from akoe.config import StageKind

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


# --------------------------------------------------------------------------
# Robust VAD for car noise (M6, issue #35). The energy VAD above stays
# available and unchanged.
# --------------------------------------------------------------------------

ROBUST_VAD_VERSION = "1"


def detect_speech(
    audio: np.ndarray,
    sr: int,
    *,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
    band_hz: tuple[float, float] = (300.0, 3400.0),
    floor_window_s: float = 5.0,
    floor_percentile: float = 10.0,
    snr_threshold_db: float = 3.0,
    whitening_window_s: float = 2.0,
    flatness_threshold: float = 0.25,
    flatness_smoothing_frames: int = 5,
    preroll_ms: float = 50.0,
    hangover_ms: float = 200.0,
    min_silence_s: float = 0.15,
    min_speech_s: float = 0.2,
) -> tuple[list[tuple[float, float]], dict]:
    """Deterministic VAD robust to car noise. Returns ``(segments, params)``.

    A frame is speech when both hold:

    1. its 300-3400 Hz energy exceeds an adaptive noise floor (running low
       percentile over ``floor_window_s``) by ``snr_threshold_db``; the band
       limit discards most engine and road rumble;
    2. the spectral flatness of the band, after whitening each bin by a
       minimum-statistics noise estimate, is below ``flatness_threshold``.
       Noise of any colour whitens to a flat spectrum (flatness about 0.56 for
       a single periodogram), voiced speech stays peaky (harmonics, formants).
       This rejects loud broadband events (road surface, wind gusts) that
       pass the energy test.

    Decisions are then smoothed: pre-roll before onsets, hangover after
    offsets, gaps shorter than ``min_silence_s`` filled, segments shorter than
    ``min_speech_s`` dropped. Unvoiced consonants rely on pre-roll/hangover.

    Defaults were set on synthetic material and must be checked on real case
    material. Known limit: a loud broadband event whose spectrum is very
    different from the running floor can whiten to a non-flat spectrum and
    pass as speech.
    """
    from scipy import ndimage

    from akoe.stages.common import as_float, to_jsonable, versions

    x = as_float(audio)
    frame_len = int(round(sr * frame_ms / 1000.0))
    hop = int(round(sr * hop_ms / 1000.0))
    hop_s = hop / sr
    params: dict = {
        "algorithm": "band_energy_adaptive_floor+whitened_flatness+hangover",
        "algorithm_version": ROBUST_VAD_VERSION,
        "sr": sr, "frame_ms": frame_ms, "hop_ms": hop_ms, "band_hz": list(band_hz),
        "floor_window_s": floor_window_s, "floor_percentile": floor_percentile,
        "snr_threshold_db": snr_threshold_db, "whitening_window_s": whitening_window_s,
        "flatness_threshold": flatness_threshold,
        "flatness_smoothing_frames": flatness_smoothing_frames,
        "preroll_ms": preroll_ms, "hangover_ms": hangover_ms,
        "min_silence_s": min_silence_s, "min_speech_s": min_speech_s,
        "window": "hann", "versions": versions(),
    }
    if frame_len < 8 or hop < 1 or len(x) < frame_len:
        params.update(n_frames=0, n_segments=0, speech_ratio=0.0, segments=[])
        return [], to_jsonable(params)

    n_frames = 1 + (len(x) - frame_len) // hop
    frames = np.lib.stride_tricks.sliding_window_view(x, frame_len)[::hop][:n_frames]
    spec = np.fft.rfft(frames * np.hanning(frame_len), axis=1)
    power = spec.real ** 2 + spec.imag ** 2
    freqs = np.fft.rfftfreq(frame_len, 1.0 / sr)
    band = (freqs >= band_hz[0]) & (freqs <= min(band_hz[1], sr / 2.0))
    p_band = power[:, band] + 1e-20

    # 1. Band energy against an adaptive floor.
    energy_db = 10.0 * np.log10(p_band.sum(axis=1))
    energy_db = ndimage.uniform_filter1d(energy_db, 3, mode="nearest")
    floor_frames = max(3, int(round(floor_window_s / hop_s)) | 1)
    floor_db = ndimage.percentile_filter(energy_db, floor_percentile, size=floor_frames,
                                         mode="nearest")
    snr_db = energy_db - floor_db

    # 2. Flatness of the noise-whitened band spectrum.
    smooth_p = ndimage.uniform_filter1d(p_band, 5, axis=0, mode="nearest")
    white_frames = max(3, int(round(whitening_window_s / hop_s)) | 1)
    noise_bins = ndimage.minimum_filter1d(smooth_p, white_frames, axis=0, mode="nearest")
    noise_bins = ndimage.uniform_filter1d(noise_bins, white_frames, axis=0, mode="nearest")
    ratio = p_band / (noise_bins + 1e-20)
    flatness = np.exp(np.mean(np.log(ratio), axis=1)) / np.mean(ratio, axis=1)
    if flatness_smoothing_frames > 1:
        flatness = ndimage.median_filter(flatness, flatness_smoothing_frames, mode="nearest")

    active = (snr_db > snr_threshold_db) & (flatness < flatness_threshold)

    # 3. Smoothing: a frame is kept if an active frame lies within
    # [-hangover, +preroll] of it; then gap filling and minimum duration.
    pre = int(round(preroll_ms / 1000.0 / hop_s))
    hang = int(round(hangover_ms / 1000.0 / hop_s))
    csum = np.concatenate([[0], np.cumsum(active.astype(np.int64))])
    i = np.arange(n_frames)
    lo = np.clip(i - hang, 0, n_frames)
    hi = np.clip(i + pre + 1, 0, n_frames)
    active = (csum[hi] - csum[lo]) > 0

    frame_start = np.arange(n_frames) * hop / sr
    segments: list[tuple[float, float]] = []
    edges = np.flatnonzero(np.diff(np.concatenate([[0], active.astype(np.int8), [0]])))
    for a, b in zip(edges[::2], edges[1::2]):
        start = float(frame_start[a])
        end = float(min(frame_start[b - 1] + frame_len / sr, len(x) / sr))
        if segments and start - segments[-1][1] < min_silence_s:
            segments[-1] = (segments[-1][0], end)
        else:
            segments.append((start, end))
    segments = [(round(s, 4), round(e, 4)) for s, e in segments if e - s >= min_speech_s]

    total = len(x) / sr
    params.update(
        n_frames=int(n_frames),
        n_segments=len(segments),
        speech_ratio=float(sum(e - s for s, e in segments) / total) if total > 0 else 0.0,
        noise_floor_db_median=float(np.median(floor_db)),
        flatness_median=float(np.median(flatness)),
        segments=[[s, e] for s, e in segments],
    )
    return segments, to_jsonable(params)


def run_robust(input_path: Path, **kwargs) -> tuple[list[tuple[float, float]], dict]:
    """File wrapper for :func:`detect_speech`. Returns ``(segments, params)``."""
    from akoe.stages.common import load_mono
    from akoe.utils import sha256_file

    audio, sr = load_mono(input_path)
    segments, params = detect_speech(audio, sr, **kwargs)
    params["input_sha256"] = sha256_file(input_path)
    return segments, params
