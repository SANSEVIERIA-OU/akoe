"""Engine harmonics: order tracking and attenuation (M6, issue #31).

1. **Tracking** (:func:`track_engine_f0`). The recording is decimated to a
   low analysis rate; long frames (default 1 s, hop 0.25 s) give a fine
   low-frequency spectrum. Each frame's log spectrum is whitened (local
   mean over ``envelope_width_hz`` subtracted, negatives set to 0) and a
   harmonic score is computed for every candidate fundamental on a grid::

       score(f0) = mean_k S(k f0) - mean_k S((k - 1/2) f0)

   The negative teeth halfway between harmonics penalise sub-octave errors
   (f0/2 would put them on real harmonics). Frames flagged as speech get
   their score multiplied by ``speech_weight`` so that the voice's own
   harmonics do not capture the track. A Viterbi path with a slope limit
   (``max_slope_hz_per_s``) and a cost per Hz of change gives a slowly
   varying track, refined by parabolic interpolation and lightly smoothed.
   The "fundamental" is the spacing of the engine comb: for a four-stroke
   engine it can be the firing frequency or a half order.

2. **Attenuation** (:func:`attenuate_engine_harmonics`). A time-varying
   comb notch along the track: for each harmonic ``k`` the recording is
   demodulated with ``exp(-j k phi(t))`` (``phi`` = integral of the track),
   low-passed with a zero-phase Butterworth of cutoff ``bandwidth_hz / 2``
   and re-modulated; this narrow-band estimate of the engine component is
   subtracted (fully, or by ``depth_db``). Only ``bandwidth_hz`` around
   each harmonic is affected, so speech energy elsewhere is untouched.
   If the track is weak (no engine comb, the tracker may be following the
   voice) nothing is removed unless forced.

Known limit: an f0 error is multiplied by the harmonic number, so during
fast accelerations (f0 doubling within a few seconds) the high harmonics
fall outside the notch and attenuation drops to roughly 5-15 dB on
synthetic tests, against 25-35 dB at steady or slowly varying speed.

Both steps are deterministic; every setting and the track itself are
returned for the processing log and for display.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage, signal

from forensicaudio.config import StageKind
from forensicaudio.stages.common import (
    Segments,
    array_sha256,
    as_float,
    frame_mask,
    load_mono,
    segments_to_list,
    to_jsonable,
    versions,
    write_mono,
)
from forensicaudio.utils import sha256_file

KIND = StageKind.DETERMINISTIC
ALGORITHM_VERSION = "1"
_CHUNK = 64  # frames per block when scoring candidates (bounds memory)


def _sample(sal: np.ndarray, freqs: np.ndarray, df: float) -> np.ndarray:
    """Linear interpolation of every row of ``sal`` (T, B) at the same
    frequencies ``freqs`` (any shape) -> (T, *freqs.shape)."""
    idx = np.clip(freqs / df, 0, sal.shape[1] - 1.001)
    i0 = idx.astype(np.int64)
    frac = idx - i0
    return sal[:, i0] * (1 - frac) + sal[:, i0 + 1] * frac


def _sample_rows(sal: np.ndarray, freqs: np.ndarray, df: float) -> np.ndarray:
    """Linear interpolation of row ``t`` of ``sal`` at ``freqs[t]`` -> freqs.shape."""
    idx = np.clip(freqs / df, 0, sal.shape[1] - 1.001)
    i0 = idx.astype(np.int64)
    frac = idx - i0
    flat = i0.reshape(len(sal), -1)
    v0 = np.take_along_axis(sal, flat, axis=1).reshape(i0.shape)
    v1 = np.take_along_axis(sal, flat + 1, axis=1).reshape(i0.shape)
    return v0 * (1 - frac) + v1 * frac


@dataclass
class EngineTrack:
    """Estimated engine fundamental over time, for display and reuse."""

    times_s: np.ndarray
    f0_hz: np.ndarray
    salience: np.ndarray  # harmonic score at the chosen f0 (whitened dB)
    speech: np.ndarray  # frames down-weighted because flagged as speech

    def f0_at(self, t: np.ndarray) -> np.ndarray:
        return np.interp(t, self.times_s, self.f0_hz)

    def to_dict(self) -> dict[str, Any]:
        return {
            "times_s": [round(float(v), 4) for v in self.times_s],
            "f0_hz": [round(float(v), 4) for v in self.f0_hz],
            "salience": [round(float(v), 3) for v in self.salience],
            "speech": [bool(v) for v in self.speech],
        }

    def summary(self, prefix: str = "") -> dict[str, Any]:
        out = {
            "n_frames": int(len(self.f0_hz)),
            "f0_min_hz": float(np.min(self.f0_hz)) if len(self.f0_hz) else None,
            "f0_median_hz": float(np.median(self.f0_hz)) if len(self.f0_hz) else None,
            "f0_max_hz": float(np.max(self.f0_hz)) if len(self.f0_hz) else None,
            "salience_median": float(np.median(self.salience)) if len(self.salience) else None,
            "sha256": array_sha256(np.stack([self.times_s, self.f0_hz])),
        }
        return {prefix + k: v for k, v in out.items()}


def track_engine_f0(
    audio: np.ndarray,
    sr: int,
    *,
    speech_segments: Segments | None = None,
    f0_min_hz: float = 15.0,
    f0_max_hz: float = 150.0,
    analysis_max_hz: float = 600.0,
    n_harmonics: int = 10,
    win_s: float = 0.5,
    hop_s: float = 0.125,
    grid_step_hz: float = 0.25,
    fine_step_hz: float = 0.01,
    max_slope_hz_per_s: float = 20.0,
    transition_cost_per_hz: float = 0.5,
    speech_weight: float = 0.25,
    envelope_width_hz: float = 20.0,
    peak_tolerance_hz: float = 1.0,
    smooth_frames: int = 3,
) -> tuple[EngineTrack, dict[str, Any]]:
    """Track the engine fundamental. Returns ``(track, params)``.

    ``speech_segments=None`` runs the robust VAD; pass ``[]`` to disable the
    speech down-weighting.
    """
    from forensicaudio.stages import vad

    x = as_float(audio)
    params: dict[str, Any] = {
        "algorithm": "harmonic_sum_with_subharmonic_penalty+viterbi+fine_refinement",
        "algorithm_version": ALGORITHM_VERSION,
        "sr": sr, "f0_min_hz": f0_min_hz, "f0_max_hz": f0_max_hz,
        "analysis_max_hz": analysis_max_hz, "n_harmonics": n_harmonics, "win_s": win_s,
        "hop_s": hop_s, "grid_step_hz": grid_step_hz, "max_slope_hz_per_s": max_slope_hz_per_s,
        "transition_cost_per_hz": transition_cost_per_hz, "speech_weight": speech_weight,
        "envelope_width_hz": envelope_width_hz, "peak_tolerance_hz": peak_tolerance_hz,
        "smooth_frames": smooth_frames, "window": "hann",
    }
    if speech_segments is None:
        speech_segments, vad_params = vad.detect_speech(x, sr)
        params["speech_segments_source"] = "vad.detect_speech"
        params["vad_params"] = vad_params
    else:
        params["speech_segments_source"] = "provided"
    params["speech_segments"] = segments_to_list(speech_segments)

    # Decimate to an analysis rate just above 2.5 x analysis_max_hz.
    q = max(1, int(sr // (2.5 * analysis_max_hz)))
    xa = signal.resample_poly(x, 1, q) if q > 1 else x
    sra = sr / q
    win = max(16, int(round(win_s * sra)))
    hop = max(1, int(round(hop_s * sra)))
    n_fft = 4 * (1 << int(np.ceil(np.log2(win))))
    padded = np.pad(xa, (win // 2, win // 2))
    n_frames = max(1, 1 + (len(padded) - win) // hop)
    if len(padded) < win:
        padded = np.pad(padded, (0, win - len(padded)))
    frames = np.lib.stride_tricks.sliding_window_view(padded, win)[::hop][:n_frames]
    df = sra / n_fft
    n_bins = int(min(analysis_max_hz, sra / 2) / df) + 2
    spec = np.fft.rfft(frames * np.hanning(win), n=n_fft, axis=1)[:, :n_bins]
    logmag = 10 * np.log10(spec.real ** 2 + spec.imag ** 2 + 1e-20)
    env = ndimage.uniform_filter1d(logmag, max(3, int(envelope_width_hz / df) | 1), axis=1,
                                   mode="nearest")
    sal_raw = np.maximum(logmag - env, 0.0)
    # Dilated salience for the coarse search: tolerant to the grid step.
    sal = ndimage.maximum_filter1d(sal_raw, max(1, int(round(peak_tolerance_hz / df)) | 1), axis=1)
    f_top = min(analysis_max_hz, sra / 2)

    grid = np.arange(f0_min_hz, f0_max_hz + grid_step_hz / 2, grid_step_hz)
    k = np.arange(1, n_harmonics + 1)
    pos_f = grid[:, None] * k[None, :]
    neg_f = grid[:, None] * (k[None, :] - 0.5)
    valid = pos_f <= f_top
    count = np.maximum(valid.sum(axis=1), 1)
    score = np.empty((n_frames, len(grid)))
    for s0 in range(0, n_frames, _CHUNK):
        chunk = sal[s0:s0 + _CHUNK]
        pos = (_sample(chunk, pos_f, df) * valid).sum(axis=2)
        neg = (_sample(chunk, neg_f, df) * valid).sum(axis=2)
        score[s0:s0 + _CHUNK] = (pos - neg) / count
    times = np.arange(n_frames) * hop / sra
    speech = frame_mask(speech_segments, times)
    weighted = np.where(speech[:, None], score * speech_weight, score)

    # Viterbi with a slope limit and a linear cost per Hz of change.
    jump = max(1, int(round(max_slope_hz_per_s * hop_s / grid_step_hz)))
    n_grid = len(grid)
    acc = weighted[0].copy()
    back = np.zeros((n_frames, n_grid), dtype=np.int64)
    for t in range(1, n_frames):
        best = np.full(n_grid, -np.inf)
        arg = np.zeros(n_grid, dtype=np.int64)
        for d in range(-jump, jump + 1):
            src = np.full(n_grid, -np.inf)
            if d >= 0:
                src[: n_grid - d] = acc[d:]
            else:
                src[-d:] = acc[: n_grid + d]
            cand = src - transition_cost_per_hz * abs(d) * grid_step_hz
            better = cand > best
            best[better] = cand[better]
            arg[better] = (np.arange(n_grid) + d)[better]
        acc = best + weighted[t]
        back[t] = arg
    path = np.zeros(n_frames, dtype=np.int64)
    path[-1] = int(np.argmax(acc))
    for t in range(n_frames - 1, 0, -1):
        path[t - 1] = back[t, path[t]]

    # Fine refinement: around the coarse path, on the un-dilated salience and
    # with every harmonic below analysis_max_hz (higher harmonics constrain f0
    # more tightly), then parabolic interpolation and light smoothing. The
    # attenuation needs this precision: an f0 error is multiplied by k.
    rows = np.arange(n_frames)
    coarse = grid[path].astype(np.float64)
    offsets = np.arange(-grid_step_hz, grid_step_hz + fine_step_hz / 2, fine_step_hz)
    cand = coarse[:, None] + offsets[None, :]  # (T, C)
    k_fine = np.arange(1, int(f_top // max(f0_min_hz, 1e-3)) + 1)
    fine = np.empty_like(cand)
    for s0 in range(0, n_frames, _CHUNK):
        freqs = cand[s0:s0 + _CHUNK, :, None] * k_fine[None, None, :]  # (t, C, K)
        ok = freqs <= f_top
        vals = _sample_rows(sal_raw[s0:s0 + _CHUNK], freqs, df)
        fine[s0:s0 + _CHUNK] = (vals * ok).sum(axis=2) / np.maximum(ok.sum(axis=2), 1)
    best = np.argmax(fine, axis=1)
    f0 = cand[rows, best]
    inner = (best > 0) & (best < len(offsets) - 1)
    if inner.any():
        r, g = rows[inner], best[inner]
        a, b, c = fine[r, g - 1], fine[r, g], fine[r, g + 1]
        den = a - 2 * b + c
        delta = np.where(np.abs(den) > 1e-12, 0.5 * (a - c) / np.where(den == 0, 1, den), 0.0)
        f0[inner] += np.clip(delta, -0.5, 0.5) * fine_step_hz
    if smooth_frames > 1:
        f0 = ndimage.uniform_filter1d(f0, smooth_frames, mode="nearest")

    track = EngineTrack(times_s=times, f0_hz=f0, salience=score[rows, path], speech=speech)
    params.update(decimation_factor=q, analysis_sr=sra, n_fft=n_fft, frame_len=win, frame_hop=hop,
                  max_jump_bins=jump, fine_step_hz=fine_step_hz,
                  n_harmonics_fine=int(len(k_fine)),
                  **track.summary(prefix="track_"),
                  versions=versions())
    return track, to_jsonable(params)


def attenuate_engine_harmonics(
    audio: np.ndarray,
    sr: int,
    *,
    track: EngineTrack | None = None,
    speech_segments: Segments | None = None,
    max_freq_hz: float = 1000.0,
    bandwidth_hz: float = 2.0,
    depth_db: float | None = None,
    lpf_order: int = 2,
    min_salience_db: float = 5.0,
    force: bool = False,
    track_options: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any], EngineTrack]:
    """Remove engine harmonics along the track. Returns ``(audio, params, track)``.

    ``depth_db=None`` removes the estimated components fully; a number
    limits the attenuation (e.g. 12 dB). ``track_options`` are passed to
    :func:`track_engine_f0` when ``track`` is not given.

    Safeguard: if the median harmonic salience of the track is below
    ``min_salience_db`` there is probably no engine comb and the tracker may
    be following the voice's own harmonics; the input is then returned
    unchanged (``applied=False``, warning in Italian) unless ``force``.
    """
    x = as_float(audio)
    params: dict[str, Any] = {
        "algorithm": "tracking_comb_notch_synchronous_demodulation",
        "algorithm_version": ALGORITHM_VERSION,
        "kind": KIND.value, "sr": sr, "max_freq_hz": max_freq_hz, "bandwidth_hz": bandwidth_hz,
        "depth_db": depth_db, "lpf": {"type": "butterworth", "order": lpf_order,
                                      "cutoff_hz": bandwidth_hz / 2,
                                      "application": "scipy.signal.sosfiltfilt (zero phase)"},
        "min_salience_db": min_salience_db, "force": force,
    }
    if track is None:
        track, tparams = track_engine_f0(x, sr, speech_segments=speech_segments,
                                         **(track_options or {}))
        params["tracking"] = tparams
    else:
        params["tracking"] = {"source": "provided", **track.summary(prefix="track_")}

    salience = track.summary()["salience_median"]
    if (salience is None or salience < min_salience_db) and not force:
        params.update(applied=False, n_harmonics=0, versions=versions(), warnings_it=[
            "Armoniche del motore deboli o assenti: la traccia potrebbe seguire la voce o la "
            "musica. Attenuazione non applicata; verificare la traccia."])
        return x.copy(), to_jsonable(params), track

    n = len(x)
    f0 = track.f0_at(np.arange(n) / sr)
    phi = 2 * np.pi * np.cumsum(f0) / sr
    f_ref = float(np.max(f0)) if n else 0.0
    n_harm = int(min(max_freq_hz, 0.45 * sr) // f_ref) if f_ref > 0 else 0
    sos = signal.butter(lpf_order, bandwidth_hz / 2, fs=sr, output="sos")
    padlen = int(min(max(n - 1, 0), 3 * sr / max(bandwidth_hz / 2, 1e-3)))
    total = np.zeros(n)
    levels: dict[str, float] = {}
    for k in range(1, n_harm + 1):
        carrier = np.exp(1j * k * phi)
        # Mirror padding: an odd extension would turn the first sample of the
        # oscillating baseband into a DC step that leaks through the low-pass.
        base = signal.sosfiltfilt(sos, x * np.conj(carrier), padlen=padlen, padtype="even")
        comp = 2.0 * np.real(base * carrier)
        total += comp
        levels[str(k)] = round(float(10 * np.log10(np.mean(comp ** 2) + 1e-20)), 3)
    gain = 1.0 if depth_db is None else 1.0 - 10 ** (-depth_db / 20)
    y = x - gain * total if n_harm else x.copy()

    params.update(
        applied=True,
        n_harmonics=n_harm,
        harmonic_levels_db=levels,
        removed_energy_db=float(10 * np.log10(np.mean((gain * total) ** 2) + 1e-20)) if n else None,
        input_energy_db=float(10 * np.log10(np.mean(x ** 2) + 1e-20)) if n else None,
        lpf_padlen=padlen,
        subtraction_gain=gain,
        warnings_it=[],
        versions=versions(),
    )
    return y, to_jsonable(params), track


def run(input_path: Path, output_path: Path, *, subtype: str = "PCM_16",
        **kwargs) -> dict[str, Any]:
    """File wrapper for :func:`attenuate_engine_harmonics`. The returned params
    include the full track (``params["track"]``) for display."""
    audio, sr = load_mono(input_path)
    y, params, track = attenuate_engine_harmonics(audio, sr, **kwargs)
    params["track"] = track.to_dict()
    params.update(write_mono(output_path, y, sr, subtype=subtype))
    params["input_sha256"] = sha256_file(input_path)
    params["output_sha256"] = sha256_file(output_path)
    return params
