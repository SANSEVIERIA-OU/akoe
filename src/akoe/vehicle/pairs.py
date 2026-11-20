# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Local / transmitted recordings of the same event (MASTER_PLAN §4.3).

The covert device records locally and transmits; aligning the two recordings
measures what the transmission does (bandwidth, level, spectral transfer,
losses) and yields real better/worse pairs for adaptation and self-assessment.

Alignment is done on band-limited signals (default 300-3400 Hz at 8 kHz, the
band both paths share whatever the codec): a coarse lag from log-energy
envelopes, refined by normalized waveform cross-correlation with sub-sample
interpolation. Clock drift is measured from windowed offsets (linear fit) and
compensated before the channel estimate. Output: a JSON-serializable report.

Convention: `offset_s` is where local time 0 falls in the transmitted file,
i.e. t_transmitted = offset_s + t_local * (1 + drift_ppm * 1e-6).
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, csd, fftconvolve, resample as fft_resample, sosfiltfilt, welch
from scipy.stats import theilslopes

from akoe.case import Case, CaseError
from akoe.config import RecordingSource
from akoe.models import ArtifactRef
from akoe.utils import sha256_file
from akoe.vehicle.audio import (
    EPS, db, environment_info, frame_powers, load_audio, power, resample, to_mono,
)
from akoe.vehicle.codecs import detect_codec

log = logging.getLogger(__name__)

REPORT_VERSION = 1
TINY = 1e-30


class PairParams(BaseModel):
    align_sr: int = 8000
    band_hz: tuple[float, float] = (300.0, 3400.0)
    envelope_hop_s: float = 0.01
    envelope_win_s: float = 0.02
    max_offset_s: float = 30.0
    fine_search_s: float = 0.03
    window_s: float = 4.0  # short: drift within a window must not smear the peak
    window_hop_s: float = 2.0
    max_windows: int = 300
    drift_max_ppm: float = 1000.0
    drift_min_ppm: float = 10.0
    drift_min_total_ms: float = 1.0
    min_confidence: float = 0.3
    channel_max_sr: int = 48000
    welch_segment_s: float = 0.032
    bandwidth_drop_db: float = 10.0
    dropout_frame_s: float = 0.02
    dropout_drop_db: float = 20.0
    dropout_silence_dbfs: float = -90.0
    max_events: int = 200


# --------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------


def _envelope(x: np.ndarray, sr: int, hop_s: float, win_s: float) -> np.ndarray:
    p = frame_powers(x, max(1, int(win_s * sr)), max(1, int(hop_s * sr)))
    floor = max(float(p.max()) * 1e-6, EPS) if p.size else EPS
    e = 10 * np.log10(p + floor)
    return (e - e.mean()) / (e.std() + EPS)


def _overlap_corr(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    """Pearson correlation of a[i] and b[i + lag] over their overlap."""
    i0, i1 = max(0, -lag), min(len(a), len(b) - lag)
    if i1 - i0 < 3:
        return 0.0
    x, y = a[i0:i1], b[i0 + lag:i1 + lag]
    if x.std() <= EPS or y.std() <= EPS:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _ncc_search(
    ref: np.ndarray, sig: np.ndarray, i0: int, center: int, search: int
) -> tuple[float, float]:
    """Normalized cross-correlation of ref[i0:i0+W] against sig around lag `center`
    (+/- search). Returns (sub-sample lag, peak correlation)."""
    w = len(ref)
    s0 = max(0, i0 + center - search)
    s1 = min(len(sig), i0 + center + w + search)
    seg = sig[s0:s1]
    if len(seg) < w or np.sum(ref**2) <= EPS:
        return float(center), 0.0
    c = fftconvolve(seg, ref[::-1], mode="valid")
    csum = np.concatenate([[0.0], np.cumsum(seg**2)])
    energy = csum[w:] - csum[:-w]
    ncc = c / (np.sqrt(np.maximum(energy, EPS)) * np.sqrt(np.sum(ref**2)))
    j = int(np.argmax(ncc))
    frac = 0.0
    if 0 < j < len(ncc) - 1:
        y0, y1, y2 = ncc[j - 1], ncc[j], ncc[j + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > EPS:
            frac = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
    return float(s0 + j + frac - i0), float(ncc[j])


def align_pair(
    local: np.ndarray, sr_local: int, transmitted: np.ndarray, sr_tx: int,
    params: PairParams | None = None,
) -> dict[str, Any]:
    """Offset, confidence and clock drift between the two recordings.

    1. coarse lag from log-energy envelopes (robust to codec and gain);
    2. short windows over the overlap, each refined by normalized waveform
       cross-correlation (short, so that drift does not smear the peak);
    3. robust linear fit of the window offsets: slope = drift. Without
       significant drift, offset = median of the reliable window offsets.
    """
    p = params or PairParams()
    A = p.align_sr
    sos = butter(4, [p.band_hz[0], min(p.band_hz[1], 0.45 * A)], btype="band", fs=A, output="sos")
    bl = sosfiltfilt(sos, resample(to_mono(local), sr_local, A))
    bt = sosfiltfilt(sos, resample(to_mono(transmitted), sr_tx, A))

    hop = max(1, int(p.envelope_hop_s * A))
    el = _envelope(bl, A, p.envelope_hop_s, p.envelope_win_s)
    et = _envelope(bt, A, p.envelope_hop_s, p.envelope_win_s)
    max_lag = int(p.max_offset_s / p.envelope_hop_s)
    c = fftconvolve(et, el[::-1], mode="full")
    zero = len(el) - 1
    lo, hi = max(0, zero - max_lag), min(len(c), zero + max_lag + 1)
    lag_frames = lo + int(np.argmax(c[lo:hi])) - zero
    coarse = lag_frames * hop

    i_start, i_end = max(0, -coarse), min(len(bl), len(bt) - coarse)
    span = max(0, i_end - i_start)
    search = int((p.fine_search_s + p.drift_max_ppm * 1e-6 * span / A) * A) + 2 * hop
    tt, ll, cc, w = _windowed_offsets(bl, bt, A, coarse, search, i_start, i_end, p)
    ok = cc >= p.min_confidence

    span_s = span / A
    drift = _fit_drift(tt, ll, ok, span_s, w / A, A, p)
    confidence_raw = float(np.median(cc[ok])) if ok.any() else None
    if ok.any():
        offset_s = drift["intercept_s"] if drift["detected"] else float(np.median(ll[ok]))
        confidence = confidence_raw
    elif len(cc):
        j = int(np.argmax(cc))
        offset_s, confidence = float(ll[j]), float(cc[j])
    else:
        offset_s, confidence = coarse / A, 0.0
    if drift["detected"]:
        # Second pass on the drift-compensated signal: windows no longer smear,
        # so the residual drift and the confidence are measured cleanly.
        n2 = int(round(len(bt) / (1 + drift["ppm"] * 1e-6)))
        bt2 = fft_resample(bt, n2)
        k1 = len(bt) / n2  # effective factor after rounding the length
        center = int(round(offset_s / k1 * A))
        search2 = 2 * hop + int((p.fine_search_s / 4 + 0.25 * abs(k1 - 1) * span_s) * A)
        tt2, ll2, cc2, w2 = _windowed_offsets(bl, bt2, A, center, search2, i_start,
                                              min(len(bl), len(bt2) - center), p)
        ok2 = cc2 >= p.min_confidence
        d2 = _fit_drift(tt2, ll2, ok2, span_s, w2 / A, A, p, force_fit=True)
        if d2["n_used"] >= 3:
            ppm = (k1 * (1 + d2["ppm"] * 1e-6) - 1) * 1e6
            if _drift_significant(ppm, span_s, p):
                offset_s = k1 * d2["intercept_s"]
                confidence = float(np.median(cc2[ok2]))
                drift.update(ppm=round(ppm, 2), intercept_s=round(offset_s, 6),
                             total_ms=round(abs(ppm) * 1e-6 * span_s * 1000, 3),
                             n_used=d2["n_used"], residual_ms=d2["residual_ms"], refined=True)
            else:  # first-pass drift was spurious
                drift.update(detected=False, ppm=round(ppm, 2),
                             note="deriva del primo passaggio non confermata")
                offset_s = float(np.median(ll[ok]))
    return {
        "offset_s": round(offset_s, 6),
        "confidence": round(confidence, 4),
        "coarse_offset_s": round(coarse / A, 4),
        "coarse_confidence": round(_overlap_corr(el, et, lag_frames), 4),
        "confidence_before_drift_compensation":
            None if confidence_raw is None else round(confidence_raw, 4),
        "n_windows": len(tt),
        "n_reliable_windows": int(ok.sum()),
        "window_s": round(w / A, 3),
        "windows": [{"t_s": round(a, 3), "offset_s": round(b, 6), "confidence": round(c, 3)}
                    for a, b, c in zip(tt.tolist(), ll.tolist(), cc.tolist())],
        "drift": drift,
        "method": "log-envelope xcorr, then windowed band-limited waveform NCC "
                  "(parabolic sub-sample) and robust linear fit",
        "analysis_sample_rate": A,
        "band_hz": list(p.band_hz),
    }


def _windowed_offsets(
    bl: np.ndarray, bt: np.ndarray, A: int, center: int, search: int, i_start: int, i_end: int,
    p: PairParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Per-window (centre time s, offset s, NCC) over local samples [i_start, i_end)."""
    span = max(0, i_end - i_start)
    w = min(int(p.window_s * A), span)
    step = int(p.window_hop_s * A)
    if span > w and p.max_windows > 1:
        step = max(step, (span - w) // (p.max_windows - 1))
    t, lags, confs = [], [], []
    for a in (range(i_start, i_end - w + 1, max(1, step)) if w > 0 else ()):
        lg, cf = _ncc_search(bl[a:a + w], bt, a, center, search)
        t.append((a + w / 2) / A)
        lags.append(lg / A)
        confs.append(cf)
    return np.asarray(t), np.asarray(lags), np.asarray(confs), w


def _drift_significant(ppm: float, span_s: float, p: PairParams) -> bool:
    return abs(ppm) >= p.drift_min_ppm and abs(ppm) * 1e-6 * span_s * 1000 >= p.drift_min_total_ms


def _fit_drift(
    tt: np.ndarray, ll: np.ndarray, ok: np.ndarray, span_s: float, w_s: float, A: int,
    p: PairParams, *, force_fit: bool = False,
) -> dict[str, Any]:
    """Robust line through the reliable window offsets: Theil-Sen start (tolerates
    windows locked on a wrong peak), then least squares on the inliers."""
    out: dict[str, Any] = {"detected": False, "ppm": 0.0, "intercept_s": None, "total_ms": 0.0,
                           "n_used": 0, "residual_ms": None}
    if ok.any():
        out["intercept_s"] = round(float(np.median(ll[ok])), 6)
    if ok.sum() < 3 or (span_s < 2 * w_s and not force_fit):
        out["note"] = "dati insufficienti per misurare la deriva"
        return out
    b, a, _, _ = theilslopes(ll[ok], tt[ok])
    res = ll - (a + b * tt)
    mad = float(np.median(np.abs(res[ok] - np.median(res[ok])))) * 1.4826
    use = ok & (np.abs(res) <= max(3 * mad, 1.0 / A))
    if use.sum() >= 3:
        b, a = np.polyfit(tt[use], ll[use], 1)
    else:
        use = ok
    res = ll[use] - (a + b * tt[use])
    out.update(
        ppm=round(float(b * 1e6), 2),
        intercept_s=round(float(a), 6),
        total_ms=round(float(abs(b) * span_s * 1000), 3),
        n_used=int(use.sum()),
        residual_ms=round(float(np.std(res) * 1000), 4),
        detected=_drift_significant(float(b * 1e6), span_s, p),
    )
    return out


def apply_alignment(
    transmitted: np.ndarray, sr: int, alignment: dict[str, Any], n_local: int
) -> tuple[np.ndarray, np.ndarray]:
    """Transmitted audio (at `sr`) mapped onto the local timeline (length
    `n_local`), with drift compensated by FFT resampling. Returns (audio, valid mask)."""
    y = to_mono(transmitted)
    offset = alignment["offset_s"]
    drift = alignment["drift"]["ppm"] * 1e-6 if alignment["drift"]["detected"] else 0.0
    if drift:
        n = int(round(len(y) / (1 + drift)))
        offset = offset * n / len(y)
        y = fft_resample(y, n)
    k_exact = offset * sr
    k = int(np.floor(k_exact))
    y = _fractional_advance(y, k_exact - k)
    out = np.zeros(n_local)
    i0, i1 = max(0, -k), min(n_local, len(y) - k)
    if i1 > i0:
        out[i0:i1] = y[i0 + k:i1 + k]
    valid = np.zeros(n_local, dtype=bool)
    valid[max(0, i0):max(0, i1)] = True
    return out, valid


def _fractional_advance(y: np.ndarray, frac: float, half: int = 32) -> np.ndarray:
    """z[m] = y(m + frac) with a Blackman-windowed sinc (accurate up to ~0.45 fs)."""
    if abs(frac) < 1e-6:
        return y
    j = np.arange(-half, half + 1)
    t = j - frac
    u = np.pi * t / (half + 1)
    g = np.sinc(t) * (0.42 + 0.5 * np.cos(u) + 0.08 * np.cos(2 * u))
    return fftconvolve(y, g[::-1], mode="full")[half:half + len(y)]


# --------------------------------------------------------------------------
# Channel estimate
# --------------------------------------------------------------------------


def _third_octaves(fmax: float) -> list[float]:
    centers, k = [], -10
    while 1000 * 2 ** ((k + 0.5) / 3) <= fmax:
        centers.append(1000 * 2 ** (k / 3))
        k += 1
    return centers


def _edge(
    f: np.ndarray, below: np.ndarray, start_hz: float, direction: int, stop_hz: float
) -> float | None:
    """Walk from start_hz in `direction`; return the last frequency before the
    gain falls below threshold, or None if it never does (down to stop_hz)."""
    j = int(np.argmin(np.abs(f - start_hz)))
    while True:
        nxt = j + direction
        if nxt < 0 or nxt >= len(f) or (direction < 0 and f[nxt] < stop_hz):
            return None
        if below[nxt]:
            return float(f[j])
        j = nxt


def estimate_channel(
    local: np.ndarray, transmitted_aligned: np.ndarray, sr: int, params: PairParams | None = None
) -> dict[str, Any]:
    """Bandwidth, level difference, spectral transfer and dropouts of the
    transmission, from time-aligned arrays of equal length at `sr`."""
    p = params or PairParams()
    x, y = np.asarray(local, dtype=np.float64), np.asarray(transmitted_aligned, dtype=np.float64)
    nper = int(2 ** round(math.log2(max(64, p.welch_segment_s * sr))))
    nper = min(nper, max(16, len(x)))
    f, pxx = welch(x, sr, nperseg=nper)
    _, pyy = welch(y, sr, nperseg=nper)
    _, pxy = csd(x, y, sr, nperseg=nper)
    # PSDs of quiet bands are far below EPS: use a much smaller floor here.
    pxx, pyy, axy = np.maximum(pxx, TINY), np.maximum(pyy, TINY), np.maximum(np.abs(pxy), TINY)
    coh = np.clip(axy**2 / pxx / pyy, 0.0, 1.0)
    g_pow = 10 * np.log10(pyy / pxx)
    g_h1 = 20 * np.log10(axy / pxx)

    df = f[1] - f[0]
    g_s = uniform_filter1d(g_pow, max(1, int(round(100 / df))), mode="nearest")
    ref_band = (f >= 500) & (f <= 2000)
    ref = float(np.median(g_s[ref_band])) if ref_band.any() else float(np.median(g_s))
    below = g_s < ref - p.bandwidth_drop_db
    high = _edge(f, below, 1000.0, +1, 0.0)
    low = _edge(f, below, 1000.0, -1, 50.0)

    speech_band = (f >= 300) & (f <= min(3400, sr / 2))
    mean_coh = float(np.mean(coh[speech_band])) if speech_band.any() else None

    bands = []
    for fc in _third_octaves(sr / 2):
        sel = (f >= fc * 2 ** (-1 / 6)) & (f < fc * 2 ** (1 / 6))
        if sel.any():
            bands.append({"center_hz": round(fc, 1),
                          "gain_db_power": round(float(np.mean(g_pow[sel])), 2),
                          "gain_db_h1": round(float(np.mean(g_h1[sel])), 2),
                          "coherence": round(float(np.mean(coh[sel])), 3)})
    step = max(1, int(np.ceil(len(f) / 256)))
    transfer = {
        "freqs_hz": [round(float(v), 1) for v in f[::step]],
        "gain_db_power": [round(float(v), 2) for v in g_pow[::step]],
        "gain_db_h1": [round(float(v), 2) for v in g_h1[::step]],
        "coherence": [round(float(v), 3) for v in coh[::step]],
    }

    # Levels and dropouts on short frames.
    frame = max(1, int(p.dropout_frame_s * sr))
    ll = 10 * np.log10(frame_powers(x, frame) + EPS)
    lt = 10 * np.log10(frame_powers(y, frame) + EPS)
    active = ll > np.percentile(ll, 10) + 10
    level_diff = db(power(y)) - db(power(x))
    if active.any():
        active_diff = float(10 * np.log10(np.mean(10 ** (lt[active] / 10))
                                          / np.mean(10 ** (ll[active] / 10))))
    else:
        active_diff = level_diff
    silent = lt < p.dropout_silence_dbfs
    drop = (active & (lt < ll + active_diff - p.dropout_drop_db)) | (
        silent & (ll > p.dropout_silence_dbfs + 20))
    events = []
    d = np.diff(np.concatenate([[0], drop.astype(np.int8), [0]]))
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        events.append({"start_s": round(a * frame / sr, 3), "end_s": round(b * frame / sr, 3),
                       "kind": "silence" if silent[a:b].all() else "attenuation"})
    return {
        "analysis_sample_rate": int(sr),
        "level_difference_db": round(level_diff, 2),
        "active_level_difference_db": round(active_diff, 2),
        "bandwidth_hz": {"low": low, "high": high,
                         "high_limited_by_sample_rate": high is None,
                         "reference_gain_db": round(ref, 2), "drop_db": p.bandwidth_drop_db},
        "mean_coherence_speech_band": None if mean_coh is None else round(mean_coh, 3),
        "third_octave_bands": bands,
        "transfer": transfer,
        "dropouts": {
            "n_events": len(events),
            "total_s": round(sum(e["end_s"] - e["start_s"] for e in events), 3),
            "fraction_of_active": round(float(drop[active].mean()), 4) if active.any() else 0.0,
            "events": events[: p.max_events],
            "truncated": len(events) > p.max_events,
        },
    }


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


def _clean(obj: Any) -> Any:
    """JSON-safe copy: numpy scalars to Python, NaN/inf to None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def analyze_pair(
    local: np.ndarray, sr_local: int, transmitted: np.ndarray, sr_tx: int,
    params: PairParams | None = None,
) -> dict[str, Any]:
    """Align the pair and estimate the transmission channel. JSON-serializable."""
    p = params or PairParams()
    warnings: list[str] = []
    alignment = align_pair(local, sr_local, transmitted, sr_tx, p)
    if alignment["confidence"] < p.min_confidence:
        warnings.append(f"Allineamento poco affidabile (confidenza {alignment['confidence']:.2f}): "
                        "verificare che le due registrazioni riguardino lo stesso evento.")
    if alignment["drift"]["detected"]:
        warnings.append(f"Deriva di clock rilevata: {alignment['drift']['ppm']:.1f} ppm "
                        f"({alignment['drift']['total_ms']:.1f} ms sulla durata); compensata.")

    C = int(min(sr_local, sr_tx, p.channel_max_sr))
    lc = resample(to_mono(local), sr_local, C)
    tc = resample(to_mono(transmitted), sr_tx, C)
    ta, valid = apply_alignment(tc, C, alignment, len(lc))
    idx = np.flatnonzero(valid)
    channel = None
    if idx.size > C // 2:
        i0, i1 = int(idx[0]), int(idx[-1]) + 1
        channel = estimate_channel(lc[i0:i1], ta[i0:i1], C, p)
        for ev in channel["dropouts"]["events"]:
            ev["start_s"] = round(ev["start_s"] + i0 / C, 3)
            ev["end_s"] = round(ev["end_s"] + i0 / C, 3)
        channel["overlap_s"] = {"start": round(i0 / C, 3), "end": round(i1 / C, 3)}
        if channel["dropouts"]["n_events"]:
            d = channel["dropouts"]
            warnings.append(f"Possibili perdite nella trasmissione: {d['n_events']} eventi, "
                            f"{d['total_s']:.2f} s in totale.")
    else:
        warnings.append("Sovrapposizione tra le registrazioni insufficiente per stimare il canale.")

    report = {
        "version": REPORT_VERSION,
        "local": {"sample_rate": int(sr_local), "duration_s": round(len(local) / sr_local, 3)},
        "transmitted": {"sample_rate": int(sr_tx),
                        "duration_s": round(len(transmitted) / sr_tx, 3)},
        "alignment": alignment,
        "channel": channel,
        "params": p.model_dump(mode="json"),
        "environment": environment_info(),
        "warnings": warnings,
    }
    for w in warnings:
        log.warning(w)
    return _clean(report)


def analyze_pair_files(
    local: Path, transmitted: Path, params: PairParams | None = None
) -> dict[str, Any]:
    """`analyze_pair` on two files, with their hashes and detected codecs."""
    local, transmitted = Path(local), Path(transmitted)
    a, sa = load_audio(local)
    b, sb = load_audio(transmitted)
    report = analyze_pair(a, sa, b, sb, params)
    for key, path in (("local", local), ("transmitted", transmitted)):
        report[key].update(file=path.name, sha256=sha256_file(path),
                           codec=_clean(detect_codec(path)))
    return report


def analyze_case_pair(
    case: Case, pair_id: str, params: PairParams | None = None
) -> tuple[dict[str, Any], ArtifactRef]:
    """Analyze the local/transmitted evidence sharing `pair_id`; the report is
    written to `outputs/pair-<pair_id>.json` and returned with its artifact."""
    items = case.pairs().get(pair_id)
    if not items:
        raise CaseError(f"unknown pair: {pair_id}")
    by_source = {i.condition.source: i for i in items}
    loc, tx = by_source.get(RecordingSource.LOCAL), by_source.get(RecordingSource.TRANSMITTED)
    if loc is None or tx is None:
        raise CaseError(f"pair {pair_id} needs one local and one transmitted recording")
    report = analyze_pair_files(case.path_of(loc), case.path_of(tx), params)
    report["pair_id"] = pair_id
    report["local"]["evidence_id"] = loc.evidence_id
    report["transmitted"]["evidence_id"] = tx.evidence_id
    out = case.dir("outputs") / f"pair-{pair_id}.json"
    out.write_text(json.dumps(report, indent=2))
    return report, ArtifactRef.from_file(out, kind="pair_report", root=case.root)
