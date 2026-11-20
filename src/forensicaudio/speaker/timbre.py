"""Timbre / spectral profile of the subject for one recording condition
(IMPLEMENTATION_PLAN M1, issue #8, MASTER_PLAN §3.5.2). DETERMINISTIC.

Measures, computed only on frames that pass an energy/SNR gate:

- **Gate.** Frame RMS (dB re full scale) on a 10 ms grid. The noise floor of
  each source is a low percentile of its frame levels over the WHOLE source
  (pauses included); a frame inside the subject's spans is kept when it is at
  least `snr_gate_db` above that floor and above an absolute floor.
- **Long-term average spectrum (LTAS).** Power spectra of gated frames, in
  1/3-octave bands (IEC 61260 base-10 centres, 100 Hz - 8 kHz; bands above
  the lowest source Nyquist are dropped). Each source's mean spectrum is normalised to its total
  band power before pooling (weighted by gated frames), so the shape does not
  depend on recording gain; the overall level is reported separately.
- **F0.** `librosa.pyin` on the subject's spans; statistics over frames that
  are voiced (pyin HMM decision) AND gated. F0 stabilises after about 10 s of
  speech (Arantes, Eriksson & Gutzeit 2017): `stable` is true when the gated
  speech in which F0 was searched reaches `f0_stable_seconds`.
- **Formants F1-F3.** LPC (Burg, `librosa.lpc`) on voiced gated frames, after
  resampling to 2 x `formant_ceiling_hz` and pre-emphasis, Praat-style:
  order = 2 x `formant_max_formants`; roots kept when above `formant_min_hz`,
  below the ceiling minus 50 Hz and with bandwidth under
  `formant_max_bandwidth_hz`; F1-F3 are the three lowest. Frames with fewer
  than three candidates are counted and skipped.

Every parameter, the software versions and the gate statistics are in the
returned dict, which is JSON-serializable and goes into
`ConditionProfile.timbre`. Warnings are in Italian (operator-facing).

Open points (IMPLEMENTATION_PLAN M1, "Da verificare"): no minimum duration
for a stable LTAS is known from the literature; self-directed speech is not
characterised.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import librosa
import numpy as np
import scipy
from pydantic import BaseModel, ConfigDict

from forensicaudio.config import StageKind

KIND = StageKind.DETERMINISTIC
METHOD = "forensicaudio.speaker.timbre/v1"
_EPS = 1e-12


class TimbreParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    sr: int = 16000
    resample_method: str = "soxr_hq"
    hop_length: int = 160  # 10 ms grid shared by gate, LTAS, F0 and formants
    # Gate
    gate_frame_length: int = 512
    noise_floor_percentile: float = 10.0
    snr_gate_db: float = 10.0
    abs_floor_dbfs: float = -60.0
    # LTAS
    ltas_n_fft: int = 2048
    ltas_window: str = "hann"
    ltas_band_fmin_hz: float = 100.0
    ltas_band_fmax_hz: float = 8000.0
    # F0
    f0_fmin_hz: float = 60.0
    f0_fmax_hz: float = 400.0
    f0_frame_length: int = 1024
    f0_resolution: float = 0.1
    f0_stable_seconds: float = 10.0
    span_context_s: float = 0.1  # audio around each span given to pyin
    # Formants
    formant_ceiling_hz: float = 5500.0  # Praat convention: 5000 male, 5500 female voices
    formant_max_formants: int = 5
    formant_window_s: float = 0.025
    formant_window: str = "hamming"
    formant_preemphasis_from_hz: float = 50.0
    formant_min_hz: float = 90.0
    formant_max_bandwidth_hz: float = 400.0
    min_formant_frames: int = 50

    @property
    def lpc_order(self) -> int:
        return 2 * self.formant_max_formants

    @property
    def formant_sr(self) -> int:
        return int(round(2 * self.formant_ceiling_hz))


@dataclass
class TimbreSource:
    """One recording of the condition. `spans` are the subject's segments in
    seconds (None = whole source). The noise floor is estimated on the whole
    source, so pass the full recording rather than pre-cut segments."""

    audio: np.ndarray
    sr: int
    spans: Sequence[tuple[float, float]] | None = None
    label: str = ""


def third_octave_bands(fmin: float, fmax: float, nyquist: float) -> list[tuple[float, float, float]]:
    """(lower, centre, upper) for IEC 61260 base-10 bands with centre in [fmin, fmax]."""
    out = []
    for n in range(-30, 20):
        fc = 1000.0 * 10 ** (n / 10)
        if fc < fmin * 0.99 or fc > fmax * 1.01:
            continue
        lo, hi = fc * 10 ** (-1 / 20), fc * 10 ** (1 / 20)
        if hi <= nyquist:
            out.append((lo, fc, hi))
    return out


def _r(x: float | None, nd: int = 3) -> float | None:
    if x is None or not np.isfinite(x):
        return None
    return round(float(x), nd)


def _stats(values: np.ndarray) -> dict[str, Any]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    pct = np.percentile(v, [5, 10, 25, 50, 75, 90, 95])
    return {
        "n": int(v.size),
        "mean": _r(v.mean()),
        "std": _r(v.std()),
        "median": _r(pct[3]),
        "percentiles": {str(p): _r(x) for p, x in zip((5, 10, 25, 50, 75, 90, 95), pct)},
        "min": _r(v.min()),
        "max": _r(v.max()),
    }


def _span_mask(n_frames: int, hop: int, sr: int, spans: Sequence[tuple[float, float]] | None) -> np.ndarray:
    if spans is None:
        return np.ones(n_frames, dtype=bool)
    t = np.arange(n_frames) * hop / sr
    mask = np.zeros(n_frames, dtype=bool)
    for s, e in spans:
        mask |= (t >= s) & (t < e)
    return mask


def _frames_power(y: np.ndarray, idx: np.ndarray, n_fft: int, hop: int, window: np.ndarray) -> np.ndarray:
    """Sum over the selected frames (centred on idx*hop) of |rfft|^2."""
    pad = n_fft // 2
    yp = np.pad(y, (pad, pad))
    acc = np.zeros(n_fft // 2 + 1)
    for chunk in np.array_split(idx, max(1, len(idx) // 512)):
        if chunk.size == 0:
            continue
        frames = np.stack([yp[k * hop : k * hop + n_fft] for k in chunk]) * window
        acc += (np.abs(np.fft.rfft(frames, axis=1)) ** 2).sum(axis=0)
    return acc


def _formants_of_frame(frame: np.ndarray, p: TimbreParams, fs: int) -> list[float] | None:
    if not np.any(frame):
        return None
    try:
        a = librosa.lpc(frame, order=p.lpc_order)
    except FloatingPointError:
        return None
    if not np.all(np.isfinite(a)):
        return None
    roots = np.roots(a)
    roots = roots[np.imag(roots) > 0]
    freqs = np.angle(roots) * fs / (2 * np.pi)
    bws = -fs / np.pi * np.log(np.abs(roots) + _EPS)
    keep = (freqs > p.formant_min_hz) & (freqs < p.formant_ceiling_hz - 50) & (bws < p.formant_max_bandwidth_hz)
    cand = np.sort(freqs[keep])
    return [float(f) for f in cand[:3]] if cand.size >= 3 else None


def compute_timbre(sources: Sequence[TimbreSource], params: TimbreParams | None = None) -> dict[str, Any]:
    """Timbre profile of one condition from one or more recordings. See module docstring."""
    p = params or TimbreParams()
    sr, hop = p.sr, p.hop_length
    warnings: list[str] = []
    # Bands above the lowest source Nyquist would only measure resampling emptiness.
    nyquist = min([sr] + [int(s.sr) for s in sources]) / 2
    bands = third_octave_bands(p.ltas_band_fmin_hz, p.ltas_band_fmax_hz, nyquist)
    freqs = np.fft.rfftfreq(p.ltas_n_fft, 1 / sr)
    band_bins = [(freqs >= lo) & (freqs < hi) for lo, _, hi in bands]
    window = scipy.signal.get_window(p.ltas_window, p.ltas_n_fft, fftbins=True)
    density = 2.0 / (p.ltas_n_fft * float(np.sum(window**2)))
    fwin_len = int(round(p.formant_window_s * p.formant_sr))
    fwin = scipy.signal.get_window(p.formant_window, fwin_len, fftbins=False)
    pre = float(np.exp(-2 * np.pi * p.formant_preemphasis_from_hz / p.formant_sr))

    src_reports: list[dict[str, Any]] = []
    shapes: list[tuple[np.ndarray, int, float]] = []  # (normalised band powers, frames, total dB)
    pooled_bins = np.zeros_like(freqs)
    f0_values: list[np.ndarray] = []
    formants: list[list[float]] = []
    n_formant_frames = 0
    n_gated_total = 0
    n_voiced_total = 0
    n_span_total = 0

    for src in sources:
        y = np.asarray(src.audio, dtype=np.float64)
        if y.ndim != 1:
            raise ValueError("audio must be mono (1-D)")
        if src.sr != sr:
            y = librosa.resample(y, orig_sr=src.sr, target_sr=sr, res_type=p.resample_method)
        n_frames = 1 + len(y) // hop
        rms = librosa.feature.rms(y=y, frame_length=p.gate_frame_length, hop_length=hop, center=True)[0]
        rms = rms[:n_frames]
        n_frames = len(rms)
        db = 20 * np.log10(np.maximum(rms, _EPS))
        floor = float(np.percentile(db, p.noise_floor_percentile))
        in_span = _span_mask(n_frames, hop, sr, src.spans)
        gate = in_span & (db >= floor + p.snr_gate_db) & (db >= p.abs_floor_dbfs)
        est_snr = float(np.percentile(db[in_span], 95) - floor) if in_span.any() else None
        n_span_total += int(in_span.sum())
        n_gated_total += int(gate.sum())
        report: dict[str, Any] = {
            "label": src.label,
            "input_sr": int(src.sr),
            "seconds_total": _r(len(y) / sr),
            "seconds_in_spans": _r(in_span.sum() * hop / sr),
            "seconds_gated": _r(gate.sum() * hop / sr),
            "noise_floor_dbfs": _r(floor),
            "estimated_snr_db": _r(est_snr),
        }
        if est_snr is not None and est_snr < p.snr_gate_db:
            warnings.append(
                f"Sorgente {src.label or '?'}: SNR stimato {est_snr:.1f} dB sotto la soglia di "
                f"{p.snr_gate_db:.0f} dB, pochi tratti utilizzabili"
            )

        # LTAS (gated frames only).
        idx = np.flatnonzero(gate)
        if idx.size:
            power = _frames_power(y, idx, p.ltas_n_fft, hop, window) * density / idx.size
            band_power = np.array([power[m].sum() for m in band_bins])
            total = float(band_power.sum())
            if total > 0:
                shapes.append((band_power / total, int(idx.size), 10 * np.log10(total)))
                pooled_bins += power / total * idx.size

        # F0 and formants, span by span (pyin only where the subject speaks).
        voiced_global = np.zeros(n_frames, dtype=bool)
        f0_global = np.full(n_frames, np.nan)
        formant_done = np.zeros(n_frames, dtype=bool)
        spans = list(src.spans) if src.spans is not None else [(0.0, len(y) / sr)]
        ctx = int(round(p.span_context_s * sr / hop)) * hop
        for s, e in spans:
            k0 = max(0, int(np.floor(s * sr / hop)))
            a = max(0, k0 * hop - ctx)
            b = min(len(y), int(np.ceil(e * sr)) + ctx)
            chunk = y[a:b]
            if len(chunk) < p.f0_frame_length:
                continue
            f0, vflag, _ = librosa.pyin(
                chunk, fmin=p.f0_fmin_hz, fmax=p.f0_fmax_hz, sr=sr,
                frame_length=p.f0_frame_length, hop_length=hop, resolution=p.f0_resolution, center=True,
            )
            ks = a // hop + np.arange(len(f0))
            ok = ks < n_frames
            ks, f0, vflag = ks[ok], f0[ok], vflag[ok]
            sel = vflag & np.isfinite(f0)
            voiced_global[ks[sel]] = True
            f0_global[ks[sel]] = f0[sel]

            # Formants on voiced + gated frames of this span.
            fk = ks[sel]
            fk = fk[gate[fk] & ~formant_done[fk]]  # overlapping spans: each frame once
            if fk.size == 0:
                continue
            formant_done[fk] = True
            c2 = librosa.resample(chunk, orig_sr=sr, target_sr=p.formant_sr, res_type=p.resample_method)
            c2 = np.append(c2[0], c2[1:] - pre * c2[:-1])
            half = fwin_len // 2
            c2p = np.pad(c2, (half, half))
            for k in fk:
                centre = int(round((k * hop - a) / sr * p.formant_sr))
                frame = c2p[centre : centre + fwin_len]
                if len(frame) < fwin_len:
                    continue
                n_formant_frames += 1
                f = _formants_of_frame(frame * fwin, p, p.formant_sr)
                if f is not None:
                    formants.append(f)

        use = voiced_global & gate
        n_voiced_total += int(use.sum())
        f0_values.append(f0_global[use])
        report["seconds_voiced"] = _r(use.sum() * hop / sr)
        src_reports.append(report)

    seconds_speech = n_gated_total * hop / sr
    seconds_voiced = n_voiced_total * hop / sr

    # Pool LTAS.
    ltas: dict[str, Any]
    if shapes:
        weights = np.array([n for _, n, _ in shapes], dtype=np.float64)
        shape = np.sum([s * n for s, n, _ in shapes], axis=0) / weights.sum()
        overall_db = float(np.sum([d * n for _, n, d in shapes]) / weights.sum())
        rel = 10 * np.log10(np.maximum(shape, _EPS))
        centres = np.array([fc for _, fc, _ in bands])
        lo_edge, hi_edge = bands[0][0], bands[-1][2]
        in_range = (freqs >= lo_edge) & (freqs < hi_edge)
        centroid = float(np.sum(freqs[in_range] * pooled_bins[in_range]) / max(np.sum(pooled_bins[in_range]), _EPS))
        slope = float(np.polyfit(np.log2(centres), rel, 1)[0]) if len(centres) >= 2 else None
        ltas = {
            "bands": "third_octave_iec61260_base10",
            "nyquist_limit_hz": _r(nyquist, 1),
            "band_centres_hz": [_r(fc, 1) for fc in centres],
            "band_edges_hz": [[_r(lo, 1), _r(hi, 1)] for lo, _, hi in bands],
            "levels_rel_db": [_r(x) for x in rel],
            "levels_db": [_r(x + overall_db) for x in rel],
            "overall_level_db": _r(overall_db),
            "level_reference": "dB re full-scale power 1.0 (full-scale sine = -3 dB)",
            "spectral_centroid_hz": _r(centroid, 1),
            "slope_db_per_octave": _r(slope),
            "seconds": _r(seconds_speech),
        }
    else:
        ltas = {"bands": "third_octave_iec61260_base10", "seconds": 0.0}
        warnings.append("Nessun tratto con SNR sufficiente: spettro medio non calcolabile")

    # F0 statistics.
    f0_all = np.concatenate(f0_values) if f0_values else np.array([])
    f0_stats = _stats(f0_all)
    stable = seconds_speech >= p.f0_stable_seconds
    f0: dict[str, Any] = {
        "seconds_speech": _r(seconds_speech),
        "seconds_voiced": _r(seconds_voiced),
        "stable": bool(stable),
        "stable_threshold_s": p.f0_stable_seconds,
        "stable_basis": "gated speech seconds (Arantes, Eriksson & Gutzeit 2017: ~10 s)",
        "hz": f0_stats,
    }
    if f0_stats["n"]:
        p5, p95 = f0_stats["percentiles"]["5"], f0_stats["percentiles"]["95"]
        f0["range_p5_p95_hz"] = _r(p95 - p5)
        f0["range_p5_p95_semitones"] = _r(12 * np.log2(p95 / p5)) if p5 and p95 else None
    if not stable:
        warnings.append(
            f"Parlato utile {seconds_speech:.1f} s, sotto i {p.f0_stable_seconds:.0f} s: "
            "distribuzione di F0 non ancora stabile"
        )

    # Formant statistics.
    fm = np.array(formants) if formants else np.zeros((0, 3))
    form: dict[str, Any] = {
        "frames_analyzed": n_formant_frames,
        "frames_with_3_formants": int(fm.shape[0]),
        "F1": _stats(fm[:, 0]) if fm.size else {"n": 0},
        "F2": _stats(fm[:, 1]) if fm.size else {"n": 0},
        "F3": _stats(fm[:, 2]) if fm.size else {"n": 0},
    }
    if fm.shape[0] < p.min_formant_frames:
        warnings.append(
            f"Solo {fm.shape[0]} finestre sonore utili per le formanti (minimo consigliato "
            f"{p.min_formant_frames}): stime poco affidabili"
        )

    params_out = p.model_dump(mode="json")
    params_out.update({"lpc_order": p.lpc_order, "formant_sr": p.formant_sr, "lpc_method": "burg (librosa.lpc)"})
    return {
        "method": METHOD,
        "kind": KIND.value,
        "params": params_out,
        "software": {"librosa": librosa.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "gate": {
            "criterion": (
                f"frame RMS >= noise floor (p{p.noise_floor_percentile:g} of the source) + "
                f"{p.snr_gate_db:g} dB and >= {p.abs_floor_dbfs:g} dBFS, inside the subject's spans"
            ),
            "seconds_in_spans": _r(n_span_total * hop / sr),
            "seconds_gated": _r(seconds_speech),
        },
        "sources": src_reports,
        "ltas": ltas,
        "f0": f0,
        "formants": form,
        "warnings": warnings,
        "notes": [
            "Durata minima per uno spettro medio stabile: non nota in letteratura (da verificare).",
            "Misure riferite a questa condizione di registrazione: canale e codec alterano F1 e lo spettro.",
        ],
    }
