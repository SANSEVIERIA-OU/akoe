"""Cancellation of a known interfering source (radio, music) given a
reference recording of it (M6, issue #32).

Identification of the track (acoustic fingerprinting) is out of scope here:
the operator supplies the reference (the original recording of the song or
broadcast). Steps:

1. **Coarse alignment** by GCC-PHAT cross-correlation between the case
   recording and the reference (sharp peak at the direct path even with
   strong reverberation). A drift estimate (ppm) from the two halves of the
   recording is logged; drift is not corrected, the adaptive filter follows
   small drifts.
2. **Adaptive filter**: partitioned-block frequency-domain NLMS
   (overlap-save, gradient-constrained), with a long multi-tap filter
   (default 128 ms) to model the cabin path. Simple subtraction "rarely"
   suffices in a car because the reverberant sound is nearly as strong as
   the direct one (Alexander, Forth & Tunstall 2012); the log reports the
   simple-subtraction baseline for comparison.
3. **Safeguards against removing the subject's voice**:
   - adaptation is frozen (or slowed, ``mu_speech``) while the subject
     speaks. Speech portions come from the caller or, by default, from the
     robust VAD run on a first-pass residual; the output pass restarts from
     zero, so the filter is learned only on portions without speech;
   - an energy guard: per block the estimated interference may not exceed
     ``guard_ratio`` times the energy of the recording (prevents an
     overshooting filter from cancelling more than is there);
   - a leakage check: after cancellation the residual should be incoherent
     with the reference; if it stays coherent during speech (and not
     outside it), the subject's voice is probably present in the reference
     and is being cancelled with it (Erkek & Ozcan 2026 warn about this). A
     warning in Italian is added to the params. Freezing cannot prevent
     this: whatever is in the reference is filtered and subtracted.

Reported: alignment, correlation and coherence with the reference before
and after (reduction in dB), interference reduction (ERLE) outside speech,
energy removed during speech, simple-subtraction baseline.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal
from scipy.fft import irfft, next_fast_len, rfft

from forensicaudio.config import StageKind
from forensicaudio.stages.common import (
    Segments,
    as_float,
    load_mono,
    sample_mask,
    segments_to_list,
    to_jsonable,
    versions,
    with_single_thread_blas,
    write_mono,
)
from forensicaudio.utils import sha256_file

KIND = StageKind.DETERMINISTIC
ALGORITHM_VERSION = "1"


# --------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------


def _gcc_phat(m: np.ndarray, r: np.ndarray, phat_exponent: float) -> tuple[np.ndarray, np.ndarray]:
    """Generalised cross-correlation; returns (lags, values) with the
    convention m[n] ~ r[n - lag]."""
    n = next_fast_len(len(m) + len(r) - 1)
    g = rfft(m, n) * np.conj(rfft(r, n))
    mag = np.abs(g)
    g = g / (mag ** phat_exponent + 1e-12 * (mag.max() ** phat_exponent + 1e-30))
    cc = irfft(g, n)
    lags = np.concatenate([np.arange(-(len(r) - 1), 0), np.arange(0, len(m))])
    vals = np.concatenate([cc[n - (len(r) - 1):], cc[: len(m)]]) if len(r) > 1 else cc[: len(m)]
    return lags, vals


def align_reference(
    mixture: np.ndarray,
    reference: np.ndarray,
    sr: int,
    *,
    max_lag_s: float | None = None,
    phat_exponent: float = 1.0,
    center_lag: int | None = None,
) -> tuple[int, dict[str, Any]]:
    """Lag (samples) such that ``mixture[n] ~ reference[n - lag]`` at the
    direct path, with a peak-to-median confidence in dB."""
    m, r = as_float(mixture), as_float(reference)
    lags, vals = _gcc_phat(m, r, phat_exponent)
    sel = np.ones(len(lags), dtype=bool)
    c = 0 if center_lag is None else center_lag
    if max_lag_s is not None:
        sel &= np.abs(lags - c) <= int(round(max_lag_s * sr))
    if not sel.any():
        raise ValueError("no admissible lag: check max_lag_s")
    a = np.abs(vals[sel])
    i = int(np.argmax(a))
    lag = int(lags[sel][i])
    conf = float(20 * np.log10(a[i] / (np.median(a) + 1e-30)))
    return lag, {"lag_samples": lag, "lag_s": lag / sr, "peak_to_median_db": conf,
                 "polarity": int(np.sign(vals[sel][i]) or 1), "phat_exponent": phat_exponent,
                 "max_lag_s": max_lag_s}


def _shift(r: np.ndarray, shift: int, n: int) -> np.ndarray:
    """x[k] = r[k - shift] for k in [0, n), zeros outside the reference."""
    x = np.zeros(n)
    src, dst = max(0, -shift), max(0, shift)
    length = min(len(r) - src, n - dst)
    if length > 0:
        x[dst:dst + length] = r[src:src + length]
    return x


# --------------------------------------------------------------------------
# Partitioned-block frequency-domain NLMS
# --------------------------------------------------------------------------


def _pbfdaf(
    x: np.ndarray,
    d: np.ndarray,
    *,
    block: int,
    n_part: int,
    mu: float,
    block_mu: np.ndarray,
    delta: float,
    beta: float,
    constrained: bool,
    guard_ratio: float | None,
    epochs: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Overlap-save PBFDAF (FFT size 2*block). ``block_mu`` scales the step
    per block (0 freezes adaptation). Returns (error, estimate, n_guarded)
    of the last epoch."""
    b = block
    n_blocks = int(np.ceil(len(d) / b))
    xp = np.zeros(n_blocks * b)
    dp = np.zeros(n_blocks * b)
    xp[: len(x)] = x
    dp[: len(d)] = d
    n_bins = b + 1
    w = np.zeros((n_part, n_bins), dtype=np.complex128)
    p_init = n_part * 2 * b * float(np.mean(x ** 2)) + 1e-12
    e_out = np.zeros_like(dp)
    y_out = np.zeros_like(dp)
    zeros_b = np.zeros(b)
    n_guarded = 0
    for _ in range(epochs):
        fdl = np.zeros((n_part, n_bins), dtype=np.complex128)  # fdl[0] = newest
        prev = np.zeros(b)
        pk = np.full(n_bins, p_init)
        n_guarded = 0
        for m in range(n_blocks):
            sl = slice(m * b, (m + 1) * b)
            xb = xp[sl]
            fdl = np.roll(fdl, 1, axis=0)
            fdl[0] = rfft(np.concatenate([prev, xb]))
            prev = xb
            y = irfft((w * fdl).sum(axis=0), 2 * b)[b:]
            db = dp[sl]
            if guard_ratio is not None:
                ey, ed = float(y @ y), float(db @ db)
                if ey > guard_ratio * ed and ey > 0:
                    y = y * np.sqrt(guard_ratio * ed / ey)
                    n_guarded += 1
            e = db - y
            e_out[sl] = e
            y_out[sl] = y
            step = mu * block_mu[m]
            if step <= 0:
                continue
            pk = beta * pk + (1 - beta) * (fdl.real ** 2 + fdl.imag ** 2).sum(axis=0)
            ef = rfft(np.concatenate([zeros_b, e]))
            grad = np.conj(fdl) * (ef / (pk + delta))[None, :]
            if constrained:
                g = irfft(grad, 2 * b, axis=1)
                g[:, b:] = 0.0
                grad = rfft(g, axis=1)
            w += step * grad
    return e_out[: len(d)], y_out[: len(d)], n_guarded


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def _max_norm_xcorr(a: np.ndarray, x: np.ndarray, max_lag: int) -> float:
    """max over lags 0..max_lag of |sum a[n] x[n - l]| / (||a|| ||x||)."""
    den = np.sqrt(float(a @ a) * float(x @ x))
    if den <= 0:
        return 0.0
    n = next_fast_len(len(a) + len(x))
    cc = irfft(rfft(a, n) * np.conj(rfft(x, n)), n)[: max_lag + 1]
    return float(np.max(np.abs(cc)) / den)


def _mean_coherence(x: np.ndarray, d: np.ndarray, sr: int, band: tuple[float, float]) -> float:
    nper = min(1024, len(x))
    if nper < 64:
        return float("nan")
    f, c = signal.coherence(x, d, fs=sr, nperseg=nper)
    sel = (f >= band[0]) & (f <= band[1])
    return float(np.mean(c[sel]))


def _db_ratio(num: float, den: float) -> float:
    return float(10 * np.log10(max(num, 1e-30) / max(den, 1e-30)))


# --------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------


@with_single_thread_blas
def cancel_reference(
    mixture: np.ndarray,
    sr: int,
    reference: np.ndarray,
    ref_sr: int | None = None,
    *,
    filter_ms: float = 128.0,
    block: int = 256,
    mu: float = 0.3,
    epochs: int = 2,
    pre_delay_ms: float = 8.0,
    max_lag_s: float | None = None,
    phat_exponent: float = 1.0,
    speech_segments: Segments | None = None,
    auto_speech_protection: bool = True,
    mu_speech: float = 0.0,
    regularization: float = 1e-3,
    power_smoothing: float = 0.9,
    constrained: bool = True,
    guard_ratio: float | None = 2.0,
    coherence_band_hz: tuple[float, float] = (300.0, 3400.0),
    leakage_margin: float = 0.04,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cancel the interference whose original is ``reference``. Returns
    ``(audio, params)``.

    ``speech_segments``: portions where the subject speaks (adaptation
    frozen there). ``None`` with ``auto_speech_protection`` runs the robust
    VAD on a first-pass residual; ``[]`` adapts everywhere.
    """
    from forensicaudio.stages import vad

    d = as_float(mixture)
    r = as_float(reference)
    ref_sr = sr if ref_sr is None else ref_sr
    params: dict[str, Any] = {
        "algorithm": "gcc_phat_alignment+pbfdaf_nlms",
        "algorithm_version": ALGORITHM_VERSION,
        "kind": KIND.value, "sr": sr, "reference_sr": ref_sr, "filter_ms": filter_ms,
        "block": block, "fft_size": 2 * block, "mu": mu, "epochs": epochs,
        "pre_delay_ms": pre_delay_ms, "mu_speech": mu_speech, "regularization": regularization,
        "power_smoothing": power_smoothing, "constrained": constrained,
        "guard_ratio": guard_ratio, "coherence_band_hz": list(coherence_band_hz),
        "leakage_margin": leakage_margin, "blas_threads": 1,
    }
    warnings: list[str] = []
    if ref_sr != sr:
        frac = Fraction(sr, ref_sr).limit_denominator(1000)
        r = signal.resample_poly(r, frac.numerator, frac.denominator)
        params["reference_resampling"] = {"method": "scipy.signal.resample_poly",
                                          "up": frac.numerator, "down": frac.denominator}

    # 1. Alignment (+ drift estimate from the two halves).
    lag, align = align_reference(d, r, sr, max_lag_s=max_lag_s, phat_exponent=phat_exponent)
    params["alignment"] = align
    if align["peak_to_median_db"] < 20.0:
        warnings.append("Allineamento incerto con il riferimento: verificare che il brano sia "
                        "quello giusto.")
    half = len(d) // 2
    if len(d) >= 4 * sr:
        try:
            l1, _ = align_reference(d[:half], r, sr, max_lag_s=0.05, center_lag=lag,
                                    phat_exponent=phat_exponent)
            l2, _ = align_reference(d[half:], r, sr, max_lag_s=0.05, center_lag=lag - half,
                                    phat_exponent=phat_exponent)
            params["drift_ppm_estimate"] = float((l2 + half - l1) / half * 1e6)
        except ValueError:
            params["drift_ppm_estimate"] = None
    pre = int(round(pre_delay_ms * sr / 1000))
    x = _shift(r, lag - pre, len(d))
    x_direct = _shift(r, lag, len(d))
    n_part = max(1, int(np.ceil(filter_ms * sr / 1000 / block)))
    params.update(n_partitions=n_part, filter_taps=n_part * block, pre_delay_samples=pre)
    delta = regularization * n_part * 2 * block * float(np.mean(x ** 2)) + 1e-12
    n_blocks = int(np.ceil(len(d) / block))

    def block_scale(segs: Segments) -> np.ndarray:
        sm = sample_mask(segs, n_blocks * block, sr).reshape(n_blocks, block).any(axis=1)
        return np.where(sm, mu_speech, 1.0)

    # 2. Speech protection: which blocks may adapt.
    if speech_segments is None and auto_speech_protection:
        e1, _, _ = _pbfdaf(x, d, block=block, n_part=n_part, mu=mu, block_mu=np.ones(n_blocks),
                           delta=delta, beta=power_smoothing, constrained=constrained,
                           guard_ratio=guard_ratio, epochs=1)
        speech_segments, vad_params = vad.detect_speech(e1, sr)
        params["speech_segments_source"] = "vad.detect_speech on first-pass residual"
        params["vad_params"] = vad_params
    elif speech_segments is None:
        speech_segments = []
        params["speech_segments_source"] = "none (adaptation everywhere)"
    else:
        params["speech_segments_source"] = "provided"
    params["speech_segments"] = segments_to_list(speech_segments)
    scale = block_scale(speech_segments)
    frozen = float(np.mean(scale < 1.0))
    params["fraction_blocks_speech_protected"] = frozen
    if frozen > 0.8:
        warnings.append("Quasi tutta la registrazione è marcata come voce: il filtro adattivo "
                        "ha poco materiale per adattarsi.")

    # 3. Adaptive cancellation (output pass, from zero weights).
    e, y, n_guarded = _pbfdaf(x, d, block=block, n_part=n_part, mu=mu, block_mu=scale,
                              delta=delta, beta=power_smoothing, constrained=constrained,
                              guard_ratio=guard_ratio, epochs=epochs)
    params["n_guarded_blocks"] = int(n_guarded)

    # 4. Report.
    speech = sample_mask(speech_segments, len(d), sr)
    ref_on = np.abs(x) > 0
    non_speech = ~speech & ref_on
    taps = n_part * block
    before = _max_norm_xcorr(d, x, taps)
    after = _max_norm_xcorr(e, x, taps)
    coh_b = _mean_coherence(x, d, sr, coherence_band_hz)
    coh_a = _mean_coherence(x, e, sr, coherence_band_hz)
    g = float(x_direct @ d) / max(float(x_direct @ x_direct), 1e-30)
    e_simple = d - g * x_direct
    report: dict[str, Any] = {
        "xcorr_before": before, "xcorr_after": after,
        "xcorr_reduction_db": float(20 * np.log10(max(before, 1e-12) / max(after, 1e-12))),
        "coherence_before": coh_b, "coherence_after": coh_a,
        "coherence_reduction_db": _db_ratio(coh_b, coh_a),
        "residual_rms_db": float(10 * np.log10(np.mean(e ** 2) + 1e-30)),
        "input_rms_db": float(10 * np.log10(np.mean(d ** 2) + 1e-30)),
    }
    def energy(a: np.ndarray, mask: np.ndarray) -> float:
        return float(a[mask] @ a[mask])

    if non_speech.any():
        report["erle_non_speech_db"] = _db_ratio(energy(d, non_speech), energy(e, non_speech))
        report["simple_subtraction_gain"] = g
        report["simple_subtraction_erle_non_speech_db"] = _db_ratio(
            energy(d, non_speech), energy(e_simple, non_speech))
    if speech.any():
        report["energy_removed_speech_db"] = _db_ratio(energy(d, speech), energy(e, speech))
    # Leakage check: once the interference is cancelled the residual is
    # incoherent with the reference. If the subject's voice is also in the
    # reference, the residual stays coherent with it during speech.
    speech_on = speech & ref_on
    if speech_on.sum() >= sr and non_speech.sum() >= sr:
        coh_s = _mean_coherence(x[speech_on], e[speech_on], sr, coherence_band_hz)
        coh_n = _mean_coherence(x[non_speech], e[non_speech], sr, coherence_band_hz)
        leak = bool(coh_s - coh_n > leakage_margin)
        report.update(residual_coherence_speech=coh_s, residual_coherence_non_speech=coh_n,
                      possible_speech_leakage=leak)
        if leak:
            warnings.append("Possibile voce del soggetto nel riferimento: la cancellazione "
                            "potrebbe attenuarla. Verificare all'ascolto.")
    params["report"] = report
    params["warnings_it"] = warnings
    params["versions"] = versions()
    return e, to_jsonable(params)


def run(input_path: Path, reference_path: Path, output_path: Path, *, subtype: str = "PCM_16",
        **kwargs) -> dict[str, Any]:
    """File wrapper for :func:`cancel_reference`. Returns the params for the log."""
    audio, sr = load_mono(input_path)
    ref, ref_sr = load_mono(reference_path)
    y, params = cancel_reference(audio, sr, ref, ref_sr, **kwargs)
    params.update(write_mono(output_path, y, sr, subtype=subtype))
    params["input_sha256"] = sha256_file(input_path)
    params["reference_sha256"] = sha256_file(reference_path)
    params["output_sha256"] = sha256_file(output_path)
    return params
