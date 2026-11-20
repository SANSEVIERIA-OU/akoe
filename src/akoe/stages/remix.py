# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Remix of cleaned and original audio, and choice of the automatic
transcriber's input (M6, issue #34).

Cleaning in front of Whisper-class models usually hurts; adding back a
fraction of the original ("observation adding", 10-25% in the studies)
recovers the damage and sometimes helps slightly (MASTER_PLAN §3.4.1).
Hence:

- :func:`remix` builds ``(1 - a) * g * cleaned + a * original`` with the
  cleaned signal time-aligned to the original (cross-correlation within
  ``max_lag_ms``) and level-matched (``g`` equalises RMS, over speech
  portions if given), so that ``a`` really is the share of original;
- :func:`choose_transcriber_input` picks the variant from the per-variant
  WER measured by the self-assessment (M5): smallest WER wins, ties (and
  improvements below ``min_improvement``) go to the original. The original
  must always be among the measured variants: the cleaned audio never
  reaches the transcriber without verification.

Provenance: a remix inherits the provenance of the cleaned input; a remix
of NEURAL cleaned audio is NEURAL (invariant 1: never merge deterministic
and neural cleaned audio, which this API cannot do: it mixes one cleaned
signal with the original only).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft

from akoe.config import StageKind
from akoe.stages.common import (
    Segments,
    as_float,
    load_mono,
    sample_mask,
    to_jsonable,
    versions,
    with_single_thread_blas,
    write_mono,
)
from akoe.utils import sha256_file

ALGORITHM_VERSION = "1"
ORIGINAL = "original"
RECOMMENDED_FRACTION_RANGE = (0.10, 0.25)


def variant_name(original_fraction: float) -> str:
    """Canonical name of a remix variant, e.g. ``remix_0.15``."""
    return f"remix_{original_fraction:.2f}"


def _lag(cleaned: np.ndarray, original: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Lag such that cleaned[n] ~ original[n - lag], and the normalised peak."""
    n = next_fast_len(len(cleaned) + len(original))
    cc = irfft(rfft(cleaned, n) * np.conj(rfft(original, n)), n)
    lags = np.concatenate([np.arange(0, max_lag + 1), np.arange(-max_lag, 0)])
    vals = np.concatenate([cc[: max_lag + 1], cc[n - max_lag:]]) if max_lag else cc[:1]
    i = int(np.argmax(vals))
    den = np.sqrt(float(np.dot(cleaned, cleaned)) * float(np.dot(original, original)))
    return int(lags[i]), float(vals[i] / den) if den > 0 else 0.0


@with_single_thread_blas
def remix(
    cleaned: np.ndarray,
    original: np.ndarray,
    sr: int,
    *,
    original_fraction: float = 0.15,
    align: bool = True,
    max_lag_ms: float = 50.0,
    level_match: bool = True,
    speech_segments: Segments | None = None,
    cleaned_kind: StageKind = StageKind.DETERMINISTIC,
    peak_limit: float = 0.99,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Mix cleaned and original audio. Returns ``(audio, params)``."""
    if not 0.0 <= original_fraction <= 1.0:
        raise ValueError("original_fraction must be in [0, 1]")
    c, o = as_float(cleaned), as_float(original)
    n = len(o)
    if len(c) < n:
        c = np.pad(c, (0, n - len(c)))
    c = c[:n]

    lag, peak = 0, None
    if align and n:
        lag, peak = _lag(c, o, int(round(max_lag_ms * sr / 1000)))
        if lag:
            # cleaned[n] ~ original[n - lag]: shift cleaned back by lag samples.
            c = np.roll(c, -lag)
            if lag > 0:
                c[n - lag:] = 0.0
            else:
                c[:-lag] = 0.0

    sel = sample_mask(speech_segments, n, sr) if speech_segments else np.ones(n, dtype=bool)
    rms_c = float(np.sqrt(np.mean(c[sel] ** 2))) if sel.any() else 0.0
    rms_o = float(np.sqrt(np.mean(o[sel] ** 2))) if sel.any() else 0.0
    gain = rms_o / rms_c if (level_match and rms_c > 0) else 1.0
    y = (1.0 - original_fraction) * gain * c + original_fraction * o
    peak_in = float(np.max(np.abs(y))) if n else 0.0
    out_gain = peak_limit / peak_in if peak_in > peak_limit else 1.0
    y = y * out_gain

    kind = StageKind.NEURAL if cleaned_kind == StageKind.NEURAL else StageKind.DETERMINISTIC
    lo, hi = RECOMMENDED_FRACTION_RANGE
    params = {
        "algorithm": "level_matched_time_aligned_remix",
        "algorithm_version": ALGORITHM_VERSION,
        "formula": "y = out_gain * ((1 - a) * g * cleaned_aligned + a * original)",
        "sr": sr,
        "original_fraction": original_fraction,
        "variant": variant_name(original_fraction),
        "in_recommended_range": bool(lo <= original_fraction <= hi),
        "recommended_range": [lo, hi],
        "align": align, "max_lag_ms": max_lag_ms, "lag_samples": lag,
        "lag_ms": lag * 1000.0 / sr, "alignment_peak": peak,
        "level_match": level_match,
        "level_reference": "speech_segments" if speech_segments else "whole_signal",
        "rms_cleaned": rms_c, "rms_original": rms_o, "cleaned_gain": gain,
        "peak_limit": peak_limit, "out_gain": out_gain,
        "cleaned_kind": cleaned_kind,
        "audio_provenance": kind,
        "versions": versions(),
    }
    return y, to_jsonable(params)


def run(cleaned_path: Path, original_path: Path, output_path: Path, *, subtype: str = "PCM_16",
        **kwargs) -> dict[str, Any]:
    """File wrapper for :func:`remix`. Returns the params for the log."""
    cleaned, sr_c = load_mono(cleaned_path)
    original, sr = load_mono(original_path)
    if sr_c != sr:
        raise ValueError("cleaned and original must have the same sample rate")
    y, params = remix(cleaned, original, sr, **kwargs)
    params.update(write_mono(output_path, y, sr, subtype=subtype))
    params["cleaned_sha256"] = sha256_file(cleaned_path)
    params["original_sha256"] = sha256_file(original_path)
    params["output_sha256"] = sha256_file(output_path)
    return params


@dataclass
class TranscriberInputDecision:
    """Which audio variant goes to the automatic transcriber, and why."""

    chosen: str
    reason_it: str
    wer_by_variant: dict[str, float]
    ranking: list[str]
    excluded: dict[str, str] = field(default_factory=dict)
    original_key: str = ORIGINAL
    tie_tolerance: float = 1e-9
    min_improvement: float = 0.0
    n_items: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self.__dict__.copy())


def choose_transcriber_input(
    wer_by_variant: dict[str, float],
    *,
    original_key: str = ORIGINAL,
    tie_tolerance: float = 1e-9,
    min_improvement: float = 0.0,
    n_items: dict[str, int] | None = None,
) -> TranscriberInputDecision:
    """Choose the transcriber input from self-assessment WERs (fractions or
    percentages, as long as they are consistent).

    Smallest WER wins. Ties within ``tie_tolerance`` go to the original, as
    does any variant whose improvement over the original is smaller than
    ``min_improvement`` (absolute WER units). Variants with missing,
    negative or non-finite WER are excluded and listed. Remaining ties
    between non-original variants are broken by name (deterministic).
    """
    if original_key not in wer_by_variant:
        raise ValueError(f"the original variant {original_key!r} must be measured: cleaned audio "
                         "never reaches the transcriber without comparison with the original")
    valid: dict[str, float] = {}
    excluded: dict[str, str] = {}
    for name, wer in wer_by_variant.items():
        try:
            value = float(wer)
        except (TypeError, ValueError):
            excluded[name] = "WER mancante o non numerico"
            continue
        if not math.isfinite(value) or value < 0:
            excluded[name] = "WER non valido"
            continue
        valid[name] = value
    if original_key not in valid:
        raise ValueError("the WER of the original variant is not valid")

    ranking = sorted(valid, key=lambda k: (valid[k], k != original_key, k))
    best = ranking[0]
    w_orig = valid[original_key]
    if best == original_key or valid[best] >= w_orig - tie_tolerance:
        chosen = original_key
        reason = ("Nessuna variante ha un WER inferiore a quello dell'audio originale: al "
                  "trascrittore automatico va l'originale.")
    elif w_orig - valid[best] < min_improvement - tie_tolerance:
        chosen = original_key
        reason = (f"La variante {best!r} migliora il WER di {w_orig - valid[best]:.4g}, meno della "
                  f"soglia minima {min_improvement:.4g}: al trascrittore automatico va "
                  "l'originale.")
    else:
        chosen = best
        reason = (f"La variante {best!r} ha il WER più basso ({valid[best]:.4g} contro "
                  f"{w_orig:.4g} dell'originale): scelta come ingresso del trascrittore "
                  "automatico per questo caso.")
    return TranscriberInputDecision(
        chosen=chosen, reason_it=reason, wer_by_variant=valid, ranking=ranking,
        excluded=excluded, original_key=original_key, tie_tolerance=tie_tolerance,
        min_improvement=min_improvement, n_items=n_items,
    )
