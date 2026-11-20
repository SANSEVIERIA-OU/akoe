# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
from pathlib import Path
from typing import Any

import noisereduce as nr
import numpy as np
import soundfile as sf
from scipy import ndimage

from akoe.stages.common import (
    Segments,
    array_sha256,
    as_float,
    frame_mask,
    istft,
    load_mono,
    segments_to_list,
    stft,
    to_jsonable,
    versions,
    write_mono,
)
from akoe.utils import sha256_file


def run_deterministic(input_path: Path, output_path: Path, *, stationary: bool = True) -> None:
    """Spectral-gating noise reduction. Reproducible from parameters + library version."""
    audio, sr = sf.read(str(input_path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    reduced = nr.reduce_noise(y=audio.astype(np.float32), sr=sr, stationary=stationary)
    sf.write(str(output_path), reduced.astype(np.float32), sr, subtype="PCM_16")


# --------------------------------------------------------------------------
# Offline noise reduction with a whole-recording noise profile (M6, issue #30)
# --------------------------------------------------------------------------

OFFLINE_VERSION = "1"


def estimate_noise_profile(
    audio: np.ndarray,
    sr: int,
    *,
    speech_segments: Segments | None = None,
    n_fft: int = 512,
    hop: int = 128,
    guard_s: float = 0.1,
    min_noise_s: float = 1.0,
    fallback_quantile: float = 0.1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Noise power spectrum averaged over ALL non-speech frames of the whole
    recording (offline). Frames within ``guard_s`` of a speech segment are
    excluded. If less than ``min_noise_s`` of noise remains, the quietest
    ``fallback_quantile`` of frames is used instead (logged).

    ``speech_segments=None`` runs the robust VAD (``vad.detect_speech``).
    """
    from akoe.stages import vad

    x = as_float(audio)
    info: dict[str, Any] = {"n_fft": n_fft, "hop": hop, "guard_s": guard_s,
                            "min_noise_s": min_noise_s, "fallback_quantile": fallback_quantile}
    if speech_segments is None:
        speech_segments, vad_params = vad.detect_speech(x, sr)
        info["speech_segments_source"] = "vad.detect_speech"
        info["vad_params"] = vad_params
    else:
        info["speech_segments_source"] = "provided"
    info["speech_segments"] = segments_to_list(speech_segments)

    power = np.abs(stft(x, n_fft, hop)) ** 2
    times = np.arange(power.shape[1]) * hop / sr
    noise = ~frame_mask(speech_segments, times, margin_s=guard_s)
    noise_s = float(noise.sum() * hop / sr)
    if noise_s >= min_noise_s:
        info["noise_profile_source"] = "non_speech_frames"
    else:
        energy = power.sum(axis=0)
        k = max(1, int(np.ceil(fallback_quantile * len(energy))))
        noise = np.zeros(len(energy), dtype=bool)
        noise[np.argsort(energy, kind="stable")[:k]] = True
        info["noise_profile_source"] = "quietest_frames_fallback"
        info["warning_it"] = ("Poco materiale senza voce: profilo del rumore stimato sui "
                              "tratti più silenziosi.")
    profile = power[:, noise].mean(axis=1)
    info.update(n_noise_frames=int(noise.sum()), noise_seconds=float(noise.sum() * hop / sr))
    return profile, info


def reduce_noise_offline(
    audio: np.ndarray,
    sr: int,
    *,
    speech_segments: Segments | None = None,
    noise_profile: np.ndarray | None = None,
    method: str = "wiener",
    n_fft: int = 512,
    hop: int = 128,
    guard_s: float = 0.1,
    min_noise_s: float = 1.0,
    fallback_quantile: float = 0.1,
    dd_alpha: float = 0.98,
    oversubtraction: float = 1.0,
    gain_floor_db: float = -15.0,
    freq_smoothing_bins: int = 3,
    time_smoothing_frames: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Deterministic noise reduction with a whole-recording noise profile.

    ``method="wiener"``: Wiener gain with decision-directed a-priori SNR
    (Ephraim-Malah), ``xi_t = a * G_{t-1}^2 * gamma_{t-1} + (1 - a) *
    max(gamma_t - 1, 0)``, ``G = xi / (1 + xi)``.
    ``method="spectral_subtraction"``: power subtraction,
    ``G = sqrt(max(1 - beta * N / |X|^2, 0))``.

    The gain is smoothed over ``freq_smoothing_bins`` x
    ``time_smoothing_frames`` (limits musical noise) and floored at
    ``gain_floor_db`` (never removes more than that, keeps the noise
    texture audible). ``noise_profile`` can be supplied (e.g. from a vehicle
    noise recording, same ``n_fft``); otherwise it is estimated from all
    non-speech portions. Returns ``(audio, params)``.
    """
    x = as_float(audio)
    if method not in ("wiener", "spectral_subtraction"):
        raise ValueError(f"unknown method: {method}")
    spec = stft(x, n_fft, hop)
    power = np.abs(spec) ** 2

    if noise_profile is None:
        noise, profile_info = estimate_noise_profile(
            x, sr, speech_segments=speech_segments, n_fft=n_fft, hop=hop, guard_s=guard_s,
            min_noise_s=min_noise_s, fallback_quantile=fallback_quantile)
    else:
        noise = np.asarray(noise_profile, dtype=np.float64)
        if noise.shape != (n_fft // 2 + 1,):
            raise ValueError("noise_profile must have n_fft // 2 + 1 bins")
        profile_info = {"noise_profile_source": "provided"}
    noise = np.maximum(noise, 1e-20)

    gamma = power / noise[:, None]
    floor = 10 ** (gain_floor_db / 20)
    if method == "wiener":
        gain = np.empty_like(gamma)
        prev = np.zeros(gamma.shape[0])  # G_{t-1}^2 * gamma_{t-1}
        for t in range(gamma.shape[1]):
            xi = dd_alpha * prev + (1 - dd_alpha) * np.maximum(gamma[:, t] - 1.0, 0.0)
            xi = np.maximum(xi, 1e-3)
            g = xi / (1.0 + xi)
            gain[:, t] = g
            prev = g * g * gamma[:, t]
    else:
        gain = np.sqrt(np.maximum(1.0 - oversubtraction / np.maximum(gamma, 1e-20), 0.0))
    if freq_smoothing_bins > 1 or time_smoothing_frames > 1:
        gain = ndimage.uniform_filter(gain, size=(max(1, freq_smoothing_bins),
                                                  max(1, time_smoothing_frames)), mode="nearest")
    gain = np.clip(gain, floor, 1.0)
    y = istft(spec * gain, n_fft, hop, len(x))

    f = np.fft.rfftfreq(n_fft, 1.0 / sr)
    octave = {}
    for fc in (125, 250, 500, 1000, 2000, 4000, 8000):
        sel = (f >= fc / np.sqrt(2)) & (f < fc * np.sqrt(2))
        if sel.any():
            octave[str(fc)] = round(float(10 * np.log10(noise[sel].sum())), 3)
    params = {
        "algorithm": f"offline_{method}_whole_recording_noise_profile",
        "algorithm_version": OFFLINE_VERSION,
        "kind": "deterministic",
        "sr": sr, "method": method, "n_fft": n_fft, "hop": hop, "window": "hann",
        "guard_s": guard_s, "min_noise_s": min_noise_s, "fallback_quantile": fallback_quantile,
        "dd_alpha": dd_alpha if method == "wiener" else None,
        "xi_min": 1e-3 if method == "wiener" else None,
        "oversubtraction": oversubtraction if method == "spectral_subtraction" else None,
        "gain_floor_db": gain_floor_db,
        "freq_smoothing_bins": freq_smoothing_bins,
        "time_smoothing_frames": time_smoothing_frames,
        **profile_info,
        "noise_profile_sha256": array_sha256(noise),
        "noise_profile_octave_db": octave,
        "mean_gain_db": float(20 * np.log10(np.mean(gain))),
        "versions": versions(),
    }
    return y, to_jsonable(params)


def run_deterministic_offline(input_path: Path, output_path: Path, *, subtype: str = "PCM_16",
                              **kwargs) -> dict[str, Any]:
    """File wrapper for :func:`reduce_noise_offline`. Returns the params for the log."""
    audio, sr = load_mono(input_path)
    y, params = reduce_noise_offline(audio, sr, **kwargs)
    params.update(write_mono(output_path, y, sr, subtype=subtype))
    params["input_sha256"] = sha256_file(input_path)
    params["output_sha256"] = sha256_file(output_path)
    return params


def run_neural(
    input_path: Path,
    output_path: Path,
    *,
    backend: Any = None,
    device: str = "cpu",
    local_files_only: bool = False,
    subtype: str = "PCM_16",
    **kwargs: Any,
) -> dict[str, Any]:
    """Neural cleaning for HUMAN LISTENING (issue #59), labelled NEURAL.

    ``backend``: a name (``unipase``, ``mossformer2_se_48k``, ``gtcrn``,
    ``deepfilternet3``) or a :class:`neural_clean.NeuralEnhancer`. There is
    no implicit backend: without one this raises ``BackendUnavailable``
    (a ``NotImplementedError``, so pipelines record the stage as skipped)
    instead of downloading models. Tier defaults:
    ``neural_clean.default_backends(tier)``. ``kwargs`` go to
    :func:`neural_clean.enhance` (``chunking``, ``match_level``). Returns the
    params for the log (model id, revision, weight SHA-256, device, chunking,
    input/output SHA-256). The output is never the automatic transcriber's
    input without self-assessment (CLAUDE.md invariant 7).
    """
    from akoe.config import Tier
    from akoe.neural import BackendUnavailable
    from akoe.stages import neural_clean

    if backend is None:
        raise BackendUnavailable(
            "Pulizia neurale: nessun backend indicato. Disponibili: "
            f"{', '.join(neural_clean.available_backends())}; predefiniti per livello: EDGE "
            f"{'+'.join(neural_clean.default_backends(Tier.EDGE))}, WORKSTATION/CLOUD "
            f"{'+'.join(neural_clean.default_backends(Tier.WORKSTATION))}.")
    enhancer = (neural_clean.get_enhancer(backend, device=device, local_files_only=local_files_only)
                if isinstance(backend, str) else backend)
    audio, sr = load_mono(input_path)
    y, params = neural_clean.enhance(audio, sr, enhancer, **kwargs)
    params.update(write_mono(output_path, y, sr, subtype=subtype))
    params["input_sha256"] = sha256_file(input_path)
    params["output_sha256"] = sha256_file(output_path)
    return params
