# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Offline noise reduction with a whole-recording noise profile (issue #30)."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from akoe.stages import denoise
from tests import m6_synth as s

SR = s.SR
DUR = 10.0
SEGS = [(1.0, 2.6), (4.0, 5.2), (7.0, 9.0)]


@pytest.fixture(scope="module")
def signals():
    speech = s.speech_like(DUR, SEGS, seed=1, level=0.1)
    noise = (s.colored_noise(DUR, seed=3, level=0.05, slope_db_per_octave=-3)
             + s.colored_noise(DUR, seed=5, level=0.01, slope_db_per_octave=0))
    return speech, noise


def _noise_only_mask(n: int) -> np.ndarray:
    mask = np.ones(n, dtype=bool)
    for a, b in SEGS:
        mask[int((a - 0.2) * SR):int((b + 0.2) * SR)] = False
    return mask


def test_wiener_with_whole_recording_profile_improves_snr(signals) -> None:
    speech, noise = signals
    x = speech + noise
    y, params = denoise.reduce_noise_offline(x, SR, speech_segments=SEGS)
    assert s.snr_db(speech, y) - s.snr_db(speech, x) >= 5.0
    quiet = _noise_only_mask(len(x))
    reduction = 10 * np.log10(np.sum(x[quiet] ** 2) / np.sum(y[quiet] ** 2))
    assert reduction == pytest.approx(-params["gain_floor_db"], abs=1.5)  # down to the floor
    assert params["noise_profile_source"] == "non_speech_frames"
    assert params["noise_seconds"] > 3.0
    s.assert_loggable(params, "denoise_offline")


def test_profile_from_robust_vad_when_segments_not_given(signals) -> None:
    speech, noise = signals
    x = speech + noise
    y, params = denoise.reduce_noise_offline(x, SR)
    assert params["speech_segments_source"] == "vad.detect_speech"
    assert params["vad_params"]["n_segments"] == len(SEGS)
    assert s.snr_db(speech, y) - s.snr_db(speech, x) >= 5.0


def test_spectral_subtraction_variant(signals) -> None:
    speech, noise = signals
    x = speech + noise
    y, params = denoise.reduce_noise_offline(x, SR, speech_segments=SEGS,
                                             method="spectral_subtraction")
    assert s.snr_db(speech, y) - s.snr_db(speech, x) >= 3.0
    assert params["oversubtraction"] == 1.0 and params["dd_alpha"] is None


def test_fallback_and_provided_profile(signals) -> None:
    speech, noise = signals
    x = speech + noise
    _, params = denoise.reduce_noise_offline(x, SR, speech_segments=[(0.0, DUR)])
    assert params["noise_profile_source"] == "quietest_frames_fallback"
    assert "warning_it" in params

    profile, _ = denoise.estimate_noise_profile(noise, SR, speech_segments=[])
    y, params = denoise.reduce_noise_offline(x, SR, noise_profile=profile)
    assert params["noise_profile_source"] == "provided"
    assert s.snr_db(speech, y) > s.snr_db(speech, x)
    with pytest.raises(ValueError):
        denoise.reduce_noise_offline(x, SR, noise_profile=profile[:-1])


def test_offline_denoise_is_bit_reproducible_and_legacy_still_works(signals,
                                                                   tmp_path: Path) -> None:
    speech, noise = signals
    x = speech + noise
    y1, p1 = denoise.reduce_noise_offline(x, SR, speech_segments=SEGS)
    y2, p2 = denoise.reduce_noise_offline(x.copy(), SR, speech_segments=SEGS)
    assert np.array_equal(y1, y2) and p1 == p2

    src = tmp_path / "in.wav"
    sf.write(src, x.astype(np.float32), SR, subtype="FLOAT")
    params = denoise.run_deterministic_offline(src, tmp_path / "a.wav", speech_segments=SEGS)
    again = denoise.run_deterministic_offline(src, tmp_path / "b.wav", speech_segments=SEGS)
    assert params["output_sha256"] == again["output_sha256"]

    denoise.run_deterministic(src, tmp_path / "legacy.wav")  # old stage untouched
    assert (tmp_path / "legacy.wav").exists()
