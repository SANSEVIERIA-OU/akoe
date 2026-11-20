# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""SII-motivated band shaping (issue #29)."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from akoe.stages import bands
from tests import m6_synth as s

SR = s.SR


def _white(seconds: float = 4.0) -> np.ndarray:
    return np.random.default_rng(0).standard_normal(int(seconds * SR)) * 0.05


def test_sii_weights_are_the_ansi_octave_values() -> None:
    assert bands.SII_OCTAVE_WEIGHTS == {250: 0.0617, 500: 0.1671, 1000: 0.2373, 2000: 0.2648,
                                        4000: 0.2142, 8000: 0.0549}
    assert sum(bands.SII_OCTAVE_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-4)


def test_measured_band_changes_follow_the_design() -> None:
    x = _white()
    y, params = bands.shape(x, SR, sii_strength=1.0)
    before, after = bands.octave_band_levels(x, SR), bands.octave_band_levels(y, SR)
    for band in ("500", "1000", "2000", "4000"):
        assert after[band] - before[band] == pytest.approx(params["gain_db"][band], abs=1.0)
    # Least important bands are cut, the most important one is untouched.
    assert params["gain_db"]["2000"] == 0.0
    assert params["gain_db"]["250"] < -5.0
    assert params["realised_gain_db"]["1000"] == pytest.approx(params["gain_db"]["1000"], abs=0.3)


def test_filtering_is_zero_phase() -> None:
    t = np.arange(SR) / SR
    tone = np.sin(2 * np.pi * 1000 * t)
    y, params = bands.shape(tone, SR)
    mid = slice(SR // 4, 3 * SR // 4)
    gain = np.dot(y[mid], tone[mid]) / np.dot(tone[mid], tone[mid])
    assert np.max(np.abs(y[mid] - gain * tone[mid])) < 1e-6  # no delay, no phase shift
    assert params["zero_phase"] is True


def test_highpass_removes_engine_rumble() -> None:
    t = np.arange(2 * SR) / SR
    rumble = np.sin(2 * np.pi * 40 * t)
    y, params = bands.shape(rumble, SR, highpass_hz=80.0)
    mid = slice(SR // 2, 3 * SR // 2)
    assert 10 * np.log10(np.mean(rumble[mid] ** 2) / np.mean(y[mid] ** 2)) > 30
    assert params["highpass"]["order"] == 4


def test_subject_spectrum_and_audibility_terms() -> None:
    ltas = {250: -10.0, 500: -3.0, 1000: 0.0, 2000: -6.0, 4000: -20.0, 8000: -30.0}
    base = bands.design_band_gains()
    subj = bands.design_band_gains(subject_ltas=ltas)
    assert subj["terms_db"]["subject"]["1000"] == 0.0
    assert subj["gain_db"]["4000"] < base["gain_db"]["4000"] - 5
    assert max(subj["gain_db"].values()) == 0.0  # normalised: no boost

    segs = [(1.0, 2.5), (3.5, 5.0)]
    x = s.speech_like(6.0, segs, seed=1) + s.colored_noise(6.0, seed=3, level=0.1)
    snr = bands.estimate_band_snr(x, SR, segs)
    assert snr["1000"] > 10 and snr["8000"] < -15
    aud = bands.design_band_gains(band_snr_db=snr)
    assert aud["terms_db"]["audibility"]["1000"] == 0.0
    assert aud["gain_db"]["8000"] < base["gain_db"]["8000"] - 5


def test_shape_is_reproducible_and_run_logs_everything(tmp_path: Path) -> None:
    x = _white(2.0)
    kw = dict(subject_ltas={500: -2.0, 1000: 0.0, 4000: -12.0})
    y1, p1 = bands.shape(x, SR, **kw)
    y2, p2 = bands.shape(x.copy(), SR, **kw)
    assert np.array_equal(y1, y2) and p1 == p2

    src, dst = tmp_path / "in.wav", tmp_path / "out.wav"
    sf.write(src, x.astype(np.float32), SR, subtype="FLOAT")
    params = bands.run(src, dst, **kw)
    assert dst.exists() and params["output_sha256"] and params["input_sha256"]
    for key in ("sii_weights", "gain_db", "terms_db", "fir", "highpass", "versions"):
        assert key in params
    s.assert_loggable(params, "bands")
    assert bands.run(src, tmp_path / "out2.wav", **kw)["output_sha256"] == params["output_sha256"]
