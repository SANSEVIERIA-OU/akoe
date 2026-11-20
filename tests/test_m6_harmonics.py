# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Engine harmonics: tracking and attenuation (issue #31)."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from akoe.stages import harmonics
from tests import m6_synth as s

SR = s.SR
DUR = 8.0
SEGS = [(1.0, 2.6), (4.0, 5.2), (6.0, 7.4)]


@pytest.fixture(scope="module")
def car():
    speech = s.speech_like(DUR, SEGS, seed=1, level=0.05)
    engine, f0 = s.engine_noise(DUR, seed=2, level=0.2, f0_start=30.0, f0_end=42.0)
    broadband = s.colored_noise(DUR, seed=3, level=0.005, slope_db_per_octave=0)
    return speech, engine, broadband, f0


def _db(a: np.ndarray, b: np.ndarray) -> float:
    return float(10 * np.log10(np.sum(a ** 2) / np.sum(b ** 2)))


def test_tracks_a_slowly_varying_engine_fundamental_under_speech(car) -> None:
    speech, engine, broadband, f0 = car
    track, params = harmonics.track_engine_f0(speech + engine + broadband, SR)
    truth = np.interp(track.times_s, np.arange(len(f0)) / SR, f0)
    err = np.abs(track.f0_hz - truth)
    assert np.median(err) < 0.05 and err.max() < 0.3
    assert params["speech_segments_source"] == "vad.detect_speech"
    assert track.speech.any()  # speech frames were down-weighted
    d = track.to_dict()
    assert len(d["f0_hz"]) == len(d["times_s"]) == params["track_n_frames"]


def test_attenuation_removes_engine_and_spares_speech(car) -> None:
    speech, engine, broadband, _ = car
    x = speech + engine + broadband
    y, params, track = harmonics.attenuate_engine_harmonics(x, SR)
    assert s.snr_db(speech, y) - s.snr_db(speech, x) >= 20.0
    assert params["n_harmonics"] >= 16

    # Same track applied to each component separately.
    y_engine, _, _ = harmonics.attenuate_engine_harmonics(engine, SR, track=track)
    assert _db(engine, y_engine) >= 25.0
    y_speech, _, _ = harmonics.attenuate_engine_harmonics(speech, SR, track=track)
    assert s.snr_db(speech, y_speech) >= 12.0  # narrow notches: little speech damage
    s.assert_loggable(params, "engine_harmonics")


def test_depth_limits_the_attenuation(car) -> None:
    _, engine, _, _ = car
    track, _ = harmonics.track_engine_f0(engine, SR, speech_segments=[])
    y, params, _ = harmonics.attenuate_engine_harmonics(engine, SR, track=track, depth_db=6.0)
    assert _db(engine, y) == pytest.approx(6.0, abs=1.0)
    assert params["subtraction_gain"] == pytest.approx(1 - 10 ** (-6 / 20))


def test_no_engine_no_attenuation(car) -> None:
    speech, _, _, _ = car
    x = speech + s.colored_noise(DUR, seed=3, level=0.05)
    y, params, _ = harmonics.attenuate_engine_harmonics(x, SR)
    assert params["applied"] is False and np.array_equal(y, x)
    assert params["warnings_it"]
    _, forced, _ = harmonics.attenuate_engine_harmonics(x, SR, force=True)
    assert forced["applied"] is True


def test_harmonics_reproducible_and_run_returns_track(car, tmp_path: Path) -> None:
    speech, engine, broadband, _ = car
    x = speech + engine + broadband
    segs = SEGS
    y1, p1, t1 = harmonics.attenuate_engine_harmonics(x, SR, speech_segments=segs)
    y2, p2, t2 = harmonics.attenuate_engine_harmonics(x.copy(), SR, speech_segments=segs)
    assert np.array_equal(y1, y2) and p1 == p2 and np.array_equal(t1.f0_hz, t2.f0_hz)

    src = tmp_path / "in.wav"
    sf.write(src, x.astype(np.float32), SR, subtype="FLOAT")
    params = harmonics.run(src, tmp_path / "out.wav", speech_segments=segs)
    assert params["track"]["f0_hz"] and params["output_sha256"]
    assert params["tracking"]["track_sha256"]
    s.assert_loggable(params, "engine_harmonics")
