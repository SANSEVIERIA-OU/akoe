"""Robust VAD for car noise (issue #35); the energy VAD stays unchanged."""
import inspect
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from forensicaudio.stages import vad
from tests import m6_synth as s

SR = s.SR
DUR = 12.0
SEGS = [(1.0, 2.6), (4.0, 5.2), (7.0, 9.0), (10.5, 11.3)]
BURST = (5.8, 6.6)


def _car_recording(burst_slope: float = -4.5) -> np.ndarray:
    speech = s.speech_like(DUR, SEGS, seed=1, level=0.1)
    engine, _ = s.engine_noise(DUR, seed=2, level=0.2)
    road = s.colored_noise(DUR, seed=3, level=0.1)
    burst = s.colored_noise(DUR, seed=4, level=0.3, slope_db_per_octave=burst_slope)
    gate = np.zeros(len(burst))
    gate[int(BURST[0] * SR):int(BURST[1] * SR)] = 1.0
    return speech + engine + road + burst * gate


def _frames(segments) -> np.ndarray:
    return s.segments_to_frame_truth(segments, int(DUR * 100), 0.01)


def test_energy_vad_is_unchanged() -> None:
    sig = inspect.signature(vad.run)
    assert list(sig.parameters) == ["input_path", "frame_ms", "threshold_db", "min_speech_s"]
    assert sig.parameters["threshold_db"].default == -40.0


def test_energy_vad_fails_in_car_noise_robust_vad_does_not(tmp_path: Path) -> None:
    x = _car_recording()
    path = tmp_path / "car.wav"
    sf.write(path, x.astype(np.float32), SR, subtype="FLOAT")
    truth = _frames(SEGS)

    old = _frames(vad.run(path))
    assert old.mean() > 0.95  # car noise is above -40 dBFS: everything is "speech"

    segments, params = vad.run_robust(path)
    pred = _frames(segments)
    assert pred[truth].mean() >= 0.95  # recall
    far = ~_frames([(a - 0.3, b + 0.3) for a, b in SEGS])
    assert pred[far].mean() <= 0.02  # false alarms away from speech (hangover excluded)
    assert params["n_segments"] == len(segments)
    assert params["input_sha256"]
    s.assert_loggable(params, "vad_robust")


@pytest.mark.parametrize("slope", [-4.5, -1.5])
def test_loud_road_noise_burst_is_not_speech(slope: float) -> None:
    segments, _ = vad.detect_speech(_car_recording(slope), SR)
    assert not _frames(segments)[_frames([BURST])].any()


def test_robust_vad_is_reproducible_and_logs_parameters() -> None:
    x = _car_recording()
    a = vad.detect_speech(x, SR)
    b = vad.detect_speech(x.copy(), SR)
    assert a == b
    params = a[1]
    for key in ("band_hz", "snr_threshold_db", "flatness_threshold", "hangover_ms",
                "floor_window_s", "versions", "algorithm_version"):
        assert key in params


def test_robust_vad_short_input() -> None:
    segments, params = vad.detect_speech(np.zeros(100), SR)
    assert segments == [] and params["n_segments"] == 0
