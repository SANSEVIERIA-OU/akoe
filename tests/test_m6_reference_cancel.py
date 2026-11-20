"""Reference cancellation of radio/music (issue #32)."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy import signal

from forensicaudio.stages import reference_cancel as rc
from tests import m6_synth as s

SR = s.SR
DUR = 10.0
SEGS = [(2.0, 3.5), (5.0, 6.2), (7.5, 9.0)]
OFFSET_S = 1.3  # the reference starts 1.3 s before the recording


@pytest.fixture(scope="module")
def scene():
    speech = s.speech_like(DUR, SEGS, seed=1, level=0.05)
    reference = s.music_like(DUR + 2.0, seed=4, level=0.1)
    rir = s.cabin_rir(seed=5)  # reverberant energy close to the direct sound
    off = int(OFFSET_S * SR)
    music_at_mic = signal.oaconvolve(reference, rir)[off:off + int(DUR * SR)]
    noise = s.colored_noise(DUR, seed=6, level=0.003, slope_db_per_octave=-3)
    return speech, reference, speech + music_at_mic + noise


@pytest.fixture(scope="module")
def protected(scene):
    _, reference, x = scene
    return rc.cancel_reference(x, SR, reference, speech_segments=SEGS)


def test_alignment_finds_the_offset(scene) -> None:
    _, reference, x = scene
    lag, info = rc.align_reference(x, reference, SR)
    assert lag == -int(OFFSET_S * SR)
    assert info["peak_to_median_db"] > 30


def test_adaptive_filter_cancels_reverberant_music(scene, protected) -> None:
    speech, _, x = scene
    y, params = protected
    rep = params["report"]
    assert rep["xcorr_reduction_db"] >= 20.0
    assert rep["coherence_reduction_db"] >= 15.0
    assert rep["erle_non_speech_db"] >= 20.0
    # Simple aligned subtraction is not enough with cabin reverberation.
    assert rep["simple_subtraction_erle_non_speech_db"] < 8.0
    assert s.snr_db(speech, y) - s.snr_db(speech, x) >= 20.0
    assert rep["possible_speech_leakage"] is False and params["warnings_it"] == []
    assert params["filter_taps"] >= 0.12 * SR
    s.assert_loggable(params, "reference_cancel")


def test_freezing_adaptation_during_speech_protects_the_voice(scene, protected) -> None:
    speech, reference, x = scene
    y_free, p_free = rc.cancel_reference(x, SR, reference, speech_segments=[])
    y_prot, _ = protected
    assert s.snr_db(speech, y_prot) > s.snr_db(speech, y_free) + 2.0
    assert p_free["speech_segments_source"] == "provided"

    y_auto, p_auto = rc.cancel_reference(x, SR, reference)  # VAD on first-pass residual
    assert p_auto["speech_segments_source"].startswith("vad.detect_speech")
    assert s.snr_db(speech, y_auto) > s.snr_db(speech, y_free) + 2.0


def test_speech_leaking_into_the_reference_is_flagged(scene) -> None:
    speech, reference, x = scene
    leaky = reference.copy()
    off = int(OFFSET_S * SR)
    leaky[off:off + len(speech)] += 0.5 * speech
    _, params = rc.cancel_reference(x, SR, leaky, speech_segments=SEGS)
    assert params["report"]["possible_speech_leakage"] is True
    assert any("voce del soggetto" in w for w in params["warnings_it"])


def test_resampled_reference_and_reproducibility(scene, protected, tmp_path: Path) -> None:
    speech, reference, x = scene
    ref44 = signal.resample_poly(reference, 441, 160)
    y44, p44 = rc.cancel_reference(x, SR, ref44, 44100, speech_segments=SEGS)
    assert p44["reference_resampling"] == {"method": "scipy.signal.resample_poly",
                                           "up": 160, "down": 441}
    assert s.snr_db(speech, y44) - s.snr_db(speech, x) >= 20.0

    y_again, p_again = rc.cancel_reference(x.copy(), SR, reference.copy(), speech_segments=SEGS)
    assert np.array_equal(protected[0], y_again) and protected[1] == p_again

    src, ref_path = tmp_path / "in.wav", tmp_path / "ref.wav"
    sf.write(src, x.astype(np.float32), SR, subtype="FLOAT")
    sf.write(ref_path, reference.astype(np.float32), SR, subtype="FLOAT")
    params = rc.run(src, ref_path, tmp_path / "out.wav", speech_segments=SEGS)
    assert params["reference_sha256"] and params["output_sha256"]
