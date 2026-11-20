import json

import librosa
import numpy as np
import pytest

from forensicaudio.speaker.timbre import TimbreParams, TimbreSource, compute_timbre, third_octave_bands
from tests.synth_voice import SPEAKER_A, SPEAKER_B, SR, bandlimit, speaker_voice


@pytest.fixture(scope="module")
def voice_a():
    return speaker_voice(SPEAKER_A, seconds=16.0, seed=1)


@pytest.fixture(scope="module")
def profile_a(voice_a):
    return compute_timbre([TimbreSource(voice_a, SR, label="a")])


def _band(out, centre):
    centres = out["ltas"]["band_centres_hz"]
    i = min(range(len(centres)), key=lambda k: abs(centres[k] - centre))
    return out["ltas"]["levels_rel_db"][i]


def test_third_octave_bands():
    bands = third_octave_bands(100, 8000, 8000)
    centres = [round(fc) for _, fc, _ in bands]
    assert centres[0] == 100 and centres[-1] == 6310  # 8 kHz band's upper edge is above Nyquist
    assert len(third_octave_bands(100, 8000, 4000)) < len(bands)


def test_f0_and_formants_of_a_known_synthetic_voice(profile_a):
    f0 = profile_a["f0"]
    assert f0["hz"]["median"] == pytest.approx(SPEAKER_A["f0"], rel=0.03)
    assert f0["stable"] is True and f0["seconds_speech"] >= 10.0
    assert 0 < f0["seconds_voiced"] <= f0["seconds_speech"]
    assert f0["range_p5_p95_hz"] > 0
    for name, (true_f, _) in zip(("F1", "F2", "F3"), SPEAKER_A["formants"]):
        assert profile_a["formants"][name]["median"] == pytest.approx(true_f, rel=0.10)
    assert profile_a["formants"]["frames_with_3_formants"] >= 50
    assert profile_a["warnings"] == []


def test_short_material_is_not_stable_and_says_so():
    y = speaker_voice(SPEAKER_B, seconds=4.0, seed=2)
    out = compute_timbre([TimbreSource(y, SR, label="b")])
    assert out["f0"]["hz"]["median"] == pytest.approx(SPEAKER_B["f0"], rel=0.03)
    assert out["f0"]["stable"] is False
    assert any("F0 non ancora stabile" in w for w in out["warnings"])


def test_output_is_json_serializable_and_records_every_parameter(profile_a):
    text = json.dumps(profile_a, allow_nan=False)
    assert json.loads(text) == profile_a
    params = profile_a["params"]
    for key in TimbreParams.model_fields:
        assert key in params
    assert params["lpc_order"] == 10 and params["formant_sr"] == 11000
    assert set(profile_a["software"]) == {"librosa", "numpy", "scipy"}
    assert "noise floor" in profile_a["gate"]["criterion"]


def test_deterministic(voice_a, profile_a):
    assert compute_timbre([TimbreSource(voice_a.copy(), SR, label="a")]) == profile_a


def test_gate_uses_only_subject_spans_above_the_noise_floor():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.003, 3 * SR).astype(np.float32)
    y = np.concatenate([noise, speaker_voice(SPEAKER_A, seconds=3.0, seed=3)])
    out = compute_timbre([TimbreSource(y, SR, spans=[(3.0, 6.0)], label="mix")])
    gated = out["gate"]["seconds_gated"]
    assert 1.5 < gated <= 3.0
    assert out["sources"][0]["estimated_snr_db"] > 20
    only_noise = compute_timbre([TimbreSource(y, SR, spans=[(0.0, 3.0)], label="noise")])
    assert only_noise["gate"]["seconds_gated"] == 0.0
    assert any("spettro medio non calcolabile" in w for w in only_noise["warnings"])
    assert any("SNR stimato" in w for w in only_noise["warnings"])
    assert only_noise["f0"]["hz"] == {"n": 0}


def test_ltas_reflects_the_channel_and_not_the_gain(voice_a):
    y = voice_a[: 4 * SR]
    full = compute_timbre([TimbreSource(y, SR)])
    quiet = compute_timbre([TimbreSource(y * 0.1, SR)])
    assert quiet["ltas"]["levels_rel_db"] == pytest.approx(full["ltas"]["levels_rel_db"], abs=0.05)
    assert quiet["ltas"]["overall_level_db"] == pytest.approx(full["ltas"]["overall_level_db"] - 20, abs=0.1)

    phone = compute_timbre([TimbreSource(bandlimit(y, 300, 3400), SR)])
    assert _band(full, 5000) - _band(phone, 5000) > 15
    assert _band(full, 125) - _band(phone, 125) > 15
    assert phone["ltas"]["spectral_centroid_hz"] != full["ltas"]["spectral_centroid_hz"]

    # 8 kHz material: no band above its Nyquist, resampled with a recorded method.
    y8 = librosa.resample(y.astype(np.float64), orig_sr=SR, target_sr=8000, res_type="soxr_hq")
    narrow = compute_timbre([TimbreSource(y8, 8000)])
    assert narrow["ltas"]["nyquist_limit_hz"] == 4000
    assert max(hi for _, hi in narrow["ltas"]["band_edges_hz"]) <= 4000
    assert narrow["sources"][0]["input_sr"] == 8000


def test_no_sources_gives_warnings_not_errors():
    out = compute_timbre([])
    assert out["f0"]["stable"] is False and out["formants"]["frames_with_3_formants"] == 0
    assert out["warnings"]
