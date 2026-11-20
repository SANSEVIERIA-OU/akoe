# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Speaker-informed listening version with supervised NMF (issue #33)."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from akoe.config import StageKind
from akoe.stages import listening
from tests import m6_synth as s

SR = s.SR
SEGS = [(0.5, 2.5), (3.0, 5.5)]


@pytest.fixture(scope="module")
def dictionaries():
    subject_clean = s.speech_like(8.0, [(0.0, 8.0)], seed=11, level=0.1)  # other recording
    engine, _ = s.engine_noise(5.0, seed=12, level=0.1, f0_start=35, f0_end=40)
    car_noise = engine + s.colored_noise(5.0, seed=13, level=0.05)
    return listening.learn_dictionaries(subject_clean, car_noise, SR, n_iter=150, seed=7)


@pytest.fixture(scope="module")
def case_audio():
    speech = s.speech_like(6.0, SEGS, seed=21, level=0.1)  # new utterances, same voice
    engine, _ = s.engine_noise(6.0, seed=22, level=0.1, f0_start=36, f0_end=39)
    noise = engine + s.colored_noise(6.0, seed=23, level=0.05)
    return speech, speech + noise


def test_nmf_listening_version_improves_snr(dictionaries, case_audio) -> None:
    speech, x = case_audio
    y, params = listening.enhance_for_listening(x, SR, dictionaries)
    assert s.snr_db(speech, y) - s.snr_db(speech, x) >= 8.0
    y_semi, _ = listening.enhance_for_listening(x, SR, dictionaries, n_free_noise=4)
    assert s.snr_db(speech, y_semi) - s.snr_db(speech, x) >= 6.0
    s.assert_loggable(params, "listening_nmf")


def test_labelled_deterministic_model_based_for_listening(dictionaries, case_audio) -> None:
    _, x = case_audio
    _, params = listening.enhance_for_listening(x, SR, dictionaries)
    assert listening.KIND == StageKind.DETERMINISTIC
    assert params["kind"] == "deterministic"
    assert params["model_based"] is True
    assert params["intended_use"] == "human_listening"
    assert "ascolto" in params["label_it"]
    assert params["seed"] == 0 and params["dictionaries_record"]["speech"]["seed"] == 7
    assert params["dictionaries_record"]["noise"]["seed"] == 8


def test_bases_are_inspectable_and_saved(dictionaries, tmp_path: Path) -> None:
    for w in (dictionaries.speech, dictionaries.noise):
        assert (w >= 0).all()
        assert np.allclose(w.sum(axis=0), 1.0)
    desc = dictionaries.describe()
    assert len(desc["speech"]) == 32 and len(desc["noise"]) == 16
    # Engine/road bases sit lower in frequency than the voice bases.
    assert (np.median([b["centroid_hz"] for b in desc["noise"]])
            < np.median([b["centroid_hz"] for b in desc["speech"]]))

    ref = dictionaries.save(tmp_path / "dict.npz")
    loaded = listening.NMFDictionaries.load(tmp_path / "dict.npz")
    assert np.array_equal(loaded.speech, dictionaries.speech)
    assert np.array_equal(loaded.noise, dictionaries.noise)
    assert loaded.sha256() == ref["bases_sha256"] == dictionaries.sha256()
    assert loaded.record["speech"]["training_audio_sha256"]


def test_nmf_is_bit_reproducible_given_seeds(dictionaries, case_audio, tmp_path: Path) -> None:
    speech_train = s.speech_like(3.0, [(0.0, 3.0)], seed=11)
    noise_train = s.colored_noise(2.0, seed=13)
    a = listening.learn_dictionaries(speech_train, noise_train, SR, n_speech=8, n_noise=4,
                                     n_iter=30, seed=3)
    b = listening.learn_dictionaries(speech_train, noise_train, SR, n_speech=8, n_noise=4,
                                     n_iter=30, seed=3)
    c = listening.learn_dictionaries(speech_train, noise_train, SR, n_speech=8, n_noise=4,
                                     n_iter=30, seed=4)
    assert a.sha256() == b.sha256() != c.sha256()

    _, x = case_audio
    y1, p1 = listening.enhance_for_listening(x, SR, dictionaries, n_free_noise=2)
    y2, p2 = listening.enhance_for_listening(x.copy(), SR, dictionaries, n_free_noise=2)
    assert np.array_equal(y1, y2) and p1 == p2

    src = tmp_path / "in.wav"
    sf.write(src, x.astype(np.float32), SR, subtype="FLOAT")
    params = listening.run(src, tmp_path / "out.wav", dictionaries)
    assert params["output_sha256"] and params["intended_use"] == "human_listening"
    with pytest.raises(ValueError):
        listening.enhance_for_listening(x, 8000, dictionaries)
