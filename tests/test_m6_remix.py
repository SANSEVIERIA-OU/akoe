"""Cleaned/original remix and choice of the transcriber input (issue #34)."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from forensicaudio.config import StageKind
from forensicaudio.stages import remix
from tests import m6_synth as s

SR = s.SR


@pytest.fixture(scope="module")
def original():
    return s.speech_like(4.0, [(0.5, 3.5)], seed=1) + s.colored_noise(4.0, seed=2, level=0.02)


def test_remix_aligns_and_level_matches(original) -> None:
    delay = 37
    cleaned = 0.5 * np.concatenate([np.zeros(delay), original[:-delay]])  # late and quieter
    y, params = remix.remix(cleaned, original, SR, original_fraction=0.2)
    assert params["lag_samples"] == delay
    assert params["cleaned_gain"] == pytest.approx(2.0, rel=1e-3)
    # Aligned and level-matched cleaned equals the original here, so the mix does too.
    core = slice(delay, len(original) - delay)
    assert np.max(np.abs(y[core] - original[core])) < 1e-3
    assert params["variant"] == "remix_0.20" and params["in_recommended_range"] is True
    assert params["audio_provenance"] == "deterministic"
    s.assert_loggable(params, "remix")


def test_remix_fraction_and_provenance(original) -> None:
    rng = np.random.default_rng(0)
    cleaned = rng.standard_normal(len(original)) * 0.01
    y, params = remix.remix(cleaned, original, SR, original_fraction=0.1, align=False,
                            cleaned_kind=StageKind.NEURAL)
    g = params["cleaned_gain"]
    assert np.allclose(y, (0.9 * g * cleaned + 0.1 * original) * params["out_gain"])
    assert params["audio_provenance"] == "neural"  # NEURAL cleaned -> NEURAL remix
    with pytest.raises(ValueError):
        remix.remix(cleaned, original, SR, original_fraction=1.5)


def test_remix_peak_limit_and_run(original, tmp_path: Path) -> None:
    loud = original / np.max(np.abs(original))
    y, params = remix.remix(loud * 3, loud, SR, original_fraction=0.5, level_match=False)
    assert np.max(np.abs(y)) <= 0.99 + 1e-12 and params["out_gain"] < 1.0

    c_path, o_path = tmp_path / "clean.wav", tmp_path / "orig.wav"
    sf.write(c_path, (0.5 * original).astype(np.float32), SR, subtype="FLOAT")
    sf.write(o_path, original.astype(np.float32), SR, subtype="FLOAT")
    p1 = remix.run(c_path, o_path, tmp_path / "a.wav", original_fraction=0.15)
    p2 = remix.run(c_path, o_path, tmp_path / "b.wav", original_fraction=0.15)
    assert p1["output_sha256"] == p2["output_sha256"]
    assert p1["original_sha256"] and p1["cleaned_sha256"]


def test_choice_smallest_wer_wins() -> None:
    d = remix.choose_transcriber_input({"original": 0.30, "remix_0.10": 0.27,
                                        "remix_0.25": 0.28, "cleaned": 0.41})
    assert d.chosen == "remix_0.10"
    assert d.ranking == ["remix_0.10", "remix_0.25", "original", "cleaned"]
    assert "remix_0.10" in d.reason_it
    json.dumps(d.to_dict())


def test_choice_ties_and_small_gains_go_to_original() -> None:
    d = remix.choose_transcriber_input({"remix_0.10": 0.25, "original": 0.25})
    assert d.chosen == "original"
    d = remix.choose_transcriber_input({"original": 0.25, "remix_0.10": 0.249},
                                       min_improvement=0.005)
    assert d.chosen == "original" and "soglia" in d.reason_it
    d = remix.choose_transcriber_input({"original": 0.20, "cleaned": 0.35})
    assert d.chosen == "original"


def test_choice_requires_the_original_and_excludes_invalid() -> None:
    with pytest.raises(ValueError):
        remix.choose_transcriber_input({"cleaned": 0.1, "remix_0.10": 0.2})
    d = remix.choose_transcriber_input({"original": 0.3, "remix_0.10": math.nan,
                                        "remix_0.25": None, "remix_0.15": 0.2})
    assert d.chosen == "remix_0.15"
    assert set(d.excluded) == {"remix_0.10", "remix_0.25"}
    # Deterministic tie-break between non-original variants.
    d = remix.choose_transcriber_input({"original": 0.3, "remix_0.25": 0.2, "remix_0.10": 0.2})
    assert d.chosen == "remix_0.10"
