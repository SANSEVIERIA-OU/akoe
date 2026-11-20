# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
import json

import numpy as np
import pytest

from akoe.transcribe.candidates import ComparisonOutcome, ScorerOutcome, compare_candidates
from akoe.transcribe.ctc_scorer import CTCForcedAlignmentScorer
from tests.fakes import FakeScorer
from tests.test_ctc_scorer import VOCAB, synthetic_emitter

SR = 16000
AUDIO = np.zeros(SR, dtype=np.float32)
CANDS = ["vado a casa", "vado a cassa", "resto qui"]


class NamedFake(FakeScorer):
    def __init__(self, truth, name):
        super().__init__(truth)
        self.name = name


def test_reports_are_separate_and_unassessed_without_thresholds():
    a, b = NamedFake("vado a casa", "a"), NamedFake("resto qui", "b")
    res = compare_candidates(AUDIO, SR, CANDS, [a, b])
    assert res.outcome == ComparisonOutcome.NOT_ASSESSED
    assert [r.scorer for r in res.reports] == ["a", "b"]
    assert [e.score for e in res.reports[0].entries] == [3.0, 2.0, 0.0]
    assert [e.rank for e in res.reports[1].entries] == [2, 3, 1] and res.reports[1].best == "resto qui"
    assert all(e.compatible is None for r in res.reports for e in r.entries)
    assert res.same_best is False and any("diverse" in w for w in res.warnings)


@pytest.mark.parametrize(
    "thr_a,thr_b,expected",
    [
        (10.0, 10.0, ComparisonOutcome.NONE_COMPATIBLE),
        (3.0, 2.0, ComparisonOutcome.SOME_COMPATIBLE),
        (3.0, 10.0, ComparisonOutcome.DISAGREEMENT),
    ],
)
def test_outcomes_with_thresholds(thr_a, thr_b, expected):
    a, b = NamedFake("vado a casa", "a"), NamedFake("resto qui", "b")
    res = compare_candidates(AUDIO, SR, CANDS, [a, b], thresholds={"a": thr_a, "b": thr_b})
    assert res.outcome == expected
    if expected == ComparisonOutcome.NONE_COMPATIBLE:
        assert all(r.outcome == ScorerOutcome.NONE_COMPATIBLE for r in res.reports)
        assert "nessuna compatibile" in res.message


def test_only_assessed_scorers_decide():
    a, b = NamedFake("vado a casa", "a"), NamedFake("resto qui", "b")
    res = compare_candidates(AUDIO, SR, CANDS, [a, b], thresholds={"a": 10.0})
    assert res.reports[1].outcome == ScorerOutcome.NOT_ASSESSED
    assert res.outcome == ComparisonOutcome.NONE_COMPATIBLE


def test_input_validation_and_warnings():
    a = NamedFake("x", "a")
    for bad in ([], ["x", "x"], ["x", "  "]):
        with pytest.raises(ValueError):
            compare_candidates(AUDIO, SR, bad, [a])
    with pytest.raises(ValueError, match="sconosciuti"):
        compare_candidates(AUDIO, SR, ["x"], [a], thresholds={"zzz": 0.0})
    with pytest.raises(ValueError, match="duplicati"):
        compare_candidates(AUDIO, SR, ["x"], [a, NamedFake("y", "a")])
    many = compare_candidates(AUDIO, SR, [f"c{i}" for i in range(7)], [a])
    assert any("più di 5" in w for w in many.warnings)


def test_ctc_details_are_kept_and_json_serializable():
    ctc = CTCForcedAlignmentScorer.from_emission_fn(synthetic_emitter("vado a casa"), VOCAB, name="synthetic")
    res = compare_candidates(AUDIO, SR, CANDS + ["99"], [ctc], thresholds={ctc.name: -0.5})
    rep = res.reports[0]
    assert rep.best == "vado a casa" and rep.entries[0].compatible is True
    assert rep.entries[3].score is None and rep.entries[3].compatible is False  # unalignable
    words = rep.entries[0].details["words"]
    assert [w["word"] for w in words] == ["vado", "a", "casa"]
    assert rep.entries[0].details["relative_per_frame"] == pytest.approx(0.0, abs=1e-9)
    json.loads(res.model_dump_json())  # -inf never leaks into JSON
    assert any("non valutabili" in w for w in res.warnings)


@pytest.mark.slow
def test_real_ctc_and_whisper_reported_separately():
    import os

    from akoe.transcribe.whisper_backend import WhisperDecoderScorer, WhisperHFBackend
    from tests.real_audio import italian_example

    audio, sr, text = italian_example()
    ctc = CTCForcedAlignmentScorer(os.environ.get("AKOE_TEST_CTC_MODEL",
                                                  "facebook/wav2vec2-base-10k-voxpopuli-ft-it"))
    wh = WhisperDecoderScorer(WhisperHFBackend("openai/whisper-tiny"))
    cands = [text, "Oggi piove molto forte in montagna"]
    res = compare_candidates(audio, sr, cands, [ctc, wh], thresholds={ctc.name: -0.5, wh.name: -1.5})
    assert [r.best for r in res.reports] == [text, text] and res.same_best
    assert res.outcome == ComparisonOutcome.SOME_COMPATIBLE
    assert [e.compatible for e in res.reports[0].entries] == [True, False]
    assert all(r.provenance and r.provenance["weights"] for r in res.reports)
    none = compare_candidates(audio, sr, cands[1:], [ctc, wh], thresholds={ctc.name: -0.5, wh.name: -1.5})
    assert none.outcome == ComparisonOutcome.NONE_COMPATIBLE
