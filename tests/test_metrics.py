# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
import jiwer
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score, roc_curve

from akoe.assess import metrics as m
from akoe.assess.metrics import (
    CandidateTrial,
    NormalizationConfig,
    candidate_separation,
    cer,
    equal_error_rate,
    expected_calibration_error,
    label_hypothesis_words,
    lexicon_insertion_rate,
    normalize_text,
    oracle_wer,
    reliability_diagram,
    roc_auc,
    unintelligible_quality,
    wer,
    word_error_counts,
)

# -- normalization -------------------------------------------------------------


def test_italian_normalization_defaults():
    assert normalize_text("L'altro giorno, all'una: VADO!") == "l altro giorno all una vado"
    # typographic apostrophe treated like the ASCII one
    assert normalize_text("dell’auto") == "dell auto"
    # accents are kept: "è" and "e" are different words
    assert normalize_text("È così, e basta.") == "è così e basta"


def test_numbers_left_as_written():
    assert normalize_text("Costa 3,5 euro, sono 10.000 persone alle 12:30.") == (
        "costa 3,5 euro sono 10.000 persone alle 12:30"
    )
    # sentence punctuation after a number is still removed; digits are not spelled out
    assert normalize_text("Ne ho 3. Tre!") == "ne ho 3 tre"


def test_apostrophe_modes_and_options():
    assert normalize_text("l'altro", NormalizationConfig(apostrophes="keep")) == "l'altro"
    assert normalize_text("l'altro", NormalizationConfig(apostrophes="join")) == "laltro"
    assert normalize_text("Perché", NormalizationConfig(strip_accents=True)) == "perche"
    assert normalize_text("Ciao", NormalizationConfig(lowercase=False)) == "Ciao"
    cfg = NormalizationConfig(remove_bracketed=True)
    assert normalize_text("vado [incomprensibile] a casa (risata)", cfg) == "vado a casa"
    assert normalize_text("vado [incomprensibile] a casa") == "vado incomprensibile a casa"


# -- WER / CER -----------------------------------------------------------------


def test_wer_counts_substitution_deletion_insertion():
    c = word_error_counts("il gatto nero salta", "il gatto è nero salto")
    assert (c.substitutions, c.deletions, c.insertions, c.hits) == (1, 0, 1, 3)
    assert c.n_ref == 4 and c.n_hyp == 5
    assert wer("il gatto nero salta", "il gatto è nero salto") == pytest.approx(0.5)

    d = word_error_counts("vado a casa domani", "vado casa")
    assert (d.substitutions, d.deletions, d.insertions) == (0, 2, 0)
    assert wer("vado a casa", "") == 1.0
    assert wer("Vado a casa.", "vado a casa") == 0.0


def test_wer_is_corpus_level_not_mean_of_items():
    refs = ["uno due tre quattro", "cinque"]
    hyps = ["uno due tre quattro", "sei"]
    # 1 error over 5 reference words, not mean(0, 1) = 0.5
    assert wer(refs, hyps) == pytest.approx(0.2)


def test_wer_empty_reference_raises():
    with pytest.raises(ValueError):
        wer("", "qualcosa")
    with pytest.raises(ValueError):
        wer(["a"], ["a", "b"])


@pytest.mark.parametrize(
    "ref,hyp",
    [
        ("il gatto nero salta sul tetto", "il gatto è nero sul tetti ieri"),
        ("vado a casa domani mattina presto", "vado a cassa domani"),
        ("allora ci vediamo dopo", "allora allora ci vediamo vediamo dopo"),
        ("non lo so", "no lo so proprio"),
        ("uno due tre", "tre due uno"),
    ],
)
def test_wer_cer_match_jiwer(ref, hyp):
    c = word_error_counts(ref, hyp)
    o = jiwer.process_words(ref, hyp)
    assert wer(ref, hyp) == pytest.approx(o.wer)
    assert c.errors == o.substitutions + o.deletions + o.insertions
    assert cer(ref, hyp) == pytest.approx(jiwer.cer(ref, hyp))


def test_wer_matches_jiwer_on_random_corpus():
    rng = np.random.default_rng(7)
    vocab = "a casa vado domani poi non so che lui lei qui là".split()
    refs, hyps = [], []
    for _ in range(40):
        refs.append(" ".join(rng.choice(vocab, size=rng.integers(1, 12))))
        hyps.append(" ".join(rng.choice(vocab, size=rng.integers(0, 12))))
    hyps = [h if h else "a" for h in hyps]  # jiwer refuses empty hypotheses strings in lists
    assert wer(refs, hyps) == pytest.approx(jiwer.wer(refs, hyps))
    assert cer(refs, hyps) == pytest.approx(jiwer.cer(refs, hyps))


def test_alignment_ops_cover_both_sequences():
    ops = m.align(["a", "b", "c"], ["a", "x", "c", "d"])
    assert [o.op for o in ops] == ["equal", "sub", "equal", "ins"]


# -- N-best --------------------------------------------------------------------


def test_oracle_wer_and_coverage():
    refs = ["vado a casa", "ci vediamo domani"]
    nbest = [["vado a cassa", "vado a casa"], ["ci vediamo dopo", "si vede domani"]]
    r = oracle_wer(refs, nbest)
    assert r.best_indices == [1, 0]
    assert r.oracle_wer == pytest.approx(1 / 6)
    assert r.exact_coverage == pytest.approx(0.5)
    # 1-best WER is never better than oracle
    assert wer(refs, [n[0] for n in nbest]) >= r.oracle_wer


def test_oracle_wer_empty_list_counts_as_deletions():
    r = oracle_wer(["vado a casa"], [[]])
    assert r.oracle_wer == 1.0 and r.best_indices == [None] and r.exact_coverage == 0.0


# -- calibration ---------------------------------------------------------------


def test_ece_zero_when_perfectly_calibrated():
    conf = [0.25] * 4 + [0.75] * 4
    ok = [True, False, False, False] + [True, True, True, False]
    assert expected_calibration_error(conf, ok, n_bins=4) == pytest.approx(0.0)


def test_ece_known_value_and_bins():
    conf = [0.9, 0.9, 0.9, 0.9, 0.1, 0.1]
    ok = [True, True, False, False, False, False]
    # bin 0.9: acc 0.5, |0.5-0.9| = 0.4 (weight 4/6); bin 0.1: acc 0, |0-0.1| = 0.1 (2/6)
    assert expected_calibration_error(conf, ok, n_bins=10) == pytest.approx(4 / 6 * 0.4 + 2 / 6 * 0.1)
    bins = reliability_diagram(conf, ok, n_bins=10)
    assert len(bins) == 10 and sum(b.count for b in bins) == 6
    assert bins[9].count == 4 and bins[9].accuracy == 0.5 and bins[9].mean_confidence == pytest.approx(0.9)
    assert bins[5].count == 0 and bins[5].accuracy is None
    # 1.0 falls in the last bin
    assert reliability_diagram([1.0], [True], n_bins=5)[-1].count == 1


def test_ece_quantile_and_validation():
    conf = list(np.linspace(0.05, 0.95, 20))
    ok = [c > 0.5 for c in conf]
    bins = reliability_diagram(conf, ok, n_bins=4, strategy="quantile")
    assert [b.count for b in bins] == [5, 5, 5, 5]
    assert expected_calibration_error([], []) is None
    # unlabeled words are skipped
    assert expected_calibration_error([0.8, 0.3], [True, None], n_bins=10) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        expected_calibration_error([1.2], [True])


def test_label_hypothesis_words_handles_elision():
    labels = label_hypothesis_words("l'altro giorno vado", ["L'altro", "giorno,", "vado", "via", "..."])
    assert labels == [True, True, True, False, None]
    assert label_hypothesis_words("vado a casa", ["vado", "a", "cassa"]) == [True, True, False]


# -- lexicon bias test ---------------------------------------------------------


def test_lexicon_insertion_rate_counts_only_absent_terms():
    refs = ["vado dal Moretti domani", "ci vediamo al porto", "niente di nuovo"]
    hyps = ["vado dal moretti domani", "ci vediamo al porto vecchio Moretti", "niente di nuovo"]
    r = lexicon_insertion_rate(refs, hyps, ["Moretti", "porto vecchio"])
    # opportunities: Moretti absent in items 2,3; "porto vecchio" absent in all three
    assert r.n_opportunities == 5
    assert r.n_insertions == 2  # Moretti and "porto vecchio" inserted in item 2
    assert r.rate == pytest.approx(2 / 5)
    assert r.n_items_at_risk == 3 and r.item_rate == pytest.approx(1 / 3)
    assert r.per_term == {"Moretti": (1, 2), "porto vecchio": (1, 3)}


def test_lexicon_insertion_rate_without_opportunities():
    r = lexicon_insertion_rate(["Moretti"], ["Moretti"], ["Moretti"])
    assert r.rate is None and r.n_opportunities == 0


# -- voiceprint ----------------------------------------------------------------


def test_eer_perfect_separation_and_full_overlap():
    e = equal_error_rate([0.8, 0.9, 0.95], [0.1, 0.2, 0.3])
    assert e.eer == 0.0
    assert 0.3 < e.threshold <= 0.8
    same = equal_error_rate([0.5, 0.5], [0.5, 0.5])
    assert same.eer == pytest.approx(0.5)
    with pytest.raises(ValueError):
        equal_error_rate([], [0.1])


def test_eer_matches_sklearn_roc():
    rng = np.random.default_rng(3)
    tgt, non = rng.normal(1.0, 1.0, 400), rng.normal(-1.0, 1.0, 600)
    e = equal_error_rate(tgt, non)
    y = np.r_[np.ones(tgt.size), np.zeros(non.size)]
    fpr, tpr, _ = roc_curve(y, np.r_[tgt, non])
    i = np.argmin(np.abs(fpr - (1 - tpr)))
    assert e.eer == pytest.approx((fpr[i] + 1 - tpr[i]) / 2, abs=0.01)
    far = np.mean(non >= e.threshold)
    frr = np.mean(tgt < e.threshold)
    assert far == pytest.approx(e.eer, abs=0.01) and frr == pytest.approx(e.eer, abs=0.01)


def test_roc_auc_matches_sklearn():
    rng = np.random.default_rng(5)
    pos, neg = rng.normal(0.5, 1, 50).round(1), rng.normal(0, 1, 70).round(1)  # rounding creates ties
    y = np.r_[np.ones(50), np.zeros(70)]
    assert roc_auc(pos, neg) == pytest.approx(roc_auc_score(y, np.r_[pos, neg]))
    assert roc_auc([], [1.0]) is None


# -- candidates and unintelligible ---------------------------------------------


def test_candidate_separation():
    trials = [
        CandidateTrial(2.0, (1.0, 0.0)),  # win
        CandidateTrial(1.0, (1.0, 0.0)),  # tie with best altered
        CandidateTrial(0.0, (1.0, 2.0)),  # loss
    ]
    s = candidate_separation(trials)
    assert s.n_trials == 3
    assert s.top1_rate == pytest.approx(1 / 3)
    assert s.tie_rate == pytest.approx(1 / 3)
    assert s.pairwise_auc == pytest.approx((1 + 1 + 0.5 + 1 + 0 + 0) / 6)
    assert s.mean_margin == pytest.approx((1 + 0 - 2) / 3)
    assert candidate_separation([]).top1_rate is None


def test_unintelligible_quality():
    q = unintelligible_quality([True, True, False, False], [0.8, 0.1, 0.9, 0.0], wer_threshold=0.5)
    assert q.flag_rate == 0.5 and q.precision == 0.5 and q.recall == 0.5
    assert unintelligible_quality([], []).flag_rate is None
