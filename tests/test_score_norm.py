# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""WeSpeaker scoring recipe: mean subtraction, AS-norm, QMF (issue #57)."""
import json
import math

import numpy as np
import pytest

from akoe.speaker.profile import (
    ProfileScorer,
    build_speaker_profile,
    calibrate_scores,
    load_normalization,
    loro_calibration_scores,
    save_calibration,
    save_normalization,
)
from akoe.speaker.score_norm import (
    CohortEntry,
    NormScore,
    ScoreNormalization,
    ScoreNormalizer,
    ScoringRecipe,
    as_norm,
    fit_qmf,
    topk_stats,
    unit,
    unit_rows,
)
from tests.fakes import FakeVoiceprint
from tests.synth_voice import SPEAKER_A, SPEAKER_B, SR, bandlimit, speaker_voice
from tests.test_speaker_profile import FAST, _car_noise, material  # noqa: F401  (fixture)


def _rand_unit(rng, n, d=16):
    return unit_rows(rng.normal(size=(n, d)))


def test_topk_stats_and_as_norm_match_the_wespeaker_formula():
    rng = np.random.default_rng(0)
    cohort = _rand_unit(rng, 50)
    v = unit(rng.normal(size=16))
    s = np.sort(cohort @ v)[::-1][:7]
    assert topk_stats(v, cohort, 7) == pytest.approx((s.mean(), s.std()))
    assert topk_stats(v, cohort, 500)[0] == pytest.approx(float((cohort @ v).mean()))  # k capped at cohort size
    me, se, mt, st = 0.2, 0.05, 0.1, 0.1
    assert as_norm(0.6, (me, se), (mt, st)) == pytest.approx(0.5 * ((0.6 - me) / se + (0.6 - mt) / st))


def test_normalizer_subtracts_the_cohort_mean_and_picks_the_best_condition():
    rng = np.random.default_rng(1)
    bias = np.zeros(16)
    bias[0] = 3.0  # a channel offset shared by everybody
    spk = rng.normal(size=16)
    subject = [("car", unit(spk + bias + 0.3 * rng.normal(size=16)), 2.0) for _ in range(6)]
    subject += [("phone", unit(spk + bias + 0.3 * rng.normal(size=16) + 1.0), 2.0) for _ in range(3)]
    cohort = [CohortEntry(unit(rng.normal(size=16) + bias), f"g{i}", "case_other_speakers", 1.0)
              for i in range(30)]
    recipe = ScoringRecipe(top_k=10, qmf=False)
    norm = ScoreNormalizer.fit(recipe, subject, cohort, use_mean=True, use_as_norm=True)
    mu = unit_rows(np.stack([c.embedding for c in cohort])).mean(axis=0)
    assert np.allclose(norm.mean, mu)
    test = unit(spk + bias + 0.3 * rng.normal(size=16))
    ns = norm.score(test, seconds=1.5)
    t = unit(test - mu)
    assert ns.cosine == pytest.approx(float(t @ norm.centroids[ns.condition_key]))
    assert ns.kind == "asnorm" and set(ns.by_condition) == {"car", "phone"}
    assert ns.score == max(ns.by_condition.values())
    assert ns.features["log_duration_test"] == pytest.approx(math.log(1.5))
    assert {"imposter_mean_test", "imposter_mean_enroll"} <= set(ns.features)
    other = unit(rng.normal(size=16) + bias)
    assert norm.score(other).score < ns.score
    # a forced condition is respected
    assert norm.score(test, condition_key="phone").condition_key == "phone"
    assert norm.steps == ["mean_subtraction", "as_norm"]


def test_qmf_fits_on_trials_and_is_skipped_with_too_few():
    rng = np.random.default_rng(2)
    trials = []
    for lab in (1, 0):
        for _ in range(30):
            s = rng.normal(2.0 if lab else 0.0, 0.7)
            trials.append((lab, NormScore(score=s, kind="asnorm", condition_key="c", cosine=0.0, features={
                "score": s, "log_duration_test": math.log(rng.uniform(0.6, 8.0)),
                "imposter_mean_test": rng.normal(0.3, 0.05), "imposter_mean_enroll": 0.3})))
    model, _ = fit_qmf(ScoringRecipe(), trials)
    assert model is not None and model.coef[0] > 0 and model.n_same == model.n_other == 30
    assert model.features == ["score", "log_duration_test", "imposter_mean_test", "imposter_mean_enroll"]
    hi = model.decision({**trials[0][1].features, "score": 4.0})
    lo = model.decision({**trials[0][1].features, "score": -2.0})
    assert hi > lo
    few, why = fit_qmf(ScoringRecipe(), trials[:5] + trials[-5:])
    assert few is None and "almeno 20" in why
    assert fit_qmf(ScoringRecipe(qmf=False), trials) == (None, None)


def test_recipe_validation():
    with pytest.raises(ValueError):
        ScoringRecipe(cohort_sources=["internet"])
    with pytest.raises(ValueError):
        ScoringRecipe(qmf_features=["log_duration_test"])


def test_artifact_round_trip_gives_the_same_scores():
    rng = np.random.default_rng(3)
    subject = [("car", unit(rng.normal(size=16) + 2), 1.0) for _ in range(5)]
    cohort = [CohortEntry(unit(rng.normal(size=16)), f"g{i}", "cohort_files", 1.0, f"clip{i}") for i in range(12)]
    recipe = ScoringRecipe(top_k=5)
    norm = ScoreNormalizer.fit(recipe, subject, cohort, use_mean=True, use_as_norm=True)
    art = norm.to_artifact(backend="b", subject_id="s", cohort=cohort, subject_items=["i1"])
    back = ScoreNormalizer.from_artifact(ScoreNormalization.model_validate(json.loads(art.model_dump_json(by_alias=True))))
    v = unit(rng.normal(size=16))
    assert back.score(v, 2.0).score == pytest.approx(norm.score(v, 2.0).score)
    assert art.effective_top_k == 5 and art.cohort_labels[0] == "clip0" and art.score_kind == "asnorm"


def _phone_b(seed):
    return bandlimit(speaker_voice(SPEAKER_B, seconds=1.0, seed=seed), 300, 3400)


def test_loro_with_recipe_excludes_the_left_out_group_from_the_cohort(material):  # noqa: F811
    case, ds = material
    cohort_clips = [(_phone_b(400 + k), SR, f"pubblica-{k}") for k in range(12)]
    recipe = ScoringRecipe(top_k=50, min_cohort=5, qmf_min_per_class=50)  # QMF skipped: few trials
    res = loro_calibration_scores(case, ds, FakeVoiceprint(), recipe=recipe, cohort_audio=cohort_clips)
    norm = res.normalization
    assert norm is not None and res.score_kind == "asnorm" and norm.steps == ["mean_subtraction", "as_norm"]
    assert res.material["score_kind"] == "asnorm"
    info = res.material["score_normalization"]
    assert info["cohort_by_source"]["cohort_files"] == 12 and info["cohort_by_source"]["case_other_speakers"] == 2
    assert info["recipe"]["top_k"] == 50 and "left-out" in info["fold_exclusion"]
    assert any("QMF non applicata" in w for w in res.warnings)
    # case spans in the cohort carry their recording group; cohort clips their own
    assert set(norm.cohort_sources) == {"case_other_speakers", "cohort_files"}
    assert all(g.startswith(("rec:", "pair:")) for g, s in zip(norm.cohort_groups, norm.cohort_sources)
               if s == "case_other_speakers")
    plain = loro_calibration_scores(case, ds, FakeVoiceprint())
    assert len(res.same) == len(plain.same) and len(res.other) == len(plain.other) == 2
    assert min(res.same) > max(res.other)


def test_loro_without_cohort_falls_back_to_cosine(material):  # noqa: F811
    case, ds = material
    res = loro_calibration_scores(case, ds, FakeVoiceprint(), recipe=ScoringRecipe(qmf_min_per_class=50))
    assert res.normalization is None and res.score_kind == "cosine"
    assert any("Coorte di altri parlanti troppo piccola" in w for w in res.warnings)
    plain = loro_calibration_scores(case, ds, FakeVoiceprint())
    assert res.same == pytest.approx(plain.same) and res.other == pytest.approx(plain.other)


def test_profile_scorer_uses_the_normalization_of_the_calibration(material):  # noqa: F811
    case, ds = material
    vp = FakeVoiceprint()
    prof = build_speaker_profile(case, ds, vp, config=FAST)
    extra = [(_phone_b(500 + k), SR) for k in range(4)]
    cohort_clips = [(_phone_b(600 + k), SR, f"c{k}") for k in range(10)]
    res = loro_calibration_scores(case, ds, vp, extra_other=extra, recipe=ScoringRecipe(min_cohort=5),
                                  cohort_audio=cohort_clips)
    assert res.normalization is not None
    ref = save_normalization(case, res.normalization)
    assert ref.verify(case.root) and load_normalization(case, ref).steps == res.normalization.steps
    cal = calibrate_scores(res.same, res.other, backend=vp.name, profile=prof.profile, normalization=ref,
                           score_kind=res.score_kind)
    assert cal.score_kind == "asnorm" and "asnorm" in cal.params["features"]
    cal_ref = save_calibration(case, cal)
    assert json.loads(cal_ref.resolve(case.root).read_text())["normalization"]["sha256"] == ref.sha256
    scorer = ProfileScorer.load(case, prof.profile, vp, cal)
    assert scorer.normalizer is not None
    b = scorer.score(_phone_b(700), SR)
    a = scorer.score(speaker_voice(SPEAKER_A, seconds=1.0, seed=701) + _car_noise(SR, 701), SR)
    assert a.score_kind == b.score_kind == "asnorm" and a.cosine is not None
    assert a.score > b.score and "normalizzato" in a.message and "identif" not in a.message.lower()
    direct = scorer.normalizer.score(unit(vp.embed(_phone_b(700), SR)), 1.0)
    assert b.score == pytest.approx(direct.score)
    # a scorer built without the normalization cannot use this calibration
    with pytest.raises(Exception, match="ProfileScorer.load"):
        ProfileScorer(prof.profile, scorer.centroids, vp, cal)
    # a different backend cannot reuse the normalization
    data = load_normalization(case, ref)
    assert data.backend == vp.name


def test_engine_subject_model_normalizes_scores_with_a_cohort(tmp_path):
    from akoe import engine
    from akoe.case import Case
    from akoe.models import ProcessingLog
    from tests.engine_synth import make_evidence, make_subject_folder
    from tests.fakes import FakeProvider
    from tests.synth_voice import write_wav

    case = Case.create(tmp_path / "case", "caso-coorte")
    folder = make_subject_folder(tmp_path / "soggetto")
    cohort = []
    for k in range(12):
        cohort.append(write_wav(tmp_path / f"coorte{k}.wav", speaker_voice(SPEAKER_B, seconds=1.5, seed=800 + k)))
    prov = FakeProvider()
    res = engine.build_subject_model(case, "s1", folder=folder, holdout_fraction=0.3, certify_all_by="op",
                                     provider=prov, cohort_files=cohort,
                                     score_recipe=ScoringRecipe(min_cohort=5, qmf_min_per_class=1000))
    cal_info = res.summary["calibration"]
    assert cal_info["performed"] and cal_info["score_kind"] == "asnorm"
    log = ProcessingLog.model_validate_json(res.log.resolve(case.root).read_text())
    assert log.parameters["score_recipe"]["min_cohort"] == 5 and len(log.parameters["cohort_files"]) == 12
    stage = next(s for s in log.stages if s.name == "score_calibration")
    info = stage.params["material"]["score_normalization"]
    assert info["cohort_by_source"]["cohort_files"] == 12 and info["steps"] == ["mean_subtraction", "as_norm"]
    assert any(a.kind == "score_normalization" for a in log.artifacts)
    assert sum(1 for i in log.inputs if i.kind == "score_cohort") == 12
    ev = case.import_evidence(make_evidence(tmp_path / "reperti")["evidence"], None)
    tr = engine.transcribe(case, ev.evidence_id, subject_id="s1", provider=prov)
    tlog = ProcessingLog.model_validate_json(tr.log.resolve(case.root).read_text())
    sc = next(s for s in tlog.stages if s.name == "speaker_scores")
    assert sc.params["score_kind"] == "asnorm" and sc.params["n_scored"] >= 1
    # plain cosine, as before, when the recipe is disabled
    case2 = Case.create(tmp_path / "case2", "caso-coseno")
    res2 = engine.build_subject_model(case2, "s1", folder=folder, holdout_fraction=0.3, certify_all_by="op",
                                      provider=prov, score_recipe=None)
    assert res2.summary["calibration"].get("score_kind", "cosine") == "cosine"
