# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Score calibration without the self-assessment holdout (issue #49)."""
import json

import numpy as np

from akoe.models import DatasetSplit
from akoe.speaker.profile import (
    LORO_SCHEME,
    ProfileBuildConfig,
    build_speaker_profile,
    calibrate_scores,
    calibration_scores,
    loro_calibration_scores,
    ProfileScorer,
)
from tests.fakes import FakeVoiceprint
from tests.synth_voice import SPEAKER_B, SR, speaker_voice
from tests.test_speaker_profile import material  # noqa: F401  (fixture)


class SpyVoiceprint(FakeVoiceprint):
    def __init__(self):
        self.lengths = []

    def embed(self, audio, sr):
        self.lengths.append(len(audio))
        return super().embed(audio, sr)


def test_loro_uses_only_training_material(material):  # noqa: F811
    case, ds = material
    spy = SpyVoiceprint()
    res = loro_calibration_scores(case, ds, spy)
    train_items = ds.select_items(splits=(DatasetSplit.TRAIN,), verified_only=True)
    train_recs = {r.recording_id for r in ds.recordings if r.split == DatasetSplit.TRAIN}
    train_spans = [s for s in ds.other_speaker_spans if s.recording_id in train_recs]
    holdout_spans = [s for s in ds.other_speaker_spans if s.recording_id not in train_recs]
    assert train_spans and holdout_spans
    # every embedding computed comes from a training item or a training other-speaker span
    assert len(spy.lengths) == len(train_items) + len(train_spans)
    assert len(res.same) == len(train_items) and len(res.other) == len(train_spans)
    assert res.material["scheme"] == LORO_SCHEME and res.material["holdout_used"] is False
    assert any("autovalutazione" in w for w in res.warnings)  # holdout spans left for the self-assessment
    # the subject scores well against centroids that never saw his recording; others do not
    assert np.median(res.same) > np.median(res.other)


def test_loro_scores_are_not_the_fixed_centroid_scores(material):  # noqa: F811
    case, ds = material
    vp = FakeVoiceprint()
    prof = build_speaker_profile(case, ds, vp, config=ProfileBuildConfig(compute_timbre=False))
    fixed = ProfileScorer.load(case, prof.profile, vp)
    loro = loro_calibration_scores(case, ds, vp)
    train_items = ds.select_items(splits=(DatasetSplit.TRAIN,), verified_only=True)
    # scoring training items against centroids that include them would be optimistic
    from akoe.speaker._io import cut, load_audio

    fixed_scores = []
    for it in train_items:
        rec = ds.recording(it.recording_id)
        a = cut(load_audio(rec.audio.resolve(case.root), SR), SR, it.start, it.end)
        fixed_scores.append(fixed.score(a, SR).score)
    assert np.mean(fixed_scores) > np.mean(loro.same)


def test_calibration_records_its_scheme_and_warns_on_holdout(material):  # noqa: F811
    case, ds = material
    vp = FakeVoiceprint()
    prof = build_speaker_profile(case, ds, vp, config=ProfileBuildConfig(compute_timbre=False)).profile
    extra = [(speaker_voice(SPEAKER_B, seconds=1.5, seed=s), SR) for s in (1, 2)]
    loro = loro_calibration_scores(case, ds, vp, extra_other=extra)
    assert loro.material["n_extra_other"] == 2
    cal = calibrate_scores(loro.same, loro.other, backend=vp.name, profile=prof, material=loro.material_json(),
                           scheme=LORO_SCHEME, holdout_used=False)
    assert cal.scheme == LORO_SCHEME and cal.holdout_used is False
    assert not any("quota di autovalutazione" in w for w in cal.warnings)
    assert json.loads(cal.material)["holdout_used"] is False

    scorer = ProfileScorer.load(case, prof, vp)
    same, other, desc = calibration_scores(case, ds, scorer)  # old default: holdout
    assert json.loads(desc)["holdout_used"] is True
    old = calibrate_scores(same, other, backend=vp.name, profile=prof, material=desc, scheme="holdout",
                           holdout_used=True)
    assert any("ottimistico" in w for w in old.warnings)
