# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
import json

import pytest

from akoe.case import Case
from akoe.config import Channel, RecordingSource
from akoe.models import DatasetSplit, RecordingCondition
from akoe.speaker.dataset import DatasetError, SubjectDataset
from akoe.speaker.profile import ProfileBuildConfig, load_profile
from akoe.speaker.verification import (
    ItemReview,
    Review,
    ReviewDecision,
    apply_review,
    eligible_items,
    read_log,
    update_profile,
    verify_log,
)
from tests.fakes import FakeVoiceprint
from tests.synth_voice import SPEAKER_A, speaker_voice, write_wav

CAR = RecordingCondition(channel=Channel.CAR, source=RecordingSource.LOCAL)
FAST = ProfileBuildConfig(compute_timbre=False)
C, U = ReviewDecision.CERTAIN, ReviewDecision.UNCERTAIN


@pytest.fixture()
def setup(tmp_path):
    case = Case.create(tmp_path / "case", "case-1")
    ds = SubjectDataset.load_or_create(case, "subj1")
    for k in range(4):
        wav = write_wav(tmp_path / f"car{k}.wav", speaker_voice(SPEAKER_A, seconds=3.0, seed=k))
        js = tmp_path / f"car{k}.json"
        js.write_text(json.dumps({"segments": [
            {"start": float(t), "end": float(t + 1), "text": f"frase {k} {t}"} for t in range(3)]}))
        ds.import_recording(case, wav, CAR, transcript_path=js)
    ds.split_by_recording(holdout_fraction=0.25)
    train = [it.item_id for it in sorted(ds.items, key=lambda it: it.item_id) if it.split == DatasetSplit.TRAIN]
    holdout = [it.item_id for it in sorted(ds.items, key=lambda it: it.item_id) if it.split == DatasetSplit.HOLDOUT]
    return case, ds, train, holdout


def _review(entries, reviewer="trascrittore-1", subject="subj1"):
    return Review(subject_id=subject, reviewer=reviewer,
                  entries=[ItemReview(item_id=i, decision=d, corrected_text=t) for i, d, t in entries])


def test_unreviewed_items_are_never_eligible(setup):
    _, ds, _, _ = setup
    assert eligible_items(ds) == []


def test_only_certain_train_items_are_eligible(setup):
    _, ds, train, holdout = setup
    out = apply_review(ds, _review([(train[0], C, None), (train[1], U, None), (holdout[0], C, None)]))
    assert out.certified == [train[0], holdout[0]] and out.marked_uncertain == [train[1]]
    assert [it.item_id for it in eligible_items(ds)] == [train[0]]
    assert "autovalutazione" in out.certain_but_not_eligible[holdout[0]]


def test_revocation_and_text_correction(setup):
    _, ds, train, _ = setup
    apply_review(ds, _review([(train[0], C, None), (train[1], C, None)]))
    out = apply_review(ds, _review([(train[0], U, None), (train[1], C, "  frase   corretta ")]))
    assert out.revoked == [train[0]] and out.text_corrected == [train[1]]
    assert ds.item(train[1]).text == "frase corretta"
    assert [it.item_id for it in eligible_items(ds)] == [train[1]]


def test_certain_but_blocked_or_unknown(setup):
    _, ds, train, _ = setup
    out = apply_review(ds, _review([(train[0], C, "   ")]))  # empty text -> blocking error
    assert "qualità" in out.certain_but_not_eligible[train[0]]
    assert eligible_items(ds) == []
    before = ds.model_dump()
    with pytest.raises(DatasetError):
        apply_review(ds, _review([(train[1], C, None), ("missing-item", C, None)]))
    assert ds.model_dump() == before  # nothing applied
    with pytest.raises(DatasetError):
        apply_review(ds, _review([(train[1], C, None)], subject="someone-else"))


def test_update_loop_bumps_version_and_records_hashes(setup):
    case, ds, train, holdout = setup
    with pytest.raises(DatasetError, match="certi"):
        update_profile(case, ds, FakeVoiceprint(), config=FAST)

    r1 = update_profile(case, ds, FakeVoiceprint(), config=FAST, review=_review(
        [(train[0], C, None), (train[1], C, None), (train[2], U, None), (holdout[0], C, None)]))
    assert r1.profile.version == 1 and r1.record.from_version is None
    assert r1.record.items_added == [train[0], train[1]] and r1.record.items_removed == []
    assert r1.record.review is not None and r1.record.review.verify(case.root)
    assert r1.record.reviewer == "trascrittore-1"
    build = json.loads(r1.record.build_record.resolve(case.root).read_text())
    assert build["item_ids"] == [train[0], train[1]]  # uncertain and holdout items never
    adaptation = json.loads(r1.record.adaptation_manifest.resolve(case.root).read_text())
    assert [i["item_id"] for i in adaptation["items"]] == [train[0], train[1]]
    assert all(ds.item(i["item_id"]).verified for i in adaptation["items"])
    assert r1.profile.conditions[0].n_segments == 2
    assert SubjectDataset.load_or_create(case, "subj1").item(train[0]).verified  # dataset saved

    r2 = update_profile(case, ds, FakeVoiceprint(), config=FAST, review=_review(
        [(train[0], U, None), (train[3], C, "testo del trascrittore")]))
    assert r2.profile.version == 2 and r2.record.from_version == 1
    assert r2.record.items_added == [train[3]] and r2.record.items_removed == [train[0]]
    assert r2.record.previous_profile == r1.record.profile
    assert r2.record.previous_record_sha256 == read_log(case, "subj1")[0].digest()
    assert r2.record.adaptation_manifest.sha256 != r1.record.adaptation_manifest.sha256
    adaptation = json.loads(r2.record.adaptation_manifest.resolve(case.root).read_text())
    assert {i["item_id"]: i["text"] for i in adaptation["items"]}[train[3]] == "testo del trascrittore"
    build2 = json.loads(r2.record.build_record.resolve(case.root).read_text())
    assert build2["parent_profile"]["sha256"] == r1.record.profile.sha256
    assert load_profile(case, r1.record.profile).version == 1  # old version untouched
    assert [r.to_version for r in read_log(case, "subj1")] == [1, 2]
    assert verify_log(case, "subj1") == []

    r3 = update_profile(case, ds, FakeVoiceprint(), config=FAST)  # no review: still a new version
    assert r3.profile.version == 3 and r3.record.items_added == r3.record.items_removed == []


def test_verify_log_detects_tampering(setup):
    case, ds, train, _ = setup
    r1 = update_profile(case, ds, FakeVoiceprint(), config=FAST, review=_review([(train[0], C, None)]))
    update_profile(case, ds, FakeVoiceprint(), config=FAST, review=_review([(train[1], C, None)]))
    assert verify_log(case, "subj1") == []

    log_path = case.root / "profiles" / "subject-subj1" / "verification_log.json"
    log = json.loads(log_path.read_text())
    log[0]["reviewer"] = "qualcun-altro"
    log_path.write_text(json.dumps(log))
    assert any("catena" in p for p in verify_log(case, "subj1"))

    log[0]["reviewer"] = "trascrittore-1"
    log_path.write_text(json.dumps(log))
    adaptation = r1.record.adaptation_manifest.resolve(case.root)
    adaptation.write_text(adaptation.read_text().replace("frase", "FRASE"))
    assert any("dati di adattamento" in p for p in verify_log(case, "subj1"))


def test_end_to_end_from_a_folder(tmp_path):
    """M1 'done when': from a folder of subject recordings, a hashed SpeakerProfile,
    a recorded split and a quality report."""
    folder = tmp_path / "soggetto"
    folder.mkdir()
    index = []
    for k, channel in enumerate(["car", "car", "phone", "room", "car"]):
        write_wav(folder / f"r{k}.wav", speaker_voice(SPEAKER_A, seconds=3.0, seed=40 + k))
        (folder / f"r{k}.json").write_text(json.dumps({"segments": [
            {"start": 0.0, "end": 1.0, "text": f"uno {k}"},
            {"start": 1.0, "end": 2.0, "text": "[incomprensibile]" if k == 0 else f"due {k}"},
        ]}))
        index.append({"audio": f"r{k}.wav", "transcript": f"r{k}.json", "channel": channel, "source": "local"})
    (folder / "recordings.json").write_text(json.dumps({"recordings": index}))

    case = Case.create(tmp_path / "case", "case-e2e")
    ds = SubjectDataset.load_or_create(case, "subj1")
    ds.import_folder(case, folder)
    ds.split_by_recording()
    train = [it.item_id for it in ds.items if it.split == DatasetSplit.TRAIN]
    res = update_profile(case, ds, FakeVoiceprint(), config=FAST,
                         review=_review([(i, C, None) for i in train]))
    assert res.record.profile.verify(case.root)
    assert {cp.condition.channel for cp in res.profile.conditions} <= {Channel.CAR, Channel.PHONE, Channel.ROOM}
    manifest = json.loads(res.profile.dataset_manifest.resolve(case.root).read_text())
    assert manifest["split_history"][0]["holdout_fraction"] == 0.2
    assert manifest["summary"]["split"]["holdout"]["recordings"] == 1
    assert "unintelligible_marker" in manifest["summary"]["warnings_by_code"]
    assert sorted(res.profile.holdout_recordings) == ds.holdout_recordings()
    assert verify_log(case, "subj1") == [] and case.verify() == []
