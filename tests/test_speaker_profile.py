import json

import librosa
import numpy as np
import pytest
import scipy.signal as ss

from forensicaudio.case import Case
from forensicaudio.config import Channel, RecordingSource, StageKind
from forensicaudio.models import DatasetSplit, RecordingCondition, SegmentFlag, SegmentHypotheses, SpeakerProfile
from forensicaudio.speaker.dataset import DatasetError, SubjectDataset
from forensicaudio.speaker.profile import (
    ProfileBuildConfig,
    ProfileScorer,
    ScoreCalibration,
    annotate_segment,
    build_speaker_profile,
    calibrate_scores,
    calibration_scores,
    equal_error_rate,
    load_calibration,
    load_profile,
    save_calibration,
)
from tests.fakes import FakeVoiceprint
from tests.synth_voice import SPEAKER_A, SPEAKER_B, SR, bandlimit, speaker_voice, write_wav

CAR = RecordingCondition(channel=Channel.CAR, source=RecordingSource.LOCAL)
PHONE = RecordingCondition(channel=Channel.PHONE, source=RecordingSource.OTHER, codec="amr-nb")
FAST = ProfileBuildConfig(compute_timbre=False)


def _car_noise(n, seed):
    rng = np.random.default_rng(seed)
    sos = ss.butter(2, 200, btype="low", fs=SR, output="sos")
    return (ss.sosfilt(sos, rng.normal(0, 0.05, n))).astype(np.float32)


def _car_recording(path, seed, seconds=4.0):
    y = speaker_voice(SPEAKER_A, seconds=seconds, seed=seed) + _car_noise(int(SR * seconds), seed)
    return write_wav(path, y)


def _phone_recording(path, seed):
    """2 s subject, 2 s other speaker, 2 s subject; band-limited, stored at 8 kHz."""
    a1 = speaker_voice(SPEAKER_A, seconds=2.0, seed=seed)
    b = speaker_voice(SPEAKER_B, seconds=2.0, seed=seed + 1)
    a2 = speaker_voice(SPEAKER_A, seconds=2.0, seed=seed + 2)
    y = bandlimit(np.concatenate([a1, b, a2]), 300, 3400)
    y8 = librosa.resample(y.astype(np.float64), orig_sr=SR, target_sr=8000, res_type="soxr_hq")
    return write_wav(path, y8.astype(np.float32), sr=8000)


def _segments(seconds):
    return [{"start": float(t), "end": float(t + 1), "text": f"frase {t}"} for t in range(int(seconds))]


def _phone_segments():
    return [
        {"start": 0.0, "end": 1.0, "text": "pronto", "speaker": "A"},
        {"start": 1.0, "end": 2.0, "text": "sono io", "speaker": "A"},
        {"start": 2.0, "end": 3.0, "text": "ciao", "speaker": "B"},
        {"start": 3.0, "end": 4.0, "text": "dimmi", "speaker": "B"},
        {"start": 4.0, "end": 5.0, "text": "va bene", "speaker": "A"},
        {"start": 5.0, "end": 6.0, "text": "a dopo", "speaker": "A"},
    ]


@pytest.fixture()
def material(tmp_path):
    case = Case.create(tmp_path / "case", "case-1")
    ds = SubjectDataset.load_or_create(case, "subj1")
    src = tmp_path / "src"
    src.mkdir()
    for k in range(3):
        wav = _car_recording(src / f"car{k}.wav", seed=10 + k)
        js = src / f"car{k}.json"
        js.write_text(json.dumps({"segments": _segments(4)}))
        ds.import_recording(case, wav, CAR, transcript_path=js)
    for k in range(2):
        wav = _phone_recording(src / f"tel{k}.wav", seed=20 + 3 * k)
        js = src / f"tel{k}.json"
        js.write_text(json.dumps({"segments": _phone_segments()}))
        ds.import_recording(case, wav, PHONE, transcript_path=js, subject_speaker="A")
    ds.split_by_recording(holdout_fraction=0.4, seed=0)  # one car + one phone recording in holdout
    for it in ds.items:
        it.verified = True
    return case, ds


def test_build_profile_one_condition_per_key(material):
    case, ds = material
    res = build_speaker_profile(case, ds, FakeVoiceprint(), config=ProfileBuildConfig())
    prof = res.profile
    assert prof.version == 1 and prof.subject_id == "subj1"
    keys = [cp.condition.key() for cp in prof.conditions]
    assert keys == sorted({CAR.key(), PHONE.key()})
    assert sorted(prof.holdout_recordings) == ds.holdout_recordings() and len(prof.holdout_recordings) == 2
    holdout_items = {it.item_id for it in ds.items if it.split == DatasetSplit.HOLDOUT}
    for cp in prof.conditions:
        assert cp.voiceprint is not None and cp.voiceprint.verify(case.root)
        data = json.loads(cp.voiceprint.resolve(case.root).read_text())
        assert data["backend"] == "fake-voiceprint" and not set(data["item_ids"]) & holdout_items
        assert cp.n_segments == data["n_embeddings"] and cp.seconds == pytest.approx(cp.n_segments * 1.0)
        assert cp.timbre["method"] == "forensicaudio.speaker.timbre/v1"
    car = next(cp for cp in prof.conditions if cp.condition.channel == Channel.CAR)
    assert car.timbre["f0"]["hz"]["median"] == pytest.approx(SPEAKER_A["f0"], rel=0.05)
    phone = next(cp for cp in prof.conditions if cp.condition.channel == Channel.PHONE)
    assert phone.timbre["ltas"]["nyquist_limit_hz"] == 4000  # 8 kHz phone material: real bandwidth
    assert phone.timbre["sources"][0]["input_sr"] == 8000
    assert prof.dataset_manifest is not None and prof.dataset_manifest.verify(case.root)
    assert res.profile_ref.verify(case.root) and res.build_record_ref.verify(case.root)
    assert load_profile(case, res.profile_ref) == prof
    record = json.loads(res.build_record_ref.resolve(case.root).read_text())
    assert record["backend"] == "fake-voiceprint" and record["version"] == 1
    assert not set(record["item_ids"]) & holdout_items


def test_profile_versions_are_immutable(material):
    case, ds = material
    build_speaker_profile(case, ds, FakeVoiceprint(), config=FAST)
    with pytest.raises(DatasetError):
        build_speaker_profile(case, ds, FakeVoiceprint(), config=FAST)  # same version, new content
    assert build_speaker_profile(case, ds, FakeVoiceprint(), config=FAST, version=2).profile.version == 2


def test_profile_requires_split_and_verified_items(material, tmp_path):
    case, ds = material
    for it in ds.items:
        it.verified = False
    with pytest.raises(DatasetError, match="verificati"):
        build_speaker_profile(case, ds, FakeVoiceprint(), config=FAST)
    res = build_speaker_profile(case, ds, FakeVoiceprint(), config=ProfileBuildConfig(compute_timbre=False,
                                                                                     verified_only=False))
    assert res.profile.conditions
    fresh = SubjectDataset(subject_id="other", recordings=ds.recordings, items=ds.items)
    for r in fresh.recordings:
        r.split = None
    with pytest.raises(DatasetError, match="Dividere"):
        build_speaker_profile(case, fresh, FakeVoiceprint(), config=FAST)


def test_scoring_reports_closest_condition(material):
    case, ds = material
    res = build_speaker_profile(case, ds, FakeVoiceprint(), config=FAST)
    scorer = ProfileScorer.load(case, res.profile, FakeVoiceprint())
    car_a = speaker_voice(SPEAKER_A, seconds=1.0, seed=99) + _car_noise(SR, 99)
    s = scorer.score(car_a, SR)
    assert s.condition_key == CAR.key() and s.score > 0.9
    assert set(s.scores_by_condition) == {CAR.key(), PHONE.key()}
    assert s.possible_other_speaker is None and "non calibrato" in s.message
    phone_a = bandlimit(speaker_voice(SPEAKER_A, seconds=1.0, seed=98), 300, 3400)
    assert scorer.score(phone_a, SR).condition_key == PHONE.key()
    forced = scorer.score(phone_a, SR, condition_key=CAR.key())
    assert forced.condition_key == CAR.key() and forced.score < scorer.score(phone_a, SR).score
    phone_b = bandlimit(speaker_voice(SPEAKER_B, seconds=1.0, seed=97), 300, 3400)
    assert scorer.score(phone_b, SR).score < scorer.score(phone_a, SR).score
    with pytest.raises(DatasetError):
        scorer.score(phone_a, SR, condition_key="room/other/unknown-codec/unknown")


def test_calibration_on_case_material_flags_other_speaker(material):
    case, ds = material
    res = build_speaker_profile(case, ds, FakeVoiceprint(), config=FAST)
    scorer = ProfileScorer.load(case, res.profile, FakeVoiceprint())
    extra = [(bandlimit(speaker_voice(SPEAKER_B, seconds=1.0, seed=200 + k), 300, 3400), SR) for k in range(4)]
    same, other, material_desc = calibration_scores(case, ds, scorer, extra_other=extra)
    assert len(same) == 4 + 4  # holdout: one car (4 items) + one phone (4 subject items)
    assert len(other) == 2 * 2 + 4  # labelled spans of B in both phone recordings + extra clips
    cal = calibrate_scores(same, other, backend=scorer.backend.name, profile=res.profile, material=material_desc)
    assert cal.reliable and cal.coef > 0
    assert min(other) < cal.threshold_score < max(same)
    assert cal.probability(max(same)) > 0.5 > cal.probability(min(other))
    assert cal.rates_at_threshold["subject_flagged"] <= 0.25
    assert any("Pochi punteggi" in w for w in cal.warnings)  # fewer than 10 per class

    ref = save_calibration(case, cal)
    assert ref.verify(case.root) and load_calibration(case, ref) == cal
    calibrated = scorer.with_calibration(cal)
    phone_b = bandlimit(speaker_voice(SPEAKER_B, seconds=1.0, seed=300), 300, 3400)
    sb = calibrated.score(phone_b, SR)
    assert sb.possible_other_speaker is True and "Possibile altro parlante" in sb.message
    car_a = speaker_voice(SPEAKER_A, seconds=1.0, seed=301) + _car_noise(SR, 301)
    sa = calibrated.score(car_a, SR)
    assert sa.possible_other_speaker is False and sa.calibrated_probability > 0.5

    seg = SegmentHypotheses(start=0.0, end=1.0, audio_provenance=StageKind.NEURAL)
    flagged = annotate_segment(seg, sb)
    assert flagged.speaker_condition == sb.condition_key and flagged.speaker_score == sb.score
    assert flagged.flags == [SegmentFlag.POSSIBLE_OTHER_SPEAKER]
    assert annotate_segment(seg, sa).flags == []


def test_calibration_guards():
    prof = SpeakerProfile(subject_id="s")
    with pytest.raises(DatasetError):
        calibrate_scores([0.9], [0.1, 0.2], backend="b", profile=prof)
    bad = calibrate_scores([0.1, 0.2, 0.15], [0.8, 0.9, 0.85], backend="b", profile=prof)
    assert not bad.reliable and bad.threshold_score is None and bad.is_possible_other_speaker(0.0) is None
    eer, t = equal_error_rate([0.8, 0.9, 0.95], [0.1, 0.2, 0.3])
    assert eer == 0.0 and 0.3 < t <= 0.8


def test_scorer_refuses_other_backend_and_tampered_artifacts(material):
    case, ds = material
    res = build_speaker_profile(case, ds, FakeVoiceprint(), config=FAST)

    class OtherBackend(FakeVoiceprint):
        name = "other-model"

    with pytest.raises(DatasetError, match="non confrontabili"):
        ProfileScorer.load(case, res.profile, OtherBackend())
    cal = calibrate_scores([0.9, 0.95], [0.1, 0.2], backend="other-model", profile=res.profile)
    scorer = ProfileScorer.load(case, res.profile, FakeVoiceprint())
    with pytest.raises(DatasetError):
        scorer.with_calibration(cal)
    path = res.profile.conditions[0].voiceprint.resolve(case.root)
    path.write_text(path.read_text().replace('"n_embeddings"', '"n_embeddings_x"'))
    with pytest.raises(ValueError, match="SHA-256"):
        ProfileScorer.load(case, res.profile, FakeVoiceprint())


def test_outputs_never_speak_of_identification(material):
    case, ds = material
    res = build_speaker_profile(case, ds, FakeVoiceprint(), config=FAST)
    scorer = ProfileScorer.load(case, res.profile, FakeVoiceprint())
    same, other, desc = calibration_scores(case, ds, scorer)
    cal = calibrate_scores(same, other, backend=scorer.backend.name, profile=res.profile, material=desc)
    texts = [
        res.profile_ref.resolve(case.root).read_text(),
        res.build_record_ref.resolve(case.root).read_text(),
        save_calibration(case, cal).resolve(case.root).read_text(),
        scorer.with_calibration(cal).score(speaker_voice(SPEAKER_B, seconds=1.0), SR).model_dump_json(),
        scorer.score(speaker_voice(SPEAKER_A, seconds=1.0), SR).model_dump_json(),
    ]
    for t in texts:
        assert "identif" not in t.lower()
    assert isinstance(cal, ScoreCalibration)
