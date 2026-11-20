import numpy as np
import pytest
import soundfile as sf

from forensicaudio.adapt.dataset import (
    DatasetBuildError,
    ExclusionReason,
    TrainingSet,
    TrainingSetConfig,
    build_training_set,
    plan_examples,
    samples_sha256,
    select_items,
)
from forensicaudio.config import Channel, RecordingSource
from forensicaudio.models import ArtifactRef, DatasetItem, DatasetSplit, RecordingCondition
from forensicaudio.utils import sha256_file

PHONE = RecordingCondition(channel=Channel.PHONE, source=RecordingSource.OTHER, codec="amr-nb")
ROOM = RecordingCondition(channel=Channel.ROOM, source=RecordingSource.OTHER)
CAR = RecordingCondition(channel=Channel.CAR, source=RecordingSource.LOCAL)


def _write(root, name, seconds=3.0, sr=8000, seed=0):
    rng = np.random.default_rng(seed)
    audio = (0.1 * rng.standard_normal(int(seconds * sr))).astype(np.float32)
    path = root / "evidence" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="FLOAT")
    return ArtifactRef.from_file(path, kind="evidence", root=root)


@pytest.fixture
def case(tmp_path):
    """6 phone items, 2 room, 1 car, plus excluded ones."""
    phone = _write(tmp_path, "phone.wav", seconds=7.0, sr=8000, seed=1)
    room = _write(tmp_path, "room.wav", seconds=3.0, sr=16000, seed=2)
    car = _write(tmp_path, "car.wav", seconds=2.0, sr=48000, seed=3)
    items = [
        DatasetItem(item_id=f"p{i}", recording_id="rec-phone", audio=phone, start=i, end=i + 1.0,
                    text=f"telefonata numero {i}", condition=PHONE, verified=True, split=DatasetSplit.TRAIN)
        for i in range(6)
    ]
    items += [
        DatasetItem(item_id=f"r{i}", recording_id="rec-room", audio=room, start=i, end=i + 1.0,
                    text=f"stanza {i}", condition=ROOM, verified=True, split=DatasetSplit.TRAIN)
        for i in range(2)
    ]
    items.append(DatasetItem(item_id="c0", recording_id="rec-car", audio=car, start=0.5, end=1.5,
                             text="vado a casa", condition=CAR, verified=True, split=DatasetSplit.TRAIN))
    # never used for training
    items.append(DatasetItem(item_id="u0", recording_id="rec-car", audio=car, start=0, end=1,
                             text="forse diceva", condition=CAR, verified=False, split=DatasetSplit.TRAIN))
    items.append(DatasetItem(item_id="h0", recording_id="rec-room", audio=room, start=2, end=3,
                             text="tenuto da parte", condition=ROOM, verified=True, split=DatasetSplit.HOLDOUT))
    items.append(DatasetItem(item_id="n0", recording_id="rec-room", audio=room, start=2, end=3,
                             text="senza divisione", condition=ROOM, verified=True, split=None))
    return tmp_path, items


def fake_augment(audio, sr, rng):
    gain = float(rng.uniform(0.2, 0.9))
    snr = float(rng.choice([0.0, 5.0, 10.0]))
    noise = rng.standard_normal(audio.size).astype(np.float32) * 0.01
    return audio * gain + noise, {"gain": np.float32(gain), "snr_db": snr, "ir": None}


fake_augment.version = "test-1"


def test_select_only_verified_training_items(case):
    _, items = case
    kept, excluded = select_items(items, TrainingSetConfig())
    assert {i.item_id for i in kept} == {"p0", "p1", "p2", "p3", "p4", "p5", "r0", "r1", "c0"}
    reasons = {e.item_id: e.reason for e in excluded}
    assert reasons == {"u0": ExclusionReason.UNVERIFIED, "h0": ExclusionReason.HOLDOUT,
                       "n0": ExclusionReason.UNSPLIT}
    kept, _ = select_items(items, TrainingSetConfig(allow_unsplit=True))
    assert "n0" in {i.item_id for i in kept}
    assert "h0" not in {i.item_id for i in kept} and "u0" not in {i.item_id for i in kept}


def test_balancing_and_clean_fraction(case):
    _, items = case
    kept, _ = select_items(items, TrainingSetConfig())
    cfg = TrainingSetConfig(clean_fraction=0.25, max_oversample=4.0)
    plan = plan_examples(kept, cfg, has_augment=True)
    by_cond = {}
    for p in plan:
        by_cond.setdefault(p.condition_key, []).append(p)
    assert len(by_cond[PHONE.key()]) == 6
    assert len(by_cond[ROOM.key()]) == 6  # 2 items x 3
    assert len(by_cond[CAR.key()]) == 4  # capped at 1 x max_oversample
    for key, exs in by_cond.items():
        assert sum(p.clean for p in exs) == round(0.25 * len(exs))
        # clean copies go to distinct items first
        clean_items = [p.item_id for p in exs if p.clean]
        assert len(clean_items) == len(set(clean_items))
    no_balance = plan_examples(kept, TrainingSetConfig(balance_conditions=False), has_augment=True)
    assert len(no_balance) == len(kept)
    no_augment = plan_examples(kept, cfg, has_augment=False)
    assert all(p.clean for p in no_augment)


def test_build_records_everything_and_is_reproducible(case):
    root, items = case
    cfg = TrainingSetConfig(seed=7, clean_fraction=0.3)
    ts1 = build_training_set(items, config=cfg, augment=fake_augment, root=root)
    ts2 = build_training_set(list(reversed(items)), config=cfg, augment=fake_augment, root=root)
    assert ts1.sha256() == ts2.sha256()
    m = ts1.manifest
    assert set(m.conditions) == {PHONE.key(), ROOM.key(), CAR.key()}
    assert m.condition_keys == sorted(m.conditions)
    assert m.augment_version == "test-1" and "fake_augment" in m.augment_name
    assert {s.item_id for s in m.sources} == {"p0", "p1", "p2", "p3", "p4", "p5", "r0", "r1", "c0"}
    assert {e.item_id for e in m.excluded} == {"u0", "h0", "n0"}
    assert all(s.audio_sha256 for s in m.sources)
    for i, rec in enumerate(m.examples):
        audio, text = ts1.example(i)
        assert audio.dtype == np.float32 and rec.sample_rate == 16000
        assert samples_sha256(audio) == rec.audio_sha256
        assert rec.n_samples == audio.size == 16000  # 1 s segments resampled to 16 kHz
        if rec.clean:
            assert rec.augmentation is None
        else:
            assert set(rec.augmentation) == {"gain", "snr_db", "ir"}
            assert isinstance(rec.augmentation["gain"], float)
    # repeats of one item get different augmentation draws
    car = [e for e in m.examples if e.item_id == "c0"]
    assert len({e.audio_sha256 for e in car}) == len(car)

    ts3 = build_training_set(items, config=TrainingSetConfig(seed=8, clean_fraction=0.3),
                             augment=fake_augment, root=root)
    assert ts3.sha256() != ts1.sha256()


def test_adding_an_item_does_not_change_other_examples(case):
    root, items = case
    cfg = TrainingSetConfig(balance_conditions=False, clean_fraction=0.0)
    base = build_training_set(items, config=cfg, augment=fake_augment, root=root)
    extra = items + [items[0].model_copy(update={"item_id": "p9", "text": "altro"})]
    more = build_training_set(extra, config=cfg, augment=fake_augment, root=root)
    before = {e.example_id: e.audio_sha256 for e in base.manifest.examples}
    after = {e.example_id: e.audio_sha256 for e in more.manifest.examples}
    assert all(after[k] == v for k, v in before.items())


def test_materialized_set_and_manifest_hash(case, tmp_path_factory):
    root, items = case
    out = tmp_path_factory.mktemp("trainset")
    ts = build_training_set(items, config=TrainingSetConfig(seed=3), augment=fake_augment, root=root, out_dir=out)
    manifest_path = out / "train_manifest.json"
    # the file hash is the manifest hash (canonical JSON, no timestamps)
    assert sha256_file(manifest_path) == ts.sha256()
    reopened = TrainingSet.from_manifest(manifest_path, expected_sha256=ts.sha256())
    a, t = reopened.example(0)
    b, _ = ts.example(0)
    assert np.array_equal(a, b) and t == ts.manifest.examples[0].text
    # a modified example file is refused
    target = out / ts.manifest.examples[0].path
    sf.write(str(target), np.zeros(10, dtype=np.float32), 16000, subtype="FLOAT")
    with pytest.raises(DatasetBuildError, match="non corrisponde al manifesto"):
        reopened.example(0)
    with pytest.raises(DatasetBuildError, match="impronta SHA-256 attesa"):
        TrainingSet.from_manifest(manifest_path, expected_sha256="0" * 64)


def test_source_file_tampering_is_refused(case):
    root, items = case
    path = root / items[0].audio.path
    path.chmod(0o644)
    sf.write(str(path), np.zeros(8000 * 7, dtype=np.float32), 8000, subtype="FLOAT")
    with pytest.raises(DatasetBuildError, match="SHA-256"):
        build_training_set(items, root=root)


def test_no_usable_items_is_an_error(case):
    root, items = case
    unverified = [i.model_copy(update={"verified": False}) for i in items]
    with pytest.raises(DatasetBuildError, match="Nessun segmento verificato"):
        build_training_set(unverified, root=root)


def test_duplicate_ids_and_bad_augment(case):
    root, items = case
    with pytest.raises(DatasetBuildError, match="duplicato"):
        build_training_set(items + [items[0]], root=root)

    def stereo(audio, sr, rng):
        return np.stack([audio, audio]), {}

    with pytest.raises(DatasetBuildError, match="mono"):
        build_training_set(items, config=TrainingSetConfig(clean_fraction=0.0), augment=stereo, root=root)


def test_custom_loader_and_clipping_recorded(case):
    _, items = case

    def loader(item):
        return np.full(8000, 0.5, dtype=np.float32), 8000

    def loud(audio, sr, rng):
        return audio * 4.0, {"gain": 4.0}

    ts = build_training_set(items, config=TrainingSetConfig(clean_fraction=0.0, balance_conditions=False),
                            augment=loud, loader=loader)
    # resampling ramps the edges; the rest of the constant signal clips
    assert all(e.clipped_samples > 15000 for e in ts.manifest.examples)
    audio, _ = ts.example(0)
    assert float(np.abs(audio).max()) <= 1.0
