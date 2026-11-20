"""Fast training tests with a tiny randomly initialised Whisper (no downloads)."""
import json
import logging

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from forensicaudio.adapt.artifact import (  # noqa: E402
    ADAPTER_JSON,
    CHECKSUMS,
    AdapterIntegrityError,
    hash_base_weights,
    load_adapter,
    load_adapter_artifact,
    load_adapter_model,
    verify_adapter,
)
from forensicaudio.adapt.dataset import TrainingSetConfig, build_training_set  # noqa: E402
from forensicaudio.adapt.train import TrainConfig, TrainError, train_adapter  # noqa: E402
from forensicaudio.config import Channel, Tier  # noqa: E402
from forensicaudio.models import (  # noqa: E402
    AdapterTechnique,
    ArtifactRef,
    DatasetItem,
    DatasetSplit,
    RecordingCondition,
)
from tests.tiny_whisper import save_base  # noqa: E402

TEXTS = ["ciao mondo", "vado a casa", "sono qui", "ciao Rossi"]


def _items():
    ref = ArtifactRef(path="/nonexistent.wav", sha256="0" * 64, kind="evidence")
    conds = [RecordingCondition(channel=Channel.PHONE), RecordingCondition(channel=Channel.CAR)]
    return [
        DatasetItem(item_id=f"i{k}", recording_id="r", audio=ref, start=0, end=1, text=t,
                    condition=conds[k % 2], verified=True, split=DatasetSplit.TRAIN)
        for k, t in enumerate(TEXTS)
    ]


def _loader(item):
    k = int(item.item_id[1:])
    t = np.arange(16000) / 16000
    return (0.3 * np.sin(2 * np.pi * (200 + 100 * k) * t)).astype(np.float32), 16000


def _augment(audio, sr, rng):
    gain = float(rng.uniform(0.5, 1.0))
    return audio * gain, {"gain": gain}


@pytest.fixture(scope="module")
def training_set():
    return build_training_set(_items(), config=TrainingSetConfig(seed=1, clean_fraction=0.5),
                              augment=_augment, loader=_loader)


@pytest.fixture(scope="module")
def base_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny-base")
    save_base(d)
    return d


def _base(base_dir):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    return (WhisperForConditionalGeneration.from_pretrained(str(base_dir)),
            WhisperProcessor.from_pretrained(str(base_dir)))


def _logits(model, processor):
    audio = _loader(_items()[1])[0]
    feats = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
    dec = torch.tensor([processor.tokenizer.prefix_tokens])
    with torch.no_grad():
        return model(input_features=feats, decoder_input_ids=dec).logits


def _config(base_dir, **kw):
    defaults = dict(base_model=str(base_dir), max_steps=4, batch_size=2, learning_rate=1e-2,
                    device="cpu", num_threads=2, seed=11)
    defaults.update(kw)
    return TrainConfig(**defaults)


def test_lora_training_produces_verified_artifact(training_set, base_dir, tmp_path):
    model, processor = _base(base_dir)
    before = _logits(model, processor).clone()
    out = tmp_path / "adapter"
    res = train_adapter(training_set, out, _config(base_dir), model=model, processor=processor,
                        tier=Tier.EDGE)
    art = res.artifact

    assert art.technique == AdapterTechnique.LORA
    assert art.base_model == str(base_dir)
    assert art.base_model_sha256 == hash_base_weights(str(base_dir)) is not None
    assert art.seed == 11
    assert art.train_manifest_sha256 == training_set.sha256()
    assert art.conditions == training_set.manifest.condition_keys
    hp = art.hyperparams
    assert hp["lora_r"] == 32 and hp["lora_alpha"] == 64
    assert set(hp["lora_target_modules"]) == {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"}
    assert hp["total_steps"] == 4 and hp["language"] == "italian"

    weights = verify_adapter(art)
    assert (weights / "adapter_config.json").is_file() and (weights / "adapter_model.safetensors").is_file()
    assert art.weights.path.endswith(CHECKSUMS)
    assert load_adapter_artifact(out / ADAPTER_JSON) == art
    assert (out / "train_manifest.json").is_file()

    rec = res.record
    assert rec.steps == 4 and len(rec.losses) == 4
    assert rec.wall_time_s > 0 and rec.peak_rss_mb >= rec.rss_before_mb > 0
    assert rec.device == "cpu" and rec.tier == Tier.EDGE and rec.cuda_peak_mb is None
    assert 0 < rec.trainable_params < rec.total_params
    assert json.loads((out / "training_record.json").read_text())["steps"] == 4

    # the injected model now carries the trained LoRA layers: reloading from disk
    # (base from its directory + PEFT adapter) must give the same output
    trained = _logits(model, processor)
    assert not torch.allclose(trained, before)
    loaded = load_adapter(out, device="cpu")
    assert loaded.device == "cpu" and loaded.base_model_sha256 == art.base_model_sha256
    assert torch.allclose(_logits(loaded.model, loaded.processor), trained, atol=1e-5)
    merged = load_adapter(out, device="cpu", merge=True)
    assert torch.allclose(_logits(merged.model, merged.processor), trained, atol=1e-4)


def test_training_is_reproducible_from_seed(training_set, base_dir, tmp_path):
    from safetensors.torch import load_file

    tensors = []
    for run in ("a", "b"):
        model, processor = _base(base_dir)
        res = train_adapter(training_set, tmp_path / run, _config(base_dir, max_steps=2),
                            model=model, processor=processor)
        tensors.append(load_file(str(tmp_path / run / "weights" / "adapter_model.safetensors")))
        assert res.record.losses
    assert tensors[0].keys() == tensors[1].keys()
    assert all(torch.equal(tensors[0][k], tensors[1][k]) for k in tensors[0])


def test_full_finetuning_option(training_set, base_dir, tmp_path):
    model, processor = _base(base_dir)
    out = tmp_path / "full"
    res = train_adapter(training_set, out, _config(base_dir, technique=AdapterTechnique.FULL, max_steps=2,
                                                   freeze_encoder=True, learning_rate=1e-3),
                        model=model, processor=processor)
    art = res.artifact
    assert art.technique == AdapterTechnique.FULL and art.hyperparams["freeze_encoder"] is True
    assert "lora_r" not in art.hyperparams
    weights = verify_adapter(art)
    assert (weights / "model.safetensors").is_file() and (weights / "tokenizer.json").is_file()
    loaded = load_adapter(out, device="cpu")
    assert torch.allclose(_logits(loaded.model, loaded.processor), _logits(model, processor), atol=1e-5)


def test_low_rank_warning_and_no_overwrite(training_set, base_dir, tmp_path, caplog):
    model, processor = _base(base_dir)
    out = tmp_path / "adapter"
    with caplog.at_level(logging.WARNING):
        train_adapter(training_set, out, _config(base_dir, lora_r=8, max_steps=1),
                      model=model, processor=processor)
    assert any("Rango LoRA 8 basso" in r.getMessage() for r in caplog.records)
    model, processor = _base(base_dir)
    with pytest.raises(TrainError, match="contiene già un adattatore"):
        train_adapter(training_set, out, _config(base_dir, max_steps=1), model=model, processor=processor)


def test_load_refuses_tampered_weights_and_wrong_base(training_set, base_dir, tmp_path):
    model, processor = _base(base_dir)
    out = tmp_path / "adapter"
    res = train_adapter(training_set, out, _config(base_dir, max_steps=1), model=model, processor=processor)

    wrong_base = res.artifact.model_copy(update={"base_model_sha256": "f" * 64})
    with pytest.raises(AdapterIntegrityError, match="modello di base .* non corrisponde"):
        load_adapter_model(wrong_base, device="cpu")

    weights = out / "weights" / "adapter_model.safetensors"
    data = bytearray(weights.read_bytes())
    data[-1] ^= 0xFF
    weights.write_bytes(bytes(data))
    with pytest.raises(AdapterIntegrityError, match="caricamento rifiutato"):
        load_adapter(out, device="cpu")


def test_examples_longer_than_window_are_skipped(base_dir, tmp_path):
    def longer(audio, sr, rng):
        return np.concatenate([audio, np.zeros(31 * sr, dtype=np.float32)]), {"pad_s": 31}

    ts = build_training_set(_items(), config=TrainingSetConfig(seed=2, clean_fraction=0.5,
                                                               balance_conditions=False),
                            augment=longer, loader=_loader)
    model, processor = _base(base_dir)
    res = train_adapter(ts, tmp_path / "a", _config(base_dir, max_steps=1), model=model, processor=processor)
    n_aug = sum(not e.clean for e in ts.manifest.examples)
    assert len(res.record.skipped) == n_aug > 0
    assert {s.reason for s in res.record.skipped} == {"audio_longer_than_window"}
    assert res.record.n_examples == len(ts) - n_aug
