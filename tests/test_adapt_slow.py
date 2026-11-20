"""CPU smoke tests with the real openai/whisper-tiny (download ~150 MB).

Run with: .venv/bin/python -m pytest -m slow tests/test_adapt_slow.py -s
The audio is synthetic (harmonic tones, no real voice): these tests check the
plumbing on a real checkpoint, not recognition quality.
"""
import re

import numpy as np
import pytest

pytestmark = pytest.mark.slow

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from forensicaudio.adapt.artifact import load_adapter, verify_adapter  # noqa: E402
from forensicaudio.adapt.dataset import TrainingSetConfig, build_training_set  # noqa: E402
from forensicaudio.adapt.lexicon import (  # noqa: E402
    BiasConfig,
    CaseLexicon,
    WhisperLexiconDecoder,
    measure_insertion_rate,
)
from forensicaudio.adapt.train import TrainConfig, train_adapter  # noqa: E402
from forensicaudio.config import Channel, Tier  # noqa: E402
from forensicaudio.models import ArtifactRef, DatasetItem, DatasetSplit, RecordingCondition  # noqa: E402

BASE = "openai/whisper-tiny"
SR = 16000
TEXTS = [
    "Vado a casa adesso.",
    "Domani chiamo il meccanico.",
    "Sono in macchina, ti richiamo.",
    "Ci vediamo alle otto davanti al bar.",
    "Porta tutto in garage stasera.",
    "Ho parlato con lui ieri mattina.",
]
CONDITIONS = [RecordingCondition(channel=c) for c in (Channel.PHONE, Channel.ROOM, Channel.CAR)]


def _voice_like(k: int, seconds: float = 2.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    f0 = 110 + 15 * k
    sig = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 12))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * (3 + k * 0.3) * t)
    return (0.2 * sig * envelope / np.abs(sig).max()).astype(np.float32)


def _items():
    ref = ArtifactRef(path="/synthetic.wav", sha256="0" * 64, kind="evidence")
    return [
        DatasetItem(item_id=f"s{k}", recording_id=f"rec{k % 3}", audio=ref, start=0, end=2, text=text,
                    condition=CONDITIONS[k % 3], verified=True, split=DatasetSplit.TRAIN)
        for k, text in enumerate(TEXTS)
    ]


def _loader(item):
    return _voice_like(int(item.item_id[1:])), SR


def _augment(audio, sr, rng):
    snr_db = float(rng.choice([5.0, 10.0, 20.0]))
    noise = rng.standard_normal(audio.size).astype(np.float32)
    noise *= np.sqrt(np.mean(audio**2) / (10 ** (snr_db / 10))) / (np.std(noise) + 1e-9)
    return audio + noise, {"kind": "white-noise", "snr_db": snr_db}


@pytest.fixture(scope="module")
def base():
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    return WhisperForConditionalGeneration.from_pretrained(BASE), WhisperProcessor.from_pretrained(BASE)


def test_whisper_tiny_lora_cpu_smoke(tmp_path):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    ts = build_training_set(_items(), config=TrainingSetConfig(seed=5, clean_fraction=0.5),
                            augment=_augment, loader=_loader)
    model = WhisperForConditionalGeneration.from_pretrained(BASE)
    processor = WhisperProcessor.from_pretrained(BASE)
    cfg = TrainConfig(base_model=BASE, max_steps=15, batch_size=2, learning_rate=1e-3,
                      device="cpu", num_threads=4, seed=5)
    res = train_adapter(ts, tmp_path / "adapter", cfg, model=model, processor=processor, tier=Tier.EDGE)
    rec, art = res.record, res.artifact
    print(f"\n[whisper-tiny LoRA r={cfg.lora_r}] steps={rec.steps} examples={rec.n_examples} "
          f"wall={rec.wall_time_s:.1f}s peak_rss={rec.peak_rss_mb:.0f}MB (before {rec.rss_before_mb:.0f}MB) "
          f"trainable={rec.trainable_params} losses={rec.losses[0]:.3f}->{rec.losses[-1]:.3f}")

    assert rec.steps == 15
    assert np.mean(rec.losses[-3:]) < rec.losses[0]
    assert re.fullmatch(r"[0-9a-f]{40}", art.hyperparams["base_model_revision"] or "")
    assert art.base_model_sha256 and len(art.base_model_sha256) == 64
    assert art.conditions == sorted(c.key() for c in CONDITIONS)
    verify_adapter(art)

    loaded = load_adapter(tmp_path / "adapter", device="cpu")
    feats = processor.feature_extractor(_voice_like(0), sampling_rate=SR, return_tensors="pt").input_features
    dec = torch.tensor([processor.tokenizer.prefix_tokens])
    with torch.no_grad():
        ref = model(input_features=feats, decoder_input_ids=dec).logits
        got = loaded.model(input_features=feats, decoder_input_ids=dec).logits
        text_ids = loaded.model.generate(input_features=feats, language="it", task="transcribe",
                                         max_new_tokens=20)
    assert torch.allclose(ref, got, atol=1e-4)
    assert isinstance(processor.batch_decode(text_ids, skip_special_tokens=True)[0], str)


def test_whisper_tiny_lexicon_insertion_test(base):
    model, processor = base
    model.eval()
    lex = CaseLexicon(lexicon_id="slow")
    for term in ("Brambilla", "Esposito", "Ferraris", "Cinisello", "Bergamini"):
        lex.add_term(term, "test")
    rng = np.random.default_rng(0)
    items = []
    for k, text in enumerate(TEXTS):
        audio = _voice_like(k) + 0.02 * rng.standard_normal(2 * SR).astype(np.float32)
        items.append(type("Item", (), {"item_id": f"s{k}", "text": text, "audio": audio})())

    controlled = WhisperLexiconDecoder(model, processor, lex, max_new_tokens=30)
    unfiltered = WhisperLexiconDecoder(
        model, processor, lex, max_new_tokens=30,
        config=BiasConfig(boost=8.0, start_margin=1e9, start_top_k=10**6, continuation_margin=1e9,
                          max_terms_per_decode=None),
    )
    report = measure_insertion_rate(
        lambda it: controlled.decode(it.audio, SR, use_lexicon=True), items, lex,
        baseline_decode=lambda it: controlled.decode(it.audio, SR, use_lexicon=False),
    )
    loose = measure_insertion_rate(lambda it: unfiltered.decode(it.audio, SR, use_lexicon=True), items, lex)
    w, wo, lo = report.with_lexicon, report.without_lexicon, loose.with_lexicon
    print(f"\n[insertion test whisper-tiny, synthetic audio] insertion rate: controlled={w.insertion_rate:.3f} "
          f"without={wo.insertion_rate:.3f} unfiltered={lo.insertion_rate:.3f}; mean WER: controlled="
          f"{w.mean_wer:.2f} without={wo.mean_wer:.2f} unfiltered={lo.mean_wer:.2f}")
    assert w.opportunities == len(items) * 5
    assert lo.insertion_rate >= w.insertion_rate
    # Unfiltered biasing damages the hypotheses (whole terms, or fragments such as
    # "Cin"+"co" when greedy decoding cannot withdraw a started term); the
    # controlled one must stay much closer to the decoding without lexicon.
    assert lo.mean_wer > w.mean_wer
    assert abs(w.mean_wer - wo.mean_wer) < abs(lo.mean_wer - wo.mean_wer)
