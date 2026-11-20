import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from forensicaudio.config import StageKind
from forensicaudio.interfaces import ASRBackend, CandidateScorer
from forensicaudio.transcribe.whisper_backend import (
    KNOWN_REVISIONS,
    SamplingConfig,
    WhisperDecoderScorer,
    WhisperHFBackend,
    mbr_select,
    word_edit_distance,
)
from tests.tiny_models import make_lora_adapter, make_tiny_whisper

SR = 16000


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    return make_tiny_whisper(tmp_path_factory.mktemp("models"))


@pytest.fixture(scope="module")
def tiny(tiny_dir):
    be = WhisperHFBackend(str(tiny_dir), max_new_tokens=10, sampling=SamplingConfig(n_samples=3, seed=0))
    be.load()
    return be


@pytest.fixture(scope="module")
def noise():
    return np.random.default_rng(0).normal(0, 0.1, SR * 3).astype(np.float32)


def test_protocols_and_names(tiny):
    assert isinstance(tiny, ASRBackend)
    assert isinstance(WhisperDecoderScorer(tiny), CandidateScorer)
    assert tiny.name == "tiny-whisper/generic"
    assert WhisperHFBackend().revision == KNOWN_REVISIONS["openai/whisper-tiny"]
    assert WhisperHFBackend().provenance is None  # lazy


def test_provenance_hashes_every_loaded_file(tiny, tiny_dir):
    prov = tiny.provenance
    assert prov.model_id == str(tiny_dir) and prov.revision is None  # local dir: hashes are the identity
    assert "model.safetensors" in prov.weights and "tokenizer.json" in prov.weights
    assert prov.device == "cpu" and prov.extra["language"] == "it"


def test_transcribe_keeps_all_candidates_with_origin_and_provenance(tiny, noise):
    segs = tiny.transcribe(noise, SR, segments=[(0.5, 2.0), (2.0, 3.0)], n_best=3)
    assert [(s.start, s.end) for s in segs] == [(0.5, 2.0), (2.0, 3.0)]
    for s in segs:
        assert s.audio_provenance == StageKind.NEURAL
        origins = {h.origin for h in s.hypotheses}
        assert origins <= {"tiny-whisper/generic", "tiny-whisper/generic/sampling"}
        assert "tiny-whisper/generic" in origins
        texts = [h.text for h in s.hypotheses]
        assert len(texts) == len(set(texts))  # deduplicated, not dropped
        scores = [h.score for h in s.hypotheses]
        assert scores == sorted(scores, reverse=True)
        # every score is the teacher-forced mean log-prob, the same measure for beam and sampling
        chunk = noise[int(s.start * SR): int(s.end * SR)]
        tf = tiny.score_texts(chunk, SR, texts)
        assert np.allclose([t.mean_logprob for t in tf], scores, atol=1e-4)
        for w in s.words:
            assert s.start <= w.start <= w.end <= s.end and w.origin == s.hypotheses[0].origin


def test_provenance_override_and_audio_variant(tiny, noise):
    segs = tiny.transcribe(noise, SR, n_best=1, audio_provenance=StageKind.DETERMINISTIC,
                           audio_variant="remix-0.2")
    assert segs[0].audio_provenance == StageKind.DETERMINISTIC
    assert all(h.audio_variant == "remix-0.2" for h in segs[0].hypotheses)


def test_long_regions_are_split_into_30s_windows(tiny):
    audio = np.zeros(SR * 65, dtype=np.float32)
    segs = tiny.transcribe(audio, SR, n_best=1)
    assert [(s.start, s.end) for s in segs] == [(0.0, 30.0), (30.0, 60.0), (60.0, 65.0)]


def test_teacher_forced_score_is_consistent(tiny, noise):
    res = tiny.score_texts(noise, SR, ["ciao come stai", "àè"])
    r = res[0]
    assert r.n_tokens == len(b" ciao come stai") + 1  # byte tokens + end-of-text
    assert r.tokens[-1].token == "<|endoftext|>"
    assert r.total_logprob == pytest.approx(sum(t.logprob for t in r.tokens))
    assert r.mean_logprob == pytest.approx(r.total_logprob / r.n_tokens)
    assert all(t.logprob <= 0 for t in r.tokens)
    # batched scoring equals one-by-one scoring (padding does not leak)
    alone = tiny.score_text(noise, SR, "àè")
    assert alone.total_logprob == pytest.approx(res[1].total_logprob, abs=1e-4)
    sc = WhisperDecoderScorer(tiny)
    assert sc.score(noise, SR, ["ciao come stai"]) == pytest.approx([r.mean_logprob])
    with pytest.raises(ValueError, match="30 s"):
        tiny.score_text(np.zeros(SR * 31, dtype=np.float32), SR, "x")


def test_word_grouping_and_timings(tiny, noise):
    ids = tiny._tok.encode(" ciao àè, va", add_special_tokens=False)
    words = [tiny._tok.decode(w).strip() for w in tiny._split_words(ids)]
    assert words == ["ciao", "àè,", "va"]  # multi-byte letters and punctuation stay in their word
    timings = tiny._word_timings(noise[:SR * 2], "ciao come stai")
    assert [t.text for t in timings] == ["ciao", "come", "stai"]
    starts = [t.start for t in timings]
    assert starts == sorted(starts)
    assert all(0.0 <= t.start <= t.end <= 2.0 and 0.0 <= t.probability <= 1.0 for t in timings)


def test_lora_adapter_is_loaded_hashed_and_named(tiny_dir, tmp_path, noise):
    adapter = make_lora_adapter(tiny_dir, tmp_path / "adapter")
    be = WhisperHFBackend(str(tiny_dir), adapter_dir=adapter, max_new_tokens=5)
    prov = be.load()
    assert be.name.startswith("tiny-whisper/adapted:") and len(be.adapter_id) == 12
    assert prov.adapter["weights"]["adapter_model.safetensors"].startswith(be.adapter_id)
    assert prov.adapter["peft_type"] == "LORA"
    segs = be.transcribe(noise, SR, n_best=2)
    assert all(h.origin.startswith(be.name) for h in segs[0].hypotheses)
    generic = WhisperHFBackend(str(tiny_dir))
    assert generic.score_text(noise, SR, "ciao").total_logprob != pytest.approx(
        be.score_text(noise, SR, "ciao").total_logprob, abs=1e-6)
    named = WhisperHFBackend(str(tiny_dir), adapter_dir=adapter, adapter_id="subj-v1")
    assert named.name == "tiny-whisper/adapted:subj-v1"


def test_adapter_for_another_base_model_is_refused(tiny_dir, tmp_path):
    adapter = make_lora_adapter(tiny_dir, tmp_path / "adapter")
    cfg_path = adapter / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["base_model_name_or_path"] = "openai/whisper-small"
    cfg_path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="whisper-small"):
        WhisperHFBackend(str(tiny_dir), adapter_dir=adapter).load()
    with pytest.raises(FileNotFoundError):
        WhisperHFBackend(str(tiny_dir), adapter_dir=tmp_path)


def test_mbr_keeps_choice_central():
    texts = ["vado a casa", "vado a casa adesso", "vado a cassa", "boh"]
    idx, risks = mbr_select(texts)
    assert idx == 0 and len(risks) == 4 and risks[3] == max(risks)
    assert word_edit_distance(["a", "b", "c"], ["a", "c"]) == 1
    with pytest.raises(ValueError):
        mbr_select([])


@pytest.mark.slow
def test_real_whisper_tiny_italian():
    from tests.real_audio import italian_example

    audio, sr, text = italian_example()
    be = WhisperHFBackend("openai/whisper-tiny", sampling=SamplingConfig(n_samples=3, seed=1))
    segs = be.transcribe(audio, sr, n_best=4)
    top = segs[0].hypotheses[0].text.lower()
    assert "tutti" in top and "bordo" in top
    beam = [h for h in segs[0].hypotheses if h.origin == "whisper-tiny/generic"]
    # regression: Whisper's own generate returned N copies of the best beam
    assert len({h.text for h in beam}) >= 2
    assert [w.text for w in segs[0].words] and segs[0].words[-1].end <= segs[0].end
    prov = be.provenance
    assert prov.revision == KNOWN_REVISIONS["openai/whisper-tiny"] and "model.safetensors" in prov.weights
    sc = WhisperDecoderScorer(be).score(audio, sr, [text, "Oggi piove molto forte in montagna"])
    assert sc[0] > sc[1]
