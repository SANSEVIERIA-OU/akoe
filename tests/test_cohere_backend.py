# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Cohere Transcribe backend (#54) on a tiny randomly initialised model with
the real transformers classes (no downloads); the real model is slow-only."""
import os

import numpy as np
import pytest

from akoe.config import StageKind
from akoe.interfaces import ASRBackend, CandidateScorer
from akoe.neural import BackendUnavailable
from akoe.transcribe.cohere_backend import KNOWN_REVISIONS, KNOWN_WEIGHTS, CohereTranscribeBackend
from akoe.transcribe.hf_asr import ModelIntegrityError, TransformersDecoderScorer
from akoe.transcribe.whisper_backend import SamplingConfig

SR = 16000
pytest.importorskip("torch")
pytest.importorskip("transformers")


class EvenTimer:
    name = "even-timer"

    def __init__(self):
        self.calls = []

    def word_times(self, audio, sr, words):
        self.calls.append(list(words))
        step = len(audio) / sr / max(len(words), 1)
        return [(i * step, (i + 1) * step) for i in range(len(words))]


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    from tests.tiny_cohere import make_cohere_dir

    return make_cohere_dir(tmp_path_factory.mktemp("cohere") / "tiny-cohere")


def _audio(seconds=2.0, seed=0):
    return np.random.default_rng(seed).normal(0, 0.1, int(SR * seconds)).astype(np.float32)


def test_defaults_and_protocol():
    be = CohereTranscribeBackend()
    assert isinstance(be, ASRBackend)
    assert be.name == "cohere-transcribe-03-2026/generic"
    assert be.revision == KNOWN_REVISIONS["CohereLabs/cohere-transcribe-03-2026"]
    assert be.expected_weights == KNOWN_WEIGHTS["CohereLabs/cohere-transcribe-03-2026"]
    assert be.dtype == "bfloat16" and be.language == "it"
    with pytest.raises(ValueError, match="Lingua non supportata"):
        CohereTranscribeBackend(language="ru")


def test_nbest_scores_words_and_provenance(tiny_dir):
    timer = EvenTimer()
    be = CohereTranscribeBackend(str(tiny_dir), aligner=timer, max_new_tokens=6, num_beams=4, dtype="float32")
    segs = be.transcribe(_audio(3.0), SR, segments=[(0.5, 2.5)], n_best=3)
    assert len(segs) == 1
    s = segs[0]
    assert s.audio_provenance == StageKind.DETERMINISTIC
    assert (s.start, s.end) == (0.5, 2.5)
    texts = [h.text for h in s.hypotheses]
    assert 1 <= len(texts) <= 3 and len(set(texts)) == len(texts)
    scores = [h.score for h in s.hypotheses]
    assert scores == sorted(scores, reverse=True) and all(sc <= 0 for sc in scores)
    assert all(h.origin == be.name for h in s.hypotheses)
    if s.hypotheses[0].text:
        assert [w.text for w in s.words] == s.hypotheses[0].text.split()
        assert all(0 < w.confidence <= 1 for w in s.words)
        assert s.words[0].start == pytest.approx(0.5) and s.words[-1].end == pytest.approx(2.5)
        assert timer.calls == [[w.text for w in s.words]]
    prov = be.provenance
    assert prov.revision is None and "model.safetensors" in prov.weights  # local folder: hashes are the identity
    assert prov.extra["trust_remote_code"] is False and prov.extra["word_timer"]["name"] == "even-timer"
    assert prov.dtype == "float32" and prov.device == "cpu"


def test_beam_scores_equal_teacher_forced_scores_and_sampling_is_reproducible(tiny_dir):
    cfg = SamplingConfig(n_samples=3, temperature=1.0, seed=7)
    be = CohereTranscribeBackend(str(tiny_dir), aligner=EvenTimer(), max_new_tokens=5, sampling=cfg,
                                 dtype="float32")
    a = _audio(2.0, seed=1)
    first = be.transcribe(a, SR, n_best=2)[0]
    again = be.transcribe(a, SR, n_best=2)[0]
    assert [(h.text, h.origin) for h in first.hypotheses] == [(h.text, h.origin) for h in again.hypotheses]
    assert {h.origin for h in first.hypotheses} <= {be.name, f"{be.name}/sampling"}
    # the same number whether it comes from transcribe or from the scorer view
    scorer = TransformersDecoderScorer(be)
    assert isinstance(scorer, CandidateScorer)
    ctx = be._prepare(a)
    cands = be._candidates(ctx, 2)
    tf = {text: float(np.mean(lp)) for (_, text, _), lp in zip(cands, be._score_ids(ctx, [c[0] for c in cands]))}
    assert {h.text: h.score for h in first.hypotheses} == pytest.approx(tf)  # ranking = teacher-forced score
    texts = [h.text for h in first.hypotheses if h.text] or ["ciao mondo"]
    for d in scorer.score_detailed(a, SR, texts):
        assert d.tokens[-1].token == "<|endoftext|>" and d.mean_logprob <= 0
    assert len(scorer.score(a, SR, ["ciao mondo", "vado a casa"])) == 2


def test_bfloat16_and_low_memory_release(tiny_dir):
    be = CohereTranscribeBackend(str(tiny_dir), aligner=EvenTimer(), max_new_tokens=4, low_memory=True)
    segs = be.transcribe(_audio(1.0), SR, n_best=1)
    assert segs[0].hypotheses and be.provenance.dtype == "bfloat16"
    assert be._model is None  # released after decoding (low_memory)
    assert be.transcribe(_audio(1.0), SR, n_best=1)[0].hypotheses  # reloads lazily


def test_long_regions_are_split_into_30_s_windows(tiny_dir):
    be = CohereTranscribeBackend(str(tiny_dir), word_timestamps=False, max_new_tokens=2, dtype="float32")
    segs = be.transcribe(np.zeros(SR * 65, dtype=np.float32), SR, n_best=1)
    assert [(s.start, s.end) for s in segs] == [(0.0, 30.0), (30.0, 60.0), (60.0, 65.0)]


def test_weights_are_checked_against_published_hashes(tiny_dir):
    be = CohereTranscribeBackend(str(tiny_dir), expected_weights={"model.safetensors": "0" * 64})
    with pytest.raises(ModelIntegrityError, match="SHA-256"):
        be.load()


def test_missing_model_is_a_skipped_stage(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    be = CohereTranscribeBackend("CohereLabs/cohere-transcribe-03-2026", local_files_only=True)
    with pytest.raises(BackendUnavailable) as exc:
        be.transcribe(np.zeros(SR, dtype=np.float32), SR)
    assert isinstance(exc.value, NotImplementedError)


@pytest.mark.slow
def test_real_model_italian():
    """Needs the gated weights: HF_TOKEN with accepted conditions, or
    AKOE_TEST_COHERE_DIR pointing at a local copy (checked against the
    published SHA-256)."""
    from tests.real_audio import italian_example

    model = os.environ.get("AKOE_TEST_COHERE_DIR", "CohereLabs/cohere-transcribe-03-2026")
    be = CohereTranscribeBackend(model, expected_weights=KNOWN_WEIGHTS["CohereLabs/cohere-transcribe-03-2026"])
    audio, sr, _ = italian_example()
    try:
        segs = be.transcribe(audio, sr, n_best=3)
    except BackendUnavailable as e:
        pytest.skip(str(e))
    assert "tutti" in segs[0].hypotheses[0].text.lower()
    assert segs[0].words and all(w.confidence is not None for w in segs[0].words)
