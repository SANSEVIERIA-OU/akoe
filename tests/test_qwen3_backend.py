# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Qwen3-ASR backend and Qwen3-ForcedAligner (#55) on tiny randomly
initialised models with the real transformers classes (no downloads); the
real models are slow-only."""
import numpy as np
import pytest

from akoe.config import StageKind
from akoe.interfaces import ASRBackend
from akoe.neural import BackendUnavailable
from akoe.transcribe.qwen3_backend import (
    DEFAULT_ALIGNER,
    DEFAULT_MODEL,
    KNOWN_REVISIONS,
    KNOWN_WEIGHTS,
    Qwen3ASRBackend,
    Qwen3ForcedAligner,
)
from akoe.transcribe.word_timing import WordTimer

SR = 16000
pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from tests.tiny_qwen3 import make_qwen3_aligner_dir, make_qwen3_asr_dir

    root = tmp_path_factory.mktemp("qwen3")
    return make_qwen3_asr_dir(root / "tiny-qwen3-asr"), make_qwen3_aligner_dir(root / "tiny-qwen3-aligner")


def _audio(seconds=2.0, seed=0):
    return np.random.default_rng(seed).normal(0, 0.1, int(SR * seconds)).astype(np.float32)


def test_defaults_and_protocol():
    be = Qwen3ASRBackend()
    assert isinstance(be, ASRBackend)
    assert be.name == "Qwen3-ASR-1.7B-hf/generic"
    assert be.revision == KNOWN_REVISIONS[DEFAULT_MODEL] and be.expected_weights == KNOWN_WEIGHTS[DEFAULT_MODEL]
    al = Qwen3ForcedAligner()
    assert isinstance(al, WordTimer) and al.revision == KNOWN_REVISIONS[DEFAULT_ALIGNER]


def test_nbest_scores_and_forced_aligner_times(tiny):
    asr_dir, aligner_dir = tiny
    be = Qwen3ASRBackend(str(asr_dir), aligner_model=str(aligner_dir), max_new_tokens=6, dtype="float32",
                         expected_weights={})
    be._aligner_arg = Qwen3ForcedAligner(str(aligner_dir), dtype="float32", expected_weights={})
    segs = be.transcribe(_audio(3.0), SR, segments=[(0.5, 2.5)], n_best=3)
    s = segs[0]
    assert s.audio_provenance == StageKind.DETERMINISTIC and (s.start, s.end) == (0.5, 2.5)
    scores = [h.score for h in s.hypotheses]
    assert 1 <= len(scores) <= 3 and scores == sorted(scores, reverse=True) and all(x <= 0 for x in scores)
    assert len({h.text for h in s.hypotheses}) == len(s.hypotheses)
    if s.hypotheses[0].text:
        assert [w.text for w in s.words] == s.hypotheses[0].text.split()
        assert all(0.5 <= w.start <= w.end <= 2.5 for w in s.words)
        assert all(w.confidence is None or 0 < w.confidence <= 1 for w in s.words)
    prov = be.provenance
    assert prov.extra["language_forcing"].startswith("assistant prefill") and prov.extra["post_editing"] is None
    if s.hypotheses[0].text:
        assert prov.extra["word_timer"]["name"] == "tiny-qwen3-aligner/forced-aligner"


def test_forced_aligner_maps_times_back_to_the_asr_words(tiny):
    _, aligner_dir = tiny
    al = Qwen3ForcedAligner(str(aligner_dir), dtype="float32", expected_weights={})
    words = ["Ciao,", "mondo", "—", "vado", "a", "casa."]
    times = al.word_times(_audio(2.0), SR, words)
    assert len(times) == len(words)
    assert times[2] is None  # punctuation only: nothing to align
    timed = [t for t in times if t is not None]
    assert len(timed) == 5 and all(0 <= s <= e for s, e in timed)
    starts = [s for s, _ in timed]
    assert starts == sorted(starts)  # monotonic after the reference LIS fix
    al.unload()
    assert al._model is None


def test_low_memory_releases_the_asr_model_before_aligning(tiny):
    asr_dir, aligner_dir = tiny
    seen = {}

    class SpyTimer:
        name = "spy"

        def word_times(self, audio, sr, words):
            seen["asr_loaded_while_aligning"] = be._model is not None
            return [None] * len(words)

    be = Qwen3ASRBackend(str(asr_dir), aligner=SpyTimer(), max_new_tokens=4, low_memory=True, expected_weights={})
    s = be.transcribe(_audio(1.0), SR, n_best=1)[0]
    assert be._model is None
    if s.hypotheses[0].text:
        assert seen == {"asr_loaded_while_aligning": False}
        assert all((w.start, w.end) == (0.0, 1.0) for w in s.words) or s.words  # untimed words span the gap


def test_score_texts_ranks_with_the_same_number_as_transcribe(tiny):
    asr_dir, _ = tiny
    be = Qwen3ASRBackend(str(asr_dir), word_timestamps=False, max_new_tokens=5, dtype="float32",
                         expected_weights={})
    a = _audio(1.5, seed=3)
    s = be.transcribe(a, SR, n_best=2)[0]
    # the ranking score is the teacher-forced mean log-probability of the generated ids
    ctx = be._prepare(a)
    cands = be._candidates(ctx, 2)
    tf = {text: float(np.mean(lp)) for (_, text, _), lp in zip(cands, be._score_ids(ctx, [c[0] for c in cands]))}
    assert {h.text: h.score for h in s.hypotheses} == pytest.approx(tf)
    texts = [h.text for h in s.hypotheses if h.text] or ["ciao mondo"]
    for d in be.score_texts(a, SR, texts):  # typed text: re-encoded, same kind of number
        assert d.tokens[-1].token == "<|im_end|>" and d.mean_logprob <= 0


def test_missing_model_is_a_skipped_stage():
    be = Qwen3ASRBackend("Qwen/akoe-test-not-a-model", local_files_only=True)
    with pytest.raises(BackendUnavailable):
        be.transcribe(np.zeros(SR, dtype=np.float32), SR)


@pytest.mark.slow
def test_real_model_italian():
    from tests.real_audio import italian_example

    audio, sr, _ = italian_example()
    be = Qwen3ASRBackend(low_memory=True)
    segs = be.transcribe(audio, sr, n_best=3)
    assert "tutti" in segs[0].hypotheses[0].text.lower()
    w = segs[0].words
    assert w and all(x.confidence is not None for x in w)
    assert w[0].start < w[-1].end <= len(audio) / sr + 1e-6
