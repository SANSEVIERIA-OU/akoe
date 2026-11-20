# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
import itertools
import os

import numpy as np
import pytest

from akoe.interfaces import CandidateScorer
from akoe.transcribe.ctc_scorer import (
    CTCForcedAlignmentScorer,
    ctc_min_frames,
    ctc_viterbi,
)

SR = 16000
VOCAB = {"<pad>": 0, "|": 1, "'": 2, **{c: i + 3 for i, c in enumerate("abcdefghilmnopqrstuvzàèéìòù")}}


def _collapse(path, blank=0):
    out, prev = [], None
    for p in path:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return out


def _log_softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


@pytest.mark.parametrize("seed,targets", [(0, [1, 2]), (1, [1, 1]), (2, [2, 3, 2]), (3, [3])])
def test_viterbi_matches_brute_force(seed, targets):
    rng = np.random.default_rng(seed)
    T, C = 6, 4
    lp = _log_softmax(rng.normal(size=(T, C)))
    best = max(
        (sum(lp[t, p[t]] for t in range(T)) for p in itertools.product(range(C), repeat=T)
         if _collapse(p) == targets),
        default=-np.inf,
    )
    res = ctc_viterbi(lp, targets, blank=0)
    assert res.logprob == pytest.approx(best)
    assert res.frame_logprobs.sum() == pytest.approx(res.logprob)
    labels = [0 if s % 2 == 0 else targets[(s - 1) // 2] for s in res.states]
    assert _collapse(labels) == targets


def test_viterbi_matches_torchaudio_when_available():
    torch = pytest.importorskip("torch")
    F = pytest.importorskip("torchaudio.functional")
    if not hasattr(F, "forced_align"):
        pytest.skip("torchaudio.functional.forced_align not available")
    rng = np.random.default_rng(42)
    for _ in range(5):
        lp = _log_softmax(rng.normal(size=(80, 12))).astype(np.float32)
        tg = list(rng.integers(1, 12, size=20))
        _, scores = F.forced_align(torch.from_numpy(lp)[None], torch.tensor([tg], dtype=torch.int32), blank=0)
        assert ctc_viterbi(lp, tg, 0).logprob == pytest.approx(float(scores.sum()), abs=1e-3)


def test_infeasible_alignment_is_minus_inf():
    lp = np.log(np.full((3, 4), 0.25))
    assert ctc_min_frames([1, 1, 2]) == 4
    assert ctc_viterbi(lp, [1, 1, 2]).logprob == float("-inf")
    assert np.isfinite(ctc_viterbi(lp, [1, 2, 3]).logprob)
    with pytest.raises(ValueError):
        ctc_viterbi(lp, [0, 1])


def synthetic_emitter(text: str, frames_per_char: int = 4, p: float = 0.9):
    """Emissions that 'say' `text`: each char held for a few frames with a
    blank in between, the rest of the mass spread uniformly."""
    seq = []
    for i, w in enumerate(text.split()):
        if i:
            seq.append(VOCAB["|"])
        seq.extend(VOCAB[c] for c in w)

    def emit(audio16k: np.ndarray) -> np.ndarray:
        labels = [0, 0]
        for tok in seq:
            labels += [tok] * frames_per_char + [0]
        labels += [0, 0]
        C = len(VOCAB)
        probs = np.full((len(labels), C), (1 - p) / (C - 1))
        probs[np.arange(len(labels)), labels] = p
        return np.log(probs)

    return emit


@pytest.fixture
def scorer():
    return CTCForcedAlignmentScorer.from_emission_fn(synthetic_emitter("vado a casa"), VOCAB, name="synthetic")


def test_true_candidate_wins_and_words_are_aligned(scorer):
    assert isinstance(scorer, CandidateScorer)
    audio = np.zeros(SR, dtype=np.float32)
    res = scorer.score_detailed(audio, SR, ["Vado a casa.", "vado a cassa", "andiamo al mare oggi"])
    assert res[0].score == pytest.approx(0.0, abs=1e-9)  # exactly the free decoding
    assert res[0].score > res[1].score > res[2].score
    assert all(r.relative_per_frame <= 1e-12 for r in res)
    assert res[0].normalized == "vado a casa"
    assert [w.word for w in res[0].words] == ["vado", "a", "casa"]
    starts = [w.start for w in res[0].words]
    assert starts == sorted(starts) and all(w.end > w.start for w in res[0].words)
    assert scorer.score(audio, SR, ["vado a casa"]) == pytest.approx([res[0].score])


def test_normalization_and_unscoreable_candidates(scorer):
    words, warns = scorer.normalize_text("Dov’è l'acqua? 3 euro, ï")
    assert words == ["dov'è", "l'acqua", "euro", "i"]
    assert any("3" in w for w in warns) and any("ï" in w for w in warns)
    audio = np.zeros(SR, dtype=np.float32)
    empty, too_long = scorer.score_detailed(audio, SR, ["123 !!", "a" * 200])
    assert not empty.feasible and empty.score == float("-inf")
    assert not too_long.feasible and "troppo lunga" in too_long.warnings[-1]


def test_scorer_without_apostrophe_in_alphabet_splits_words():
    vocab = {k: v for k, v in VOCAB.items() if k != "'"}
    sc = CTCForcedAlignmentScorer.from_emission_fn(synthetic_emitter("vado"), vocab)
    assert sc.normalize_text("l'acqua")[0] == ["l", "acqua"]


def test_local_model_load_takes_blank_from_vocab_and_hashes_weights(tmp_path):
    pytest.importorskip("transformers")
    from tests.tiny_models import make_tiny_wav2vec2_ctc

    d = make_tiny_wav2vec2_ctc(tmp_path)
    sc = CTCForcedAlignmentScorer(str(d))
    prov = sc.load()
    assert sc._blank == 0 and prov.extra["config_pad_token_id"] == 1
    assert "model.safetensors" in prov.weights and prov.revision is None
    audio = np.random.default_rng(0).normal(0, 0.1, SR * 2).astype(np.float32)
    res = sc.score_detailed(audio, SR, ["vado a casa", "si"])
    assert all(r.feasible and r.score <= 1e-9 for r in res)
    assert res[0].n_frames == sc.emissions(audio, SR)[0].shape[0]


@pytest.mark.slow
def test_real_ctc_model_prefers_the_spoken_sentence():
    from tests.real_audio import italian_example

    model_id = os.environ.get("AKOE_TEST_CTC_MODEL", "facebook/wav2vec2-base-10k-voxpopuli-ft-it")
    audio, sr, text = italian_example()
    sc = CTCForcedAlignmentScorer(model_id)
    res = sc.score_detailed(audio, sr, [text, text.replace("bordo", "borgo"), "Oggi piove molto forte in montagna"])
    assert res[0].score > res[1].score > res[2].score
    assert [w.word for w in res[0].words] == text.lower().split()
    assert res[0].words[-1].end <= len(audio) / sr + 0.05
    prov = sc.provenance
    assert prov.revision and prov.license and any(k.endswith((".bin", ".safetensors")) for k in prov.weights)
