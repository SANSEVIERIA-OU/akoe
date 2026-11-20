# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""omniASR CTC v2 candidate scorer (issue #58): weight mapping, folding, tiers."""
import numpy as np
import pytest

from akoe import engine
from akoe.config import Tier
from akoe.interfaces import CandidateScorer
from akoe.neural import BackendUnavailable
from akoe.transcribe import omniasr_ctc as oc
from akoe.transcribe.ctc_aligner import CTCTranscriptAligner
from akoe.transcribe.ctc_scorer import DEFAULT_CTC_MODEL, CTCForcedAlignmentScorer

# Representative parameter names of the official omniASR-CTC-1B-v2.pt (read from its pickle).
FS2_KEYS = [
    "encoder_frontend.feature_extractor.layers.0.conv.weight",
    "encoder_frontend.feature_extractor.layers.6.layer_norm.bias",
    "encoder_frontend.post_extract_layer_norm.weight",
    "encoder_frontend.model_dim_proj.bias",
    "encoder_frontend.pos_encoder.conv.weight_g",
    "encoder.layers.47.self_attn.output_proj.weight",
    "encoder.layers.3.ffn.inner_proj.bias",
    "encoder.layers.3.ffn_layer_norm.weight",
    "encoder.layer_norm.bias",
    "final_proj.weight",
]


def test_fairseq2_names_map_onto_transformers_wav2vec2():
    g, v = "G", "V"
    got = [oc.fairseq2_to_hf_key(k, g, v) for k in FS2_KEYS]
    assert got == [
        "wav2vec2.feature_extractor.conv_layers.0.conv.weight",
        "wav2vec2.feature_extractor.conv_layers.6.layer_norm.bias",
        "wav2vec2.feature_projection.layer_norm.weight",
        "wav2vec2.feature_projection.projection.bias",
        "G",
        "wav2vec2.encoder.layers.47.attention.out_proj.weight",
        "wav2vec2.encoder.layers.3.feed_forward.intermediate_dense.bias",
        "wav2vec2.encoder.layers.3.final_layer_norm.weight",
        "wav2vec2.encoder.layer_norm.bias",
        "lm_head.weight",
    ]
    with pytest.raises(KeyError):
        oc.fairseq2_to_hf_key("encoder.layers.0.mystery.weight", g, v)


def _inverse(hf_key: str) -> str:
    """fairseq2 name of a transformers key (test helper, inverse of the mapping)."""
    rules = [
        ("wav2vec2.feature_extractor.conv_layers.", "encoder_frontend.feature_extractor.layers."),
        ("wav2vec2.feature_projection.layer_norm.", "encoder_frontend.post_extract_layer_norm."),
        ("wav2vec2.feature_projection.projection.", "encoder_frontend.model_dim_proj."),
        ("wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0", "encoder_frontend.pos_encoder.conv.weight_g"),
        ("wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1", "encoder_frontend.pos_encoder.conv.weight_v"),
        ("wav2vec2.encoder.pos_conv_embed.conv.bias", "encoder_frontend.pos_encoder.conv.bias"),
        ("wav2vec2.encoder.layer_norm.", "encoder.layer_norm."),
        ("lm_head.", "final_proj."),
    ]
    for a, b in rules:
        if hf_key.startswith(a):
            return b + hf_key[len(a):]
    idx, rest = hf_key[len("wav2vec2.encoder.layers."):].split(".", 1)
    for src, dst in oc._LAYER_MAP.items():
        if rest.startswith(dst + "."):
            return f"encoder.layers.{idx}.{src}{rest[len(dst):]}"
    raise KeyError(hf_key)


def test_state_dict_mapping_is_strict_and_loads_into_wav2vec2():
    torch = pytest.importorskip("torch")
    from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

    cfg = Wav2Vec2Config(vocab_size=12, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                         intermediate_size=64, feat_extract_norm="layer", conv_bias=True,
                         do_stable_layer_norm=True, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=4,
                         mask_time_prob=0.0, mask_feature_prob=0.0)
    torch.manual_seed(0)
    ref = Wav2Vec2ForCTC(cfg).eval()
    fs2 = {_inverse(k): t.clone() for k, t in ref.state_dict().items()}
    with torch.device("meta"):
        m = Wav2Vec2ForCTC(cfg)
    mapped = oc.map_state_dict(fs2, set(m.state_dict().keys()))
    m.load_state_dict(mapped, strict=True, assign=True)
    m.eval()
    x = torch.randn(1, 4000)
    with torch.inference_mode():
        assert torch.allclose(m(x).logits, ref(x).logits, atol=1e-5)
    with pytest.raises(RuntimeError, match="incompleta"):
        oc.map_state_dict({k: t for k, t in list(fs2.items())[1:]}, set(m.state_dict().keys()))


PIECES = ["<s>", "<pad>", "</s>", "<unk>", " ", "a", "A", "b", "B", "c", "'", ",", ".", "?", "-", "3", "è", "È"]


def test_folding_merges_case_and_treats_punctuation_as_blank():
    cls, alphabet, blank, summary = oc.build_folding(PIECES, blank_id=0)
    ids = {p: i for i, p in enumerate(PIECES)}
    assert blank == 0
    for p in ("<s>", "<pad>", "</s>", "<unk>", ",", ".", "?", "-"):
        assert cls[ids[p]] == 0
    assert cls[ids["A"]] == cls[ids["a"]] and cls[ids["È"]] == cls[ids["è"]]
    assert cls[ids["'"]] not in (0, cls[ids["a"]]) and alphabet["'"] == cls[ids["'"]]
    assert " " in alphabet and "3" in alphabet and "," not in alphabet
    assert summary["n_classes"] == int(cls.max()) + 1 and summary["case_folded_tokens"] == 3
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, len(PIECES)))
    lp = x - np.log(np.exp(x).sum(axis=1, keepdims=True))
    f = oc.fold_log_probs(lp, cls, summary["n_classes"])
    assert np.allclose(np.log(np.exp(f.astype(np.float64)).sum(axis=1)), 0.0, atol=1e-5)  # still a distribution
    assert np.allclose(f[:, cls[ids["a"]]], np.logaddexp(lp[:, ids["a"]], lp[:, ids["A"]]), atol=1e-5)


def _toy_scorer(text: str, frames_per_char: int = 3):
    """OmniASRCTCScorer whose emissions 'say' `text` in written form (case, comma)."""
    cls, alphabet, blank, summary = oc.build_folding(PIECES, blank_id=0)
    ids = {p: i for i, p in enumerate(PIECES)}
    seq = []
    for ch in text:
        seq += [ids[ch]] * frames_per_char + [0]

    def emit(a16):
        lp = np.full((len(seq), len(PIECES)), np.log(0.02 / (len(PIECES) - 1)))
        lp[np.arange(len(seq)), seq] = np.log(0.98)
        return oc.fold_log_probs(lp, cls, summary["n_classes"])

    s = oc.OmniASRCTCScorer()
    s._emit, s._vocab, s._blank, s._delim = emit, alphabet, blank, " "
    return s


def test_scorer_ignores_case_and_punctuation_of_the_candidate():
    s = _toy_scorer("Abc, cab.")
    assert isinstance(s, CandidateScorer) and s.name == "omniASR_CTC_1B_v2/ctc-forced-alignment"
    sc = s.score(np.zeros(1600, dtype=np.float32), 16000, ["abc cab", "ABC, CAB!", "abc bab"])
    assert sc[0] == pytest.approx(sc[1]) and sc[0] > sc[2]
    assert sc[0] > -0.05  # what the model would decode freely, up to case and punctuation
    words, _ = s.normalize_text("L'a-b, 3")
    assert words == ["l'a", "b", "3"] or words == ["'a", "b", "3"]


def test_unpinned_weights_are_refused_before_download(tmp_path, monkeypatch):
    s = oc.OmniASRCTCScorer(oc.WORKSTATION_OMNIASR_MODEL, cache_dir=tmp_path)
    assert not oc.weights_pinned(oc.WORKSTATION_OMNIASR_MODEL)
    called = []
    monkeypatch.setattr(oc, "download_verified", lambda *a, **k: called.append(a))
    with pytest.raises(BackendUnavailable, match="SHA-256"):
        s._fetch(oc.OMNIASR_MODELS[oc.WORKSTATION_OMNIASR_MODEL]["file"])
    assert not called
    off = oc.OmniASRCTCScorer(cache_dir=tmp_path, local_files_only=True)
    with pytest.raises(BackendUnavailable, match="offline"):
        off._fetch(oc.TOKENIZER_FILE)
    (tmp_path / oc.TOKENIZER_FILE).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="diverso"):
        off._fetch(oc.TOKENIZER_FILE)
    with pytest.raises(ValueError):
        oc.OmniASRCTCScorer("omniASR_CTC_99B_v9")


def test_tier_defaults_and_overrides(monkeypatch):
    assert oc.weights_pinned(oc.DEFAULT_OMNIASR_MODEL)
    assert engine.default_candidate_ctc_model(Tier.EDGE) == "omniASR_CTC_1B_v2"
    # the 3B is the workstation model once its hash is pinned; until then the 1B
    assert engine.default_candidate_ctc_model(Tier.WORKSTATION) == "omniASR_CTC_1B_v2"
    monkeypatch.setitem(oc.OMNIASR_KNOWN_SHA256, "omniASR-CTC-3B-v2.pt", "0" * 64)
    assert engine.default_candidate_ctc_model(Tier.WORKSTATION) == "omniASR_CTC_3B_v2"
    assert engine.default_candidate_ctc_model(Tier.CLOUD) == "omniASR_CTC_3B_v2"
    assert engine.default_candidate_ctc_model(Tier.EDGE) == "omniASR_CTC_1B_v2"

    b = engine.DefaultBackends()
    cand = b.candidate_ctc_scorer(Tier.EDGE)
    assert isinstance(cand, oc.OmniASRCTCScorer) and cand.model_card == "omniASR_CTC_1B_v2"
    assert cand.provenance is None  # nothing loaded
    al = b.aligner()
    assert type(al.scorer) is CTCForcedAlignmentScorer and al.scorer.model_id == DEFAULT_CTC_MODEL
    assert DEFAULT_CTC_MODEL == "jonatasgrosman/wav2vec2-xls-r-1b-italian"
    assert isinstance(CTCTranscriptAligner().scorer, CTCForcedAlignmentScorer)
    o = engine.DefaultBackends(ctc_model="facebook/wav2vec2-base-10k-voxpopuli-ft-it")
    assert o.candidate_ctc_scorer(Tier.EDGE) is o.ctc_scorer()  # explicit model: both roles, as before


@pytest.mark.slow
def test_real_omniasr_prefers_the_spoken_sentence():
    from tests.real_audio import italian_example

    audio, sr, text = italian_example()
    s = oc.OmniASRCTCScorer()
    scores = s.score_detailed(audio, sr, [text, "Buongiorno a tutti e benvenuti a Roma",
                                          "Buonasera a tutti e benvenuti a bordo"])
    assert scores[0].score > scores[1].score and scores[0].score > scores[2].score
    prov = s.provenance
    assert prov.weights["omniASR-CTC-1B-v2.pt"] == oc.OMNIASR_KNOWN_SHA256["omniASR-CTC-1B-v2.pt"]


def test_download_verified_checks_the_hash_before_the_final_name(tmp_path):
    import hashlib

    from akoe.neural import download_verified

    src = tmp_path / "src.bin"
    src.write_bytes(b"weights" * 1000)
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    dest = tmp_path / "cache" / "w.bin"
    assert download_verified(src.as_uri(), dest, sha) == sha and dest.read_bytes() == src.read_bytes()
    bad = tmp_path / "cache" / "bad.bin"
    with pytest.raises(RuntimeError, match="diverso"):
        download_verified(src.as_uri(), bad, "0" * 64)
    assert not bad.exists() and not (tmp_path / "cache" / "bad.bin.part").exists()
    with pytest.raises(RuntimeError, match="rifiutato"):
        download_verified(src.as_uri(), tmp_path / "x.bin", None)
    assert download_verified(src.as_uri(), tmp_path / "y.bin", None, allow_unknown=True) == sha
