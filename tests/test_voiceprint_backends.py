# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
import numpy as np
import pytest

from akoe.interfaces import VoiceprintBackend
from akoe.speaker.voiceprint_backends import (
    ECAPA_REVISION,
    REDIMNET_KNOWN_SHA256,
    EcapaVoiceprint,
    ReDimNetVoiceprint,
)


def test_backends_satisfy_protocol_and_are_lazy():
    for be in (EcapaVoiceprint(), ReDimNetVoiceprint()):
        assert isinstance(be, VoiceprintBackend)
        assert be.provenance is None  # nothing loaded or downloaded yet
    assert EcapaVoiceprint().revision == ECAPA_REVISION
    assert EcapaVoiceprint("someone/other-ecapa").revision is None


def test_too_short_segment_is_rejected_before_loading():
    be = EcapaVoiceprint()
    with pytest.raises(ValueError, match="troppo corto"):
        be.embed(np.zeros(1000, dtype=np.float32), 16000)
    assert be.provenance is None


def test_redimnet_refuses_changed_or_unknown_weights(tmp_path):
    be = ReDimNetVoiceprint()
    fake = tmp_path / be.weights_file
    fake.write_bytes(b"not the verified weights")
    with pytest.raises(RuntimeError, match="diverso"):
        be._verify(fake)
    other = ReDimNetVoiceprint(model_name="zz")
    f2 = tmp_path / other.weights_file
    f2.write_bytes(b"x")
    assert other.weights_file not in REDIMNET_KNOWN_SHA256
    with pytest.raises(RuntimeError, match="non hanno un SHA-256 verificato"):
        other._verify(f2)
    allowed = ReDimNetVoiceprint(model_name="zz", allow_unknown_weights=True)
    assert len(allowed._verify(f2)) == 64


def _cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


@pytest.mark.slow
@pytest.mark.parametrize("backend_cls", [EcapaVoiceprint, ReDimNetVoiceprint])
def test_real_embeddings_separate_speakers(backend_cls):
    from tests.real_audio import english_examples, italian_example

    it, sr, _ = italian_example()
    be = backend_cls()
    h = len(it) // 2
    a, b = be.embed(it[:h], sr), be.embed(it[h:], sr)
    others = [be.embed(x, s) for x, s in english_examples()]
    assert a.shape == (192,) and a.dtype == np.float32
    same = _cos(a, b)
    assert all(same > _cos(a, o) + 0.2 for o in others)
    prov = be.provenance
    assert prov is not None and prov.revision and prov.device == "cpu"
    assert prov.weights and all(len(v) == 64 for v in prov.weights.values())
    # deterministic given fixed weights
    assert np.allclose(be.embed(it[:h], sr), a, atol=1e-5)


# --------------------------------------------------------------------------
# ReDimNet2 and w2v-BERT 2.0 SV (issue #57)
# --------------------------------------------------------------------------


def test_new_backends_are_lazy_and_w2v_bert_is_the_default():
    from akoe.speaker.voiceprint_backends import (
        REDIMNET2_KNOWN_SHA256,
        ReDimNet2Voiceprint,
        W2VBertSVVoiceprint,
        default_voiceprint_backend,
    )

    for be in (ReDimNet2Voiceprint(), W2VBertSVVoiceprint()):
        assert isinstance(be, VoiceprintBackend) and be.provenance is None
    assert isinstance(default_voiceprint_backend(), W2VBertSVVoiceprint)
    d = default_voiceprint_backend("redimnet2-b6")
    assert isinstance(d, ReDimNet2Voiceprint) and d.name == "redimnet2-b6-vb2+vox2_v0-lm"
    assert d.weights_file in REDIMNET2_KNOWN_SHA256
    assert d.weights_url.endswith("/releases/download/v1.0.0/b6-vb2+vox2_v0-lm.pt")
    assert default_voiceprint_backend("redimnet2-b6-vox2").weights_file == "b6-vox2-lm.pt"
    assert isinstance(default_voiceprint_backend("w2v-bert-sv"), W2VBertSVVoiceprint)
    assert isinstance(default_voiceprint_backend("redimnet-b2"), ReDimNetVoiceprint)
    assert isinstance(default_voiceprint_backend("ecapa"), EcapaVoiceprint)
    with pytest.raises(ValueError, match="sconosciuto"):
        default_voiceprint_backend("x-vector")


def test_new_backends_refuse_changed_or_unknown_weights(tmp_path):
    from akoe.speaker.voiceprint_backends import ReDimNet2Voiceprint, W2VBertSVVoiceprint

    be = ReDimNet2Voiceprint()
    f = tmp_path / be.weights_file
    f.write_bytes(b"not the release asset")
    with pytest.raises(RuntimeError, match="diverso"):
        be._verify(f)
    unk = ReDimNet2Voiceprint(model_name="b6", dataset="vox9")
    f2 = tmp_path / unk.weights_file
    f2.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="non hanno un SHA-256 verificato"):
        unk._verify(f2)
    assert len(ReDimNet2Voiceprint(dataset="vox9", allow_unknown_weights=True)._verify(f2)) == 64
    w = W2VBertSVVoiceprint()
    f3 = tmp_path / "model_lmft_0.14.pth"
    f3.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="diverso"):
        w._verify(f3)
    with pytest.raises(RuntimeError, match="non hanno un SHA-256"):
        W2VBertSVVoiceprint("other.pth")._verify(f3)


def test_new_backends_reject_short_segments_before_loading():
    from akoe.speaker.voiceprint_backends import ReDimNet2Voiceprint, W2VBertSVVoiceprint

    for be in (ReDimNet2Voiceprint(), W2VBertSVVoiceprint()):
        with pytest.raises(ValueError, match="troppo corto"):
            be.embed(np.zeros(1000, dtype=np.float32), 16000)
        assert be.provenance is None


def test_w2vbert_sv_module_has_the_reference_parameter_names():
    torch = pytest.importorskip("torch")
    from transformers import Wav2Vec2BertConfig

    from akoe.speaker.voiceprint_backends import _build_w2vbert_sv

    cfg = Wav2Vec2BertConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64,
                             feature_projection_input_dim=160, position_embeddings_type="relative_key",
                             left_max_position_embeddings=8, right_max_position_embeddings=2,
                             conv_depthwise_kernel_size=3, mask_time_prob=0.0, mask_feature_prob=0.0)
    m = _build_w2vbert_sv(cfg, n_mfa_layers=3, adapter_dim=8, embd_dim=16).eval()
    keys = set(m.state_dict())
    # names of Audio2Vec_based_Adapter in ZXHY-82/w2v-BERT-2.0_SV
    for k in ("front.encoder.feature_projection.projection.weight",
              "front.encoder.encoder.layers.1.self_attn.distance_embedding.weight",
              "adapter_layers.2.0.weight", "adapter_layers.2.1.weight", "adapter_layers.2.3.bias",
              "pooling.attention.0.weight", "pooling.attention.2.running_var", "bottleneck.weight"):
        assert k in keys, k
    assert not any("masked_spec_embed" in k for k in keys)
    assert m.bottleneck.in_features == 8 * 3 * 2
    with torch.inference_mode():
        out = m(torch.randn(1, 20, 160))
    assert out.shape == (1, 16) and torch.isfinite(out).all()


@pytest.mark.slow
def test_new_backends_real_embeddings_separate_speakers():
    from akoe.speaker.voiceprint_backends import ReDimNet2Voiceprint, W2VBertSVVoiceprint
    from tests.real_audio import english_examples, italian_example

    it, sr, _ = italian_example()
    h = len(it) // 2
    for be in (ReDimNet2Voiceprint(), W2VBertSVVoiceprint()):
        a, b = be.embed(it[:h], sr), be.embed(it[h:], sr)
        others = [be.embed(x, s) for x, s in english_examples()]
        assert a.shape == (be.provenance.extra["embedding_dim"],) and a.dtype == np.float32
        assert all(_cos(a, b) > _cos(a, o) + 0.2 for o in others)
        assert be.provenance.weights and all(len(v) == 64 for v in be.provenance.weights.values())
        assert np.allclose(be.embed(it[:h], sr), a, atol=1e-5)
