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
