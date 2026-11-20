import os
from types import SimpleNamespace

import numpy as np
import pytest

from forensicaudio.config import StageKind
from forensicaudio.interfaces import ASRBackend
from forensicaudio.neural import BackendUnavailable, ModelProvenance
from forensicaudio.transcribe.parakeet_backend import (
    KNOWN_REVISIONS,
    ParakeetTDTBackend,
    nemo_available,
)

SR = 16000


def test_protocol_and_defaults():
    be = ParakeetTDTBackend()
    assert isinstance(be, ASRBackend)
    assert be.name == "parakeet-tdt-0.6b-v3/generic"
    assert be.nemo_file == "parakeet-tdt-0.6b-v3.nemo"
    assert be.revision == KNOWN_REVISIONS["nvidia/parakeet-tdt-0.6b-v3"]
    assert be.confidence.entropy_type == "tsallis" and be.confidence.alpha == 0.33


@pytest.mark.skipif(nemo_available(), reason="NeMo is installed here")
def test_without_nemo_the_stage_is_skipped_with_instructions():
    be = ParakeetTDTBackend()
    with pytest.raises(BackendUnavailable, match="NeMo") as exc:
        be.transcribe(np.zeros(SR, dtype=np.float32), SR)
    assert isinstance(exc.value, NotImplementedError)  # pipeline logs it as "skipped"
    assert "nemo_toolkit[asr]" in str(exc.value)


class FakeNeMoModel:
    """Mimics `ASRModel.transcribe(..., return_hypotheses=True, timestamps=True)`."""

    def __init__(self, conf):
        self.conf = conf

    def transcribe(self, audio, **kw):
        stamps = {"word": [{"word": "buongiorno", "start": 0.1, "end": 0.6},
                           {"word": "a", "start": 0.7, "end": 0.8},
                           {"word": "tutti", "start": 0.8, "end": 9.0}]}
        return [SimpleNamespace(text="buongiorno a tutti", score=-1.5, timestamp=stamps, word_confidence=self.conf)]


def _with_fake(conf):
    be = ParakeetTDTBackend()
    be._model = FakeNeMoModel(conf)
    be._provenance = ModelProvenance(backend=be.name, model_id=be.model_id)
    return be


def test_hypotheses_words_and_confidence_from_nemo_output():
    be = _with_fake([0.9, 0.5, 0.2])
    segs = be.transcribe(np.zeros(SR * 2, dtype=np.float32), SR, segments=[(1.0, 2.0)], n_best=1)
    s = segs[0]
    assert s.audio_provenance == StageKind.DETERMINISTIC  # default input: original audio
    assert [(h.text, h.score, h.origin) for h in s.hypotheses] == [("buongiorno a tutti", -1.5, be.name)]
    assert [w.text for w in s.words] == ["buongiorno", "a", "tutti"]
    assert [w.confidence for w in s.words] == [0.9, 0.5, 0.2]
    assert s.words[0].start == pytest.approx(1.1) and s.words[-1].end == pytest.approx(2.0)  # clipped


def test_unpairable_confidence_is_dropped_not_misassigned():
    be = _with_fake([0.9])
    s = be.transcribe(np.zeros(SR, dtype=np.float32), SR, n_best=1)[0]
    assert all(w.confidence is None for w in s.words)


@pytest.mark.slow
@pytest.mark.skipif(not nemo_available(), reason="NeMo not installed (see parakeet_backend docstring)")
def test_real_nemo_model():
    from tests.real_audio import italian_example

    model_id = os.environ.get("WHISPER_TEST_NEMO_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
    audio, sr, _ = italian_example()
    be = ParakeetTDTBackend(model_id)
    segs = be.transcribe(audio, sr, n_best=3)
    assert "tutti" in segs[0].hypotheses[0].text.lower()
    assert segs[0].words and all(w.confidence is not None for w in segs[0].words)
    prov = be.provenance
    assert prov.weights and prov.extra["confidence"]["alpha"] == 0.33
