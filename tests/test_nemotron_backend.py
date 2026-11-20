# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Nemotron 3.5 Transcribe backend (#56): CLOUD only, never run here.
Checks the refusal messages and the output mapping with a fake SALM model."""
from types import SimpleNamespace

import numpy as np
import pytest

from akoe.config import StageKind
from akoe.interfaces import ASRBackend
from akoe.neural import BackendUnavailable, ModelProvenance
from akoe.transcribe import nemotron_backend as nb
from akoe.transcribe.nemotron_backend import KNOWN_LICENSES, KNOWN_REVISIONS, NemotronTranscribeBackend

SR = 16000


def test_defaults_and_protocol():
    be = NemotronTranscribeBackend()
    assert isinstance(be, ASRBackend)
    assert be.name == "NVIDIA-Nemotron-3.5-Transcribe-30B-A4B/generic"
    assert be.revision == KNOWN_REVISIONS[nb.DEFAULT_MODEL] and KNOWN_LICENSES[nb.DEFAULT_MODEL] == "openmdw-1.1"
    assert "Italian" in be.prompt and be.device == "cuda" and be.min_gpu_memory_gb >= 78


def test_without_nemo_speech_the_stage_is_skipped_with_instructions(monkeypatch):
    monkeypatch.setattr(nb, "nemo_speech_available", lambda: False)
    be = NemotronTranscribeBackend()
    with pytest.raises(BackendUnavailable, match="NeMo Speech") as exc:
        be.transcribe(np.zeros(SR, dtype=np.float32), SR)
    assert isinstance(exc.value, NotImplementedError)
    assert "CLOUD" in str(exc.value) and "backend_neurali.md" in str(exc.value)


def test_gpu_checks(monkeypatch):
    monkeypatch.setattr(nb, "nemo_speech_available", lambda: True)
    monkeypatch.setattr(nb, "gpu_memory_gb", lambda: [])
    assert "CUDA" in NemotronTranscribeBackend().unavailable_reason()
    assert "cpu" in NemotronTranscribeBackend(device="cpu").unavailable_reason()
    monkeypatch.setattr(nb, "gpu_memory_gb", lambda: [48.0])
    reason = NemotronTranscribeBackend().unavailable_reason()
    assert "insufficiente" in reason and "48 GB" in reason
    assert "nessuna GPU" in NemotronTranscribeBackend(device="cuda:1").unavailable_reason()
    monkeypatch.setattr(nb, "gpu_memory_gb", lambda: [80.0])
    assert NemotronTranscribeBackend().unavailable_reason() is None


class FakeSALM:
    audio_locator_tag = "<|audio|>"
    text_bos_id, text_eos_id, text_pad_id = 1, 2, 0

    def __init__(self):
        self.tokenizer = SimpleNamespace(ids_to_text=lambda ids: " ".join({5: "vado", 6: "a", 7: "casa",
                                                                           8: "cassa"}[i] for i in ids))
        self.calls = []

    def generate(self, prompts, audios, audio_lens, generation_config, enable_thinking):
        self.calls.append((prompts, generation_config))
        rows = [[5, 6, 7, 2, 0], [5, 6, 8, 2, 0], [5, 6, 7, 2, 0]]
        return rows[: generation_config.num_return_sequences]


class EvenTimer:
    name = "even"

    def word_times(self, audio, sr, words):
        step = len(audio) / sr / len(words)
        return [(i * step, (i + 1) * step) for i in range(len(words))]


def test_output_mapping_with_a_fake_model():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    be = NemotronTranscribeBackend(device="cpu", aligner=EvenTimer())
    be._model = FakeSALM()
    be._provenance = ModelProvenance(backend=be.name, model_id=be.model_id)
    s = be.transcribe(np.zeros(SR * 2, dtype=np.float32), SR, segments=[(0.0, 1.5)], n_best=3)[0]
    assert s.audio_provenance == StageKind.DETERMINISTIC
    assert [h.text for h in s.hypotheses] == ["vado a casa", "vado a cassa"]  # duplicates removed
    assert all(h.score is None and h.origin == be.name for h in s.hypotheses)
    assert [w.text for w in s.words] == ["vado", "a", "casa"]
    assert all(w.confidence is None for w in s.words)
    assert s.words[-1].end == pytest.approx(1.5)
    prompts, gen = be._model.calls[0]
    assert prompts[0][0]["content"].endswith("<|audio|>") and gen.num_beams >= 3
