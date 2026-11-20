# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""ASR ensemble of the default backend provider (engine.py): which
transcribers per tier, configuration, low-memory release. No model is
loaded: backends are lazy."""
import numpy as np
import pytest

from akoe import engine
from akoe.case import Case
from akoe.config import Channel, RecordingSource, Tier
from akoe.models import ProcessingLog, RecordingCondition
from akoe.transcribe.cohere_backend import CohereTranscribeBackend
from akoe.transcribe.nemotron_backend import NemotronTranscribeBackend
from akoe.transcribe.parakeet_backend import nemo_available
from akoe.transcribe.qwen3_backend import Qwen3ASRBackend
from akoe.transcribe.whisper_backend import WhisperHFBackend
from tests.engine_synth import make_evidence
from tests.fakes import FakeASR, FakeProvider


def test_default_ensemble_per_tier():
    assert engine.default_asr_ensemble(Tier.EDGE) == ("cohere", "qwen3", "parakeet", "whisper")
    assert engine.default_asr_ensemble(Tier.WORKSTATION) == ("cohere", "qwen3", "parakeet", "whisper")
    assert engine.default_asr_ensemble(Tier.CLOUD)[0] == "nemotron"


def test_default_backends_build_the_edge_ensemble_lazily(monkeypatch):
    monkeypatch.delenv(engine.ASR_ENSEMBLE_ENV, raising=False)
    b = engine.DefaultBackends()
    asr = b.asr(tier=Tier.EDGE)
    labels = list(asr)
    expected = ["cohere", "qwen3"] + (["parakeet"] if nemo_available() else []) + ["generic"]
    assert labels == expected  # Cohere first: skeleton of the word alignment
    assert isinstance(asr["cohere"], CohereTranscribeBackend) and asr["cohere"].low_memory
    assert isinstance(asr["qwen3"], Qwen3ASRBackend) and asr["qwen3"].dtype == "bfloat16"
    assert isinstance(asr["generic"], WhisperHFBackend) and asr["generic"].model_id == "openai/whisper-small"
    assert all(getattr(x, "_model", None) is None for x in asr.values())  # nothing loaded
    assert b.asr(tier=Tier.EDGE)["cohere"] is asr["cohere"]  # cached


def test_cloud_adds_nemotron_which_refuses_without_gpu():
    b = engine.DefaultBackends(asr_ensemble=["nemotron", "whisper"])
    asr = b.asr(tier=Tier.CLOUD)
    assert list(asr) == ["nemotron", "generic"]
    assert isinstance(asr["nemotron"], NemotronTranscribeBackend)
    assert asr["nemotron"].unavailable_reason()  # no NeMo Speech / no GPU on this machine


def test_ensemble_from_argument_or_environment(monkeypatch):
    assert list(engine.DefaultBackends(asr_ensemble="qwen3, whisper").asr(tier=Tier.EDGE)) == ["qwen3", "generic"]
    monkeypatch.setenv(engine.ASR_ENSEMBLE_ENV, "cohere")
    assert list(engine.DefaultBackends().asr(tier=Tier.EDGE)) == ["cohere"]
    with pytest.raises(engine.EngineError, match="sconosciuti"):
        engine.DefaultBackends(asr_ensemble="cohere,granite")


def test_adapter_always_adds_the_adapted_whisper():
    from types import SimpleNamespace

    b = engine.DefaultBackends(asr_ensemble="cohere")
    sentinel = object()
    b.adapted_whisper = lambda adapter: sentinel  # type: ignore[method-assign]
    adapter = SimpleNamespace(artifact=SimpleNamespace(base_model="openai/whisper-small"))
    assert b.asr(tier=Tier.EDGE, adapter=adapter)["adapted"] is sentinel  # type: ignore[arg-type]


class Unloadable(FakeASR):
    def __init__(self, text, name, log):
        super().__init__(text, name=name)
        self.log = log

    def unload(self):
        self.log.append(self.name)


class EnsembleProvider(FakeProvider):
    def __init__(self):
        super().__init__()
        self.released: list[str] = []
        self.backends = {
            "cohere": Unloadable("vado a casa adesso", "fake-cohere/generic", self.released),
            "qwen3": Unloadable("vado a casa adesso", "fake-qwen3/generic", self.released),
            "generic": Unloadable("vado a cassa adesso", "fake-whisper/generic", self.released),
        }

    def asr(self, *, tier, adapter=None, whisper_model=None):
        return dict(self.backends)

    def release(self, backend):
        backend.unload()


@pytest.fixture
def case_with_evidence(tmp_path):
    case = Case.create(tmp_path / "case", "caso-ensemble")
    ev = case.import_evidence(make_evidence(tmp_path / "reperti")["evidence"],
                              RecordingCondition(channel=Channel.CAR, source=RecordingSource.LOCAL))
    return case, ev


def test_transcribe_runs_the_ensemble_and_releases_each_model(case_with_evidence):
    case, ev = case_with_evidence
    prov = EnsembleProvider()
    res = engine.transcribe(case, ev.evidence_id, provider=prov)
    log = ProcessingLog.model_validate_json(res.log.resolve(case.root).read_text())
    assert [s.name for s in log.stages if s.name.startswith("asr:")] == ["asr:cohere", "asr:qwen3", "asr:generic"]
    assert prov.released == ["fake-cohere/generic", "fake-qwen3/generic", "fake-whisper/generic"]
    models = " ".join(res.summary["models"])
    assert all(n in models for n in ("fake-cohere", "fake-qwen3", "fake-whisper"))


def test_transcribe_can_select_part_of_the_ensemble(case_with_evidence):
    case, ev = case_with_evidence
    prov = EnsembleProvider()
    res = engine.transcribe(case, ev.evidence_id, provider=prov, asr_systems=["qwen3", "missing"])
    log = ProcessingLog.model_validate_json(res.log.resolve(case.root).read_text())
    assert [s.name for s in log.stages if s.name.startswith("asr:")] == ["asr:qwen3"]
    assert log.parameters["asr_systems"] == ["qwen3", "missing"]
    assert any("missing" in n for n in log.notes)


def test_explicit_backends_are_not_released(case_with_evidence):
    case, ev = case_with_evidence
    log: list[str] = []
    b = Unloadable("vado a casa", "fake/generic", log)
    engine.transcribe(case, ev.evidence_id, backends=[b], provider=EnsembleProvider())
    assert log == []


def test_default_backends_release_only_in_low_memory_mode():
    calls: list[str] = []
    b = Unloadable("x", "fake", calls)
    engine.DefaultBackends(low_memory=False).release(b)
    assert calls == []
    engine.DefaultBackends().release(b)
    assert calls == ["fake"]
    engine.DefaultBackends().release(np.zeros(1))  # objects without unload are ignored
