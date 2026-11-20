import numpy as np

from forensicaudio.config import StageKind
from forensicaudio.interfaces import ASRBackend, CandidateScorer, VoiceprintBackend
from tests.fakes import FakeASR, FakeScorer, FakeVoiceprint


def test_fakes_satisfy_protocols():
    assert isinstance(FakeVoiceprint(), VoiceprintBackend)
    assert isinstance(FakeASR(), ASRBackend)
    assert isinstance(FakeScorer("a"), CandidateScorer)


def test_fake_asr_output_carries_provenance_and_origin():
    audio = np.zeros(16000, dtype=np.float32)
    segs = FakeASR().transcribe(audio, 16000)
    assert segs and all(s.audio_provenance == StageKind.NEURAL for s in segs)
    assert all(a.origin for s in segs for a in s.hypotheses)
