"""Deterministic fake backends for tests (no models, no downloads)."""
from __future__ import annotations

import numpy as np

from forensicaudio.config import StageKind
from forensicaudio.models import Alternative, SegmentHypotheses, WordHypothesis


class FakeVoiceprint:
    """Embedding = normalized band energies: same signal -> same vector."""

    name = "fake-voiceprint"

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        spec = np.abs(np.fft.rfft(audio.astype(np.float64), n=2048))
        bands = np.array_split(spec, 16)
        v = np.array([b.mean() for b in bands]) + 1e-9
        return v / np.linalg.norm(v)


class FakeASR:
    name = "fake-asr"

    def __init__(self, text: str = "buongiorno a tutti") -> None:
        self.text = text

    def transcribe(self, audio, sr, *, segments=None, n_best=5):
        dur = len(audio) / sr
        segs = segments or [(0.0, dur)]
        out = []
        for start, end in segs:
            words = self.text.split()
            step = (end - start) / max(len(words), 1)
            out.append(
                SegmentHypotheses(
                    start=start,
                    end=end,
                    audio_provenance=StageKind.NEURAL,
                    hypotheses=[Alternative(text=self.text, score=0.0, origin=self.name)],
                    words=[
                        WordHypothesis(
                            text=w, start=start + i * step, end=start + (i + 1) * step,
                            confidence=0.9, origin=self.name,
                        )
                        for i, w in enumerate(words)
                    ],
                )
            )
        return out


class FakeScorer:
    """Prefers candidates sharing more words with a hidden truth."""

    name = "fake-scorer"

    def __init__(self, truth: str) -> None:
        self.truth = set(truth.split())

    def score(self, audio, sr, candidates):
        return [float(len(self.truth & set(c.split()))) for c in candidates]
