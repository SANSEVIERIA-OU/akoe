"""Backend interfaces. Concrete backends (SpeechBrain, NeMo, Whisper...) live in
their own modules and are optional; the engine depends only on these protocols.

Audio is always mono float32 in [-1, 1] at the given sample rate.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from forensicaudio.models import SegmentHypotheses


@runtime_checkable
class VoiceprintBackend(Protocol):
    """Speaker-verification embeddings (discriminative, not a synthesis voice)."""

    name: str

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Return a 1-D embedding for one speech segment."""
        ...


@runtime_checkable
class ASRBackend(Protocol):
    name: str

    def transcribe(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        segments: list[tuple[float, float]] | None = None,
        n_best: int = 5,
    ) -> list[SegmentHypotheses]:
        """Return hypotheses per segment. Every SegmentHypotheses must carry
        audio_provenance and every Alternative its origin."""
        ...


@runtime_checkable
class CandidateScorer(Protocol):
    """Scores operator-typed candidate phrases against the audio ("did he say X or Y?")."""

    name: str

    def score(self, audio: np.ndarray, sr: int, candidates: list[str]) -> list[float]:
        """Return one score per candidate, higher = more compatible with the audio.
        Scores are comparable only within one call."""
        ...
