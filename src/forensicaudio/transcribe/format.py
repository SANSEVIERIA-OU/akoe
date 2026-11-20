"""Hypothesis transcript format (IMPLEMENTATION_PLAN M2, issue #17).

Builds and writes `transcript.json` as a `HypothesisTranscript`:
- every segment keeps the `audio_provenance` it declared: it describes the
  AUDIO the text was transcribed from (deterministic original/cleaned audio,
  or neural audio), not the text. Machine text is always an AI hypothesis
  through `HypothesisTranscript.is_ai_hypothesis`; the audio variant of each
  reading is in `Alternative.audio_variant`;
- every word with a confidence gets a confidence band;
- segments whose confidence is below a configurable threshold, or that
  have no readable text, are labelled UNINTELLIGIBLE (label added, text kept);
- the written file is returned as an `ArtifactRef` with its SHA-256.
"""
from __future__ import annotations

import logging
import os
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from forensicaudio.models import ArtifactRef, HypothesisTranscript, SegmentFlag, SegmentHypotheses
from forensicaudio.transcribe.combine import (
    COMBINED_PREFIX,
    DEFAULT_BANDS,
    BandThresholds,
    confidence_band,
)
from forensicaudio.utils import sha256_file

log = logging.getLogger(__name__)

TRANSCRIPT_KIND = "hypothesis_transcript"
TRANSCRIPT_FILENAME = "transcript.json"


@dataclass(frozen=True)
class FormatConfig:
    # UNINTELLIGIBLE below this segment confidence (to be tuned in M5).
    unintelligible_threshold: float = 0.4
    aggregate: Literal["mean", "min", "median"] = "mean"
    # A segment with no readable text at all is labelled UNINTELLIGIBLE.
    empty_is_unintelligible: bool = True
    bands: BandThresholds = DEFAULT_BANDS
    # False: only fill missing bands. True: recompute every band from confidence.
    recompute_bands: bool = False


DEFAULT_FORMAT = FormatConfig()


def segment_confidence(seg: SegmentHypotheses, aggregate: str = "mean") -> float | None:
    """Aggregate of word confidences; None when no word has one."""
    confs = [w.confidence for w in seg.words if w.confidence is not None and w.text.strip()]
    if not confs:
        return None
    if aggregate == "mean":
        return sum(confs) / len(confs)
    if aggregate == "min":
        return min(confs)
    if aggregate == "median":
        return float(statistics.median(confs))
    raise ValueError(f"unknown aggregate: {aggregate}")


def has_text(seg: SegmentHypotheses) -> bool:
    return any(a.text.strip() for a in seg.hypotheses) or any(w.text.strip() for w in seg.words)


def is_unintelligible(seg: SegmentHypotheses, cfg: FormatConfig | None = None) -> bool:
    """True when no reading reaches the threshold. Without any confidence the
    label is not applied (nothing is invented), unless the segment is empty."""
    cfg = cfg or DEFAULT_FORMAT
    if not has_text(seg):
        return cfg.empty_is_unintelligible
    conf = segment_confidence(seg, cfg.aggregate)
    return conf is not None and conf < cfg.unintelligible_threshold


def prepare_segment(seg: SegmentHypotheses, cfg: FormatConfig | None = None) -> SegmentHypotheses:
    """Confidence bands and UNINTELLIGIBLE label. Provenance, existing flags,
    texts and alternatives are never changed."""
    cfg = cfg or DEFAULT_FORMAT
    seg = seg.model_copy(deep=True)
    for w in seg.words:
        if w.confidence is not None and (cfg.recompute_bands or w.confidence_band is None):
            w.confidence_band = confidence_band(w.confidence, cfg.bands)
    if is_unintelligible(seg, cfg) and SegmentFlag.UNINTELLIGIBLE not in seg.flags:
        seg.flags.append(SegmentFlag.UNINTELLIGIBLE)
    return seg


def transcript_models(segments: Sequence[SegmentHypotheses]) -> list[str]:
    """Origins that contributed to the segments (combined labels expanded)."""
    out: dict[str, None] = {}
    for seg in segments:
        for alt in seg.hypotheses:
            if not alt.origin.startswith(COMBINED_PREFIX):
                out.setdefault(alt.origin)
        for w in seg.words:
            for alt in w.alternatives:
                if not alt.origin.startswith(COMBINED_PREFIX):
                    out.setdefault(alt.origin)
            if not w.origin.startswith(COMBINED_PREFIX):
                out.setdefault(w.origin)
    return list(out)


def build_transcript(
    case_id: str,
    segments: Sequence[SegmentHypotheses],
    *,
    evidence_id: str | None = None,
    models: Sequence[str] | None = None,
    cfg: FormatConfig | None = None,
) -> HypothesisTranscript:
    """Machine transcript ready to write: segments sorted by time and prepared
    with `prepare_segment`. `models` defaults to the origins found."""
    cfg = cfg or DEFAULT_FORMAT
    prepared = [prepare_segment(s, cfg) for s in sorted(segments, key=lambda s: (s.start, s.end))]
    return HypothesisTranscript(
        case_id=case_id,
        evidence_id=evidence_id,
        segments=prepared,
        models=list(models) if models is not None else transcript_models(prepared),
    )


def write_transcript(
    transcript: HypothesisTranscript,
    path: Path,
    *,
    root: Path | None = None,
    kind: str = TRANSCRIPT_KIND,
) -> ArtifactRef:
    """Write JSON atomically and return its ArtifactRef (path relative to
    `root` when given, e.g. the case root)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return ArtifactRef.from_file(path, kind=kind, root=root)


def read_transcript(path: Path, expected: ArtifactRef | None = None) -> HypothesisTranscript:
    """Read a transcript; when `expected` is given, the file's SHA-256 must match it."""
    if expected is not None and sha256_file(path) != expected.sha256:
        raise ValueError(f"transcript {path} does not match its recorded SHA-256")
    return HypothesisTranscript.model_validate_json(path.read_text(encoding="utf-8"))
