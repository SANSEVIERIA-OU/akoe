# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
import json
import logging

import numpy as np
import pytest

from akoe.config import StageKind
from akoe.models import (
    Alternative,
    ConfidenceBand,
    HypothesisTranscript,
    SegmentFlag,
    SegmentHypotheses,
    WordHypothesis,
)
from akoe.transcribe.combine import HypothesisSource, combine_hypotheses
from akoe.transcribe.format import (
    TRANSCRIPT_KIND,
    FormatConfig,
    build_transcript,
    is_unintelligible,
    read_transcript,
    segment_confidence,
    transcript_models,
    write_transcript,
)
from akoe.transcribe.guard import guard_segments
from akoe.utils import sha256_file
from tests.fakes import FakeASR


def seg(confs, start=0.0, prov=StageKind.NEURAL, origin="whisper/generic"):
    toks = [f"w{i}" for i in range(len(confs))]
    return SegmentHypotheses(
        start=start, end=start + 1, audio_provenance=prov,
        hypotheses=[Alternative(text=" ".join(toks), origin=origin)] if toks else [],
        words=[WordHypothesis(text=t, start=start, end=start + 1, confidence=c, origin=origin)
               for t, c in zip(toks, confs)],
    )


def test_segment_confidence_aggregates():
    s = seg([0.2, 0.6, 1.0])
    assert segment_confidence(s) == pytest.approx(0.6)
    assert segment_confidence(s, "min") == pytest.approx(0.2)
    assert segment_confidence(s, "median") == pytest.approx(0.6)
    assert segment_confidence(seg([None, None])) is None
    with pytest.raises(ValueError):
        segment_confidence(s, "max")


def test_unintelligible_threshold_is_configurable():
    low, high = seg([0.2, 0.3]), seg([0.8, 0.9])
    assert is_unintelligible(low) and not is_unintelligible(high)
    assert is_unintelligible(high, FormatConfig(unintelligible_threshold=0.95))
    assert is_unintelligible(seg([0.9, 0.1]), FormatConfig(aggregate="min"))
    # no confidence at all: nothing is invented
    assert not is_unintelligible(seg([None]))
    # empty segment: explicit label (configurable)
    empty = SegmentHypotheses(start=0, end=1, audio_provenance=StageKind.NEURAL)
    assert is_unintelligible(empty)
    assert not is_unintelligible(empty, FormatConfig(empty_is_unintelligible=False))


def test_build_transcript_labels_bands_and_provenance():
    segments = [seg([0.9, 0.95], start=5.0), seg([0.2, 0.3], start=0.0, prov=StageKind.DETERMINISTIC)]
    t = build_transcript("case-1", segments, evidence_id="ev-1")
    assert t.is_ai_hypothesis is True and t.evidence_id == "ev-1"
    assert [s.start for s in t.segments] == [0.0, 5.0]  # sorted by time
    # audio_provenance describes the audio, so it is kept as declared
    assert [s.audio_provenance for s in t.segments] == [StageKind.DETERMINISTIC, StageKind.NEURAL]
    assert t.segments[0].flags == [SegmentFlag.UNINTELLIGIBLE]
    assert t.segments[1].flags == []
    assert [w.confidence_band for w in t.segments[1].words] == [ConfidenceBand.HIGH, ConfidenceBand.HIGH]
    assert [w.confidence_band for w in t.segments[0].words] == [ConfidenceBand.LOW, ConfidenceBand.LOW]
    # text is kept even when unintelligible
    assert t.segments[0].hypotheses[0].text == "w0 w1"
    # inputs untouched
    assert segments[1].audio_provenance == StageKind.DETERMINISTIC and segments[1].flags == []


def test_existing_bands_kept_unless_recomputed():
    s = seg([0.9])
    s.words[0].confidence_band = ConfidenceBand.LOW
    assert build_transcript("c", [s]).segments[0].words[0].confidence_band == ConfidenceBand.LOW
    t = build_transcript("c", [s], cfg=FormatConfig(recompute_bands=True))
    assert t.segments[0].words[0].confidence_band == ConfidenceBand.HIGH


def test_unintelligible_flag_not_duplicated():
    s = seg([0.1])
    s.flags.append(SegmentFlag.UNINTELLIGIBLE)
    assert build_transcript("c", [s]).segments[0].flags == [SegmentFlag.UNINTELLIGIBLE]


def test_write_and_read_with_sha256(tmp_path):
    t = build_transcript("case-1", [seg([0.9])])
    path = tmp_path / "outputs" / "transcript.json"
    ref = write_transcript(t, path, root=tmp_path)
    assert ref.path == "outputs/transcript.json" and ref.kind == TRANSCRIPT_KIND
    assert ref.sha256 == sha256_file(path) and ref.verify(tmp_path)
    data = json.loads(path.read_text())
    assert data["is_ai_hypothesis"] is True
    assert all(s["audio_provenance"] == "neural" for s in data["segments"])
    assert read_transcript(path, ref) == t
    path.write_text(path.read_text().replace("w0", "XX"))
    with pytest.raises(ValueError):
        read_transcript(path, ref)


def test_write_keeps_audio_provenance(tmp_path):
    t = HypothesisTranscript(case_id="c", segments=[seg([0.9], prov=StageKind.DETERMINISTIC)])
    ref = write_transcript(t, tmp_path / "t.json")
    back = read_transcript(tmp_path / "t.json", expected=ref)
    assert back.segments[0].audio_provenance == StageKind.DETERMINISTIC and back.is_ai_hypothesis


def test_end_to_end_two_models(tmp_path):
    audio = np.zeros(32000, dtype=np.float32)
    spans = [(0.0, 1.0), (1.0, 2.0)]
    a = FakeASR("vado a casa").transcribe(audio, 16000, segments=spans)
    b = FakeASR("vado a cassa").transcribe(audio, 16000, segments=spans)
    combined = combine_hypotheses([
        HypothesisSource(a, origin="whisper-large-v3/generic"),
        HypothesisSource(b, origin="parakeet-tdt-0.6b-v3/generic"),
    ])
    guarded = guard_segments(combined, vad_regions=[(0.0, 1.0)]).segments
    t = build_transcript("case-1", guarded)
    assert t.models == ["whisper-large-v3/generic", "parakeet-tdt-0.6b-v3/generic"]
    assert transcript_models(t.segments) == t.models
    assert len(t.segments) == 2
    assert SegmentFlag.POSSIBLE_HALLUCINATION in t.segments[1].flags  # second span has no speech
    for s in t.segments:
        assert s.audio_provenance == StageKind.NEURAL
        assert SegmentFlag.MODEL_DISAGREEMENT in s.flags
        for w in s.words:
            assert w.origin and w.alternatives and all(x.origin for x in w.alternatives)
    ref = write_transcript(t, tmp_path / "transcript.json")
    assert read_transcript(tmp_path / "transcript.json", ref) == t
