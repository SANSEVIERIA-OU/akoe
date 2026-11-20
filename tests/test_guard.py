from forensicaudio.config import StageKind
from forensicaudio.models import Alternative, SegmentFlag, SegmentHypotheses, WordHypothesis
from forensicaudio.transcribe.combine import HypothesisSource, combine_hypotheses
from forensicaudio.transcribe.guard import (
    GuardConfig,
    GuardReason,
    compression_ratio,
    disagreement_score,
    guard_segments,
    match_known_phrases,
    repetition_loop,
    speech_fraction,
)

HALL = SegmentFlag.POSSIBLE_HALLUCINATION


def seg(text, start=0.0, end=2.0, origin="whisper/generic", flags=()):
    toks = text.split()
    step = (end - start) / max(len(toks), 1)
    return SegmentHypotheses(
        start=start, end=end, audio_provenance=StageKind.NEURAL,
        hypotheses=[Alternative(text=text, origin=origin)],
        words=[WordHypothesis(text=w, start=start + i * step, end=start + (i + 1) * step, confidence=0.8,
                              origin=origin) for i, w in enumerate(toks)],
        flags=list(flags),
    )


def reasons(result, i=0):
    return {f.reason for f in result.for_segment(i)}


# -- speech activity -----------------------------------------------------------


def test_speech_fraction():
    assert speech_fraction(0, 2, [(0, 1)]) == 0.5
    assert speech_fraction(0, 2, [(0, 1.5), (1, 2)]) == 1.0  # overlapping regions merged
    assert speech_fraction(0, 2, []) == 0.0
    assert speech_fraction(1, 1, [(0, 2)]) == 1.0


def test_text_without_speech_is_flagged():
    r = guard_segments([seg("vado a casa", 0, 2)], vad_regions=[(5.0, 6.0)])
    assert HALL in r.segments[0].flags
    assert GuardReason.NO_SPEECH in reasons(r)
    # VAD found no speech at all
    assert HALL in guard_segments([seg("vado a casa")], vad_regions=[]).segments[0].flags


def test_text_on_speech_is_not_flagged_and_no_vad_skips_check():
    assert guard_segments([seg("vado a casa", 0, 2)], vad_regions=[(0.0, 2.0)]).segments[0].flags == []
    assert guard_segments([seg("vado a casa", 0, 2)], vad_regions=None).segments[0].flags == []


def test_words_outside_speech():
    # 30% of the span is speech (above the segment threshold) but most words fall outside it
    r = guard_segments([seg("uno due tre quattro cinque", 0, 10)], vad_regions=[(0.0, 3.0)])
    assert GuardReason.WORDS_OUTSIDE_SPEECH in reasons(r)


def test_empty_segment_on_silence_is_not_a_hallucination():
    empty = SegmentHypotheses(start=0, end=1, audio_provenance=StageKind.NEURAL)
    assert guard_segments([empty], vad_regions=[]).segments[0].flags == []


# -- bag of hallucinations -----------------------------------------------------


def test_known_phrases_italian_and_english():
    assert match_known_phrases("Sottotitoli creati dalla comunità Amara.org") == [
        "Sottotitoli creati dalla comunità Amara.org"
    ]
    assert match_known_phrases("sottotitoli e revisione a cura di QTSS") == ["Sottotitoli e revisione a cura di QTSS"]
    assert match_known_phrases("Grazie per la visione!") == ["Grazie per la visione"]
    assert match_known_phrases("THANK YOU FOR WATCHING.") == ["Thank you for watching"]
    assert match_known_phrases("Grazie.") == ["Grazie"]
    # exact-list entries are ordinary words inside a sentence
    assert match_known_phrases("grazie mille per il passaggio") == []
    assert match_known_phrases("vado a casa") == []


def test_known_phrase_flags_segment_with_origin():
    r = guard_segments([seg("Grazie per la visione", origin="whisper-large-v3/generic")], vad_regions=[(0, 2)])
    assert HALL in r.segments[0].flags
    [f] = r.for_segment(0)
    assert f.reason == GuardReason.KNOWN_PHRASE and f.origin == "whisper-large-v3/generic"
    assert "allucinazioni" in f.message and "whisper-large-v3/generic" in f.message


def test_custom_phrase_list():
    cfg = GuardConfig(contains_phrases=("frase sospetta",), exact_phrases=())
    assert match_known_phrases("una frase sospetta qui", cfg) == ["frase sospetta"]
    assert match_known_phrases("Grazie per la visione", cfg) == []


# -- repetition loops ----------------------------------------------------------


def test_repetition_loops():
    assert repetition_loop("e poi e poi e poi e poi".split()) == ("e poi", 4)
    assert repetition_loop("no no no no no".split()) == ("no", 5)
    assert repetition_loop("no no no".split()) is None
    assert repetition_loop("vado a casa e poi torno".split()) is None
    r = guard_segments([seg("allora e poi e poi e poi andiamo")])
    assert GuardReason.REPETITION_LOOP in reasons(r)


def test_compression_ratio_detects_long_repetitive_text():
    text = "ci vediamo domani mattina alle otto, " * 6
    assert compression_ratio(text) > 2.4
    cfg = GuardConfig(min_repeats_ngram=99, min_repeats_unigram=99)  # isolate the compression signal
    r = guard_segments([seg(text)], cfg=cfg)
    assert GuardReason.HIGH_COMPRESSION in reasons(r)
    assert compression_ratio("vado a casa") < 2.4


# -- cross-model disagreement --------------------------------------------------


def test_disagreement_score():
    assert disagreement_score(["vado a casa", "vado a casa"]) == 0.0
    assert disagreement_score(["vado a casa", ""]) == 1.0
    assert disagreement_score(["vado a casa"]) is None


def test_strong_disagreement_flags_hallucination():
    sources = [
        HypothesisSource([seg("Grazie a tutti per aver seguito il programma", origin="w")], origin="w"),
        HypothesisSource([seg("allora domani vado", origin="p")], origin="p"),
    ]
    combined = combine_hypotheses(sources)
    r = guard_segments(combined, vad_regions=[(0, 2)])
    assert GuardReason.STRONG_DISAGREEMENT in reasons(r)
    assert HALL in r.segments[0].flags and SegmentFlag.MODEL_DISAGREEMENT in r.segments[0].flags


def test_mild_disagreement_is_not_a_hallucination():
    sources = [
        HypothesisSource([seg("vado a casa domani", origin="w")], origin="w"),
        HypothesisSource([seg("vado a cassa domani", origin="p")], origin="p"),
    ]
    r = guard_segments(combine_hypotheses(sources), vad_regions=[(0, 2)])
    assert HALL not in r.segments[0].flags


# -- never delete --------------------------------------------------------------


def test_guard_only_adds_flags():
    segments = [
        seg("Sottotitoli creati dalla comunità Amara.org", 0, 2, flags=[SegmentFlag.MODEL_DISAGREEMENT]),
        seg("vado a casa", 2, 4),
        seg("e poi e poi e poi e poi", 4, 6),
    ]
    before = [s.model_copy(deep=True) for s in segments]
    r = guard_segments(segments, vad_regions=[(2.0, 4.0)])
    assert len(r.segments) == len(segments)
    for out, orig in zip(r.segments, before):
        assert out.hypotheses == orig.hypotheses and out.words == orig.words
        assert (out.start, out.end, out.audio_provenance) == (orig.start, orig.end, orig.audio_provenance)
        assert set(orig.flags) <= set(out.flags)
        assert len(out.flags) == len(set(out.flags))
    assert segments == before  # input not mutated
    assert r.segments[0].flags == [SegmentFlag.MODEL_DISAGREEMENT, HALL]
    assert r.segments[1].flags == []
    assert HALL in r.segments[2].flags
