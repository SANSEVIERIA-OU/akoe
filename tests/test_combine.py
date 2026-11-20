import pytest

from forensicaudio.config import StageKind
from forensicaudio.models import Alternative, ConfidenceBand, SegmentFlag, SegmentHypotheses, WordHypothesis
from forensicaudio.transcribe.combine import (
    ABSENT,
    COMBINED_PREFIX,
    BandThresholds,
    CombineConfig,
    HypothesisSource,
    TimedWord,
    align_word_sequences,
    combine_hypotheses,
    confidence_band,
    consensus_text,
    join_nbest,
    source_texts,
)


def seg(text, start=0.0, end=2.0, origin="m", conf=0.9, timed=True, prov=StageKind.NEURAL, n_best=()):
    toks = text.split()
    step = (end - start) / max(len(toks), 1)
    words = (
        [WordHypothesis(text=w, start=start + i * step, end=start + (i + 1) * step, confidence=conf, origin=origin)
         for i, w in enumerate(toks)]
        if timed else []
    )
    hyps = [Alternative(text=text, score=-1.0, origin=origin)] + [
        Alternative(text=t, score=-2.0 - i, origin=origin) for i, t in enumerate(n_best)
    ]
    return SegmentHypotheses(start=start, end=end, audio_provenance=prov, hypotheses=hyps, words=words)


def src(text, origin, **kw):
    seg_kw = {k: kw.pop(k) for k in list(kw) if k in ("start", "end", "conf", "timed", "prov", "n_best")}
    return HypothesisSource(segments=[seg(text, origin=origin, **seg_kw)], origin=origin, **kw)


# -- bands ---------------------------------------------------------------------


def test_confidence_bands_configurable():
    assert confidence_band(0.9) == ConfidenceBand.HIGH
    assert confidence_band(0.7) == ConfidenceBand.MEDIUM
    assert confidence_band(0.2) == ConfidenceBand.LOW
    assert confidence_band(None) is None
    t = BandThresholds(high=0.95, medium=0.8)
    assert confidence_band(0.9, t) == ConfidenceBand.MEDIUM
    with pytest.raises(ValueError):
        BandThresholds(high=0.5, medium=0.8)


# -- alignment -----------------------------------------------------------------


def test_alignment_keeps_every_word_in_order():
    a = [TimedWord(w) for w in "vado a casa domani".split()]
    b = [TimedWord(w) for w in "vado a cassa domani sera".split()]
    slots = align_word_sequences([a, b])
    for k, seq in enumerate([a, b]):
        assert [s[k] for s in slots if s[k] is not None] == seq
    texts = [(s[0].text if s[0] else None, s[1].text if s[1] else None) for s in slots]
    assert texts == [("vado", "vado"), ("a", "a"), ("casa", "cassa"), ("domani", "domani"), (None, "sera")]


def test_alignment_is_time_aware():
    # Same word, far apart in time: never paired.
    a = [TimedWord("sì", 0.0, 0.3)]
    b = [TimedWord("sì", 5.0, 5.3)]
    assert len(align_word_sequences([a, b])) == 2
    # Without timestamps they would pair.
    assert len(align_word_sequences([[TimedWord("sì")], [TimedWord("sì")]])) == 1
    # A repeated word pairs with the occurrence at the right time.
    a = [TimedWord("no", 0.0, 0.4), TimedWord("no", 3.0, 3.4)]
    b = [TimedWord("no", 3.05, 3.4)]
    slots = align_word_sequences([a, b])
    assert slots[0][1] is None and slots[1][1] is b[0]


def test_alignment_prefers_similar_words():
    a = [TimedWord(w) for w in "ho visto la cassa".split()]
    b = [TimedWord(w) for w in "visto casa".split()]
    slots = align_word_sequences([a, b])
    pairs = {s[0].text: s[1].text for s in slots if s[0] and s[1]}
    assert pairs == {"visto": "visto", "cassa": "casa"}


# -- combination -----------------------------------------------------------------


def test_majority_vote_with_alternatives_and_origins():
    sources = [
        src("vado a casa", "whisper/generic"),
        src("vado a casa", "parakeet/generic"),
        src("vado a cassa", "whisper/adapted"),
    ]
    [out] = combine_hypotheses(sources)
    assert consensus_text(out) == "vado a casa"
    assert out.hypotheses[0].origin.startswith(COMBINED_PREFIX)
    casa = out.words[2]
    assert casa.text == "casa"
    assert casa.origin == "rover:whisper/generic+parakeet/generic"
    assert {(a.text, a.origin) for a in casa.alternatives} == {
        ("casa", "whisper/generic"), ("casa", "parakeet/generic"), ("cassa", "whisper/adapted")
    }
    assert casa.confidence == pytest.approx(2 / 3 * 0.9)
    assert casa.confidence_band == ConfidenceBand.MEDIUM
    assert out.words[0].confidence == pytest.approx(0.9)
    assert SegmentFlag.MODEL_DISAGREEMENT in out.flags
    # every source's own hypothesis is kept at segment level
    assert [a.text for a in out.hypotheses[1:]] == ["vado a casa", "vado a casa", "vado a cassa"]
    assert out.audio_provenance == StageKind.NEURAL


def test_no_silent_dropping_of_minority_words():
    sources = [
        src("vado a casa", "a"),
        src("vado a casa", "b"),
        src("vado eh a casa", "c", conf=0.5),
    ]
    [out] = combine_hypotheses(sources)
    assert [w.text for w in out.words] == ["vado", "eh", "a", "casa"]
    eh = out.words[1]
    # the word stays, with its origin and the declared absence of the others
    assert {(a.text, a.origin) for a in eh.alternatives} == {(ABSENT, "a"), (ABSENT, "b"), ("eh", "c")}
    assert eh.confidence_band == ConfidenceBand.LOW
    # but the consensus follows the majority
    assert consensus_text(out) == "vado a casa"
    all_alt_texts = {(a.text, a.origin) for w in out.words for a in w.alternatives}
    for s in sources:
        for w in s.segments[0].words:
            assert (w.text, s.origin) in all_alt_texts


def test_agreement_gives_no_disagreement_flag():
    [out] = combine_hypotheses([src("ci vediamo dopo", "a"), src("Ci vediamo dopo.", "b")])
    assert SegmentFlag.MODEL_DISAGREEMENT not in out.flags
    assert consensus_text(out) == "ci vediamo dopo"


def test_source_silent_in_a_span_is_declared():
    a = HypothesisSource([seg("buongiorno", 0, 1, origin="a"), seg("vado via", 3, 4, origin="a")], origin="a")
    b = HypothesisSource([seg("buongiorno", 0, 1, origin="b")], origin="b")
    out = combine_hypotheses([a, b])
    assert [(s.start, s.end) for s in out] == [(0, 1), (3, 4)]
    second = out[1]
    assert source_texts(second) == {"a": "vado via", "b": ABSENT}
    assert SegmentFlag.MODEL_DISAGREEMENT in second.flags
    assert SegmentFlag.MODEL_DISAGREEMENT not in out[0].flags


def test_single_source_is_preserved():
    s = HypothesisSource([seg("vado a casa", origin="whisper/generic", conf=0.5, n_best=("vado a cassa",))])
    [out] = combine_hypotheses([s])
    assert [w.text for w in out.words] == ["vado", "a", "casa"]
    assert all(w.origin == "whisper/generic" and w.confidence == 0.5 for w in out.words)
    assert all(w.confidence_band == ConfidenceBand.LOW for w in out.words)
    assert [a.text for a in out.hypotheses] == ["vado a casa", "vado a cassa"]
    assert out.flags == []


def test_untimed_source_gets_interpolated_times():
    [out] = combine_hypotheses([HypothesisSource([seg("uno due tre quattro", 2.0, 6.0, timed=False)], origin="x")])
    starts = [w.start for w in out.words]
    assert starts == sorted(starts) and starts[0] == pytest.approx(2.0) and out.words[-1].end == pytest.approx(6.0)
    assert all(w.confidence is None and w.confidence_band is None for w in out.words)


def test_untimed_mixed_with_timed_sources():
    a = HypothesisSource([seg("vado a casa", 0, 3, origin="a")], origin="a")
    b = HypothesisSource([seg("vado a casa", 0, 3, origin="b", timed=False)], origin="b")
    [out] = combine_hypotheses([a, b])
    assert [w.text for w in out.words] == ["vado", "a", "casa"]
    assert out.words[2].start == pytest.approx(2.0)
    assert SegmentFlag.MODEL_DISAGREEMENT not in out.flags
    # a source without confidence contributes its vote share
    assert out.words[0].confidence == pytest.approx(0.9)


def test_lexicon_dependent_flag():
    sources = [src("vado da Moretti", "adapted+lex", uses_lexicon=True), src("vado da Moreno", "generic")]
    [out] = combine_hypotheses(sources)
    assert SegmentFlag.LEXICON_DEPENDENT in out.flags
    [same] = combine_hypotheses([src("vado da Moretti", "x", uses_lexicon=True), src("vado da Moretti", "y")])
    assert SegmentFlag.LEXICON_DEPENDENT not in same.flags


def test_weights_decide_votes():
    sources = [src("vado a cassa", "weak"), src("vado a casa", "strong", weight=2.0)]
    [out] = combine_hypotheses(sources)
    assert out.words[2].text == "casa"


def test_audio_variants_are_distinct_sources():
    sources = [
        src("vado a casa", "whisper/generic"),
        src("vado a cassa", "whisper/generic", audio_variant="remixed"),
    ]
    [out] = combine_hypotheses(sources)
    variants = {(a.origin, a.audio_variant) for a in out.words[2].alternatives}
    assert variants == {("whisper/generic", "original"), ("whisper/generic", "remixed")}
    assert out.hypotheses[0].audio_variant == "mixed"
    assert set(source_texts(out)) == {"whisper/generic", "whisper/generic@remixed"}
    with pytest.raises(ValueError):
        combine_hypotheses([src("a", "same"), src("b", "same")])


def test_provenance_and_flags_are_carried():
    a = HypothesisSource([seg("vado", prov=StageKind.DETERMINISTIC, origin="a")], origin="a")
    flagged = seg("vado", origin="b")
    flagged.flags.append(SegmentFlag.POSSIBLE_OTHER_SPEAKER)
    b = HypothesisSource([flagged], origin="b")
    [out] = combine_hypotheses([a, b])
    assert out.audio_provenance == StageKind.NEURAL
    assert SegmentFlag.POSSIBLE_OTHER_SPEAKER in out.flags


def test_disagreement_threshold_configurable():
    sources = [src("uno due tre quattro", "a"), src("uno due tre cinque", "b")]
    [out] = combine_hypotheses(sources, CombineConfig(disagreement_min_fraction=0.5))
    assert SegmentFlag.MODEL_DISAGREEMENT not in out.flags


def test_join_nbest_over_several_segments():
    s1 = seg("vado", 0, 1, n_best=("bado",))
    s2 = seg("a casa", 1, 2)
    alts = join_nbest([s2, s1], audio_variant="remixed")
    assert [a.text for a in alts] == ["vado a casa", "bado a casa"]
    assert all(a.audio_variant == "remixed" for a in alts)
    assert combine_hypotheses([]) == []
