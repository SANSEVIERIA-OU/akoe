# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Combination of hypotheses from several sources (IMPLEMENTATION_PLAN M2, issue #14).

A *source* is one run of one transcriber on one audio variant: Parakeet or
Whisper, generic or adapted model, with or without lexicon, original or
remixed audio. Sources are aligned word by word, ROVER style (Fiscus 1997):
the first source is the skeleton, each further source is aligned to the
growing word network by dynamic programming. When both sides have word
timestamps the alignment is time-aware (words far apart in time are not
paired).

Nothing is dropped:
- every aligned slot becomes a `WordHypothesis`, even when most sources
  heard no word there;
- each word lists, as `alternatives`, the reading of every source (one entry
  per source, including those agreeing with the winner, so agreement is
  countable) and an empty-text entry for each source that heard nothing;
- segment `hypotheses` start with the voted consensus, followed by every
  source's own N-best list, plus an empty-text entry for any source that
  produced nothing in that time span.

Voting: each reading gets (share of source weight) x (mean source
confidence). A word is part of the consensus text only when it outvotes
"no word here"; it stays in `words` either way, with its (low) confidence.
"""
from __future__ import annotations

import difflib
from collections.abc import Sequence
from dataclasses import dataclass

from akoe.assess.metrics import DEFAULT_NORMALIZATION, NormalizationConfig, normalize_text
from akoe.config import StageKind
from akoe.models import (
    Alternative,
    ConfidenceBand,
    SegmentFlag,
    SegmentHypotheses,
    WordHypothesis,
)

COMBINED_PREFIX = "rover:"
ORIGIN_JOIN = "+"
ABSENT = ""  # text of an Alternative declaring that a source heard no word
MIXED_VARIANT = "mixed"


# --------------------------------------------------------------------------
# Confidence bands
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BandThresholds:
    """HIGH if confidence >= high, MEDIUM if >= medium, else LOW. To be tuned on
    the self-assessment calibration (M5); confidence is not a raw probability."""

    high: float = 0.85
    medium: float = 0.6

    def __post_init__(self) -> None:
        if not 0.0 <= self.medium <= self.high <= 1.0:
            raise ValueError("band thresholds must satisfy 0 <= medium <= high <= 1")


DEFAULT_BANDS = BandThresholds()


def confidence_band(
    confidence: float | None, thresholds: BandThresholds | None = None
) -> ConfidenceBand | None:
    if confidence is None:
        return None
    t = thresholds or DEFAULT_BANDS
    if confidence >= t.high:
        return ConfidenceBand.HIGH
    if confidence >= t.medium:
        return ConfidenceBand.MEDIUM
    return ConfidenceBand.LOW


# --------------------------------------------------------------------------
# Sources and configuration
# --------------------------------------------------------------------------


@dataclass
class HypothesisSource:
    """Output of one transcriber run.

    `origin` labels the source (e.g. "whisper-large-v3/generic"); when given it
    becomes the origin of every alternative of this source, since the caller
    knows which model and configuration it ran. If None, the origin is taken
    from the backend's first alternative or word. `audio_variant`, when given,
    likewise overrides the variant of every alternative (the caller knows what
    audio it fed). Two sources may share an origin only if their audio
    variants differ (same model on original and remixed audio)."""

    segments: list[SegmentHypotheses]
    origin: str | None = None
    audio_variant: str | None = None
    weight: float = 1.0
    uses_lexicon: bool = False

    def label(self, index: int = 0) -> str:
        if self.origin:
            return self.origin
        for seg in self.segments:
            for alt in seg.hypotheses:
                if alt.origin:
                    return alt.origin
            for w in seg.words:
                if w.origin:
                    return w.origin
        return f"source-{index}"

    def variant(self) -> str:
        if self.audio_variant:
            return self.audio_variant
        for seg in self.segments:
            for alt in seg.hypotheses:
                return alt.audio_variant
        return "original"


@dataclass(frozen=True)
class CombineConfig:
    bands: BandThresholds = DEFAULT_BANDS
    # Time-aware alignment: extra cost time_weight * min(1, gap / tolerance).
    # With the defaults, words more than ~0.6 s apart are never paired.
    time_tolerance_s: float = 1.0
    time_weight: float = 2.5
    # Substitution cost is 1 - similarity_weight * char_similarity, so that
    # "casa" pairs with "cassa" rather than with an unrelated word.
    similarity_weight: float = 0.5
    # Used only to rank readings when a source gives no confidence.
    missing_confidence: float = 0.5
    # Confidence attributed to "this source heard no word here" when ranking.
    absence_confidence: float = 0.7
    # Flag MODEL_DISAGREEMENT when the fraction of disagreeing slots exceeds this.
    disagreement_min_fraction: float = 0.0
    # Segments of different sources are grouped when they overlap by more than this.
    cluster_min_overlap_s: float = 0.0
    normalization: NormalizationConfig = DEFAULT_NORMALIZATION


DEFAULT_COMBINE = CombineConfig()


@dataclass
class TimedWord:
    text: str
    start: float | None = None
    end: float | None = None
    confidence: float | None = None
    alternatives: tuple[Alternative, ...] = ()

    @property
    def timed(self) -> bool:
        return self.start is not None and self.end is not None


# --------------------------------------------------------------------------
# ROVER alignment
# --------------------------------------------------------------------------


def _key(w: TimedWord | None, cfg: CombineConfig) -> str:
    """Comparison key of a reading; ABSENT for "no word"."""
    if w is None:
        return ABSENT
    return normalize_text(w.text, cfg.normalization) or w.text.strip() or "\x00"


def _pair_cost(a: TimedWord, b: TimedWord, cfg: CombineConfig) -> float:
    ka, kb = _key(a, cfg), _key(b, cfg)
    if ka == kb:
        cost = 0.0
    else:
        sim = difflib.SequenceMatcher(None, ka, kb).ratio()
        cost = 1.0 - cfg.similarity_weight * sim
    if a.timed and b.timed and cfg.time_weight > 0:
        gap = max(0.0, max(a.start, b.start) - min(a.end, b.end))  # type: ignore[type-var,arg-type]
        tol = cfg.time_tolerance_s if cfg.time_tolerance_s > 0 else 1e-9
        cost += cfg.time_weight * min(1.0, gap / tol)
    return cost


def _slot_cost(w: TimedWord, slot: list[TimedWord | None], cfg: CombineConfig) -> float:
    return min(_pair_cost(w, e, cfg) for e in slot if e is not None)


def align_word_sequences(
    sequences: Sequence[Sequence[TimedWord]], cfg: CombineConfig | None = None
) -> list[list[TimedWord | None]]:
    """Align word sequences into slots. Returns one row per slot with one entry
    per input sequence (None where that sequence has no word in the slot).
    Every input word appears in exactly one slot, in order."""
    cfg = cfg or DEFAULT_COMBINE
    slots: list[list[TimedWord | None]] = []
    for s, seq in enumerate(sequences):
        if s == 0:
            slots = [[w] for w in seq]
            continue
        m, n = len(slots), len(seq)
        inf = float("inf")
        dp = [[inf] * (n + 1) for _ in range(m + 1)]
        bt = [[""] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 0.0
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 and j == 0:
                    continue
                best, move = inf, ""
                if i > 0 and j > 0:
                    c = dp[i - 1][j - 1] + _slot_cost(seq[j - 1], slots[i - 1], cfg)
                    if c < best:
                        best, move = c, "m"
                if i > 0:
                    c = dp[i - 1][j] + 1.0
                    if c < best:
                        best, move = c, "d"
                if j > 0:
                    c = dp[i][j - 1] + 1.0
                    if c < best:
                        best, move = c, "i"
                dp[i][j], bt[i][j] = best, move
        merged: list[list[TimedWord | None]] = []
        i, j = m, n
        while i > 0 or j > 0:
            move = bt[i][j]
            if move == "m":
                merged.append(slots[i - 1] + [seq[j - 1]])
                i, j = i - 1, j - 1
            elif move == "d":
                merged.append(slots[i - 1] + [None])
                i -= 1
            else:
                merged.append([None] * s + [seq[j - 1]])
                j -= 1
        merged.reverse()
        slots = merged
    return slots


# --------------------------------------------------------------------------
# Helpers on segments
# --------------------------------------------------------------------------


def source_tag(origin: str, audio_variant: str) -> str:
    """Readable source identifier: the origin, plus "@variant" unless original."""
    return origin if audio_variant == "original" else f"{origin}@{audio_variant}"


def _relabel(alt: Alternative, origin: str | None = None, variant: str | None = None) -> Alternative:
    update: dict[str, str] = {}
    if origin is not None and alt.origin != origin:
        update["origin"] = origin
    if variant is not None and alt.audio_variant != variant:
        update["audio_variant"] = variant
    return alt.model_copy(update=update) if update else alt


def join_nbest(
    segments: Sequence[SegmentHypotheses],
    audio_variant: str | None = None,
    origin: str | None = None,
) -> list[Alternative]:
    """Segment-level alternatives for consecutive segments of one source.

    One segment: its own hypotheses. Several: the k-th alternative joins the
    k-th hypothesis of each segment (falling back to the segment's best), so
    every candidate text survives. Segments without hypotheses contribute the
    text of their words. `origin` / `audio_variant`, when given, relabel the
    alternatives."""
    segs = sorted(segments, key=lambda s: (s.start, s.end))
    per_seg: list[list[Alternative]] = []
    for seg in segs:
        if seg.hypotheses:
            per_seg.append(list(seg.hypotheses))
        elif seg.words:
            text = " ".join(w.text for w in seg.words)
            per_seg.append([Alternative(text=text, origin=seg.words[0].origin)])
    if not per_seg:
        return []
    if len(per_seg) == 1:
        return [_relabel(a, origin, audio_variant) for a in per_seg[0]]
    out = []
    for k in range(max(len(p) for p in per_seg)):
        parts = [p[k] if k < len(p) else p[0] for p in per_seg]
        scores = [a.score for a in parts]
        out.append(
            Alternative(
                text=" ".join(a.text for a in parts if a.text.strip()),
                score=sum(scores) if all(x is not None for x in scores) else None,  # type: ignore[misc]
                origin=origin or parts[0].origin,
                audio_variant=audio_variant or parts[0].audio_variant,
            )
        )
    return out


def segment_text(segments: Sequence[SegmentHypotheses]) -> str:
    """Best text of consecutive segments (top hypothesis, else the words)."""
    alts = join_nbest(segments)
    return alts[0].text if alts else ""


def source_texts(segment: SegmentHypotheses) -> dict[str, str]:
    """Top text per source in a (possibly combined) segment, keyed by
    `source_tag(origin, audio_variant)`; the combined consensus is excluded.
    An empty string means the source declared it heard nothing."""
    out: dict[str, str] = {}
    for alt in segment.hypotheses:
        if alt.origin.startswith(COMBINED_PREFIX):
            continue
        out.setdefault(source_tag(alt.origin, alt.audio_variant), alt.text)
    if not out and segment.words:
        out[segment.words[0].origin] = " ".join(w.text for w in segment.words)
    return out


def _source_words(segs: Sequence[SegmentHypotheses]) -> list[TimedWord]:
    out: list[TimedWord] = []
    for seg in sorted(segs, key=lambda s: (s.start, s.end)):
        if seg.words:
            for w in sorted(seg.words, key=lambda w: (w.start, w.end)):
                out.append(TimedWord(w.text, w.start, w.end, w.confidence, tuple(w.alternatives)))
        elif seg.hypotheses and seg.hypotheses[0].text.strip():
            out.extend(TimedWord(tok) for tok in seg.hypotheses[0].text.split())
    return out


def _clusters(
    sources: Sequence[HypothesisSource], cfg: CombineConfig
) -> list[tuple[float, float, dict[int, list[SegmentHypotheses]]]]:
    items = sorted(
        ((seg.start, seg.end, s, seg) for s, src in enumerate(sources) for seg in src.segments),
        key=lambda x: (x[0], x[1], x[2]),
    )
    clusters: list[tuple[float, float, dict[int, list[SegmentHypotheses]]]] = []
    for start, end, s, seg in items:
        if clusters and start < clusters[-1][1] - cfg.cluster_min_overlap_s:
            c_start, c_end, members = clusters[-1]
            members.setdefault(s, []).append(seg)
            clusters[-1] = (c_start, max(c_end, end), members)
        else:
            clusters.append((start, end, {s: [seg]}))
    return clusters


def _fill_times(
    times: list[tuple[float | None, float | None]], lo: float, hi: float
) -> list[tuple[float, float]]:
    """Give approximate, evenly spread times to slots that have none."""
    out: list[tuple[float, float]] = []
    n = len(times)
    i = 0
    while i < n:
        s, e = times[i]
        if s is not None and e is not None:
            out.append((s, e))
            i += 1
            continue
        j = i
        while j < n and (times[j][0] is None or times[j][1] is None):
            j += 1
        left = out[-1][1] if out else lo
        right = times[j][0] if j < n else hi
        right = max(right, left)  # type: ignore[type-var]
        step = (right - left) / (j - i)  # type: ignore[operator]
        out.extend((left + k * step, left + (k + 1) * step) for k in range(j - i))
        i = j
    return out


def _mean(xs: Sequence[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


# --------------------------------------------------------------------------
# Combination
# --------------------------------------------------------------------------


@dataclass
class _SlotResult:
    word: WordHypothesis
    in_consensus: bool
    disagreement: bool
    lexicon_dependent: bool
    times_known: bool
    start: float | None = None
    end: float | None = None


def _vote_slot(
    slot: list[TimedWord | None],
    sources: Sequence[HypothesisSource],
    labels: list[str],
    variants: list[str],
    cfg: CombineConfig,
) -> _SlotResult:
    n_src = len(sources)
    keys = [_key(e, cfg) for e in slot]
    total_w = sum(max(src.weight, 0.0) for src in sources) or float(n_src)

    groups: dict[str, list[int]] = {}
    for s, k in enumerate(keys):
        groups.setdefault(k, []).append(s)

    def share(k: str) -> float:
        return sum(max(sources[s].weight, 0.0) for s in groups[k]) / total_w

    def known_conf(k: str) -> list[float]:
        entries = [slot[s] for s in groups[k]]
        return [e.confidence for e in entries if e is not None and e.confidence is not None]

    def rank(k: str) -> float:
        if k == ABSENT:
            return share(k) * cfg.absence_confidence
        c = _mean(known_conf(k))
        return share(k) * (c if c is not None else cfg.missing_confidence)

    words_keys = [k for k in groups if k != ABSENT]
    winner = max(words_keys, key=lambda k: (rank(k), share(k), -groups[k][0]))
    win_src = groups[winner]
    first = slot[win_src[0]]
    assert first is not None
    in_consensus = ABSENT not in groups or rank(winner) >= rank(ABSENT)

    confs = known_conf(winner)
    if n_src == 1:
        confidence = confs[0] if confs else None
    else:
        c = _mean(confs)
        confidence = share(winner) * c if c is not None else share(winner)

    timed = [slot[s] for s in win_src if slot[s] is not None and slot[s].timed]  # type: ignore[union-attr]
    if not timed:
        timed = [e for e in slot if e is not None and e.timed]
    start = _mean([w.start for w in timed]) if timed else None  # type: ignore[misc]
    end = _mean([w.end for w in timed]) if timed else None  # type: ignore[misc]

    alternatives: list[Alternative] = []
    seen: set[tuple[str, str, str]] = set()
    if n_src > 1:
        for s, e in enumerate(slot):
            alternatives.append(
                Alternative(
                    text=e.text if e is not None else ABSENT,
                    score=e.confidence if e is not None else None,
                    origin=labels[s],
                    audio_variant=variants[s],
                )
            )
            seen.add((alternatives[-1].text, labels[s], variants[s]))
    for s, e in enumerate(slot):
        if e is None:
            continue
        for alt in e.alternatives:
            alt = _relabel(alt, sources[s].origin, sources[s].audio_variant)
            sig = (alt.text, alt.origin, alt.audio_variant)
            if sig not in seen:
                seen.add(sig)
                alternatives.append(alt)

    if n_src == 1:
        origin = labels[0]
    else:
        origin = COMBINED_PREFIX + ORIGIN_JOIN.join(
            dict.fromkeys(source_tag(labels[s], variants[s]) for s in win_src)
        )

    lex = [s for s in range(n_src) if sources[s].uses_lexicon]
    plain = [s for s in range(n_src) if not sources[s].uses_lexicon]
    lex_keys = {keys[s] for s in lex}
    plain_keys = {keys[s] for s in plain}
    lexicon_dependent = bool(lex and plain and not (lex_keys & plain_keys) and (lex_keys - {ABSENT}))

    word = WordHypothesis(
        text=first.text,
        start=start if start is not None else 0.0,
        end=end if end is not None else 0.0,
        confidence=confidence,
        confidence_band=confidence_band(confidence, cfg.bands),
        origin=origin,
        alternatives=alternatives,
    )
    return _SlotResult(
        word=word,
        in_consensus=in_consensus,
        disagreement=len(groups) > 1,
        lexicon_dependent=lexicon_dependent,
        times_known=start is not None and end is not None,
        start=start,
        end=end,
    )


def combine_hypotheses(
    sources: Sequence[HypothesisSource], cfg: CombineConfig | None = None
) -> list[SegmentHypotheses]:
    """Combine several sources into one list of segments with per-word
    alternatives and declared origins. See the module docstring."""
    cfg = cfg or DEFAULT_COMBINE
    if not sources:
        return []
    labels = [src.label(i) for i, src in enumerate(sources)]
    variants = [src.variant() for src in sources]
    tags = [source_tag(lab, var) for lab, var in zip(labels, variants)]
    if len(set(tags)) != len(tags):
        raise ValueError(f"sources must differ by origin or audio variant, got {tags}")
    n_src = len(sources)

    out: list[SegmentHypotheses] = []
    for c_start, c_end, members in _clusters(sources, cfg):
        seqs = [_source_words(members.get(s, [])) for s in range(n_src)]
        slots = align_word_sequences(seqs, cfg)
        results = [_vote_slot(slot, sources, labels, variants, cfg) for slot in slots]

        filled = _fill_times([(r.start, r.end) for r in results], c_start, c_end)
        words = [
            r.word if r.times_known else r.word.model_copy(update={"start": t[0], "end": t[1]})
            for r, t in zip(results, filled)
        ]

        hypotheses: list[Alternative] = []
        if n_src > 1:
            cons = [w for w, r in zip(words, results) if r.in_consensus]
            confs = [w.confidence for w in cons if w.confidence is not None]
            hypotheses.append(
                Alternative(
                    text=" ".join(w.text for w in cons),
                    score=_mean(confs) if confs else None,
                    origin=COMBINED_PREFIX + ORIGIN_JOIN.join(tags),
                    audio_variant=variants[0] if len(set(variants)) == 1 else MIXED_VARIANT,
                )
            )
        for s in range(n_src):
            alts = join_nbest(members.get(s, []), sources[s].audio_variant, sources[s].origin)
            if not alts and n_src > 1:
                alts = [Alternative(text=ABSENT, origin=labels[s], audio_variant=variants[s])]
            hypotheses.extend(alts)

        segs = [seg for s in range(n_src) for seg in members.get(s, [])]
        flags: list[SegmentFlag] = []
        for seg in segs:
            for f in seg.flags:
                if f not in flags:
                    flags.append(f)
        if n_src > 1 and results:
            frac = sum(r.disagreement for r in results) / len(results)
            if frac > cfg.disagreement_min_fraction and SegmentFlag.MODEL_DISAGREEMENT not in flags:
                flags.append(SegmentFlag.MODEL_DISAGREEMENT)
        if any(r.lexicon_dependent for r in results) and SegmentFlag.LEXICON_DEPENDENT not in flags:
            flags.append(SegmentFlag.LEXICON_DEPENDENT)

        provenance = (
            StageKind.NEURAL
            if any(seg.audio_provenance == StageKind.NEURAL for seg in segs)
            else StageKind.DETERMINISTIC
        )
        out.append(
            SegmentHypotheses(
                start=c_start,
                end=c_end,
                audio_provenance=provenance,
                hypotheses=hypotheses,
                words=words,
                flags=flags,
                speaker_score=next(
                    (seg.speaker_score for seg in segs if seg.speaker_score is not None), None
                ),
                speaker_condition=next(
                    (seg.speaker_condition for seg in segs if seg.speaker_condition is not None), None
                ),
            )
        )
    return out


def consensus_text(segment: SegmentHypotheses) -> str:
    """Consensus (or single-source best) text of a combined segment."""
    return segment.hypotheses[0].text if segment.hypotheses else " ".join(w.text for w in segment.words)
