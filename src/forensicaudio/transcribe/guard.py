"""Hallucination guard (IMPLEMENTATION_PLAN M2, issue #16).

Signals (MASTER_PLAN §3.6.3):
- text produced on spans where the voice-activity detector finds little or
  no speech (Whisper large-v3 writes text on ~40% of non-speech inputs);
- phrases typical of hallucinations ("Bag of Hallucinations", Barański et
  al. 2025), e.g. subtitle credits and video sign-offs in Italian and English;
- repetition loops (repeated n-grams, high compression ratio);
- strong disagreement between sources (HALAS, Barański et al. 2026).

The guard only adds flags and explains them; it never deletes or edits text,
words, alternatives or segments.
"""
from __future__ import annotations

import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations

from forensicaudio.assess.metrics import DEFAULT_NORMALIZATION, NormalizationConfig, edit_counts, words
from forensicaudio.models import SegmentFlag, SegmentHypotheses
from forensicaudio.transcribe.combine import COMBINED_PREFIX, source_texts

# Flagged when contained anywhere in a source's text (matched on normalized tokens).
DEFAULT_CONTAINS_PHRASES: tuple[str, ...] = (
    # Italian
    "Sottotitoli creati dalla comunità Amara.org",
    "Sottotitoli e revisione a cura di QTSS",
    "Sottotitoli a cura di QTSS",
    "Sottotitoli a cura di",
    "Sottotitoli creati da",
    "Amara.org",
    "Grazie per la visione",
    "Grazie per aver guardato",
    "Grazie per l'attenzione",
    "Grazie per l'ascolto",
    "Iscriviti al canale",
    "Iscrivetevi al canale",
    "Non dimenticate di iscrivervi",
    "Lascia un like",
    "Ci vediamo nel prossimo video",
    "Al prossimo video",
    "Trascrizione a cura di",
    # English
    "Thank you for watching",
    "Thanks for watching",
    "Please subscribe",
    "Like and subscribe",
    "Subscribe to my channel",
    "Don't forget to subscribe",
    "Subtitles by the Amara.org community",
    "Subtitles by",
    "Transcribed by",
    "Captions by",
    "See you in the next video",
    "♪",
    "♫",
)

# Flagged only when they are the whole text of a source (they are ordinary
# words otherwise).
DEFAULT_EXACT_PHRASES: tuple[str, ...] = (
    "Grazie",
    "Grazie a tutti",
    "Grazie mille",
    "Grazie e arrivederci",
    "Ciao a tutti",
    "Buona visione",
    "Sottotitoli",
    "Musica",
    "Applausi",
    "Risate",
    "Silenzio",
    "Thank you",
    "Thank you very much",
    "Thanks",
    "You",
    "Bye",
    "Bye bye",
    "Music",
    "Applause",
    "Laughter",
    "Silence",
    "Subtitles",
)


class GuardReason(str, Enum):
    NO_SPEECH = "no_speech"
    WORDS_OUTSIDE_SPEECH = "words_outside_speech"
    KNOWN_PHRASE = "known_phrase"
    REPETITION_LOOP = "repetition_loop"
    HIGH_COMPRESSION = "high_compression"
    STRONG_DISAGREEMENT = "strong_disagreement"


@dataclass(frozen=True)
class GuardConfig:
    # Text on a span whose speech coverage is below this fraction is flagged.
    min_speech_fraction: float = 0.2
    # Flag when more than this fraction of timed words fall outside speech.
    max_words_outside_speech: float = 0.5
    contains_phrases: tuple[str, ...] = DEFAULT_CONTAINS_PHRASES
    exact_phrases: tuple[str, ...] = DEFAULT_EXACT_PHRASES
    max_ngram: int = 6
    min_repeats_unigram: int = 5
    min_repeats_ngram: int = 3
    # Whisper's own loop heuristic (gzip ratio > 2.4); None disables it.
    max_compression_ratio: float | None = 2.4
    min_chars_for_compression: int = 40
    # Mean pairwise normalized word edit distance between sources.
    strong_disagreement: float = 0.6
    normalization: NormalizationConfig = DEFAULT_NORMALIZATION


DEFAULT_GUARD = GuardConfig()


@dataclass(frozen=True)
class GuardFinding:
    segment_index: int
    start: float
    end: float
    flag: SegmentFlag
    reason: GuardReason
    message: str  # operator-facing, Italian
    origin: str | None = None


@dataclass
class GuardResult:
    segments: list[SegmentHypotheses]
    findings: list[GuardFinding] = field(default_factory=list)

    def for_segment(self, index: int) -> list[GuardFinding]:
        return [f for f in self.findings if f.segment_index == index]


def _it(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


# --------------------------------------------------------------------------
# Individual signals
# --------------------------------------------------------------------------


def merge_regions(regions: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for s, e in sorted((float(a), float(b)) for a, b in regions if b > a):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def speech_fraction(start: float, end: float, regions: Sequence[tuple[float, float]]) -> float:
    """Fraction of [start, end] covered by speech regions (point spans: 1 or 0)."""
    merged = merge_regions(regions)
    if end <= start:
        return 1.0 if any(s <= start <= e for s, e in merged) else 0.0
    covered = sum(max(0.0, min(end, e) - max(start, s)) for s, e in merged)
    return min(1.0, covered / (end - start))


def _inside(t: float, merged: Sequence[tuple[float, float]]) -> bool:
    return any(s <= t <= e for s, e in merged)


def _contains(tokens: Sequence[str], term: Sequence[str]) -> bool:
    k = len(term)
    return k > 0 and any(list(tokens[i : i + k]) == list(term) for i in range(len(tokens) - k + 1))


def match_known_phrases(text: str, cfg: GuardConfig | None = None) -> list[str]:
    """Hallucination-list phrases found in `text` (as configured, original spelling)."""
    cfg = cfg or DEFAULT_GUARD
    toks = words(text, cfg.normalization)
    if not toks:
        return []
    found = [p for p in cfg.contains_phrases if _contains(toks, words(p, cfg.normalization))]
    found += [p for p in cfg.exact_phrases if toks == words(p, cfg.normalization)]
    # Report only the longest of overlapping matches ("Sottotitoli a cura di QTSS",
    # not also "Sottotitoli a cura di").
    norm = {p: words(p, cfg.normalization) for p in found}
    return [
        p for p in dict.fromkeys(found)
        if not any(q != p and len(norm[q]) > len(norm[p]) and _contains(norm[q], norm[p]) for q in found)
    ]


def repetition_loop(tokens: Sequence[str], cfg: GuardConfig | None = None) -> tuple[str, int] | None:
    """First n-gram repeated consecutively at least the configured number of
    times, as (n-gram, count); None if there is no loop."""
    cfg = cfg or DEFAULT_GUARD
    L = len(tokens)
    for n in range(1, min(cfg.max_ngram, L) + 1):
        need = cfg.min_repeats_unigram if n == 1 else cfg.min_repeats_ngram
        if need * n > L:
            continue
        for i in range(L - n + 1):
            gram = list(tokens[i : i + n])
            k = 1
            while list(tokens[i + k * n : i + (k + 1) * n]) == gram:
                k += 1
            if k >= need:
                return " ".join(gram), k
    return None


def compression_ratio(text: str) -> float:
    raw = text.encode("utf-8")
    return len(raw) / len(zlib.compress(raw)) if raw else 0.0


def disagreement_score(texts: Sequence[str], cfg: GuardConfig | None = None) -> float | None:
    """Mean pairwise word edit distance normalized by the longer text (0 =
    identical, 1 = nothing in common; an empty text against a non-empty one
    counts 1). None with fewer than two texts."""
    cfg = cfg or DEFAULT_GUARD
    toks = [words(t, cfg.normalization) for t in texts]
    if len(toks) < 2:
        return None
    dists = []
    for a, b in combinations(toks, 2):
        longest = max(len(a), len(b))
        dists.append(edit_counts(a, b).errors / longest if longest else 0.0)
    return sum(dists) / len(dists)


# --------------------------------------------------------------------------
# Guard
# --------------------------------------------------------------------------


def _segment_texts(seg: SegmentHypotheses) -> list[tuple[str | None, str]]:
    """Texts to inspect: the best text of each source, plus the consensus."""
    out: list[tuple[str | None, str]] = [(o, t) for o, t in source_texts(seg).items()]
    for alt in seg.hypotheses:
        if alt.origin.startswith(COMBINED_PREFIX):
            out.append((alt.origin, alt.text))
            break
    if not out and seg.words:
        out.append((seg.words[0].origin, " ".join(w.text for w in seg.words)))
    return out


def check_segment(
    seg: SegmentHypotheses,
    index: int,
    vad_regions: Sequence[tuple[float, float]] | None = None,
    cfg: GuardConfig | None = None,
) -> list[GuardFinding]:
    cfg = cfg or DEFAULT_GUARD
    findings: list[GuardFinding] = []

    def add(flag: SegmentFlag, reason: GuardReason, message: str, origin: str | None = None) -> None:
        findings.append(GuardFinding(index, seg.start, seg.end, flag, reason, message, origin))

    hall = SegmentFlag.POSSIBLE_HALLUCINATION
    texts = _segment_texts(seg)
    has_text = any(t.strip() for _, t in texts) or any(w.text.strip() for w in seg.words)

    if vad_regions is not None and has_text:
        merged = merge_regions(vad_regions)
        frac = speech_fraction(seg.start, seg.end, merged)
        if frac < cfg.min_speech_fraction:
            add(hall, GuardReason.NO_SPEECH,
                "Testo prodotto su un tratto in cui il rilevatore di voce trova poco o nessun parlato "
                f"({_it(100 * frac, 0)}% del tratto).")
        else:
            timed = [w for w in seg.words if w.end >= w.start and w.text.strip()]
            outside = [w for w in timed if not _inside((w.start + w.end) / 2, merged)]
            if timed and len(outside) / len(timed) > cfg.max_words_outside_speech:
                add(hall, GuardReason.WORDS_OUTSIDE_SPEECH,
                    f"{len(outside)} parole su {len(timed)} cadono fuori dai tratti di parlato rilevati.")

    seen_phrases: set[str] = set()
    for origin, text in texts:
        for phrase in match_known_phrases(text, cfg):
            if phrase in seen_phrases:
                continue
            seen_phrases.add(phrase)
            add(hall, GuardReason.KNOWN_PHRASE,
                "Frase tipica delle allucinazioni dei trascrittori automatici: "
                f"«{phrase}» (fonte: {origin}).",
                origin)
        loop = repetition_loop(words(text, cfg.normalization), cfg)
        if loop is not None:
            add(hall, GuardReason.REPETITION_LOOP,
                f"Ripetizione anomala: «{loop[0]}» ripetuto {loop[1]} volte di seguito (fonte: {origin}).",
                origin)
        elif cfg.max_compression_ratio is not None and len(text) >= cfg.min_chars_for_compression:
            ratio = compression_ratio(text)
            if ratio > cfg.max_compression_ratio:
                add(hall, GuardReason.HIGH_COMPRESSION,
                    f"Testo molto ripetitivo (rapporto di compressione {_it(ratio, 1)}) (fonte: {origin}).",
                    origin)

    per_source = list(source_texts(seg).values())
    d = disagreement_score(per_source, cfg)
    if d is not None and d >= cfg.strong_disagreement:
        msg = (f"Forte disaccordo tra i modelli (distanza media {_it(d)}): "
               "il testo potrebbe essere inventato da uno di essi.")
        add(hall, GuardReason.STRONG_DISAGREEMENT, msg)
        add(SegmentFlag.MODEL_DISAGREEMENT, GuardReason.STRONG_DISAGREEMENT, msg)
    return findings


def guard_segments(
    segments: Sequence[SegmentHypotheses],
    vad_regions: Sequence[tuple[float, float]] | None = None,
    cfg: GuardConfig | None = None,
) -> GuardResult:
    """Return copies of `segments` with hallucination flags added, and the
    reasons. `vad_regions=None` skips the speech check (no VAD available);
    an empty list means the VAD found no speech at all."""
    cfg = cfg or DEFAULT_GUARD
    out: list[SegmentHypotheses] = []
    findings: list[GuardFinding] = []
    for i, seg in enumerate(segments):
        f = check_segment(seg, i, vad_regions, cfg)
        findings.extend(f)
        flags = list(seg.flags)
        for x in f:
            if x.flag not in flags:
                flags.append(x.flag)
        out.append(seg.model_copy(update={"flags": flags}, deep=True))
    return GuardResult(out, findings)
