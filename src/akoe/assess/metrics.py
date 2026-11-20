# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Metrics for self-assessment (IMPLEMENTATION_PLAN M5, issue #27).

Model-independent, pure functions:

- Italian text normalization (configurable) and word/char tokenization;
- WER / CER with explicit substitution, deletion and insertion counts
  (own edit-distance implementation, cross-checked against jiwer in tests);
- oracle WER and exact-match coverage over N-best lists;
- confidence calibration: ECE and reliability-diagram data;
- insertion rate of lexicon terms on items where they are absent (bias test);
- EER and threshold for voiceprint scores;
- separation of candidate-phrase scores (true phrase against altered ones);
- quality of the "unintelligible" label.

Conventions: corpus-level rates are total errors over total reference units
(not a mean of per-item rates). Every function that cannot compute a value
returns None instead of inventing one.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.stats import rankdata

# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

_APOSTROPHES = "’‘ʼ´`′"
_APOS_TABLE = str.maketrans({c: "'" for c in _APOSTROPHES})
_NUMBER_INNER = set(".,:/")
_BRACKETED = re.compile(r"\[[^\]]*\]|\([^)]*\)|<[^>]*>|\{[^}]*\}")


@dataclass(frozen=True)
class NormalizationConfig:
    """How reference and hypothesis texts are normalized before scoring.

    Defaults suit Italian: lowercase, punctuation removed, elided forms split
    at the apostrophe ("l'altro" -> "l altro"), accents kept ("è" != "e"),
    numbers left as written ("3,5" and "10.000" stay intact; no digit-to-word
    conversion, so "tre" and "3" count as different words).
    """

    lowercase: bool = True
    unicode_form: Literal["NFC", "NFKC"] = "NFC"
    remove_punctuation: bool = True
    # split: "l'altro" -> "l altro"; keep: "l'altro"; join: "laltro"
    apostrophes: Literal["split", "keep", "join"] = "split"
    strip_accents: bool = False
    # Drop transcriber annotations such as "[incomprensibile]" or "(risata)".
    remove_bracketed: bool = False
    keep_number_punctuation: bool = True


DEFAULT_NORMALIZATION = NormalizationConfig()


def normalize_text(text: str, cfg: NormalizationConfig | None = None) -> str:
    """Return the normalized text, tokens separated by single spaces."""
    cfg = cfg or DEFAULT_NORMALIZATION
    t = unicodedata.normalize(cfg.unicode_form, text).translate(_APOS_TABLE)
    if cfg.remove_bracketed:
        t = _BRACKETED.sub(" ", t)
    if cfg.lowercase:
        t = t.lower()
    if cfg.strip_accents:
        t = "".join(c for c in unicodedata.normalize("NFD", t) if not unicodedata.combining(c))
        t = unicodedata.normalize("NFC", t)

    out: list[str] = []
    n = len(t)
    for i, c in enumerate(t):
        if c == "'":
            if cfg.apostrophes == "keep":
                out.append(c)
            elif cfg.apostrophes == "split":
                out.append(" ")
            # join: drop it
            continue
        if cfg.remove_punctuation and unicodedata.category(c).startswith("P"):
            if (
                cfg.keep_number_punctuation
                and c in _NUMBER_INNER
                and 0 < i < n - 1
                and t[i - 1].isdigit()
                and t[i + 1].isdigit()
            ):
                out.append(c)
            else:
                out.append(" ")
            continue
        out.append(c)
    return " ".join("".join(out).split())


def words(text: str, cfg: NormalizationConfig | None = None) -> list[str]:
    """Normalized word tokens."""
    return normalize_text(text, cfg).split()


def chars(text: str, cfg: NormalizationConfig | None = None, *, include_spaces: bool = True) -> list[str]:
    """Normalized character tokens (single spaces between words, as jiwer's CER)."""
    t = normalize_text(text, cfg)
    return list(t if include_spaces else t.replace(" ", ""))


# --------------------------------------------------------------------------
# Edit distance with S/D/I
# --------------------------------------------------------------------------

OpKind = Literal["equal", "sub", "del", "ins"]


@dataclass(frozen=True)
class AlignOp:
    op: OpKind
    ref_index: int | None
    hyp_index: int | None


def align(ref: Sequence[str], hyp: Sequence[str]) -> list[AlignOp]:
    """Minimum-edit alignment (unit costs). Ties prefer match/substitution,
    then deletion, then insertion. O(len(ref) * len(hyp))."""
    n, m = len(ref), len(hyp)
    # dp[i][j]: cost of aligning ref[:i] with hyp[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        row, prev, r = dp[i], dp[i - 1], ref[i - 1]
        for j in range(1, m + 1):
            diag = prev[j - 1] + (0 if r == hyp[j - 1] else 1)
            dele = prev[j] + 1
            ins = row[j - 1] + 1
            row[j] = min(diag, dele, ins)

    ops: list[AlignOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            same = ref[i - 1] == hyp[j - 1]
            if dp[i][j] == dp[i - 1][j - 1] + (0 if same else 1):
                ops.append(AlignOp("equal" if same else "sub", i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(AlignOp("del", i - 1, None))
            i -= 1
        else:
            ops.append(AlignOp("ins", None, j - 1))
            j -= 1
    ops.reverse()
    return ops


@dataclass
class EditCounts:
    """Edit counts against a reference. `rate` is (S + D + I) / reference length."""

    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    hits: int = 0

    @property
    def n_ref(self) -> int:
        return self.substitutions + self.deletions + self.hits

    @property
    def n_hyp(self) -> int:
        return self.substitutions + self.insertions + self.hits

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float | None:
        return self.errors / self.n_ref if self.n_ref else None

    def __add__(self, other: "EditCounts") -> "EditCounts":
        return EditCounts(
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
            self.hits + other.hits,
        )

    @classmethod
    def from_ops(cls, ops: Sequence[AlignOp]) -> "EditCounts":
        c = cls()
        for o in ops:
            if o.op == "equal":
                c.hits += 1
            elif o.op == "sub":
                c.substitutions += 1
            elif o.op == "del":
                c.deletions += 1
            else:
                c.insertions += 1
        return c


def edit_counts(ref: Sequence[str], hyp: Sequence[str]) -> EditCounts:
    return EditCounts.from_ops(align(ref, hyp))


def _as_list(x: str | Sequence[str]) -> list[str]:
    return [x] if isinstance(x, str) else list(x)


def _pairs(reference: str | Sequence[str], hypothesis: str | Sequence[str]) -> list[tuple[str, str]]:
    refs, hyps = _as_list(reference), _as_list(hypothesis)
    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} references but {len(hyps)} hypotheses")
    return list(zip(refs, hyps))


def word_error_counts(
    reference: str | Sequence[str],
    hypothesis: str | Sequence[str],
    cfg: NormalizationConfig | None = None,
) -> EditCounts:
    """Corpus-level word edit counts (one text or parallel lists of texts)."""
    total = EditCounts()
    for r, h in _pairs(reference, hypothesis):
        total = total + edit_counts(words(r, cfg), words(h, cfg))
    return total


def char_error_counts(
    reference: str | Sequence[str],
    hypothesis: str | Sequence[str],
    cfg: NormalizationConfig | None = None,
    *,
    include_spaces: bool = True,
) -> EditCounts:
    total = EditCounts()
    for r, h in _pairs(reference, hypothesis):
        total = total + edit_counts(
            chars(r, cfg, include_spaces=include_spaces), chars(h, cfg, include_spaces=include_spaces)
        )
    return total


def _rate_or_raise(c: EditCounts, what: str) -> float:
    if c.rate is None:
        raise ValueError(f"{what}: the reference is empty after normalization")
    return c.rate


def wer(
    reference: str | Sequence[str],
    hypothesis: str | Sequence[str],
    cfg: NormalizationConfig | None = None,
) -> float:
    """Word error rate. Raises ValueError if the (total) reference is empty."""
    return _rate_or_raise(word_error_counts(reference, hypothesis, cfg), "WER")


def cer(
    reference: str | Sequence[str],
    hypothesis: str | Sequence[str],
    cfg: NormalizationConfig | None = None,
    *,
    include_spaces: bool = True,
) -> float:
    """Character error rate on normalized text (spaces count, as in jiwer)."""
    return _rate_or_raise(
        char_error_counts(reference, hypothesis, cfg, include_spaces=include_spaces), "CER"
    )


# --------------------------------------------------------------------------
# N-best: oracle WER and coverage
# --------------------------------------------------------------------------


@dataclass
class OracleResult:
    oracle_wer: float | None
    counts: EditCounts
    best_indices: list[int | None]  # index of the closest hypothesis per item
    exact_coverage: float | None  # fraction of items whose reference is in the list


def oracle_wer(
    references: Sequence[str],
    nbest_lists: Sequence[Sequence[str]],
    cfg: NormalizationConfig | None = None,
) -> OracleResult:
    """Oracle WER: for each item, the hypothesis closest to the reference.

    An empty list counts as an empty hypothesis (all deletions). Coverage is
    the fraction of items whose normalized reference appears verbatim among
    the alternatives ("how often the right sentence is offered").
    """
    if len(references) != len(nbest_lists):
        raise ValueError(f"{len(references)} references but {len(nbest_lists)} N-best lists")
    total = EditCounts()
    best_idx: list[int | None] = []
    covered = 0
    for ref, nbest in zip(references, nbest_lists):
        r = words(ref, cfg)
        if not nbest:
            total = total + edit_counts(r, [])
            best_idx.append(None)
            continue
        cands = [words(h, cfg) for h in nbest]
        scored = [edit_counts(r, h) for h in cands]
        k = min(range(len(scored)), key=lambda i: (scored[i].errors, i))
        total = total + scored[k]
        best_idx.append(k)
        covered += any(h == r for h in cands)
    n = len(references)
    return OracleResult(
        oracle_wer=total.rate,
        counts=total,
        best_indices=best_idx,
        exact_coverage=covered / n if n else None,
    )


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def label_hypothesis_words(
    reference: str,
    hypothesis_words: Sequence[str],
    cfg: NormalizationConfig | None = None,
) -> list[bool | None]:
    """Correctness of each hypothesis word against the reference.

    A word is correct when all its normalized tokens align as matches
    (a word such as "l'altro" may yield two tokens). Words with no token
    after normalization (pure punctuation) get None and should be skipped.
    """
    owner: list[int] = []
    flat: list[str] = []
    for wi, w in enumerate(hypothesis_words):
        for tok in words(w, cfg):
            flat.append(tok)
            owner.append(wi)
    ok_tok = [False] * len(flat)
    for op in align(words(reference, cfg), flat):
        if op.op == "equal" and op.hyp_index is not None:
            ok_tok[op.hyp_index] = True
    result: list[bool | None] = [None] * len(hypothesis_words)
    for ti, wi in enumerate(owner):
        result[wi] = ok_tok[ti] if result[wi] is None else (result[wi] and ok_tok[ti])
    return result


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float | None
    accuracy: float | None


def _clean_conf(
    confidences: Sequence[float], correct: Sequence[bool | None]
) -> tuple[np.ndarray, np.ndarray]:
    if len(confidences) != len(correct):
        raise ValueError("confidences and correctness labels differ in length")
    pairs = [(float(c), bool(k)) for c, k in zip(confidences, correct) if c is not None and k is not None]
    conf = np.array([p[0] for p in pairs], dtype=np.float64)
    ok = np.array([p[1] for p in pairs], dtype=bool)
    if conf.size and (np.any(conf < 0.0) or np.any(conf > 1.0) or np.any(np.isnan(conf))):
        raise ValueError("confidences must lie in [0, 1]")
    return conf, ok


def reliability_diagram(
    confidences: Sequence[float],
    correct: Sequence[bool | None],
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
) -> list[ReliabilityBin]:
    """Reliability-diagram data. `uniform`: equal-width bins on [0, 1] (the last
    bin includes 1.0), empty bins kept with None values. `quantile`: bins with
    (nearly) equal counts; bounds are the min/max confidence in each bin."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    conf, ok = _clean_conf(confidences, correct)
    bins: list[ReliabilityBin] = []
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip(np.floor(conf * n_bins).astype(int), 0, n_bins - 1)
        for b in range(n_bins):
            sel = idx == b
            cnt = int(sel.sum())
            bins.append(
                ReliabilityBin(
                    lower=float(edges[b]),
                    upper=float(edges[b + 1]),
                    count=cnt,
                    mean_confidence=float(conf[sel].mean()) if cnt else None,
                    accuracy=float(ok[sel].mean()) if cnt else None,
                )
            )
    elif strategy == "quantile":
        order = np.argsort(conf, kind="stable")
        for group in np.array_split(order, min(n_bins, max(conf.size, 1))):
            if group.size == 0:
                continue
            c = conf[group]
            bins.append(
                ReliabilityBin(
                    lower=float(c.min()),
                    upper=float(c.max()),
                    count=int(group.size),
                    mean_confidence=float(c.mean()),
                    accuracy=float(ok[group].mean()),
                )
            )
    else:
        raise ValueError(f"unknown binning strategy: {strategy}")
    return bins


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool | None],
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
) -> float | None:
    """ECE = sum_b (n_b / N) * |accuracy_b - mean_confidence_b|. None if no data."""
    bins = reliability_diagram(confidences, correct, n_bins, strategy)
    total = sum(b.count for b in bins)
    if total == 0:
        return None
    gaps = [b.count / total * abs(b.accuracy - b.mean_confidence) for b in bins if b.count]  # type: ignore[operator]
    return float(sum(gaps))


# --------------------------------------------------------------------------
# Lexicon insertion (bias test)
# --------------------------------------------------------------------------


def _contains(tokens: Sequence[str], term: Sequence[str]) -> bool:
    k = len(term)
    if k == 0:
        return False
    return any(list(tokens[i : i + k]) == list(term) for i in range(len(tokens) - k + 1))


@dataclass
class LexiconInsertionResult:
    """`rate`: insertions / opportunities, where an opportunity is an
    (item, term) pair with the term absent from the reference. `item_rate`:
    fraction of items with at least one absent term whose hypothesis contains
    at least one of them."""

    rate: float | None
    n_opportunities: int
    n_insertions: int
    item_rate: float | None
    n_items_at_risk: int
    per_term: dict[str, tuple[int, int]] = field(default_factory=dict)  # term -> (insertions, opportunities)


def lexicon_insertion_rate(
    references: Sequence[str],
    hypotheses: Sequence[str],
    lexicon: Sequence[str],
    cfg: NormalizationConfig | None = None,
) -> LexiconInsertionResult:
    """How often lexicon terms appear in hypotheses where the speaker did not
    say them. Multi-word terms match as contiguous normalized token runs."""
    if len(references) != len(hypotheses):
        raise ValueError(f"{len(references)} references but {len(hypotheses)} hypotheses")
    terms = [(t, words(t, cfg)) for t in dict.fromkeys(lexicon)]
    terms = [(t, toks) for t, toks in terms if toks]
    per_term = {t: [0, 0] for t, _ in terms}
    n_opp = n_ins = items_at_risk = items_hit = 0
    for ref, hyp in zip(references, hypotheses):
        r, h = words(ref, cfg), words(hyp, cfg)
        at_risk = hit = False
        for t, toks in terms:
            if _contains(r, toks):
                continue
            at_risk = True
            n_opp += 1
            per_term[t][1] += 1
            if _contains(h, toks):
                hit = True
                n_ins += 1
                per_term[t][0] += 1
        items_at_risk += at_risk
        items_hit += hit
    return LexiconInsertionResult(
        rate=n_ins / n_opp if n_opp else None,
        n_opportunities=n_opp,
        n_insertions=n_ins,
        item_rate=items_hit / items_at_risk if items_at_risk else None,
        n_items_at_risk=items_at_risk,
        per_term={t: (v[0], v[1]) for t, v in per_term.items()},
    )


# --------------------------------------------------------------------------
# Voiceprint: EER, AUC
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EERResult:
    """Equal error rate. Decision rule: accept as target when score >= threshold."""

    eer: float
    threshold: float
    n_target: int
    n_nontarget: int


def equal_error_rate(target_scores: Sequence[float], nontarget_scores: Sequence[float]) -> EERResult:
    """EER with linear interpolation between operating points.

    Higher scores mean "more compatible with the subject". FAR(t) is the
    fraction of non-target scores >= t, FRR(t) the fraction of target
    scores < t.
    """
    tgt = np.asarray(target_scores, dtype=np.float64)
    non = np.asarray(nontarget_scores, dtype=np.float64)
    if tgt.size == 0 or non.size == 0:
        raise ValueError("EER needs at least one target and one non-target score")
    thr = np.unique(np.concatenate([tgt, non]))
    span = float(thr[-1] - thr[0]) or 1.0
    thr = np.append(thr, thr[-1] + 1e-6 * span)
    tgt_sorted, non_sorted = np.sort(tgt), np.sort(non)
    frr = np.searchsorted(tgt_sorted, thr, side="left") / tgt.size
    far = 1.0 - np.searchsorted(non_sorted, thr, side="left") / non.size
    k = int(np.argmax(frr >= far))  # first crossing; exists because frr[-1] = 1, far[-1] = 0
    if k == 0:
        return EERResult(float(far[0]), float(thr[0]), int(tgt.size), int(non.size))
    d0 = far[k - 1] - frr[k - 1]
    d1 = far[k] - frr[k]
    a = d0 / (d0 - d1) if d0 != d1 else 1.0
    eer = far[k - 1] + a * (far[k] - far[k - 1])
    t = thr[k - 1] + a * (thr[k] - thr[k - 1])
    return EERResult(float(eer), float(t), int(tgt.size), int(non.size))


def roc_auc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float | None:
    """P(positive > negative) + 0.5 * P(tie) (Mann-Whitney). None if a class is empty."""
    pos = np.asarray(positive_scores, dtype=np.float64)
    neg = np.asarray(negative_scores, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return None
    ranks = rankdata(np.concatenate([pos, neg]))
    u = ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / den) if den > 0 else 0.0


# --------------------------------------------------------------------------
# Candidate phrases: true against altered
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateTrial:
    """Scores from ONE CandidateScorer call (scores are comparable only
    within a call): the true phrase and its altered versions."""

    true_score: float
    altered_scores: tuple[float, ...]


@dataclass(frozen=True)
class CandidateSeparation:
    n_trials: int
    top1_rate: float | None  # true phrase strictly above every altered one
    tie_rate: float | None  # true phrase tied with the best altered one
    pairwise_auc: float | None  # within-call pairs, ties count 0.5
    mean_margin: float | None  # mean of (true - best altered)


def candidate_separation(trials: Sequence[CandidateTrial]) -> CandidateSeparation:
    """How often the true phrase outscores altered phrases.

    The AUC is computed over (true, altered) pairs within each call only,
    because scores from different calls are not comparable.
    """
    valid = [t for t in trials if t.altered_scores]
    if not valid:
        return CandidateSeparation(0, None, None, None, None)
    wins = ties = 0
    pair_sum = 0.0
    n_pairs = 0
    margins = []
    for t in valid:
        best = max(t.altered_scores)
        wins += t.true_score > best
        ties += t.true_score == best
        margins.append(t.true_score - best)
        for a in t.altered_scores:
            pair_sum += 1.0 if t.true_score > a else 0.5 if t.true_score == a else 0.0
            n_pairs += 1
    n = len(valid)
    return CandidateSeparation(
        n_trials=n,
        top1_rate=wins / n,
        tie_rate=ties / n,
        pairwise_auc=pair_sum / n_pairs,
        mean_margin=float(np.mean(margins)),
    )


# --------------------------------------------------------------------------
# "Unintelligible" label quality
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class UnintelligibleQuality:
    """`precision`: flagged items whose WER is >= the threshold (the label was
    deserved). `recall`: items with WER >= threshold that were flagged."""

    flag_rate: float | None
    precision: float | None
    recall: float | None


def unintelligible_quality(
    flagged: Sequence[bool], item_wer: Sequence[float], wer_threshold: float = 0.5
) -> UnintelligibleQuality:
    if len(flagged) != len(item_wer):
        raise ValueError("flags and WERs differ in length")
    n = len(flagged)
    if n == 0:
        return UnintelligibleQuality(None, None, None)
    bad = [w >= wer_threshold for w in item_wer]
    tp = sum(f and b for f, b in zip(flagged, bad))
    n_flag, n_bad = sum(flagged), sum(bad)
    return UnintelligibleQuality(
        flag_rate=n_flag / n,
        precision=tp / n_flag if n_flag else None,
        recall=tp / n_bad if n_bad else None,
    )
