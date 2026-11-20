"""Candidate phrase comparison ("did he say X or Y?", MASTER_PLAN §4.6, issue #15).

Runs several independent scorers (typically the CTC forced alignment and the
Whisper teacher-forced likelihood) on the same audio and candidates, and
reports each scorer **separately**: scores are never merged into one number.

Thresholds are calibrated elsewhere (self-assessment, M5) and passed in by the
caller, one per scorer name. Per scorer: a candidate is "compatible" when its
score reaches the threshold; "none compatible" when no candidate does. Overall:
- NONE_COMPATIBLE: every assessed scorer finds no compatible candidate;
- SOME_COMPATIBLE: every assessed scorer finds at least one;
- DISAGREEMENT: the scorers disagree on that;
- NOT_ASSESSED: no threshold given (ranking only, no compatibility verdict).

Example::

    whisper = WhisperHFBackend("openai/whisper-small")
    scorers = [CTCForcedAlignmentScorer(), WhisperDecoderScorer(whisper)]
    result = compare_candidates(audio, sr, ["vado a casa", "vado a cassa"], scorers,
                                thresholds={scorers[0].name: -0.35, scorers[1].name: -1.2})
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from forensicaudio.interfaces import CandidateScorer

RELIABLE_MAX_CANDIDATES = 5  # score/error correlation weakens with more (Novosad 2026)


class ScorerOutcome(str, Enum):
    COMPATIBLE_FOUND = "compatible_found"
    NONE_COMPATIBLE = "none_compatible"
    NOT_ASSESSED = "not_assessed"


class ComparisonOutcome(str, Enum):
    SOME_COMPATIBLE = "some_compatible"
    NONE_COMPATIBLE = "none_compatible"
    DISAGREEMENT = "disagreement"
    NOT_ASSESSED = "not_assessed"


class CandidateEntry(BaseModel):
    candidate: str
    score: float | None  # None when the candidate cannot be aligned at all (-inf)
    rank: int  # 1 = best under this scorer
    compatible: bool | None = None  # None when no threshold was given
    details: dict[str, Any] | None = None  # scorer-specific (alignment, per-token...)


class ScorerReport(BaseModel):
    scorer: str
    threshold: float | None = None
    outcome: ScorerOutcome
    best: str | None = None
    entries: list[CandidateEntry] = Field(default_factory=list)  # in input order
    provenance: dict[str, Any] | None = None


class CandidateComparison(BaseModel):
    candidates: list[str]
    reports: list[ScorerReport]
    outcome: ComparisonOutcome
    same_best: bool | None = None  # do all scorers rank the same candidate first?
    message: str  # for the operator (Italian)
    warnings: list[str] = Field(default_factory=list)


def _details(obj: Any) -> dict[str, Any] | None:
    if obj is None:
        return None
    if dataclasses.is_dataclass(obj):
        d = dataclasses.asdict(obj)
        for prop in ("score", "per_frame", "per_token", "relative_per_frame", "mean_logprob", "n_tokens"):
            if hasattr(obj, prop):
                d[prop] = getattr(obj, prop)
        return _finite(d)
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    return None


def _finite(x: Any) -> Any:
    """Replace non-finite floats with None so the result serializes to JSON."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    if isinstance(x, np.ndarray):
        return _finite(x.tolist())
    return x


def _report(scorer: CandidateScorer, audio: np.ndarray, sr: int, candidates: list[str],
            threshold: float | None) -> ScorerReport:
    detailed = None
    if hasattr(scorer, "score_detailed"):
        detailed = scorer.score_detailed(audio, sr, candidates)
        scores = [float(d.score) for d in detailed]
    else:
        scores = [float(s) for s in scorer.score(audio, sr, candidates)]
    if len(scores) != len(candidates):
        raise RuntimeError(f"Lo scorer {scorer.name} ha restituito {len(scores)} punteggi per "
                           f"{len(candidates)} candidate.")
    finite = [s if math.isfinite(s) else -math.inf for s in scores]
    order = sorted(range(len(candidates)), key=lambda i: finite[i], reverse=True)
    ranks = {i: r + 1 for r, i in enumerate(order)}
    entries = []
    for i, cand in enumerate(candidates):
        ok = None if threshold is None else bool(math.isfinite(finite[i]) and finite[i] >= threshold)
        entries.append(CandidateEntry(
            candidate=cand, score=finite[i] if math.isfinite(finite[i]) else None, rank=ranks[i],
            compatible=ok, details=_details(detailed[i]) if detailed is not None else None,
        ))
    if threshold is None:
        outcome = ScorerOutcome.NOT_ASSESSED
    elif any(e.compatible for e in entries):
        outcome = ScorerOutcome.COMPATIBLE_FOUND
    else:
        outcome = ScorerOutcome.NONE_COMPATIBLE
    prov = getattr(scorer, "provenance", None)
    best = candidates[order[0]] if math.isfinite(finite[order[0]]) else None
    return ScorerReport(
        scorer=scorer.name, threshold=threshold, outcome=outcome, best=best, entries=entries,
        provenance=prov.model_dump(mode="json") if isinstance(prov, BaseModel) else None,
    )


_MESSAGES = {
    ComparisonOutcome.SOME_COMPATIBLE: "Tutti i metodi trovano almeno una frase compatibile con l'audio "
    "(soglie tarate nell'autovalutazione). I punteggi dei metodi sono riportati separatamente.",
    ComparisonOutcome.NONE_COMPATIBLE: "Nessuna frase candidata è compatibile con l'audio secondo tutti i "
    "metodi valutati: nessuna compatibile.",
    ComparisonOutcome.DISAGREEMENT: "I metodi non concordano: almeno uno trova una frase compatibile, "
    "almeno un altro nessuna. Riascoltare e consultare i punteggi separati.",
    ComparisonOutcome.NOT_ASSESSED: "Nessuna soglia tarata disponibile: solo ordinamento delle frasi, "
    "senza giudizio di compatibilità.",
}


def compare_candidates(
    audio: np.ndarray,
    sr: int,
    candidates: list[str],
    scorers: Sequence[CandidateScorer],
    thresholds: Mapping[str, float] | None = None,
) -> CandidateComparison:
    """Score `candidates` with every scorer and report each one separately."""
    cands = [c for c in candidates]
    if not cands or any(not c.strip() for c in cands):
        raise ValueError("Servono una o più frasi candidate non vuote.")
    if len(set(cands)) != len(cands):
        raise ValueError("Le frasi candidate contengono duplicati.")
    if not scorers:
        raise ValueError("Serve almeno un metodo di punteggio.")
    names = [s.name for s in scorers]
    if len(set(names)) != len(names):
        raise ValueError(f"Nomi dei metodi di punteggio duplicati: {names}")
    thresholds = dict(thresholds or {})
    unknown = set(thresholds) - set(names)
    if unknown:
        raise ValueError(f"Soglie per metodi sconosciuti: {sorted(unknown)}; metodi: {names}")
    warnings = []
    if len(cands) > RELIABLE_MAX_CANDIDATES:
        warnings.append(
            f"{len(cands)} frasi candidate: con più di {RELIABLE_MAX_CANDIDATES} la relazione tra "
            "punteggio ed errore si indebolisce. Ridurre le candidate se possibile."
        )
    reports = [_report(s, audio, sr, cands, thresholds.get(s.name)) for s in scorers]
    assessed = [r.outcome for r in reports if r.outcome != ScorerOutcome.NOT_ASSESSED]
    if not assessed:
        outcome = ComparisonOutcome.NOT_ASSESSED
    elif all(o == ScorerOutcome.NONE_COMPATIBLE for o in assessed):
        outcome = ComparisonOutcome.NONE_COMPATIBLE
    elif all(o == ScorerOutcome.COMPATIBLE_FOUND for o in assessed):
        outcome = ComparisonOutcome.SOME_COMPATIBLE
    else:
        outcome = ComparisonOutcome.DISAGREEMENT
    bests = [r.best for r in reports]
    same_best = None if len(reports) < 2 else (len(set(bests)) == 1 and bests[0] is not None)
    if same_best is False:
        warnings.append("I metodi indicano frasi diverse come la più compatibile.")
    unalignable = [e.candidate for r in reports for e in r.entries if e.score is None]
    if unalignable:
        warnings.append("Frasi non valutabili da almeno un metodo: " + "; ".join(sorted(set(unalignable))))
    return CandidateComparison(
        candidates=cands, reports=reports, outcome=outcome, same_best=same_best,
        message=_MESSAGES[outcome], warnings=warnings,
    )
