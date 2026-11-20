# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Voiceprint score normalization: the WeSpeaker scoring recipe (issue #57).

Steps, in the order WeSpeaker applies them (`wespeaker/bin/score.py`,
`score_norm.py`, `score_calibration.py`):

1. **Mean subtraction.** The mean of the impostor cohort embeddings is
   subtracted from every embedding (subject, test segment, cohort) before the
   cosine (WeSpeaker `cal_mean`). Embeddings are L2-normalized first, so
   backends with different embedding norms are treated alike; centroids are
   the normalized mean of the transformed subject embeddings.
2. **AS-norm** (adaptive symmetric normalization, Matějka et al. 2017; WeSpeaker
   `asnorm`): with the cosines of the enrollment side (condition centroid) and
   of the test side against the cohort, keep the `top_k` highest of each and
   ``s' = 0.5 * ((s - mu_e) / sd_e + (s - mu_t) / sd_t)`` (population std).
3. **QMF** (quality measure functions): logistic regression over the AS-norm
   score and quality measures - log duration of the test segment and the
   "imposter mean" (mean of the top-k cohort scores) of test and enrollment
   side, as in the WeSpeaker VoxSRC-23 system. Fitted on the
   leave-one-recording-out trials of the training material; its output (a
   log-odds-like score) then goes to the existing calibration
   (`profile.calibrate_scores`).

The cohort is configurable (`ScoringRecipe.cohort_sources`): labelled
non-subject speakers of the case training recordings, the operator's extra
other-speaker clips, and/or cohort-only clips (e.g. a public Italian cohort
computed offline). Every parameter, the cohort composition and the fitted
weights are saved in the `ScoreNormalization` artifact.

The result is still a similarity score for triage, never an identification
(CLAUDE.md invariant 6).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, Field

NORMALIZATION_SCHEMA = "akoe.score-normalization/v1"
COHORT_SOURCES = ("case_other_speakers", "extra_other", "cohort_files")
QMF_FEATURES = ("score", "log_duration_test", "imposter_mean_test", "imposter_mean_enroll", "log_duration_enroll")
_STD_FLOOR = 1e-6


class ScoringRecipe(BaseModel):
    """Parameters of the scoring recipe (all recorded with the calibration)."""

    mean_subtraction: bool = True
    as_norm: bool = True
    top_k: int = Field(300, ge=1)  # WeSpeaker VoxCeleb recipe: top_n = 300
    min_cohort: int = Field(10, ge=2)  # below this cohort size, mean subtraction and AS-norm are skipped
    cohort_sources: list[str] = Field(default_factory=lambda: list(COHORT_SOURCES))
    qmf: bool = True
    qmf_features: list[str] = Field(
        default_factory=lambda: ["score", "log_duration_test", "imposter_mean_test", "imposter_mean_enroll"])
    qmf_min_per_class: int = Field(20, ge=2)  # fewer trials: QMF skipped (a 4-feature fit would overfit)
    qmf_C: float = Field(1.0, gt=0)

    def model_post_init(self, __context: Any) -> None:
        bad = [s for s in self.cohort_sources if s not in COHORT_SOURCES]
        if bad:
            raise ValueError(f"Unknown cohort sources {bad}; allowed: {list(COHORT_SOURCES)}")
        badf = [f for f in self.qmf_features if f not in QMF_FEATURES]
        if badf or "score" not in self.qmf_features:
            raise ValueError(f"qmf_features must include 'score' and be among {list(QMF_FEATURES)}")


DEFAULT_SCORING_RECIPE = ScoringRecipe()


def unit(v: np.ndarray) -> np.ndarray | None:
    v = np.asarray(v, dtype=np.float64).ravel()
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n == 0.0:
        return None
    return v / n


def unit_rows(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def topk_stats(v: np.ndarray, cohort: np.ndarray, k: int) -> tuple[float, float]:
    """Mean and population std of the `k` highest cosines of unit `v` with the
    unit rows of `cohort` (WeSpeaker `get_mean_std`)."""
    s = np.sort(cohort @ v)[::-1][: max(1, min(k, cohort.shape[0]))]
    return float(s.mean()), float(s.std())


def as_norm(score: float, enroll: tuple[float, float], test: tuple[float, float]) -> float:
    (me, se), (mt, st) = enroll, test
    return 0.5 * ((score - me) / max(se, _STD_FLOOR) + (score - mt) / max(st, _STD_FLOOR))


@dataclass
class CohortEntry:
    embedding: np.ndarray  # unit-normalized, as produced by the backend
    group: str  # recording group (pair or recording) or "extra:i" / "cohort:i"
    source: str  # one of COHORT_SOURCES
    seconds: float
    label: str = ""


class QMFModel(BaseModel):
    """Logistic regression over standardized quality features; `decision()` is
    the log-odds of the subject class (balanced classes)."""

    features: list[str]
    coef: list[float]
    intercept: float
    mean: list[float]
    std: list[float]
    C: float
    n_same: int
    n_other: int
    solver: str = "lbfgs"
    class_weight: str = "balanced"

    def decision(self, feats: dict[str, float]) -> float:
        x = np.array([feats[f] for f in self.features], dtype=np.float64)
        z = (x - np.array(self.mean)) / np.array(self.std)
        return float(z @ np.array(self.coef) + self.intercept)


class NormScore(BaseModel):
    score: float  # final score: QMF if fitted, else AS-norm, else (mean-subtracted) cosine
    kind: str  # "qmf" | "asnorm" | "cosine"
    condition_key: str
    cosine: float  # cosine in the transformed space for the chosen condition
    asnorm: float | None = None
    by_condition: dict[str, float] = Field(default_factory=dict)  # pre-QMF score per condition
    features: dict[str, float] = Field(default_factory=dict)


class ScoreNormalization(BaseModel):
    """Everything needed to reproduce normalized scores (saved as an artifact)."""

    schema_: str = Field(default=NORMALIZATION_SCHEMA, alias="schema")
    backend: str
    subject_id: str
    recipe: ScoringRecipe
    steps: list[str]  # applied steps, in order
    mean: list[float] | None = None
    cohort: list[list[float]] = Field(default_factory=list)  # unit embeddings (before mean subtraction)
    cohort_groups: list[str] = Field(default_factory=list)
    cohort_sources: list[str] = Field(default_factory=list)
    cohort_labels: list[str] = Field(default_factory=list)
    centroids: dict[str, list[float]] = Field(default_factory=dict)  # transformed space, per condition
    enroll_seconds: dict[str, float] = Field(default_factory=dict)
    subject_items: list[str] = Field(default_factory=list)
    qmf: QMFModel | None = None
    effective_top_k: int | None = None
    warnings: list[str] = Field(default_factory=list)  # Italian
    material: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}

    @property
    def score_kind(self) -> str:
        return "qmf" if self.qmf is not None else ("asnorm" if "as_norm" in self.steps else "cosine")


@dataclass
class ScoreNormalizer:
    """Fitted mean / cohort / centroids; scores unit embeddings of test segments."""

    recipe: ScoringRecipe
    centroids: dict[str, np.ndarray]  # transformed space, unit
    enroll_seconds: dict[str, float]
    mean: np.ndarray | None = None
    cohort: np.ndarray | None = None  # transformed space, unit rows
    qmf: QMFModel | None = None
    enroll_stats: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def steps(self) -> list[str]:
        out = []
        if self.mean is not None:
            out.append("mean_subtraction")
        if self.cohort is not None:
            out.append("as_norm")
        if self.qmf is not None:
            out.append("qmf")
        return out

    def transform(self, v: np.ndarray) -> np.ndarray:
        u = unit(v)
        if u is None:
            raise ValueError("embedding is zero or not finite")
        if self.mean is None:
            return u
        t = unit(u - self.mean)
        if t is None:
            raise ValueError("embedding equals the cohort mean")
        return t

    @classmethod
    def fit(
        cls,
        recipe: ScoringRecipe,
        subject: Sequence[tuple[str, np.ndarray, float]],
        cohort: Sequence[CohortEntry],
        *,
        use_mean: bool,
        use_as_norm: bool,
        qmf: QMFModel | None = None,
    ) -> ScoreNormalizer | None:
        """`subject`: (condition key, unit embedding, seconds). Returns None when
        no centroid can be built or AS-norm is requested with < 2 cohort entries."""
        if not subject:
            return None
        coh = unit_rows(np.stack([c.embedding for c in cohort])) if cohort else None
        mean = None
        if use_mean:
            if coh is None:
                return None
            mean = coh.mean(axis=0)
        norm = cls(recipe=recipe, centroids={}, enroll_seconds={}, mean=mean, qmf=qmf)
        groups: dict[str, list[np.ndarray]] = {}
        secs: dict[str, float] = {}
        for key, v, s in subject:
            groups.setdefault(key, []).append(norm.transform(v))
            secs[key] = secs.get(key, 0.0) + float(s)
        for key in sorted(groups):
            c = unit(np.stack(groups[key]).mean(axis=0))
            if c is not None:
                norm.centroids[key] = c
                norm.enroll_seconds[key] = round(secs[key], 3)
        if not norm.centroids:
            return None
        if use_as_norm:
            if coh is None or coh.shape[0] < 2:
                return None
            norm.cohort = coh if mean is None else unit_rows(coh - mean)
            norm.enroll_stats = {k: topk_stats(c, norm.cohort, recipe.top_k) for k, c in norm.centroids.items()}
        return norm

    def score(self, v: np.ndarray, seconds: float | None = None, condition_key: str | None = None) -> NormScore:
        t = self.transform(v)
        cos = {k: float(t @ c) for k, c in self.centroids.items()}
        test_stats = topk_stats(t, self.cohort, self.recipe.top_k) if self.cohort is not None else None
        if test_stats is not None:
            pre = {k: as_norm(cos[k], self.enroll_stats[k], test_stats) for k in cos}
        else:
            pre = dict(cos)
        if condition_key is not None:
            if condition_key not in pre:
                raise KeyError(condition_key)
            best = condition_key
        else:
            best = max(sorted(pre), key=lambda k: pre[k])
        feats: dict[str, float] = {"score": pre[best]}
        if seconds is not None and seconds > 0:
            feats["log_duration_test"] = math.log(seconds)
        if self.enroll_seconds.get(best):
            feats["log_duration_enroll"] = math.log(self.enroll_seconds[best])
        if test_stats is not None:
            feats["imposter_mean_test"] = test_stats[0]
            feats["imposter_mean_enroll"] = self.enroll_stats[best][0]
        kind = "asnorm" if test_stats is not None else "cosine"
        final = pre[best]
        if self.qmf is not None:
            missing = [f for f in self.qmf.features if f not in feats]
            if missing:
                raise ValueError(f"QMF features unavailable for this segment: {missing}")
            final, kind = self.qmf.decision(feats), "qmf"
        return NormScore(score=final, kind=kind, condition_key=best, cosine=cos[best],
                         asnorm=pre[best] if test_stats is not None else None,
                         by_condition=pre, features=feats)

    # ------------------------------------------------------------ artifact
    def to_artifact(self, *, backend: str, subject_id: str, cohort: Sequence[CohortEntry],
                    subject_items: Sequence[str], warnings: Sequence[str] = (),
                    material: dict[str, Any] | None = None) -> ScoreNormalization:
        entries = list(cohort) if (self.cohort is not None or self.mean is not None) else []
        return ScoreNormalization(
            backend=backend, subject_id=subject_id, recipe=self.recipe, steps=self.steps,
            mean=None if self.mean is None else [float(x) for x in self.mean],
            cohort=[[float(x) for x in unit(c.embedding)] for c in entries],
            cohort_groups=[c.group for c in entries], cohort_sources=[c.source for c in entries],
            cohort_labels=[c.label for c in entries],
            centroids={k: [float(x) for x in c] for k, c in self.centroids.items()},
            enroll_seconds=dict(self.enroll_seconds), subject_items=list(subject_items), qmf=self.qmf,
            effective_top_k=None if self.cohort is None else min(self.recipe.top_k, self.cohort.shape[0]),
            warnings=list(warnings), material=material or {},
        )

    @classmethod
    def from_artifact(cls, art: ScoreNormalization) -> ScoreNormalizer:
        mean = None if art.mean is None else np.asarray(art.mean, dtype=np.float64)
        norm = cls(recipe=art.recipe, centroids={k: np.asarray(v, dtype=np.float64) for k, v in art.centroids.items()},
                   enroll_seconds=dict(art.enroll_seconds), mean=mean, qmf=art.qmf)
        if "as_norm" in art.steps:
            coh = unit_rows(np.asarray(art.cohort, dtype=np.float64))
            norm.cohort = coh if mean is None else unit_rows(coh - mean)
            norm.enroll_stats = {k: topk_stats(c, norm.cohort, art.recipe.top_k) for k, c in norm.centroids.items()}
        return norm


def fit_qmf(recipe: ScoringRecipe, trials: Sequence[tuple[int, NormScore]]) -> tuple[QMFModel | None, str | None]:
    """Fit the QMF logistic regression on labelled trials (1 = subject).
    Returns (model, None) or (None, reason in Italian)."""
    from sklearn.linear_model import LogisticRegression

    if not recipe.qmf:
        return None, None
    feats = [f for f in recipe.qmf_features if all(f in ns.features for _, ns in trials)]
    if not trials or feats[:1] != ["score"]:
        return None, "QMF non applicata: punteggi mancanti."
    y = np.array([lab for lab, _ in trials], dtype=np.int64)
    n_same, n_other = int(y.sum()), int((1 - y).sum())
    if min(n_same, n_other) < recipe.qmf_min_per_class:
        return None, (f"QMF non applicata: servono almeno {recipe.qmf_min_per_class} prove per classe "
                      f"({n_same} del soggetto, {n_other} di altri parlanti); si usa il punteggio "
                      "normalizzato senza misure di qualità.")
    dropped = [f for f in recipe.qmf_features if f not in feats]
    x = np.array([[ns.features[f] for f in feats] for _, ns in trials], dtype=np.float64)
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd == 0] = 1.0
    lr = LogisticRegression(C=recipe.qmf_C, class_weight="balanced", solver="lbfgs", max_iter=1000)
    lr.fit((x - mu) / sd, y)
    model = QMFModel(features=feats, coef=[float(c) for c in lr.coef_[0]], intercept=float(lr.intercept_[0]),
                     mean=[float(m) for m in mu], std=[float(s) for s in sd], C=recipe.qmf_C,
                     n_same=n_same, n_other=n_other)
    if model.coef[0] <= 0:
        return None, "QMF non applicata: il punteggio non separa il soggetto dagli altri parlanti."
    note = (f"QMF senza le misure non disponibili: {', '.join(dropped)}." if dropped else None)
    return model, note
