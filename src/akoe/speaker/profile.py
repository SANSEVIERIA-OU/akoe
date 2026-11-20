# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Subject profile across contexts (IMPLEMENTATION_PLAN M1, issue #10).

- `build_speaker_profile`: one `ConditionProfile` per `RecordingCondition.key()`
  from the training items of a split `SubjectDataset`: a voiceprint centroid
  (from any `VoiceprintBackend`; the model is chosen elsewhere) and a timbre
  profile (`timbre.py`). Versions are immutable files with SHA-256; a build
  record next to each profile lists the items, parameters and warnings.
- `ProfileScorer`: cosine similarity of a segment's embedding with the
  closest condition centroid. Every score states the condition it refers to
  (MASTER_PLAN §3.5.3, §4.2).
- `calibrate_scores`: logistic regression on case material (subject segments
  vs other speakers) and a calibrated threshold below which a segment is
  flagged "possibile altro parlante". The scores come from
  `loro_calibration_scores` (leave-one-recording-out on training material,
  issue #49), so the self-assessment holdout never enters the calibration;
  `calibration_scores` (holdout by default) is kept for compatibility.

The voiceprint is a similarity score for triage, selection and warnings. It
is never an attribution of the voice to a person (CLAUDE.md invariant 6):
outputs only ever speak of scores and similarity.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

import numpy as np
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression

from akoe.case import Case
from akoe.config import StageKind
from akoe.interfaces import VoiceprintBackend
from akoe.models import (
    ArtifactRef,
    ConditionProfile,
    DatasetItem,
    DatasetSplit,
    RecordingCondition,
    SegmentFlag,
    SegmentHypotheses,
    SpeakerProfile,
)
from akoe.speaker._io import (
    cut,
    dump_json_bytes,
    load_audio,
    read_verified_json,
    subject_dir,
    write_content_addressed,
)
from akoe.speaker.dataset import DatasetError, SubjectDataset
from akoe.speaker.timbre import TimbreParams, TimbreSource, compute_timbre

# Embeddings come from fixed weights: deterministic given the backend.
KIND = StageKind.DETERMINISTIC
CENTROID_SCHEMA = "akoe.voiceprint-centroid/v1"
BUILD_SCHEMA = "akoe.speaker-profile-build/v1"
CALIBRATION_SCHEMA = "akoe.score-calibration/v1"
SCORE_NOTE = "Punteggio di somiglianza per smistamento e avvisi, non una prova di attribuzione."


class ProfileBuildConfig(BaseModel):
    analysis_sr: int = 16000
    verified_only: bool = True  # only items marked 'certain' by the transcriber
    include_unaligned: bool = False  # whole-recording items of untimed transcriptions
    min_embed_seconds: float = 0.5
    compute_timbre: bool = True
    timbre: TimbreParams = Field(default_factory=TimbreParams)


class ProfileBuildResult(BaseModel):
    profile: SpeakerProfile
    profile_ref: ArtifactRef
    build_record_ref: ArtifactRef
    warnings: list[str] = Field(default_factory=list)


def _unit(v: np.ndarray) -> np.ndarray | None:
    v = np.asarray(v, dtype=np.float64).ravel()
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n == 0.0:
        return None
    return v / n


def _key_condition(c: RecordingCondition) -> RecordingCondition:
    """Only the fields that make up the condition key."""
    return RecordingCondition(channel=c.channel, vocal_effort=c.vocal_effort, source=c.source, codec=c.codec)


def profile_path(case: Case, subject_id: str, version: int):
    return subject_dir(case, subject_id) / f"profile-v{version:04d}.json"


def _write_immutable(path, data: Any) -> None:
    payload = dump_json_bytes(data)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise DatasetError(f"La versione {path.name} esiste già: le versioni del profilo non si sovrascrivono")
    path.write_bytes(payload)


class _AudioCache:
    def __init__(self, case: Case, dataset: SubjectDataset, sr: int) -> None:
        self.case, self.dataset, self.sr = case, dataset, sr
        self._audio: dict[str, np.ndarray] = {}

    def recording(self, recording_id: str) -> np.ndarray:
        if recording_id not in self._audio:
            rec = self.dataset.recording(recording_id)
            if not rec.audio.verify(self.case.root):
                raise DatasetError(f"{rec.original_name}: audio mancante o modificato (SHA-256)")
            self._audio[recording_id] = load_audio(rec.audio.resolve(self.case.root), self.sr)
        return self._audio[recording_id]

    def segment(self, recording_id: str, start: float, end: float) -> np.ndarray:
        return cut(self.recording(recording_id), self.sr, start, end)

    def native(self, recording_id: str) -> tuple[np.ndarray, int]:
        """Audio at its own sample rate (timbre must know the real bandwidth)."""
        rec = self.dataset.recording(recording_id)
        if not rec.audio.verify(self.case.root):
            raise DatasetError(f"{rec.original_name}: audio mancante o modificato (SHA-256)")
        return load_audio(rec.audio.resolve(self.case.root), rec.sample_rate), rec.sample_rate


def build_speaker_profile(
    case: Case,
    dataset: SubjectDataset,
    backend: VoiceprintBackend,
    *,
    version: int = 1,
    config: ProfileBuildConfig | None = None,
    parent: ArtifactRef | None = None,
    items: Sequence[DatasetItem] | None = None,
) -> ProfileBuildResult:
    """Build and save version `version` of the subject profile.

    By default uses the TRAIN items selected by `config` (verified, aligned,
    not blocked by quality errors); holdout material never enters the profile.
    """
    cfg = config or ProfileBuildConfig()
    if not dataset.is_split:
        raise DatasetError(
            "Dividere il dataset in addestramento e autovalutazione prima di costruire il profilo"
        )
    if items is None:
        items = dataset.select_items(
            splits=(DatasetSplit.TRAIN,), verified_only=cfg.verified_only,
            include_unaligned=cfg.include_unaligned,
        )
    if any(it.split != DatasetSplit.TRAIN for it in items):
        raise DatasetError("Il profilo si costruisce solo con segmenti della quota di addestramento")
    if not items:
        raise DatasetError(
            "Nessun segmento utilizzabile per il profilo (servono segmenti di addestramento "
            + ("verificati come certi, " if cfg.verified_only else "")
            + "allineati e senza errori di qualità)"
        )
    warnings: list[str] = []
    audio = _AudioCache(case, dataset, cfg.analysis_sr)
    sdir = subject_dir(case, dataset.subject_id)

    by_condition: dict[str, list[DatasetItem]] = {}
    for it in sorted(items, key=lambda it: it.item_id):
        by_condition.setdefault(it.condition.key(), []).append(it)

    conditions: list[ConditionProfile] = []
    used_ids: list[str] = []
    for key in sorted(by_condition):
        group = by_condition[key]
        embeds: list[np.ndarray] = []
        emb_ids: list[str] = []
        for it in group:
            if it.end - it.start < cfg.min_embed_seconds:
                warnings.append(
                    f"Segmento {it.item_id}: più breve di {cfg.min_embed_seconds:.1f} s, escluso dall'impronta"
                )
                continue
            v = _unit(backend.embed(audio.segment(it.recording_id, it.start, it.end), cfg.analysis_sr))
            if v is None:
                warnings.append(f"Segmento {it.item_id}: impronta non valida (vettore nullo), escluso")
                continue
            embeds.append(v)
            emb_ids.append(it.item_id)

        voiceprint_ref = None
        if embeds:
            mat = np.stack(embeds)
            centroid = _unit(mat.mean(axis=0))
            assert centroid is not None
            sims = mat @ centroid
            voiceprint_ref = write_content_addressed(
                case, sdir / "voiceprints", "centroid",
                {
                    "schema": CENTROID_SCHEMA,
                    "backend": backend.name,
                    "condition_key": key,
                    "dim": int(centroid.size),
                    "centroid": [float(x) for x in centroid],
                    "n_embeddings": len(embeds),
                    "item_ids": emb_ids,
                    "within_condition_similarity": {
                        "min": float(sims.min()), "median": float(np.median(sims)), "max": float(sims.max()),
                    },
                    "analysis_sr": cfg.analysis_sr,
                },
                kind="voiceprint_centroid",
            )
        else:
            warnings.append(f"Condizione {key}: nessun segmento valido per l'impronta vocale")

        timbre: dict[str, Any] = {}
        if cfg.compute_timbre:
            spans: dict[str, list[tuple[float, float]]] = {}
            for it in group:
                spans.setdefault(it.recording_id, []).append((it.start, it.end))
            sources = []
            for rid, s in sorted(spans.items()):
                y, native_sr = audio.native(rid)
                sources.append(TimbreSource(y, native_sr, spans=s, label=rid))
            timbre = compute_timbre(sources, cfg.timbre)
            warnings.extend(f"Condizione {key}: {w}" for w in timbre.get("warnings", []))

        conditions.append(
            ConditionProfile(
                condition=_key_condition(group[0].condition),
                seconds=round(sum(it.end - it.start for it in group), 3),
                n_segments=len(group),
                voiceprint=voiceprint_ref,
                timbre=timbre,
            )
        )
        used_ids.extend(it.item_id for it in group)

    manifest_ref = dataset.write_manifest(case)
    profile = SpeakerProfile(
        subject_id=dataset.subject_id,
        version=version,
        conditions=conditions,
        dataset_manifest=manifest_ref,
        holdout_recordings=dataset.holdout_recordings(),
    )
    path = profile_path(case, dataset.subject_id, version)
    _write_immutable(path, profile.model_dump(mode="json"))
    profile_ref = ArtifactRef.from_file(path, kind="speaker_profile", root=case.root)
    record = {
        "schema": BUILD_SCHEMA,
        "kind": KIND.value,
        "subject_id": dataset.subject_id,
        "version": version,
        "profile": profile_ref.model_dump(mode="json"),
        "parent_profile": parent.model_dump(mode="json") if parent else None,
        "backend": backend.name,
        "config": cfg.model_dump(mode="json"),
        "dataset_manifest": manifest_ref.model_dump(mode="json"),
        "item_ids": used_ids,
        "warnings": warnings,
        "note": SCORE_NOTE,
    }
    record_path = path.with_name(path.stem + ".build.json")
    _write_immutable(record_path, record)
    record_ref = ArtifactRef.from_file(record_path, kind="speaker_profile_build", root=case.root)
    return ProfileBuildResult(profile=profile, profile_ref=profile_ref, build_record_ref=record_ref, warnings=warnings)


def load_profile(case: Case, ref: ArtifactRef) -> SpeakerProfile:
    return SpeakerProfile.model_validate(read_verified_json(case, ref))


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def _summary(x: Sequence[float]) -> dict[str, float]:
    a = np.asarray(x, dtype=np.float64)
    return {"n": int(a.size), "min": float(a.min()), "median": float(np.median(a)),
            "mean": float(a.mean()), "max": float(a.max())}


def equal_error_rate(same: Sequence[float], other: Sequence[float]) -> tuple[float, float]:
    """(EER, threshold) sweeping thresholds over observed scores; flag = score < threshold."""
    s, o = np.asarray(same, dtype=np.float64), np.asarray(other, dtype=np.float64)
    best = (1.0, float("nan"), float("inf"))
    for t in np.unique(np.concatenate([s, o])):
        frr = float(np.mean(s < t))  # subject flagged
        far = float(np.mean(o >= t))  # other speaker not flagged
        gap = abs(frr - far)
        if gap < best[2]:
            best = ((frr + far) / 2, float(t), gap)
    return best[0], best[1]


class ScoreCalibration(BaseModel):
    """Logistic calibration of similarity scores on the case material.

    `probability(score)` is the calibrated probability that the segment comes
    from the subject, with equal priors (balanced classes). Below
    `threshold_score` a segment is flagged "possibile altro parlante".
    """

    schema_: str = Field(default=CALIBRATION_SCHEMA, alias="schema")
    method: str = "logistic_regression"
    backend: str
    subject_id: str
    profile_version: int
    material: str
    coef: float
    intercept: float
    standardization: dict[str, float]
    threshold_probability: float
    threshold_score: float | None
    reliable: bool
    n_same: int
    n_other: int
    same_scores: dict[str, float]
    other_scores: dict[str, float]
    rates_at_threshold: dict[str, float] = Field(default_factory=dict)
    eer: float
    eer_threshold: float
    params: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    # How the subject scores were obtained (issue #49): "leave-one-recording-out"
    # on training material keeps the self-assessment holdout out of the
    # calibration; "holdout" (the old default of `calibration_scores`) makes the
    # EER measured later on the same holdout optimistic.
    scheme: str = "unspecified"
    holdout_used: bool | None = None

    model_config = {"populate_by_name": True}

    def probability(self, score: float) -> float:
        return float(1.0 / (1.0 + np.exp(-(self.coef * score + self.intercept))))

    def is_possible_other_speaker(self, score: float) -> bool | None:
        if not self.reliable or self.threshold_score is None:
            return None
        return bool(score < self.threshold_score)


def calibrate_scores(
    same: Sequence[float],
    other: Sequence[float],
    *,
    backend: str,
    profile: SpeakerProfile,
    material: str = "",
    threshold_probability: float = 0.5,
    C: float = 1.0,
    min_per_class: int = 10,
    scheme: str = "unspecified",
    holdout_used: bool | None = None,
) -> ScoreCalibration:
    """Fit P(subject | score) with scikit-learn's LogisticRegression on standardized scores.

    `scheme` / `holdout_used` record where the scores come from (see
    `loro_calibration_scores`); a calibration on holdout material gets a warning.
    """
    if len(same) < 2 or len(other) < 2:
        raise DatasetError(
            "Calibrazione impossibile: servono almeno 2 punteggi del soggetto e 2 di altri parlanti"
        )
    if not 0.0 < threshold_probability < 1.0:
        raise ValueError("threshold_probability must be in (0, 1)")
    warnings: list[str] = []
    s = np.asarray(same, dtype=np.float64)
    o = np.asarray(other, dtype=np.float64)
    x = np.concatenate([s, o])
    y = np.concatenate([np.ones(s.size), np.zeros(o.size)])
    mu, sd = float(x.mean()), float(x.std()) or 1.0
    lr = LogisticRegression(C=C, class_weight="balanced", solver="lbfgs", max_iter=1000)
    lr.fit(((x - mu) / sd).reshape(-1, 1), y)
    cz, iz = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    coef, intercept = cz / sd, iz - cz * mu / sd

    reliable = coef > 0
    threshold = None
    rates: dict[str, float] = {}
    if reliable:
        threshold = (float(np.log(threshold_probability / (1 - threshold_probability))) - intercept) / coef
        rates = {
            "subject_flagged": float(np.mean(s < threshold)),
            "other_not_flagged": float(np.mean(o >= threshold)),
        }
    else:
        warnings.append(
            "I punteggi del soggetto non risultano più alti di quelli degli altri parlanti: "
            "calibrazione non affidabile, nessun avviso automatico"
        )
    if min(s.size, o.size) < min_per_class:
        warnings.append(
            f"Pochi punteggi per la calibrazione ({s.size} del soggetto, {o.size} di altri parlanti; "
            f"consigliati almeno {min_per_class} per classe): soglia indicativa"
        )
    if holdout_used:
        warnings.append(
            "Calibrazione fatta sulla quota di autovalutazione: l'errore dell'impronta misurato poi "
            "sulla stessa quota risulterà ottimistico"
        )
    eer, eer_t = equal_error_rate(s, o)
    return ScoreCalibration(
        backend=backend,
        subject_id=profile.subject_id,
        profile_version=profile.version,
        material=material,
        coef=coef,
        intercept=intercept,
        standardization={"mean": mu, "std": sd},
        threshold_probability=threshold_probability,
        threshold_score=threshold,
        reliable=reliable,
        n_same=int(s.size),
        n_other=int(o.size),
        same_scores=_summary(s),
        other_scores=_summary(o),
        rates_at_threshold=rates,
        eer=eer,
        eer_threshold=eer_t,
        params={"C": C, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 1000,
                "features": "standardized similarity score"},
        warnings=warnings,
        scheme=scheme,
        holdout_used=holdout_used,
    )


def save_calibration(case: Case, calibration: ScoreCalibration) -> ArtifactRef:
    return write_content_addressed(
        case, subject_dir(case, calibration.subject_id) / "calibration", "calibration",
        calibration.model_dump(mode="json", by_alias=True), kind="score_calibration",
    )


def load_calibration(case: Case, ref: ArtifactRef) -> ScoreCalibration:
    return ScoreCalibration.model_validate(read_verified_json(case, ref))


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


class SegmentScore(BaseModel):
    """Similarity of one segment with the subject profile. Always states the
    reference condition; it is a score, not an attribution."""

    score: float  # cosine similarity with the closest condition centroid
    condition_key: str  # condition the score refers to
    scores_by_condition: dict[str, float]
    calibrated_probability: float | None = None
    possible_other_speaker: bool | None = None  # None = no reliable calibration
    backend: str
    profile_version: int
    message: str


class ProfileScorer:
    def __init__(
        self,
        profile: SpeakerProfile,
        centroids: dict[str, np.ndarray],
        backend: VoiceprintBackend,
        calibration: ScoreCalibration | None = None,
        analysis_sr: int = 16000,
    ) -> None:
        if not centroids:
            raise DatasetError("Il profilo non contiene impronte vocali utilizzabili")
        if calibration is not None and calibration.backend != backend.name:
            raise DatasetError(
                f"Calibrazione fatta con '{calibration.backend}', modello attuale '{backend.name}': "
                "punteggi non confrontabili"
            )
        self.profile = profile
        self.centroids = {k: c for k, c in sorted(centroids.items())}
        self.backend = backend
        self.calibration = calibration
        self.analysis_sr = analysis_sr

    @classmethod
    def load(
        cls,
        case: Case,
        profile: SpeakerProfile,
        backend: VoiceprintBackend,
        calibration: ScoreCalibration | None = None,
    ) -> "ProfileScorer":
        """Load centroids after checking their SHA-256 and that they come from `backend`."""
        centroids: dict[str, np.ndarray] = {}
        sr = 16000
        for cp in profile.conditions:
            if cp.voiceprint is None:
                continue
            data = read_verified_json(case, cp.voiceprint)
            if data.get("backend") != backend.name:
                raise DatasetError(
                    f"Impronta della condizione {cp.condition.key()} calcolata con '{data.get('backend')}', "
                    f"modello attuale '{backend.name}': punteggi non confrontabili"
                )
            centroids[data["condition_key"]] = np.asarray(data["centroid"], dtype=np.float64)
            sr = int(data.get("analysis_sr", sr))
        return cls(profile, centroids, backend, calibration, analysis_sr=sr)

    def with_calibration(self, calibration: ScoreCalibration) -> "ProfileScorer":
        return ProfileScorer(self.profile, self.centroids, self.backend, calibration, self.analysis_sr)

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray | None:
        return _unit(self.backend.embed(audio, sr))

    def score_embedding(self, embedding: np.ndarray, condition_key: str | None = None) -> SegmentScore:
        v = _unit(embedding)
        if v is None:
            raise ValueError("embedding is zero or not finite")
        if condition_key is not None and condition_key not in self.centroids:
            raise DatasetError(f"Condizione {condition_key} assente dal profilo")
        scores = {k: float(np.dot(v, c)) for k, c in self.centroids.items()}
        candidates = {condition_key: scores[condition_key]} if condition_key else scores
        best_key = max(sorted(candidates), key=lambda k: candidates[k])
        best = scores[best_key]
        prob = flag = None
        msg = f"Somiglianza {best:.3f} rispetto alla condizione {best_key}."
        if self.calibration is not None:
            prob = self.calibration.probability(best)
            flag = self.calibration.is_possible_other_speaker(best)
            if flag:
                msg += (f" Possibile altro parlante: sotto la soglia calibrata "
                        f"{self.calibration.threshold_score:.3f}.")
            elif flag is None:
                msg += " Calibrazione non affidabile: nessun avviso automatico."
            else:
                msg += f" Sopra la soglia calibrata {self.calibration.threshold_score:.3f}."
        else:
            msg += " Punteggio non calibrato: nessun avviso automatico."
        return SegmentScore(
            score=best, condition_key=best_key, scores_by_condition=scores,
            calibrated_probability=prob, possible_other_speaker=flag,
            backend=self.backend.name, profile_version=self.profile.version,
            message=f"{msg} {SCORE_NOTE}",
        )

    def score(self, audio: np.ndarray, sr: int, condition_key: str | None = None) -> SegmentScore:
        v = self.embed(audio, sr)
        if v is None:
            raise ValueError("backend returned a zero or non-finite embedding")
        return self.score_embedding(v, condition_key)


def annotate_segment(segment: SegmentHypotheses, score: SegmentScore) -> SegmentHypotheses:
    """Copy of `segment` with speaker_score, speaker_condition and, when flagged,
    SegmentFlag.POSSIBLE_OTHER_SPEAKER."""
    flags = list(segment.flags)
    if score.possible_other_speaker and SegmentFlag.POSSIBLE_OTHER_SPEAKER not in flags:
        flags.append(SegmentFlag.POSSIBLE_OTHER_SPEAKER)
    return segment.model_copy(
        update={"speaker_score": score.score, "speaker_condition": score.condition_key, "flags": flags}
    )


def calibration_scores(
    case: Case,
    dataset: SubjectDataset,
    scorer: ProfileScorer,
    *,
    subject_splits: Iterable[DatasetSplit] = (DatasetSplit.HOLDOUT,),
    extra_other: Sequence[tuple[np.ndarray, int]] = (),
    min_seconds: float = 0.5,
) -> tuple[list[float], list[float], str]:
    """Scores of the subject's segments (by default the HOLDOUT ones, never seen
    by the centroids) and of other speakers' labelled spans in the case
    material, plus optional extra other-speaker audio `(audio, sr)`.

    Returns (subject scores, other-speaker scores, description of the material).

    Prefer `loro_calibration_scores` (issue #49): calibrating on the holdout
    makes the self-assessment of the voiceprint on that same holdout
    optimistic. The description records `holdout_used`.
    """
    splits = tuple(subject_splits)
    sr = scorer.analysis_sr
    audio = _AudioCache(case, dataset, sr)
    same: list[float] = []
    for it in dataset.select_items(splits=splits, verified_only=False):
        if it.end - it.start < min_seconds:
            continue
        v = scorer.embed(audio.segment(it.recording_id, it.start, it.end), sr)
        if v is not None:
            same.append(scorer.score_embedding(v).score)
    other: list[float] = []
    for span in dataset.other_speaker_spans:
        if span.end - span.start < min_seconds:
            continue
        v = scorer.embed(audio.segment(span.recording_id, span.start, span.end), sr)
        if v is not None:
            other.append(scorer.score_embedding(v).score)
    for a, a_sr in extra_other:
        v = scorer.embed(np.asarray(a, dtype=np.float32), a_sr)
        if v is not None:
            other.append(scorer.score_embedding(v).score)
    material = json.dumps(
        {
            "subject_segments": f"items in splits {[s.value for s in splits]}",
            "other_speakers": f"{len(dataset.other_speaker_spans)} labelled spans + {len(extra_other)} extra clips",
            "scheme": "fixed centroids",
            "holdout_used": DatasetSplit.HOLDOUT in splits,
        },
        sort_keys=True,
    )
    return same, other, material


LORO_SCHEME = "leave-one-recording-out"


class LOROScores(BaseModel):
    """Subject and other-speaker scores for a leave-one-recording-out calibration."""

    same: list[float]
    other: list[float]
    material: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)  # Italian

    def material_json(self) -> str:
        return json.dumps(self.material, sort_keys=True)


def _recording_group(dataset: SubjectDataset, recording_id: str) -> str:
    rec = dataset.recording(recording_id)
    return f"pair:{rec.pair_id}" if rec.pair_id else f"rec:{rec.recording_id}"


def _centroids(embeds: Sequence[tuple[str, np.ndarray]]) -> dict[str, np.ndarray]:
    groups: dict[str, list[np.ndarray]] = {}
    for key, v in embeds:
        groups.setdefault(key, []).append(v)
    out = {}
    for key in sorted(groups):
        c = _unit(np.stack(groups[key]).mean(axis=0))
        if c is not None:
            out[key] = c
    return out


def _best(v: np.ndarray, centroids: dict[str, np.ndarray]) -> float:
    return max(float(np.dot(v, c)) for c in centroids.values())


def loro_calibration_scores(
    case: Case,
    dataset: SubjectDataset,
    backend: VoiceprintBackend,
    *,
    items: Sequence[DatasetItem] | None = None,
    analysis_sr: int = 16000,
    min_seconds: float = 0.5,
    extra_other: Sequence[tuple[np.ndarray, int]] = (),
) -> LOROScores:
    """Leave-one-recording-out scores on TRAINING material only (issue #49).

    For each recording group (a `pair_id`, otherwise one recording) of the
    training split, per-condition centroids are rebuilt from the training
    items of all the OTHER groups; the subject's items of the left-out group
    and the other speakers' labelled spans of the same group are scored
    against them, as `ProfileScorer` would (best cosine over conditions).
    No holdout recording is touched, so the self-assessment of the voiceprint
    on the holdout stays an honest estimate. `extra_other` clips (other
    speakers outside the case recordings) are scored against the centroids
    of all training items.

    `items` defaults to the items eligible for the profile (verified, train,
    aligned, without blocking errors); any non-train item is refused.
    """
    if items is None:
        items = dataset.select_items(splits=(DatasetSplit.TRAIN,), verified_only=True)
    if any(it.split != DatasetSplit.TRAIN for it in items):
        raise DatasetError("La calibrazione leave-one-recording-out usa solo segmenti di addestramento")
    warnings: list[str] = []
    audio = _AudioCache(case, dataset, analysis_sr)

    subj: list[tuple[str, str, np.ndarray]] = []  # (group, condition key, unit embedding)
    for it in sorted(items, key=lambda it: it.item_id):
        if it.end - it.start < min_seconds:
            continue
        v = _unit(backend.embed(audio.segment(it.recording_id, it.start, it.end), analysis_sr))
        if v is not None:
            subj.append((_recording_group(dataset, it.recording_id), it.condition.key(), v))
    train_recs = {r.recording_id for r in dataset.recordings if r.split == DatasetSplit.TRAIN}
    holdout_recs = {r.recording_id for r in dataset.recordings if r.split == DatasetSplit.HOLDOUT}
    oth: list[tuple[str, np.ndarray]] = []
    n_holdout_spans = 0
    for span in dataset.other_speaker_spans:
        if span.recording_id not in train_recs:
            n_holdout_spans += int(span.recording_id in holdout_recs)
            continue
        if span.end - span.start < min_seconds:
            continue
        v = _unit(backend.embed(audio.segment(span.recording_id, span.start, span.end), analysis_sr))
        if v is not None:
            oth.append((_recording_group(dataset, span.recording_id), v))

    groups = sorted({g for g, _, _ in subj} | {g for g, _ in oth})
    same: list[float] = []
    other: list[float] = []
    skipped: list[str] = []
    for g in groups:
        cents = _centroids([(k, v) for gg, k, v in subj if gg != g])
        if not cents:
            skipped.append(g)
            continue
        same += [_best(v, cents) for gg, _, v in subj if gg == g]
        other += [_best(v, cents) for gg, v in oth if gg == g]
    if skipped:
        warnings.append(
            f"{len(skipped)} gruppi di registrazioni esclusi dalla calibrazione: senza altre registrazioni "
            "di addestramento non si può costruire l'impronta che li esclude"
        )
    full = _centroids([(k, v) for _, k, v in subj])
    n_extra = 0
    if full:
        for a, a_sr in extra_other:
            v = _unit(backend.embed(np.asarray(a, dtype=np.float32), a_sr))
            if v is not None:
                other.append(_best(v, full))
                n_extra += 1
    if n_holdout_spans:
        warnings.append(
            f"{n_holdout_spans} tratti di altri parlanti nelle registrazioni di autovalutazione non usati "
            "per la calibrazione (restano per l'autovalutazione)"
        )
    material = {
        "scheme": LORO_SCHEME,
        "holdout_used": False,
        "subject_segments": "training items, each scored against centroids rebuilt without its "
                            "recording group (pair_id, otherwise recording)",
        "other_speakers": "labelled other-speaker spans in training recordings, scored like the subject "
                          "items of the same group; extra clips against all training centroids",
        "n_groups": len(groups),
        "skipped_groups": skipped,
        "n_subject_scores": len(same),
        "n_other_scores": len(other),
        "n_extra_other": n_extra,
        "min_seconds": min_seconds,
        "analysis_sr": analysis_sr,
        "backend": backend.name,
    }
    return LOROScores(same=same, other=other, material=material, warnings=warnings)
