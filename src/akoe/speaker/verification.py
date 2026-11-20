# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Verification loop (IMPLEMENTATION_PLAN M1, issue #11, MASTER_PLAN §4.2).

The human transcriber reviews the subject's segments and marks each one
'certain' or 'uncertain' (optionally correcting the text: the transcription
belongs to the human). Then:

- only items marked certain enter the profile and the adaptation data, and
  only if they are in the TRAIN split, aligned and free of blocking quality
  errors. Uncertain or never-reviewed items are never eligible; a later
  'uncertain' revokes an earlier 'certain';
- every update writes a new, immutable `SpeakerProfile` version
  (version + 1), a hashed adaptation-data manifest, the hashed review, and
  appends a record to a hash-chained log (`verification_log.json`) that
  `verify_log` can check end to end.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from akoe.case import Case
from akoe.interfaces import VoiceprintBackend
from akoe.models import ArtifactRef, DatasetItem, DatasetSplit, SpeakerProfile
from akoe.speaker._io import (
    dump_json_bytes,
    read_verified_json,
    subject_dir,
    write_content_addressed,
    write_json_atomic,
)
from akoe.speaker.dataset import DatasetError, SubjectDataset, normalize_text
from akoe.speaker.profile import ProfileBuildConfig, build_speaker_profile, load_profile

REVIEW_SCHEMA = "akoe.subject-review/v1"
ADAPTATION_SCHEMA = "akoe.adaptation-data/v1"
UPDATE_SCHEMA = "akoe.profile-update/v1"
LOG_NAME = "verification_log.json"
ELIGIBILITY_POLICY = (
    "only items marked 'certain' by the transcriber, in the train split, aligned, "
    "without blocking quality errors; uncertain, unreviewed and holdout items never"
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ReviewDecision(str, Enum):
    CERTAIN = "certain"
    UNCERTAIN = "uncertain"


class ItemReview(BaseModel):
    item_id: str
    decision: ReviewDecision
    corrected_text: str | None = None  # the transcriber's text replaces the item text
    note: str | None = None


class Review(BaseModel):
    """One review session of the human transcriber."""

    schema_: str = Field(default=REVIEW_SCHEMA, alias="schema")
    subject_id: str
    reviewer: str
    reviewed_at: datetime = Field(default_factory=_now)
    entries: list[ItemReview]

    model_config = {"populate_by_name": True}


class ReviewOutcome(BaseModel):
    certified: list[str] = Field(default_factory=list)
    marked_uncertain: list[str] = Field(default_factory=list)
    revoked: list[str] = Field(default_factory=list)  # were certain, now uncertain
    text_corrected: list[str] = Field(default_factory=list)
    certain_but_not_eligible: dict[str, str] = Field(default_factory=dict)  # item -> reason (Italian)


def apply_review(dataset: SubjectDataset, review: Review) -> ReviewOutcome:
    """Apply the transcriber's decisions to the dataset items (in place)."""
    if review.subject_id != dataset.subject_id:
        raise DatasetError(
            f"Revisione del soggetto '{review.subject_id}' applicata al dataset di '{dataset.subject_id}'"
        )
    for e in review.entries:  # fail before changing anything
        dataset.item(e.item_id)
    out = ReviewOutcome()
    for e in review.entries:
        item = dataset.item(e.item_id)
        if e.corrected_text is not None and normalize_text(e.corrected_text) != item.text:
            item.text = normalize_text(e.corrected_text)
            out.text_corrected.append(item.item_id)
        if e.decision == ReviewDecision.CERTAIN:
            item.verified = True
            out.certified.append(item.item_id)
        else:
            if item.verified:
                out.revoked.append(item.item_id)
            item.verified = False
            out.marked_uncertain.append(item.item_id)
    dataset.run_quality_checks()
    eligible = {it.item_id for it in eligible_items(dataset)}
    pending = dataset.pending_alignment_ids()
    blocked = dataset.blocked_item_ids()
    for iid in out.certified:
        if iid in eligible:
            continue
        item = dataset.item(iid)
        if item.split != DatasetSplit.TRAIN:
            reason = "nella quota di autovalutazione: resta fuori da profilo e adattamento"
        elif iid in pending:
            reason = "trascrizione da allineare all'audio"
        elif iid in blocked:
            reason = "errore di qualità bloccante"
        else:
            reason = "non idoneo"
        out.certain_but_not_eligible[iid] = reason
    return out


def eligible_items(dataset: SubjectDataset) -> list[DatasetItem]:
    """Items eligible for the profile and the adaptation data (see ELIGIBILITY_POLICY)."""
    items = dataset.select_items(
        splits=(DatasetSplit.TRAIN,), verified_only=True, include_unaligned=False, exclude_blocked=True
    )
    # Defensive: the invariant must hold whatever select_items does.
    return [it for it in items if it.verified is True and it.split == DatasetSplit.TRAIN]


def write_adaptation_manifest(case: Case, dataset: SubjectDataset) -> ArtifactRef:
    """Hashed list of the adaptation data (its SHA-256 is what an AdapterArtifact
    records as train_manifest_sha256)."""
    items = eligible_items(dataset)
    if any(not it.verified or it.split != DatasetSplit.TRAIN for it in items):
        raise RuntimeError("invariant violated: non-verified or non-train item in adaptation data")
    by_condition: dict[str, dict[str, float]] = {}
    for it in items:
        d = by_condition.setdefault(it.condition.key(), {"items": 0, "seconds": 0.0})
        d["items"] += 1
        d["seconds"] = round(d["seconds"] + (it.end - it.start), 3)
    data = {
        "schema": ADAPTATION_SCHEMA,
        "subject_id": dataset.subject_id,
        "policy": ELIGIBILITY_POLICY,
        "items": [
            {
                "item_id": it.item_id,
                "recording_id": it.recording_id,
                "audio": {"path": it.audio.path, "sha256": it.audio.sha256},
                "start": it.start,
                "end": it.end,
                "text": it.text,
                "condition_key": it.condition.key(),
                "condition": it.condition.model_dump(mode="json"),
            }
            for it in items
        ],
        "n_items": len(items),
        "seconds": round(sum(it.end - it.start for it in items), 3),
        "by_condition": dict(sorted(by_condition.items())),
    }
    return write_content_addressed(
        case, subject_dir(case, dataset.subject_id) / "adaptation", "adaptation", data, kind="adaptation_data"
    )


# --------------------------------------------------------------------------
# Versioned updates with a hash-chained log
# --------------------------------------------------------------------------


class ProfileUpdateRecord(BaseModel):
    schema_: str = Field(default=UPDATE_SCHEMA, alias="schema")
    subject_id: str
    from_version: int | None
    to_version: int
    previous_profile: ArtifactRef | None
    profile: ArtifactRef
    build_record: ArtifactRef
    dataset_manifest: ArtifactRef
    adaptation_manifest: ArtifactRef
    review: ArtifactRef | None = None
    reviewer: str | None = None
    items_added: list[str] = Field(default_factory=list)
    items_removed: list[str] = Field(default_factory=list)
    n_eligible: int
    eligible_items_sha256: str
    previous_record_sha256: str | None
    created_at: datetime = Field(default_factory=_now)

    model_config = {"populate_by_name": True}

    def digest(self) -> str:
        return hashlib.sha256(dump_json_bytes(self.model_dump(mode="json", by_alias=True))).hexdigest()


class ProfileUpdateResult(BaseModel):
    profile: SpeakerProfile
    record: ProfileUpdateRecord
    review_outcome: ReviewOutcome | None = None
    warnings: list[str] = Field(default_factory=list)


def _log_path(case: Case, subject_id: str):
    return subject_dir(case, subject_id) / LOG_NAME


def read_log(case: Case, subject_id: str) -> list[ProfileUpdateRecord]:
    path = _log_path(case, subject_id)
    if not path.exists():
        return []
    return [ProfileUpdateRecord.model_validate(r) for r in json.loads(path.read_text(encoding="utf-8"))]


def _ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def _next_version(case: Case, subject_id: str, log: list[ProfileUpdateRecord]) -> int:
    existing = [
        int(m.group(1))
        for p in subject_dir(case, subject_id).glob("profile-v*.json")
        if (m := re.fullmatch(r"profile-v(\d+)\.json", p.name))
    ]
    return 1 + max([0] + existing + [r.to_version for r in log])


def update_profile(
    case: Case,
    dataset: SubjectDataset,
    backend: VoiceprintBackend,
    *,
    review: Review | None = None,
    config: ProfileBuildConfig | None = None,
) -> ProfileUpdateResult:
    """Apply a review (if any), rebuild the profile from the eligible items only,
    bump the version, write the adaptation manifest and append the log record.
    The dataset is saved."""
    log = read_log(case, dataset.subject_id)
    previous = log[-1] if log else None
    outcome = None
    review_ref = None
    if review is not None:
        review_ref = write_content_addressed(
            case, subject_dir(case, dataset.subject_id) / "reviews", "review",
            review.model_dump(mode="json", by_alias=True), kind="subject_review",
        )
        outcome = apply_review(dataset, review)

    eligible = eligible_items(dataset)
    if not eligible:
        raise DatasetError(
            "Nessun segmento idoneo: servono segmenti di addestramento marcati come certi dal trascrittore"
        )
    version = _next_version(case, dataset.subject_id, log)
    cfg = (config or ProfileBuildConfig()).model_copy(update={"verified_only": True, "include_unaligned": False})
    build = build_speaker_profile(
        case, dataset, backend, version=version, config=cfg,
        parent=previous.profile if previous else None, items=eligible,
    )
    adaptation_ref = write_adaptation_manifest(case, dataset)

    ids = [it.item_id for it in eligible]
    before: set[str] = set()
    if previous is not None:
        prev_data = read_verified_json(case, previous.adaptation_manifest)
        before = {i["item_id"] for i in prev_data["items"]}
    record = ProfileUpdateRecord(
        subject_id=dataset.subject_id,
        from_version=previous.to_version if previous else None,
        to_version=version,
        previous_profile=previous.profile if previous else None,
        profile=build.profile_ref,
        build_record=build.build_record_ref,
        dataset_manifest=build.profile.dataset_manifest,
        adaptation_manifest=adaptation_ref,
        review=review_ref,
        reviewer=review.reviewer if review else None,
        items_added=sorted(set(ids) - before),
        items_removed=sorted(before - set(ids)),
        n_eligible=len(ids),
        eligible_items_sha256=_ids_sha256(ids),
        previous_record_sha256=previous.digest() if previous else None,
    )
    write_json_atomic(
        _log_path(case, dataset.subject_id),
        [r.model_dump(mode="json", by_alias=True) for r in [*log, record]],
    )
    dataset.save(case)
    return ProfileUpdateResult(profile=build.profile, record=record, review_outcome=outcome, warnings=build.warnings)


def verify_log(case: Case, subject_id: str) -> list[str]:
    """Check the hash chain, version sequence and every referenced artifact.
    Returns problems (Italian); an empty list means the log is intact."""
    problems: list[str] = []
    prev: ProfileUpdateRecord | None = None
    for i, rec in enumerate(read_log(case, subject_id)):
        where = f"voce {i + 1} (v{rec.to_version})"
        if prev is None:
            if rec.previous_record_sha256 is not None:
                problems.append(f"{where}: catena di hash non valida (prima voce con predecessore)")
        else:
            if rec.previous_record_sha256 != prev.digest():
                problems.append(f"{where}: catena di hash interrotta, la voce precedente è stata modificata")
            if rec.to_version <= prev.to_version or rec.from_version != prev.to_version:
                problems.append(f"{where}: sequenza delle versioni non valida")
        refs: dict[str, Any] = {
            "profilo": rec.profile, "registro di costruzione": rec.build_record,
            "manifest del dataset": rec.dataset_manifest, "dati di adattamento": rec.adaptation_manifest,
            "revisione": rec.review,
        }
        for name, ref in refs.items():
            if ref is not None and not ref.verify(case.root):
                problems.append(f"{where}: {name} mancante o modificato ({ref.path})")
        if rec.profile.verify(case.root) and load_profile(case, rec.profile).version != rec.to_version:
            problems.append(f"{where}: la versione nel profilo non corrisponde al registro")
        prev = rec
    return problems
