# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Case layout on disk and read-only evidence import (IMPLEMENTATION_PLAN M0).

    cases/<id>/
      case.json      manifest (evidence items, conditions, pairs)
      evidence/      original recordings, read-only, hashed at import
      working/       intermediate audio
      profiles/      subject and vehicle profiles
      adapters/      trained subject models
      outputs/       hypotheses, self-assessment, exports
      logs/          processing logs
"""
from __future__ import annotations

import os
import shutil
import stat
import threading
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from akoe.config import RecordingSource
from akoe.models import ArtifactRef, EvidenceItem, EvidenceRole, RecordingCondition
from akoe.utils import sha256_file

CASE_SUBDIRS = ("evidence", "working", "profiles", "adapters", "outputs", "logs")
MANIFEST = "case.json"


class CaseError(Exception):
    pass


class CaseManifest(BaseModel):
    case_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    evidence: list[EvidenceItem] = Field(default_factory=list)
    # Processing logs of the case workflows (engine.py), with their SHA-256.
    logs: list[ArtifactRef] = Field(default_factory=list)


_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(root: Path) -> threading.RLock:
    key = str(Path(root).resolve())
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


class Case:
    def __init__(self, root: Path, manifest: CaseManifest) -> None:
        self.root = root
        self.manifest = manifest

    def lock(self) -> threading.RLock:
        """Process-wide lock of this case folder: manifest updates from
        concurrent jobs (API) are read-modify-write under it."""
        return _lock_for(self.root)

    def refresh(self) -> None:
        """Reload the manifest from disk (another Case object may have saved it)."""
        path = self.root / MANIFEST
        if path.exists():
            self.manifest = CaseManifest.model_validate_json(path.read_text())

    def register_log(self, ref: ArtifactRef) -> None:
        """Record a processing log (path relative to the case root, SHA-256) in the manifest."""
        with self.lock():
            self.refresh()
            if not any(r.path == ref.path and r.sha256 == ref.sha256 for r in self.manifest.logs):
                self.manifest.logs.append(ref)
                self.save()

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def create(cls, root: Path, case_id: str) -> "Case":
        if (root / MANIFEST).exists():
            raise CaseError(f"case already exists at {root}")
        root.mkdir(parents=True, exist_ok=True)
        for sub in CASE_SUBDIRS:
            (root / sub).mkdir(exist_ok=True)
        case = cls(root, CaseManifest(case_id=case_id))
        case.save()
        return case

    @classmethod
    def open(cls, root: Path) -> "Case":
        path = root / MANIFEST
        if not path.exists():
            raise CaseError(f"no case at {root}")
        return cls(root, CaseManifest.model_validate_json(path.read_text()))

    def save(self) -> None:
        tmp = self.root / (MANIFEST + ".tmp")
        tmp.write_text(self.manifest.model_dump_json(indent=2))
        os.replace(tmp, self.root / MANIFEST)

    def dir(self, name: str) -> Path:
        if name not in CASE_SUBDIRS:
            raise CaseError(f"unknown case subdirectory: {name}")
        return self.root / name

    # -- evidence ----------------------------------------------------------

    def import_evidence(
        self,
        src: Path,
        condition: RecordingCondition | None = None,
        pair_id: str | None = None,
        role: EvidenceRole = EvidenceRole.UNDER_ANALYSIS,
    ) -> EvidenceItem:
        """Copy a recording into evidence/, verify the copy, make it read-only.

        Importing the same bytes twice returns the existing item.
        """
        if not src.is_file():
            raise CaseError(f"not a file: {src}")
        digest = sha256_file(src)
        with self.lock():
            self.refresh()
            for item in self.manifest.evidence:
                if item.artifact.sha256 == digest:
                    return item
            return self._import_new(src, digest, condition, pair_id, role)

    def _import_new(
        self,
        src: Path,
        digest: str,
        condition: RecordingCondition | None,
        pair_id: str | None,
        role: EvidenceRole,
    ) -> EvidenceItem:
        evidence_id = f"ev-{digest[:12]}"
        dest = self.dir("evidence") / f"{evidence_id}{src.suffix.lower()}"
        if dest.exists():  # left by an interrupted import: replace only if it differs
            if sha256_file(dest) != digest:
                dest.chmod(stat.S_IRUSR | stat.S_IWUSR)
                dest.unlink()
        if not dest.exists():
            shutil.copyfile(src, dest)
        if sha256_file(dest) != digest:
            dest.unlink(missing_ok=True)
            raise CaseError(f"copy of {src} does not match the original hash")
        dest.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

        item = EvidenceItem(
            evidence_id=evidence_id,
            original_name=src.name,
            artifact=ArtifactRef(
                path=str(dest.relative_to(self.root)), sha256=digest, kind="evidence"
            ),
            condition=condition or RecordingCondition(),
            pair_id=pair_id,
            role=role,
        )
        self.manifest.evidence.append(item)
        self.save()
        return item

    def get(self, evidence_id: str) -> EvidenceItem:
        for item in self.manifest.evidence:
            if item.evidence_id == evidence_id:
                return item
        raise CaseError(f"unknown evidence: {evidence_id}")

    def path_of(self, item: EvidenceItem) -> Path:
        return item.artifact.resolve(self.root)

    def verify(self) -> list[str]:
        """Return the evidence ids whose file is missing or changed."""
        return [i.evidence_id for i in self.manifest.evidence if not i.artifact.verify(self.root)]

    # -- local / transmitted pairs -----------------------------------------

    def pairs(self) -> dict[str, list[EvidenceItem]]:
        out: dict[str, list[EvidenceItem]] = {}
        for item in self.manifest.evidence:
            if item.pair_id:
                out.setdefault(item.pair_id, []).append(item)
        return out

    def preferred_input(self, pair_id: str) -> EvidenceItem:
        """Local recording when available (usually better quality), else transmitted."""
        items = self.pairs().get(pair_id)
        if not items:
            raise CaseError(f"unknown pair: {pair_id}")
        for source in (RecordingSource.LOCAL, RecordingSource.TRANSMITTED, RecordingSource.OTHER):
            for item in items:
                if item.condition.source == source:
                    return item
        return items[0]
