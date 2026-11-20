# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""High-level case workflows (IMPLEMENTATION_PLAN M7, issue #36).

Used by both the engine API (`api/server.py`) and the CLI (`cli.py`):

- `build_subject_model`: subject recordings (reference role) -> transcript
  alignment (CTC, issue #7) -> split by recording -> verification -> profile
  with voiceprint and timbre per condition -> score calibration
  (leave-one-recording-out on training material, issue #49);
- `build_vehicle_profile`: measured IRs and noise, or an estimate from an
  evidence item;
- `clean`: listening variants for the human transcriber, each a separate,
  labelled artifact (never merged with neural audio);
- `transcribe`: original audio (local recording of a pair first) -> robust VAD
  -> ASR backends (generic, adapted, Parakeet), optional lexicon readings ->
  combine -> hallucination guard -> `HypothesisTranscript`;
- `score_candidates`: "did he say X or Y?", every scorer reported separately;
- `train_subject_adapter`: verified training items, vehicle channel + variable
  codecs augmentation -> verified `AdapterArtifact`;
- `self_assess`: held-out material degraded with the vehicle channel and
  codecs -> `SelfAssessment` per condition + Italian summary.

Every workflow writes a `ProcessingLog` under ``logs/`` (input hashes, every
parameter, model ids and weight hashes, tier, device), records it in the case
manifest and registers each derived artifact (`ArtifactRef` with SHA-256).
Evidence is verified against its hash before it is read.

Backends are injectable: explicit arguments, or a `BackendProvider`
(`set_backend_provider`). `DefaultBackends` loads real models lazily, only when
a workflow needs them; fast tests use the fakes in ``tests/fakes.py``.
Identifiers are English, operator-facing messages Italian.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import stat
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile as sf
from pydantic import BaseModel, Field

from akoe import __version__
from akoe.case import Case, CaseError
from akoe.config import Channel, Mode, RecordingSource, RunProfile, StageKind, Tier, VocalEffort
from akoe.interfaces import ASRBackend, CandidateScorer, VoiceprintBackend
from akoe.models import (
    AdapterArtifact,
    Alternative,
    ArtifactRef,
    DatasetItem,
    DatasetSplit,
    EvidenceItem,
    HypothesisTranscript,
    ProcessingLog,
    RecordingCondition,
    SegmentHypotheses,
    SelfAssessment,
    StageLog,
)
from akoe.neural import TARGET_SR, iter_segments, prepare_audio
from akoe.speaker._io import (
    check_id,
    cut,
    dump_json_bytes,
    load_audio,
    subject_dir,
    write_content_addressed,
    write_json_atomic,
)
from akoe.speaker.dataset import (
    DatasetError,
    Severity,
    SubjectDataset,
    TranscriptAligner,
    TranscriptFormat,
)
from akoe.stages.common import load_mono, to_jsonable, write_mono
from akoe.utils import sha256_file

ENGINE_SCHEMA = "akoe.engine/v1"
SUBJECT_MODEL_FILE = "subject_model.json"
VEHICLE_POINTER_FILE = "current_profile.json"
TRANSCRIPT_FILENAME = "transcript.json"
SELF_ASSESSMENT_DIR = "self_assessment"
MAX_CANDIDATE_SECONDS = 60.0

ProgressFn = Callable[[float, str], None]


class EngineError(Exception):
    """A workflow cannot run; the message is for the operator (Italian)."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _noop(fraction: float, message: str = "") -> None:
    return None


# --------------------------------------------------------------------------
# Files under the chain of custody
# --------------------------------------------------------------------------


def _archive_previous(path: Path, new_sha: str | None = None) -> None:
    """Move an existing file to ``archive/<stem>-<sha12><suffix>`` unless it
    has the same content (used before a module overwrites a file in place,
    e.g. the measured vehicle profile): a file referenced by an older log
    never vanishes."""
    if not path.exists():
        return
    old = sha256_file(path)
    if new_sha is not None and old == new_sha:
        return
    dest = path.parent / "archive" / f"{path.stem}-{old[:12]}{path.suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        path.unlink()
    else:
        os.replace(path, dest)


def _finalize(case: Case, tmp: Path, path: Path, kind: str) -> ArtifactRef:
    """Publish a freshly written file.

    The content is kept as an immutable version,
    ``versions/<stem>-<sha16><suffix>`` (read-only, hard-linked when the file
    system allows it, so no extra space), and `path` becomes the latest
    version (a convenience for the API). The returned ArtifactRef points to
    the immutable version, so every log keeps verifying after later runs.
    """
    digest = sha256_file(tmp)
    vdir = path.parent / "versions"
    vdir.mkdir(parents=True, exist_ok=True)
    version = vdir / f"{path.stem}-{digest[:16]}{path.suffix}"
    if not version.exists():
        try:
            os.link(tmp, version)
        except OSError:
            version.write_bytes(tmp.read_bytes())
        version.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    if path.exists() and sha256_file(path) == digest:
        tmp.unlink()
    else:
        os.replace(tmp, path)
    ref = ArtifactRef.from_file(version, kind=kind, root=case.root)
    if ref.sha256 != digest:
        raise EngineError(f"La versione {version.name} esiste con un contenuto diverso")
    return ref


def _write_versioned(case: Case, path: Path, payload: bytes, kind: str) -> ArtifactRef:
    """`_finalize` for in-memory content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_bytes(payload)
    return _finalize(case, tmp, path, kind)


def _store_input(case: Case, src: Path, subdir: str, kind: str) -> ArtifactRef:
    """Copy an operator-supplied input into the case (read-only, hash-named)."""
    src = Path(src)
    if not src.is_file():
        raise EngineError(f"File inesistente: {src}")
    digest = sha256_file(src)
    dest_dir = case.root / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{kind}-{digest[:12]}{src.suffix.lower()}"
    if not dest.exists():
        tmp = dest.with_name(dest.name + ".tmp")
        tmp.write_bytes(src.read_bytes())
        if sha256_file(tmp) != digest:
            tmp.unlink()
            raise EngineError(f"Copia di {src.name} non corrispondente all'originale (SHA-256)")
        os.replace(tmp, dest)
        dest.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    elif sha256_file(dest) != digest:
        raise EngineError(f"{dest} esiste con un contenuto diverso")
    return ArtifactRef(path=str(dest.relative_to(case.root)), sha256=digest, kind=kind)


def verified_evidence(case: Case, evidence_id: str) -> tuple[EvidenceItem, Path]:
    """The evidence item and its file, after checking the SHA-256 recorded at import."""
    try:
        item = case.get(evidence_id)
    except CaseError as e:
        raise EngineError(f"Reperto sconosciuto: {evidence_id}") from e
    if not item.artifact.verify(case.root):
        raise EngineError(
            f"Il reperto {evidence_id} manca o non corrisponde allo SHA-256 registrato all'importazione: "
            "elaborazione rifiutata."
        )
    return item, case.path_of(item)


# --------------------------------------------------------------------------
# Processing log of one workflow
# --------------------------------------------------------------------------


class WorkflowLog:
    """Builds the `ProcessingLog` of one workflow run and writes it under logs/."""

    def __init__(self, case: Case, workflow: str, *, mode: Mode, tier: Tier,
                 parameters: Mapping[str, Any]) -> None:
        self.case = case
        self.workflow = workflow
        self.mode = mode
        self.tier = tier
        self.log = ProcessingLog(
            case_id=case.manifest.case_id, mode=mode, tier=tier,
            profile=RunProfile.VEHICLE_SINGLE_SPEAKER, input_files={}, workflow=workflow,
            parameters=to_jsonable(dict(parameters)), engine_version=__version__,
        )

    def input(self, label: str, ref: ArtifactRef) -> None:
        self.log.input_files[label] = ref.path
        if not any(r.path == ref.path and r.sha256 == ref.sha256 for r in self.log.inputs):
            self.log.inputs.append(ref)

    def register(self, ref: ArtifactRef | None, label: str | None = None) -> None:
        if ref is None:
            return
        if label:
            self.log.output_files[label] = ref.path
        if not any(r.path == ref.path and r.sha256 == ref.sha256 for r in self.log.artifacts):
            self.log.register(ref)

    def note(self, message: str) -> None:
        if message not in self.log.notes:
            self.log.notes.append(message)

    def stage(
        self,
        name: str,
        kind: StageKind,
        params: Mapping[str, Any],
        started: datetime,
        *,
        input_sha256: str | None = None,
        output_sha256: str | None = None,
        provenance: Any = None,
        model_version: str | None = None,
        model_sha256: str | None = None,
        device: str | None = None,
    ) -> StageLog:
        p = dict(params)
        fields: dict[str, str] = {}
        if provenance is not None and hasattr(provenance, "stage_log_fields"):
            fields = provenance.stage_log_fields()
            p["model_provenance"] = provenance.as_params()
        st = StageLog(
            name=name, kind=kind, mode=self.mode, params=to_jsonable(p),
            input_sha256=input_sha256, output_sha256=output_sha256,
            model_version=model_version or fields.get("model_version"),
            model_sha256=model_sha256 or fields.get("model_sha256"),
            tier=self.tier, device=device or fields.get("device") or "cpu",
            started_at=started, finished_at=_now(),
        )
        self.log.stages.append(st)
        return st

    def skipped(self, name: str, kind: StageKind, params: Mapping[str, Any], reason: str,
                started: datetime | None = None) -> None:
        self.stage(f"{name}_skipped", kind, {**params, "reason": reason}, started or _now())
        self.note(reason)

    def write(self) -> ArtifactRef:
        self.log.finished_at = _now()
        ts = self.log.created_at.strftime("%Y%m%dT%H%M%S")
        path = self.case.dir("logs") / f"{self.workflow}-{ts}-{uuid.uuid4().hex[:8]}.json"
        path.write_text(self.log.model_dump_json(indent=2), encoding="utf-8")
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        ref = ArtifactRef.from_file(path, kind="processing_log", root=self.case.root)
        self.case.register_log(ref)
        return ref


class WorkflowResult(BaseModel):
    """What a workflow returns (also the API job result)."""

    workflow: str
    case_id: str
    log: ArtifactRef
    artifacts: dict[str, ArtifactRef] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)  # Italian


def _finish(wl: WorkflowLog, artifacts: dict[str, ArtifactRef | None], summary: dict[str, Any]) -> WorkflowResult:
    log_ref = wl.write()
    return WorkflowResult(
        workflow=wl.workflow, case_id=wl.case.manifest.case_id, log=log_ref,
        artifacts={k: v for k, v in artifacts.items() if v is not None},
        summary=to_jsonable(summary), notes=list(wl.log.notes),
    )


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------


@dataclass
class VerifiedAdapter:
    """An adapter whose weights matched its artifact (adapt.artifact.verify_adapter)."""

    artifact: AdapterArtifact
    artifact_ref: ArtifactRef
    weights_dir: Path


class BackendProvider(Protocol):
    """Factory of the backends a workflow needs when none is passed explicitly."""

    def voiceprint(self) -> VoiceprintBackend: ...

    def aligner(self) -> TranscriptAligner: ...

    def asr(self, *, tier: Tier, adapter: VerifiedAdapter | None = None,
            whisper_model: str | None = None) -> dict[str, ASRBackend]: ...

    def scorers(self, *, tier: Tier, whisper_model: str | None = None) -> list[CandidateScorer]: ...

    def trainer(self) -> Callable[..., Any]: ...


def default_whisper_model(tier: Tier) -> str:
    """Whisper small on a laptop CPU, large-v3 with a GPU (MASTER_PLAN §4.5)."""
    return "openai/whisper-small" if tier == Tier.EDGE else "openai/whisper-large-v3"


class DefaultBackends:
    """Real models (lazy, cached per process): ReDimNet-B2 voiceprint, Italian
    CTC model for alignment and candidate scoring, Whisper generic / adapted,
    Parakeet-TDT when NeMo is installed, LoRA trainer."""

    def __init__(self, *, device: str = "cpu", voiceprint_name: str | None = None,
                 ctc_model: str | None = None, whisper_model: str | None = None,
                 local_files_only: bool = False) -> None:
        self.device = device
        self.voiceprint_name = voiceprint_name
        self.ctc_model = ctc_model
        self.whisper_model = whisper_model
        self.local_files_only = local_files_only
        self._cache: dict[tuple, Any] = {}
        self._lock = threading.RLock()  # re-entrant: a factory may need another backend (aligner -> CTC)

    def _get(self, key: tuple, factory: Callable[[], Any]) -> Any:
        with self._lock:
            if key not in self._cache:
                self._cache[key] = factory()
            return self._cache[key]

    def voiceprint(self) -> VoiceprintBackend:
        from akoe.speaker.voiceprint_backends import default_voiceprint_backend

        return self._get(("voiceprint", self.voiceprint_name),
                         lambda: default_voiceprint_backend(self.voiceprint_name, device=self.device))

    def ctc_scorer(self) -> Any:
        from akoe.transcribe.ctc_scorer import DEFAULT_CTC_MODEL, CTCForcedAlignmentScorer

        model = self.ctc_model or DEFAULT_CTC_MODEL
        return self._get(("ctc", model), lambda: CTCForcedAlignmentScorer(
            model, device=self.device, local_files_only=self.local_files_only))

    def aligner(self) -> TranscriptAligner:
        from akoe.transcribe.ctc_aligner import CTCTranscriptAligner

        return self._get(("aligner", self.ctc_model), lambda: CTCTranscriptAligner(self.ctc_scorer()))

    def whisper(self, model_id: str) -> Any:
        from akoe.transcribe.whisper_backend import WhisperHFBackend

        return self._get(("whisper", model_id), lambda: WhisperHFBackend(
            model_id, device=self.device, local_files_only=self.local_files_only))

    def adapted_whisper(self, adapter: VerifiedAdapter) -> Any:
        from akoe.adapt.artifact import HP_BASE_REVISION, AdapterIntegrityError, hash_base_weights
        from akoe.models import AdapterTechnique
        from akoe.transcribe.whisper_backend import WhisperHFBackend

        art = adapter.artifact
        rev = art.hyperparams.get(HP_BASE_REVISION)

        def build() -> Any:
            if art.base_model_sha256:
                actual = hash_base_weights(art.base_model, rev)
                if actual is not None and actual != art.base_model_sha256:
                    raise AdapterIntegrityError(
                        f"Adattatore {art.adapter_id}: il modello di base {art.base_model} non corrisponde a "
                        "quello usato per l'addestramento (SHA-256 diverso). Caricamento rifiutato.")
            if art.technique == AdapterTechnique.LORA:
                return WhisperHFBackend(art.base_model, revision=rev, adapter_dir=adapter.weights_dir,
                                        adapter_id=art.adapter_id, device=self.device,
                                        local_files_only=self.local_files_only)
            b = WhisperHFBackend(str(adapter.weights_dir), device=self.device)
            b.name = f"{art.base_model.rstrip('/').split('/')[-1]}/adapted:{art.adapter_id}"
            return b

        return self._get(("adapted", art.adapter_id, adapter.artifact_ref.sha256), build)

    def asr(self, *, tier: Tier, adapter: VerifiedAdapter | None = None,
            whisper_model: str | None = None) -> dict[str, ASRBackend]:
        from akoe.transcribe.parakeet_backend import ParakeetTDTBackend, nemo_available

        base = (whisper_model or (adapter.artifact.base_model if adapter else None)
                or self.whisper_model or default_whisper_model(tier))
        out: dict[str, ASRBackend] = {"generic": self.whisper(base)}
        if adapter is not None:
            out["adapted"] = self.adapted_whisper(adapter)
        if nemo_available():
            out["parakeet"] = self._get(("parakeet",), lambda: ParakeetTDTBackend(
                device=self.device, local_files_only=self.local_files_only))
        return out

    def scorers(self, *, tier: Tier, whisper_model: str | None = None) -> list[CandidateScorer]:
        from akoe.transcribe.whisper_backend import WhisperDecoderScorer

        model = whisper_model or self.whisper_model or default_whisper_model(tier)
        return [self.ctc_scorer(), WhisperDecoderScorer(self.whisper(model))]

    def trainer(self) -> Callable[..., Any]:
        from akoe.adapt.train import train_adapter

        return train_adapter


_PROVIDER: BackendProvider | None = None


def get_backend_provider() -> BackendProvider:
    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = DefaultBackends()
    return _PROVIDER


def set_backend_provider(provider: BackendProvider | None) -> None:
    """Replace the process-wide provider (None restores DefaultBackends lazily)."""
    global _PROVIDER
    _PROVIDER = provider


def _provider(p: BackendProvider | None) -> BackendProvider:
    return p if p is not None else get_backend_provider()


def _systems(asr: Mapping[str, ASRBackend] | Sequence[ASRBackend]) -> list[tuple[str, ASRBackend]]:
    if isinstance(asr, Mapping):
        return list(asr.items())
    out, used = [], set()
    for b in asr:
        label, k = b.name, 2
        while label in used:
            label, k = f"{b.name}#{k}", k + 1
        used.add(label)
        out.append((label, b))
    return out


def _transcribe_kwargs(backend: Any) -> set[str]:
    try:
        sig = inspect.signature(backend.transcribe)
    except (TypeError, ValueError):
        return set()
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return {"audio_provenance", "audio_variant"}
    return {k for k in ("audio_provenance", "audio_variant") if k in sig.parameters}


def _run_asr(backend: ASRBackend, audio: np.ndarray, sr: int, segments: list[tuple[float, float]],
             n_best: int, provenance: StageKind, variant: str) -> list[SegmentHypotheses]:
    kw: dict[str, Any] = {}
    accepted = _transcribe_kwargs(backend)
    if "audio_provenance" in accepted:
        kw["audio_provenance"] = provenance
    if "audio_variant" in accepted:
        kw["audio_variant"] = variant
    return list(backend.transcribe(audio, sr, segments=segments, n_best=n_best, **kw))


class LexiconASR:
    """ASR view of a lexicon decoder (`decode_both(audio, sr) -> DualHypothesis`):
    one segment per region with the reading WITH the case lexicon; the
    reading without it comes from the plain backend. Segments where the two
    readings differ are flagged LEXICON_DEPENDENT."""

    def __init__(self, backend: ASRBackend, decoder: Any, *, name: str | None = None) -> None:
        self.backend = backend
        self.decoder = decoder
        sha = str(getattr(decoder, "lexicon_sha256", "") or "")
        self.lexicon_sha256 = sha
        self.name = name or f"{backend.name}+lexicon:{sha[:12]}"
        self.duals: list[dict[str, Any]] = []

    @property
    def provenance(self) -> Any:
        return getattr(self.backend, "provenance", None)

    def transcribe(self, audio: np.ndarray, sr: int, *, segments: list[tuple[float, float]] | None = None,
                   n_best: int = 5, audio_provenance: StageKind | None = None,
                   audio_variant: str = "original") -> list[SegmentHypotheses]:
        prov = audio_provenance or StageKind.DETERMINISTIC
        a = prepare_audio(np.asarray(audio, dtype=np.float32), sr)
        out = []
        for start, end, chunk in iter_segments(a, TARGET_SR, segments, max_seconds=30.0):
            dual = self.decoder.decode_both(chunk, TARGET_SR)
            self.duals.append({"start": round(start, 3), "end": round(end, 3), "differs": dual.differs,
                               "terms_introduced": dual.terms_introduced, "terms_removed": dual.terms_removed})
            out.append(SegmentHypotheses(
                start=start, end=end, audio_provenance=prov,
                hypotheses=[Alternative(text=dual.with_lexicon, origin=self.name, audio_variant=audio_variant)],
                flags=dual.flags(),
            ))
        return out


def _lexicon_decoder_for(backend: Any, lexicon: Any) -> Any:
    factory = getattr(backend, "lexicon_decoder", None)
    if factory is None:
        return None
    return factory(lexicon)


def _load_lexicon(case: Case, lexicon: Any) -> tuple[Any, ArtifactRef | None]:
    from akoe.adapt.lexicon import CaseLexicon

    if lexicon is None:
        return None, None
    if isinstance(lexicon, CaseLexicon):
        path = case.dir("working") / "lexicon" / f"lexicon-{lexicon.sha256()[:16]}.json"
        if not path.exists():
            lexicon.save(path)
        return lexicon, ArtifactRef.from_file(path, kind="lexicon", root=case.root)
    path = Path(lexicon)
    path = path if path.is_absolute() else case.root / path
    if not path.is_file():
        raise EngineError(f"Lessico non trovato: {lexicon}")
    lex = CaseLexicon.load(path)
    root = case.root if path.resolve().is_relative_to(case.root.resolve()) else None
    return lex, ArtifactRef.from_file(path.resolve() if root is None else path, kind="lexicon", root=root)


# --------------------------------------------------------------------------
# Subject model state and vehicle pointer
# --------------------------------------------------------------------------


class SubjectModelState(BaseModel):
    """Pointer to the current subject model (profile, calibration, adapter).
    Not itself under custody: the artifacts it points to are, through the logs."""

    schema_: str = Field(default="akoe.subject-model/v1", alias="schema")
    subject_id: str
    voiceprint_backend: str | None = None
    profile: ArtifactRef | None = None
    profile_version: int | None = None
    calibration: ArtifactRef | None = None
    calibration_scheme: str | None = None
    report: ArtifactRef | None = None
    adapter: ArtifactRef | None = None
    log: ArtifactRef | None = None
    updated_at: datetime = Field(default_factory=_now)

    model_config = {"populate_by_name": True}


def load_subject_model(case: Case, subject_id: str) -> SubjectModelState | None:
    path = subject_dir(case, subject_id) / SUBJECT_MODEL_FILE
    if not path.exists():
        return None
    return SubjectModelState.model_validate_json(path.read_text(encoding="utf-8"))


def _save_subject_model(case: Case, state: SubjectModelState) -> None:
    state.updated_at = _now()
    write_json_atomic(subject_dir(case, state.subject_id) / SUBJECT_MODEL_FILE,
                      state.model_dump(mode="json", by_alias=True))


def load_vehicle_profile(case: Case, vehicle_id: str) -> tuple[Any, ArtifactRef]:
    """Current profile of the vehicle, after checking its SHA-256."""
    from akoe.vehicle.profile import load_profile, profile_dir

    check_id(vehicle_id, "identificativo del veicolo")
    pointer = profile_dir(case, vehicle_id) / VEHICLE_POINTER_FILE
    if not pointer.exists():
        raise EngineError(f"Nessun profilo per il veicolo {vehicle_id}: costruirlo prima")
    ref = ArtifactRef.model_validate(json.loads(pointer.read_text())["profile"])
    if not ref.verify(case.root):
        raise EngineError(f"Profilo del veicolo {vehicle_id} mancante o modificato (SHA-256)")
    return load_profile(ref.resolve(case.root)), ref


def resolve_adapter(case: Case, adapter: str | Path, subject_id: str | None = None) -> VerifiedAdapter:
    """``"latest"`` (the subject's last verified adapter), an adapter folder or
    its ``adapter.json``, relative to the case root or absolute. Verified."""
    from akoe.adapt.artifact import ADAPTER_JSON, load_adapter_artifact, verify_adapter

    if str(adapter) == "latest":
        if not subject_id:
            raise EngineError("L'adattatore 'latest' richiede il soggetto (subject_id)")
        state = load_subject_model(case, subject_id)
        if state is None or state.adapter is None:
            raise EngineError("Nessun adattatore addestrato per il soggetto: indicarne uno")
        if not state.adapter.verify(case.root):
            raise EngineError("L'artefatto dell'adattatore del soggetto è mancante o modificato (SHA-256)")
        path = state.adapter.resolve(case.root)
    else:
        path = Path(adapter)
        path = path if path.is_absolute() else case.root / path
        if path.is_dir():
            path = path / ADAPTER_JSON
    try:
        art = load_adapter_artifact(path)
        weights = verify_adapter(art, case.root)
    except Exception as e:  # AdapterIntegrityError and I/O errors: refuse
        raise EngineError(f"Adattatore non utilizzabile: {e}") from e
    root = case.root if path.resolve().is_relative_to(case.root.resolve()) else None
    ref = ArtifactRef.from_file(path if root else path.resolve(), kind="adapter", root=root)
    return VerifiedAdapter(artifact=art, artifact_ref=ref, weights_dir=weights)


def _log_adapter(wl: WorkflowLog, adapter: VerifiedAdapter) -> None:
    started = _now()
    wl.input("adapter", adapter.artifact_ref)
    wl.input("adapter_weights", adapter.artifact.weights)
    wl.stage("adapter_verification", StageKind.DETERMINISTIC, {
        "adapter_id": adapter.artifact.adapter_id, "base_model": adapter.artifact.base_model,
        "base_model_sha256": adapter.artifact.base_model_sha256, "technique": adapter.artifact.technique.value,
        "train_manifest_sha256": adapter.artifact.train_manifest_sha256,
        "weights_checksums_sha256": adapter.artifact.weights.sha256, "verified": True,
    }, started, model_version=adapter.artifact.base_model, model_sha256=adapter.artifact.weights.sha256)


# --------------------------------------------------------------------------
# Vehicle channel simulation (augmentation and degradation)
# --------------------------------------------------------------------------


class VehicleChannelAugment:
    """``augment(audio, sr, rng) -> (audio, params)`` for adaptation and
    ``degrade(audio, sr, rng) -> audio`` for self-assessment: cabin IR + real
    noise of that car at the case SNR + a codec drawn per example. Audio at
    another rate is resampled to the simulator rate and back."""

    version = "1"

    def __init__(self, simulator: Any, name: str) -> None:
        self.simulator = simulator
        self.name = name

    def __call__(self, audio: np.ndarray, sr: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
        from akoe.vehicle.audio import pad_or_trim, resample

        seed = int(rng.integers(2**62))
        x = np.asarray(audio, dtype=np.float32)
        ssr = self.simulator.sr
        y_in = x if sr == ssr else resample(x, sr, ssr)
        y, params = self.simulator.simulate(y_in, ssr, seed)
        if sr != ssr:
            y = pad_or_trim(resample(y, ssr, sr), len(x))
        return np.asarray(y, dtype=np.float32), params

    def degrade(self, audio: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
        return self(audio, sr, rng)[0]

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "simulator": self.simulator.describe()}


def channel_augment_for(
    case: Case,
    vehicle_id: str | None,
    *,
    codecs: Sequence[str] | None = None,
    codec_probability: float = 0.5,
    sr: int = TARGET_SR,
) -> tuple[VehicleChannelAugment, float | None, ArtifactRef | None]:
    """(augment, median case SNR in dB or None, vehicle profile ref or None).
    Without a vehicle the simulation applies codecs only."""
    from akoe.vehicle.simulate import ChannelSimConfig, ChannelSimulator, SNRDistribution

    if vehicle_id is None:
        sim = ChannelSimulator(sr, (), (), ChannelSimConfig(codec_probability=codec_probability,
                                                            codecs=list(codecs) if codecs else None))
        return VehicleChannelAugment(sim, "codecs-only"), None, None
    profile, ref = load_vehicle_profile(case, vehicle_id)
    est = getattr(profile, "noise_estimate", None)
    snr_db = None
    snr = SNRDistribution()
    if est is not None and est.snr_db:
        snr = SNRDistribution.from_noise_estimate(est)
        snr_db = float(est.snr_db.get("p50")) if est.snr_db.get("p50") is not None else None
    cfg = ChannelSimConfig(snr=snr, codec_probability=codec_probability,
                           codecs=list(codecs) if codecs else None)
    sim = ChannelSimulator.from_profile(profile, sr=sr, root=case.root, config=cfg)
    return VehicleChannelAugment(sim, f"vehicle-{vehicle_id}+codecs"), snr_db, ref


# --------------------------------------------------------------------------
# 1. Subject model
# --------------------------------------------------------------------------


class RecordingSpec(BaseModel):
    """One recording of the subject with its (optional) transcription."""

    audio: Path
    transcript: Path | None = None
    transcript_format: TranscriptFormat | None = None
    channel: Channel = Channel.UNKNOWN
    source: RecordingSource = RecordingSource.OTHER
    vocal_effort: VocalEffort = VocalEffort.UNKNOWN
    codec: str | None = None
    pair_id: str | None = None
    subject_speaker: str | None = None
    notes: str | None = None

    def condition(self) -> RecordingCondition:
        return RecordingCondition(channel=self.channel, source=self.source, vocal_effort=self.vocal_effort,
                                  codec=self.codec, notes=self.notes)


_AUTO_CERTIFY_EXCLUDED = ("possible_misalignment", "unintelligible_marker", "annotation")


def _certify_all(ds: SubjectDataset, reviewer: str) -> tuple[Any, int]:
    """Review marking as certain the items whose existing transcription the
    operator accepts, except items possibly misaligned, with unintelligible
    markers or annotations, blocked or still to align (those need a real review)."""
    from akoe.speaker.verification import ItemReview, ReviewDecision, Review

    ds.run_quality_checks()
    excluded = {w.item_id for w in ds.warnings if w.code in _AUTO_CERTIFY_EXCLUDED and w.item_id}
    excluded |= ds.blocked_item_ids() | ds.pending_alignment_ids()
    entries = [
        ItemReview(item_id=it.item_id, decision=ReviewDecision.CERTAIN,
                   note="Trascrizione esistente accettata dall'operatore")
        for it in sorted(ds.items, key=lambda it: it.item_id)
        if not it.verified and it.item_id not in excluded
    ]
    n_excluded = sum(1 for it in ds.items if not it.verified and it.item_id in excluded)
    if not entries:
        return None, n_excluded
    return Review(subject_id=ds.subject_id, reviewer=reviewer, entries=entries), n_excluded


def build_subject_model(
    case: Case,
    subject_id: str,
    *,
    folder: Path | None = None,
    recordings: Sequence[RecordingSpec] = (),
    condition: RecordingCondition | None = None,
    subject_speaker: str | None = None,
    holdout_fraction: float = 0.2,
    seed: int = 0,
    align: bool = True,
    aligner: TranscriptAligner | None = None,
    min_alignment_score: float | None = None,
    review: Any = None,
    certify_all_by: str | None = None,
    voiceprint: VoiceprintBackend | None = None,
    compute_timbre: bool = True,
    calibrate: bool = True,
    other_speaker_files: Sequence[Path] = (),
    score_recipe: Any = "default",
    cohort_files: Sequence[Path] = (),
    mode: Mode = Mode.FORENSIC,
    tier: Tier = Tier.EDGE,
    provider: BackendProvider | None = None,
    progress: ProgressFn | None = None,
) -> WorkflowResult:
    """Import the subject's recordings (reference role) and build his model.

    Voiceprint scores are normalized with the WeSpeaker recipe before the
    calibration (`speaker/score_norm.py`, issue #57): `score_recipe` is a
    `ScoringRecipe` ("default" = `DEFAULT_SCORING_RECIPE`, None = plain cosine
    as before); `cohort_files` are cohort-only clips of non-subject speakers
    (e.g. a public Italian cohort), added to the other speakers of the case.

    Untimed transcriptions are aligned with the CTC aligner (spans below its
    threshold are flagged), recordings are split train/holdout by recording,
    a review (or `certify_all_by`: the operator accepts the existing
    transcriptions, see `_certify_all`) marks the items usable, the profile
    (voiceprint + timbre per condition) is built from verified training items
    only, and the score calibration uses leave-one-recording-out on training
    material: the holdout stays for the self-assessment (issue #49).
    """
    from akoe.speaker import profile as sp
    from akoe.speaker.verification import eligible_items, update_profile

    progress = progress or _noop
    check_id(subject_id, "identificativo del soggetto")
    if review is not None and certify_all_by:
        raise EngineError("Indicare una revisione oppure l'accettazione delle trascrizioni esistenti, non entrambe")
    params = {
        "subject_id": subject_id, "folder": str(folder) if folder else None,
        "recordings": [r.model_dump(mode="json") for r in recordings],
        "condition": condition.model_dump(mode="json") if condition else None,
        "subject_speaker": subject_speaker, "holdout_fraction": holdout_fraction, "seed": seed,
        "align": align, "min_alignment_score": min_alignment_score,
        "review": review.reviewer if review is not None else None, "certify_all_by": certify_all_by,
        "compute_timbre": compute_timbre, "calibrate": calibrate,
        "other_speaker_files": [str(p) for p in other_speaker_files],
        "cohort_files": [str(p) for p in cohort_files],
    }
    from akoe.speaker.score_norm import DEFAULT_SCORING_RECIPE, ScoringRecipe

    if isinstance(score_recipe, str):
        if score_recipe != "default":
            raise EngineError(f"Ricetta di punteggio sconosciuta: {score_recipe!r}")
        score_recipe = DEFAULT_SCORING_RECIPE
    elif score_recipe is not None and not isinstance(score_recipe, ScoringRecipe):
        score_recipe = ScoringRecipe.model_validate(score_recipe)
    params["score_recipe"] = score_recipe.model_dump(mode="json") if score_recipe is not None else None
    wl = WorkflowLog(case, "build_subject_model", mode=mode, tier=tier, parameters=params)
    ds = SubjectDataset.load_or_create(case, subject_id)

    # -- import ------------------------------------------------------------
    progress(0.05, "Importazione delle registrazioni del soggetto")
    started = _now()
    before = {r.recording_id for r in ds.recordings}
    if folder is not None:
        ds.import_folder(case, Path(folder), condition, subject_speaker=subject_speaker)
    for spec in recordings:
        ds.import_recording(case, Path(spec.audio), spec.condition(), pair_id=spec.pair_id,
                            transcript_path=Path(spec.transcript) if spec.transcript else None,
                            transcript_format=spec.transcript_format,
                            subject_speaker=spec.subject_speaker or subject_speaker)
    if not ds.recordings:
        raise EngineError("Nessuna registrazione del soggetto: indicare una cartella o dei file da importare")
    for rec in ds.recordings:
        wl.input(f"recording:{rec.recording_id}", rec.audio)
        if rec.transcript is not None:
            wl.input(f"transcript:{rec.recording_id}", rec.transcript)
    new = [r for r in ds.recordings if r.recording_id not in before]
    wl.stage("subject_import", StageKind.DETERMINISTIC, {
        "n_recordings": len(ds.recordings), "n_new": len(new),
        "new": [{"recording_id": r.recording_id, "original_name": r.original_name, "sha256": r.audio.sha256,
                 "condition": r.condition.key(), "pair_id": r.pair_id,
                 "transcript_sha256": r.transcript.sha256 if r.transcript else None,
                 "transcript_format": r.transcript_format.value if r.transcript_format else None}
                for r in new],
        "import_errors": [w.message for w in ds.recorded_warnings if w.code == "import_failed"],
    }, started)

    # -- alignment of untimed transcriptions ------------------------------
    pending = [r for r in ds.recordings if r.needs_alignment]
    if pending and align:
        progress(0.2, "Allineamento delle trascrizioni senza tempi")
        started = _now()
        al = aligner or _provider(provider).aligner()
        thr = min_alignment_score if min_alignment_score is not None else getattr(al, "min_score", None)
        n_before = len(ds.recorded_warnings)
        created = ds.align_pending(case, al, min_score=thr)
        new_w = ds.recorded_warnings[n_before:]
        flagged = sorted(w.item_id for w in new_w if w.code == "possible_misalignment" and w.item_id)
        failed = [w.message for w in new_w if w.code == "alignment_failed"]
        params_fn = getattr(al, "params", None)
        wl.stage("transcript_alignment", StageKind.NEURAL, {
            "aligner": al.name, "aligner_params": params_fn() if callable(params_fn) else None,
            "min_score": thr, "recordings": [r.recording_id for r in pending],
            "n_items": len(created), "flagged_items": flagged, "failed": failed,
            "spans": [{"item_id": it.item_id, "start": it.start, "end": it.end} for it in created],
        }, started, provenance=getattr(al, "provenance", None))
        if flagged:
            wl.note(f"{len(flagged)} segmenti allineati con punteggio basso: possibile allineamento errato, "
                    "da verificare prima dell'uso")
        for msg in failed:
            wl.note(msg)
    elif pending:
        wl.note(f"{len(pending)} registrazioni con trascrizione senza tempi non allineate: "
                "non entrano nel profilo finché non vengono allineate")

    # -- split -------------------------------------------------------------
    progress(0.3, "Divisione per registrazione")
    started = _now()
    ds.split_by_recording(holdout_fraction, seed)
    wl.stage("split_by_recording", StageKind.DETERMINISTIC, {
        **ds.split_history[-1].model_dump(mode="json"),
        "holdout_recordings": ds.holdout_recordings(),
        "train_recordings": sorted(r.recording_id for r in ds.recordings if r.split == DatasetSplit.TRAIN),
    }, started)

    # -- review ------------------------------------------------------------
    if certify_all_by:
        review, n_excluded = _certify_all(ds, certify_all_by)
        if n_excluded:
            wl.note(f"{n_excluded} segmenti non accettati automaticamente (allineamento dubbio, parti "
                    "incomprensibili, annotazioni o errori): richiedono la revisione del trascrittore")

    # -- profile -----------------------------------------------------------
    progress(0.45, "Profilo del soggetto (impronta vocale e timbro per condizione)")
    vp = voiceprint or _provider(provider).voiceprint()
    upd = None
    if review is not None or eligible_items(ds):
        started = _now()
        upd = update_profile(case, ds, vp, review=review,
                             config=sp.ProfileBuildConfig(compute_timbre=compute_timbre))
        rec = upd.record
        wl.stage("speaker_profile", sp.KIND, {
            "version": rec.to_version, "from_version": rec.from_version, "backend": vp.name,
            "n_eligible": rec.n_eligible, "items_added": len(rec.items_added),
            "items_removed": len(rec.items_removed), "reviewer": rec.reviewer,
            "conditions": [{"condition": c.condition.key(), "seconds": c.seconds, "n_segments": c.n_segments,
                            "voiceprint": c.voiceprint is not None,
                            "timbre_stable": (c.timbre.get("f0") or {}).get("stable") if c.timbre else None}
                           for c in upd.profile.conditions],
            "warnings": upd.warnings, "note": sp.SCORE_NOTE,
        }, started, provenance=getattr(vp, "provenance", None))
        for ref in (rec.profile, rec.build_record, rec.dataset_manifest, rec.adaptation_manifest, rec.review):
            wl.register(ref)
        for c in upd.profile.conditions:
            wl.register(c.voiceprint)
        if upd.review_outcome and upd.review_outcome.certain_but_not_eligible:
            wl.note(f"{len(upd.review_outcome.certain_but_not_eligible)} segmenti certi ma non idonei al profilo "
                    "(quota di autovalutazione o altro motivo)")
    else:
        ds.save(case)
        wl.note("Profilo non costruito: nessun segmento verificato. Serve la revisione del trascrittore "
                "(segmenti 'certi') oppure l'accettazione delle trascrizioni esistenti.")
    for ref in [r.transcript for r in ds.recordings if r.transcript is not None]:
        wl.register(ref)

    # -- calibration (leave-one-recording-out, issue #49) ------------------
    cal = cal_ref = None
    cal_info: dict[str, Any] = {"performed": False}
    if upd is not None and calibrate:
        progress(0.75, "Calibrazione dei punteggi (leave-one-recording-out)")
        started = _now()
        extra = []
        for p in other_speaker_files:
            ref = _store_input(case, Path(p), "working/other_speakers", "other_speaker")
            wl.input(f"other_speaker:{ref.sha256[:12]}", ref)
            extra.append((load_audio(ref.resolve(case.root), TARGET_SR), TARGET_SR))
        cohort = []
        for p in cohort_files:
            ref = _store_input(case, Path(p), "working/score_cohort", "score_cohort")
            wl.input(f"score_cohort:{ref.sha256[:12]}", ref)
            cohort.append((load_audio(ref.resolve(case.root), TARGET_SR), TARGET_SR, f"sha256:{ref.sha256}"))
        loro = sp.loro_calibration_scores(case, ds, vp, extra_other=extra, recipe=score_recipe,
                                          cohort_audio=cohort)
        for w in loro.warnings:
            wl.note(w)
        if len(loro.same) >= 2 and len(loro.other) >= 2:
            norm_ref = None
            if loro.normalization is not None:
                from akoe.speaker.profile import save_normalization

                norm_ref = save_normalization(case, loro.normalization)
                wl.register(norm_ref, "score_normalization")
            cal = sp.calibrate_scores(loro.same, loro.other, backend=vp.name, profile=upd.profile,
                                      material=loro.material_json(), scheme=sp.LORO_SCHEME, holdout_used=False,
                                      normalization=norm_ref, score_kind=loro.score_kind)
            cal_ref = sp.save_calibration(case, cal)
            wl.register(cal_ref, "calibration")
            cal_info = {"performed": True, "scheme": cal.scheme, "holdout_used": cal.holdout_used,
                        "score_kind": cal.score_kind, "eer": cal.eer, "threshold_score": cal.threshold_score,
                        "reliable": cal.reliable, "n_same": cal.n_same, "n_other": cal.n_other,
                        "warnings": cal.warnings}
            wl.stage("score_calibration", StageKind.DETERMINISTIC, {**cal_info, "material": loro.material,
                                                                    "params": cal.params}, started)
            for w in cal.warnings:
                wl.note(w)
        else:
            reason = ("Calibrazione non eseguita: servono almeno 2 punteggi del soggetto e 2 di altri parlanti "
                      "sulle registrazioni di addestramento (leave-one-recording-out; la quota di "
                      "autovalutazione non si usa). Nessun avviso automatico 'possibile altro parlante'.")
            cal_info = {"performed": False, "scheme": sp.LORO_SCHEME, "reason": reason,
                        "n_same": len(loro.same), "n_other": len(loro.other)}
            wl.skipped("score_calibration", StageKind.DETERMINISTIC, {"material": loro.material}, reason, started)

    # -- report and state ---------------------------------------------------
    progress(0.9, "Rapporto del modello del soggetto")
    summary = ds.summary()
    lines = [
        f"Soggetto {subject_id}: {summary['recordings']} registrazioni, {summary['items']} segmenti "
        f"({summary['verified_items']} verificati).",
        f"Divisione per registrazione: {summary['split']['train']['recordings']} in addestramento, "
        f"{summary['split']['holdout']['recordings']} messe da parte per l'autovalutazione.",
    ]
    if upd is not None:
        lines.append(f"Profilo versione {upd.profile.version}: {len(upd.profile.conditions)} condizioni "
                     f"({', '.join(c.condition.key() for c in upd.profile.conditions)}).")
    else:
        lines.append("Profilo non costruito: nessun segmento verificato.")
    if cal is not None:
        lines.append(
            "Calibrazione dell'impronta vocale: leave-one-recording-out sulle sole registrazioni di "
            "addestramento; la quota di autovalutazione non è stata usata. "
            f"Errore a pari errore sul materiale di calibrazione circa {round(100 * cal.eer)} su 100"
            + ("." if cal.reliable else " (calibrazione non affidabile: nessun avviso automatico)."))
    elif upd is not None and calibrate:
        lines.append(cal_info.get("reason", "Calibrazione non eseguita."))
    lines.append("L'impronta vocale è un punteggio di somiglianza, mai un'identificazione.")
    report = {
        "schema": "akoe.subject-model-report/v1",
        "subject_id": subject_id,
        "dataset_summary": summary,
        "split": {"holdout_fraction": holdout_fraction, "seed": seed,
                  "holdout_recordings": ds.holdout_recordings()},
        "profile": None if upd is None else {
            "version": upd.profile.version, "ref": upd.record.profile.model_dump(mode="json"),
            "conditions": [c.condition.key() for c in upd.profile.conditions],
            "voiceprint_backend": vp.name,
        },
        "calibration": {**cal_info, "ref": cal_ref.model_dump(mode="json") if cal_ref else None},
        "warnings": [w.message for w in ds.warnings if w.severity != Severity.INFO]
                    + (upd.warnings if upd else []),
        "summary_it": lines,
    }
    report_ref = write_content_addressed(case, subject_dir(case, subject_id) / "reports", "model-report",
                                         to_jsonable(report), kind="subject_model_report")
    wl.register(report_ref, "subject_model_report")

    state = load_subject_model(case, subject_id) or SubjectModelState(subject_id=subject_id)
    if upd is not None:
        state.voiceprint_backend = vp.name
        state.profile = upd.record.profile
        state.profile_version = upd.profile.version
        state.calibration = cal_ref
        state.calibration_scheme = cal.scheme if cal is not None else None
    state.report = report_ref
    result = _finish(wl, {"profile": upd.record.profile if upd else None, "calibration": cal_ref,
                          "report": report_ref,
                          "dataset_manifest": upd.record.dataset_manifest if upd else None,
                          "adaptation_manifest": upd.record.adaptation_manifest if upd else None},
                     {"subject_id": subject_id, "dataset": summary,
                      "profile_version": upd.profile.version if upd else None,
                      "calibration": cal_info, "summary_it": lines})
    state.log = result.log
    _save_subject_model(case, state)
    progress(1.0, "Modello del soggetto pronto")
    return result


# --------------------------------------------------------------------------
# 2. Vehicle profile
# --------------------------------------------------------------------------


def build_vehicle_profile(
    case: Case,
    vehicle_id: str,
    *,
    manifest: Path | None = None,
    impulse_responses: Sequence[Any] = (),
    noise_recordings: Sequence[Any] = (),
    evidence_id: str | None = None,
    estimate_params: Any = None,
    track_engine: bool = True,
    mode: Mode = Mode.FORENSIC,
    tier: Tier = Tier.EDGE,
    progress: ProgressFn | None = None,
) -> WorkflowResult:
    """Measured profile (`misura.json` manifest, or IR and noise files) or,
    as a fallback, a profile estimated from the non-speech parts of an
    evidence item. Becomes the vehicle's current profile."""
    from akoe.vehicle import profile as vp

    progress = progress or _noop
    check_id(vehicle_id, "identificativo del veicolo")
    measured = bool(manifest or impulse_responses or noise_recordings)
    if not measured and evidence_id is None:
        raise EngineError("Indicare le misure del veicolo (manifest, risposte all'impulso, rumori) "
                          "oppure un reperto da cui stimare il profilo")
    params = {"vehicle_id": vehicle_id, "manifest": str(manifest) if manifest else None,
              "impulse_responses": [str(getattr(x, "path", x)) for x in impulse_responses],
              "noise_recordings": [{"path": str(getattr(x, "path", "")), "label": getattr(x, "label", None)}
                                   for x in noise_recordings],
              "evidence_id": evidence_id, "track_engine": track_engine,
              "estimate_params": estimate_params.model_dump(mode="json") if estimate_params is not None else None}
    wl = WorkflowLog(case, "build_vehicle_profile", mode=mode, tier=tier, parameters=params)
    pdir = vp.profile_dir(case, vehicle_id)
    started = _now()
    progress(0.1, "Profilo del veicolo")
    if measured:
        _archive_previous(pdir / vp.PROFILE_FILE)
        if manifest is not None:
            session = vp.MeasurementSession.model_validate_json(Path(manifest).read_text())
            if session.vehicle_id != vehicle_id:
                raise EngineError(f"Il manifest di misura riguarda il veicolo '{session.vehicle_id}', "
                                  f"non '{vehicle_id}'")
            prof = vp.build_profile_from_manifest(Path(manifest), case=case)
        else:
            irs = [x if isinstance(x, vp.ImpulseResponseSource) else vp.ImpulseResponseSource(path=Path(x))
                   for x in impulse_responses]
            noises = [x if isinstance(x, vp.NoiseSource) else vp.NoiseSource.model_validate(x)
                      for x in noise_recordings]
            prof = vp.build_vehicle_profile(vehicle_id, irs, noises, case=case)
        path = pdir / vp.PROFILE_FILE
        for ref in prof.impulse_responses:
            wl.input(f"ir:{ref.sha256[:12]}", ref)
        for rec in prof.noise_recordings:
            wl.input(f"noise:{rec.label}", rec.artifact)
        stage_params = {"estimated": False, "n_impulse_responses": len(prof.impulse_responses),
                        "n_noise_recordings": len(prof.noise_recordings),
                        "ir_details": [d.model_dump(mode="json") for d in prof.ir_details],
                        "noise_details": [d.model_dump(mode="json") for d in prof.noise_details],
                        "observed_codecs": prof.observed_codecs, "build": prof.build,
                        "measurement_metadata": prof.measurement_metadata}
        name = "vehicle_profile_measured"
        input_sha = None
    else:
        item, epath = verified_evidence(case, evidence_id)  # type: ignore[arg-type]
        wl.input(f"evidence:{item.evidence_id}", item.artifact)
        prof = vp.estimate_vehicle_profile(vehicle_id, item, case=case, params=estimate_params,
                                           track_engine=track_engine)
        est = prof.noise_estimate
        assert est is not None
        tag = f"{item.artifact.sha256[:12]}-{hashlib.sha256(est.params.model_dump_json().encode()).hexdigest()[:8]}"
        path = pdir / f"vehicle_profile_estimated-{tag}.json"
        stage_params = {"estimated": True, "source_evidence": item.evidence_id,
                        "params": est.params.model_dump(mode="json"), "nonspeech_s": est.nonspeech_s,
                        "nonspeech_fraction": est.nonspeech_fraction, "noise_level_dbfs": est.noise_level_dbfs,
                        "snr_db": est.snr_db, "octave_bands_db": est.octave_bands_db,
                        "engine": est.engine.summary if est.engine else None,
                        "observed_codecs": prof.observed_codecs, "warnings": est.warnings}
        for w in est.warnings:
            wl.note(w)
        if not prof.noise_recordings:
            wl.note("Nessun tratto senza voce abbastanza lungo: il profilo non contiene rumore utilizzabile "
                    "per la simulazione del canale")
        name = "vehicle_profile_estimated"
        input_sha = item.artifact.sha256
    ref = ArtifactRef.from_file(path, kind="vehicle_profile", root=case.root)
    wl.stage(name, StageKind.DETERMINISTIC, stage_params, started, input_sha256=input_sha,
             output_sha256=ref.sha256)
    wl.register(ref, "vehicle_profile")
    for r in prof.impulse_responses:
        wl.register(r)
    for rec in prof.noise_recordings:
        wl.register(rec.artifact)
    result = _finish(wl, {"vehicle_profile": ref},
                     {"vehicle_id": vehicle_id, "estimated": prof.estimated,
                      "n_impulse_responses": len(prof.impulse_responses),
                      "noise_labels": [r.label for r in prof.noise_recordings],
                      "observed_codecs": prof.observed_codecs})
    write_json_atomic(pdir / VEHICLE_POINTER_FILE, {"profile": ref.model_dump(mode="json"),
                                                    "estimated": prof.estimated,
                                                    "log": result.log.model_dump(mode="json")})
    progress(1.0, "Profilo del veicolo pronto")
    return result


# --------------------------------------------------------------------------
# 3. Listening variants
# --------------------------------------------------------------------------

LISTENING_VARIANTS = ("bands", "denoise_offline", "engine_harmonics", "reference_cancel", "nmf_listening")
_LISTENING_TAIL = "Versione d'ascolto per il trascrittore umano: non è l'audio originale."
LISTENING_LABELS_IT = {
    "original": "Audio originale del reperto (sola lettura).",
    "bands": "Filtro per bande motivato dall'SII e dal profilo spettrale del soggetto, riproducibile. "
             + _LISTENING_TAIL,
    "denoise_offline": "Riduzione del rumore con profilo stimato sull'intera registrazione, riproducibile. "
                       + _LISTENING_TAIL,
    "engine_harmonics": "Attenuazione delle armoniche del motore lungo la traccia stimata, riproducibile. "
                        + _LISTENING_TAIL,
    "reference_cancel": "Cancellazione della radio/musica con la registrazione di riferimento, riproducibile. "
                        + _LISTENING_TAIL,
    "neural": "Pulizia neurale (ipotesi, modalità investigativa). " + _LISTENING_TAIL,
}


def _subject_ltas(case: Case, subject_id: str, channel: Channel) -> dict[str, float] | None:
    from akoe.speaker.profile import load_profile

    state = load_subject_model(case, subject_id)
    if state is None or state.profile is None:
        return None
    prof = load_profile(case, state.profile)
    conds = sorted(prof.conditions, key=lambda c: (c.condition.channel != channel, -c.seconds))
    for c in conds:
        ltas = (c.timbre or {}).get("ltas") or {}
        centres, levels = ltas.get("band_centres_hz"), ltas.get("levels_rel_db")
        if centres and levels:
            return {str(f): float(v) for f, v in zip(centres, levels) if f is not None and v is not None}
    return None


def _subject_speech(case: Case, subject_id: str, sr: int, channel: Channel,
                    max_seconds: float) -> tuple[np.ndarray, list[str]]:
    from akoe.speaker.verification import eligible_items

    ds = SubjectDataset.load_or_create(case, subject_id)
    items = eligible_items(ds)
    same = [it for it in items if it.condition.channel == channel]
    chosen = same or items
    chunks, used, total = [], [], 0.0
    cache: dict[str, np.ndarray] = {}
    for it in chosen:
        rec = ds.recording(it.recording_id)
        if it.recording_id not in cache:
            if not rec.audio.verify(case.root):
                raise EngineError(f"{rec.original_name}: audio del soggetto mancante o modificato (SHA-256)")
            cache = {it.recording_id: load_audio(rec.audio.resolve(case.root), sr)}
        seg = cut(cache[it.recording_id], sr, it.start, it.end)
        if seg.size:
            chunks.append(seg)
            used.append(it.item_id)
            total += len(seg) / sr
        if total >= max_seconds:
            break
    return (np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)), used


def _vehicle_noise(case: Case, vehicle_id: str, sr: int, max_seconds: float) -> tuple[np.ndarray, list[ArtifactRef]]:
    from akoe.vehicle.audio import load_audio as v_load

    prof, _ = load_vehicle_profile(case, vehicle_id)
    chunks, refs, total = [], [], 0.0
    for rec in prof.noise_recordings:
        if not rec.artifact.verify(case.root):
            raise EngineError(f"Rumore del veicolo {rec.label} mancante o modificato (SHA-256)")
        a, _ = v_load(rec.artifact.resolve(case.root), sr=sr)
        chunks.append(np.asarray(a, dtype=np.float64))
        refs.append(rec.artifact)
        total += len(a) / sr
        if total >= max_seconds:
            break
    audio = np.concatenate(chunks)[: int(max_seconds * sr)] if chunks else np.zeros(0)
    return audio, refs


def clean(
    case: Case,
    evidence_id: str,
    *,
    variants: Sequence[str] | None = None,
    subject_id: str | None = None,
    vehicle_id: str | None = None,
    reference: Path | None = None,
    stage_params: Mapping[str, Mapping[str, Any]] | None = None,
    nmf_max_seconds: float = 120.0,
    mode: Mode = Mode.FORENSIC,
    tier: Tier = Tier.EDGE,
    progress: ProgressFn | None = None,
) -> WorkflowResult:
    """Listening variants of an evidence item for the HUMAN transcriber.

    Each variant is computed from the original audio by one stage and written
    as its own artifact (``outputs/<evidence>/listening/<variant>.wav`` plus a
    JSON record with its Italian label, provenance and parameters): variants
    are never chained or merged, and neural audio never shares an artifact
    with deterministic audio. The automatic transcriber keeps using the
    original audio (CLAUDE.md invariant 7).
    """
    from akoe.stages import bands, denoise, harmonics, listening, reference_cancel
    from akoe.stages.vad import detect_speech

    progress = progress or _noop
    sp_ = {k: dict(v) for k, v in (stage_params or {}).items()}
    requested = list(variants) if variants is not None else list(LISTENING_VARIANTS)
    unknown = [v for v in requested if v not in LISTENING_VARIANTS]
    if unknown:
        raise EngineError(f"Varianti d'ascolto sconosciute: {unknown}; disponibili: {list(LISTENING_VARIANTS)}")
    params = {"evidence_id": evidence_id, "variants": requested, "subject_id": subject_id,
              "vehicle_id": vehicle_id, "reference": str(reference) if reference else None,
              "stage_params": sp_, "nmf_max_seconds": nmf_max_seconds}
    wl = WorkflowLog(case, "clean", mode=mode, tier=tier, parameters=params)
    item, path = verified_evidence(case, evidence_id)
    wl.input(f"evidence:{evidence_id}", item.artifact)
    ev_sha = item.artifact.sha256
    audio, sr = load_mono(path)
    out_dir = case.dir("outputs") / evidence_id / "listening"
    out_dir.mkdir(parents=True, exist_ok=True)

    started = _now()
    speech, vad_params = detect_speech(audio, sr, **sp_.get("vad", {}))
    wl.stage("vad_robust", StageKind.DETERMINISTIC, vad_params, started, input_sha256=ev_sha)
    produced: list[dict[str, Any]] = []
    artifacts: dict[str, ArtifactRef | None] = {}

    def emit(variant: str, y: np.ndarray, p: dict[str, Any], started: datetime, *,
             kind: StageKind = StageKind.DETERMINISTIC, label: str | None = None,
             extra: dict[str, Any] | None = None) -> None:
        tmp = out_dir / f".{variant}.{uuid.uuid4().hex[:8]}.tmp.wav"
        write_info = write_mono(tmp, y, sr)
        ref = _finalize(case, tmp, out_dir / f"{variant}.wav", "listening_variant")
        label = label or LISTENING_LABELS_IT[variant]
        record = {
            "schema": "akoe.listening-variant/v1", "variant": variant, "label_it": label,
            "audio_provenance": kind.value, "intended_use": "human_listening",
            "is_original": False, "evidence_id": evidence_id, "evidence_sha256": ev_sha,
            "audio": ref.model_dump(mode="json"), "params": {**p, **write_info}, **(extra or {}),
        }
        rec_ref = _write_versioned(case, out_dir / f"{variant}.json", dump_json_bytes(to_jsonable(record)),
                                   "listening_variant_record")
        wl.stage(f"listening:{variant}", kind, {**p, **write_info, "label_it": label}, started,
                 input_sha256=ev_sha, output_sha256=ref.sha256)
        wl.register(ref, f"listening:{variant}")
        wl.register(rec_ref)
        artifacts[variant] = ref
        produced.append({"variant": variant, "label_it": label, "audio_provenance": kind.value,
                         "audio": ref.model_dump(mode="json"), "record": rec_ref.model_dump(mode="json")})

    if mode in (Mode.FORENSIC, Mode.BOTH):
        n = len(requested)
        for i, variant in enumerate(requested):
            progress(0.1 + 0.8 * i / max(n, 1), f"Versione d'ascolto: {variant}")
            started = _now()
            p = sp_.get(variant, {})
            if variant == "bands":
                ltas = _subject_ltas(case, subject_id, item.condition.channel) if subject_id else None
                snr = bands.estimate_band_snr(audio, sr, speech) if speech else None
                y, prm = bands.shape(audio, sr, subject_ltas=ltas, band_snr_db=snr, **p)
                emit(variant, y, prm, started)
            elif variant == "denoise_offline":
                y, prm = denoise.reduce_noise_offline(audio, sr, speech_segments=speech or None, **p)
                emit(variant, y, prm, started)
            elif variant == "engine_harmonics":
                y, prm, track = harmonics.attenuate_engine_harmonics(audio, sr, speech_segments=speech or None, **p)
                if not prm.get("applied"):
                    reason = (prm.get("warnings_it") or ["Armoniche del motore non rilevate"])[0]
                    wl.skipped("listening:engine_harmonics", StageKind.DETERMINISTIC, prm, reason, started)
                else:
                    emit(variant, y, prm, started, extra={"engine_track": track.to_dict()})
            elif variant == "reference_cancel":
                if reference is None:
                    if variants is not None:
                        wl.skipped("listening:reference_cancel", StageKind.DETERMINISTIC, {},
                                   "Cancellazione con riferimento non eseguita: nessuna registrazione di "
                                   "riferimento (radio/musica) fornita", started)
                    continue
                ref_in = _store_input(case, Path(reference), "working/references", "reference")
                wl.input("reference", ref_in)
                r_audio, r_sr = load_mono(ref_in.resolve(case.root))
                y, prm = reference_cancel.cancel_reference(audio, sr, r_audio, r_sr, **p)
                for w in prm.get("warnings_it", []):
                    wl.note(f"Cancellazione con riferimento: {w}")
                emit(variant, y, {**prm, "reference_sha256": ref_in.sha256}, started)
            elif variant == "nmf_listening":
                missing = [x for x, ok in (("soggetto", subject_id), ("veicolo", vehicle_id)) if not ok]
                if missing:
                    if variants is not None:
                        wl.skipped("listening:nmf_listening", StageKind.DETERMINISTIC, {},
                                   "Versione d'ascolto NMF non eseguita: servono il profilo del soggetto e il "
                                   f"rumore del veicolo (mancano: {', '.join(missing)})", started)
                    continue
                speech_audio, used = _subject_speech(case, subject_id, sr, item.condition.channel,
                                                     nmf_max_seconds)  # type: ignore[arg-type]
                noise_audio, noise_refs = _vehicle_noise(case, vehicle_id, sr, nmf_max_seconds)  # type: ignore[arg-type]
                if speech_audio.size < sr or noise_audio.size < sr:
                    wl.skipped("listening:nmf_listening", StageKind.DETERMINISTIC,
                               {"speech_s": speech_audio.size / sr, "noise_s": noise_audio.size / sr},
                               "Versione d'ascolto NMF non eseguita: parlato verificato del soggetto o rumore del "
                               "veicolo insufficienti (almeno 1 s ciascuno)", started)
                    continue
                learn = dict(p.get("learn", {}))
                enhance = dict(p.get("enhance", {}))
                dicts = listening.learn_dictionaries(speech_audio, noise_audio, sr, **learn)
                ndir = subject_dir(case, subject_id) / "nmf"  # type: ignore[arg-type]
                ndir.mkdir(parents=True, exist_ok=True)
                tmp = ndir / f".dict-{uuid.uuid4().hex[:8]}.npz"
                dicts.save(tmp)
                dref = _finalize(case, tmp, ndir / f"dictionaries-{vehicle_id}-{sr}-{dicts.sha256()[:12]}.npz",
                                 "nmf_dictionaries")
                wl.register(dref, "nmf_dictionaries")
                for r in noise_refs:
                    wl.input(f"vehicle_noise:{r.sha256[:12]}", r)
                y, prm = listening.enhance_for_listening(audio, sr, dicts, **enhance)
                emit(variant, y, {**prm, "speech_items": used, "dictionaries": dref.model_dump(mode="json"),
                                  "vehicle_noise": [r.model_dump(mode="json") for r in noise_refs]},
                     started, label=listening.LABEL_IT)
    else:
        wl.note("Modalità investigativa: le versioni d'ascolto riproducibili non sono state calcolate")

    if mode in (Mode.INVESTIGATIVE, Mode.BOTH):
        started = _now()
        ndir = out_dir / "neural"
        ndir.mkdir(exist_ok=True)
        try:
            denoise.run_neural(path, ndir / "neural.wav")
        except NotImplementedError as e:
            wl.skipped("listening:neural", StageKind.NEURAL, {"purpose": "human listening"},
                       f"Pulizia neurale non disponibile: {e or 'non implementata'}", started)

    original = {"variant": "original", "label_it": LISTENING_LABELS_IT["original"],
                "audio_provenance": StageKind.DETERMINISTIC.value, "is_original": True,
                "audio": item.artifact.model_dump(mode="json")}
    index = {"schema": "akoe.listening-index/v1", "evidence_id": evidence_id, "evidence_sha256": ev_sha,
             "note_it": "Le versioni d'ascolto servono al trascrittore umano; il trascrittore automatico usa "
                        "l'audio originale.",
             "variants": [original] + produced}
    index_ref = _write_versioned(case, out_dir / "index.json", dump_json_bytes(to_jsonable(index)), "listening_index")
    wl.register(index_ref, "listening_index")
    artifacts["index"] = index_ref
    progress(1.0, "Versioni d'ascolto pronte")
    return _finish(wl, artifacts, {"evidence_id": evidence_id,
                                   "variants": [p["variant"] for p in produced],
                                   "vad_segments": len(speech)})


def listening_index(case: Case, evidence_id: str) -> dict[str, Any] | None:
    path = case.dir("outputs") / evidence_id / "listening" / "index.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --------------------------------------------------------------------------
# 4. Transcription
# --------------------------------------------------------------------------


def _pad_merge(segments: Sequence[tuple[float, float]], pad: float, duration: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for s, e in sorted(segments):
        s, e = max(0.0, s - pad), min(duration, e + pad)
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((round(s, 4), round(e, 4)))
    return out


def transcript_path(case: Case, evidence_id: str) -> Path:
    return case.dir("outputs") / evidence_id / TRANSCRIPT_FILENAME


def transcribe(
    case: Case,
    evidence_id: str | None = None,
    *,
    pair_id: str | None = None,
    backends: Mapping[str, ASRBackend] | Sequence[ASRBackend] | None = None,
    adapter: str | Path | None = None,
    lexicon: Any = None,
    lexicon_weight: float = 0.0,
    subject_id: str | None = None,
    voiceprint: VoiceprintBackend | None = None,
    whisper_model: str | None = None,
    n_best: int = 5,
    vad_params: Mapping[str, Any] | None = None,
    segment_pad_s: float = 0.1,
    combine_config: Any = None,
    guard_config: Any = None,
    format_config: Any = None,
    mode: Mode = Mode.FORENSIC,
    tier: Tier = Tier.EDGE,
    provider: BackendProvider | None = None,
    progress: ProgressFn | None = None,
) -> WorkflowResult:
    """Machine hypotheses (`HypothesisTranscript`) of an evidence item or of
    a local/transmitted pair (the local recording is preferred).

    The ASR input is the ORIGINAL audio (audio_provenance DETERMINISTIC). The
    robust VAD gives the regions; every backend runs on them (default:
    Whisper generic, plus the adapted Whisper when a verified adapter is
    given, plus Parakeet when NeMo is installed). With a lexicon, each
    backend that can decode with it adds the reading "with lexicon" as a
    separate source (weight `lexicon_weight`, 0 by default: it is shown and
    flagged, never voted into the consensus). Sources are combined word by
    word, checked by the hallucination guard and written to
    ``outputs/<evidence>/transcript.json``. With `subject_id`, every segment
    also gets the voiceprint score against the subject profile (a score,
    never an identification).
    """
    from akoe.stages.vad import detect_speech
    from akoe.transcribe.combine import HypothesisSource, combine_hypotheses
    from akoe.transcribe.format import TRANSCRIPT_KIND, build_transcript
    from akoe.transcribe.guard import guard_segments

    progress = progress or _noop
    if (evidence_id is None) == (pair_id is None):
        raise EngineError("Indicare un reperto oppure una coppia locale/trasmessa (non entrambi)")
    params = {"evidence_id": evidence_id, "pair_id": pair_id, "adapter": str(adapter) if adapter else None,
              "lexicon": None if lexicon is None else str(getattr(lexicon, "lexicon_id", lexicon)),
              "lexicon_weight": lexicon_weight, "subject_id": subject_id, "whisper_model": whisper_model,
              "n_best": n_best, "vad_params": dict(vad_params or {}), "segment_pad_s": segment_pad_s,
              "audio_variant": "original", "audio_provenance": StageKind.DETERMINISTIC.value}
    wl = WorkflowLog(case, "transcribe", mode=mode, tier=tier, parameters=params)
    if pair_id is not None:
        try:
            chosen = case.preferred_input(pair_id)
        except CaseError as e:
            raise EngineError(f"Coppia sconosciuta: {pair_id}") from e
        others = [i.evidence_id for i in case.pairs()[pair_id] if i.evidence_id != chosen.evidence_id]
        wl.note(f"Coppia {pair_id}: trascritta la registrazione {chosen.condition.source.value} "
                f"{chosen.evidence_id} (la locale ha la precedenza); altre: {', '.join(others) or 'nessuna'}")
        evidence_id = chosen.evidence_id
    item, path = verified_evidence(case, evidence_id)  # type: ignore[arg-type]
    wl.input(f"evidence:{item.evidence_id}", item.artifact)
    ev_sha = item.artifact.sha256

    verified = resolve_adapter(case, adapter, subject_id) if adapter is not None else None
    if verified is not None:
        _log_adapter(wl, verified)

    audio64, sr = load_mono(path)
    audio = audio64.astype(np.float32)
    duration = len(audio) / sr
    progress(0.05, "Rilevamento della voce")
    started = _now()
    speech, vparams = detect_speech(audio64, sr, **dict(vad_params or {}))
    regions = _pad_merge(speech, segment_pad_s, duration)
    wl.stage("vad_robust", StageKind.DETERMINISTIC, {**vparams, "asr_regions": regions,
                                                      "segment_pad_s": segment_pad_s}, started, input_sha256=ev_sha)
    if not regions:
        wl.note("Nessun parlato rilevato dal rilevatore di voce: trascrizione vuota")

    systems = _systems(backends) if backends is not None else list(
        _provider(provider).asr(tier=tier, adapter=verified, whisper_model=whisper_model).items())
    sources: list[Any] = []
    ok_systems: list[tuple[str, ASRBackend]] = []
    expected = StageKind.DETERMINISTIC
    for i, (label, backend) in enumerate(systems):
        progress(0.1 + 0.5 * i / max(len(systems), 1), f"Trascrizione: {label}")
        started = _now()
        try:
            segs = _run_asr(backend, audio, sr, regions, n_best, expected, "original") if regions else []
        except NotImplementedError as e:
            wl.skipped(f"asr:{label}", StageKind.NEURAL, {"backend": backend.name},
                       f"Trascrittore {label} non disponibile: {e or 'non implementato'}", started)
            continue
        fixed = 0
        for seg in segs:
            if seg.audio_provenance != expected:
                seg.audio_provenance = expected
                fixed += 1
        if fixed:
            wl.note(f"{label}: provenienza dell'audio dichiarata dal trascrittore diversa dall'ingresso "
                    f"(audio originale): corretta in '{expected.value}' per {fixed} segmenti")
        wl.stage(f"asr:{label}", StageKind.NEURAL, {
            "backend": backend.name, "label": label, "n_best": n_best, "n_regions": len(regions),
            "n_segments": len(segs), "audio_variant": "original", "audio_provenance": expected.value,
            "provenance_corrected": fixed,
        }, started, input_sha256=ev_sha, provenance=getattr(backend, "provenance", None))
        sources.append(HypothesisSource(segments=segs, audio_variant="original"))
        ok_systems.append((label, backend))
    if not ok_systems:
        raise EngineError("Nessun trascrittore disponibile: trascrizione non eseguita")

    lex, lex_ref = _load_lexicon(case, lexicon)
    if lex is not None:
        wl.input("lexicon", lex_ref)  # type: ignore[arg-type]
        if not lex.terms():
            wl.note("Il lessico non contiene termini confermati dall'operatore: lettura con lessico non eseguita")
        else:
            for label, backend in ok_systems:
                started = _now()
                dec = _lexicon_decoder_for(backend, lex)
                if dec is None:
                    wl.note(f"Lettura con lessico non disponibile per {label}")
                    continue
                lb = LexiconASR(backend, dec)
                segs = lb.transcribe(audio, sr, segments=regions, audio_provenance=expected) if regions else []
                sources.append(HypothesisSource(segments=segs, audio_variant="original",
                                                weight=lexicon_weight, uses_lexicon=True))
                n_diff = sum(1 for d in lb.duals if d["differs"])
                wl.stage(f"asr_lexicon:{label}", StageKind.NEURAL, {
                    "backend": lb.name, "lexicon_sha256": lb.lexicon_sha256, "n_terms": len(lex.terms()),
                    "weight": lexicon_weight, "n_regions": len(regions), "n_differing": n_diff,
                    "regions": lb.duals,
                }, started, input_sha256=ev_sha, provenance=lb.provenance)
                if n_diff:
                    wl.note(f"{label}: {n_diff} tratti in cui la lettura con lessico differisce da quella senza "
                            "(entrambe riportate, segnalate come dipendenti dal lessico)")

    progress(0.7, "Combinazione delle ipotesi")
    started = _now()
    combined = combine_hypotheses(sources, combine_config) if sources else []
    wl.stage("combine", StageKind.DETERMINISTIC, {"n_sources": len(sources), "n_segments": len(combined),
                                           "config": repr(combine_config) if combine_config else "default"},
             started)

    if subject_id is not None and combined:
        started = _now()
        combined = _speaker_scores(case, subject_id, combined, audio, sr, voiceprint, provider, wl, started)

    started = _now()
    guarded = guard_segments(combined, vad_regions=speech, cfg=guard_config)
    wl.stage("hallucination_guard", StageKind.DETERMINISTIC, {
        "n_findings": len(guarded.findings),
        "findings": [{"segment": f.segment_index, "start": f.start, "end": f.end, "flag": f.flag.value,
                      "reason": f.reason.value, "message": f.message, "origin": f.origin}
                     for f in guarded.findings],
    }, started)

    started = _now()
    tr = build_transcript(case.manifest.case_id, guarded.segments, evidence_id=item.evidence_id, cfg=format_config)
    tref = _write_versioned(case, transcript_path(case, item.evidence_id),
                            tr.model_dump_json(indent=2).encode("utf-8"), TRANSCRIPT_KIND)
    wl.stage("format", StageKind.DETERMINISTIC, {"n_segments": len(tr.segments), "models": tr.models,
                                                 "is_ai_hypothesis": True}, started, output_sha256=tref.sha256)
    wl.register(tref, "transcript")
    progress(1.0, "Trascrizione pronta")
    flags: dict[str, int] = {}
    for seg in tr.segments:
        for f in seg.flags:
            flags[f.value] = flags.get(f.value, 0) + 1
    return _finish(wl, {"transcript": tref}, {
        "evidence_id": item.evidence_id, "pair_id": pair_id, "n_segments": len(tr.segments),
        "models": tr.models, "flags": flags, "adapter": verified.artifact.adapter_id if verified else None,
    })


def _speaker_scores(case: Case, subject_id: str, segments: list[SegmentHypotheses], audio: np.ndarray, sr: int,
                    voiceprint: VoiceprintBackend | None, provider: BackendProvider | None, wl: WorkflowLog,
                    started: datetime) -> list[SegmentHypotheses]:
    from akoe.speaker.profile import ProfileScorer, annotate_segment, load_calibration, load_profile

    state = load_subject_model(case, subject_id)
    if state is None or state.profile is None:
        wl.note(f"Punteggi dell'impronta non calcolati: nessun profilo per il soggetto {subject_id}")
        return segments
    vp = voiceprint or _provider(provider).voiceprint()
    if state.voiceprint_backend and state.voiceprint_backend != vp.name:
        wl.note(f"Punteggi dell'impronta non calcolati: profilo costruito con '{state.voiceprint_backend}', "
                f"modello attuale '{vp.name}'")
        return segments
    wl.input("subject_profile", state.profile)
    cal = None
    if state.calibration is not None:
        wl.input("score_calibration", state.calibration)
        cal = load_calibration(case, state.calibration)
        if cal.normalization is not None:
            wl.input("score_normalization", cal.normalization)
    try:
        scorer = ProfileScorer.load(case, load_profile(case, state.profile), vp, cal)
    except (DatasetError, ValueError) as e:
        wl.note(f"Punteggi dell'impronta non calcolati: {e}")
        return segments
    out, scored, flagged = [], 0, 0
    for seg in segments:
        chunk = cut(audio, sr, seg.start, seg.end)
        try:
            score = scorer.score(chunk, sr)
        except ValueError:
            out.append(seg)
            continue
        scored += 1
        flagged += int(bool(score.possible_other_speaker))
        out.append(annotate_segment(seg, score))
    wl.stage("speaker_scores", StageKind.DETERMINISTIC, {
        "subject_id": subject_id, "profile_version": state.profile_version, "backend": vp.name,
        "calibrated": cal is not None, "calibration_scheme": cal.scheme if cal else None,
        "score_kind": cal.score_kind if cal else "cosine",
        "n_scored": scored, "n_possible_other_speaker": flagged,
        "note": "Punteggio di somiglianza, mai un'identificazione.",
    }, started, provenance=getattr(vp, "provenance", None))
    return out


def read_transcript(case: Case, evidence_id: str) -> tuple[HypothesisTranscript, str] | None:
    path = transcript_path(case, evidence_id)
    if not path.exists():
        return None
    return HypothesisTranscript.model_validate_json(path.read_text(encoding="utf-8")), sha256_file(path)


# --------------------------------------------------------------------------
# 5. Candidate phrases
# --------------------------------------------------------------------------


def candidate_thresholds_path(case: Case, subject_id: str) -> Path:
    return case.dir("outputs") / SELF_ASSESSMENT_DIR / check_id(subject_id, "identificativo del soggetto") \
        / "candidate_thresholds.json"


def score_candidates(
    case: Case,
    evidence_id: str,
    start: float,
    end: float,
    candidates: Sequence[str],
    thresholds: Mapping[str, float] | None = None,
    *,
    thresholds_from: str | None = None,
    scorers: Sequence[CandidateScorer] | None = None,
    whisper_model: str | None = None,
    mode: Mode = Mode.FORENSIC,
    tier: Tier = Tier.EDGE,
    provider: BackendProvider | None = None,
    progress: ProgressFn | None = None,
) -> WorkflowResult:
    """Compatibility of operator-typed phrases with [start, end] of the
    original audio. Each scorer (CTC forced alignment, Whisper decoder) is
    reported separately; thresholds come from the caller or from the
    subject's self-assessment (`thresholds_from`), otherwise only a ranking."""
    from akoe.transcribe.candidates import compare_candidates

    progress = progress or _noop
    if not 0 <= start < end:
        raise EngineError("Tratto non valido: serve 0 <= inizio < fine")
    if end - start > MAX_CANDIDATE_SECONDS:
        raise EngineError(f"Tratto troppo lungo per il confronto ({end - start:.1f} s; massimo "
                          f"{MAX_CANDIDATE_SECONDS:.0f} s)")
    params = {"evidence_id": evidence_id, "start": start, "end": end, "candidates": list(candidates),
              "thresholds": dict(thresholds) if thresholds else None, "thresholds_from": thresholds_from,
              "whisper_model": whisper_model}
    wl = WorkflowLog(case, "score_candidates", mode=mode, tier=tier, parameters=params)
    item, path = verified_evidence(case, evidence_id)
    wl.input(f"evidence:{evidence_id}", item.artifact)
    info = sf.info(str(path))
    if start >= info.duration:
        raise EngineError(f"Il tratto inizia oltre la fine del reperto ({info.duration:.2f} s)")
    audio, sr = sf.read(str(path), start=int(start * info.samplerate), stop=int(end * info.samplerate),
                        dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    scorer_list = list(scorers) if scorers is not None else _provider(provider).scorers(
        tier=tier, whisper_model=whisper_model)
    source = "caller" if thresholds else None
    if thresholds is None and thresholds_from:
        tpath = candidate_thresholds_path(case, thresholds_from)
        if tpath.exists():
            data = json.loads(tpath.read_text(encoding="utf-8"))
            names = {s.name for s in scorer_list}
            thresholds = {k: v["threshold"] for k, v in data.get("scorers", {}).items()
                          if k in names and v.get("threshold") is not None}
            source = f"self_assessment:{thresholds_from}"
            wl.input("candidate_thresholds", ArtifactRef.from_file(tpath, kind="candidate_thresholds",
                                                                   root=case.root))
        else:
            wl.note(f"Nessuna soglia tarata per il soggetto {thresholds_from}: solo ordinamento")
    progress(0.2, "Punteggio delle frasi candidate")
    started = _now()
    try:
        comparison = compare_candidates(audio, sr, list(candidates), scorer_list, thresholds)
    except ValueError as e:
        raise EngineError(str(e)) from e
    for rep, scorer in zip(comparison.reports, scorer_list):
        wl.stage(f"candidate_scoring:{rep.scorer}", StageKind.NEURAL, {
            "threshold": rep.threshold, "outcome": rep.outcome.value, "best": rep.best,
            "scores": [{"candidate": e.candidate, "score": e.score, "rank": e.rank, "compatible": e.compatible}
                       for e in rep.entries],
        }, started, input_sha256=item.artifact.sha256, provenance=getattr(scorer, "provenance", None))
    record = {"schema": "akoe.candidate-comparison/v1", "evidence_id": evidence_id,
              "evidence_sha256": item.artifact.sha256, "start": start, "end": end,
              "audio_variant": "original", "thresholds_source": source,
              "comparison": comparison.model_dump(mode="json")}
    ref = write_content_addressed(case, case.dir("outputs") / evidence_id / "candidates", "candidates",
                                  to_jsonable(record), kind="candidate_comparison")
    wl.register(ref, "candidate_comparison")
    progress(1.0, "Confronto completato")
    return _finish(wl, {"candidate_comparison": ref}, {**record, "comparison": comparison.model_dump(mode="json")})


# --------------------------------------------------------------------------
# 6. Subject adapter
# --------------------------------------------------------------------------


def train_subject_adapter(
    case: Case,
    subject_id: str,
    *,
    vehicle_id: str | None = None,
    train_config: Any = None,
    training_set_config: Any = None,
    augment: Callable[..., Any] | None = None,
    codecs: Sequence[str] | None = None,
    codec_probability: float = 0.5,
    trainer: Callable[..., Any] | None = None,
    adapter_id: str | None = None,
    mode: Mode = Mode.FORENSIC,
    tier: Tier = Tier.EDGE,
    provider: BackendProvider | None = None,
    progress: ProgressFn | None = None,
) -> WorkflowResult:
    """Adapt the transcriber to the subject on his VERIFIED training items
    only (never holdout), augmented with the vehicle channel simulation and
    variable codecs; the result is verified before it is recorded."""
    from akoe.adapt.artifact import TRAIN_MANIFEST, verify_adapter, verify_train_manifest
    from akoe.adapt.dataset import TrainingSetConfig, build_training_set
    from akoe.adapt.train import TrainConfig, base_model_for_tier
    from akoe.speaker.verification import eligible_items

    progress = progress or _noop
    check_id(subject_id, "identificativo del soggetto")
    ds = SubjectDataset.load_or_create(case, subject_id)
    items = eligible_items(ds)
    if not items:
        raise EngineError("Nessun segmento verificato della quota di addestramento: adattamento non possibile")
    cfg = train_config or TrainConfig(base_model=base_model_for_tier(tier))
    ts_cfg = training_set_config or TrainingSetConfig()
    params = {"subject_id": subject_id, "vehicle_id": vehicle_id, "train_config": cfg.model_dump(mode="json"),
              "training_set_config": ts_cfg.model_dump(mode="json"), "codecs": list(codecs) if codecs else None,
              "codec_probability": codec_probability, "adapter_id": adapter_id}
    wl = WorkflowLog(case, "train_subject_adapter", mode=mode, tier=tier, parameters=params)
    for it in items:
        wl.input(f"recording:{it.recording_id}", it.audio)
    progress(0.05, "Insieme di addestramento (canale del veicolo e codec)")
    started = _now()
    aug_desc: dict[str, Any] | None = None
    if augment is None:
        aug, _, vref = channel_augment_for(case, vehicle_id, codecs=codecs, codec_probability=codec_probability)
        if vref is not None:
            wl.input("vehicle_profile", vref)
        else:
            wl.note("Nessun profilo del veicolo: simulazione dei soli codec")
        augment, aug_desc = aug, aug.describe()
    tset = build_training_set(items, config=ts_cfg, augment=augment, root=case.root)
    eligible_ids = {it.item_id for it in items}
    if any(s.item_id not in eligible_ids for s in tset.manifest.sources):
        raise EngineError("Invariante violato: elementi non verificati o di autovalutazione nei dati di adattamento")
    wl.stage("training_set", StageKind.DETERMINISTIC, {
        "manifest_sha256": tset.sha256(), "n_sources": len(tset.manifest.sources),
        "n_examples": len(tset), "n_excluded": len(tset.manifest.excluded),
        "conditions": {k: v.model_dump(mode="json") for k, v in tset.manifest.conditions.items()},
        "augment": aug_desc or {"name": tset.manifest.augment_name, "version": tset.manifest.augment_version},
    }, started)
    aid = adapter_id or f"{subject_id}-{cfg.technique.value}-{tset.sha256()[:12]}-s{cfg.seed}"
    check_id(aid, "identificativo dell'adattatore")
    out_dir = case.dir("adapters") / aid
    progress(0.2, "Addestramento dell'adattatore")
    started = _now()
    train = trainer or _provider(provider).trainer()
    res = train(tset, out_dir, cfg, adapter_id=aid, root=case.root, tier=tier)
    try:
        verify_adapter(res.artifact, case.root)
        verify_train_manifest(res.artifact, out_dir / TRAIN_MANIFEST)
    except Exception as e:
        raise EngineError(f"Adattatore prodotto ma non verificabile: {e}") from e
    if res.artifact.train_manifest_sha256 != tset.sha256():
        raise EngineError("L'adattatore non dichiara il manifesto dell'insieme di addestramento usato")
    rec = res.record
    wl.stage("adapter_training", StageKind.NEURAL, {
        "adapter_id": res.artifact.adapter_id, "base_model": res.artifact.base_model,
        "technique": res.artifact.technique.value, "hyperparams": res.artifact.hyperparams,
        "seed": res.artifact.seed, "train_manifest_sha256": res.artifact.train_manifest_sha256,
        "conditions": res.artifact.conditions, "wall_time_s": rec.wall_time_s, "peak_rss_mb": rec.peak_rss_mb,
        "cuda_peak_mb": rec.cuda_peak_mb, "steps": rec.steps, "n_examples": rec.n_examples,
        "final_loss": rec.losses[-1] if rec.losses else None, "versions": rec.versions, "verified": True,
    }, started, model_version=res.artifact.base_model, model_sha256=res.artifact.base_model_sha256,
        device=rec.device)
    for ref, label in ((res.artifact_ref, "adapter"), (res.manifest_ref, "train_manifest"),
                       (res.record_ref, "training_record"), (res.artifact.weights, "adapter_weights")):
        wl.register(ref, label)
    state = load_subject_model(case, subject_id) or SubjectModelState(subject_id=subject_id)
    state.adapter = res.artifact_ref
    result = _finish(wl, {"adapter": res.artifact_ref, "train_manifest": res.manifest_ref,
                          "training_record": res.record_ref},
                     {"adapter_id": res.artifact.adapter_id, "adapter_dir": str(out_dir.relative_to(case.root)),
                      "base_model": res.artifact.base_model, "conditions": res.artifact.conditions,
                      "n_examples": rec.n_examples, "steps": rec.steps, "peak_rss_mb": rec.peak_rss_mb,
                      "wall_time_s": rec.wall_time_s})
    _save_subject_model(case, state)
    progress(1.0, "Adattatore pronto e verificato")
    return result


# --------------------------------------------------------------------------
# 7. Self-assessment
# --------------------------------------------------------------------------


def self_assessment_dir(case: Case, subject_id: str) -> Path:
    return case.dir("outputs") / SELF_ASSESSMENT_DIR / check_id(subject_id, "identificativo del soggetto")


def _span_item(ds: SubjectDataset, span: Any, k: int) -> DatasetItem:
    rec = ds.recording(span.recording_id)
    return DatasetItem(item_id=f"{span.recording_id}-other{k:04d}", recording_id=span.recording_id,
                       audio=rec.audio, start=span.start, end=span.end, text=span.text or "",
                       condition=rec.condition, verified=False, split=rec.split)


def _merge_assessment(base: SelfAssessment, other: SelfAssessment) -> None:
    for k, v in other.overall.items():
        base.overall.setdefault(k, v)
    by = {c.condition_key: c for c in base.by_condition}
    for c in other.by_condition:
        if c.condition_key in by:
            for k, v in c.metrics.items():
                by[c.condition_key].metrics.setdefault(k, v)
        else:
            base.by_condition.append(c)
    base.models += [m for m in other.models if m not in base.models]
    base.limitations += [x for x in other.limitations if x not in base.limitations]


def self_assess(
    case: Case,
    subject_id: str,
    *,
    vehicle_id: str | None = None,
    asr: Mapping[str, ASRBackend] | Sequence[ASRBackend] | None = None,
    adapter: str | Path | None = None,
    scorers: Sequence[CandidateScorer] | None = None,
    voiceprint: VoiceprintBackend | None = None,
    lexicon: Any = None,
    whisper_model: str | None = None,
    degrade: bool = True,
    codecs: Sequence[str] | None = None,
    codec_probability: float = 0.5,
    snr_db: float | None = None,
    include_candidates: bool = True,
    include_voiceprint: bool = True,
    config: Any = None,
    mode: Mode = Mode.FORENSIC,
    tier: Tier = Tier.EDGE,
    provider: BackendProvider | None = None,
    progress: ProgressFn | None = None,
) -> WorkflowResult:
    """Self-assessment on the subject's held-out material, degraded with the
    vehicle channel and variable codecs, reported per condition. Writes
    ``self_assessment.json``, the Italian summary and the candidate
    thresholds (EER of true vs altered phrases, per scorer)."""
    from akoe.assess.metrics import equal_error_rate
    from akoe.assess.selfcheck import (
        FileAudioLoader,
        SelfCheckConfig,
        item_rng,
        render_summary_it,
        run_selfcheck,
    )
    from akoe.speaker.verification import eligible_items

    progress = progress or _noop
    check_id(subject_id, "identificativo del soggetto")
    cfg = config or SelfCheckConfig()
    params = {"subject_id": subject_id, "vehicle_id": vehicle_id, "adapter": str(adapter) if adapter else None,
              "whisper_model": whisper_model, "degrade": degrade, "codecs": list(codecs) if codecs else None,
              "codec_probability": codec_probability, "snr_db": snr_db,
              "include_candidates": include_candidates, "include_voiceprint": include_voiceprint,
              "lexicon": None if lexicon is None else str(getattr(lexicon, "lexicon_id", lexicon)),
              "config": repr(cfg)}
    wl = WorkflowLog(case, "self_assess", mode=mode, tier=tier, parameters=params)
    ds = SubjectDataset.load_or_create(case, subject_id)
    if not ds.recordings:
        raise EngineError(f"Soggetto sconosciuto o senza registrazioni: {subject_id}")
    holdout = ds.select_items(splits=(DatasetSplit.HOLDOUT,), verified_only=True)
    enrollment = eligible_items(ds)
    holdout_recs = {r.recording_id for r in ds.recordings if r.split == DatasetSplit.HOLDOUT}
    impostors = [_span_item(ds, s, k) for k, s in enumerate(ds.other_speaker_spans) if s.recording_id in holdout_recs]
    for it in holdout:
        wl.input(f"holdout:{it.recording_id}", it.audio)
    verified = resolve_adapter(case, adapter, subject_id) if adapter is not None else None
    if verified is not None:
        _log_adapter(wl, verified)
    systems = dict(_systems(asr)) if asr is not None else dict(
        _provider(provider).asr(tier=tier, adapter=verified, whisper_model=whisper_model))
    lex, lex_ref = _load_lexicon(case, lexicon)
    terms: list[str] = []
    if lex is not None and lex.terms():
        wl.input("lexicon", lex_ref)  # type: ignore[arg-type]
        terms = lex.terms()
        first_label, first = next(iter(systems.items()))
        dec = _lexicon_decoder_for(first, lex)
        if dec is not None:
            systems[f"{first_label}+lexicon"] = LexiconASR(first, dec)
        else:
            wl.note(f"Test di inserimento del lessico: {first_label} non decodifica con il lessico")
    scorer_list = list(scorers) if scorers is not None else (
        _provider(provider).scorers(tier=tier, whisper_model=whisper_model) if include_candidates else [])
    vp = voiceprint if voiceprint is not None else (_provider(provider).voiceprint() if include_voiceprint else None)

    aug = None
    est_snr = None
    if degrade:
        aug, est_snr, vref = channel_augment_for(case, vehicle_id, codecs=codecs, codec_probability=codec_probability)
        if vref is not None:
            wl.input("vehicle_profile", vref)
        else:
            wl.note("Nessun profilo del veicolo: materiale degradato con i soli codec")
    snr = snr_db if snr_db is not None else est_snr

    progress(0.1, "Autovalutazione sul materiale messo da parte")
    started = _now()
    first_scorer = scorer_list[0] if scorer_list else None
    res = run_selfcheck(case.manifest.case_id, holdout, systems, scorer=first_scorer, voiceprint=vp,
                        enrollment=enrollment, impostors=impostors, degrade=aug.degrade if aug else None,
                        snr_db=snr, lexicon=terms, root=case.root, config=cfg)
    sa = res.assessment
    trials = {first_scorer.name: res.candidate_trials} if first_scorer is not None else {}
    for s in scorer_list[1:]:
        progress(0.6, f"Autovalutazione delle frasi candidate: {s.name}")
        r2 = run_selfcheck(case.manifest.case_id, holdout, None, scorer=s, degrade=aug.degrade if aug else None,
                           snr_db=snr, root=case.root, config=cfg)
        if r2.computable:
            _merge_assessment(sa, r2.assessment)
            trials[s.name] = r2.candidate_trials

    thresholds: dict[str, Any] = {}
    for name, tr in trials.items():
        alt = [x for t in tr for x in t.altered_scores]
        if tr and alt:
            e = equal_error_rate([t.true_score for t in tr], alt)
            thresholds[name] = {"threshold": e.threshold, "eer": e.eer, "n_true": e.n_target, "n_altered": e.n_nontarget}

    # -- voiceprint calibration statement and check on the holdout (#49) ---
    state = load_subject_model(case, subject_id)
    cal_statement = None
    if state is not None and state.calibration is not None and state.calibration.verify(case.root):
        from akoe.speaker.profile import ProfileScorer, load_calibration, load_profile

        cal = load_calibration(case, state.calibration)
        wl.input("score_calibration", state.calibration)
        used = "usata" if cal.holdout_used else "non è stata usata"
        cal_statement = (f"Soglia dell'impronta vocale tarata con il metodo '{cal.scheme}' sulle registrazioni di "
                         f"addestramento: la quota di autovalutazione {used} per la calibrazione.")
        if vp is not None and cal.backend == vp.name and cal.reliable and state.profile is not None and holdout:
            scorer = ProfileScorer.load(case, load_profile(case, state.profile), vp, cal)
            loader = FileAudioLoader(case.root)

            def flags(items: Sequence[DatasetItem], salt: str) -> list[bool]:
                out = []
                for it in items:
                    try:
                        a, s_ = loader(it)
                        if aug is not None:
                            a = aug.degrade(a, s_, item_rng(cfg.seed, it.item_id, salt))
                        out.append(bool(scorer.score(a, s_).possible_other_speaker))
                    except Exception:  # too short or unreadable: not counted
                        continue
                return out

            fs, fo = flags(holdout, "degrade"), flags(impostors, "impostor")
            if fs:
                sa.overall[f"voiceprint_flag_rate_subject@{vp.name}"] = float(np.mean(fs))
            if fo:
                sa.overall[f"voiceprint_miss_rate_other@{vp.name}"] = float(1.0 - np.mean(fo))
            if fs or fo:
                parts = []
                if fs:
                    parts.append(f"{round(100 * np.mean(fs))} tratti del soggetto su 100 segnalati come possibile "
                                 "altro parlante")
                if fo:
                    parts.append(f"{round(100 * (1 - np.mean(fo)))} tratti di altri parlanti su 100 non segnalati")
                cal_statement += " Sui tratti messi da parte: " + "; ".join(parts) + "."
    elif vp is not None:
        cal_statement = ("Impronta vocale senza calibrazione sul materiale del caso: nessuna soglia per l'avviso "
                         "'possibile altro parlante'.")
    if cal_statement:
        sa.limitations.insert(0, cal_statement)
    for name, t in thresholds.items():
        sa.overall[f"candidate_threshold@{name}"] = float(t["threshold"])
    summary_it = render_summary_it(sa)
    wl.stage("self_assessment", StageKind.DETERMINISTIC, {
        "n_holdout_items": len(holdout), "n_enrollment_items": len(enrollment), "n_impostor_spans": len(impostors),
        "systems": {k: b.name for k, b in systems.items()},
        "system_provenance": {k: (p.as_params() if (p := getattr(b, "provenance", None)) is not None
                                  and hasattr(p, "as_params") else None) for k, b in systems.items()},
        "scorers": [s.name for s in scorer_list], "voiceprint": vp.name if vp is not None else None,
        "degrade": aug.describe() if aug else None, "snr_db": snr, "computable": res.computable,
        "candidate_thresholds": thresholds, "lexicon_terms": len(terms),
    }, started)
    out = self_assessment_dir(case, subject_id)
    sa_ref = _write_versioned(case, out / "self_assessment.json", sa.model_dump_json(indent=2).encode("utf-8"),
                              "self_assessment")
    sum_ref = _write_versioned(case, out / "summary_it.txt", (summary_it + "\n").encode("utf-8"),
                               "self_assessment_summary")
    th_ref = _write_versioned(case, out / "candidate_thresholds.json", dump_json_bytes(to_jsonable({
        "schema": "akoe.candidate-thresholds/v1", "subject_id": subject_id,
        "method": "EER frase giusta contro frasi alterate sul materiale messo da parte",
        "degraded": aug is not None, "scorers": thresholds,
    })), "candidate_thresholds")
    for ref, label in ((sa_ref, "self_assessment"), (sum_ref, "self_assessment_summary"),
                       (th_ref, "candidate_thresholds")):
        wl.register(ref, label)
    progress(1.0, "Autovalutazione pronta")
    return _finish(wl, {"self_assessment": sa_ref, "summary": sum_ref, "candidate_thresholds": th_ref},
                   {"subject_id": subject_id, "computable": res.computable, "overall": sa.overall,
                    "conditions": [c.condition_key for c in sa.by_condition], "summary_it": summary_it,
                    "candidate_thresholds": thresholds})


def read_self_assessment(case: Case, subject_id: str) -> tuple[SelfAssessment, str, str] | None:
    out = self_assessment_dir(case, subject_id)
    path = out / "self_assessment.json"
    if not path.exists():
        return None
    summary = (out / "summary_it.txt").read_text(encoding="utf-8") if (out / "summary_it.txt").exists() else ""
    return SelfAssessment.model_validate_json(path.read_text(encoding="utf-8")), summary, sha256_file(path)


__all__ = [
    "BackendProvider", "DefaultBackends", "EngineError", "LISTENING_VARIANTS", "LexiconASR", "RecordingSpec",
    "SubjectModelState", "VehicleChannelAugment", "VerifiedAdapter", "WorkflowLog", "WorkflowResult",
    "build_subject_model", "build_vehicle_profile", "channel_augment_for", "clean", "get_backend_provider",
    "listening_index", "load_subject_model", "load_vehicle_profile", "read_self_assessment", "read_transcript",
    "resolve_adapter", "score_candidates", "self_assess", "set_backend_provider", "train_subject_adapter",
    "transcribe", "verified_evidence",
]
