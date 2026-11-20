"""Engine API (IMPLEMENTATION_PLAN M7, issue #36).

The Tauri app talks to the engine only through this API: locally (sidecar on
EDGE) or over an encrypted tunnel to a workstation/cloud VM the user controls.

Security defaults: bind to 127.0.0.1, every request needs the bearer token
generated at startup (the app receives it from the sidecar). Audio never
leaves machines the user controls.
"""
from __future__ import annotations

import secrets
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from forensicaudio import __version__
from forensicaudio.case import Case, CaseError
from forensicaudio.config import Channel, RecordingSource, Tier, VocalEffort
from forensicaudio.models import EvidenceItem, RecordingCondition

API_VERSION = "1"


class JobState(BaseModel):
    job_id: str
    case_id: str
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)
    status: str = "queued"  # queued | running | done | failed
    progress: float = 0.0
    message: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None


class NewCase(BaseModel):
    case_id: str


class NewJob(BaseModel):
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)


JobFn = Callable[[Case, dict[str, Any], Callable[[float, str], None]], dict[str, Any]]


class JobRunner:
    """Background jobs (profile building, transcription, self-assessment...)."""

    def __init__(self, max_workers: int = 2) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()
        self.registry: dict[str, JobFn] = {}

    def register(self, kind: str, fn: JobFn) -> None:
        self.registry[kind] = fn

    def submit(self, case: Case, kind: str, params: dict[str, Any]) -> JobState:
        if kind not in self.registry:
            raise KeyError(kind)
        job = JobState(job_id=uuid.uuid4().hex, case_id=case.manifest.case_id, kind=kind, params=params)
        with self._lock:
            self._jobs[job.job_id] = job

        def progress(fraction: float, message: str = "") -> None:
            with self._lock:
                job.progress = max(0.0, min(1.0, fraction))
                job.message = message

        def run() -> None:
            job.status = "running"
            try:
                job.result = self.registry[kind](case, params, progress)
                job.status = "done"
                job.progress = 1.0
            except Exception as e:  # reported to the operator, never swallowed
                job.status = "failed"
                job.error = f"{type(e).__name__}: {e}"
                job.message = traceback.format_exc(limit=3)
            finally:
                job.finished_at = datetime.now(timezone.utc)

        self._pool.submit(run)
        return job

    def get(self, job_id: str) -> JobState:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id]


def _job_verify(case: Case, params: dict[str, Any], progress: Callable[[float, str], None]) -> dict[str, Any]:
    progress(0.1, "Verifica degli hash dei reperti")
    bad = case.verify()
    return {"ok": not bad, "mismatched": bad}


def spectrogram(audio: np.ndarray, sr: int, *, n_fft: int = 512, hop: int = 128,
                max_frames: int = 2000) -> tuple[np.ndarray, dict[str, Any]]:
    """Log-magnitude spectrogram computed by the engine, with every parameter
    returned so the display is reproducible (MASTER_PLAN §4.8)."""
    from scipy.signal import stft

    _, _, z = stft(audio, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop,
                   boundary=None, padded=False)
    db = 20.0 * np.log10(np.abs(z) + 1e-10)
    decim = max(1, int(np.ceil(db.shape[1] / max_frames)))
    if decim > 1:
        usable = (db.shape[1] // decim) * decim
        db = db[:, :usable].reshape(db.shape[0], -1, decim).max(axis=2)
    params = {"window": "hann", "n_fft": n_fft, "hop": hop, "sr": sr, "scale": "dB (20*log10|STFT|)",
              "time_decimation": decim, "decimation_rule": "max", "library": "scipy.signal.stft"}
    return db.astype(np.float32), params


def create_app(cases_root: Path, token: str | None = None, tier: Tier = Tier.EDGE) -> FastAPI:
    cases_root = cases_root.resolve()
    cases_root.mkdir(parents=True, exist_ok=True)
    api_token = token or secrets.token_urlsafe(32)
    runner = JobRunner()
    runner.register("verify", _job_verify)

    app = FastAPI(title="WHISPER engine", version=__version__)
    app.state.token = api_token
    app.state.runner = runner

    def auth(authorization: str | None = Header(default=None)) -> None:
        expected = f"Bearer {api_token}"
        if authorization is None or not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="Token mancante o non valido")

    def open_case(case_id: str) -> Case:
        root = (cases_root / case_id).resolve()
        if root.parent != cases_root:
            raise HTTPException(status_code=400, detail="Identificativo del caso non valido")
        try:
            return Case.open(root)
        except CaseError:
            raise HTTPException(status_code=404, detail=f"Caso non trovato: {case_id}")

    def evidence(case: Case, evidence_id: str) -> EvidenceItem:
        try:
            return case.get(evidence_id)
        except CaseError:
            raise HTTPException(status_code=404, detail=f"Reperto non trovato: {evidence_id}")

    deps = [Depends(auth)]

    @app.get("/health", dependencies=deps)
    def health() -> dict[str, Any]:
        return {"engine": "whisper", "version": __version__, "api": API_VERSION, "tier": tier.value,
                "jobs": sorted(runner.registry)}

    @app.get("/cases", dependencies=deps)
    def list_cases() -> list[str]:
        return sorted(p.name for p in cases_root.iterdir() if (p / "case.json").exists())

    @app.post("/cases", dependencies=deps, status_code=201)
    def create_case(body: NewCase) -> dict[str, Any]:
        if not body.case_id or "/" in body.case_id or body.case_id.startswith("."):
            raise HTTPException(status_code=400, detail="Identificativo del caso non valido")
        try:
            case = Case.create(cases_root / body.case_id, body.case_id)
        except CaseError as e:
            raise HTTPException(status_code=409, detail=str(e))
        return case.manifest.model_dump(mode="json")

    @app.get("/cases/{case_id}", dependencies=deps)
    def get_case(case_id: str) -> dict[str, Any]:
        return open_case(case_id).manifest.model_dump(mode="json")

    @app.post("/cases/{case_id}/evidence", dependencies=deps, status_code=201)
    def import_evidence(
        case_id: str,
        file: UploadFile = File(...),
        channel: Channel = Form(Channel.UNKNOWN),
        source: RecordingSource = Form(RecordingSource.OTHER),
        effort: VocalEffort = Form(VocalEffort.UNKNOWN),
        codec: str | None = Form(None),
        pair_id: str | None = Form(None),
    ) -> dict[str, Any]:
        case = open_case(case_id)
        staging = case.dir("working") / f".upload-{uuid.uuid4().hex}-{Path(file.filename or 'audio').name}"
        try:
            with staging.open("wb") as f:
                while chunk := file.file.read(1 << 20):
                    f.write(chunk)
            item = case.import_evidence(
                staging,
                RecordingCondition(channel=channel, source=source, vocal_effort=effort, codec=codec),
                pair_id=pair_id,
            )
            if item.original_name == staging.name:
                item.original_name = Path(file.filename or staging.name).name
                case.save()
        finally:
            staging.unlink(missing_ok=True)
        return item.model_dump(mode="json")

    @app.get("/cases/{case_id}/evidence/{evidence_id}/audio", dependencies=deps)
    def evidence_audio(case_id: str, evidence_id: str) -> FileResponse:
        case = open_case(case_id)
        return FileResponse(case.path_of(evidence(case, evidence_id)))

    @app.get("/cases/{case_id}/evidence/{evidence_id}/spectrogram", dependencies=deps)
    def evidence_spectrogram(case_id: str, evidence_id: str, start: float = 0.0, end: float | None = None,
                             n_fft: int = 512, hop: int = 128) -> dict[str, Any]:
        import soundfile as sf

        case = open_case(case_id)
        item = evidence(case, evidence_id)
        path = case.path_of(item)
        info = sf.info(str(path))
        sr = info.samplerate
        stop = None if end is None else int(end * sr)
        audio, _ = sf.read(str(path), start=int(start * sr), stop=stop, always_2d=True, dtype="float32")
        db, params = spectrogram(audio.mean(axis=1), sr, n_fft=n_fft, hop=hop)
        return {"evidence_sha256": item.artifact.sha256, "start": start, "end": end,
                "params": params, "shape": list(db.shape), "db": db.round(1).tolist()}

    @app.post("/cases/{case_id}/jobs", dependencies=deps, status_code=202)
    def submit_job(case_id: str, body: NewJob) -> JobState:
        case = open_case(case_id)
        try:
            return runner.submit(case, body.kind, body.params)
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Tipo di job sconosciuto: {body.kind}")

    @app.get("/jobs/{job_id}", dependencies=deps)
    def get_job(job_id: str) -> JobState:
        try:
            return runner.get(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job non trovato")

    return app
