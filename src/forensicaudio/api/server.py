"""Engine API (IMPLEMENTATION_PLAN M7, issue #36).

The Tauri app talks to the engine only through this API: locally (sidecar on
EDGE) or over an encrypted tunnel to a workstation/cloud VM the user controls.

Security defaults: bind to 127.0.0.1, every request needs the bearer token
generated at startup (the app receives it from the sidecar). Audio never
leaves machines the user controls.

Case workflows (`forensicaudio.engine`) run as background jobs with progress:
``build_subject_model``, ``build_vehicle_profile``, ``clean``, ``transcribe``,
``score_candidates``, ``train_subject_adapter``, ``self_assess`` (plus
``verify``). File parameters are paths relative to the case folder (files
sent with ``POST /cases/{id}/uploads`` land in ``working/uploads/``) or
absolute paths inside the import folders configured at startup; anything
else is refused before the job starts.
"""
from __future__ import annotations

import re
import secrets
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import quote

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from forensicaudio import __version__
from forensicaudio.case import Case, CaseError
from forensicaudio.config import Channel, Mode, RecordingSource, Tier, VocalEffort
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


class CandidateRequest(BaseModel):
    start: float
    end: float
    candidates: list[str]
    thresholds: dict[str, float] | None = None
    thresholds_from: str | None = None
    whisper_model: str | None = None
    timeout_s: float = Field(default=30.0, ge=0.0, le=600.0)


JobFn = Callable[[Case, dict[str, Any], Callable[[float, str], None]], dict[str, Any]]
PrepareFn = Callable[[Case, dict[str, Any]], dict[str, Any]]


class JobRunner:
    """Background jobs (profile building, transcription, self-assessment...)."""

    def __init__(self, max_workers: int = 2) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()
        self.registry: dict[str, JobFn] = {}
        self.prepare: dict[str, PrepareFn] = {}

    def register(self, kind: str, fn: JobFn, prepare: PrepareFn | None = None) -> None:
        """`prepare(case, params)` runs synchronously at submission: it validates
        the parameters (ValueError -> 400) and returns those given to `fn`."""
        self.registry[kind] = fn
        if prepare is not None:
            self.prepare[kind] = prepare

    def submit(self, case: Case, kind: str, params: dict[str, Any]) -> JobState:
        if kind not in self.registry:
            raise KeyError(kind)
        run_params = self.prepare[kind](case, dict(params)) if kind in self.prepare else params
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
                job.result = self.registry[kind](case, run_params, progress)
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

    def wait(self, job_id: str, timeout_s: float) -> JobState:
        deadline = time.monotonic() + timeout_s
        job = self.get(job_id)
        while job.status in ("queued", "running") and time.monotonic() < deadline:
            time.sleep(0.02)
        return job


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


# --------------------------------------------------------------------------
# Case workflows as jobs
# --------------------------------------------------------------------------


class PathPolicy:
    """Case-relative paths stay inside the case; absolute paths only inside the
    configured import folders. Refusals are ValueError (HTTP 400)."""

    def __init__(self, import_roots: Sequence[Path] = ()) -> None:
        self.roots = [Path(r).resolve() for r in import_roots]

    def resolve(self, case: Case, value: Any, *, must_exist: bool = True, what: str = "file") -> Path:
        if not isinstance(value, str) or not value or "\x00" in value:
            raise ValueError(f"Percorso non valido per {what}")
        p = Path(value)
        case_root = case.root.resolve()
        if p.is_absolute():
            real = p.resolve()
            allowed = [case_root, *self.roots]
            if not any(real == r or real.is_relative_to(r) for r in allowed):
                raise ValueError(f"Percorso fuori dalle cartelle consentite ({what}): {value}")
        else:
            real = (case_root / p).resolve()
            if not real.is_relative_to(case_root):
                raise ValueError(f"Percorso fuori dalla cartella del caso ({what}): {value}")
        if must_exist and not real.exists():
            raise ValueError(f"File inesistente ({what}): {value}")
        return real


def _common(params: dict[str, Any], tier: Tier) -> dict[str, Any]:
    out: dict[str, Any] = {"tier": tier}
    if params.get("mode") is not None:
        try:
            out["mode"] = Mode(params["mode"])
        except ValueError as e:
            raise ValueError(f"Modalità sconosciuta: {params['mode']}") from e
    return out


def _check_keys(params: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(params) - allowed - {"mode"})
    if unknown:
        raise ValueError(f"Parametri sconosciuti: {', '.join(unknown)}")


def _id(params: dict[str, Any], key: str, required: bool = True) -> str | None:
    value = params.get(key)
    if value is None:
        if required:
            raise ValueError(f"Parametro obbligatorio mancante: {key}")
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value):
        raise ValueError(f"Identificativo non valido per {key}: {value!r}")
    return value


def register_workflows(runner: JobRunner, *, tier: Tier, paths: PathPolicy, backends: Any = None) -> None:
    """Register the engine workflows as job kinds."""
    from forensicaudio import engine

    def provider() -> Any:
        return backends if backends is not None else engine.get_backend_provider()

    def done(result: Any) -> dict[str, Any]:
        return result.model_dump(mode="json")

    # -- build_subject_model ---------------------------------------------
    def prep_subject(case: Case, p: dict[str, Any]) -> dict[str, Any]:
        _check_keys(p, {"subject_id", "folder", "recordings", "condition", "subject_speaker", "holdout_fraction",
                        "seed", "align", "min_alignment_score", "certify_all_by", "compute_timbre", "calibrate",
                        "other_speaker_files", "review"})
        kw: dict[str, Any] = {"subject_id": _id(p, "subject_id"), **_common(p, tier)}
        if p.get("folder"):
            kw["folder"] = paths.resolve(case, p["folder"], what="cartella del soggetto")
        specs = []
        for r in p.get("recordings") or []:
            r = dict(r)
            r["audio"] = paths.resolve(case, r.get("audio"), what="registrazione")
            if r.get("transcript"):
                r["transcript"] = paths.resolve(case, r["transcript"], what="trascrizione")
            specs.append(engine.RecordingSpec.model_validate(r))
        kw["recordings"] = specs
        if p.get("condition"):
            kw["condition"] = RecordingCondition.model_validate(p["condition"])
        if p.get("review"):
            from forensicaudio.speaker.verification import Review

            kw["review"] = Review.model_validate(p["review"])
        kw["other_speaker_files"] = [paths.resolve(case, x, what="altro parlante")
                                     for x in p.get("other_speaker_files") or []]
        for key in ("subject_speaker", "holdout_fraction", "seed", "align", "min_alignment_score",
                    "certify_all_by", "compute_timbre", "calibrate"):
            if key in p and p[key] is not None:
                kw[key] = p[key]
        return kw

    def job_subject(case: Case, kw: dict[str, Any], progress: Callable[[float, str], None]) -> dict[str, Any]:
        sid = kw.pop("subject_id")
        return done(engine.build_subject_model(case, sid, provider=provider(), progress=progress, **kw))

    # -- build_vehicle_profile -------------------------------------------
    def prep_vehicle(case: Case, p: dict[str, Any]) -> dict[str, Any]:
        from forensicaudio.vehicle.profile import ImpulseResponseSource, NoiseSource

        _check_keys(p, {"vehicle_id", "manifest", "impulse_responses", "noise_recordings", "evidence_id",
                        "track_engine"})
        kw: dict[str, Any] = {"vehicle_id": _id(p, "vehicle_id"), **_common(p, tier)}
        if p.get("manifest"):
            kw["manifest"] = paths.resolve(case, p["manifest"], what="manifest di misura")
        irs = []
        for x in p.get("impulse_responses") or []:
            x = {"path": x} if isinstance(x, str) else dict(x)
            x["path"] = paths.resolve(case, x.get("path"), what="risposta all'impulso")
            irs.append(ImpulseResponseSource.model_validate(x))
        noises = []
        for x in p.get("noise_recordings") or []:
            x = dict(x)
            x["path"] = paths.resolve(case, x.get("path"), what="rumore")
            noises.append(NoiseSource.model_validate(x))
        kw.update(impulse_responses=irs, noise_recordings=noises, evidence_id=_id(p, "evidence_id", False))
        if "track_engine" in p:
            kw["track_engine"] = bool(p["track_engine"])
        return kw

    def job_vehicle(case: Case, kw: dict[str, Any], progress: Callable[[float, str], None]) -> dict[str, Any]:
        vid = kw.pop("vehicle_id")
        return done(engine.build_vehicle_profile(case, vid, progress=progress, **kw))

    # -- clean -------------------------------------------------------------
    def prep_clean(case: Case, p: dict[str, Any]) -> dict[str, Any]:
        _check_keys(p, {"evidence_id", "variants", "subject_id", "vehicle_id", "reference", "stage_params"})
        kw: dict[str, Any] = {"evidence_id": _id(p, "evidence_id"), **_common(p, tier),
                              "subject_id": _id(p, "subject_id", False), "vehicle_id": _id(p, "vehicle_id", False)}
        if p.get("variants") is not None:
            kw["variants"] = list(p["variants"])
        if p.get("reference"):
            kw["reference"] = paths.resolve(case, p["reference"], what="riferimento")
        if p.get("stage_params"):
            kw["stage_params"] = dict(p["stage_params"])
        return kw

    def job_clean(case: Case, kw: dict[str, Any], progress: Callable[[float, str], None]) -> dict[str, Any]:
        eid = kw.pop("evidence_id")
        return done(engine.clean(case, eid, progress=progress, **kw))

    # -- transcribe --------------------------------------------------------
    def adapter_param(case: Case, value: Any) -> Any:
        if value is None or value == "latest":
            return value
        return paths.resolve(case, value, what="adattatore")

    def prep_transcribe(case: Case, p: dict[str, Any]) -> dict[str, Any]:
        _check_keys(p, {"evidence_id", "pair_id", "adapter", "lexicon", "subject_id", "whisper_model", "n_best",
                        "lexicon_weight"})
        kw: dict[str, Any] = {"evidence_id": _id(p, "evidence_id", False), "pair_id": _id(p, "pair_id", False),
                              "subject_id": _id(p, "subject_id", False), **_common(p, tier),
                              "adapter": adapter_param(case, p.get("adapter"))}
        if (kw["evidence_id"] is None) == (kw["pair_id"] is None):
            raise ValueError("Indicare evidence_id oppure pair_id")
        if p.get("lexicon"):
            kw["lexicon"] = paths.resolve(case, p["lexicon"], what="lessico")
        for key in ("whisper_model", "n_best", "lexicon_weight"):
            if p.get(key) is not None:
                kw[key] = p[key]
        return kw

    def job_transcribe(case: Case, kw: dict[str, Any], progress: Callable[[float, str], None]) -> dict[str, Any]:
        eid = kw.pop("evidence_id")
        return done(engine.transcribe(case, eid, provider=provider(), progress=progress, **kw))

    # -- score_candidates --------------------------------------------------
    def prep_candidates(case: Case, p: dict[str, Any]) -> dict[str, Any]:
        _check_keys(p, {"evidence_id", "start", "end", "candidates", "thresholds", "thresholds_from",
                        "whisper_model"})
        cands = p.get("candidates")
        if not isinstance(cands, list) or not all(isinstance(c, str) for c in cands):
            raise ValueError("candidates deve essere una lista di frasi")
        return {"evidence_id": _id(p, "evidence_id"), "start": float(p["start"]), "end": float(p["end"]),
                "candidates": cands, "thresholds": p.get("thresholds"),
                "thresholds_from": _id(p, "thresholds_from", False), "whisper_model": p.get("whisper_model"),
                **_common(p, tier)}

    def job_candidates(case: Case, kw: dict[str, Any], progress: Callable[[float, str], None]) -> dict[str, Any]:
        args = [kw.pop(k) for k in ("evidence_id", "start", "end", "candidates", "thresholds")]
        return done(engine.score_candidates(case, *args, provider=provider(), progress=progress, **kw))

    # -- train_subject_adapter ---------------------------------------------
    def prep_train(case: Case, p: dict[str, Any]) -> dict[str, Any]:
        from forensicaudio.adapt.dataset import TrainingSetConfig
        from forensicaudio.adapt.train import TrainConfig, base_model_for_tier

        _check_keys(p, {"subject_id", "vehicle_id", "train_config", "training_set_config", "codecs",
                        "codec_probability", "adapter_id"})
        kw: dict[str, Any] = {"subject_id": _id(p, "subject_id"), "vehicle_id": _id(p, "vehicle_id", False),
                              "adapter_id": _id(p, "adapter_id", False), **_common(p, tier)}
        tc = {"base_model": base_model_for_tier(tier), **(p.get("train_config") or {})}
        kw["train_config"] = TrainConfig.model_validate(tc)
        if p.get("training_set_config"):
            kw["training_set_config"] = TrainingSetConfig.model_validate(p["training_set_config"])
        if p.get("codecs"):
            kw["codecs"] = list(p["codecs"])
        if p.get("codec_probability") is not None:
            kw["codec_probability"] = float(p["codec_probability"])
        return kw

    def job_train(case: Case, kw: dict[str, Any], progress: Callable[[float, str], None]) -> dict[str, Any]:
        sid = kw.pop("subject_id")
        return done(engine.train_subject_adapter(case, sid, provider=provider(), progress=progress, **kw))

    # -- self_assess -------------------------------------------------------
    def prep_assess(case: Case, p: dict[str, Any]) -> dict[str, Any]:
        _check_keys(p, {"subject_id", "vehicle_id", "adapter", "lexicon", "whisper_model", "degrade", "codecs",
                        "codec_probability", "snr_db", "include_candidates", "include_voiceprint"})
        kw: dict[str, Any] = {"subject_id": _id(p, "subject_id"), "vehicle_id": _id(p, "vehicle_id", False),
                              "adapter": adapter_param(case, p.get("adapter")), **_common(p, tier)}
        if p.get("lexicon"):
            kw["lexicon"] = paths.resolve(case, p["lexicon"], what="lessico")
        for key in ("whisper_model", "degrade", "codecs", "codec_probability", "snr_db", "include_candidates",
                    "include_voiceprint"):
            if p.get(key) is not None:
                kw[key] = p[key]
        return kw

    def job_assess(case: Case, kw: dict[str, Any], progress: Callable[[float, str], None]) -> dict[str, Any]:
        sid = kw.pop("subject_id")
        return done(engine.self_assess(case, sid, provider=provider(), progress=progress, **kw))

    runner.register("build_subject_model", job_subject, prep_subject)
    runner.register("build_vehicle_profile", job_vehicle, prep_vehicle)
    runner.register("clean", job_clean, prep_clean)
    runner.register("transcribe", job_transcribe, prep_transcribe)
    runner.register("score_candidates", job_candidates, prep_candidates)
    runner.register("train_subject_adapter", job_train, prep_train)
    runner.register("self_assess", job_assess, prep_assess)


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def create_app(cases_root: Path, token: str | None = None, tier: Tier = Tier.EDGE, *,
               backends: Any = None, import_roots: Sequence[Path] = ()) -> FastAPI:
    """`backends`: an `engine.BackendProvider` (default: the process-wide one,
    real models loaded lazily). `import_roots`: folders whose absolute paths
    job parameters may reference (e.g. the operator's folder on EDGE)."""
    from forensicaudio import engine

    cases_root = cases_root.resolve()
    cases_root.mkdir(parents=True, exist_ok=True)
    api_token = token or secrets.token_urlsafe(32)
    runner = JobRunner()
    runner.register("verify", _job_verify)
    paths = PathPolicy(import_roots)
    register_workflows(runner, tier=tier, paths=paths, backends=backends)

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

    def safe_id(value: str, what: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value):
            raise HTTPException(status_code=400, detail=f"Identificativo non valido ({what})")
        return value

    def submit(case: Case, kind: str, params: dict[str, Any]) -> JobState:
        try:
            return runner.submit(case, kind, params)
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Tipo di job sconosciuto: {kind}")
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=400, detail=str(e))

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
                with case.lock():
                    case.refresh()
                    stored = case.get(item.evidence_id)
                    stored.original_name = Path(file.filename or staging.name).name
                    case.save()
                    item = stored
        finally:
            staging.unlink(missing_ok=True)
        return item.model_dump(mode="json")

    @app.post("/cases/{case_id}/uploads", dependencies=deps, status_code=201)
    def upload(case_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
        """Stage an input file (subject recording, transcription, IR, noise,
        reference, lexicon...) in ``working/uploads/``; job parameters refer to
        the returned case-relative path. Evidence goes through /evidence."""
        import hashlib

        case = open_case(case_id)
        updir = case.dir("working") / "uploads"
        updir.mkdir(parents=True, exist_ok=True)
        name = _SAFE_NAME.sub("_", Path(file.filename or "file").name).lstrip(".") or "file"
        tmp = updir / f".tmp-{uuid.uuid4().hex}"
        h = hashlib.sha256()
        size = 0
        with tmp.open("wb") as f:
            while chunk := file.file.read(1 << 20):
                h.update(chunk)
                size += len(chunk)
                f.write(chunk)
        digest = h.hexdigest()
        dest = updir / f"{digest[:12]}-{name}"
        if dest.exists():
            tmp.unlink()
        else:
            tmp.replace(dest)
        return {"path": str(dest.relative_to(case.root)), "sha256": digest, "size": size,
                "original_name": Path(file.filename or name).name}

    @app.get("/cases/{case_id}/evidence/{evidence_id}/audio", dependencies=deps)
    def evidence_audio(case_id: str, evidence_id: str) -> FileResponse:
        case = open_case(case_id)
        item = evidence(case, evidence_id)
        return FileResponse(case.path_of(item), headers={
            "X-Audio-Variant": "original", "X-Audio-Provenance": "deterministic",
            "X-Audio-Label": quote(engine.LISTENING_LABELS_IT["original"]),
            "X-Content-SHA256": item.artifact.sha256})

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

    @app.get("/cases/{case_id}/evidence/{evidence_id}/transcript", dependencies=deps)
    def evidence_transcript(case_id: str, evidence_id: str) -> JSONResponse:
        case = open_case(case_id)
        evidence(case, evidence_id)
        found = engine.read_transcript(case, evidence_id)
        if found is None:
            raise HTTPException(status_code=404, detail="Nessuna trascrizione automatica per questo reperto")
        tr, sha = found
        return JSONResponse(tr.model_dump(mode="json"), headers={"X-Content-SHA256": sha})

    @app.get("/cases/{case_id}/evidence/{evidence_id}/listening", dependencies=deps)
    def listening_variants(case_id: str, evidence_id: str) -> dict[str, Any]:
        case = open_case(case_id)
        item = evidence(case, evidence_id)
        index = engine.listening_index(case, evidence_id)
        if index is None:
            original = {"variant": "original", "label_it": engine.LISTENING_LABELS_IT["original"],
                        "audio_provenance": "deterministic", "is_original": True,
                        "audio": item.artifact.model_dump(mode="json")}
            return {"evidence_id": evidence_id, "variants": [original]}
        return index

    @app.get("/cases/{case_id}/evidence/{evidence_id}/listening/{variant}/audio", dependencies=deps)
    def listening_audio(case_id: str, evidence_id: str, variant: str) -> FileResponse:
        case = open_case(case_id)
        item = evidence(case, evidence_id)
        if variant == "original":
            return evidence_audio(case_id, evidence_id)
        index = engine.listening_index(case, evidence_id) or {}
        entry = next((v for v in index.get("variants", []) if v.get("variant") == variant
                      and variant in engine.LISTENING_VARIANTS), None)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Versione d'ascolto non trovata: {variant}")
        from forensicaudio.models import ArtifactRef

        ref = ArtifactRef.model_validate(entry["audio"])
        path = ref.resolve(case.root).resolve()
        if not path.is_relative_to(case.root.resolve()) or not ref.verify(case.root):
            raise HTTPException(status_code=409, detail="Versione d'ascolto mancante o modificata (SHA-256)")
        return FileResponse(path, filename=f"{item.evidence_id}-{variant}.wav", headers={
            "X-Audio-Variant": variant, "X-Audio-Provenance": entry["audio_provenance"],
            "X-Audio-Label": quote(entry["label_it"]), "X-Audio-Intended-Use": "human_listening",
            "X-Content-SHA256": ref.sha256})

    @app.post("/cases/{case_id}/evidence/{evidence_id}/candidates", dependencies=deps)
    def candidates(case_id: str, evidence_id: str, body: CandidateRequest) -> JSONResponse:
        """Synchronous with a timeout: 200 with the comparison when the job ends
        in time, 202 with the job state otherwise (poll /jobs/{id})."""
        case = open_case(case_id)
        evidence(case, evidence_id)
        params = {"evidence_id": evidence_id, **body.model_dump(exclude={"timeout_s"}, exclude_none=True)}
        job = submit(case, "score_candidates", params)
        job = runner.wait(job.job_id, body.timeout_s)
        if job.status == "done":
            return JSONResponse(job.result or {})
        if job.status == "failed":
            raise HTTPException(status_code=422, detail=job.error)
        return JSONResponse(job.model_dump(mode="json"), status_code=202)

    @app.get("/cases/{case_id}/subjects/{subject_id}/model", dependencies=deps)
    def subject_model(case_id: str, subject_id: str) -> dict[str, Any]:
        case = open_case(case_id)
        state = engine.load_subject_model(case, safe_id(subject_id, "soggetto"))
        if state is None:
            raise HTTPException(status_code=404, detail=f"Nessun modello per il soggetto {subject_id}")
        report = None
        if state.report is not None and state.report.verify(case.root):
            import json

            report = json.loads(state.report.resolve(case.root).read_text(encoding="utf-8"))
        return {"state": state.model_dump(mode="json", by_alias=True), "report": report}

    @app.get("/cases/{case_id}/subjects/{subject_id}/self-assessment", dependencies=deps)
    def self_assessment(case_id: str, subject_id: str) -> JSONResponse:
        case = open_case(case_id)
        found = engine.read_self_assessment(case, safe_id(subject_id, "soggetto"))
        if found is None:
            raise HTTPException(status_code=404, detail=f"Nessuna autovalutazione per il soggetto {subject_id}")
        sa, summary, sha = found
        return JSONResponse({"assessment": sa.model_dump(mode="json"), "summary_it": summary},
                            headers={"X-Content-SHA256": sha})

    @app.get("/cases/{case_id}/logs", dependencies=deps)
    def logs(case_id: str) -> list[dict[str, Any]]:
        return [r.model_dump(mode="json") for r in open_case(case_id).manifest.logs]

    @app.post("/cases/{case_id}/jobs", dependencies=deps, status_code=202)
    def submit_job(case_id: str, body: NewJob) -> JobState:
        case = open_case(case_id)
        return submit(case, body.kind, body.params)

    @app.get("/jobs/{job_id}", dependencies=deps)
    def get_job(job_id: str) -> JobState:
        try:
            return runner.get(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Job non trovato")

    return app
