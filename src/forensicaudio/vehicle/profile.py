"""Vehicle profile: measured impulse responses and noise, or an estimate from
the recording itself (IMPLEMENTATION_PLAN M3, MASTER_PLAN §4.3).

- Measured (preferred): IR files (from `measure.deconvolve_file`) and noise
  recordings per condition are hashed and, when a `Case` is given, copied
  read-only into `profiles/vehicle-<id>/`.
- Estimated (fallback): non-speech frames are detected over the WHOLE
  recording with a deterministic energy / spectral-flatness rule; their mean
  spectrum is the noise profile, their audio becomes a noise recording usable
  by the channel simulation, and the engine harmonics are tracked over time.

`DetailedVehicleProfile` extends the shared `VehicleProfile` with the details
(per-IR metadata, noise estimate, build parameters). Its JSON is still a valid
`VehicleProfile` for consumers that only know the base model (extra fields are
ignored by pydantic).
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.ndimage import uniform_filter1d

from forensicaudio.case import Case
from forensicaudio.models import ArtifactRef, EvidenceItem, NoiseRecording, VehicleProfile
from forensicaudio.utils import sha256_file
from forensicaudio.vehicle.audio import (
    EPS, environment_info, load_audio, resample, to_mono, write_audio,
)
from forensicaudio.vehicle.codecs import detect_codec

log = logging.getLogger(__name__)

ChannelPath = Literal["local", "transmitted", "unknown"]
PROFILE_FILE = "vehicle_profile.json"
OCTAVE_CENTERS_HZ = (63, 125, 250, 500, 1000, 2000, 4000, 8000)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


class ImpulseResponseSource(BaseModel):
    path: Path
    position: str = "driver"  # driver / passenger
    channel_path: ChannelPath = "local"
    notes: str | None = None


class NoiseSource(BaseModel):
    path: Path
    label: str  # e.g. "highway-windows-closed-vent-off"
    speed_kmh: float | None = None
    windows_open: bool | None = None
    ventilation: bool | None = None
    channel_path: ChannelPath = "local"
    notes: str | None = None


class ImpulseResponseInfo(BaseModel):
    sha256: str
    original_name: str
    position: str
    channel_path: ChannelPath
    sample_rate: int
    n_samples: int
    direct_path_index: int
    peak_to_noise_db: float | None = None
    deconvolution: dict[str, Any] | None = None  # sidecar written by measure.deconvolve_file
    notes: str | None = None


class NoiseInfo(BaseModel):
    sha256: str
    original_name: str
    label: str
    channel_path: ChannelPath
    sample_rate: int
    duration_s: float
    level_dbfs: float
    notes: str | None = None


class NoiseEstimateParams(BaseModel):
    """Non-speech detection. Deterministic; every value is logged in the profile."""

    frame_s: float = 0.032
    hop_s: float = 0.016
    band_hz: tuple[float, float] = (300.0, 3400.0)
    floor_percentile: float = 20.0  # noise floor = this percentile of band energy
    margin_db: float = 6.0  # frames above floor + margin are speech
    flatness_threshold: float = 0.2  # tonal frames above floor + margin/2 are speech too
    hangover_s: float = 0.2  # speech regions are widened by this on each side
    min_noise_run_s: float = 0.3  # shorter non-speech runs are not used as noise audio
    crossfade_s: float = 0.01
    block_frames: int = 4096


class EngineHarmonics(BaseModel):
    params: dict[str, Any]
    times_s: list[float]
    f0_hz: list[float | None]
    salience_db: list[float]
    speech: list[bool]
    summary: dict[str, Any]


class NoiseEstimate(BaseModel):
    source_sha256: str | None = None
    sample_rate: int
    duration_s: float
    params: NoiseEstimateParams
    n_frames: int
    n_nonspeech_frames: int
    nonspeech_s: float
    nonspeech_fraction: float
    noise_level_dbfs: float | None
    freqs_hz: list[float]
    psd_db: list[float]  # dB re full scale^2 / Hz
    octave_bands_db: dict[str, float]
    snr_db: dict[str, float] | None  # distribution over speech frames, speech band
    nonspeech_segments: list[tuple[float, float]]
    engine: EngineHarmonics | None = None
    warnings: list[str] = Field(default_factory=list)


class DetailedVehicleProfile(VehicleProfile):
    ir_details: list[ImpulseResponseInfo] = Field(default_factory=list)
    noise_details: list[NoiseInfo] = Field(default_factory=list)
    noise_estimate: NoiseEstimate | None = None
    measurement_metadata: dict[str, Any] = Field(default_factory=dict)
    build: dict[str, Any] = Field(default_factory=dict)


class MeasurementSession(BaseModel):
    """`misura.json` written by the operator (docs/runbook_misura_veicolo.md).
    File paths are relative to the manifest's directory. Extra keys (playback,
    environment, ...) are kept in the profile's measurement metadata."""

    model_config = ConfigDict(extra="allow")

    vehicle_id: str
    vehicle: dict[str, Any] = Field(default_factory=dict)
    date: str | None = None
    operators: list[str] = Field(default_factory=list)
    microphone: dict[str, Any] = Field(default_factory=dict)
    impulse_responses: list[ImpulseResponseSource] = Field(default_factory=list)
    noise: list[NoiseSource] = Field(default_factory=list)
    notes: str | None = None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def direct_path_index(ir: np.ndarray, threshold_db: float = -6.0, refine_s: float = 0.0003,
                      sr: int = 16000) -> int:
    """Index of the direct-path peak: the first sample within `threshold_db` of
    the absolute maximum, refined to the local maximum in the next `refine_s`."""
    a = np.abs(np.asarray(ir, dtype=np.float64))
    if not a.size or a.max() <= 0:
        return 0
    first = int(np.argmax(a >= a.max() * 10 ** (threshold_db / 20)))
    end = first + max(1, int(refine_s * sr))
    return first + int(np.argmax(a[first:end]))


def profile_dir(case: Case, vehicle_id: str) -> Path:
    return case.dir("profiles") / f"vehicle-{vehicle_id}"


def _store(src: Path, dest_dir: Path, kind: str, root: Path | None) -> ArtifactRef:
    """Copy `src` read-only into `dest_dir` (hash-named, verified) or reference it in place."""
    src = Path(src)
    digest = sha256_file(src)
    if root is None:
        return ArtifactRef(path=str(src.resolve()), sha256=digest, kind=kind)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{kind}-{digest[:12]}{src.suffix.lower()}"
    if not dest.exists():
        shutil.copyfile(src, dest)
        if sha256_file(dest) != digest:
            dest.unlink(missing_ok=True)
            raise OSError(f"copy of {src} does not match the original hash")
        dest.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    elif sha256_file(dest) != digest:
        raise OSError(f"{dest} exists with different content")
    return ArtifactRef(path=str(dest.relative_to(root)), sha256=digest, kind=kind)


def save_profile(profile: VehicleProfile, path: Path, root: Path | None = None) -> ArtifactRef:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(profile.model_dump_json(indent=2))
    return ArtifactRef.from_file(path, kind="vehicle_profile", root=root)


def load_profile(path: Path) -> DetailedVehicleProfile:
    return DetailedVehicleProfile.model_validate_json(Path(path).read_text())


# --------------------------------------------------------------------------
# Measured profile
# --------------------------------------------------------------------------


def build_vehicle_profile(
    vehicle_id: str,
    impulse_responses: Sequence[ImpulseResponseSource | Path] = (),
    noise_recordings: Sequence[NoiseSource] = (),
    *,
    case: Case | None = None,
    metadata: dict[str, Any] | None = None,
    notes: str | None = None,
) -> DetailedVehicleProfile:
    """Profile from measured IRs and noise recordings, every file hashed.

    With a case the files are copied read-only under `profiles/vehicle-<id>/`
    and the profile is saved there as `vehicle_profile.json`; without a case
    the artifacts reference the files in place (absolute paths).
    """
    root = case.root if case else None
    base = profile_dir(case, vehicle_id) if case else None
    profile = DetailedVehicleProfile(vehicle_id=vehicle_id, estimated=False, notes=notes,
                                     measurement_metadata=dict(metadata or {}))

    for src in impulse_responses:
        if not isinstance(src, ImpulseResponseSource):
            src = ImpulseResponseSource(path=Path(src))
        audio, sr = load_audio(src.path)
        ref = _store(src.path, base / "ir" if base else Path(), "vehicle_ir", root)
        sidecar = src.path.with_suffix(".json")
        deconv = None
        if sidecar.exists():
            deconv = json.loads(sidecar.read_text())
            deconv["sidecar_sha256"] = sha256_file(sidecar)
            if base is not None:
                _store(sidecar, base / "ir", "vehicle_ir_sidecar", root)
        tail = audio[int(0.9 * len(audio)):]
        peak = float(np.max(np.abs(audio)))
        noise = float(np.sqrt(np.mean(tail.astype(np.float64) ** 2)))
        profile.impulse_responses.append(ref)
        profile.ir_details.append(ImpulseResponseInfo(
            sha256=ref.sha256, original_name=src.path.name, position=src.position,
            channel_path=src.channel_path, sample_rate=sr, n_samples=len(audio),
            direct_path_index=direct_path_index(audio, sr=sr),
            peak_to_noise_db=float(20 * np.log10(peak / noise)) if noise > 0 and peak > 0 else None,
            deconvolution=deconv, notes=src.notes,
        ))

    for src in noise_recordings:
        audio, sr = load_audio(src.path)
        ref = _store(src.path, base / "noise" if base else Path(), "vehicle_noise", root)
        profile.noise_recordings.append(NoiseRecording(
            label=src.label, artifact=ref, speed_kmh=src.speed_kmh,
            windows_open=src.windows_open, ventilation=src.ventilation,
        ))
        p = float(np.mean(audio.astype(np.float64) ** 2)) if len(audio) else 0.0
        profile.noise_details.append(NoiseInfo(
            sha256=ref.sha256, original_name=src.path.name, label=src.label,
            channel_path=src.channel_path, sample_rate=sr, duration_s=len(audio) / sr,
            level_dbfs=float(10 * np.log10(max(p, EPS))), notes=src.notes,
        ))
        codec = detect_codec(src.path).get("codec")
        if codec and codec not in profile.observed_codecs and not codec.startswith("pcm_"):
            profile.observed_codecs.append(codec)

    profile.build = {
        "function": "build_vehicle_profile",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment_info(),
    }
    if case is not None:
        save_profile(profile, base / PROFILE_FILE, root)
    return profile


def build_profile_from_manifest(
    manifest: Path, *, case: Case | None = None
) -> DetailedVehicleProfile:
    """Profile from the `misura.json` session manifest described in the runbook."""
    manifest = Path(manifest)
    session = MeasurementSession.model_validate_json(manifest.read_text())
    here = manifest.parent

    def _abs(p: Path) -> Path:
        return p if p.is_absolute() else here / p

    irs = [s.model_copy(update={"path": _abs(s.path)}) for s in session.impulse_responses]
    noise = [s.model_copy(update={"path": _abs(s.path)}) for s in session.noise]
    meta = session.model_dump(mode="json", exclude={"impulse_responses", "noise"})
    meta["manifest_sha256"] = sha256_file(manifest)
    if case is not None:
        _store(manifest, profile_dir(case, session.vehicle_id), "measurement_manifest", case.root)
    return build_vehicle_profile(session.vehicle_id, irs, noise, case=case, metadata=meta,
                                 notes=session.notes)


# --------------------------------------------------------------------------
# Estimated profile
# --------------------------------------------------------------------------


def _frames(x: np.ndarray, frame: int, hop: int, start: int, count: int) -> np.ndarray:
    idx = np.arange(frame)[None, :] + hop * (start + np.arange(count))[:, None]
    return x[idx]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs where mask is True."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def detect_nonspeech(
    audio: np.ndarray, sr: int, params: NoiseEstimateParams | None = None
) -> dict[str, Any]:
    """Frame-level speech / non-speech decision over the whole signal.

    Returns {speech_raw, nonspeech (after hangover), band_power, flatness,
    floor_db, frame, hop}. Arrays are per frame.
    """
    p = params or NoiseEstimateParams()
    x = to_mono(audio)
    frame, hop = max(16, int(round(p.frame_s * sr))), max(1, int(round(p.hop_s * sr)))
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n = 1 + (len(x) - frame) // hop
    win = np.hanning(frame)
    freqs = np.fft.rfftfreq(frame, 1 / sr)
    band = (freqs >= p.band_hz[0]) & (freqs <= min(p.band_hz[1], sr / 2))
    eb, flat = np.empty(n), np.empty(n)
    for s in range(0, n, p.block_frames):
        c = min(p.block_frames, n - s)
        P = np.abs(np.fft.rfft(_frames(x, frame, hop, s, c) * win, axis=1)) ** 2
        Pb = P[:, band] + EPS
        eb[s:s + c] = Pb.sum(axis=1)
        flat[s:s + c] = np.exp(np.mean(np.log(Pb), axis=1)) / np.mean(Pb, axis=1)
    eb_db = 10 * np.log10(eb + EPS)
    floor_db = float(np.percentile(eb_db, p.floor_percentile))
    speech = (eb_db > floor_db + p.margin_db) | (
        (eb_db > floor_db + p.margin_db / 2) & (flat < p.flatness_threshold))
    k = int(round(p.hangover_s / p.hop_s))
    widened = speech
    if k > 0:
        widened = np.convolve(speech.astype(float), np.ones(2 * k + 1), mode="same") > 0
    return {"speech_raw": speech, "nonspeech": ~widened, "band_power": eb, "flatness": flat,
            "floor_db": floor_db, "frame": frame, "hop": hop}


def _concat_runs(x: np.ndarray, runs: list[tuple[int, int]], xfade: int) -> np.ndarray:
    out = np.zeros(0)
    for a, b in runs:
        seg = x[a:b].astype(np.float64)
        if out.size and xfade > 0 and len(seg) > xfade and len(out) > xfade:
            ramp = np.linspace(0.0, 1.0, xfade)
            out[-xfade:] = out[-xfade:] * (1 - ramp) + seg[:xfade] * ramp
            seg = seg[xfade:]
        out = np.concatenate([out, seg])
    return out


def track_engine_harmonics(
    audio: np.ndarray,
    sr: int,
    *,
    speech_mask: np.ndarray | None = None,
    frame_s: float = 1.0,
    hop_s: float = 0.5,
    fmin_hz: float = 20.0,
    fmax_hz: float = 200.0,
    step_hz: float = 0.25,
    n_harmonics: int = 4,
    min_salience_db: float = 9.0,
    analysis_sr: int = 2000,
) -> EngineHarmonics:
    """Track the engine firing frequency f0 over time with a harmonic sum.

    Salience(f0) = mean peak height (dB above the local spectral mean) at
    k*f0 minus half of it at (k-1/2)*f0, which penalizes octave errors. A frame
    is a detection when salience >= `min_salience_db`. `speech_mask` (per
    sample at `sr`) flags frames dominated by speech; the summary uses only
    non-speech frames.
    """
    params = {k: v for k, v in locals().items() if k not in ("audio", "sr", "speech_mask")}
    x = resample(to_mono(audio), sr, analysis_sr)
    frame, hop = int(round(frame_s * analysis_sr)), int(round(hop_s * analysis_sr))
    nfft = 1 << int(np.ceil(np.log2(max(frame, analysis_sr / 0.25))))
    df = analysis_sr / nfft
    fmax_bin = min(nfft // 2, int(np.ceil((n_harmonics + 1) * fmax_hz / df)) + 1)
    grid = np.arange(fmin_hz, fmax_hz + step_hz / 2, step_hz)
    k = np.arange(1, n_harmonics + 1)
    on = np.clip(np.round(np.outer(grid, k) / df).astype(int), 0, fmax_bin - 1)
    off = np.clip(np.round(np.outer(grid, k - 0.5) / df).astype(int), 0, fmax_bin - 1)
    smooth = max(3, int(round(20.0 / df)) | 1)

    times, f0s, sals, speech = [], [], [], []
    n = 0 if len(x) < frame else 1 + (len(x) - frame) // hop
    win = np.hanning(frame)
    for s in range(0, n, 256):
        c = min(256, n - s)
        X = np.fft.rfft(_frames(x, frame, hop, s, c) * win, nfft, axis=1)[:, :fmax_bin]
        L = 20 * np.log10(np.abs(X) + EPS)
        P = np.maximum(L - uniform_filter1d(L, smooth, axis=1, mode="nearest"), 0.0)
        sal = P[:, on].mean(axis=2) - 0.5 * P[:, off].mean(axis=2)
        best = np.argmax(sal, axis=1)
        for i in range(c):
            t0 = (s + i) * hop / analysis_sr
            sv = float(sal[i, best[i]])
            times.append(round(t0 + frame_s / 2, 3))
            sals.append(round(sv, 2))
            f0s.append(round(float(grid[best[i]]), 2) if sv >= min_salience_db else None)
            if speech_mask is not None:
                a, b = int(t0 * sr), int((t0 + frame_s) * sr)
                speech.append(bool(np.mean(speech_mask[a:b]) > 0.5) if b > a else False)
            else:
                speech.append(False)

    usable = [f for f, sp in zip(f0s, speech) if f is not None and not sp]
    n_ns = sum(1 for sp in speech if not sp)
    summary: dict[str, Any] = {
        "n_frames": len(times),
        "n_nonspeech_frames": n_ns,
        "n_detected_nonspeech": len(usable),
        "detected_fraction_nonspeech": round(len(usable) / n_ns, 3) if n_ns else 0.0,
        "basis": "nonspeech_frames",
    }
    if len(usable) < 5:
        # Too little speech-free material: summarize every frame (speech
        # harmonics may then be mistaken for the engine; flagged by `basis`).
        usable = [f for f in f0s if f is not None]
        summary["basis"] = "all_frames"
    summary["n_detected_used"] = len(usable)
    if usable:
        u = np.asarray(usable)
        summary.update(
            f0_median_hz=round(float(np.median(u)), 2),
            f0_p10_hz=round(float(np.percentile(u, 10)), 2),
            f0_p90_hz=round(float(np.percentile(u, 90)), 2),
            # Firing frequency = rpm / 30 for a 4-cylinder 4-stroke engine (assumption).
            rpm_median_if_4cyl_4stroke=round(float(np.median(u)) * 30, 0),
        )
    return EngineHarmonics(params=params, times_s=times, f0_hz=f0s, salience_db=sals,
                           speech=speech, summary=summary)


def estimate_noise(
    audio: np.ndarray,
    sr: int,
    params: NoiseEstimateParams | None = None,
    *,
    track_engine: bool = True,
) -> tuple[NoiseEstimate, np.ndarray]:
    """Noise profile from the non-speech frames of the whole signal.

    Returns (estimate, noise_audio) where noise_audio is the concatenation of
    the non-speech runs longer than `min_noise_run_s` (crossfaded).
    """
    p = params or NoiseEstimateParams()
    x = to_mono(audio)
    det = detect_nonspeech(x, sr, p)
    frame, hop, ns = det["frame"], det["hop"], det["nonspeech"]
    n = len(ns)
    warnings: list[str] = []

    win = np.hanning(frame)
    scale = 2.0 / (sr * np.sum(win**2))
    freqs = np.fft.rfftfreq(frame, 1 / sr)
    acc = np.zeros(len(freqs))
    idx = np.flatnonzero(ns)
    for s in range(0, len(idx), p.block_frames):
        sel = idx[s:s + p.block_frames]
        fr = x[np.arange(frame)[None, :] + hop * sel[:, None]]
        acc += (np.abs(np.fft.rfft(fr * win, axis=1)) ** 2).sum(axis=0)
    if len(idx):
        psd = acc / len(idx) * scale
    else:
        psd = np.full(len(freqs), np.nan)
        warnings.append("Nessun tratto senza voce rilevato: profilo del rumore non stimabile.")

    # Sample-level non-speech runs (frame hop grid), and the noise audio.
    sample_mask = np.zeros(len(x), dtype=bool)
    for a, b in _runs(ns):
        sample_mask[a * hop: min(len(x), (b - 1) * hop + frame)] = True
    runs = [(a, b) for a, b in _runs(sample_mask) if (b - a) >= p.min_noise_run_s * sr]
    noise_audio = _concat_runs(x, runs, int(p.crossfade_s * sr))

    octave: dict[str, float] = {}
    if len(idx):
        for fc in OCTAVE_CENTERS_HZ:
            sel = (freqs >= fc / np.sqrt(2)) & (freqs < fc * np.sqrt(2))
            if fc * np.sqrt(2) <= sr / 2 and sel.any():
                band_power = np.sum(psd[sel]) * (freqs[1] - freqs[0])
                octave[str(fc)] = round(float(10 * np.log10(band_power + EPS)), 2)

    snr = None
    speech = det["speech_raw"]
    if len(idx) and speech.any():
        nb = float(np.mean(det["band_power"][ns]))
        v = 10 * np.log10(np.maximum(det["band_power"][speech] - nb, EPS) / max(nb, EPS))
        snr = {f"p{q}": round(float(np.percentile(v, q)), 2) for q in (10, 25, 50, 75, 90)}
        snr.update(mean=round(float(v.mean()), 2), std=round(float(v.std()), 2),
                   n_frames=int(v.size))

    nonspeech_s = float(sample_mask.sum() / sr)
    if len(x) and nonspeech_s / (len(x) / sr) < 0.1:
        warnings.append("Meno del 10% della registrazione è senza voce: "
                        "stima del rumore poco affidabile.")

    engine = None
    if track_engine:
        engine = track_engine_harmonics(x, sr, speech_mask=~sample_mask)

    est = NoiseEstimate(
        sample_rate=int(sr),
        duration_s=len(x) / sr,
        params=p,
        n_frames=n,
        n_nonspeech_frames=int(len(idx)),
        nonspeech_s=round(nonspeech_s, 3),
        nonspeech_fraction=round(nonspeech_s / (len(x) / sr), 4) if len(x) else 0.0,
        noise_level_dbfs=(round(float(10 * np.log10(np.mean(noise_audio**2) + EPS)), 2)
                          if noise_audio.size else None),
        freqs_hz=[round(float(f), 2) for f in freqs],
        psd_db=[round(float(10 * np.log10(max(v, 1e-30))), 2) if np.isfinite(v) else -300.0
                for v in psd],
        octave_bands_db=octave,
        snr_db=snr,
        nonspeech_segments=[(round(a / sr, 3), round(b / sr, 3)) for a, b in _runs(sample_mask)],
        engine=engine,
        warnings=warnings,
    )
    for w in warnings:
        log.warning(w)
    return est, noise_audio.astype(np.float32)


def estimate_vehicle_profile(
    vehicle_id: str,
    recording: Path | EvidenceItem,
    *,
    case: Case | None = None,
    out_dir: Path | None = None,
    params: NoiseEstimateParams | None = None,
    track_engine: bool = True,
) -> DetailedVehicleProfile:
    """Fallback profile estimated from a recording (estimated=True).

    The non-speech audio is written as a read-only noise recording (FLOAT WAV)
    under `profiles/vehicle-<id>/` of the case, or `out_dir`, together with
    `vehicle_profile_estimated-<source>-<params>.json` (a measured profile in
    the same folder is never overwritten); without either, the profile holds
    only the spectral estimate.
    """
    root = case.root if case else None
    if isinstance(recording, EvidenceItem):
        if case is None:
            raise ValueError("an EvidenceItem needs its case")
        path = case.path_of(recording)
    else:
        path = Path(recording)
    digest = sha256_file(path)
    audio, sr = load_audio(path)
    est, noise_audio = estimate_noise(audio, sr, params, track_engine=track_engine)
    est.source_sha256 = digest

    profile = DetailedVehicleProfile(
        vehicle_id=vehicle_id,
        estimated=True,
        noise_estimate=est,
        notes=f"Profilo stimato dai tratti senza voce di {path.name} (sha256 {digest[:12]}).",
    )
    codec = detect_codec(path).get("codec")
    if codec and not codec.startswith("pcm_"):
        profile.observed_codecs.append(codec)

    target = profile_dir(case, vehicle_id) if case else (Path(out_dir) if out_dir else None)
    # Source and parameters identify the estimate: re-running it reuses the files.
    tag = f"{digest[:12]}-{hashlib.sha256(est.params.model_dump_json().encode()).hexdigest()[:8]}"
    if target is not None and noise_audio.size:
        wav = target / "noise" / f"vehicle_noise_estimated-{tag}.wav"
        _write_once(wav, noise_audio, sr)
        ref = ArtifactRef.from_file(wav if root else wav.resolve(), kind="vehicle_noise_estimated",
                                    root=root)
        profile.noise_recordings.append(NoiseRecording(label="estimated-nonspeech", artifact=ref))
        profile.noise_details.append(NoiseInfo(
            sha256=ref.sha256, original_name=wav.name, label="estimated-nonspeech",
            channel_path="unknown", sample_rate=sr, duration_s=len(noise_audio) / sr,
            level_dbfs=est.noise_level_dbfs if est.noise_level_dbfs is not None else -240.0,
            notes=f"Tratti senza voce di {path.name}",
        ))
    elif noise_audio.size == 0:
        log.warning("Nessun tratto senza voce abbastanza lungo: "
                    "nessun rumore utilizzabile per la simulazione.")

    profile.build = {
        "function": "estimate_vehicle_profile",
        "source": {"file": path.name, "sha256": digest,
                   "evidence_id": (recording.evidence_id if isinstance(recording, EvidenceItem)
                                   else None)},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment_info(),
    }
    if target is not None:
        save_profile(profile, target / f"vehicle_profile_estimated-{tag}.json", root)
    return profile


def _write_once(wav: Path, audio: np.ndarray, sr: int) -> None:
    """Write a read-only FLOAT WAV; if it already exists it must be identical."""
    tmp = wav.with_name(wav.stem + ".tmp" + wav.suffix)
    write_audio(tmp, audio, sr, subtype="FLOAT")
    if wav.exists():
        same = sha256_file(tmp) == sha256_file(wav)
        tmp.unlink()
        if not same:
            raise OSError(f"{wav} exists with different content")
        return
    tmp.rename(wav)
    wav.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
