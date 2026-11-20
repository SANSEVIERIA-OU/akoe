"""Channel simulation for adaptation and self-assessment (IMPLEMENTATION_PLAN M3).

clean speech -> cabin IR, aligned on its direct-path peak (a misaligned
convolution raises error rates, Tang et al.; aligned output stays in sync with
the transcript) -> real noise of that car at an SNR drawn from a distribution
(real noise matters as much as the IR, Szöke et al. 2019) -> optional gain ->
optional codec drawn per example (`codecs.CodecAugmenter`).

Every random choice comes from one seed and every parameter is returned, so
each output can be regenerated exactly. `simulate_dataset` writes the outputs
with SHA-256 and a manifest (seed, configuration, inputs, per-item parameters).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from pydantic import BaseModel, Field
from scipy.signal import fftconvolve

from forensicaudio.case import Case
from forensicaudio.config import Channel, RecordingSource
from forensicaudio.models import ArtifactRef, DatasetItem, RecordingCondition, VehicleProfile
from forensicaudio.vehicle.audio import (
    active_power, db, environment_info, load_audio, power, resample, to_mono, write_audio,
)
from forensicaudio.vehicle.codecs import CodecAugmenter
from forensicaudio.vehicle.profile import NoiseEstimate, direct_path_index

log = logging.getLogger(__name__)

SIM_VERSION = 1
MANIFEST_FILE = "simulation_manifest.json"


class SNRDistribution(BaseModel):
    """Distribution the per-example SNR (dB) is drawn from, clipped to [min_db, max_db]."""

    kind: Literal["uniform", "normal", "choice", "fixed"] = "uniform"
    low: float = 0.0
    high: float = 20.0
    mean: float = 10.0
    std: float = 5.0
    values: list[float] = Field(default_factory=list)
    min_db: float = -10.0
    max_db: float = 40.0

    def sample(self, rng: np.random.Generator) -> float:
        if self.kind == "uniform":
            v = rng.uniform(self.low, self.high)
        elif self.kind == "normal":
            v = rng.normal(self.mean, self.std)
        elif self.kind == "choice":
            if not self.values:
                raise ValueError("choice distribution without values")
            v = self.values[int(rng.integers(len(self.values)))]
        else:
            v = self.mean
        return float(np.clip(v, self.min_db, self.max_db))

    @classmethod
    def from_noise_estimate(
        cls, estimate: NoiseEstimate, *, min_std: float = 2.0
    ) -> "SNRDistribution":
        """Normal distribution matching the case's estimated speech-band SNR
        (median, and interquartile range / 1.349 as a robust std)."""
        snr = estimate.snr_db
        if not snr:
            raise ValueError("the noise estimate has no SNR distribution")
        std = max((snr["p75"] - snr["p25"]) / 1.349, min_std)
        return cls(kind="normal", mean=snr["p50"], std=std)


class ChannelSimConfig(BaseModel):
    snr: SNRDistribution = Field(default_factory=SNRDistribution)
    ir_probability: float = 1.0
    noise_probability: float = 1.0
    gain_db: tuple[float, float] | None = None  # uniform range, applied after mixing
    level_reference: Literal["active", "rms"] = "active"  # speech level used for the SNR
    preserve_level: bool = True  # rescale the convolved speech to the clean speech level
    direct_path_threshold_db: float = -6.0
    noise_labels: list[str] | None = None  # restrict to these noise recordings
    loop_crossfade_s: float = 0.02
    codec_probability: float = 0.0
    codecs: list[str] | None = None  # None = codecs.DEFAULT_CODECS
    codec_weights: list[float] | None = None
    peak_limit: float = 0.99


@dataclass
class LoadedIR:
    ir: np.ndarray  # float64 at the simulator rate
    direct_index: int
    sha256: str | None
    path: str | None
    source_sample_rate: int


@dataclass
class LoadedNoise:
    audio: np.ndarray  # float32 at the simulator rate
    label: str
    sha256: str | None
    path: str | None


def _tile(noise: np.ndarray, n: int, start: int, xfade: int) -> np.ndarray:
    """Loop `noise` from `start` to length `n`, crossfading at the seams."""
    out = noise[start:].astype(np.float64)
    while len(out) < n:
        nxt = noise.astype(np.float64)
        if xfade and len(out) > xfade and len(nxt) > xfade:
            r = np.linspace(0.0, 1.0, xfade)
            seam = out[-xfade:] * (1 - r) + nxt[:xfade] * r
            out = np.concatenate([out[:-xfade], seam, nxt[xfade:]])
        else:
            out = np.concatenate([out, nxt])
    return out[:n]


class ChannelSimulator:
    def __init__(
        self,
        sr: int,
        irs: Sequence[LoadedIR] = (),
        noises: Sequence[LoadedNoise] = (),
        config: ChannelSimConfig | None = None,
    ) -> None:
        self.sr = int(sr)
        self.config = config or ChannelSimConfig()
        self.irs = list(irs)
        labels = self.config.noise_labels
        self.noises = [n for n in noises if labels is None or n.label in labels]
        if not self.irs:
            log.warning("Nessuna risposta all'impulso: "
                        "la simulazione non applicherà l'acustica dell'abitacolo.")
        if not self.noises:
            log.warning("Nessuna registrazione di rumore: la simulazione non aggiungerà rumore.")
        self.codec_augmenter = None
        if self.config.codec_probability > 0:
            self.codec_augmenter = CodecAugmenter(
                codecs=self.config.codecs, weights=self.config.codec_weights,
                probability=self.config.codec_probability,
            )

    # -- construction -------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        sr: int,
        irs: Sequence[np.ndarray] = (),
        noises: Sequence[np.ndarray] = (),
        config: ChannelSimConfig | None = None,
        noise_labels: Sequence[str] | None = None,
    ) -> "ChannelSimulator":
        config = config or ChannelSimConfig()
        loaded_irs = [
            LoadedIR(np.asarray(h, dtype=np.float64),
                     direct_path_index(h, config.direct_path_threshold_db, sr=sr), None, None, sr)
            for h in irs
        ]
        labels = list(noise_labels or [f"noise-{i}" for i in range(len(noises))])
        loaded_noises = [LoadedNoise(np.asarray(a, dtype=np.float32), lab, None, None)
                         for a, lab in zip(noises, labels)]
        return cls(sr, loaded_irs, loaded_noises, config)

    @classmethod
    def from_profile(
        cls,
        profile: VehicleProfile,
        *,
        sr: int = 16000,
        root: Path | None = None,
        config: ChannelSimConfig | None = None,
        verify: bool = True,
    ) -> "ChannelSimulator":
        """Load the profile's IRs and noise recordings at `sr`.

        `root` is the case root for case-relative artifacts. With `verify`, a
        file whose SHA-256 differs from the profile raises ValueError.
        """
        config = config or ChannelSimConfig()

        def _load(ref: ArtifactRef) -> tuple[np.ndarray, int]:
            path = ref.resolve(root)
            if verify and not ref.verify(root):
                raise ValueError(f"{path}: file missing or SHA-256 differs from the profile")
            return load_audio(path)

        irs = []
        for ref in profile.impulse_responses:
            audio, s = _load(ref)
            h = resample(audio, s, sr)
            irs.append(LoadedIR(h, direct_path_index(h, config.direct_path_threshold_db, sr=sr),
                                ref.sha256, ref.path, s))
        noises = []
        for rec in profile.noise_recordings:
            if config.noise_labels is not None and rec.label not in config.noise_labels:
                continue
            audio, s = _load(rec.artifact)
            noises.append(LoadedNoise(resample(audio, s, sr).astype(np.float32), rec.label,
                                      rec.artifact.sha256, rec.artifact.path))
        return cls(sr, irs, noises, config)

    def describe(self) -> dict[str, Any]:
        return {
            "version": SIM_VERSION,
            "sample_rate": self.sr,
            "config": self.config.model_dump(mode="json"),
            "impulse_responses": [
                {"sha256": i.sha256, "path": i.path, "direct_path_index": i.direct_index,
                 "source_sample_rate": i.source_sample_rate, "n_samples": len(i.ir)}
                for i in self.irs
            ],
            "noises": [
                {"label": n.label, "sha256": n.sha256, "path": n.path,
                 "duration_s": round(len(n.audio) / self.sr, 3)} for n in self.noises
            ],
            "codecs": self.codec_augmenter.describe() if self.codec_augmenter else None,
            "environment": environment_info(),
        }

    # -- simulation ---------------------------------------------------------

    def _level(self, x: np.ndarray) -> float:
        return active_power(x, self.sr) if self.config.level_reference == "active" else power(x)

    def simulate(self, speech: np.ndarray, sr: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
        """Simulate the car channel on clean `speech`. Returns (float32 audio, params).

        Same seed, same input, same simulator -> identical output and params.
        Output has the input's length and is time-aligned with it.
        """
        if int(sr) != self.sr:
            raise ValueError(f"speech at {sr} Hz, simulator at {self.sr} Hz")
        cfg = self.config
        rng = np.random.default_rng(seed)
        x = to_mono(speech)
        n = len(x)
        # All draws happen in a fixed order, whether used or not.
        u_ir, u_noise, u_offset = rng.random(3)
        ir_idx = int(rng.integers(len(self.irs))) if self.irs else -1
        noise_idx = int(rng.integers(len(self.noises))) if self.noises else -1
        snr = cfg.snr.sample(rng)
        gain = float(rng.uniform(*cfg.gain_db)) if cfg.gain_db else 0.0
        codec_seed = int(rng.integers(2**62))

        params: dict[str, Any] = {"version": SIM_VERSION, "seed": int(seed), "sample_rate": self.sr,
                                  "n_samples": n, "ir": None, "noise": None}
        clean_level = self._level(x)

        wet = x
        if self.irs and u_ir < cfg.ir_probability:
            L = self.irs[ir_idx]
            d = L.direct_index
            wet = fftconvolve(x, L.ir)[d:d + n]
            scale = 1.0
            if cfg.preserve_level and clean_level > 0 and self._level(wet) > 0:
                scale = float(np.sqrt(clean_level / self._level(wet)))
                wet = wet * scale
            params["ir"] = {"index": ir_idx, "sha256": L.sha256, "path": L.path,
                            "direct_path_index": d,
                            "level_scale_db": round(20 * np.log10(scale), 4)}

        mix = wet
        if self.noises and u_noise < cfg.noise_probability:
            N = self.noises[noise_idx]
            looped = len(N.audio) < n
            if looped:
                start = int(u_offset * len(N.audio))
                excerpt = _tile(N.audio, n, start, int(cfg.loop_crossfade_s * self.sr))
            else:
                start = int(u_offset * (len(N.audio) - n + 1))
                excerpt = N.audio[start:start + n].astype(np.float64)
            ps, pn = self._level(wet), power(excerpt)
            if ps > 0 and pn > 0:
                k = float(np.sqrt(ps / (pn * 10 ** (snr / 10))))
                noise = excerpt * k
                mix = wet + noise
                params["noise"] = {
                    "index": noise_idx, "label": N.label, "sha256": N.sha256, "path": N.path,
                    "start_s": round(start / self.sr, 6), "looped": looped,
                    "snr_db_target": round(snr, 4),
                    "snr_db_actual": round(db(ps) - db(power(noise)), 4),
                    "noise_gain_db": round(20 * np.log10(k), 4),
                }
            else:
                log.warning("Parlato o rumore silenzioso: rumore non aggiunto.")

        mix = mix * 10 ** (gain / 20)
        params["gain_db"] = round(gain, 4)
        peak = float(np.max(np.abs(mix))) if n else 0.0
        peak_scale = cfg.peak_limit / peak if peak > cfg.peak_limit else 1.0
        mix = mix * peak_scale
        params["peak_limit_scale_db"] = round(20 * np.log10(peak_scale), 4)
        params["level_reference"] = cfg.level_reference

        codec_name = None
        params["codec"] = None
        if self.codec_augmenter is not None:
            mix, cparams = self.codec_augmenter(mix, self.sr, np.random.default_rng(codec_seed))
            params["codec"] = cparams
            codec_name = cparams.get("codec") if cparams.get("codec") != "none" else None
        params["codec_seed"] = codec_seed
        params["condition"] = RecordingCondition(
            channel=Channel.CAR,
            source=RecordingSource.TRANSMITTED if codec_name else RecordingSource.LOCAL,
            codec=codec_name, sample_rate=self.sr, notes="simulated",
        ).model_dump(mode="json")
        return np.asarray(mix, dtype=np.float32), params


# --------------------------------------------------------------------------
# Dataset generation
# --------------------------------------------------------------------------


class SimulatedItem(BaseModel):
    item_id: str
    source_item_id: str
    recording_id: str
    variant: int
    input_sha256: str
    start: float
    end: float
    text: str
    seed: int
    output: ArtifactRef
    params: dict[str, Any]
    condition: RecordingCondition


class SimulationManifest(BaseModel):
    name: str
    master_seed: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    simulator: dict[str, Any]
    items: list[SimulatedItem] = Field(default_factory=list)


def derive_seed(master_seed: int, key: str) -> int:
    """Per-item seed from the master seed and a stable key (independent of order)."""
    digest = hashlib.sha256(f"{int(master_seed)}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _ref(path: Path, kind: str, case: Case | None) -> ArtifactRef:
    """Case-relative artifact when `path` is inside the case, absolute otherwise."""
    path = Path(path).resolve()
    if case is not None and path.is_relative_to(case.root.resolve()):
        return ArtifactRef.from_file(path, kind=kind, root=case.root.resolve())
    return ArtifactRef.from_file(path, kind=kind)


def simulate_dataset(
    simulator: ChannelSimulator,
    items: Sequence[DatasetItem],
    out_dir: Path | None = None,
    *,
    seed: int,
    case: Case | None = None,
    root: Path | None = None,
    name: str = "simulated",
    variants: int = 1,
    subtype: str = "PCM_16",
    verify_inputs: bool = True,
) -> tuple[SimulationManifest, ArtifactRef]:
    """Simulate every item (`variants` times each) and write WAVs plus a manifest.

    Item audio is resolved against the case root (or `root`); outputs go to
    `working/simulated/<name>/` of the case unless `out_dir` is given. Input
    hashes are verified once per file. Returns (manifest, manifest artifact).
    """
    base = case.root if case else root
    if out_dir is None:
        if case is None:
            raise ValueError("out_dir is required without a case")
        out_dir = case.dir("working") / "simulated" / _safe(name)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = SimulationManifest(name=name, master_seed=int(seed), simulator=simulator.describe())

    cache: tuple[str, np.ndarray] | None = None
    for item in items:
        if cache is None or cache[0] != item.audio.sha256:
            path = item.audio.resolve(base)
            if verify_inputs and not item.audio.verify(base):
                raise ValueError(f"{path}: file missing or SHA-256 differs from the dataset item")
            audio, sr = load_audio(path, sr=simulator.sr)
            cache = (item.audio.sha256, audio)
        audio = cache[1]
        a, b = int(round(item.start * simulator.sr)), int(round(item.end * simulator.sr))
        segment = audio[a:b] if b > a else audio[a:]
        for v in range(variants):
            item_seed = derive_seed(seed, f"{item.item_id}#{v}")
            y, params = simulator.simulate(segment, simulator.sr, item_seed)
            wav = out_dir / f"{_safe(item.item_id)}__v{v}.wav"
            write_audio(wav, y, simulator.sr, subtype=subtype)
            ref = _ref(wav, "simulated_audio", case)
            manifest.items.append(SimulatedItem(
                item_id=f"{item.item_id}__v{v}", source_item_id=item.item_id,
                recording_id=item.recording_id, variant=v, input_sha256=item.audio.sha256,
                start=item.start, end=item.end, text=item.text, seed=item_seed, output=ref,
                params=params, condition=RecordingCondition.model_validate(params["condition"]),
            ))

    mpath = out_dir / MANIFEST_FILE
    mpath.write_text(manifest.model_dump_json(indent=2))
    mref = _ref(mpath, "simulation_manifest", case)
    log.info("Simulati %d esempi in %s (seme %d).", len(manifest.items), out_dir, seed)
    return manifest, mref


def write_simulated(
    path: Path, audio: np.ndarray, sr: int, params: dict[str, Any], *, subtype: str = "PCM_16",
    case: Case | None = None,
) -> ArtifactRef:
    """Write one simulated example and its parameters (`<name>.json` with the hash)."""
    path = Path(path)
    digest = write_audio(path, audio, sr, subtype=subtype)
    path.with_suffix(".json").write_text(json.dumps({"sha256": digest, "params": params}, indent=2))
    return _ref(path, "simulated_audio", case)
