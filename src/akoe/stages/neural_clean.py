# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Neural cleaning for HUMAN LISTENING (M6, issue #59).

Principle (CLAUDE.md invariants 1 and 7, MASTER_PLAN §3.5.1, §4.4): the
output of a neural enhancer is a **listening version** for the human
transcriber, labelled ``StageKind.NEURAL``. It is never merged with
deterministic audio and is never the automatic transcriber's input unless
the self-assessment shows that a remix with the original helps
(:mod:`akoe.stages.remix`).

This module holds the backend-independent part:

- :class:`NeuralEnhancer`, the protocol every backend implements
  (:mod:`akoe.stages.neural_backends`: UniPASE, MossFormer2_SE_48K, GTCRN,
  DeepFilterNet 3);
- :func:`enhance`: sample-rate conversion to the model rate and back,
  chunking of long recordings with overlap and raised-cosine cross-fade,
  optional level matching, and the full parameter record (model id,
  revision, SHA-256 of every weight and code file, device, chunking, rates,
  timing);
- pinned downloads: every weight and every third-party source file is
  fetched at a fixed revision and checked against a SHA-256 written in this
  repository before it is used (or unpickled);
- defaults per hardware tier (``docs/catalogo_modelli.md``).
"""
from __future__ import annotations

import importlib
import os
import sys
import threading
import time
import types
import urllib.request
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from akoe.config import StageKind, Tier
from akoe.neural import BackendUnavailable, ModelProvenance, sha256_cached
from akoe.stages.common import as_float, to_jsonable, versions

KIND = StageKind.NEURAL
INTENDED_USE = "human_listening"
ALGORITHM_VERSION = "1"
VARIANT_PREFIX = "neural_"
MIN_MODEL_INPUT_S = 0.5  # shorter inputs are zero-padded for the model, then trimmed

#: Backends in order of preference per tier (decision 2026-11-20, measured on
#: 7 real Italian clips in car noise: MossFormer2 gave the best ESTOI gain
#: (+0.148) and lowered Whisper WER 0.327 -> 0.286, while UniPASE resynthesised
#: the speech and raised WER to 0.406; see docs/catalogo_modelli.md). On EDGE
#: GTCRN gives a quick listening version, MossFormer2 the offline one
#: (~0.4x real time on CPU). UniPASE stays available on request.
DEFAULT_BACKENDS: dict[Tier, tuple[str, ...]] = {
    Tier.EDGE: ("gtcrn", "mossformer2_se_48k"),
    Tier.WORKSTATION: ("mossformer2_se_48k",),
    Tier.CLOUD: ("mossformer2_se_48k",),
}

_TAIL_IT = ("Ipotesi neurale, solo per l'ascolto del trascrittore umano: non è l'audio originale, "
            "può attenuare o alterare parti della voce e non è l'ingresso del trascrittore "
            "automatico. Confrontare sempre con l'originale.")
LABELS_IT: dict[str, str] = {
    "gtcrn": "Pulizia neurale GTCRN (modello piccolo e rapido su CPU). ",
    "deepfilternet3": "Pulizia neurale DeepFilterNet 3 (filtraggio profondo a 48 kHz). ",
    "mossformer2_se_48k": "Pulizia neurale MossFormer2_SE_48K (maschera spettrale a 48 kHz). ",
    "unipase": ("Pulizia neurale UniPASE: modello generativo a bassa allucinazione che "
                "ricostruisce la voce da rappresentazioni WavLM; può produrre suoni plausibili "
                "assenti nell'originale. "),
}


def label_it(backend: str) -> str:
    return LABELS_IT.get(backend, f"Pulizia neurale {backend}. ") + _TAIL_IT


def variant_name(backend: str) -> str:
    """Name of the listening variant produced by ``backend``, e.g. ``neural_gtcrn``."""
    return f"{VARIANT_PREFIX}{backend}"


def is_neural_variant(name: str) -> bool:
    return name.startswith(VARIANT_PREFIX) and name[len(VARIANT_PREFIX):] in available_backends()


def available_backends() -> tuple[str, ...]:
    from akoe.stages.neural_backends import BACKENDS

    return tuple(BACKENDS)


def default_backends(tier: Tier) -> tuple[str, ...]:
    return DEFAULT_BACKENDS[Tier(tier)]


def resolve_backends(spec: str | Sequence[str] | None, tier: Tier) -> list[str]:
    """Expand ``"default"`` to the tier defaults, check names, drop duplicates."""
    if spec is None:
        return []
    names = [spec] if isinstance(spec, str) else list(spec)
    out: list[str] = []
    for n in names:
        for b in (default_backends(tier) if n == "default" else (n,)):
            if b not in available_backends():
                raise ValueError(f"Backend di pulizia neurale sconosciuto: {b!r}; disponibili: "
                                 f"{', '.join(available_backends())} (oppure 'default').")
            if b not in out:
                out.append(b)
    return out


def get_enhancer(name: str, *, device: str = "cpu", local_files_only: bool = False,
                 **kwargs: Any) -> "NeuralEnhancer":
    """Instantiate a backend by name. Nothing is loaded or downloaded until
    :meth:`NeuralEnhancer.load` (or :func:`enhance`) is called."""
    from akoe.stages.neural_backends import BACKENDS

    if name not in BACKENDS:
        raise ValueError(f"Backend di pulizia neurale sconosciuto: {name!r}; disponibili: "
                         f"{', '.join(BACKENDS)}.")
    return BACKENDS[name](device=device, local_files_only=local_files_only, **kwargs)


# --------------------------------------------------------------------------
# Backend protocol
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunking:
    """How long recordings are cut. ``chunk_s=None``: the whole signal at
    once. Consecutive chunks overlap by at least ``overlap_s``; inside each
    overlap the output switches from one chunk to the next with a
    raised-cosine (equal-gain) cross-fade of ``crossfade_s`` centred in the
    overlap, so each chunk contributes only where it has context on both
    sides."""

    chunk_s: float | None = None
    overlap_s: float = 0.0
    crossfade_s: float = 0.0

    def __post_init__(self) -> None:
        if self.chunk_s is not None:
            if self.chunk_s <= 0 or not 0 <= self.overlap_s < self.chunk_s:
                raise ValueError("chunking: serve 0 <= overlap_s < chunk_s")
            if not 0 <= self.crossfade_s <= self.overlap_s:
                raise ValueError("chunking: serve 0 <= crossfade_s <= overlap_s")


@runtime_checkable
class NeuralEnhancer(Protocol):
    """A speech enhancer. ``enhance_chunk`` takes mono float32 audio at
    ``native_sr`` and returns audio of the same length and rate."""

    name: str
    native_sr: int
    chunking: Chunking

    def load(self) -> ModelProvenance: ...

    def enhance_chunk(self, audio: np.ndarray) -> np.ndarray: ...

    def describe(self) -> dict[str, Any]: ...


# --------------------------------------------------------------------------
# Chunking and resampling
# --------------------------------------------------------------------------


def chunk_bounds(n: int, chunk: int, overlap: int) -> list[tuple[int, int]]:
    """Full-length chunks, evenly spaced, first at 0 and last ending at
    ``n``; consecutive chunks overlap by at least ``overlap`` samples."""
    if chunk <= 0 or n <= chunk:
        return [(0, n)]
    hop = chunk - overlap
    if hop <= 0:
        raise ValueError("overlap must be smaller than the chunk")
    k = int(np.ceil((n - chunk) / hop)) + 1
    starts = [int(round(i * (n - chunk) / (k - 1))) for i in range(k)]
    return [(s, s + chunk) for s in starts]


def _ramp(f: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(np.pi * (np.arange(f) + 0.5) / f) if f > 0 else np.zeros(0)


def chunk_weights(bounds: list[tuple[int, int]], crossfade: int) -> tuple[list[np.ndarray], int]:
    """Per-chunk weights summing to one at every sample. Returns the weights
    and the cross-fade length actually used (clamped to the overlap and to
    the chunk spacing so that fades never overlap each other)."""
    weights = [np.ones(e - s) for s, e in bounds]
    if len(bounds) < 2:
        return weights, 0
    spacing = min(bounds[i + 1][0] - bounds[i][0] for i in range(len(bounds) - 1))
    used = crossfade
    for i in range(len(bounds) - 1):
        (s0, e0), (s1, _) = bounds[i], bounds[i + 1]
        ov = e0 - s1
        f = max(0, min(crossfade, ov, spacing))
        used = min(used, f)
        a = s1 + (ov - f) // 2  # start of the fade, absolute
        r = _ramp(f)
        w0, w1 = weights[i], weights[i + 1]
        w0[a - s0: a - s0 + f] *= 1.0 - r
        w0[a - s0 + f:] = 0.0
        w1[: a - s1] = 0.0
        w1[a - s1: a - s1 + f] *= r
    return weights, int(used)


def process_chunked(fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray, sr: int,
                    chunking: Chunking,
                    progress: Callable[[float, str], None] | None = None,
                    ) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply ``fn`` chunk by chunk and stitch the outputs with cross-fades."""
    n = len(x)
    chunk = int(round(chunking.chunk_s * sr)) if chunking.chunk_s else 0
    overlap = int(round(chunking.overlap_s * sr))
    bounds = chunk_bounds(n, chunk, overlap)
    weights, used = chunk_weights(bounds, int(round(chunking.crossfade_s * sr)))
    y = np.zeros(n, dtype=np.float64)
    for i, ((s, e), w) in enumerate(zip(bounds, weights)):
        if progress is not None:
            progress(i / len(bounds), f"Pulizia neurale: blocco {i + 1} di {len(bounds)}")
        out = np.asarray(fn(np.ascontiguousarray(x[s:e], dtype=np.float32)), dtype=np.float64)
        out = out.reshape(-1)
        if len(out) < e - s:
            out = np.pad(out, (0, e - s - len(out)))
        y[s:e] += w * out[: e - s]
    info = {
        "chunk_s": chunking.chunk_s, "overlap_s": chunking.overlap_s,
        "crossfade_s": chunking.crossfade_s, "crossfade_used_s": used / sr if sr else 0.0,
        "crossfade": "raised_cosine_equal_gain_centred_in_overlap",
        "n_chunks": len(bounds),
        "chunk_starts_s": [round(s / sr, 6) for s, _ in bounds],
    }
    return y, info


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Polyphase resampling (``scipy.signal.resample_poly``), deterministic."""
    if sr_from == sr_to:
        return np.asarray(x, dtype=np.float64)
    from scipy.signal import resample_poly

    g = gcd(int(sr_from), int(sr_to))
    return resample_poly(np.asarray(x, dtype=np.float64), int(sr_to) // g, int(sr_from) // g)


def _fit_length(y: np.ndarray, n: int) -> np.ndarray:
    return np.pad(y, (0, n - len(y))) if len(y) < n else y[:n]


def _runtime_versions() -> dict[str, str]:
    out = dict(versions())
    for mod in ("torch", "torchaudio", "onnxruntime"):
        m = sys.modules.get(mod)
        if m is not None and hasattr(m, "__version__"):
            out[mod] = str(m.__version__)
    return out


def _threads() -> dict[str, Any]:
    out: dict[str, Any] = {}
    torch = sys.modules.get("torch")
    if torch is not None:
        out["torch_num_threads"] = int(torch.get_num_threads())
    return out


# --------------------------------------------------------------------------
# Enhancement with full parameter record
# --------------------------------------------------------------------------


def enhance(
    audio: np.ndarray,
    sr: int,
    enhancer: NeuralEnhancer,
    *,
    chunking: Chunking | None = None,
    match_level: str | None = None,
    peak_limit: float = 0.99,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run ``enhancer`` on mono ``audio`` at ``sr``. Returns ``(audio, params)``.

    The output has the input's sample rate and length, so that it stays
    aligned with the original for A/B listening and for :func:`remix.remix`.
    ``match_level``: ``"none"``, ``"rms"`` or ``"peak"`` (match the input);
    default from the backend (``describe()["match_level"]``, else
    ``"none"``). If the output peak exceeds 1.0 it is scaled to
    ``peak_limit`` (logged), never clipped silently.
    """
    x = as_float(audio)
    if sr <= 0:
        raise ValueError(f"Frequenza di campionamento non valida: {sr}.")
    if len(x) == 0:
        raise ValueError("Audio vuoto: niente da pulire.")
    prov = enhancer.load()
    desc = dict(enhancer.describe())
    plan = chunking or enhancer.chunking
    level = match_level or desc.get("match_level") or "none"
    if level not in ("none", "rms", "peak"):
        raise ValueError(f"match_level sconosciuto: {level!r} (none, rms, peak)")

    native = enhancer.native_sr

    min_len = int(MIN_MODEL_INPUT_S * native)

    def run_chunk(chunk: np.ndarray) -> np.ndarray:
        # Chunks are cut at the input rate and resampled one by one, so long
        # recordings never exist at the (possibly higher) model rate in full.
        # Resampling edge effects fall inside the overlaps, away from the
        # cross-fades. Very short inputs are zero-padded for the model and
        # trimmed back.
        xm = resample(chunk, sr, native)
        n_native = len(xm)
        if n_native < min_len:
            xm = np.pad(xm, (0, min_len - n_native))
        ym = np.asarray(enhancer.enhance_chunk(xm.astype(np.float32)), dtype=np.float64).reshape(-1)
        return _fit_length(resample(_fit_length(ym, n_native), native, sr), len(chunk))

    t0 = time.perf_counter()
    y, chunk_info = process_chunked(run_chunk, x, sr, plan, progress)
    elapsed = time.perf_counter() - t0

    gain = 1.0
    if level == "rms":
        rx, ry = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0, float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
        gain = rx / ry if ry > 0 else 1.0
    elif level == "peak":
        px, py = float(np.max(np.abs(x))) if len(x) else 0.0, float(np.max(np.abs(y))) if len(y) else 0.0
        gain = px / py if py > 0 else 1.0
    y = y * gain
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    protect = peak_limit / peak if peak > 1.0 else 1.0
    y = y * protect

    duration = len(x) / sr
    params = {
        "algorithm": "neural_speech_enhancement",
        "algorithm_version": ALGORITHM_VERSION,
        "kind": KIND.value,
        "backend": enhancer.name,
        "variant": variant_name(enhancer.name),
        "intended_use": INTENDED_USE,
        "label_it": label_it(enhancer.name),
        "model_id": prov.model_id,
        "model_revision": prov.revision,
        "model_weights_sha256": dict(prov.weights),
        "model_weights_digest": prov.weights_digest(),
        "model_license": prov.license,
        "device": prov.device,
        "dtype": prov.dtype,
        "model_provenance": prov.as_params(),
        "input_sr": sr,
        "model_sr": enhancer.native_sr,
        "output_sr": sr,
        "resampling": ("none" if sr == native
                       else "scipy.signal.resample_poly per chunk (in and out)"),
        "chunking": chunk_info,
        "backend_params": desc,
        "match_level": level,
        "level_gain": gain,
        "peak_limit": peak_limit,
        "peak_protection_gain": protect,
        "audio_s": duration,
        "processing_s": elapsed,
        "real_time_factor": elapsed / duration if duration > 0 else None,
        **_threads(),
        "versions": _runtime_versions(),
    }
    return y, to_jsonable(params)


# --------------------------------------------------------------------------
# Pinned downloads and third-party code snapshots
# --------------------------------------------------------------------------


def model_dir() -> Path:
    """Cache of files fetched outside the Hugging Face hub (``AKOE_MODEL_DIR``
    or ``~/.cache/akoe/models``)."""
    return Path(os.environ.get("AKOE_MODEL_DIR", Path.home() / ".cache" / "akoe" / "models"))


@dataclass(frozen=True)
class PinnedFile:
    """A file at a fixed URL (a commit or a tag) with its expected SHA-256."""

    url: str
    sha256: str
    name: str  # path under the cache directory


_DL_LOCK = threading.Lock()


def _check_sha(path: Path, expected: str, what: str) -> str:
    actual = sha256_cached(path)
    if actual != expected:
        raise RuntimeError(
            f"{what}: SHA-256 {actual} diverso da quello verificato ({expected}). Il file è "
            f"stato cambiato: non viene usato ({path}).")
    return actual


def fetch_pinned(f: PinnedFile, *, local_files_only: bool = False, timeout: float = 120.0) -> Path:
    """Return the cached copy of ``f`` (downloading it if allowed), always
    checked against its SHA-256 before use."""
    dest = model_dir() / f.name
    with _DL_LOCK:
        if not dest.exists():
            if local_files_only:
                raise BackendUnavailable(
                    f"File del modello non presente in locale: {dest}. Scaricarlo una volta con "
                    f"accesso alla rete (local_files_only=False) da {f.url}, oppure copiarlo in "
                    "quella posizione (AKOE_MODEL_DIR).")
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex[:8]}.part")
            try:
                req = urllib.request.Request(f.url, headers={"User-Agent": "akoe-model-fetch"})
                with urllib.request.urlopen(req, timeout=timeout) as r, tmp.open("wb") as fh:
                    while True:
                        block = r.read(1 << 20)
                        if not block:
                            break
                        fh.write(block)
                _check_sha(tmp, f.sha256, f.name)
                os.replace(tmp, dest)
            except OSError as e:
                raise BackendUnavailable(f"Scaricamento non riuscito di {f.url}: {e}") from e
            finally:
                tmp.unlink(missing_ok=True)
    _check_sha(dest, f.sha256, f.name)
    return dest


def fetch_hf_pinned(repo: str, filename: str, revision: str, sha256: str, *,
                    local_files_only: bool = False) -> Path:
    """Hugging Face file at a pinned commit, checked against its SHA-256."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover
        raise BackendUnavailable("Serve huggingface_hub per scaricare i pesi del modello.") from e
    try:
        path = Path(hf_hub_download(repo, filename, revision=revision,
                                    local_files_only=local_files_only))
    except Exception as e:  # LocalEntryNotFoundError, network errors...
        raise BackendUnavailable(
            f"Pesi {repo}/{filename}@{revision[:12]} non disponibili"
            f"{' in locale' if local_files_only else ''}: {e}") from e
    _check_sha(path, sha256, f"{repo}/{filename}")
    return path


@dataclass(frozen=True)
class CodeSnapshot:
    """Third-party inference code at a pinned commit. Every file is fetched
    from ``raw.githubusercontent.com`` and checked against the SHA-256 listed
    here before it is imported; the code is imported under a private package
    name so that generic module names (``models``, ``utils``) never clash."""

    name: str
    repo: str
    commit: str
    files: Mapping[str, str]  # path relative to ``prefix`` -> SHA-256
    prefix: str = ""
    license: str = ""

    @property
    def alias(self) -> str:
        return f"_akoe_ext_{self.name}_{self.commit[:8]}"

    def root(self) -> Path:
        return model_dir() / "code" / f"{self.name}-{self.commit[:12]}"

    def ensure(self, *, local_files_only: bool = False) -> Path:
        base = f"https://raw.githubusercontent.com/{self.repo}/{self.commit}/{self.prefix}"
        rel_root = self.root().relative_to(model_dir())
        for rel, sha in self.files.items():
            fetch_pinned(PinnedFile(base + rel, sha, str(rel_root / rel)),
                         local_files_only=local_files_only)
        return self.root()

    def import_module(self, module: str, *, local_files_only: bool = False) -> types.ModuleType:
        root = self.ensure(local_files_only=local_files_only)
        with _DL_LOCK:
            if self.alias not in sys.modules:
                pkg = types.ModuleType(self.alias)
                pkg.__path__ = [str(root)]  # type: ignore[attr-defined]
                pkg.__package__ = self.alias
                sys.modules[self.alias] = pkg
        return importlib.import_module(f"{self.alias}.{module}")

    def record(self) -> dict[str, Any]:
        return {"repo": self.repo, "commit": self.commit, "prefix": self.prefix,
                "license": self.license, "files_sha256": dict(self.files)}


@dataclass
class _Loaded:
    """Helper for backends: lazily loaded state guarded by a lock."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    provenance: ModelProvenance | None = None
    state: Any = None


def missing_dependency(backend: str, packages: str, error: Exception) -> BackendUnavailable:
    """Italian message explaining how to enable a backend."""
    return BackendUnavailable(
        f"Backend di pulizia neurale '{backend}' non disponibile: manca {packages} ({error}). "
        f"Per abilitarlo: pip install {packages} nell'ambiente di Akoe (nessun aggiornamento di "
        "torch/numpy richiesto).")
