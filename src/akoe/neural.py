# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Shared plumbing for neural backends.

- `ModelProvenance`: what a backend actually loaded (model id, revision, SHA-256
  of every weight file, device), for the chain of custody (MASTER_PLAN §5.5).
- Hugging Face snapshot resolution with pinned revisions.
- Device selection and audio preparation (mono float32, 16 kHz).
- `BackendUnavailable`: a `NotImplementedError` so the pipeline logs the stage
  as "skipped" instead of crashing (IMPLEMENTATION_PLAN §6, invariant 5).

Heavy libraries (torch, transformers, huggingface_hub) are imported lazily so
that importing a backend module never requires the `ml` extras.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from akoe.utils import sha256_file

TARGET_SR = 16000
_SHA1_OR_SHA256 = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$")


class BackendUnavailable(NotImplementedError):
    """A backend cannot run in this environment (missing optional dependency,
    unsupported platform...). Subclass of NotImplementedError on purpose: the
    pipeline records it as a skipped stage. The message is for the operator."""


class ModelProvenance(BaseModel):
    """Everything needed to say which model produced an output."""

    backend: str  # backend name, also the origin prefix (e.g. "whisper-tiny/generic")
    model_id: str
    revision: str | None = None  # commit hash when resolved from the HF hub
    weights: dict[str, str] = Field(default_factory=dict)  # file name -> SHA-256
    device: str = "cpu"
    dtype: str | None = None
    license: str | None = None  # as verified on the model card (see docs)
    adapter: dict[str, Any] | None = None  # e.g. LoRA adapter id, dir, file hashes
    extra: dict[str, Any] = Field(default_factory=dict)
    loaded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def weights_digest(self) -> str:
        """Single SHA-256 over the canonical JSON of all weight hashes (base
        model and adapter). Goes into `StageLog.model_sha256`; the per-file
        hashes go into `StageLog.params` via `as_params()`."""
        payload = {"weights": self.weights, "adapter": (self.adapter or {}).get("weights", {})}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def model_version(self) -> str:
        return f"{self.model_id}@{self.revision}" if self.revision else self.model_id

    def stage_log_fields(self) -> dict[str, str]:
        """Fields of `StageLog` filled from this provenance."""
        return {
            "model_version": self.model_version(),
            "model_sha256": self.weights_digest(),
            "device": self.device,
        }

    def as_params(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# --------------------------------------------------------------------------
# Hashing and model resolution
# --------------------------------------------------------------------------

_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def sha256_cached(path: Path) -> str:
    """SHA-256 of a file, cached per (real path, size, mtime) within the process."""
    real = Path(os.path.realpath(path))
    st = real.stat()
    key = (str(real), st.st_size, st.st_mtime_ns)
    if key not in _HASH_CACHE:
        _HASH_CACHE[key] = sha256_file(real)
    return _HASH_CACHE[key]


def hash_files(paths: Iterable[Path], root: Path | None = None) -> dict[str, str]:
    """Map each file (name relative to `root` when given) to its SHA-256."""
    out: dict[str, str] = {}
    for p in sorted(Path(x) for x in paths):
        name = str(p.relative_to(root)) if root is not None else p.name
        out[name] = sha256_cached(p)
    return out


def resolve_hf_snapshot(
    model_id: str,
    revision: str | None = None,
    allow_patterns: list[str] | None = None,
    local_files_only: bool = False,
) -> tuple[Path, str | None]:
    """Return (local directory, commit hash) for a Hugging Face model.

    `model_id` may also be a local directory: then the revision is None and the
    weight hashes are the only identity of the model.
    """
    p = Path(model_id).expanduser()
    if p.is_dir():
        return p, None
    from huggingface_hub import snapshot_download

    local = Path(
        snapshot_download(
            model_id,
            revision=revision,
            allow_patterns=allow_patterns,
            local_files_only=local_files_only,
        )
    )
    commit = local.name if _SHA1_OR_SHA256.match(local.name) else revision
    return local, commit


def download_verified(
    url: str,
    dest: Path,
    expected_sha256: str | None,
    *,
    allow_unknown: bool = False,
    chunk_size: int = 1 << 20,
) -> str:
    """Download `url` to `dest` and return its SHA-256, checked BEFORE the file
    takes its final name (nothing unverified is ever unpickled from `dest`).

    The bytes go to `dest.part` and are hashed while streaming; on a mismatch
    with `expected_sha256` the partial file is deleted and a RuntimeError
    (Italian) is raised. Without an expected hash the download is refused
    unless `allow_unknown=True` (the hash is still returned for the record).
    """
    import urllib.request

    if expected_sha256 is None and not allow_unknown:
        raise RuntimeError(
            f"Il file {dest.name} non ha uno SHA-256 verificato in questo progetto: download rifiutato. "
            "Verificarlo e aggiungerlo all'elenco, oppure consentire esplicitamente pesi non verificati."
        )
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    try:
        with urllib.request.urlopen(url) as r, open(part, "wb") as f:
            while True:
                block = r.read(chunk_size)
                if not block:
                    break
                h.update(block)
                f.write(block)
        sha = h.hexdigest()
        if expected_sha256 is not None and sha != expected_sha256:
            raise RuntimeError(
                f"Il file scaricato da {url} ha SHA-256 {sha}, diverso da quello verificato "
                f"({expected_sha256}): scartato."
            )
        os.replace(part, dest)
        return sha
    finally:
        if part.exists():
            part.unlink()


def resolve_device(device: str = "cpu") -> str:
    """'cpu', 'cuda', 'cuda:N' or 'auto'. Asking for CUDA without CUDA is an
    explicit error (IMPLEMENTATION_PLAN M8), never a silent fallback."""
    if device == "cpu":
        return "cpu"
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"È stato richiesto il dispositivo '{device}', ma CUDA non è disponibile su questa "
            "macchina. Usare device='cpu' oppure installare PyTorch con supporto CUDA."
        )
    return device


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------


def prepare_audio(audio: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Mono float32 at `target_sr` (polyphase resampling, deterministic)."""
    a = np.asarray(audio)
    if a.ndim != 1:
        raise ValueError(
            f"L'audio deve essere mono (vettore 1-D); ricevuto un array con forma {a.shape}."
        )
    if sr <= 0:
        raise ValueError(f"Frequenza di campionamento non valida: {sr}.")
    a = a.astype(np.float32, copy=False)
    if sr != target_sr:
        from scipy.signal import resample_poly

        g = gcd(int(sr), int(target_sr))
        a = resample_poly(a, target_sr // g, int(sr) // g).astype(np.float32)
    return a


def iter_segments(
    audio: np.ndarray,
    sr: int,
    segments: list[tuple[float, float]] | None,
    max_seconds: float | None = None,
) -> list[tuple[float, float, np.ndarray]]:
    """Cut (start, end) regions in seconds out of `audio`, clipped to its
    duration. Regions longer than `max_seconds` are split into consecutive
    chunks of at most `max_seconds` (each chunk becomes its own segment)."""
    dur = len(audio) / sr
    regions = segments if segments is not None else [(0.0, dur)]
    out: list[tuple[float, float, np.ndarray]] = []
    for start, end in regions:
        s, e = max(0.0, float(start)), min(dur, float(end))
        if e <= s:
            raise ValueError(
                f"Segmento non valido ({start:.3f}-{end:.3f} s) per un audio di {dur:.3f} s."
            )
        step = max_seconds if max_seconds else e - s
        t = s
        while t < e - 1e-9:
            u = min(e, t + step)
            out.append((t, u, audio[round(t * sr) : round(u * sr)]))
            t = u
    return out
