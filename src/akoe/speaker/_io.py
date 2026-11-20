# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Small I/O helpers shared by the speaker modules (internal).

- Deterministic JSON: sorted keys, fixed indentation, UTF-8, so that the same
  content always has the same SHA-256.
- Content-addressed artifacts: the file name embeds the hash prefix, an
  existing file is never overwritten with different content.
- Audio loading: mono float32, resampled with a fixed, recorded method.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from akoe.case import Case
from akoe.models import ArtifactRef

RESAMPLE_METHOD = "soxr_hq"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def check_id(value: str, what: str = "identificativo") -> str:
    if not _SAFE_ID.match(value):
        raise ValueError(
            f"{what} non valido: '{value}' (ammessi lettere, cifre, '_', '.', '-')"
        )
    return value


def subject_dir(case: Case, subject_id: str) -> Path:
    path = case.dir("profiles") / f"subject-{check_id(subject_id, 'identificativo del soggetto')}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_json_bytes(data: Any) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(dump_json_bytes(data))
    os.replace(tmp, path)


def write_content_addressed(
    case: Case, directory: Path, prefix: str, data: Any, kind: str
) -> ArtifactRef:
    """Write `data` as JSON to `<directory>/<prefix>-<sha16>.json` and return its ArtifactRef
    (path relative to the case root). Identical content maps to the same file."""
    payload = dump_json_bytes(data)
    digest = hashlib.sha256(payload).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prefix}-{digest[:16]}.json"
    if not path.exists():
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, path)
    ref = ArtifactRef.from_file(path, kind=kind, root=case.root)
    if ref.sha256 != digest:
        raise RuntimeError(f"hash mismatch writing {path}")
    return ref


def read_verified_json(case: Case, ref: ArtifactRef) -> Any:
    """Read a JSON artifact after checking its SHA-256 against the reference."""
    if not ref.verify(case.root):
        raise ValueError(
            f"L'artefatto {ref.path} manca o è stato modificato (SHA-256 non corrispondente)"
        )
    return json.loads(ref.resolve(case.root).read_text(encoding="utf-8"))


def load_audio(path: Path, sr: int) -> np.ndarray:
    """Mono float32 at `sr`, resampled with RESAMPLE_METHOD."""
    try:
        data, native_sr = sf.read(str(path), dtype="float32", always_2d=True)
        audio = data.mean(axis=1)
    except Exception:  # formats libsndfile cannot read: fall back to librosa/audioread
        audio, native_sr = librosa.load(str(path), sr=None, mono=True)
    if native_sr != sr:
        audio = librosa.resample(audio, orig_sr=native_sr, target_sr=sr, res_type=RESAMPLE_METHOD)
    return np.ascontiguousarray(audio, dtype=np.float32)


def cut(audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    a = max(0, int(round(start * sr)))
    b = min(len(audio), int(round(end * sr)))
    return audio[a:b] if b > a else audio[:0]
