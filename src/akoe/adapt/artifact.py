# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Saving, verifying and loading subject adapters (IMPLEMENTATION_PLAN M4, issue #25).

"Trained where there is compute, usable anywhere": an adapter trained on a GPU
workstation must load on the laptop (CPU). Layout of one adapter::

    <adapter_dir>/
      adapter.json            AdapterArtifact (canonical JSON)
      train_manifest.json     training-set manifest; its SHA-256 is train_manifest_sha256
      training_record.json    wall time, peak memory, losses, library versions
      weights/                PEFT ``save_pretrained`` output (LoRA) or full model
      weights/SHA256SUMS      SHA-256 of every file in weights/ (``sha256sum -c`` format)

``AdapterArtifact.weights`` points to ``weights/SHA256SUMS``: its hash covers the
hash of every weight and config file, so one SHA-256 in the artifact protects
the whole directory. The LoRA directory loads with
``PeftModel.from_pretrained(base, weights_dir)``; call :func:`verify_adapter`
first (or use :func:`load_adapter_model`, which does both).
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from akoe.models import AdapterArtifact, AdapterTechnique, ArtifactRef
from akoe.utils import sha256_file

log = logging.getLogger(__name__)

ADAPTER_JSON = "adapter.json"
WEIGHTS_DIR = "weights"
CHECKSUMS = "SHA256SUMS"
TRAIN_MANIFEST = "train_manifest.json"
TRAINING_RECORD = "training_record.json"
WEIGHTS_KIND = "adapter_weights"
# Keys of AdapterArtifact.hyperparams read back by the loader.
HP_BASE_REVISION = "base_model_revision"
HP_LANGUAGE = "language"
HP_TASK = "task"


class AdapterIntegrityError(Exception):
    """The adapter on disk does not match its artifact. Message in Italian."""


# --------------------------------------------------------------------------
# checksums
# --------------------------------------------------------------------------


def _list_files(directory: Path) -> list[str]:
    return sorted(
        p.relative_to(directory).as_posix()
        for p in directory.rglob("*")
        if p.is_file() and p.name != CHECKSUMS
    )


def write_checksums(weights_dir: Path) -> Path:
    """Write ``SHA256SUMS`` for every file in ``weights_dir`` (sorted, sha256sum format)."""
    lines = [f"{sha256_file(weights_dir / name)}  {name}\n" for name in _list_files(weights_dir)]
    if not lines:
        raise AdapterIntegrityError(f"La cartella dei pesi {weights_dir} è vuota.")
    path = weights_dir / CHECKSUMS
    path.write_text("".join(lines), encoding="utf-8")
    return path


def read_checksums(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        if len(digest) != 64 or not name:
            raise AdapterIntegrityError(f"Riga non valida nel file delle impronte {path}: {line!r}.")
        out[name] = digest
    return out


def weights_ref(weights_dir: Path, root: Path | None = None) -> ArtifactRef:
    """Write the checksums of ``weights_dir`` and return the ArtifactRef to store
    in ``AdapterArtifact.weights``."""
    return ArtifactRef.from_file(write_checksums(weights_dir), kind=WEIGHTS_KIND, root=root)


def weights_dir_of(artifact: AdapterArtifact, root: Path | None = None) -> Path:
    return artifact.weights.resolve(root).parent


def verify_adapter(artifact: AdapterArtifact, root: Path | None = None) -> Path:
    """Check every file of the adapter against the artifact; return the weights dir.

    Refuses (``AdapterIntegrityError``) when the checksum list is missing or
    changed, when a listed file is missing or changed, or when an unlisted file
    appears in the weights directory.
    """
    checksums = artifact.weights.resolve(root)
    if not checksums.is_file():
        raise AdapterIntegrityError(
            f"Adattatore {artifact.adapter_id}: elenco delle impronte non trovato ({checksums}). "
            "Caricamento rifiutato."
        )
    if sha256_file(checksums) != artifact.weights.sha256:
        raise AdapterIntegrityError(
            f"Adattatore {artifact.adapter_id}: l'impronta SHA-256 dei pesi non corrisponde a quella "
            "registrata nell'artefatto. I file potrebbero essere stati modificati o sostituiti: "
            "caricamento rifiutato."
        )
    directory = checksums.parent
    expected = read_checksums(checksums)
    present = set(_list_files(directory))
    missing = sorted(set(expected) - present)
    extra = sorted(present - set(expected))
    if missing:
        raise AdapterIntegrityError(
            f"Adattatore {artifact.adapter_id}: file mancanti nella cartella dei pesi: {', '.join(missing)}. "
            "Caricamento rifiutato."
        )
    if extra:
        raise AdapterIntegrityError(
            f"Adattatore {artifact.adapter_id}: file non previsti nella cartella dei pesi: {', '.join(extra)}. "
            "Caricamento rifiutato."
        )
    changed = [name for name, digest in expected.items() if sha256_file(directory / name) != digest]
    if changed:
        raise AdapterIntegrityError(
            f"Adattatore {artifact.adapter_id}: l'impronta SHA-256 non corrisponde per {', '.join(changed)}. "
            "I pesi potrebbero essere stati modificati: caricamento rifiutato."
        )
    return directory


def verify_train_manifest(artifact: AdapterArtifact, manifest_path: Path) -> None:
    if not manifest_path.is_file() or sha256_file(manifest_path) != artifact.train_manifest_sha256:
        raise AdapterIntegrityError(
            f"Adattatore {artifact.adapter_id}: il manifesto dei dati di addestramento {manifest_path} "
            "non corrisponde all'impronta registrata nell'artefatto."
        )


# --------------------------------------------------------------------------
# artifact file
# --------------------------------------------------------------------------


def save_adapter_artifact(artifact: AdapterArtifact, path: Path, root: Path | None = None) -> ArtifactRef:
    """Write the artifact as canonical JSON; return its ArtifactRef (kind "adapter")."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = artifact.model_dump(mode="json")
    path.write_text(json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return ArtifactRef.from_file(path, kind="adapter", root=root)


def load_adapter_artifact(path: Path, expected_sha256: str | None = None) -> AdapterArtifact:
    if not path.is_file():
        raise AdapterIntegrityError(f"Artefatto dell'adattatore non trovato: {path}.")
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise AdapterIntegrityError(
            f"L'artefatto {path} non corrisponde all'impronta SHA-256 attesa: caricamento rifiutato."
        )
    return AdapterArtifact.model_validate_json(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# base model hash
# --------------------------------------------------------------------------

def _combine(files: dict[str, str]) -> str:
    if len(files) == 1:
        return next(iter(files.values()))
    blob = "".join(f"{digest}  {name}\n" for name, digest in sorted(files.items()))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _hash_local_weights(directory: Path) -> str | None:
    for pattern in (("model.safetensors", "model-*.safetensors"), ("pytorch_model.bin", "pytorch_model-*.bin")):
        found = sorted({p for pat in pattern for p in directory.glob(pat)})
        if found:
            return _combine({p.name: sha256_file(p) for p in found})
    return None


def hash_base_weights(base_model: str, revision: str | None = None) -> str | None:
    """SHA-256 of the base model weights, or None when not computable offline.

    One weight file: the hash of that file (comparable with the Hub's LFS
    sha256). Sharded checkpoints: the SHA-256 of the sorted
    ``"<sha256>  <filename>"`` lines of all shards. Only ``*.safetensors`` are
    considered when present, otherwise ``pytorch_model*.bin``.
    """
    local = Path(base_model)
    if local.is_dir():
        return _hash_local_weights(local)
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:  # pragma: no cover
        return None
    for single, index in (("model.safetensors", "model.safetensors.index.json"),
                          ("pytorch_model.bin", "pytorch_model.bin.index.json")):
        path = try_to_load_from_cache(base_model, single, revision=revision)
        if isinstance(path, str):
            return sha256_file(Path(path))
        idx = try_to_load_from_cache(base_model, index, revision=revision)
        if isinstance(idx, str):
            shards = sorted(set(json.loads(Path(idx).read_text())["weight_map"].values()))
            files: dict[str, str] = {}
            for shard in shards:
                sp = try_to_load_from_cache(base_model, shard, revision=revision)
                if not isinstance(sp, str):
                    return None
                files[shard] = sha256_file(Path(sp))
            return _combine(files)
    return None


def resolve_revision(base_model: str, revision: str | None = None) -> str | None:
    """Commit hash of a cached Hub model (None for local directories / unknown)."""
    if Path(base_model).is_dir():
        return None
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:  # pragma: no cover
        return revision
    path = try_to_load_from_cache(base_model, "config.json", revision=revision)
    if isinstance(path, str):
        parts = Path(path).parts
        if "snapshots" in parts:
            return parts[parts.index("snapshots") + 1]
    return revision


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def resolve_device(device: str | None = None) -> str:
    """CUDA when available, otherwise CPU (EDGE)."""
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class LoadedAdapter:
    model: Any  # PeftModel (LoRA) or WhisperForConditionalGeneration (full / merged)
    processor: Any  # WhisperProcessor
    artifact: AdapterArtifact
    weights_dir: Path
    device: str
    base_model_sha256: str | None


def load_adapter_model(
    artifact: AdapterArtifact,
    root: Path | None = None,
    *,
    device: str | None = None,
    base_model: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    check_base_hash: bool = True,
    strict_base_hash: bool = False,
    merge: bool = False,
    model: Any = None,
    processor: Any = None,
) -> LoadedAdapter:
    """Verify and load an adapter on any tier (CPU when no GPU is present).

    Args:
        base_model: override the base model location (e.g. a local copy on an
            offline laptop); defaults to ``artifact.base_model``.
        revision: base revision; defaults to the one recorded at training time.
        check_base_hash: compare the base weights with ``artifact.base_model_sha256``
            and refuse on mismatch.
        strict_base_hash: also refuse when the base hash cannot be computed.
        merge: merge LoRA weights into the base model (faster inference).
        model, processor: an already-loaded base model/processor (LoRA only).
    """
    weights_dir = verify_adapter(artifact, root)
    device = resolve_device(device)
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    source = base_model or artifact.base_model
    rev = revision or artifact.hyperparams.get(HP_BASE_REVISION)
    base_hash: str | None = None

    if artifact.technique == AdapterTechnique.FULL:
        loaded = WhisperForConditionalGeneration.from_pretrained(str(weights_dir))
        if processor is None:
            try:
                processor = WhisperProcessor.from_pretrained(str(weights_dir))
            except (OSError, ValueError):
                processor = WhisperProcessor.from_pretrained(
                    source, revision=rev, local_files_only=local_files_only
                )
    else:
        from peft import PeftModel

        base = model
        if base is None:
            base = WhisperForConditionalGeneration.from_pretrained(
                source, revision=rev, local_files_only=local_files_only
            )
        if processor is None:
            processor = WhisperProcessor.from_pretrained(source, revision=rev, local_files_only=local_files_only)
        if check_base_hash and artifact.base_model_sha256:
            base_hash = hash_base_weights(source, rev)
            if base_hash is None:
                msg = (
                    f"Adattatore {artifact.adapter_id}: impossibile verificare l'impronta del modello di base "
                    f"{source}."
                )
                if strict_base_hash:
                    raise AdapterIntegrityError(msg + " Caricamento rifiutato.")
                log.warning(msg)
            elif base_hash != artifact.base_model_sha256:
                raise AdapterIntegrityError(
                    f"Adattatore {artifact.adapter_id}: il modello di base {source} non corrisponde a quello "
                    "usato per l'addestramento (impronta SHA-256 diversa). Caricamento rifiutato."
                )
        loaded = PeftModel.from_pretrained(base, str(weights_dir))
        if merge:
            loaded = loaded.merge_and_unload()

    loaded.to(device)
    loaded.eval()
    return LoadedAdapter(
        model=loaded,
        processor=processor,
        artifact=artifact,
        weights_dir=weights_dir,
        device=device,
        base_model_sha256=base_hash,
    )


def load_adapter(adapter_dir: Path, root: Path | None = None, **kwargs: Any) -> LoadedAdapter:
    """Load ``<adapter_dir>/adapter.json`` and the model it describes."""
    artifact = load_adapter_artifact(adapter_dir / ADAPTER_JSON)
    return load_adapter_model(artifact, root, **kwargs)
