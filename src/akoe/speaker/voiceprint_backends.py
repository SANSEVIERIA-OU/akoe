# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Speaker-verification embedding backends (`VoiceprintBackend`).

The embedding is a discriminative representation used to compute a *score*
against the subject profile; it is never an identification (CLAUDE.md,
invariant 6) and never a voice for synthesis (invariant 5).

Backends:
- `EcapaVoiceprint`: SpeechBrain ECAPA-TDNN trained on VoxCeleb
  (`speechbrain/spkrec-ecapa-voxceleb`, Apache-2.0 on the model card).
- `ReDimNetVoiceprint`: ID R&D ReDimNet (MIT code; weights from the GitHub
  release, SHA-256 pinned here because the release tag is mutable).

Both are lazy: nothing is downloaded or loaded until the first `embed()` or an
explicit `load()`. After loading, `provenance` records model id, revision,
SHA-256 of every weight file and the device. See docs/decisione_impronta_vocale.md
for the model choice.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from akoe.neural import (
    TARGET_SR,
    BackendUnavailable,
    ModelProvenance,
    hash_files,
    prepare_audio,
    resolve_device,
    resolve_hf_snapshot,
)

# Revisions verified on 2026-11-20 (licence read on the model card at this commit).
ECAPA_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
ECAPA_FILES = [
    "hyperparams.yaml",
    "embedding_model.ckpt",
    "mean_var_norm_emb.ckpt",
    "classifier.ckpt",
    "label_encoder.txt",
]

REDIMNET_REPO = "IDRnD/ReDimNet"
REDIMNET_COMMIT = "ce039a624cb99fe127702ceb94c6080090e5032f"
# SHA-256 of the release assets downloaded on 2026-11-20. The release tag is
# "latest" and GitHub publishes no digest, so a different file is refused.
REDIMNET_KNOWN_SHA256 = {
    "b2-vox2-ft_lm.pt": "c9b6bb2f6747caa28a41eaf2e372d66b0d1563baef186d18f5e99abd5e71e06f",
    "b6-vox2-ft_lm.pt": "e0f3034320c9cceaa8ed7c917a36e0affb501a922631d55c1007660eb1bbedea",
}

DEFAULT_MIN_SECONDS = 0.5


def _check_length(audio16k: np.ndarray, min_seconds: float) -> None:
    dur = len(audio16k) / TARGET_SR
    if dur < min_seconds:
        raise ValueError(
            f"Segmento troppo corto per l'impronta vocale ({dur:.2f} s; minimo {min_seconds:.2f} s)."
        )


class EcapaVoiceprint:
    """ECAPA-TDNN (SpeechBrain) speaker embeddings, 192-D, for cosine scoring.

    `embed()` returns the raw embedding (not L2-normalized): cosine scoring is
    invariant to the norm, and keeping it raw loses nothing.
    """

    def __init__(
        self,
        model_id: str = ECAPA_MODEL_ID,
        revision: str | None = None,
        device: str = "cpu",
        savedir: str | Path | None = None,
        local_files_only: bool = False,
        min_seconds: float = DEFAULT_MIN_SECONDS,
    ) -> None:
        self.model_id = model_id
        self.revision = revision if revision is not None else (
            ECAPA_REVISION if model_id == ECAPA_MODEL_ID else None
        )
        self.device = device
        self.savedir = Path(savedir) if savedir else None
        self.local_files_only = local_files_only
        self.min_seconds = min_seconds
        self.name = model_id.rstrip("/").split("/")[-1]
        self._model: Any = None
        self._provenance: ModelProvenance | None = None
        self._lock = threading.Lock()

    @property
    def provenance(self) -> ModelProvenance | None:
        """None until the model is loaded."""
        return self._provenance

    def load(self) -> ModelProvenance:
        with self._lock:
            if self._model is None:
                self._load()
        assert self._provenance is not None
        return self._provenance

    def _load(self) -> None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy
        except ImportError as e:  # pragma: no cover - depends on environment
            raise BackendUnavailable(
                "Il backend ECAPA-TDNN richiede SpeechBrain e PyTorch: installare gli extra "
                "'ml' (pip install -e '.[ml]')."
            ) from e
        device = resolve_device(self.device)
        local, commit = resolve_hf_snapshot(
            self.model_id, self.revision, allow_patterns=ECAPA_FILES + ["*.json"],
            local_files_only=self.local_files_only,
        )
        savedir = self.savedir or (
            Path.home() / ".cache" / "akoe" / "speechbrain"
            / f"{self.model_id.replace('/', '--')}-{commit or 'local'}"
        )
        savedir.mkdir(parents=True, exist_ok=True)
        # `pretrained_path` in hyperparams.yaml points at the hub "main" branch:
        # override it so that the pinned local snapshot is what gets loaded.
        self._model = EncoderClassifier.from_hparams(
            source=str(local),
            savedir=str(savedir),
            overrides={"pretrained_path": str(local)},
            run_opts={"device": device},
            local_strategy=LocalStrategy.NO_LINK,
        )
        self._model.eval()
        files = [local / f for f in ECAPA_FILES if (local / f).exists()]
        self._provenance = ModelProvenance(
            backend=self.name,
            model_id=self.model_id,
            revision=commit,
            weights=hash_files(files, root=local),
            device=device,
            dtype="float32",
            license="apache-2.0" if self.model_id == ECAPA_MODEL_ID else None,
            extra={"embedding_dim": 192, "sample_rate": TARGET_SR},
        )

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import torch

        a = prepare_audio(audio, sr)
        _check_length(a, self.min_seconds)
        if self._model is None:
            self.load()
        wav = torch.from_numpy(a)[None]
        with torch.inference_mode():
            emb = self._model.encode_batch(wav.to(self._provenance.device), torch.ones(1))
        return emb.reshape(-1).cpu().numpy().astype(np.float32)


class ReDimNetVoiceprint:
    """ReDimNet speaker embeddings (192-D), loaded through torch.hub at a pinned
    commit of the official repository. Default: B2 trained on VoxCeleb2 with
    large-margin fine-tuning (`b2`, `vox2`, `ft_lm`).

    Weights not listed in `REDIMNET_KNOWN_SHA256` are accepted only with
    `allow_unknown_weights=True`, and their hash is still recorded.
    """

    def __init__(
        self,
        model_name: str = "b2",
        dataset: str = "vox2",
        train_type: str = "ft_lm",
        commit: str = REDIMNET_COMMIT,
        device: str = "cpu",
        allow_unknown_weights: bool = False,
        min_seconds: float = DEFAULT_MIN_SECONDS,
    ) -> None:
        self.model_name = model_name
        self.dataset = dataset
        self.train_type = train_type
        self.commit = commit
        self.device = device
        self.allow_unknown_weights = allow_unknown_weights
        self.min_seconds = min_seconds
        self.weights_file = f"{model_name}-{dataset}-{train_type}.pt"
        self.name = f"redimnet-{model_name}-{dataset}-{train_type}"
        self._model: Any = None
        self._provenance: ModelProvenance | None = None
        self._lock = threading.Lock()

    @property
    def provenance(self) -> ModelProvenance | None:
        return self._provenance

    def _verify(self, path: Path) -> str:
        sha = hash_files([path])[path.name]
        known = REDIMNET_KNOWN_SHA256.get(self.weights_file)
        if known is not None and sha != known:
            raise RuntimeError(
                f"I pesi ReDimNet '{self.weights_file}' hanno SHA-256 {sha}, diverso da quello "
                f"verificato ({known}). Il file è stato cambiato: non viene usato."
            )
        if known is None and not self.allow_unknown_weights:
            raise RuntimeError(
                f"I pesi ReDimNet '{self.weights_file}' non hanno un SHA-256 verificato in questo "
                "progetto. Verificarli e aggiungerli a REDIMNET_KNOWN_SHA256, oppure usare "
                "allow_unknown_weights=True (l'hash viene comunque registrato)."
            )
        return sha

    def load(self) -> ModelProvenance:
        with self._lock:
            if self._model is None:
                self._load()
        assert self._provenance is not None
        return self._provenance

    def _load(self) -> None:
        try:
            import torch
        except ImportError as e:  # pragma: no cover
            raise BackendUnavailable(
                "Il backend ReDimNet richiede PyTorch: installare gli extra 'ml'."
            ) from e
        device = resolve_device(self.device)
        ckpt = Path(torch.hub.get_dir()) / "checkpoints" / self.weights_file
        if ckpt.exists():  # verify before unpickling a cached file
            self._verify(ckpt)
        model = torch.hub.load(
            f"{REDIMNET_REPO}:{self.commit}", "ReDimNet",
            model_name=self.model_name, train_type=self.train_type, dataset=self.dataset,
            trust_repo=True, skip_validation=True, verbose=False,
        )
        sha = self._verify(ckpt)
        self._model = model.to(device).eval()
        self._provenance = ModelProvenance(
            backend=self.name,
            model_id=f"{REDIMNET_REPO}/{self.weights_file}",
            revision=self.commit,
            weights={self.weights_file: sha},
            device=device,
            dtype="float32",
            license="MIT (code, verified); weights: release asset, licence not stated separately",
            extra={
                "embedding_dim": 192,
                "sample_rate": TARGET_SR,
                "weights_url": f"https://github.com/{REDIMNET_REPO}/releases/download/latest/"
                f"{self.weights_file}",
            },
        )

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import torch

        a = prepare_audio(audio, sr)
        _check_length(a, self.min_seconds)
        if self._model is None:
            self.load()
        with torch.inference_mode():
            emb = self._model(torch.from_numpy(a)[None].to(self._provenance.device))
        return emb.reshape(-1).cpu().numpy().astype(np.float32)


DEFAULT_VOICEPRINT = "redimnet-b2"


def default_voiceprint_backend(name: str | None = None, *, device: str = "cpu") -> Any:
    """The project's voiceprint backend (docs/decisione_impronta_vocale.md):
    ReDimNet-B2 by default, ECAPA-TDNN as the independent reference
    (``name="ecapa"``). Nothing is loaded until the first `embed()`."""
    key = (name or DEFAULT_VOICEPRINT).lower()
    if key in ("redimnet", "redimnet-b2", "b2"):
        return ReDimNetVoiceprint(device=device)
    if key in ("redimnet-b6", "b6"):
        return ReDimNetVoiceprint(model_name="b6", device=device)
    if key in ("ecapa", "ecapa-tdnn"):
        return EcapaVoiceprint(device=device)
    raise ValueError(f"Modello di impronta vocale sconosciuto: {name!r} (redimnet-b2, redimnet-b6, ecapa)")
