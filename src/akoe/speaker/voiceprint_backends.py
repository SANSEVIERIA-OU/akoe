# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Speaker-verification embedding backends (`VoiceprintBackend`).

The embedding is a discriminative representation used to compute a *score*
against the subject profile; it is never an identification (CLAUDE.md,
invariant 6) and never a voice for synthesis (invariant 5).

Backends:
- `ReDimNet2Voiceprint` (default, issue #57): PalabraAI ReDimNet2-B6 (MIT
  repository; weights from the v1.0.0 GitHub release, SHA-256 pinned here and
  checked before unpickling).
- `W2VBertSVVoiceprint`: w2v-BERT 2.0 + layer adapters + multi-layer feature
  aggregation (MFA) + attentive statistics pooling (Li, Cheng & Li 2025,
  `ZXHY-82/w2v-BERT-2.0_SV`); the high-accuracy option, ~590M parameters.
  The weights (`zl389/w2v-bert-2.0_SV` on the HF hub) carry no licence of
  their own: the repository licence applies (owner decision), and it is
  CC-BY-NC-SA-4.0 (`License.md` of the GitHub repository).
- `EcapaVoiceprint`: SpeechBrain ECAPA-TDNN trained on VoxCeleb
  (`speechbrain/spkrec-ecapa-voxceleb`, Apache-2.0 on the model card).
- `ReDimNetVoiceprint`: ID R&D ReDimNet v1 (MIT code; weights from the GitHub
  release, SHA-256 pinned here because the release tag is mutable).

All are lazy: nothing is downloaded or loaded until the first `embed()` or an
explicit `load()`. After loading, `provenance` records model id, revision,
SHA-256 of every weight file and the device. See docs/decisione_impronta_vocale.md
for the model choice; the score normalization (mean subtraction, AS-norm,
QMF) is in `speaker/score_norm.py`.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from akoe.neural import (
    TARGET_SR,
    BackendUnavailable,
    ModelProvenance,
    download_verified,
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

REDIMNET2_REPO = "PalabraAI/redimnet2"
# `main` on 2026-11-20; licence MIT (GitHub API and LICENSE file).
REDIMNET2_COMMIT = "c5bbe0b76e37df698c403f8844e41304ceab6307"
REDIMNET2_RELEASE = "v1.0.0"
# SHA-256 published by GitHub for the v1.0.0 release assets (API field `digest`,
# read on 2026-11-20). The three B6 `lm` files were also downloaded and hashed
# here, with the same result. A file with a different hash is refused before
# it is unpickled.
REDIMNET2_KNOWN_SHA256 = {
    "b0-vox2-lm.pt": "4b5a8c00f813ab0d92fc81ea7c504475d8201bed608fb2c6aaba65be723e9c61",
    "b1-vox2-lm.pt": "9f5b2364524c3ce42f4852b8d9e6c8eb0c14a5a8878dba27f0c93ca500ee37f3",
    "b2-vox2-lm.pt": "0545a29679a87fe1c662d2bbd05e3b3fe0d1b392832729abaa135e4079a2f77a",
    "b3-vox2-lm.pt": "02d14efdc184573873b5c3f62bea847fcdb6919fbba19a8da7af7c1ba36b76a2",
    "b4-vox2-lm.pt": "4f0b1e8bc6bcc10406705996b41a8ed0c71bfcfb7af6fbccd751c2d6a226ef6a",
    "b5-vox2-lm.pt": "65d478edc0ad9c386401a52d80d8a46ee1984625525fdfb2ee82ed069aa67359",
    "b6-vox2-lm.pt": "bc45f032099f3c9f0fb6ad756bca23daeeba456d00c515056e388824ef36f41b",
    "b6-vb2+vox2_v0-lm.pt": "e0a7d340a92f798720d1208949aa6a6bd0cddcb0ba7d4cec33596a17a484e6a2",
    "b6-vb2+vox2+cnc2_v0-lm.pt": "287365f6f485b19e65e5176554f8f7123bfa8d85185f3d2c040eab51acec9868",
    "b3-vb2+vox2+cnc2_v0-lm.pt": "b64989536ef8822447d72517ad9019b870a951e85a3f176f0ed42d2a4c3d811d",
}
# Default variant: B6, large-margin fine-tuned, trained on VoxBlink2 + VoxCeleb2.
# Lowest EER of the release on every reported test set, including the
# out-of-domain SITW (0.74%) and VOICES (2.55%) (README of the repository);
# VoxBlink2 adds many languages to VoxCeleb2. See docs/decisione_impronta_vocale.md.
REDIMNET2_DEFAULT_DATASET = "vb2+vox2_v0"

W2VBERT_SV_REPO = "zl389/w2v-bert-2.0_SV"  # checkpoints of github.com/ZXHY-82/w2v-BERT-2.0_SV
W2VBERT_SV_REVISION = "4e7b92e88f69b6a0aa5a6fdd3dfdb458ed615967"  # 2026-04-27
W2VBERT_SV_CODE = "ZXHY-82/w2v-BERT-2.0_SV@98ff0e29c41d22f8164f4ac5bff7f21cc5d9a95d"
# Weights without a licence of their own: the repository licence applies
# (owner decision), CC-BY-NC-SA-4.0 (non-commercial, share-alike).
W2VBERT_SV_LICENSE = (
    "CC-BY-NC-SA-4.0 (licence of the GitHub repository ZXHY-82/w2v-BERT-2.0_SV, applied to the "
    "unlicensed HF weights by owner decision); base model facebook/w2v-bert-2.0: MIT"
)
# SHA-256 = Git LFS object ids published by the HF hub at W2VBERT_SV_REVISION.
W2VBERT_SV_KNOWN_SHA256 = {
    # VoxCeleb2 + VoxBlink2, LoRA + adapters + MFA, large-margin fine-tuning: Vox1-O EER 0.14% (README)
    "model_lmft_0.14.pth": "454398fe4010bb7d1095517f3aff683c90a1eed644f68f83272c38ea81f953f9",
    # VoxCeleb2 only, adapters + MFA, stage 3 (LMFT): Vox1-O EER 0.18%
    "Adapter_MFA_voxceleb2/s3/adapter_mfa_vox2_s3_0.18.pth":
        "457d222a1ca8a57f3b404a9131a5522f3845596fdebd904b2e341b7c384f951f",
}
W2VBERT_BASE_MODEL = "facebook/w2v-bert-2.0"  # MIT; only config.json and preprocessor_config.json are used
W2VBERT_BASE_REVISION = "da985ba0987f70aaeb84a80f2851cfac8c697a7b"

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


class ReDimNet2Voiceprint:
    """ReDimNet2 speaker embeddings (PalabraAI, Interspeech 2026), loaded through
    torch.hub at a pinned commit of the official repository.

    Default: B6, large-margin fine-tuning (`lm`), trained on VoxBlink2 +
    VoxCeleb2 (`vb2+vox2_v0`). The weight file is downloaded by this class and
    its SHA-256 checked against `REDIMNET2_KNOWN_SHA256` before torch.hub
    unpickles it; an already cached file is checked the same way. Weights not
    listed are accepted only with `allow_unknown_weights=True` (hash recorded).
    """

    def __init__(
        self,
        model_name: str = "b6",
        dataset: str = REDIMNET2_DEFAULT_DATASET,
        train_type: str = "lm",
        commit: str = REDIMNET2_COMMIT,
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
        self.weights_url = (
            f"https://github.com/{REDIMNET2_REPO}/releases/download/{REDIMNET2_RELEASE}/{self.weights_file}"
        )
        self.name = f"redimnet2-{model_name}-{dataset}-{train_type}"
        self._model: Any = None
        self._provenance: ModelProvenance | None = None
        self._lock = threading.Lock()

    @property
    def provenance(self) -> ModelProvenance | None:
        return self._provenance

    def _verify(self, path: Path) -> str:
        sha = hash_files([path])[path.name]
        known = REDIMNET2_KNOWN_SHA256.get(self.weights_file)
        if known is not None and sha != known:
            raise RuntimeError(
                f"I pesi ReDimNet2 '{self.weights_file}' hanno SHA-256 {sha}, diverso da quello "
                f"verificato ({known}). Il file è stato cambiato: non viene usato."
            )
        if known is None and not self.allow_unknown_weights:
            raise RuntimeError(
                f"I pesi ReDimNet2 '{self.weights_file}' non hanno un SHA-256 verificato in questo "
                "progetto. Verificarli e aggiungerli a REDIMNET2_KNOWN_SHA256, oppure usare "
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
                "Il backend ReDimNet2 richiede PyTorch e torchaudio: installare gli extra 'ml'."
            ) from e
        device = resolve_device(self.device)
        ckpt = Path(torch.hub.get_dir()) / "checkpoints" / self.weights_file
        if ckpt.exists():  # verify before unpickling a cached file
            self._verify(ckpt)
        else:  # download and verify here: torch.hub would unpickle before any check
            download_verified(self.weights_url, ckpt, REDIMNET2_KNOWN_SHA256.get(self.weights_file),
                              allow_unknown=self.allow_unknown_weights)
        model = torch.hub.load(
            f"{REDIMNET2_REPO}:{self.commit}", "redimnet2",
            model_name=self.model_name, train_type=self.train_type, dataset=self.dataset,
            pretrained=True, trust_repo=True, skip_validation=True, verbose=False,
        )
        sha = self._verify(ckpt)
        self._model = model.to(device).eval()
        self._provenance = ModelProvenance(
            backend=self.name,
            model_id=f"{REDIMNET2_REPO}/{self.weights_file}",
            revision=self.commit,
            weights={self.weights_file: sha},
            device=device,
            dtype="float32",
            license="MIT (repository PalabraAI/redimnet2; weights are release assets of the same "
                    "repository, covered by its licence per owner decision)",
            extra={
                "embedding_dim": int(getattr(self._model, "embed_dim", 0)) or None,
                "sample_rate": TARGET_SR,
                "release": REDIMNET2_RELEASE,
                "weights_url": self.weights_url,
                "training_data": self.dataset,
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


def _build_w2vbert_sv(config: Any, n_mfa_layers: int, adapter_dim: int, embd_dim: int) -> Any:
    """w2v-BERT 2.0 encoder + one adapter per hidden state + attentive statistics
    pooling + linear bottleneck, with the parameter names of
    `Audio2Vec_based_Adapter` in ZXHY-82/w2v-BERT-2.0_SV (so the published
    checkpoints load as they are). The vendored transformers of that repository
    only adds pruning gates; with `prune=False` (the released models) its
    conformer equals the upstream `Wav2Vec2BertModel` used here."""
    import torch
    from torch import nn
    from transformers import Wav2Vec2BertModel

    class ASP(nn.Module):
        """Attentive statistics pooling (`pooling_v2.ASP` of the reference code)."""

        expansion = 2

        def __init__(self, input_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.attention = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_dim),
                nn.Conv1d(hidden_dim, input_dim, kernel_size=1),
                nn.Softmax(dim=2),
            )

        def forward(self, x: Any) -> Any:  # x: (B, T, D)
            w = self.attention(x.transpose(1, 2)).transpose(1, 2)
            mu = torch.sum(x * w, dim=1)
            sg = torch.sqrt((torch.sum((x ** 2) * w, dim=1) - mu ** 2).clamp(min=1e-5))
            return torch.cat([mu, sg], dim=1)

    class Front(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = Wav2Vec2BertModel(config)

    class W2VBertSV(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.front = Front()
            d = config.hidden_size
            self.n_mfa_layers = n_mfa_layers
            self.adapter_layers = nn.ModuleList([
                nn.Sequential(nn.Linear(d, adapter_dim), nn.LayerNorm(adapter_dim), nn.ReLU(True),
                              nn.Linear(adapter_dim, adapter_dim))
                for _ in range(n_mfa_layers)
            ])
            feat_dim = adapter_dim * n_mfa_layers
            self.pooling = ASP(feat_dim, adapter_dim)
            self.bottleneck = nn.Linear(feat_dim * ASP.expansion, embd_dim)

        def forward(self, input_features: Any) -> Any:
            enc = self.front.encoder
            x = enc.feature_projection(input_features)[0]
            hidden = [x]
            for layer in enc.encoder.layers:
                out = layer(x)
                x = out[0] if isinstance(out, tuple) else out
                hidden.append(x)
            hidden = hidden[-self.n_mfa_layers:]
            x = torch.cat([ad(h) for ad, h in zip(self.adapter_layers, hidden)], dim=-1)
            return self.bottleneck(self.pooling(x))

    return W2VBertSV()


class W2VBertSVVoiceprint:
    """w2v-BERT 2.0 speaker-verification embeddings (256-D), the high-accuracy option.

    Default checkpoint `model_lmft_0.14.pth` (VoxCeleb2 + VoxBlink2, Vox1-O EER
    0.14% as stated in the repository README). The checkpoint (~2.35 GB, fp32)
    is memory-mapped and loaded with `weights_only=True` after its SHA-256 is
    checked, so the weights are not copied into process memory twice. Input
    longer than `max_seconds` is truncated, as in the reference embedding
    script (`get_embd_w2v.py`, 60 s). Inference is float32 on CPU (the
    reference runs bfloat16 autocast on GPU); both are recorded in `provenance`.
    """

    def __init__(
        self,
        checkpoint: str = "model_lmft_0.14.pth",
        *,
        repo_id: str = W2VBERT_SV_REPO,
        revision: str = W2VBERT_SV_REVISION,
        device: str = "cpu",
        local_files_only: bool = False,
        allow_unknown_weights: bool = False,
        min_seconds: float = DEFAULT_MIN_SECONDS,
        max_seconds: float = 60.0,
        n_mfa_layers: int = 25,
        adapter_dim: int = 128,
        embd_dim: int = 256,
    ) -> None:
        self.checkpoint = checkpoint
        self.repo_id = repo_id
        self.revision = revision
        self.device = device
        self.local_files_only = local_files_only
        self.allow_unknown_weights = allow_unknown_weights
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        self.n_mfa_layers = n_mfa_layers
        self.adapter_dim = adapter_dim
        self.embd_dim = embd_dim
        stem = Path(checkpoint).stem
        self.name = f"w2v-bert-2.0-sv/{stem}"
        self._model: Any = None
        self._fe: Any = None
        self._provenance: ModelProvenance | None = None
        self._lock = threading.Lock()

    @property
    def provenance(self) -> ModelProvenance | None:
        return self._provenance

    def _verify(self, path: Path) -> str:
        sha = hash_files([path])[path.name]
        known = W2VBERT_SV_KNOWN_SHA256.get(self.checkpoint) if self.repo_id == W2VBERT_SV_REPO else None
        if known is not None and sha != known:
            raise RuntimeError(
                f"I pesi w2v-BERT SV '{self.checkpoint}' hanno SHA-256 {sha}, diverso da quello "
                f"verificato ({known}). Il file è stato cambiato: non viene usato."
            )
        if known is None and not self.allow_unknown_weights:
            raise RuntimeError(
                f"I pesi w2v-BERT SV '{self.checkpoint}' non hanno un SHA-256 verificato in questo "
                "progetto. Verificarli e aggiungerli a W2VBERT_SV_KNOWN_SHA256, oppure usare "
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
            from huggingface_hub import hf_hub_download
            from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertConfig
        except ImportError as e:  # pragma: no cover
            raise BackendUnavailable(
                "Il backend w2v-BERT 2.0 SV richiede PyTorch e transformers: installare gli extra 'ml'."
            ) from e
        device = resolve_device(self.device)
        base, base_commit = resolve_hf_snapshot(
            W2VBERT_BASE_MODEL, W2VBERT_BASE_REVISION, ["config.json", "preprocessor_config.json"],
            self.local_files_only,
        )
        cfg_dict = json.loads((base / "config.json").read_text(encoding="utf-8"))
        cfg_dict.update(mask_time_prob=0.0, mask_feature_prob=0.0)  # no masked_spec_embed (deleted in training)
        config = Wav2Vec2BertConfig(**cfg_dict)
        if self.n_mfa_layers > config.num_hidden_layers + 1:
            raise ValueError("n_mfa_layers exceeds the number of hidden states")
        ckpt = Path(hf_hub_download(self.repo_id, self.checkpoint, revision=self.revision,
                                    local_files_only=self.local_files_only))
        sha = self._verify(ckpt)  # before unpickling
        data = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=True)
        state = data["modules"]["spk_model"] if "modules" in data else data
        with torch.device("meta"):
            model = _build_w2vbert_sv(config, self.n_mfa_layers, self.adapter_dim, self.embd_dim)
        res = model.load_state_dict(state, strict=False, assign=True)
        if res.missing_keys:
            raise RuntimeError(f"Checkpoint w2v-BERT SV incompleto: mancano {res.missing_keys[:5]}...")
        left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
        if left:
            raise RuntimeError(f"Parametri non inizializzati dopo il caricamento: {left[:5]}")
        self._model = model.to(device).eval()
        self._fe = SeamlessM4TFeatureExtractor.from_pretrained(str(base))
        self._provenance = ModelProvenance(
            backend=self.name,
            model_id=f"{self.repo_id}/{self.checkpoint}",
            revision=self.revision,
            weights={self.checkpoint: sha},
            device=device,
            dtype="float32",
            license=W2VBERT_SV_LICENSE,
            extra={
                "embedding_dim": self.embd_dim, "sample_rate": TARGET_SR,
                "architecture": "w2v-BERT 2.0 + adapters + MFA + ASP (Audio2Vec_based_Adapter)",
                "n_mfa_layers": self.n_mfa_layers, "adapter_dim": self.adapter_dim,
                "max_seconds": self.max_seconds, "code_reference": W2VBERT_SV_CODE,
                "base_config": f"{W2VBERT_BASE_MODEL}@{base_commit}",
                "unexpected_keys": len(res.unexpected_keys),
                "reference_precision": "bfloat16 autocast on GPU (get_embd_w2v.py); here float32",
            },
        )

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import torch

        a = prepare_audio(audio, sr)
        _check_length(a, self.min_seconds)
        if self._model is None:
            self.load()
        a = a[: int(self.max_seconds * TARGET_SR)]
        feats = self._fe(a, sampling_rate=TARGET_SR, return_tensors="pt", padding=False,
                         truncation=False, return_attention_mask=False).input_features
        with torch.inference_mode():
            emb = self._model(feats.to(self._provenance.device, torch.float32))
        return emb.reshape(-1).cpu().numpy().astype(np.float32)


DEFAULT_VOICEPRINT = "redimnet2-b6"
VOICEPRINT_CHOICES = "redimnet2-b6 (predefinito), redimnet2-b6-vox2, w2v-bert-sv, redimnet-b2, redimnet-b6, ecapa"


def default_voiceprint_backend(name: str | None = None, *, device: str = "cpu") -> Any:
    """The project's voiceprint backend (docs/decisione_impronta_vocale.md):
    ReDimNet2-B6 by default; w2v-BERT 2.0 SV as the high-accuracy option
    (``name="w2v-bert-sv"``); ReDimNet v1 and ECAPA-TDNN for comparison and
    fallback. Nothing is loaded until the first `embed()`."""
    key = (name or DEFAULT_VOICEPRINT).lower()
    if key in ("redimnet2", "redimnet2-b6"):
        return ReDimNet2Voiceprint(device=device)
    if key in ("redimnet2-b6-vox2",):
        return ReDimNet2Voiceprint(dataset="vox2", device=device)
    if key in ("w2v-bert-sv", "w2v-bert", "w2vbert", "w2v-bert-2.0-sv"):
        return W2VBertSVVoiceprint(device=device)
    if key in ("redimnet", "redimnet-b2", "b2"):
        return ReDimNetVoiceprint(device=device)
    if key in ("redimnet-b6", "b6"):
        return ReDimNetVoiceprint(model_name="b6", device=device)
    if key in ("ecapa", "ecapa-tdnn"):
        return EcapaVoiceprint(device=device)
    raise ValueError(f"Modello di impronta vocale sconosciuto: {name!r} ({VOICEPRINT_CHOICES})")
