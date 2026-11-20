# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Fine-tuning Hugging Face Whisper on the subject (IMPLEMENTATION_PLAN M4, issue #24).

- LoRA (PEFT) is the default, with a non-minimal rank: on a single speaker,
  rank 8 was 15-39% relative worse than full fine-tuning (Huber & Waibel 2026a,
  MASTER_PLAN §3.4.3). Default r=32, alpha=64 on all attention and MLP
  projections of encoder and decoder.
- Full fine-tuning is an option (workstation / cloud), chosen when the
  self-assessment shows a gain.
- Seeded; peak memory and wall time are measured and recorded.
- Output: PEFT ``save_pretrained`` directory (loads with
  ``PeftModel.from_pretrained``) plus an ``AdapterArtifact`` with base model,
  revision, base weights hash, technique, hyperparameters, seed, training
  manifest hash, conditions covered and the SHA-256 of the saved weights
  (layout in ``akoe.adapt.artifact``).

Training on CPU (EDGE) is realistic only for small models (tiny/small) and is slow.
"""
from __future__ import annotations

import logging
import math
import os
import platform
import random
import resource
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from akoe.adapt.artifact import (
    ADAPTER_JSON,
    HP_BASE_REVISION,
    HP_LANGUAGE,
    HP_TASK,
    TRAIN_MANIFEST,
    TRAINING_RECORD,
    WEIGHTS_DIR,
    hash_base_weights,
    resolve_device,
    resolve_revision,
    save_adapter_artifact,
    weights_ref,
)
from akoe.adapt.dataset import TrainingSet
from akoe.config import Tier
from akoe.models import AdapterArtifact, AdapterTechnique, ArtifactRef

log = logging.getLogger(__name__)

# All linear projections of Whisper attention (self and cross) and MLP blocks.
DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
LOW_RANK_WARNING = 16


def base_model_for_tier(tier: Tier) -> str:
    """Suggested base model: small on a laptop CPU, large-v3 with a GPU."""
    return "openai/whisper-small" if tier == Tier.EDGE else "openai/whisper-large-v3"


class TrainError(Exception):
    """Raised with an operator-facing (Italian) message."""


class TrainConfig(BaseModel):
    base_model: str = "openai/whisper-small"
    base_model_revision: str | None = None
    technique: AdapterTechnique = AdapterTechnique.LORA
    # LoRA
    lora_r: int = Field(default=32, ge=1)
    lora_alpha: int = Field(default=64, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    lora_target_modules: list[str] = Field(default_factory=lambda: list(DEFAULT_LORA_TARGETS))
    # Full fine-tuning
    freeze_encoder: bool = False
    # Decoding target
    language: str = "italian"
    task: str = "transcribe"
    # Optimisation (learning_rate None -> 5e-4 for LoRA, 1e-5 for full)
    learning_rate: float | None = None
    num_epochs: float = Field(default=3.0, gt=0)
    max_steps: int | None = Field(default=None, ge=1)
    batch_size: int = Field(default=4, ge=1)
    grad_accum_steps: int = Field(default=1, ge=1)
    warmup_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    seed: int = 1234
    gradient_checkpointing: bool = False
    # Execution
    device: str | None = None  # None: cuda if available, else cpu
    # torch CPU threads, set process-wide (None: leave torch's setting). Useful
    # on a shared machine.
    num_threads: int | None = None

    def resolved_learning_rate(self) -> float:
        if self.learning_rate is not None:
            return self.learning_rate
        return 5e-4 if self.technique == AdapterTechnique.LORA else 1e-5


class SkippedExample(BaseModel):
    example_id: str
    reason: str


class TrainingRecord(BaseModel):
    adapter_id: str
    base_model: str
    base_model_revision: str | None = None
    technique: AdapterTechnique
    started_at: datetime
    finished_at: datetime
    wall_time_s: float
    rss_before_mb: float
    peak_rss_mb: float  # sampled during loading + training + saving
    cuda_peak_mb: float | None = None
    device: str
    tier: Tier | None = None
    num_threads: int
    steps: int
    epochs_completed: float
    n_examples: int
    skipped: list[SkippedExample] = Field(default_factory=list)
    losses: list[float] = Field(default_factory=list)
    trainable_params: int
    total_params: int
    versions: dict[str, str] = Field(default_factory=dict)


class TrainResult(BaseModel):
    artifact: AdapterArtifact
    artifact_ref: ArtifactRef
    manifest_ref: ArtifactRef
    record_ref: ArtifactRef
    record: TrainingRecord
    adapter_dir: str


# --------------------------------------------------------------------------
# memory measurement
# --------------------------------------------------------------------------


def _rss_bytes() -> int:
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):  # non-Linux: lifetime peak instead
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss if platform.system() == "Darwin" else rss * 1024


class PeakRSS:
    """Samples the resident set size in a background thread; ``peak_mb`` is the
    maximum observed while the context was open."""

    def __init__(self, interval_s: float = 0.02) -> None:
        self.interval_s = interval_s
        self.before_mb = 0.0
        self._peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._peak = max(self._peak, _rss_bytes())
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "PeakRSS":
        self._peak = _rss_bytes()
        self.before_mb = self._peak / 2**20
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self._peak = max(self._peak, _rss_bytes())

    @property
    def peak_mb(self) -> float:
        return self._peak / 2**20


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for name in ("torch", "transformers", "peft", "numpy"):
        try:
            out[name] = __import__(name).__version__
        except ImportError:  # pragma: no cover
            pass
    return out


def _collate(examples: list[tuple[np.ndarray, str]], processor: Any, sr: int, decoder_start_id: int):
    import torch

    feats = processor.feature_extractor(
        [a for a, _ in examples], sampling_rate=sr, return_tensors="pt"
    ).input_features
    enc = processor.tokenizer([t for _, t in examples], padding="longest", return_tensors="pt")
    labels = enc.input_ids.masked_fill(enc.attention_mask.ne(1), -100)
    # The model prepends decoder_start_token_id itself when shifting labels.
    if bool((labels[:, 0] == decoder_start_id).all()):
        labels = labels[:, 1:]
    return feats, labels.to(torch.long)


def _check_examples(
    training_set: TrainingSet, processor: Any, max_seconds: float, max_label_tokens: int
) -> tuple[list[int], list[SkippedExample]]:
    keep: list[int] = []
    skipped: list[SkippedExample] = []
    for i, rec in enumerate(training_set.manifest.examples):
        if rec.n_samples / rec.sample_rate > max_seconds:
            skipped.append(SkippedExample(example_id=rec.example_id, reason="audio_longer_than_window"))
            continue
        if len(processor.tokenizer(rec.text).input_ids) > max_label_tokens:
            skipped.append(SkippedExample(example_id=rec.example_id, reason="label_too_long"))
            continue
        keep.append(i)
    return keep, skipped


def _apply_technique(model: Any, config: TrainConfig) -> Any:
    if config.technique == AdapterTechnique.LORA:
        from peft import LoraConfig, get_peft_model

        lora = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_target_modules),
            bias="none",
        )
        return get_peft_model(model, lora)
    if config.freeze_encoder:
        model.model.encoder.requires_grad_(False)
    return model


def train_adapter(
    training_set: TrainingSet,
    out_dir: Path,
    config: TrainConfig | None = None,
    *,
    adapter_id: str | None = None,
    root: Path | None = None,
    tier: Tier | None = None,
    model: Any = None,
    processor: Any = None,
) -> TrainResult:
    """Fine-tune Whisper on ``training_set`` and save a versioned adapter in ``out_dir``.

    Args:
        training_set: output of ``akoe.adapt.dataset.build_training_set``.
        out_dir: adapter directory (must not already contain an adapter).
        config: hyperparameters; LoRA r=32 by default.
        adapter_id: identifier; default ``<technique>-<manifest sha[:12]>-s<seed>``.
        root: case root; artifact paths are stored relative to it when given.
        tier: recorded in the training record.
        model, processor: preloaded ``WhisperForConditionalGeneration`` and
            ``WhisperProcessor`` (tests, offline use); otherwise loaded from
            ``config.base_model``.
    """
    import torch
    from transformers import get_linear_schedule_with_warmup

    config = config or TrainConfig()
    out_dir = Path(out_dir).resolve()
    root = Path(root).resolve() if root is not None else None
    if config.technique == AdapterTechnique.LORA and config.lora_r < LOW_RANK_WARNING:
        log.warning(
            "Rango LoRA %d basso: su un singolo parlante il rango 8 è risultato nettamente peggiore del "
            "fine-tuning completo (Huber & Waibel 2026a). Si consiglia un rango di almeno %d.",
            config.lora_r,
            LOW_RANK_WARNING,
        )
    weights_dir = out_dir / WEIGHTS_DIR
    if weights_dir.exists() and any(weights_dir.iterdir()):
        raise TrainError(
            f"La cartella {out_dir} contiene già un adattatore: scegliere una cartella nuova "
            "(gli adattatori esistenti non vengono sovrascritti)."
        )
    manifest_sha = training_set.sha256()
    adapter_id = adapter_id or f"{config.technique.value}-{manifest_sha[:12]}-s{config.seed}"

    if config.num_threads:
        torch.set_num_threads(config.num_threads)
    device = resolve_device(config.device)
    if device == "cpu":
        log.warning(
            "Addestramento su CPU: realistico solo per modelli piccoli (tiny/small), con tempi lunghi."
        )
    cuda = device.startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    with PeakRSS() as mem:
        _seed_everything(config.seed)
        if model is None:
            from transformers import WhisperForConditionalGeneration

            model = WhisperForConditionalGeneration.from_pretrained(
                config.base_model, revision=config.base_model_revision
            )
        if processor is None:
            from transformers import WhisperProcessor

            processor = WhisperProcessor.from_pretrained(config.base_model, revision=config.base_model_revision)
        tokenizer = processor.tokenizer
        tokenizer.set_prefix_tokens(language=config.language, task=config.task, predict_timestamps=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        decoder_start_id = model.config.decoder_start_token_id
        max_seconds = float(getattr(processor.feature_extractor, "chunk_length", 30))
        keep, skipped = _check_examples(
            training_set, processor, max_seconds, int(model.config.max_target_positions)
        )
        if not keep:
            raise TrainError("Nessun esempio utilizzabile per l'addestramento (tutti troppo lunghi).")

        _seed_everything(config.seed)  # LoRA initialisation is seeded too
        lora = config.technique == AdapterTechnique.LORA
        model = _apply_technique(model, config)
        model.to(device)
        if config.gradient_checkpointing:
            core = model.get_base_model() if lora else model
            core.config.use_cache = False
            core.gradient_checkpointing_enable()
            if lora:
                # the encoder input (log-mel features) carries no grad; LoRA needs one
                core.model.encoder.conv1.register_forward_hook(lambda _m, _i, out: out.requires_grad_(True))
        model.train()

        params = [p for p in model.parameters() if p.requires_grad]
        trainable = sum(p.numel() for p in params)
        total_params = sum(p.numel() for p in model.parameters())
        lr = config.resolved_learning_rate()
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=config.weight_decay)
        n = len(keep)
        batches_per_epoch = math.ceil(n / config.batch_size)
        steps_per_epoch = max(1, math.ceil(batches_per_epoch / config.grad_accum_steps))
        total_steps = config.max_steps or max(1, math.ceil(steps_per_epoch * config.num_epochs))
        warmup = int(round(config.warmup_ratio * total_steps))
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
        autocast_dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else None

        gen = torch.Generator().manual_seed(config.seed)
        losses: list[float] = []
        step = 0
        micro = 0
        running = 0.0
        batches_done = 0
        optimizer.zero_grad(set_to_none=True)
        sr = training_set.sample_rate
        while step < total_steps:
            order = [keep[i] for i in torch.randperm(n, generator=gen).tolist()]
            for b in range(0, n, config.batch_size):
                batch = [training_set.example(i) for i in order[b : b + config.batch_size]]
                feats, labels = _collate(batch, processor, sr, decoder_start_id)
                feats, labels = feats.to(device), labels.to(device)
                if autocast_dtype is not None:
                    with torch.autocast("cuda", dtype=autocast_dtype):
                        loss = model(input_features=feats, labels=labels).loss
                else:
                    loss = model(input_features=feats, labels=labels).loss
                (loss / config.grad_accum_steps).backward()
                running += float(loss.detach()) / config.grad_accum_steps
                micro += 1
                batches_done += 1
                if micro == config.grad_accum_steps:
                    torch.nn.utils.clip_grad_norm_(params, config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    losses.append(round(running, 6))
                    step += 1
                    micro = 0
                    running = 0.0
                    log.info("passo %d/%d, loss %.4f", step, total_steps, losses[-1])
                    if step >= total_steps:
                        break
        model.eval()

        weights_dir.mkdir(parents=True, exist_ok=True)
        # LoRA: PEFT adapter files only. Full: standalone model + processor.
        model.save_pretrained(str(weights_dir))
        if not lora:
            processor.save_pretrained(str(weights_dir))
    wall = time.perf_counter() - t0
    finished_at = datetime.now(timezone.utc)

    revision = resolve_revision(config.base_model, config.base_model_revision)
    base_hash = hash_base_weights(config.base_model, revision)
    w_ref = weights_ref(weights_dir, root)
    manifest_ref = training_set.write_manifest(out_dir / TRAIN_MANIFEST, root)
    if manifest_ref.sha256 != manifest_sha:  # canonical JSON: must be identical
        raise TrainError("Il manifesto scritto non corrisponde a quello dell'insieme di addestramento.")
    used_ids = set(keep)
    conditions = sorted(
        {e.condition_key for i, e in enumerate(training_set.manifest.examples) if i in used_ids}
    )

    hyperparams: dict[str, Any] = {
        HP_BASE_REVISION: revision,
        HP_LANGUAGE: config.language,
        HP_TASK: config.task,
        "learning_rate": lr,
        "num_epochs": config.num_epochs,
        "max_steps": config.max_steps,
        "total_steps": total_steps,
        "batch_size": config.batch_size,
        "grad_accum_steps": config.grad_accum_steps,
        "effective_batch_size": config.batch_size * config.grad_accum_steps,
        "warmup_steps": warmup,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "optimizer": "adamw",
        "scheduler": "linear",
        "gradient_checkpointing": config.gradient_checkpointing,
        "precision": "bf16-autocast" if autocast_dtype is not None else "fp32",
        "n_train_examples": n,
    }
    if config.technique == AdapterTechnique.LORA:
        hyperparams.update(
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            lora_target_modules=list(config.lora_target_modules),
        )
    else:
        hyperparams["freeze_encoder"] = config.freeze_encoder

    artifact = AdapterArtifact(
        adapter_id=adapter_id,
        base_model=config.base_model,
        base_model_sha256=base_hash,
        technique=config.technique,
        hyperparams=hyperparams,
        seed=config.seed,
        train_manifest_sha256=manifest_sha,
        conditions=conditions,
        weights=w_ref,
    )
    artifact_ref = save_adapter_artifact(artifact, out_dir / ADAPTER_JSON, root)

    record = TrainingRecord(
        adapter_id=adapter_id,
        base_model=config.base_model,
        base_model_revision=revision,
        technique=config.technique,
        started_at=started_at,
        finished_at=finished_at,
        wall_time_s=round(wall, 3),
        rss_before_mb=round(mem.before_mb, 1),
        peak_rss_mb=round(mem.peak_mb, 1),
        cuda_peak_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else None,
        device=device,
        tier=tier,
        num_threads=torch.get_num_threads(),
        steps=step,
        epochs_completed=round(batches_done / batches_per_epoch, 3),
        n_examples=n,
        skipped=skipped,
        losses=losses,
        trainable_params=trainable,
        total_params=total_params,
        versions=_versions(),
    )
    record_path = out_dir / TRAINING_RECORD
    record_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    record_ref = ArtifactRef.from_file(record_path, kind="training_record", root=root)
    return TrainResult(
        artifact=artifact,
        artifact_ref=artifact_ref,
        manifest_ref=manifest_ref,
        record_ref=record_ref,
        record=record,
        adapter_dir=str(out_dir),
    )
