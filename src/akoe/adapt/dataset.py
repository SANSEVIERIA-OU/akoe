# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Multi-context training data for the subject adapter (IMPLEMENTATION_PLAN M4, issue #23).

The subject model learns his voice from every occasion he speaks (phone calls,
rooms, the car). This module turns the subject's *verified* dataset items into a
training set:

- only verified items, never holdout items (the holdout is for self-assessment);
- every recording condition is kept, and conditions can be balanced by
  oversampling the scarce ones (each repeat gets its own augmentation draw);
- an optional caller-supplied ``augment(audio, sr, rng) -> (audio, params)``
  simulates channels, codecs and noise (written elsewhere, M3); a configurable
  fraction of the examples stays clean so normal speech is not degraded;
- everything is reproducible from a seed: each example has its own RNG seed,
  derived from (seed, item_id, repeat), so adding an item does not change the
  augmentation of the others;
- a manifest lists every source item, every example, its augmentation
  parameters and the SHA-256 of its samples. The manifest is written as
  canonical JSON without timestamps, so its file hash is reproducible and goes
  into ``AdapterArtifact.train_manifest_sha256``.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from akoe.models import ArtifactRef, DatasetItem, DatasetSplit
from akoe.utils import sha256_file

MANIFEST_SCHEMA_VERSION = 1
TARGET_SAMPLE_RATE = 16_000
WHISPER_MAX_SECONDS = 30.0

# augment(audio, sr, rng) -> (augmented_audio, params). `params` must describe
# the simulation completely (channel, IR, codec, SNR...), it goes in the manifest.
Augment = Callable[[np.ndarray, int, np.random.Generator], tuple[np.ndarray, dict[str, Any]]]
# load(item) -> (mono float32 audio of the segment, sample rate)
AudioLoader = Callable[[DatasetItem], tuple[np.ndarray, int]]


class DatasetBuildError(Exception):
    """Raised with an operator-facing (Italian) message."""


class ExclusionReason:
    UNVERIFIED = "unverified"
    HOLDOUT = "holdout"
    UNSPLIT = "unsplit"
    EMPTY_TEXT = "empty_text"
    TOO_LONG = "too_long"
    EMPTY_AUDIO = "empty_audio"


class TrainingSetConfig(BaseModel):
    seed: int = 1234
    # Fraction of examples (per condition) left without augmentation.
    clean_fraction: float = Field(default=0.3, ge=0.0, le=1.0)
    # Every eligible item appears this many times before balancing.
    copies_per_item: int = Field(default=1, ge=1)
    # Oversample scarce conditions towards the largest one (by item count).
    balance_conditions: bool = True
    # Upper bound of the oversampling factor of a condition (avoids repeating
    # a handful of items dozens of times).
    max_oversample: float = Field(default=4.0, ge=1.0)
    target_sample_rate: int = TARGET_SAMPLE_RATE
    max_seconds: float = WHISPER_MAX_SECONDS
    # Items with split=None are excluded unless explicitly allowed: the holdout
    # must be assigned before training (M1), otherwise self-assessment leaks.
    allow_unsplit: bool = False


class SourceRecord(BaseModel):
    item_id: str
    recording_id: str
    condition_key: str
    audio_sha256: str  # hash of the source file declared in the DatasetItem
    start: float
    end: float
    text: str


class ExcludedRecord(BaseModel):
    item_id: str
    reason: str


class ExampleRecord(BaseModel):
    example_id: str
    item_id: str
    condition_key: str
    repeat: int
    clean: bool
    rng_seed: int
    augmentation: dict[str, Any] | None = None
    text: str
    sample_rate: int
    n_samples: int
    # SHA-256 of the example samples as little-endian float32 (format independent).
    audio_sha256: str
    clipped_samples: int = 0
    # Relative to the manifest directory, when the set was materialized on disk.
    path: str | None = None


class ConditionSummary(BaseModel):
    n_items: int
    n_examples: int
    n_clean: int
    seconds: float


class TrainingManifest(BaseModel):
    schema_version: int = MANIFEST_SCHEMA_VERSION
    config: TrainingSetConfig
    augment_name: str | None = None
    augment_version: str | None = None
    sources: list[SourceRecord] = Field(default_factory=list)
    excluded: list[ExcludedRecord] = Field(default_factory=list)
    examples: list[ExampleRecord] = Field(default_factory=list)
    conditions: dict[str, ConditionSummary] = Field(default_factory=dict)

    def canonical_json(self) -> bytes:
        data = self.model_dump(mode="json")
        return (json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json()).hexdigest()

    def write(self, path: Path, root: Path | None = None) -> ArtifactRef:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.canonical_json())
        return ArtifactRef.from_file(path, kind="train_manifest", root=root)

    @classmethod
    def read(cls, path: Path) -> "TrainingManifest":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    @property
    def condition_keys(self) -> list[str]:
        return sorted({e.condition_key for e in self.examples})


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def samples_sha256(audio: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(audio, dtype="<f4").tobytes()).hexdigest()


def derive_seed(seed: int, *parts: object) -> int:
    """Stable 63-bit seed from the global seed and identifying parts."""
    key = "|".join([str(seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") >> 1


def _rng(seed: int, *parts: object) -> np.random.Generator:
    return np.random.default_rng(derive_seed(seed, *parts))


def _json_safe(value: Any) -> Any:
    """Make augmentation parameters JSON-serialisable (numpy, paths, tuples)."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio)
    if a.ndim == 2:
        # soundfile layout is (frames, channels)
        a = a.mean(axis=1)
    elif a.ndim != 1:
        raise DatasetBuildError(f"Audio con forma non supportata: {a.shape}.")
    return a.astype(np.float32, copy=False)


def resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio.astype(np.float32, copy=False)
    from scipy.signal import resample_poly

    g = gcd(int(sr), int(target_sr))
    return resample_poly(audio, target_sr // g, sr // g).astype(np.float32)


class FileAudioLoader:
    """Reads the item's segment from its audio file and checks the file hash
    once per file (chain of custody: refuse audio that does not match)."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root
        self._verified: set[Path] = set()

    def __call__(self, item: DatasetItem) -> tuple[np.ndarray, int]:
        import soundfile as sf

        path = item.audio.resolve(self.root)
        if not path.exists():
            raise DatasetBuildError(f"File audio mancante per il segmento {item.item_id}: {path}.")
        if path not in self._verified:
            if sha256_file(path) != item.audio.sha256:
                raise DatasetBuildError(
                    f"L'impronta SHA-256 del file {path} non corrisponde a quella registrata "
                    f"(segmento {item.item_id}): file modificato o sostituito, addestramento interrotto."
                )
            self._verified.add(path)
        info = sf.info(str(path))
        start = max(0, int(round(item.start * info.samplerate)))
        stop = min(info.frames, int(round(item.end * info.samplerate)))
        audio, sr = sf.read(str(path), start=start, stop=max(start, stop), dtype="float32", always_2d=False)
        return to_mono_float32(audio), int(sr)


# --------------------------------------------------------------------------
# selection and planning (no audio involved: fully determined by items + config)
# --------------------------------------------------------------------------


class PlannedExample(BaseModel):
    item_id: str
    condition_key: str
    repeat: int
    clean: bool
    rng_seed: int


def select_items(
    items: Sequence[DatasetItem], config: TrainingSetConfig
) -> tuple[list[DatasetItem], list[ExcludedRecord]]:
    """Keep verified training items only. Duplicated item ids are an error."""
    seen: set[str] = set()
    kept: list[DatasetItem] = []
    excluded: list[ExcludedRecord] = []
    for item in items:
        if item.item_id in seen:
            raise DatasetBuildError(f"Segmento duplicato nel dataset: {item.item_id}.")
        seen.add(item.item_id)
        reason = None
        if not item.verified:
            reason = ExclusionReason.UNVERIFIED
        elif item.split == DatasetSplit.HOLDOUT:
            reason = ExclusionReason.HOLDOUT
        elif item.split is None and not config.allow_unsplit:
            reason = ExclusionReason.UNSPLIT
        elif not item.text.strip():
            reason = ExclusionReason.EMPTY_TEXT
        elif item.end - item.start > config.max_seconds:
            reason = ExclusionReason.TOO_LONG
        elif item.end <= item.start:
            reason = ExclusionReason.EMPTY_AUDIO
        if reason:
            excluded.append(ExcludedRecord(item_id=item.item_id, reason=reason))
        else:
            kept.append(item)
    kept.sort(key=lambda i: i.item_id)
    excluded.sort(key=lambda e: e.item_id)
    return kept, excluded


def plan_examples(
    items: Sequence[DatasetItem], config: TrainingSetConfig, *, has_augment: bool
) -> list[PlannedExample]:
    """Decide how many times each item appears and which copies stay clean.

    Per condition c with n_c items: every item gets ``copies_per_item`` copies;
    with balancing, the condition is brought towards ``max_c n_c`` items, capped
    at ``n_c * max_oversample``. Clean copies are exactly
    ``round(clean_fraction * n_examples_c)``, spread over distinct items first.
    """
    groups: dict[str, list[DatasetItem]] = {}
    for item in sorted(items, key=lambda i: i.item_id):
        groups.setdefault(item.condition.key(), []).append(item)
    if not groups:
        return []
    largest = max(len(g) for g in groups.values())

    planned: list[PlannedExample] = []
    for key in sorted(groups):
        group = groups[key]
        n = len(group)
        factor = 1.0
        if config.balance_conditions:
            factor = min(config.max_oversample, largest / n)
        target = int(math.floor(n * factor + 1e-9)) * config.copies_per_item
        base, extra = divmod(target, n)
        repeats = {item.item_id: base for item in group}
        if extra:
            order = _rng(config.seed, "oversample", key).permutation(n)
            for idx in order[:extra]:
                repeats[group[idx].item_id] += 1

        copies = [(item.item_id, r) for item in group for r in range(repeats[item.item_id])]
        n_clean = len(copies) if not has_augment else int(round(config.clean_fraction * len(copies)))
        # clean copies go to repeat 0 of distinct items first, in a seeded order
        tie = _rng(config.seed, "clean", key).permutation(len(copies))
        ranked = sorted(range(len(copies)), key=lambda i: (copies[i][1], tie[i]))
        clean_set = set(ranked[:n_clean])
        for i, (item_id, r) in enumerate(copies):
            planned.append(
                PlannedExample(
                    item_id=item_id,
                    condition_key=key,
                    repeat=r,
                    clean=i in clean_set,
                    rng_seed=derive_seed(config.seed, "example", item_id, r),
                )
            )
    return planned


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------


class TrainingSet:
    """Examples ready for training plus their manifest.

    Audio is either kept in memory or materialized as float32 WAV files next to
    the manifest (``example(i)`` reads them back and checks their hash).
    """

    def __init__(
        self,
        manifest: TrainingManifest,
        *,
        audio: list[np.ndarray] | None = None,
        directory: Path | None = None,
    ) -> None:
        if audio is None and directory is None:
            raise ValueError("either in-memory audio or a directory is required")
        self.manifest = manifest
        self._audio = audio
        self.directory = directory

    def __len__(self) -> int:
        return len(self.manifest.examples)

    @property
    def sample_rate(self) -> int:
        return self.manifest.config.target_sample_rate

    def sha256(self) -> str:
        return self.manifest.sha256()

    def example(self, i: int) -> tuple[np.ndarray, str]:
        rec = self.manifest.examples[i]
        if self._audio is not None:
            return self._audio[i], rec.text
        import soundfile as sf

        assert self.directory is not None and rec.path is not None
        audio, sr = sf.read(str(self.directory / rec.path), dtype="float32", always_2d=False)
        if sr != rec.sample_rate or samples_sha256(audio) != rec.audio_sha256:
            raise DatasetBuildError(
                f"L'esempio di addestramento {rec.example_id} non corrisponde al manifesto "
                f"(file {rec.path} modificato)."
            )
        return audio, rec.text

    def write_manifest(self, path: Path, root: Path | None = None) -> ArtifactRef:
        return self.manifest.write(path, root)

    @classmethod
    def from_manifest(cls, manifest_path: Path, expected_sha256: str | None = None) -> "TrainingSet":
        """Open a materialized training set (e.g. built on the laptop, trained on a
        workstation). Example files are verified when read."""
        if expected_sha256 is not None and sha256_file(manifest_path) != expected_sha256:
            raise DatasetBuildError(
                f"Il manifesto {manifest_path} non corrisponde all'impronta SHA-256 attesa."
            )
        manifest = TrainingManifest.read(manifest_path)
        if any(e.path is None for e in manifest.examples):
            raise DatasetBuildError(
                "Il manifesto non elenca i file audio degli esempi: l'insieme non è stato salvato su disco."
            )
        return cls(manifest, directory=manifest_path.parent)


def _augment_identity(augment: Augment | None) -> tuple[str | None, str | None]:
    if augment is None:
        return None, None
    name = getattr(augment, "name", None) or getattr(augment, "__qualname__", None)
    name = name or type(augment).__qualname__
    version = getattr(augment, "version", None)
    return str(name), (str(version) if version is not None else None)


def build_training_set(
    items: Sequence[DatasetItem],
    *,
    config: TrainingSetConfig | None = None,
    augment: Augment | None = None,
    root: Path | None = None,
    loader: AudioLoader | None = None,
    out_dir: Path | None = None,
) -> TrainingSet:
    """Build the multi-condition training set.

    Args:
        items: the subject's dataset items (any mix; only verified training items are used).
        config: selection, balancing and augmentation settings.
        augment: optional channel/codec/noise simulation, called as
            ``augment(audio, sr, rng)`` and returning ``(audio, params)``.
        root: case root used to resolve relative audio paths (default loader).
        loader: custom segment loader (default: read from file and verify SHA-256).
        out_dir: when given, examples are written there as float32 WAV with the
            manifest (``train_manifest.json``); otherwise they stay in memory.
    """
    config = config or TrainingSetConfig()
    loader = loader or FileAudioLoader(root)
    kept, excluded = select_items(items, config)
    if not kept:
        raise DatasetBuildError(
            "Nessun segmento verificato utilizzabile per l'addestramento "
            f"({len(excluded)} esclusi: non verificati, di autovalutazione o senza divisione)."
        )
    by_id = {i.item_id: i for i in kept}
    plan = plan_examples(kept, config, has_augment=augment is not None)

    if out_dir is not None:
        (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    sr_out = config.target_sample_rate
    cache: dict[str, np.ndarray] = {}
    examples: list[ExampleRecord] = []
    audio_mem: list[np.ndarray] | None = [] if out_dir is None else None
    empty_ids: set[str] = set()

    for p in plan:
        item = by_id[p.item_id]
        base = cache.get(item.item_id)
        if base is None:
            raw, sr = loader(item)
            base = resample(to_mono_float32(raw), sr, sr_out)
            if base.size == 0:
                empty_ids.add(item.item_id)
                continue
            # keep only the current item's segment: plans are grouped by item
            cache.clear()
            cache[item.item_id] = base

        params: dict[str, Any] | None = None
        audio = base
        if not p.clean:
            assert augment is not None
            out, params = augment(base.copy(), sr_out, np.random.default_rng(p.rng_seed))
            audio = np.asarray(out)
            if audio.ndim != 1:
                raise DatasetBuildError("La funzione di simulazione deve restituire audio mono (1-D).")
            if not np.all(np.isfinite(audio)):
                raise DatasetBuildError(
                    f"La simulazione ha prodotto valori non finiti per l'esempio {p.item_id}#{p.repeat}."
                )
            params = _json_safe(dict(params or {}))
        audio = audio.astype(np.float32, copy=False)
        clipped = int(np.count_nonzero(np.abs(audio) > 1.0))
        if clipped:
            audio = np.clip(audio, -1.0, 1.0)

        example_id = f"{p.item_id}#{p.repeat}"
        rel_path = None
        if out_dir is not None:
            import soundfile as sf

            rel_path = f"audio/{len(examples):06d}.wav"
            sf.write(str(out_dir / rel_path), audio, sr_out, subtype="FLOAT")
        else:
            assert audio_mem is not None
            audio_mem.append(audio)
        examples.append(
            ExampleRecord(
                example_id=example_id,
                item_id=p.item_id,
                condition_key=p.condition_key,
                repeat=p.repeat,
                clean=p.clean,
                rng_seed=p.rng_seed,
                augmentation=params,
                text=item.text.strip(),
                sample_rate=sr_out,
                n_samples=int(audio.size),
                audio_sha256=samples_sha256(audio),
                clipped_samples=clipped,
                path=rel_path,
            )
        )

    for item_id in sorted(empty_ids):
        excluded.append(ExcludedRecord(item_id=item_id, reason=ExclusionReason.EMPTY_AUDIO))
    excluded.sort(key=lambda e: e.item_id)
    used = [i for i in kept if i.item_id not in empty_ids]

    conditions: dict[str, ConditionSummary] = {}
    for key in sorted({i.condition.key() for i in used}):
        exs = [e for e in examples if e.condition_key == key]
        conditions[key] = ConditionSummary(
            n_items=sum(1 for i in used if i.condition.key() == key),
            n_examples=len(exs),
            n_clean=sum(1 for e in exs if e.clean),
            seconds=round(sum(e.n_samples for e in exs) / sr_out, 3),
        )

    name, version = _augment_identity(augment)
    manifest = TrainingManifest(
        config=config,
        augment_name=name,
        augment_version=version,
        sources=[
            SourceRecord(
                item_id=i.item_id,
                recording_id=i.recording_id,
                condition_key=i.condition.key(),
                audio_sha256=i.audio.sha256,
                start=i.start,
                end=i.end,
                text=i.text.strip(),
            )
            for i in used
        ],
        excluded=excluded,
        examples=examples,
        conditions=conditions,
    )
    if not examples:
        raise DatasetBuildError("Nessun esempio di addestramento prodotto: tutti i segmenti sono vuoti.")
    if out_dir is not None:
        manifest.write(out_dir / "train_manifest.json")
        return TrainingSet(manifest, directory=out_dir)
    return TrainingSet(manifest, audio=audio_mem)
