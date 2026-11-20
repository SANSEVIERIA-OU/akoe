# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Deterministic fake backends for tests (no models, no downloads)."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from akoe.config import StageKind
from akoe.models import Alternative, SegmentHypotheses, WordHypothesis


class FakeVoiceprint:
    """Embedding = normalized band energies: same signal -> same vector."""

    name = "fake-voiceprint"

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        spec = np.abs(np.fft.rfft(audio.astype(np.float64), n=2048))
        bands = np.array_split(spec, 16)
        v = np.array([b.mean() for b in bands]) + 1e-9
        return v / np.linalg.norm(v)


class FakeASR:
    name = "fake-asr"

    def __init__(self, text: str = "buongiorno a tutti", name: str | None = None) -> None:
        self.text = text
        if name is not None:
            self.name = name

    def transcribe(self, audio, sr, *, segments=None, n_best=5, audio_provenance=None,
                   audio_variant="original"):
        dur = len(audio) / sr
        segs = segments or [(0.0, dur)]
        prov = audio_provenance or StageKind.NEURAL  # historical default of this fake
        out = []
        for start, end in segs:
            words = self.text.split()
            step = (end - start) / max(len(words), 1)
            out.append(
                SegmentHypotheses(
                    start=start,
                    end=end,
                    audio_provenance=prov,
                    hypotheses=[Alternative(text=self.text, score=0.0, origin=self.name,
                                            audio_variant=audio_variant)],
                    words=[
                        WordHypothesis(
                            text=w, start=start + i * step, end=start + (i + 1) * step,
                            confidence=0.9, origin=self.name,
                        )
                        for i, w in enumerate(words)
                    ],
                )
            )
        return out


class FakeScorer:
    """Prefers candidates sharing more words with a hidden truth."""

    name = "fake-scorer"

    def __init__(self, truth: str, name: str | None = None) -> None:
        self.truth = set(truth.split())
        if name is not None:
            self.name = name

    def score(self, audio, sr, candidates):
        return [float(len(self.truth & set(c.split()))) for c in candidates]


# --------------------------------------------------------------------------
# Engine workflows (engine.py): lexicon decoding, alignment, training
# --------------------------------------------------------------------------


class FakeLexiconDecoder:
    """`decode_both` like adapt.lexicon.WhisperLexiconDecoder: with the lexicon
    the first confirmed term replaces the last word of the plain text."""

    def __init__(self, text: str, terms) -> None:
        from akoe.adapt.lexicon import _resolve_terms

        self.text = text
        self.terms, self.lexicon_sha256 = _resolve_terms(terms)

    def decode(self, audio, sr, *, use_lexicon: bool) -> str:
        if not use_lexicon or not self.terms:
            return self.text
        words = self.text.split()
        return " ".join(words[:-1] + [self.terms[0]])

    def decode_both(self, audio, sr):
        from akoe.adapt.lexicon import dual_decode

        return dual_decode(lambda use: self.decode(audio, sr, use_lexicon=use), self.terms)


class FakeLexiconASR(FakeASR):
    """FakeASR that can also decode with a case lexicon (duck-typed like
    WhisperHFBackend.lexicon_decoder)."""

    def lexicon_decoder(self, terms):
        return FakeLexiconDecoder(self.text, terms)


def uniform_ctc_scorer(name: str = "fake-ctc"):
    """CTC scorer over uniform emissions (one frame per 320 samples): every
    path is equally likely, so any text aligns with score 0."""
    from akoe.transcribe.ctc_scorer import CTCForcedAlignmentScorer

    letters = "abcdefghijklmnopqrstuvwxyzàèéìòù'"
    vocab = {"<pad>": 0, "|": 1, **{c: i + 2 for i, c in enumerate(letters)}}

    def emit(a16: np.ndarray) -> np.ndarray:
        n = max(1, (len(a16) - 400) // 320 + 1)
        return np.full((n, len(vocab)), -np.log(len(vocab)), dtype=np.float32)

    return CTCForcedAlignmentScorer.from_emission_fn(emit, vocab, name=name)


def fake_trainer(training_set, out_dir, config, *, adapter_id, root=None, tier=None):
    """Writes an adapter folder with the real artifact layout (weights +
    SHA256SUMS, train manifest, adapter.json, training record) without
    training anything."""
    from akoe.adapt.artifact import (
        ADAPTER_JSON,
        TRAIN_MANIFEST,
        TRAINING_RECORD,
        WEIGHTS_DIR,
        save_adapter_artifact,
        weights_ref,
    )
    from akoe.adapt.train import TrainingRecord, TrainResult
    from akoe.models import AdapterArtifact, ArtifactRef

    out_dir = Path(out_dir)
    wdir = out_dir / WEIGHTS_DIR
    wdir.mkdir(parents=True, exist_ok=False)
    seed = hashlib.sha256(training_set.sha256().encode()).digest()
    (wdir / "adapter_model.safetensors").write_bytes(seed * 64)
    (wdir / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": config.base_model,
                                                          "peft_type": "LORA", "r": config.lora_r}))
    w_ref = weights_ref(wdir, root)
    manifest_ref = training_set.write_manifest(out_dir / TRAIN_MANIFEST, root)
    conditions = sorted({e.condition_key for e in training_set.manifest.examples})
    artifact = AdapterArtifact(
        adapter_id=adapter_id, base_model=config.base_model, base_model_sha256=None,
        technique=config.technique, hyperparams={"lora_r": config.lora_r, "fake": True}, seed=config.seed,
        train_manifest_sha256=training_set.sha256(), conditions=conditions, weights=w_ref,
    )
    artifact_ref = save_adapter_artifact(artifact, out_dir / ADAPTER_JSON, root)
    now = datetime.now(timezone.utc)
    record = TrainingRecord(
        adapter_id=adapter_id, base_model=config.base_model, technique=config.technique, started_at=now,
        finished_at=now, wall_time_s=0.0, rss_before_mb=0.0, peak_rss_mb=0.0, device="cpu", tier=tier,
        num_threads=1, steps=0, epochs_completed=0.0, n_examples=len(training_set), trainable_params=0,
        total_params=0,
    )
    (out_dir / TRAINING_RECORD).write_text(record.model_dump_json(indent=2))
    record_ref = ArtifactRef.from_file(out_dir / TRAINING_RECORD, kind="training_record", root=root)
    return TrainResult(artifact=artifact, artifact_ref=artifact_ref, manifest_ref=manifest_ref,
                       record_ref=record_ref, record=record, adapter_dir=str(out_dir))


class FakeProvider:
    """`engine.BackendProvider` made of fakes."""

    def __init__(self, generic_text: str = "vado a casa adesso", adapted_text: str = "vado a casa adesso") -> None:
        self.generic_text = generic_text
        self.adapted_text = adapted_text
        self.calls: list[str] = []

    def voiceprint(self):
        self.calls.append("voiceprint")
        return FakeVoiceprint()

    def aligner(self):
        from akoe.transcribe.ctc_aligner import CTCTranscriptAligner

        self.calls.append("aligner")
        return CTCTranscriptAligner(uniform_ctc_scorer())

    def asr(self, *, tier, adapter=None, whisper_model=None):
        self.calls.append("asr")
        out = {"generic": FakeLexiconASR(self.generic_text, name="fake-asr/generic")}
        if adapter is not None:
            out["adapted"] = FakeLexiconASR(self.adapted_text,
                                            name=f"fake-asr/adapted:{adapter.artifact.adapter_id}")
        return out

    def scorers(self, *, tier, whisper_model=None):
        self.calls.append("scorers")
        return [FakeScorer("vado a casa", name="fake-ctc"), FakeScorer("vado a casa adesso", name="fake-whisper")]

    def trainer(self):
        self.calls.append("trainer")
        return fake_trainer
