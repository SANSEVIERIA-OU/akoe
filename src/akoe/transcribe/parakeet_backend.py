# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""NVIDIA Parakeet-TDT backend via NeMo (`ASRBackend`), with entropy-based
word confidence (issue #12, MASTER_PLAN §3.6.2).

NeMo is an optional, heavy dependency and is **not** installed in the main
environment. Feasibility checked on 2026-11-20 in a throwaway venv:
`nemo_toolkit[asr]` 3.0.0 installs and imports on Python 3.13.5 with
torch 2.15.0+cpu (install torch CPU first, then
`pip install "nemo_toolkit[asr]" --extra-index-url https://download.pytorch.org/whl/cpu`;
about 2.2 GB on disk). Without NeMo every call raises `BackendUnavailable`
(a NotImplementedError, logged as a skipped stage) with instructions in Italian.

Confidence: NeMo `confidence_cfg` with Tsallis entropy, alpha = 0.33,
exponential normalization, per-word aggregation "min", blank excluded
(Laptev & Ginsburg 2022). Values are raw (uncalibrated): calibration happens
in self-assessment (M5).

CTC head: Parakeet-TDT 0.6B v3 has none. Its .nemo targets
`EncDecRNNTBPEModel` (pure TDT); the config keeps a vestigial `aux_ctc` block
(num_classes -1, empty vocabulary) but the checkpoint contains only
preprocessor/encoder/decoder/joint weights (state-dict keys inspected without
downloading the weights). NeMo Forced Aligner needs a CTC or hybrid model, so
forced alignment uses the separate CTC scorer (`ctc_scorer.py`).

Memory: the 0.6B model is 2.5 GB in float32; restoring a .nemo instantiates the
model and then loads the state dict, so expect a peak of roughly twice that.
"""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from akoe.config import StageKind
from akoe.models import Alternative, SegmentHypotheses, WordHypothesis
from akoe.neural import (
    TARGET_SR,
    BackendUnavailable,
    ModelProvenance,
    hash_files,
    iter_segments,
    prepare_audio,
    resolve_device,
)

DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
KNOWN_REVISIONS = {DEFAULT_MODEL: "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"}  # cc-by-4.0, 2026-11-20
KNOWN_LICENSES = {DEFAULT_MODEL: "cc-by-4.0"}
MAX_WINDOW_S = 30.0

NEMO_MISSING_MESSAGE = (
    "Il backend Parakeet richiede NVIDIA NeMo, che non è installato in questo ambiente. "
    "Per abilitarlo, in un ambiente Python separato (verificato con Python 3.13, NeMo 3.0.0, "
    "PyTorch 2.15 CPU): installare prima PyTorch CPU "
    "(pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu), poi "
    "pip install \"nemo_toolkit[asr]\" --extra-index-url https://download.pytorch.org/whl/cpu "
    "e infine questo pacchetto. Servono circa 2,2 GB su disco e, per Parakeet-TDT 0.6B, "
    "circa 5 GB di RAM al caricamento."
)


@dataclass
class EntropyConfidence:
    """NeMo confidence settings (recorded in the provenance)."""

    entropy_type: str = "tsallis"
    alpha: float = 0.33
    entropy_norm: str = "exp"
    aggregation: str = "min"
    exclude_blank: bool = True
    tdt_include_duration: bool = False


def nemo_available() -> bool:
    try:
        import nemo.collections.asr  # noqa: F401
    except Exception:  # noqa: BLE001 - NeMo can fail at import with non-ImportError errors
        return False
    return True


class ParakeetTDTBackend:
    """Parakeet-TDT (or any NeMo transducer model) with word timestamps and
    entropy-based word confidence. N-best (n_best > 1) comes from NeMo beam
    search; beam hypotheses carry NeMo's beam score, the greedy hypothesis its
    greedy score, so the two kinds are marked by origin and not mixed silently."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        *,
        revision: str | None = None,
        nemo_file: str | None = None,
        device: str = "cpu",
        confidence: EntropyConfidence | None = None,
        beam_size: int = 4,
        local_files_only: bool = False,
        audio_provenance: StageKind = StageKind.DETERMINISTIC,
    ) -> None:
        self.model_id = model_id
        self.revision = revision if revision is not None else KNOWN_REVISIONS.get(model_id)
        self.short_name = Path(model_id).stem if model_id.endswith(".nemo") else model_id.rstrip("/").split("/")[-1]
        self.nemo_file = nemo_file or f"{self.short_name}.nemo"
        self.device = device
        self.confidence = confidence or EntropyConfidence()
        self.beam_size = beam_size
        self.local_files_only = local_files_only
        self.audio_provenance = audio_provenance
        self.name = f"{self.short_name}/generic"
        self._model: Any = None
        self._greedy_cfg: Any = None
        self._provenance: ModelProvenance | None = None
        self._lock = threading.Lock()

    @property
    def provenance(self) -> ModelProvenance | None:
        return self._provenance

    def load(self) -> ModelProvenance:
        with self._lock:
            if self._model is None:
                self._load()
        assert self._provenance is not None
        return self._provenance

    def unload(self) -> None:
        """Free the weights (provenance is kept; the next call reloads)."""
        import gc

        with self._lock:
            self._model = self._greedy_cfg = None
        gc.collect()

    def _resolve_nemo(self) -> tuple[Path, str | None]:
        p = Path(self.model_id).expanduser()
        if p.suffix == ".nemo" and p.is_file():
            return p, None
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(self.model_id, self.nemo_file, revision=self.revision,
                                    local_files_only=self.local_files_only))
        commit = path.parent.name if path.parent.parent.name == "snapshots" else self.revision
        return path, commit

    def _load(self) -> None:
        try:
            import nemo
            import nemo.collections.asr as nemo_asr
            from omegaconf import open_dict
        except ImportError as e:
            raise BackendUnavailable(NEMO_MISSING_MESSAGE) from e
        device = resolve_device(self.device)
        path, commit = self._resolve_nemo()
        sha = hash_files([path])
        model = nemo_asr.models.ASRModel.restore_from(str(path), map_location=device)
        model.eval()
        cfg = model.cfg.decoding
        with open_dict(cfg):
            cfg.strategy = "greedy_batch"
            cfg.confidence_cfg = {
                "preserve_word_confidence": True,
                "preserve_token_confidence": True,
                "exclude_blank": self.confidence.exclude_blank,
                "aggregation": self.confidence.aggregation,
                "tdt_include_duration": self.confidence.tdt_include_duration,
                "method_cfg": {
                    "name": "entropy",
                    "entropy_type": self.confidence.entropy_type,
                    "alpha": self.confidence.alpha,
                    "entropy_norm": self.confidence.entropy_norm,
                },
            }
        model.change_decoding_strategy(cfg, verbose=False)
        self._model, self._greedy_cfg = model, cfg
        self._provenance = ModelProvenance(
            backend=self.name, model_id=self.model_id, revision=commit, weights=sha,
            device=device, dtype="float32", license=KNOWN_LICENSES.get(self.model_id),
            extra={
                "nemo_version": getattr(nemo, "__version__", None),
                "model_class": type(model).__name__,
                "has_ctc_head": hasattr(model, "ctc_decoder"),
                "confidence": asdict(self.confidence),
                "beam_size": self.beam_size,
            },
        )

    # ------------------------------------------------------------- decoding
    @staticmethod
    def _first(out: Any) -> Any:
        # Older NeMo / hybrid models may return (best_hypotheses, all_hypotheses).
        if isinstance(out, tuple):
            out = out[0]
        return out[0]

    def _greedy(self, chunk: np.ndarray) -> Any:
        out = self._model.transcribe([chunk], batch_size=1, return_hypotheses=True,
                                     timestamps=True, verbose=False)
        return self._first(out)

    def _beam(self, chunk: np.ndarray, n_best: int) -> list[Any]:
        import copy

        from omegaconf import open_dict

        cfg = copy.deepcopy(self._greedy_cfg)
        with open_dict(cfg):
            cfg.strategy = "malsd_batch"  # NeMo's recommended beam strategy (RNNT and TDT)
            cfg.beam.beam_size = max(self.beam_size, n_best)
            cfg.beam.return_best_hypothesis = False
            cfg.confidence_cfg.preserve_word_confidence = False
            cfg.confidence_cfg.preserve_token_confidence = False
        self._model.change_decoding_strategy(cfg, verbose=False)
        try:
            out = self._first(self._model.transcribe([chunk], batch_size=1, return_hypotheses=True,
                                                     verbose=False))
        finally:
            self._model.change_decoding_strategy(self._greedy_cfg, verbose=False)
        if hasattr(out, "n_best_hypotheses"):
            hyps = list(out.n_best_hypotheses)
        elif isinstance(out, (list, tuple)):
            hyps = list(out)
        else:
            hyps = [out]
        return hyps[:n_best]

    def _words(self, hyp: Any, offset: float, seg_dur: float) -> list[WordHypothesis]:
        ts = getattr(hyp, "timestamp", None)
        stamps = ts.get("word", []) if isinstance(ts, dict) else []
        conf = list(getattr(hyp, "word_confidence", None) or [])
        if conf and len(conf) != len(stamps):
            conf = []  # cannot pair them reliably: report no confidence rather than a wrong one
        words = []
        for i, st in enumerate(stamps):
            s = float(min(max(st.get("start", 0.0), 0.0), seg_dur))
            e = float(min(max(st.get("end", s), s), seg_dur))
            c = float(conf[i]) if conf else None
            words.append(WordHypothesis(text=str(st.get("word", "")), start=offset + s, end=offset + e,
                                        confidence=c, origin=self.name))
        return words

    def transcribe(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        segments: list[tuple[float, float]] | None = None,
        n_best: int = 5,
        audio_provenance: StageKind | None = None,
        audio_variant: str = "original",
    ) -> list[SegmentHypotheses]:
        if self._model is None:
            self.load()
        prov = audio_provenance or self.audio_provenance
        a = prepare_audio(audio, sr)
        results = []
        for start, end, chunk in iter_segments(a, TARGET_SR, segments, max_seconds=MAX_WINDOW_S):
            greedy = self._greedy(chunk)
            hyps = [Alternative(text=greedy.text.strip(), score=float(greedy.score), origin=self.name,
                                audio_variant=audio_variant)]
            if n_best > 1:
                for h in self._beam(chunk, n_best):
                    text = (h.text or "").strip()
                    if all(text != x.text for x in hyps):
                        hyps.append(Alternative(text=text, score=float(h.score), origin=f"{self.name}/beam",
                                                audio_variant=audio_variant))
            results.append(SegmentHypotheses(
                start=start, end=end, audio_provenance=prov, hypotheses=hyps,
                words=self._words(greedy, start, end - start),
            ))
        return results
