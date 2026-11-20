# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Common flow of the attention-decoder ASR backends on Hugging Face
transformers (Cohere Transcribe, Qwen3-ASR), same conventions as
`whisper_backend`:

- **Alternatives.** Beam-search N-best (origin "<model>/generic"), plus
  optional temperature / epsilon sampling (origin ".../sampling", seed
  recorded). Duplicates are removed keeping the first origin.
- **One score for every candidate:** mean per-token log-probability of the
  candidate (plus the end token) under the decoder, teacher-forced on the
  same audio, so beam and sampled candidates are comparable. Candidates are
  ranked by it; nothing is dropped.
- **Word confidence:** product of the probabilities of the word's tokens
  (teacher-forced, top hypothesis), raw and uncalibrated.
- **Word times** from a `WordTimer` (forced aligner); without one, words
  span the whole segment (no invented precision).
- **Memory** (EDGE laptop, 14 GB shared): weights load lazily in bfloat16 by
  default; with `low_memory` the ASR weights are released after decoding and
  before the aligner is loaded, and an aligner owned by the backend is
  released at the end of the call. `unload()` frees everything.
- **Provenance:** model id, revision, SHA-256 of every loaded file (checked
  against the published hashes when known), dtype, device, decoding settings
  and the aligner's own provenance.
"""
from __future__ import annotations

import copy
import gc
import threading
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from akoe.config import StageKind
from akoe.models import Alternative, SegmentHypotheses
from akoe.neural import (
    TARGET_SR,
    BackendUnavailable,
    ModelProvenance,
    hash_files,
    iter_segments,
    prepare_audio,
    resolve_hf_snapshot,
)
from akoe.transcribe.whisper_backend import SamplingConfig, TeacherForcedScore, TokenScore
from akoe.transcribe.word_timing import (
    WordTimer,
    build_words,
    resolve_timer,
    timer_provenance,
    token_word_groups,
)


class ModelIntegrityError(ValueError):
    """A downloaded or local model file does not match its published SHA-256."""


def resolve_snapshot(model_id: str, revision: str | None, patterns: list[str], local_files_only: bool,
                     family: str) -> tuple[Path, str | None]:
    """`resolve_hf_snapshot`, with "model not obtainable" turned into
    `BackendUnavailable` (Italian message) so one missing model skips its
    stage instead of stopping the whole ensemble."""
    try:
        return resolve_hf_snapshot(model_id, revision, patterns, local_files_only)
    except Exception as e:
        from huggingface_hub import errors as hfe

        rev = f"@{revision}" if revision else ""
        if isinstance(e, hfe.GatedRepoError):
            raise BackendUnavailable(
                f"Il modello {model_id}{rev} ({family}) è ad accesso controllato su Hugging Face: il "
                "proprietario deve accettarne le condizioni con il proprio account e impostare HF_TOKEN "
                "su questa macchina, oppure scaricare i file in una cartella locale e indicarla come modello."
            ) from e
        if isinstance(e, OSError | hfe.RepositoryNotFoundError | hfe.RevisionNotFoundError):
            raise BackendUnavailable(
                f"Il modello {model_id}{rev} ({family}) non è disponibile su questa macchina e non è stato "
                f"possibile scaricarlo: {e}"
            ) from e
        raise


def check_weights(weights: dict[str, str], expected: dict[str, str] | None, model_id: str) -> None:
    for name, sha in (expected or {}).items():
        if name in weights and weights[name] != sha:
            raise ModelIntegrityError(
                f"Il file {name} del modello {model_id} ha SHA-256 {weights[name]}, diverso da quello "
                f"pubblicato ({sha}). Caricamento rifiutato: verificare la provenienza dei pesi."
            )


class TransformersASRBackend:
    """Base class; subclasses implement `_load_model`, `_prepare`,
    `_generate_ids`, `_score_ids`, `_text_ids`, `_decode` and `_default_timer`."""

    family = "ASR"
    max_window_s = 30.0

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        device: str = "cpu",
        dtype: str = "bfloat16",
        language: str = "it",
        num_beams: int = 4,
        sampling: SamplingConfig | None = None,
        word_timestamps: bool = True,
        aligner: WordTimer | Callable[[], WordTimer] | None = None,
        max_new_tokens: int = 256,
        local_files_only: bool = False,
        audio_provenance: StageKind = StageKind.DETERMINISTIC,
        low_memory: bool = False,
        expected_weights: dict[str, str] | None = None,
    ) -> None:
        if dtype not in ("bfloat16", "float32", "float16"):
            raise ValueError(f"Tipo numerico non supportato: {dtype} (usare bfloat16, float16 o float32).")
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.dtype = dtype
        self.language = language
        self.num_beams = num_beams
        self.sampling = sampling
        self.word_timestamps = word_timestamps
        self.max_new_tokens = max_new_tokens
        self.local_files_only = local_files_only
        self.audio_provenance = audio_provenance
        self.low_memory = low_memory
        self.expected_weights = expected_weights
        self.short_name = model_id.rstrip("/").split("/")[-1]
        self.name = f"{self.short_name}/generic"
        self._aligner_arg = aligner
        self._timer: WordTimer | None = None
        self._owns_timer = False
        self._model: Any = None
        self._provenance: ModelProvenance | None = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------- loading
    @property
    def provenance(self) -> ModelProvenance | None:
        return self._provenance

    def load(self) -> ModelProvenance:
        with self._lock:
            if self._model is None:
                self._load_model()
        assert self._provenance is not None
        return self._provenance

    def _torch_dtype(self) -> Any:
        import torch

        return getattr(torch, self.dtype)

    def _resolve(self, patterns: list[str]) -> tuple[Path, str | None]:
        return resolve_snapshot(self.model_id, self.revision, patterns, self.local_files_only, self.family)

    def _hash_and_check(self, local: Path, suffixes: tuple[str, ...]) -> dict[str, str]:
        files = [p for p in local.iterdir() if p.is_file() and p.suffix in suffixes]
        weights = hash_files(files, root=local)
        check_weights(weights, self.expected_weights, self.model_id)
        return weights

    def _decoding_params(self) -> dict[str, Any]:
        return {"language": self.language, "num_beams": self.num_beams,
                "sampling": asdict(self.sampling) if self.sampling else None,
                "max_new_tokens": self.max_new_tokens, "low_memory": self.low_memory,
                "score": "mean per-token log-probability, teacher-forced (end token included)"}

    def release_model(self) -> None:
        """Free the ASR weights (provenance is kept; the next call reloads)."""
        with self._lock:
            self._model = None
            self._release_model_extra()
        gc.collect()

    def _release_model_extra(self) -> None:
        pass

    def unload(self) -> None:
        """Free the ASR weights and the aligner owned by this backend."""
        self.release_model()
        self._release_timer()

    def _release_timer(self) -> None:
        if self._owns_timer and self._timer is not None and hasattr(self._timer, "unload"):
            self._timer.unload()
            gc.collect()

    def _get_timer(self) -> WordTimer | None:
        if not self.word_timestamps:
            return None
        if self._timer is None:
            if self._aligner_arg is not None:
                self._timer = resolve_timer(self._aligner_arg)
            else:
                self._timer = self._default_timer()
                self._owns_timer = self._timer is not None
        return self._timer

    # ------------------------------------------------------------- subclass hooks
    def _load_model(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _prepare(self, chunk: np.ndarray) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def _generate_ids(self, ctx: Any, overrides: dict[str, Any]) -> list[list[int]]:  # pragma: no cover
        """Generated token ids per returned sequence, without prompt, end or padding."""
        raise NotImplementedError

    def _score_ids(self, ctx: Any, id_lists: list[list[int]]) -> list[list[float]]:  # pragma: no cover
        """Teacher-forced log-probability of each token of each list, plus the end token (last)."""
        raise NotImplementedError

    def _text_ids(self, text: str) -> list[int]:  # pragma: no cover - abstract
        raise NotImplementedError

    def _decode(self, ids: list[int]) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def _end_token_text(self) -> str:
        return "<eos>"

    def _default_timer(self) -> WordTimer | None:
        return None

    # ------------------------------------------------------------- generation
    def _generation_config(self, base: Any, **overrides: Any) -> Any:
        gc_ = copy.deepcopy(base)
        gc_.max_length = None
        gc_.update(max_new_tokens=self.max_new_tokens, return_dict_in_generate=True, **overrides)
        return gc_

    def _candidates(self, ctx: Any, n_best: int) -> list[tuple[list[int], str, str]]:
        """(ids, text, origin): beam N-best first, then sampling; unique texts."""
        import torch

        beams = max(self.num_beams, n_best)
        out = [(ids, self.name) for ids in self._generate_ids(
            ctx, {"num_beams": beams, "num_return_sequences": min(n_best, beams), "do_sample": False})]
        s = self.sampling
        if s is not None and s.n_samples > 0:
            extra: dict[str, Any] = {"temperature": s.temperature}
            if s.epsilon_cutoff is not None:
                extra["epsilon_cutoff"] = s.epsilon_cutoff
            if s.top_p is not None:
                extra["top_p"] = s.top_p
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(s.seed)
                sampled = self._generate_ids(
                    ctx, {"num_beams": 1, "num_return_sequences": s.n_samples, "do_sample": True, **extra})
            out += [(ids, f"{self.name}/sampling") for ids in sampled]
        seen: dict[str, tuple[list[int], str]] = {}
        for ids, origin in out:
            seen.setdefault(self._decode(ids), (ids, origin))
        return [(ids, text, origin) for text, (ids, origin) in seen.items()]

    # ------------------------------------------------------------- scoring
    def _token_scores(self, ids: list[int], lps: list[float]) -> list[TokenScore]:
        toks = [TokenScore(token=self._decode([t]), token_id=int(t), logprob=lp) for t, lp in zip(ids, lps)]
        if len(lps) > len(ids):
            toks.append(TokenScore(token=self._end_token_text(), token_id=-1, logprob=lps[-1]))
        return toks

    def score_texts(self, audio: np.ndarray, sr: int, texts: list[str]) -> list[TeacherForcedScore]:
        """Teacher-forced log-likelihood of each text on one audio stretch
        (at most `max_window_s` seconds)."""
        self.load()
        a = prepare_audio(audio, sr)
        if len(a) > self.max_window_s * TARGET_SR + 1:
            raise ValueError(
                f"{self.family} valuta al massimo {self.max_window_s:.0f} s per volta; il tratto dura "
                f"{len(a) / TARGET_SR:.1f} s. Selezionare un tratto più breve."
            )
        ctx = self._prepare(a)
        id_lists = [self._text_ids(t) for t in texts]
        out = []
        for text, ids, lps in zip(texts, id_lists, self._score_ids(ctx, id_lists)):
            out.append(TeacherForcedScore(text=text.strip(), total_logprob=float(sum(lps)),
                                          tokens=self._token_scores(ids, lps)))
        return out

    # ------------------------------------------------------------- transcribe
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
        """Hypotheses per segment (VAD regions in seconds, or the whole audio;
        regions longer than `max_window_s` are split). `audio_provenance`
        declares how the given audio was produced (default: original)."""
        self.load()
        prov = audio_provenance or self.audio_provenance
        a = prepare_audio(audio, sr)
        decoded = []
        for start, end, chunk in iter_segments(a, TARGET_SR, segments, max_seconds=self.max_window_s):
            ctx = self._prepare(chunk)
            cands = self._candidates(ctx, max(1, n_best))
            scores = self._score_ids(ctx, [ids for ids, _, _ in cands])
            ranked = sorted(zip(cands, scores), key=lambda x: float(np.mean(x[1])), reverse=True)
            hyps = [Alternative(text=text, score=float(np.mean(lps)), origin=origin, audio_variant=audio_variant)
                    for (_, text, origin), lps in ranked]
            (top_ids, top_text, _), top_lps = ranked[0]
            groups = token_word_groups(self._decode, top_ids) if top_text else []
            decoded.append((start, end, chunk, hyps, groups, top_lps[:len(top_ids)]))
            del ctx
        if self.low_memory:
            self.release_model()
        timer = self._get_timer() if any(d[4] for d in decoded) else None
        results = []
        for start, end, chunk, hyps, groups, lps in decoded:
            times = timer.word_times(chunk, TARGET_SR, [w for w, _ in groups]) if timer is not None and groups else None
            words = build_words(groups, lps, times, offset=start, seg_dur=end - start, origin=hyps[0].origin)
            results.append(SegmentHypotheses(start=start, end=end, audio_provenance=prov, hypotheses=hyps,
                                             words=words))
        if timer is not None and self._provenance is not None:
            self._provenance.extra["word_timer"] = timer_provenance(timer)
        if self.low_memory:
            self._release_timer()
        return results


class TransformersDecoderScorer:
    """`CandidateScorer` view of a transformers backend: score = mean per-token
    log-probability of the candidate, teacher-forced (as `WhisperDecoderScorer`)."""

    def __init__(self, backend: TransformersASRBackend) -> None:
        self.backend = backend
        self.name = f"{backend.name}/teacher-forced"

    @property
    def provenance(self) -> ModelProvenance | None:
        return self.backend.provenance

    def score_detailed(self, audio: np.ndarray, sr: int, candidates: list[str]) -> list[TeacherForcedScore]:
        return self.backend.score_texts(audio, sr, candidates)

    def score(self, audio: np.ndarray, sr: int, candidates: list[str]) -> list[float]:
        return [s.mean_logprob for s in self.score_detailed(audio, sr, candidates)]


__all__ = ["BackendUnavailable", "ModelIntegrityError", "TransformersASRBackend", "TransformersDecoderScorer",
           "check_weights", "resolve_snapshot"]
