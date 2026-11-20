# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Qwen3-ASR-1.7B backend with Qwen3-ForcedAligner word times (`ASRBackend`,
issue #55).

Models (Apache-2.0 on both model cards, read 2026-11-20), in the official
transformers format published by Qwen (the "-hf" repositories; the original
repositories target the `qwen-asr` package, not needed here):
- `Qwen/Qwen3-ASR-1.7B-hf`: audio encoder + Qwen3 1.7B language model,
  30 languages, Italian included; FLEURS-it 2.41 (MASTER_PLAN §3.11);
- `Qwen/Qwen3-ForcedAligner-0.6B-hf`: non-autoregressive timestamp
  prediction (80 ms classes) for 11 languages, Italian included.
Both are native in transformers 5.23 (`Qwen3ASRForConditionalGeneration`,
`Qwen3ASRForTokenClassification`); no extra package.

How it is run (common flow in `hf_asr.TransformersASRBackend`):
- the language is forced as in the reference implementation: the assistant
  turn is prefilled with "language Italian<asr_text>", so the model writes
  only the transcription (no language detection, no switch to another
  language on unclear audio);
- the audio encoder runs once per segment: the prompt embeddings (text +
  audio) are built once and reused by beam search, sampling and the
  teacher-forced scoring;
- N-best from beam search (+ sampling if configured), teacher-forced mean
  log-probability for every candidate, word confidence from the top
  hypothesis' token probabilities;
- the output is not post-edited (the reference package's repetition fixer
  is not applied: repetitions are left for the hallucination guard);
- word times: `Qwen3ForcedAligner` on the top hypothesis. Punctuation-only
  words get the gap between their neighbours. With `low_memory`, the ASR
  weights (3.9 GB bf16) are released before the aligner (1.7 GB) loads, so
  the peak stays near the larger of the two.
"""
from __future__ import annotations

import gc
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from akoe.config import StageKind
from akoe.neural import BackendUnavailable, ModelProvenance, prepare_audio, resolve_device
from akoe.transcribe.hf_asr import TransformersASRBackend, check_weights, resolve_snapshot
from akoe.transcribe.whisper_backend import SamplingConfig
from akoe.transcribe.word_timing import Times, WordTimer

DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B-hf"
DEFAULT_ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
KNOWN_REVISIONS = {  # 2026-11-20
    DEFAULT_MODEL: "bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
    DEFAULT_ALIGNER: "c07281df297b9905d24a508279258cccf987a064",
    "Qwen/Qwen3-ASR-0.6B-hf": "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c",
}
KNOWN_LICENSES = {DEFAULT_MODEL: "apache-2.0", DEFAULT_ALIGNER: "apache-2.0", "Qwen/Qwen3-ASR-0.6B-hf": "apache-2.0"}
KNOWN_WEIGHTS = {  # SHA-256 published by the hub (LFS metadata)
    DEFAULT_MODEL: {"model.safetensors": "2db53c7d81bd9b8cbc6a074e89be2c968a0d373fb4ee68bb1b1e14f7042dfee1"},
    DEFAULT_ALIGNER: {"model.safetensors": "00568245ceca5af1991d28562a75fe1ddc9bfeb041c27fda66947ea05c47fb86"},
    "Qwen/Qwen3-ASR-0.6B-hf": {"model.safetensors": "d3f212dd20abecd315d830bc54ae3865e56ebfc3276484e57b771288ba27fd35"},
}
PATTERNS = ["*.json", "*.jinja", "*.safetensors"]
_MISSING = ("Il backend Qwen3-ASR richiede PyTorch e transformers >= 5.23 (classi Qwen3ASR*): "
            "aggiornare gli extra 'ml'.")


@dataclass
class _Ctx:
    prompt: list[int]
    embeds: Any  # (1, P, H) prompt embeddings with the audio inserted


class Qwen3ForcedAligner:
    """`WordTimer` on Qwen3-ForcedAligner (lazy; `unload()` frees it)."""

    def __init__(
        self,
        model_id: str = DEFAULT_ALIGNER,
        *,
        revision: str | None = None,
        device: str = "cpu",
        dtype: str = "bfloat16",
        language: str = "it",
        local_files_only: bool = False,
        expected_weights: dict[str, str] | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision if revision is not None else KNOWN_REVISIONS.get(model_id)
        self.device = device
        self.dtype = dtype
        self.language = language
        self.local_files_only = local_files_only
        self.expected_weights = expected_weights if expected_weights is not None else KNOWN_WEIGHTS.get(model_id)
        self.name = f"{model_id.rstrip('/').split('/')[-1]}/forced-aligner"
        self.last_report: dict[str, Any] = {}
        self._model: Any = None
        self._processor: Any = None
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

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Qwen3ASRForTokenClassification
        except ImportError as e:
            raise BackendUnavailable(_MISSING) from e
        from akoe.neural import hash_files

        device = resolve_device(self.device)
        local, commit = resolve_snapshot(self.model_id, self.revision, PATTERNS, self.local_files_only,
                                         "Qwen3-ForcedAligner")
        files = [p for p in local.iterdir() if p.is_file() and p.suffix in (".safetensors", ".json", ".jinja")]
        weights = hash_files(files, root=local)
        check_weights(weights, self.expected_weights, self.model_id)
        self._processor = AutoProcessor.from_pretrained(str(local))
        model = Qwen3ASRForTokenClassification.from_pretrained(str(local), dtype=getattr(torch, self.dtype))
        self._model = model.to(device).eval()
        self._provenance = ModelProvenance(
            backend=self.name, model_id=self.model_id, revision=commit, weights=weights, device=device,
            dtype=self.dtype, license=KNOWN_LICENSES.get(self.model_id),
            extra={"language": self.language, "timestamp_segment_time_ms": self._processor.timestamp_segment_time},
        )

    def unload(self) -> None:
        with self._lock:
            self._model = self._processor = None
        gc.collect()

    def word_times(self, audio: np.ndarray, sr: int, words: list[str]) -> Times:
        import torch

        self.load()
        self.last_report = {"aligner": self.name, "warnings": []}
        a16 = prepare_audio(audio, sr)
        parts = [self._processor.split_words_for_alignment(w, self.language) for w in words]
        flat = [p for ps in parts for p in ps]
        if not flat:
            return [None] * len(words)
        inputs, word_lists = self._processor.prepare_forced_aligner_inputs(
            audio=[a16], transcript=[" ".join(flat)], language=self.language)
        if word_lists[0] != flat:
            self.last_report["warnings"].append(
                "Suddivisione in parole dell'allineatore diversa da quella della trascrizione: tempi non assegnati.")
            return [None] * len(words)
        dev = self._provenance.device  # type: ignore[union-attr]
        model_inputs = {k: v.to(dev) for k, v in inputs.items() if hasattr(v, "to") and k != "num_audio_tokens"}
        model_inputs["input_features"] = model_inputs["input_features"].to(self._model.dtype)
        with torch.inference_mode():
            logits = self._model(**model_inputs).logits
        items = self._processor.decode_forced_alignment(
            logits.float(), model_inputs["input_ids"], word_lists,
            timestamp_token_id=self._model.config.timestamp_token_id)[0]
        out: Times = []
        k = 0
        for ps in parts:
            if not ps:
                out.append(None)
                continue
            out.append((float(items[k]["start_time"]), float(items[k + len(ps) - 1]["end_time"])))
            k += len(ps)
        return out


class Qwen3ASRBackend(TransformersASRBackend):
    """Qwen3-ASR with N-best, teacher-forced scores, word confidence and
    Qwen3-ForcedAligner word times."""

    family = "Qwen3-ASR"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        *,
        revision: str | None = None,
        device: str = "cpu",
        dtype: str = "bfloat16",
        language: str = "it",
        num_beams: int = 4,
        sampling: SamplingConfig | None = None,
        word_timestamps: bool = True,
        aligner: WordTimer | Any = None,
        aligner_model: str = DEFAULT_ALIGNER,
        max_new_tokens: int = 256,
        local_files_only: bool = False,
        audio_provenance: StageKind = StageKind.DETERMINISTIC,
        low_memory: bool = False,
        expected_weights: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            model_id, revision=revision if revision is not None else KNOWN_REVISIONS.get(model_id),
            device=device, dtype=dtype, language=language, num_beams=num_beams, sampling=sampling,
            word_timestamps=word_timestamps, aligner=aligner, max_new_tokens=max_new_tokens,
            local_files_only=local_files_only, audio_provenance=audio_provenance, low_memory=low_memory,
            expected_weights=expected_weights if expected_weights is not None else KNOWN_WEIGHTS.get(model_id),
        )
        self.aligner_model = aligner_model
        self._processor: Any = None
        self._tok: Any = None

    # ------------------------------------------------------------- loading
    def _load_model(self) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
        except ImportError as e:
            raise BackendUnavailable(_MISSING) from e
        device = resolve_device(self.device)
        local, commit = self._resolve(PATTERNS)
        weights = self._hash_and_check(local, (".safetensors", ".json", ".jinja"))
        processor = AutoProcessor.from_pretrained(str(local))
        model = Qwen3ASRForConditionalGeneration.from_pretrained(str(local), dtype=self._torch_dtype())
        self._model = model.to(device).eval()
        self._processor, self._tok = processor, processor.tokenizer
        self._provenance = ModelProvenance(
            backend=self.name, model_id=self.model_id, revision=commit, weights=weights, device=device,
            dtype=self.dtype, license=KNOWN_LICENSES.get(self.model_id),
            extra={**self._decoding_params(), "model_class": type(model).__name__,
                   "language_forcing": "assistant prefill 'language <name><asr_text>'",
                   "post_editing": None},
        )

    def _release_model_extra(self) -> None:
        self._processor = self._tok = None

    def _default_timer(self) -> WordTimer | None:
        return Qwen3ForcedAligner(self.aligner_model, device=self.device, dtype=self.dtype, language=self.language,
                                  local_files_only=self.local_files_only)

    # ------------------------------------------------------------- model calls
    def _prepare(self, chunk: np.ndarray) -> _Ctx:
        import torch

        if self._model is None:
            self.load()
        dev = self._provenance.device  # type: ignore[union-attr]
        inputs = self._processor.apply_transcription_request(audio=[chunk], language=self.language)
        ids = inputs["input_ids"].to(dev)
        feats = inputs["input_features"].to(dev, dtype=self._model.dtype)
        fmask = inputs["input_features_mask"].to(dev)
        m = self._model.model
        with torch.inference_mode():
            emb = m.get_input_embeddings()(ids)
            audio = m.get_audio_features(feats, fmask, return_dict=True).pooler_output
            mask = m.get_placeholder_mask(ids, inputs_embeds=emb, audio_features=audio)
            emb = emb.masked_scatter(mask, audio.to(emb.dtype))
        return _Ctx(prompt=[int(t) for t in ids[0]], embeds=emb)

    def _eos_ids(self) -> set[int]:
        eos = self._model.generation_config.eos_token_id
        if eos is None:
            eos = self._model.config.eos_token_id
        return set(eos) if isinstance(eos, list | tuple) else {int(eos)}

    def _end_id(self) -> int:
        im_end = self._tok.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end != self._tok.unk_token_id and im_end in self._eos_ids():
            return im_end
        return min(self._eos_ids())

    def _end_token_text(self) -> str:
        return str(self._tok.convert_ids_to_tokens(self._end_id()))

    def _strip(self, seq: list[int]) -> list[int]:
        eos, pad = self._eos_ids(), self._tok.pad_token_id
        out = []
        for t in seq:
            if t in eos:
                break
            if t != pad:
                out.append(int(t))
        return out

    def _generate_ids(self, ctx: _Ctx, overrides: dict[str, Any]) -> list[list[int]]:
        import torch

        gc_ = self._generation_config(self._model.generation_config, **overrides)
        mask = torch.ones(ctx.embeds.shape[:2], dtype=torch.long, device=ctx.embeds.device)
        with torch.inference_mode():
            out = self._model.generate(inputs_embeds=ctx.embeds, attention_mask=mask, generation_config=gc_)
        p = len(ctx.prompt)
        seqs = []
        for s in out.sequences.tolist():
            if len(s) >= p and s[:p] == ctx.prompt:  # some versions return the prompt ids too
                s = s[p:]
            seqs.append(self._strip(s))
        return seqs

    def _score_ids(self, ctx: _Ctx, id_lists: list[list[int]]) -> list[list[float]]:
        import torch

        end = self._end_id()
        pad = self._tok.pad_token_id if self._tok.pad_token_id is not None else end
        seqs = [ids + [end] for ids in id_lists]
        n, width = len(seqs), max(len(s) for s in seqs)
        dev = ctx.embeds.device
        toks = torch.full((n, width), pad, dtype=torch.long, device=dev)
        tmask = torch.zeros((n, width), dtype=torch.long, device=dev)
        for i, s in enumerate(seqs):
            toks[i, : len(s)] = torch.tensor(s)
            tmask[i, : len(s)] = 1
        p = ctx.embeds.shape[1]
        with torch.inference_mode():
            emb = torch.cat([ctx.embeds.expand(n, -1, -1), self._model.get_input_embeddings()(toks)], dim=1)
            mask = torch.cat([torch.ones((n, p), dtype=torch.long, device=dev), tmask], dim=1)
            # keep only the logits predicting the text: positions p-1 .. p+width-2
            out = self._model(inputs_embeds=emb, attention_mask=mask, logits_to_keep=width + 1, use_cache=False)
        result = []
        for b, s in enumerate(seqs):
            logp = torch.log_softmax(out.logits[b].float(), dim=-1)
            result.append([float(logp[k, t]) for k, t in enumerate(s)])
        return result

    def _text_ids(self, text: str) -> list[int]:
        text = text.strip()
        return list(self._tok.encode(text, add_special_tokens=False)) if text else []

    def _decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True).strip()


__all__ = ["DEFAULT_ALIGNER", "DEFAULT_MODEL", "Qwen3ASRBackend", "Qwen3ForcedAligner"]

