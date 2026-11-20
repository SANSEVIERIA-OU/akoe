# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Cohere Transcribe 03-2026 backend (`ASRBackend`, issue #54).

Model: `CohereLabs/cohere-transcribe-03-2026`, 2B parameters, Apache-2.0
(model card, read 2026-11-20). Conformer encoder (FastConformer / Parakeet
style, 48 layers) and a light 8-layer Transformer decoder; 14 languages,
Italian included. Native in transformers >= 5.4
(`CohereAsrForConditionalGeneration`, no remote code): this backend never
uses `trust_remote_code`.

How it is run here (see `hf_asr.TransformersASRBackend` for the common flow):
- decoder prompt from the processor (`<|it|>`, punctuation on by default,
  no ITN, no timestamps, no diarization);
- the encoder runs once per segment; beam search, optional sampling and the
  teacher-forced scoring reuse its output;
- N-best from beam search (+ sampling if configured); one score for every
  candidate: mean per-token log-probability, teacher-forced;
- word confidence = product of the word's token probabilities;
- **timestamps: the model has none** (model card: "does not feature"
  timestamps or diarization), so word times come from the Italian CTC
  aligner already in the repository (`ctc_aligner.CTCTranscriptAligner`),
  which records its own model and hashes;
- the feature extractor's dither is seeded by the clip length in
  transformers (deterministic), and recorded in the provenance;
- regions longer than 30 s are split (the processor would chunk above 30 s
  by energy and glue the texts, which would hide the chunk boundaries).

Access: the Hugging Face repository is gated (automatic approval after
accepting the conditions). Without an accepted token the backend raises
`BackendUnavailable` with instructions; a local folder with the same files
can be given as `model_id`. The SHA-256 of `model.safetensors` is checked
against the value published by the hub (LFS metadata, read 2026-11-20).

Memory: 4.1 GB of weights in bfloat16 (the published dtype); float32 would
need 8.3 GB. On a CPU without native bfloat16 (AVX2 only) bfloat16 is
slower than float32, but it is the only option that fits on EDGE.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from akoe.config import StageKind
from akoe.neural import TARGET_SR, BackendUnavailable, ModelProvenance, resolve_device
from akoe.transcribe.hf_asr import TransformersASRBackend
from akoe.transcribe.whisper_backend import SamplingConfig
from akoe.transcribe.word_timing import WordTimer

DEFAULT_MODEL = "CohereLabs/cohere-transcribe-03-2026"
KNOWN_REVISIONS = {DEFAULT_MODEL: "d552dadc2922ba6f6c1f994f8db904d52dcaf942"}  # 2026-11-20
KNOWN_LICENSES = {DEFAULT_MODEL: "apache-2.0"}
KNOWN_WEIGHTS = {DEFAULT_MODEL: {
    "model.safetensors": "987bd3e141c7bfdb5a78f5db11397ee7737308357e6cc0a3f36a4979b158137a",
}}
LANGUAGES = {"ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "vi", "zh"}


@dataclass
class _Ctx:
    enc: Any  # encoder output (last_hidden_state, attention_mask)
    prompt: list[int]


class CohereTranscribeBackend(TransformersASRBackend):
    """Cohere Transcribe with N-best, teacher-forced scores, word confidence
    and CTC-aligned word times."""

    family = "Cohere Transcribe"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        *,
        revision: str | None = None,
        device: str = "cpu",
        dtype: str = "bfloat16",
        language: str = "it",
        punctuation: bool = True,
        num_beams: int = 4,
        sampling: SamplingConfig | None = None,
        word_timestamps: bool = True,
        aligner: WordTimer | Any = None,
        max_new_tokens: int = 256,
        local_files_only: bool = False,
        audio_provenance: StageKind = StageKind.DETERMINISTIC,
        low_memory: bool = False,
        expected_weights: dict[str, str] | None = None,
    ) -> None:
        if language not in LANGUAGES:
            raise ValueError(f"Lingua non supportata da Cohere Transcribe: {language}. "
                             f"Lingue disponibili: {', '.join(sorted(LANGUAGES))}.")
        super().__init__(
            model_id, revision=revision if revision is not None else KNOWN_REVISIONS.get(model_id),
            device=device, dtype=dtype, language=language, num_beams=num_beams, sampling=sampling,
            word_timestamps=word_timestamps, aligner=aligner, max_new_tokens=max_new_tokens,
            local_files_only=local_files_only, audio_provenance=audio_provenance, low_memory=low_memory,
            expected_weights=expected_weights if expected_weights is not None else KNOWN_WEIGHTS.get(model_id),
        )
        self.punctuation = punctuation
        self._processor: Any = None
        self._tok: Any = None

    # ------------------------------------------------------------- loading
    def _load_model(self) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoProcessor, CohereAsrForConditionalGeneration
        except ImportError as e:
            raise BackendUnavailable(
                "Il backend Cohere Transcribe richiede PyTorch e transformers >= 5.4 (classe "
                "CohereAsrForConditionalGeneration): aggiornare gli extra 'ml'."
            ) from e
        device = resolve_device(self.device)
        local, commit = self._resolve(["*.json", "*.safetensors", "tokenizer.model"])
        weights = self._hash_and_check(local, (".safetensors", ".json", ".model"))
        processor = AutoProcessor.from_pretrained(str(local))
        model = CohereAsrForConditionalGeneration.from_pretrained(str(local), dtype=self._torch_dtype())
        self._model = model.to(device).eval()
        self._processor, self._tok = processor, processor.tokenizer
        fe = processor.feature_extractor
        self._provenance = ModelProvenance(
            backend=self.name, model_id=self.model_id, revision=commit, weights=weights, device=device,
            dtype=self.dtype, license=KNOWN_LICENSES.get(self.model_id),
            extra={**self._decoding_params(), "punctuation": self.punctuation,
                   "model_class": type(model).__name__, "trust_remote_code": False,
                   "feature_extractor": {"dither": getattr(fe, "dither", None),
                                         "dither_seed": "clip length (transformers)",
                                         "max_audio_clip_s": getattr(fe, "max_audio_clip_s", None)},
                   "timestamps": "model has none: CTC forced alignment (word_timer)"},
        )

    def _release_model_extra(self) -> None:
        self._processor = self._tok = None

    def _default_timer(self) -> WordTimer | None:
        from akoe.transcribe.ctc_aligner import CTCTranscriptAligner
        from akoe.transcribe.ctc_scorer import CTCForcedAlignmentScorer

        return CTCTranscriptAligner(CTCForcedAlignmentScorer(device=self.device,
                                                             local_files_only=self.local_files_only))

    # ------------------------------------------------------------- model calls
    def _prepare(self, chunk: np.ndarray) -> _Ctx:
        import torch

        if self._model is None:
            self.load()
        dev = self._provenance.device  # type: ignore[union-attr]
        feats = self._processor(chunk, sampling_rate=TARGET_SR, language=self.language,
                                punctuation=self.punctuation, return_tensors="pt")
        x = feats["input_features"].to(dev, dtype=self._model.dtype)
        mask = feats["attention_mask"].to(dev)
        with torch.inference_mode():
            enc = self._model.model.encoder(x, attention_mask=mask)
        return _Ctx(enc=enc, prompt=[int(t) for t in feats["decoder_input_ids"][0]])

    def _eos_ids(self) -> set[int]:
        eos = self._model.generation_config.eos_token_id
        if eos is None:
            eos = self._tok.eos_token_id
        return set(eos) if isinstance(eos, list | tuple) else {int(eos)}

    def _end_id(self) -> int:
        return int(self._tok.eos_token_id if self._tok.eos_token_id is not None else min(self._eos_ids()))

    def _end_token_text(self) -> str:
        return str(self._tok.eos_token or "<eos>")

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
        dec = torch.tensor([ctx.prompt], dtype=torch.long, device=ctx.enc.last_hidden_state.device)
        # generate() expands the encoder output for the beams in place: give it a fresh wrapper
        enc = ctx.enc.__class__(last_hidden_state=ctx.enc.last_hidden_state, attention_mask=ctx.enc.attention_mask)
        with torch.inference_mode():
            out = self._model.generate(encoder_outputs=enc, decoder_input_ids=dec, generation_config=gc_)
        return [self._strip(s[len(ctx.prompt):].tolist()) for s in out.sequences]

    def _score_ids(self, ctx: _Ctx, id_lists: list[list[int]]) -> list[list[float]]:
        import torch

        end = self._end_id()
        pad = self._tok.pad_token_id if self._tok.pad_token_id is not None else end
        seqs = [ctx.prompt + ids + [end] for ids in id_lists]
        n, width = len(seqs), max(len(s) for s in seqs)
        dev = ctx.enc.last_hidden_state.device
        batch = torch.full((n, width), pad, dtype=torch.long, device=dev)
        mask = torch.zeros((n, width), dtype=torch.long, device=dev)
        for i, s in enumerate(seqs):
            batch[i, : len(s)] = torch.tensor(s)
            mask[i, : len(s)] = 1
        enc = ctx.enc
        expanded = enc.__class__(
            last_hidden_state=enc.last_hidden_state.expand(n, -1, -1),
            attention_mask=None if enc.attention_mask is None else enc.attention_mask.expand(n, -1),
        )
        with torch.inference_mode():
            out = self._model(encoder_outputs=expanded, decoder_input_ids=batch, decoder_attention_mask=mask,
                              use_cache=False)
        p0 = len(ctx.prompt)
        result = []
        for b, ids in enumerate(id_lists):
            logp = torch.log_softmax(out.logits[b].float(), dim=-1)
            # the token at position p0+k is predicted by the logits at position p0+k-1
            result.append([float(logp[p0 + k - 1, t]) for k, t in enumerate(ids + [end])])
        return result

    def _text_ids(self, text: str) -> list[int]:
        text = text.strip()
        return list(self._tok.encode(text, add_special_tokens=False)) if text else []

    def _decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True).strip()
