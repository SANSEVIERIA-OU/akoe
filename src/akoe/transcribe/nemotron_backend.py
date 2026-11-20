# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""NVIDIA Nemotron 3.5 Transcribe backend via NeMo Speech (`ASRBackend`,
issue #56). CLOUD tier only.

Model: `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` (October 2026),
licence OpenMDW-1.1 (model card and LICENSE file, read 2026-11-20). A
speech-augmented language model (SALM): 600M-parameter transformer ASR
encoder, Nemotron-3 diarization encoder, CTC timestamp heads and the hybrid
Mamba-Transformer MoE Nemotron 3.5 Lightning 30B-A3B (about 4.3B active
parameters). Best open Italian in November 2026: FLEURS-it 1.41, language
mean 3.61 (model card). One `model.safetensors` of 67.5 GB in BF16; the card
targets one 80 GB H100 (FP8/NVFP4 variants exist for other GPUs).

**Never run on the development machine** (CPU, 14 GB RAM): the backend is
import-guarded and refuses to load, with an Italian message
(`BackendUnavailable`, a NotImplementedError: the engine logs a skipped
stage), when
- NeMo Speech with `nemo.collections.speechlm2` is not installed, or
- there is no CUDA GPU with at least `min_gpu_memory_gb` of memory.

In-process inference follows the NeMo Speech SALM API
(`SALMAutomodel.from_pretrained`, `generate(prompts=..., audios=...,
audio_lens=..., generation_config=...)`, `tokenizer.ids_to_text`). This path
has not been executed (no GPU here): the class to load and the prompt format
must be confirmed on the CLOUD VM against the NeMo Speech version installed
there (see docs/backend_neurali.md). The model card's recommended serving
path is a vLLM server with the NeMo Speech plugin on the user's own VM.

Outputs:
- N-best from beam search (`num_return_sequences`), origin
  "<model>/generic"; scores are not computed (None) because the in-process
  API returns token ids only; the ensemble's other members carry scores;
- word times from the repository's Italian CTC aligner (the checkpoint's own
  CTC timestamp heads are exposed only through the vLLM middleware);
- no per-word confidence (None).
"""
from __future__ import annotations

import gc
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from akoe.config import StageKind
from akoe.models import Alternative, SegmentHypotheses
from akoe.neural import TARGET_SR, BackendUnavailable, ModelProvenance, hash_files, iter_segments, prepare_audio
from akoe.transcribe.hf_asr import check_weights, resolve_snapshot
from akoe.transcribe.word_timing import WordTimer, build_words, resolve_timer, timer_provenance

DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
KNOWN_REVISIONS = {DEFAULT_MODEL: "c17667641642c4265ff2b7037dd101b7d4567634"}  # 2026-11-20
KNOWN_LICENSES = {DEFAULT_MODEL: "openmdw-1.1"}
# Minimum GPU memory: 67.5 GB of BF16 weights plus activations (card: one 80 GB H100).
MIN_GPU_MEMORY_GB = {DEFAULT_MODEL: 78.0}
# Verbatim transcription prompt of the model card, with the language fixed to Italian.
PROMPT_IT = ("Audio transcription. Language: Italian. Wording: verbatim, including fillers and repetitions. "
             "Formatting: punctuation and capitalization. Output: transcript only.")
MAX_WINDOW_S = 30.0

NEMO_SPEECH_MISSING = (
    "Il backend Nemotron 3.5 Transcribe richiede NeMo Speech (modulo nemo.collections.speechlm2), che non è "
    "installato in questo ambiente. È previsto solo per il livello CLOUD, su una VM con GPU controllata "
    "dall'utente: installare NeMo Speech dal ramo principale (git clone https://github.com/NVIDIA-NeMo/Speech; "
    "uv sync --extra asr) in un ambiente Python 3.13 separato, insieme a questo pacchetto. Istruzioni complete "
    "in docs/backend_neurali.md."
)


def nemo_speech_available() -> bool:
    try:
        import nemo.collections.speechlm2  # noqa: F401
    except Exception:  # noqa: BLE001 - NeMo can fail at import with non-ImportError errors
        return False
    return True


def gpu_memory_gb() -> list[float]:
    """Total memory of each visible CUDA GPU in GB ([] without CUDA)."""
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.get_device_properties(i).total_memory / 1e9 for i in range(torch.cuda.device_count())]


class NemotronTranscribeBackend:
    """Nemotron 3.5 Transcribe (NeMo Speech SALM) for the CLOUD tier."""

    family = "Nemotron 3.5 Transcribe"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        *,
        revision: str | None = None,
        device: str = "cuda",
        min_gpu_memory_gb: float | None = None,
        prompt: str = PROMPT_IT,
        num_beams: int = 4,
        max_new_tokens: int = 512,
        word_timestamps: bool = True,
        aligner: WordTimer | Callable[[], WordTimer] | None = None,
        local_files_only: bool = False,
        audio_provenance: StageKind = StageKind.DETERMINISTIC,
        expected_weights: dict[str, str] | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision if revision is not None else KNOWN_REVISIONS.get(model_id)
        self.device = device
        self.min_gpu_memory_gb = (min_gpu_memory_gb if min_gpu_memory_gb is not None
                                  else MIN_GPU_MEMORY_GB.get(model_id, 78.0))
        self.prompt = prompt
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.word_timestamps = word_timestamps
        self.local_files_only = local_files_only
        self.audio_provenance = audio_provenance
        self.expected_weights = expected_weights
        self.short_name = model_id.rstrip("/").split("/")[-1]
        self.name = f"{self.short_name}/generic"
        self._aligner_arg = aligner
        self._timer: WordTimer | None = None
        self._model: Any = None
        self._provenance: ModelProvenance | None = None
        self._lock = threading.Lock()

    @property
    def provenance(self) -> ModelProvenance | None:
        return self._provenance

    # ------------------------------------------------------------- availability
    def unavailable_reason(self) -> str | None:
        """Italian reason why the model cannot run here, or None."""
        if not nemo_speech_available():
            return NEMO_SPEECH_MISSING
        if not self.device.startswith("cuda"):
            return (f"Nemotron 3.5 Transcribe non è eseguibile su '{self.device}': serve una GPU NVIDIA con "
                    f"almeno {self.min_gpu_memory_gb:.0f} GB di memoria (livello CLOUD).")
        mem = gpu_memory_gb()
        if not mem:
            return ("Nemotron 3.5 Transcribe richiede una GPU NVIDIA con CUDA, non disponibile su questa macchina "
                    f"(servono almeno {self.min_gpu_memory_gb:.0f} GB di memoria della GPU; livello CLOUD).")
        idx = int(self.device.split(":")[1]) if ":" in self.device else 0
        if idx >= len(mem) or mem[idx] < self.min_gpu_memory_gb:
            found = f"{mem[idx]:.0f} GB" if idx < len(mem) else "nessuna GPU con quell'indice"
            return (f"Memoria della GPU insufficiente per Nemotron 3.5 Transcribe in BF16: servono almeno "
                    f"{self.min_gpu_memory_gb:.0f} GB, trovati {found}. Usare una GPU da 80 GB (H100, A100 80 GB) "
                    "oppure la variante FP8 su una GPU che la supporta.")
        return None

    def load(self) -> ModelProvenance:
        with self._lock:
            if self._model is None:
                self._load()
        assert self._provenance is not None
        return self._provenance

    def _load(self) -> None:
        reason = self.unavailable_reason()
        if reason is not None:
            raise BackendUnavailable(reason)
        import nemo  # type: ignore[import-not-found]
        from nemo.collections.speechlm2.models import SALMAutomodel  # type: ignore[import-not-found]

        local, commit = resolve_snapshot(self.model_id, self.revision, None, self.local_files_only, self.family)
        files = [p for p in local.rglob("*") if p.is_file() and p.suffix in (".safetensors", ".json")]
        weights = hash_files(files, root=local)
        check_weights(weights, self.expected_weights, self.model_id)
        model = SALMAutomodel.from_pretrained(str(local))
        self._model = model.to(self.device).eval()
        self._provenance = ModelProvenance(
            backend=self.name, model_id=self.model_id, revision=commit, weights=weights, device=self.device,
            dtype="bfloat16", license=KNOWN_LICENSES.get(self.model_id),
            extra={"nemo_version": getattr(nemo, "__version__", None), "model_class": type(model).__name__,
                   "prompt": self.prompt, "num_beams": self.num_beams, "max_new_tokens": self.max_new_tokens,
                   "enable_thinking": False, "scores": None, "word_confidence": None},
        )

    def unload(self) -> None:
        with self._lock:
            self._model = None
        gc.collect()

    def _get_timer(self) -> WordTimer | None:
        if not self.word_timestamps:
            return None
        if self._timer is None:
            if self._aligner_arg is not None:
                self._timer = resolve_timer(self._aligner_arg)
            else:
                from akoe.transcribe.ctc_aligner import CTCTranscriptAligner

                self._timer = CTCTranscriptAligner()
        return self._timer

    # ------------------------------------------------------------- decoding
    def _generate(self, chunk: np.ndarray, n_best: int) -> list[str]:
        import torch
        from transformers import GenerationConfig

        m = self._model
        audios = torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32))[None].to(self.device)
        lens = torch.tensor([audios.shape[1]], dtype=torch.long, device=self.device)
        beams = max(self.num_beams, n_best)
        gen = GenerationConfig(num_beams=beams, num_return_sequences=min(n_best, beams), do_sample=False,
                               max_new_tokens=self.max_new_tokens, bos_token_id=m.text_bos_id,
                               eos_token_id=m.text_eos_id, pad_token_id=m.text_pad_id)
        prompts = [[{"role": "user", "content": f"{self.prompt} {m.audio_locator_tag}"}]]
        with torch.inference_mode():
            ids = m.generate(prompts=prompts, audios=audios, audio_lens=lens, generation_config=gen,
                             enable_thinking=False)
        texts = []
        for row in ids:
            row = [int(t) for t in row if int(t) not in (m.text_pad_id, m.text_eos_id)]
            texts.append(m.tokenizer.ids_to_text(row).strip())
        return texts

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
        self.load()
        prov = audio_provenance or self.audio_provenance
        a = prepare_audio(audio, sr)
        timer = self._get_timer()
        results = []
        for start, end, chunk in iter_segments(a, TARGET_SR, segments, max_seconds=MAX_WINDOW_S):
            texts = list(dict.fromkeys(self._generate(chunk, max(1, n_best))))
            hyps = [Alternative(text=t, score=None, origin=self.name, audio_variant=audio_variant) for t in texts]
            words_text = hyps[0].text.split() if hyps and hyps[0].text else []
            groups = [(w, []) for w in words_text]  # no token log-probabilities: confidence None
            times = timer.word_times(chunk, TARGET_SR, words_text) if timer is not None and words_text else None
            words = build_words(groups, [], times, offset=start, seg_dur=end - start, origin=self.name)
            results.append(SegmentHypotheses(start=start, end=end, audio_provenance=prov, hypotheses=hyps,
                                             words=words))
        if timer is not None and self._provenance is not None:
            self._provenance.extra["word_timer"] = timer_provenance(timer)
        return results


__all__ = ["DEFAULT_MODEL", "NemotronTranscribeBackend", "gpu_memory_gb", "nemo_speech_available"]
