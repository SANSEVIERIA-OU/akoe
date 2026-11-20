"""Whisper backend on Hugging Face transformers (`ASRBackend`), with optional
PEFT/LoRA adapter, N-best, sampling candidates, word timestamps and a
teacher-forced scorer (`CandidateScorer`).

Design notes (MASTER_PLAN §3.6, §4.6; issue #13):
- Every candidate is kept, with its origin: "whisper-tiny/generic" or
  "whisper-tiny/adapted:<id>" for beam-search candidates, the same plus
  "/sampling" for candidates found only by sampling.
- One score for every candidate, whatever produced it: the mean per-token
  log-probability of the candidate text (plus end-of-text) under the decoder,
  teacher-forced on the same audio. The beam's own `sequences_scores` use a
  different length convention and suppression processors, so they are not
  comparable with sampled candidates and are not reported.
- Word timestamps come from cross-attention + DTW on the alignment heads (the
  method of openai/whisper `timing.py`), computed for the top hypothesis only.
  Word confidence is the joint probability of the word's tokens given the
  preceding text (raw, uncalibrated; calibration is done in self-assessment, M5).
- Whisper sees at most 30 s: longer regions are split into ≤30 s chunks, each
  returned as its own segment.

Limitations:
- Teacher-forced scores are sensitive to casing and punctuation (Whisper writes
  cased, punctuated text). Compare candidates written in the same style.
- Raw probabilities, no suppress-token renormalization as in generation.
- If the model has no `alignment_heads` (some fine-tuned checkpoints), word
  times are set to the segment bounds (no fake precision).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from forensicaudio.config import StageKind
from forensicaudio.models import Alternative, SegmentHypotheses, WordHypothesis
from forensicaudio.neural import (
    TARGET_SR,
    BackendUnavailable,
    ModelProvenance,
    hash_files,
    iter_segments,
    prepare_audio,
    resolve_device,
    resolve_hf_snapshot,
)

# Revisions verified on 2026-11-20 (licence: apache-2.0 on the model cards).
KNOWN_REVISIONS = {
    "openai/whisper-tiny": "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
    "openai/whisper-base": "e37978b90ca9030d5170a5c07aadb050351a65bb",
}
MAX_WINDOW_S = 30.0
HOP_LENGTH = 160  # feature frames: 10 ms; encoder frames: 20 ms
SECONDS_PER_ENCODER_FRAME = 2 * HOP_LENGTH / TARGET_SR
MEDIAN_FILTER_WIDTH = 7


@dataclass
class SamplingConfig:
    """Temperature / epsilon sampling to widen the candidate set (beam N-best
    has little diversity, HyPoradise). `seed` is recorded for reproducibility."""

    n_samples: int = 5
    temperature: float = 0.7
    epsilon_cutoff: float | None = 3e-4
    top_p: float | None = None
    seed: int = 0


@dataclass
class TokenScore:
    token: str
    token_id: int
    logprob: float


@dataclass
class TeacherForcedScore:
    """Log-likelihood of a given text under the decoder, forced token by token.
    `tokens` covers the text tokens plus the final end-of-text token."""

    text: str
    total_logprob: float
    tokens: list[TokenScore] = field(default_factory=list)

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    @property
    def mean_logprob(self) -> float:
        return self.total_logprob / max(self.n_tokens, 1)

    @property
    def score(self) -> float:
        """The value used for ranking: `mean_logprob`."""
        return self.mean_logprob


@dataclass
class _WordTiming:
    text: str
    start: float
    end: float
    probability: float


# --------------------------------------------------------------------------
# MBR selection (keeps every candidate; only points at the most central one)
# --------------------------------------------------------------------------


def word_edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, wb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (wa != wb))
        prev = cur
    return prev[-1]


def mbr_select(texts: list[str], logprobs: list[float] | None = None) -> tuple[int, list[float]]:
    """Minimum-Bayes-risk choice under a word-level edit loss.

    Loss = word edit distance / max(len(a), len(b)): symmetric and bounded in
    [0, 1], so a short outlier among the candidates cannot dominate (plain WER,
    normalized by the reference length only, would favour it).
    Returns (index of the chosen text, expected risk of every text). Weights are
    softmax(logprobs) when given, uniform otherwise. All candidates must be kept:
    on conversational speech MBR can be worse than plain decoding (Jinnai 2025).
    """
    if not texts:
        raise ValueError("Nessun candidato per la scelta MBR.")
    if logprobs is None:
        w = np.full(len(texts), 1.0 / len(texts))
    else:
        lp = np.asarray(logprobs, dtype=np.float64)
        w = np.exp(lp - lp.max())
        w /= w.sum()
    toks = [t.lower().split() for t in texts]
    risks = []
    for i, ti in enumerate(toks):
        r = 0.0
        for j, tj in enumerate(toks):
            if i != j:
                r += w[j] * word_edit_distance(ti, tj) / max(len(ti), len(tj), 1)
        risks.append(r)
    return int(np.argmin(risks)), [float(r) for r in risks]


# --------------------------------------------------------------------------
# DTW alignment helpers
# --------------------------------------------------------------------------


def _median_filter(x, width: int):
    import torch

    pad = width // 2
    if x.shape[-1] <= pad:
        return x
    x = torch.nn.functional.pad(x, (pad, pad, 0, 0), mode="reflect")
    return x.unfold(-1, width, 1).sort()[0][..., pad]


def _dtw(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Monotonic alignment minimizing `cost` (rows: tokens, cols: frames)."""
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    trace = np.zeros((n + 1, m + 1), dtype=np.int8)
    acc[0, 0] = 0.0
    for j in range(1, m + 1):
        for i in range(1, n + 1):
            c0, c1, c2 = acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]
            if c0 <= c1 and c0 <= c2:
                acc[i, j], trace[i, j] = cost[i - 1, j - 1] + c0, 0
            elif c1 <= c2:
                acc[i, j], trace[i, j] = cost[i - 1, j - 1] + c1, 1
            else:
                acc[i, j], trace[i, j] = cost[i - 1, j - 1] + c2, 2
    i, j = n, m
    rows, cols = [], []
    while i > 0 and j > 0:
        rows.append(i - 1)
        cols.append(j - 1)
        t = trace[i, j]
        if t == 0:
            i, j = i - 1, j - 1
        elif t == 1:
            i -= 1
        else:
            j -= 1
    return np.array(rows[::-1]), np.array(cols[::-1])


# --------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------


class WhisperHFBackend:
    """Whisper (transformers) with optional LoRA adapter saved by peft
    `save_pretrained` and loaded with `PeftModel.from_pretrained`."""

    def __init__(
        self,
        model_id: str = "openai/whisper-tiny",
        *,
        revision: str | None = None,
        adapter_dir: str | Path | None = None,
        adapter_id: str | None = None,
        device: str = "cpu",
        language: str = "it",
        task: str = "transcribe",
        num_beams: int = 5,
        sampling: SamplingConfig | None = None,
        word_timestamps: bool = True,
        max_new_tokens: int = 200,
        local_files_only: bool = False,
        audio_provenance: StageKind = StageKind.NEURAL,
        allow_base_mismatch: bool = False,
    ) -> None:
        self.model_id = model_id
        self.revision = revision if revision is not None else KNOWN_REVISIONS.get(model_id)
        self.adapter_dir = Path(adapter_dir) if adapter_dir else None
        self.device = device
        self.language = language
        self.task = task
        self.num_beams = num_beams
        self.sampling = sampling
        self.word_timestamps = word_timestamps
        self.max_new_tokens = max_new_tokens
        self.local_files_only = local_files_only
        self.audio_provenance = audio_provenance
        self.allow_base_mismatch = allow_base_mismatch
        self.short_name = model_id.rstrip("/").split("/")[-1]
        if self.adapter_dir is not None:
            self.adapter_id = adapter_id or self._adapter_fingerprint(self.adapter_dir)[:12]
            self.name = f"{self.short_name}/adapted:{self.adapter_id}"
        else:
            self.adapter_id = None
            self.name = f"{self.short_name}/generic"
        self._model: Any = None
        self._peft: Any = None
        self._tok: Any = None
        self._fe: Any = None
        self._provenance: ModelProvenance | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------- loading
    @staticmethod
    def _adapter_files(adapter_dir: Path) -> list[Path]:
        files = sorted(adapter_dir.glob("adapter_model.*")) + sorted(adapter_dir.glob("adapter_config.json"))
        if not any(f.name.startswith("adapter_model.") for f in files):
            raise FileNotFoundError(
                f"Nessun file 'adapter_model.*' nella cartella dell'adattatore {adapter_dir}: "
                "serve una cartella salvata con peft save_pretrained."
            )
        return files

    @classmethod
    def _adapter_fingerprint(cls, adapter_dir: Path) -> str:
        weights = [f for f in cls._adapter_files(adapter_dir) if f.name.startswith("adapter_model.")]
        return hash_files(weights)[weights[0].name]

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
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
        except ImportError as e:  # pragma: no cover
            raise BackendUnavailable(
                "Il backend Whisper richiede PyTorch e transformers: installare gli extra 'ml'."
            ) from e
        device = resolve_device(self.device)
        patterns = ["*.json", "*.txt", "*.safetensors"]
        local, commit = resolve_hf_snapshot(self.model_id, self.revision, patterns, self.local_files_only)
        if not list(local.glob("*.safetensors")):
            local, commit = resolve_hf_snapshot(
                self.model_id, self.revision, patterns + ["pytorch_model*.bin"], self.local_files_only
            )
        processor = WhisperProcessor.from_pretrained(str(local))
        attn = "eager" if self.word_timestamps else "sdpa"
        model = WhisperForConditionalGeneration.from_pretrained(
            str(local), attn_implementation=attn, dtype=torch.float32
        )
        weight_files = [p for p in local.iterdir() if p.is_file() and (
            p.suffix in (".safetensors", ".bin", ".json", ".txt"))]
        adapter_info = None
        if self.adapter_dir is not None:
            model, adapter_info = self._attach_adapter(model)
        self._model = model.to(device).eval()
        self._tok, self._fe = processor.tokenizer, processor.feature_extractor
        self._provenance = ModelProvenance(
            backend=self.name, model_id=self.model_id, revision=commit,
            weights=hash_files(weight_files, root=local), device=device, dtype="float32",
            license="apache-2.0" if self.model_id in KNOWN_REVISIONS else None,
            adapter=adapter_info,
            extra={"language": self.language, "task": self.task, "num_beams": self.num_beams,
                   "sampling": self.sampling.__dict__ if self.sampling else None,
                   "attn_implementation": attn},
        )

    def _attach_adapter(self, model: Any) -> tuple[Any, dict[str, Any]]:
        import json

        try:
            from peft import PeftModel
        except ImportError as e:  # pragma: no cover
            raise BackendUnavailable(
                "Per caricare un adattatore LoRA serve la libreria peft (extra 'ml')."
            ) from e
        assert self.adapter_dir is not None
        files = self._adapter_files(self.adapter_dir)
        cfg = json.loads((self.adapter_dir / "adapter_config.json").read_text())
        base = str(cfg.get("base_model_name_or_path") or "")
        base_short = base.rstrip("/").split("/")[-1]
        if base_short and base_short != self.short_name and not self.allow_base_mismatch:
            raise ValueError(
                f"L'adattatore in {self.adapter_dir} è stato addestrato su '{base}', ma il modello "
                f"di base caricato è '{self.model_id}'. Usare il modello di base corretto."
            )
        peft_model = PeftModel.from_pretrained(model, str(self.adapter_dir), is_trainable=False)
        # LoRA layers are injected in place: use the underlying transformers model
        # for generate/forward, whatever task_type the adapter config declares.
        self._peft = peft_model
        info = {
            "adapter_id": self.adapter_id,
            "dir": str(self.adapter_dir),
            "weights": hash_files(files, root=self.adapter_dir),
            "base_model_name_or_path": base,
            "peft_type": cfg.get("peft_type"),
            "r": cfg.get("r"),
        }
        return peft_model.get_base_model(), info

    # ------------------------------------------------------------- helpers
    def _prompt_ids(self) -> list[int]:
        toks = ["<|startoftranscript|>", f"<|{self.language}|>", f"<|{self.task}|>", "<|notimestamps|>"]
        ids = self._tok.convert_tokens_to_ids(toks)
        if any(i is None or i == self._tok.unk_token_id for i in ids):
            raise ValueError(f"Token speciali Whisper mancanti nel tokenizer: {toks}")
        return list(ids)

    def _text_ids(self, text: str) -> list[int]:
        text = text.strip()
        ids = self._tok.encode(" " + text, add_special_tokens=False) if text else []
        return list(ids) + [self._tok.eos_token_id]

    def _features(self, audio16k: np.ndarray):
        import torch

        if len(audio16k) > MAX_WINDOW_S * TARGET_SR + 1:
            raise ValueError(
                f"Whisper accetta al massimo {MAX_WINDOW_S:.0f} s per volta; il tratto dura "
                f"{len(audio16k) / TARGET_SR:.1f} s. Dividerlo in segmenti più brevi."
            )
        feats = self._fe(audio16k, sampling_rate=TARGET_SR, return_tensors="pt").input_features
        return feats.to(self._provenance.device, dtype=torch.float32)

    def _forced_forward(self, feats, id_lists: list[list[int]], attentions: bool = False):
        """One encoder pass, then the decoder on every (prompt + ids) sequence,
        right-padded (causal masking makes padding irrelevant for real positions)."""
        import torch

        prompt = self._prompt_ids()
        seqs = [prompt + ids for ids in id_lists]
        width = max(len(s) for s in seqs)
        pad = self._tok.pad_token_id if self._tok.pad_token_id is not None else self._tok.eos_token_id
        batch = torch.full((len(seqs), width), pad, dtype=torch.long)
        for i, s in enumerate(seqs):
            batch[i, : len(s)] = torch.tensor(s)
        batch = batch.to(feats.device)
        with torch.inference_mode():
            enc = self._model.get_encoder()(feats).last_hidden_state
            out = self._model(
                encoder_outputs=(enc.expand(len(seqs), -1, -1),),
                decoder_input_ids=batch,
                output_attentions=attentions,
                use_cache=False,
            )
        return prompt, out

    # ------------------------------------------------------------- scoring
    def score_texts(self, audio: np.ndarray, sr: int, texts: list[str]) -> list[TeacherForcedScore]:
        """Teacher-forced log-likelihood of each text (one encoder pass)."""
        import torch

        if self._model is None:
            self.load()
        a = prepare_audio(audio, sr)
        feats = self._features(a)
        id_lists = [self._text_ids(t) for t in texts]
        prompt, out = self._forced_forward(feats, id_lists)
        logp = torch.log_softmax(out.logits.float(), dim=-1)
        results = []
        for b, (text, ids) in enumerate(zip(texts, id_lists)):
            p0 = len(prompt)
            # token at position p0+k is predicted by the logits at position p0+k-1
            lps = [float(logp[b, p0 + k - 1, tid]) for k, tid in enumerate(ids)]
            toks = [TokenScore(token=self._tok.decode([tid]), token_id=int(tid), logprob=lp)
                    for tid, lp in zip(ids, lps)]
            results.append(TeacherForcedScore(text=text.strip(), total_logprob=float(sum(lps)), tokens=toks))
        return results

    def score_text(self, audio: np.ndarray, sr: int, text: str) -> TeacherForcedScore:
        return self.score_texts(audio, sr, [text])[0]

    # ------------------------------------------------------------- words
    def _split_words(self, ids: list[int]) -> list[list[int]]:
        """Group text token ids into words: a token whose text starts with a
        space opens a new word; multi-token UTF-8 characters are kept whole."""
        words: list[list[int]] = []
        pending: list[int] = []
        for tid in ids:
            pending.append(tid)
            piece = self._tok.decode(pending)
            if "\ufffd" in piece:  # incomplete UTF-8 character
                continue
            if not words or piece.startswith(" "):
                words.append(list(pending))
            else:
                words[-1].extend(pending)
            pending = []
        if pending:  # trailing incomplete bytes: attach to the last word
            if words:
                words[-1].extend(pending)
            else:
                words.append(pending)
        return words

    def _word_timings(self, audio16k: np.ndarray, text: str) -> list[_WordTiming]:
        import torch

        text_ids = self._text_ids(text)[:-1]  # without end-of-text
        if not text_ids:
            return []
        feats = self._features(audio16k)
        heads = getattr(self._model.generation_config, "alignment_heads", None)
        prompt, out = self._forced_forward(feats, [text_ids + [self._tok.eos_token_id]],
                                           attentions=bool(heads and self.word_timestamps))
        logp = torch.log_softmax(out.logits[0].float(), dim=-1)
        p0 = len(prompt)
        token_lp = [float(logp[p0 + k - 1, tid]) for k, tid in enumerate(text_ids)]
        words = self._split_words(text_ids)
        seg_dur = len(audio16k) / TARGET_SR
        # token index boundaries of each word
        bounds = np.cumsum([0] + [len(w) for w in words])
        if heads and self.word_timestamps and out.cross_attentions is not None:
            n_frames = max(1, int(np.ceil(len(audio16k) / HOP_LENGTH)) // 2)
            w = torch.stack([out.cross_attentions[layer][0, head] for layer, head in heads])
            # rows predicting text tokens and end-of-text: positions p0-1 .. p0-1+len(text_ids)
            w = w[:, p0 - 1 : p0 + len(text_ids), :n_frames].float()
            std, mean = torch.std_mean(w, dim=-2, keepdim=True, correction=0)
            w = _median_filter((w - mean) / (std + 1e-10), MEDIAN_FILTER_WIDTH).mean(dim=0)
            rows, cols = _dtw(-w.double().cpu().numpy())
            jumps = np.concatenate([[True], np.diff(rows) > 0])
            jump_times = cols[jumps] * SECONDS_PER_ENCODER_FRAME
            starts = jump_times[bounds[:-1]]
            ends = jump_times[bounds[1:]]
        else:
            starts = np.zeros(len(words))
            ends = np.full(len(words), seg_dur)
        out_words = []
        for i, wids in enumerate(words):
            lp = sum(token_lp[bounds[i] : bounds[i + 1]])
            s = float(min(max(starts[i], 0.0), seg_dur))
            e = float(min(max(ends[i], s), seg_dur))
            out_words.append(_WordTiming(self._tok.decode(wids).strip(), s, e, float(np.exp(lp))))
        return out_words

    # ------------------------------------------------------------- decoding
    def _generation_config(self, **overrides: Any):
        """Copy of the model's generation config for the *generic* transformers
        `generate`: the decoder prompt is given explicitly, timestamp tokens are
        suppressed, Whisper's suppress lists are kept.

        Why not `WhisperForConditionalGeneration.generate`: in transformers 5.x
        it repeats the input `num_return_sequences` times and runs independent
        beam searches, so the "N-best" are N copies of the best beam.
        """
        import copy

        gc = copy.deepcopy(self._model.generation_config)
        gc.forced_decoder_ids = None
        gc.max_length = None
        if hasattr(gc, "return_timestamps"):
            gc.return_timestamps = False
        suppress = set(gc.suppress_tokens or [])
        no_ts = getattr(gc, "no_timestamps_token_id", None)
        if no_ts is not None:
            suppress |= set(range(no_ts + 1, self._model.config.vocab_size))
        gc.suppress_tokens = sorted(suppress)
        gc.update(max_new_tokens=self.max_new_tokens, return_dict_in_generate=True, **overrides)
        return gc

    def _decode_sequences(self, seqs, n_prompt: int) -> list[str]:
        return [t.strip() for t in self._tok.batch_decode(seqs[:, n_prompt:], skip_special_tokens=True)]

    def _generate(self, feats, n_best: int) -> list[tuple[str, str]]:
        """Return (text, origin) candidates: beam N-best, then sampling."""
        import torch
        from transformers import GenerationMixin

        prompt = self._prompt_ids()
        dec = torch.tensor([prompt], dtype=torch.long, device=feats.device)
        beams = max(self.num_beams, n_best)
        gc = self._generation_config(num_beams=beams, num_return_sequences=n_best, do_sample=False)
        with torch.inference_mode():
            out = GenerationMixin.generate(
                self._model, input_features=feats, decoder_input_ids=dec, generation_config=gc
            )
        cands = [(t, self.name) for t in self._decode_sequences(out.sequences, len(prompt))]
        s = self.sampling
        if s is not None and s.n_samples > 0:
            extra: dict[str, Any] = {"temperature": s.temperature}
            if s.epsilon_cutoff is not None:
                extra["epsilon_cutoff"] = s.epsilon_cutoff
            if s.top_p is not None:
                extra["top_p"] = s.top_p
            gc = self._generation_config(
                num_beams=1, num_return_sequences=s.n_samples, do_sample=True, **extra
            )
            with torch.random.fork_rng(devices=[]), torch.inference_mode():
                torch.manual_seed(s.seed)
                out = GenerationMixin.generate(
                    self._model, input_features=feats, decoder_input_ids=dec, generation_config=gc
                )
            cands += [(t, f"{self.name}/sampling") for t in self._decode_sequences(out.sequences, len(prompt))]
        # deduplicate on text, keeping the first origin (beam before sampling)
        seen: dict[str, str] = {}
        for text, origin in cands:
            seen.setdefault(text, origin)
        return list(seen.items())

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
        """Hypotheses per segment (VAD regions in seconds, or the whole audio).

        `audio_provenance` declares how the audio given here was produced; it
        defaults to the constructor value (NEURAL unless told otherwise).
        """
        if self._model is None:
            self.load()
        prov = audio_provenance or self.audio_provenance
        a = prepare_audio(audio, sr)
        results: list[SegmentHypotheses] = []
        for start, end, chunk in iter_segments(a, TARGET_SR, segments, max_seconds=MAX_WINDOW_S):
            feats = self._features(chunk)
            cands = self._generate(feats, max(1, n_best))
            scores = self.score_texts(chunk, TARGET_SR, [t for t, _ in cands])
            ranked = sorted(zip(cands, scores), key=lambda x: x[1].mean_logprob, reverse=True)
            hyps = [Alternative(text=t, score=sc.mean_logprob, origin=o, audio_variant=audio_variant)
                    for (t, o), sc in ranked]
            words = []
            if hyps and hyps[0].text:
                for wt in self._word_timings(chunk, hyps[0].text):
                    words.append(WordHypothesis(
                        text=wt.text, start=start + wt.start, end=start + wt.end,
                        confidence=wt.probability, origin=hyps[0].origin,
                    ))
            results.append(SegmentHypotheses(
                start=start, end=end, audio_provenance=prov, hypotheses=hyps, words=words,
            ))
        return results


class WhisperDecoderScorer:
    """`CandidateScorer` view of a Whisper backend: score = mean per-token
    log-probability of the candidate under the decoder (teacher-forced)."""

    def __init__(self, backend: WhisperHFBackend) -> None:
        self.backend = backend
        self.name = f"{backend.name}/teacher-forced"

    @property
    def provenance(self) -> ModelProvenance | None:
        return self.backend.provenance

    def score_detailed(self, audio: np.ndarray, sr: int, candidates: list[str]) -> list[TeacherForcedScore]:
        return self.backend.score_texts(audio, sr, candidates)

    def score(self, audio: np.ndarray, sr: int, candidates: list[str]) -> list[float]:
        return [s.mean_logprob for s in self.score_detailed(audio, sr, candidates)]
