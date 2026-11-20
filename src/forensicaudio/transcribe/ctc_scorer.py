"""CTC forced-alignment scorer for "did he say X or Y?" (`CandidateScorer`).

Each operator-typed candidate is force-aligned to the emissions of an Italian
CTC model (wav2vec2 / XLS-R) with a Viterbi pass over the CTC trellis,
implemented here in numpy. Why not torchaudio: `torchaudio.functional.forced_align`
was announced as deprecated in 2.8 and slated for removal in 2.9; torchaudio
2.11.1 (installed here) still ships it and it works, but depending on it would
pin the project to an API marked for removal. The tests use it, when present,
only as a cross-check of this implementation.

Scores (all length-normalized, higher = more compatible):
- `per_frame`: Viterbi log-probability of the candidate / number of frames;
- `relative_per_frame` (the `score()` value): the same minus the per-frame
  log-probability of the unconstrained best path, so 0 means "exactly what the
  model would decode freely" and the number is less sensitive to how noisy the
  audio is overall. Always <= 0;
- `per_token`: Viterbi log-probability / number of target tokens.
An impossible alignment (more characters than frames) scores -inf.

Caution (MASTER_PLAN §3.6.4, §3.8): on poor audio an alignment can "appear to
work" on a wrong text (Loakes 2022). Thresholds for "none compatible" must be
calibrated per case in self-assessment; this module never decides alone.
"""
from __future__ import annotations

import itertools
import json
import threading
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from forensicaudio.neural import (
    TARGET_SR,
    BackendUnavailable,
    ModelProvenance,
    hash_files,
    prepare_audio,
    resolve_device,
    resolve_hf_snapshot,
)

# Default Italian CTC model: XLS-R 53 fine-tuned on Common Voice 6.1 Italian.
# Licence apache-2.0 on the model card, verified 2026-11-20 at this revision.
DEFAULT_CTC_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-italian"
KNOWN_REVISIONS = {
    DEFAULT_CTC_MODEL: "dab04a3e00d8326052f3fb22a6ff276b822f6131",
    # Small model for tests only: licence cc-by-nc-4.0 (non-commercial).
    "facebook/wav2vec2-base-10k-voxpopuli-ft-it": "653b3c789c7d83134924bda6fabe0c6bb84f579c",
}
KNOWN_LICENSES = {
    DEFAULT_CTC_MODEL: "apache-2.0",
    "facebook/wav2vec2-base-10k-voxpopuli-ft-it": "cc-by-nc-4.0",
}
MAX_SECONDS = 60.0
_APOSTROPHES = {"’": "'", "‘": "'", "`": "'", "´": "'", "ʼ": "'"}


# --------------------------------------------------------------------------
# Viterbi forced alignment on the CTC trellis
# --------------------------------------------------------------------------


@dataclass
class CTCPath:
    """Best CTC path for a target sequence. `states[t]` is the index in the
    blank-extended sequence (even = blank, odd = target index (s-1)//2)."""

    logprob: float
    states: np.ndarray  # (T,) int
    frame_logprobs: np.ndarray  # (T,) log-prob of the emitted label at each frame

    def token_spans(self, n_targets: int) -> list[tuple[int, int]]:
        """[start, end) frame span of each target token."""
        spans = [(-1, -1)] * n_targets
        for t, s in enumerate(self.states):
            if s % 2 == 1:
                k = (s - 1) // 2
                a, _ = spans[k]
                spans[k] = (t if a < 0 else a, t + 1)
        return spans


def ctc_min_frames(targets: list[int]) -> int:
    """Minimum number of frames for a CTC alignment: one per token plus one
    blank between equal consecutive tokens."""
    return len(targets) + sum(1 for a, b in itertools.pairwise(targets) if a == b)


def ctc_viterbi(log_probs: np.ndarray, targets: list[int], blank: int = 0) -> CTCPath:
    """Most probable CTC alignment of `targets` to `log_probs` (T x C, natural log).

    Returns a path with logprob -inf when no alignment exists.
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    n_frames = lp.shape[0]
    if blank in targets:
        raise ValueError("La sequenza di destinazione non può contenere il simbolo blank.")
    n_states = 2 * len(targets) + 1
    ext = np.full(n_states, blank, dtype=np.int64)
    ext[1::2] = targets
    if n_frames == 0 or n_frames < ctc_min_frames(targets):
        return CTCPath(float("-inf"), np.zeros(n_frames, dtype=np.int64), np.full(n_frames, -np.inf))
    can_skip = np.zeros(n_states, dtype=bool)
    if n_states > 2:
        can_skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
    neg = -np.inf
    alpha = np.full(n_states, neg)
    alpha[0] = lp[0, ext[0]]
    if n_states > 1:
        alpha[1] = lp[0, ext[1]]
    back = np.zeros((n_frames, n_states), dtype=np.int8)  # 0 stay, 1 from s-1, 2 from s-2
    idx = np.arange(n_states)
    for t in range(1, n_frames):
        prev1 = np.concatenate(([neg], alpha[:-1]))
        prev2 = np.concatenate(([neg, neg], alpha[:-2]))
        prev2[~can_skip] = neg
        cand = np.stack([alpha, prev1, prev2])
        best = cand.argmax(axis=0)
        alpha = cand[best, idx] + lp[t, ext]
        back[t] = best
    finals = [n_states - 1] + ([n_states - 2] if n_states > 1 else [])
    s = max(finals, key=lambda k: alpha[k])
    score = float(alpha[s])
    states = np.zeros(n_frames, dtype=np.int64)
    for t in range(n_frames - 1, -1, -1):
        states[t] = s
        s -= int(back[t, s])
    frame_lp = lp[np.arange(n_frames), ext[states]]
    return CTCPath(score, states, frame_lp)


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass
class WordAlignment:
    word: str
    start: float  # seconds, relative to the scored audio
    end: float
    mean_logprob: float  # over the frames emitting this word's characters
    n_frames: int


@dataclass
class CTCCandidateScore:
    text: str
    normalized: str  # words separated by spaces, as aligned
    feasible: bool
    viterbi_logprob: float
    free_logprob: float
    n_frames: int
    n_tokens: int
    words: list[WordAlignment] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def per_frame(self) -> float:
        return self.viterbi_logprob / max(self.n_frames, 1)

    @property
    def per_token(self) -> float:
        return self.viterbi_logprob / max(self.n_tokens, 1)

    @property
    def relative_per_frame(self) -> float:
        return (self.viterbi_logprob - self.free_logprob) / max(self.n_frames, 1)

    @property
    def score(self) -> float:
        return self.relative_per_frame


# --------------------------------------------------------------------------
# Scorer
# --------------------------------------------------------------------------


class CTCForcedAlignmentScorer:
    """Scores candidate phrases with an Italian CTC model (lazy-loaded)."""

    def __init__(
        self,
        model_id: str = DEFAULT_CTC_MODEL,
        *,
        revision: str | None = None,
        device: str = "cpu",
        local_files_only: bool = False,
    ) -> None:
        self.model_id = model_id
        self.revision = revision if revision is not None else KNOWN_REVISIONS.get(model_id)
        self.device = device
        self.local_files_only = local_files_only
        self.name = f"{model_id.rstrip('/').split('/')[-1]}/ctc-forced-alignment"
        self._emit: Callable[[np.ndarray], np.ndarray] | None = None
        self._vocab: dict[str, int] = {}
        self._blank = 0
        self._delim = "|"
        self._provenance: ModelProvenance | None = None
        self._lock = threading.Lock()

    @classmethod
    def from_emission_fn(
        cls,
        emission_fn: Callable[[np.ndarray], np.ndarray],
        vocab: dict[str, int],
        *,
        blank_id: int = 0,
        word_delimiter: str = "|",
        name: str = "custom-ctc",
    ) -> CTCForcedAlignmentScorer:
        """Scorer over any function mapping 16 kHz audio to (T x C) log-probs."""
        self = cls(name)
        self.name = f"{name}/ctc-forced-alignment"
        self._emit, self._vocab, self._blank, self._delim = emission_fn, dict(vocab), blank_id, word_delimiter
        self._provenance = ModelProvenance(backend=self.name, model_id=name, device="cpu",
                                           extra={"note": "funzione di emissione fornita dal chiamante"})
        return self

    @property
    def provenance(self) -> ModelProvenance | None:
        return self._provenance

    def load(self) -> ModelProvenance:
        with self._lock:
            if self._emit is None:
                self._load()
        assert self._provenance is not None
        return self._provenance

    def _load(self) -> None:
        try:
            import torch
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
        except ImportError as e:  # pragma: no cover
            raise BackendUnavailable(
                "Il punteggio CTC richiede PyTorch e transformers: installare gli extra 'ml'."
            ) from e
        device = resolve_device(self.device)
        patterns = ["*.json", "*.safetensors"]
        local, commit = resolve_hf_snapshot(self.model_id, self.revision, patterns, self.local_files_only)
        if not list(local.glob("*.safetensors")):
            local, commit = resolve_hf_snapshot(
                self.model_id, self.revision, patterns + ["pytorch_model.bin"], self.local_files_only
            )
        fe = Wav2Vec2FeatureExtractor.from_pretrained(str(local))
        model = Wav2Vec2ForCTC.from_pretrained(str(local), dtype=torch.float32).to(device).eval()
        vocab = json.loads((local / "vocab.json").read_text(encoding="utf-8"))
        cfg = json.loads((local / "config.json").read_text(encoding="utf-8"))
        tok_cfg_path = local / "tokenizer_config.json"
        tok_cfg = json.loads(tok_cfg_path.read_text()) if tok_cfg_path.exists() else {}
        self._vocab = vocab
        # The CTC blank is the tokenizer's pad token. Take it from the vocabulary:
        # some checkpoints have a config.pad_token_id that disagrees with vocab.json
        # (facebook/wav2vec2-base-10k-voxpopuli-ft-it: config 1, "<pad>" = 0).
        pad = tok_cfg.get("pad_token", "<pad>")
        pad = pad.get("content", "<pad>") if isinstance(pad, dict) else pad
        self._blank = int(vocab[pad]) if pad in vocab else int(cfg.get("pad_token_id", 0))
        self._delim = tok_cfg.get("word_delimiter_token", "|")

        def emit(a16: np.ndarray) -> np.ndarray:
            x = fe(a16, sampling_rate=TARGET_SR, return_tensors="pt").input_values.to(device)
            with torch.inference_mode():
                logits = model(x).logits[0].float()
            return torch.log_softmax(logits, dim=-1).cpu().numpy()

        self._emit = emit
        weights = [p for p in local.iterdir() if p.is_file() and p.suffix in (".safetensors", ".bin", ".json")]
        self._provenance = ModelProvenance(
            backend=self.name, model_id=self.model_id, revision=commit,
            weights=hash_files(weights, root=local), device=device, dtype="float32",
            license=KNOWN_LICENSES.get(self.model_id),
            extra={"blank_id": self._blank, "config_pad_token_id": cfg.get("pad_token_id"),
                   "word_delimiter": self._delim, "vocab_size": len(vocab)},
        )

    # ------------------------------------------------------------- text
    def normalize_text(self, text: str) -> tuple[list[str], list[str]]:
        """Map a candidate to words over the model's alphabet.

        Lowercase, NFC; apostrophes and hyphens kept if the alphabet has them,
        otherwise treated as word boundaries; punctuation is a boundary;
        accented letters missing from the alphabet fall back to the base letter;
        anything else (digits, symbols) is dropped with a warning (Italian).
        """
        if self._emit is None:
            self.load()
        warnings: list[str] = []
        s = unicodedata.normalize("NFC", text).lower()
        s = "".join(_APOSTROPHES.get(c, c) for c in s)
        out: list[str] = []
        dropped: list[str] = []
        degraded: list[str] = []
        for c in s:
            if c != self._delim and c in self._vocab:
                out.append(c)
            elif c.isspace() or c in "'-" or unicodedata.category(c).startswith("P"):
                out.append(" ")
            else:
                base = unicodedata.normalize("NFD", c)[0]
                if base != c and base in self._vocab:
                    out.append(base)
                    degraded.append(c)
                else:
                    out.append(" ")
                    dropped.append(c)
        words = "".join(out).split()
        if degraded:
            warnings.append(f"Lettere accentate non nell'alfabeto del modello, sostituite: {''.join(sorted(set(degraded)))}")
        if dropped:
            warnings.append(
                f"Caratteri ignorati perché non valutabili dal modello: {''.join(sorted(set(dropped)))} "
                "(scrivere i numeri in lettere)."
            )
        return words, warnings

    # ------------------------------------------------------------- scoring
    def emissions(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        """(T x C) log-probabilities and the number of 16 kHz samples."""
        if self._emit is None:
            self.load()
        a = prepare_audio(audio, sr)
        if len(a) > MAX_SECONDS * TARGET_SR:
            raise ValueError(
                f"Tratto troppo lungo per il confronto tra frasi candidate ({len(a) / TARGET_SR:.1f} s; "
                f"massimo {MAX_SECONDS:.0f} s). Selezionare un tratto più breve."
            )
        return self._emit(a), len(a)

    def _score_one(self, lp: np.ndarray, n_samples: int, text: str) -> CTCCandidateScore:
        n_frames = lp.shape[0]
        free = float(lp.max(axis=1).sum())
        words, warnings = self.normalize_text(text)
        targets: list[int] = []
        word_token_ranges: list[tuple[int, int]] = []
        for i, w in enumerate(words):
            if i > 0:
                targets.append(self._vocab[self._delim])
            a = len(targets)
            targets.extend(self._vocab[c] for c in w)
            word_token_ranges.append((a, len(targets)))
        base = {"text": text, "normalized": " ".join(words), "free_logprob": free,
                "n_frames": n_frames, "n_tokens": len(targets), "warnings": warnings}
        if not targets:
            warnings.append("Candidato vuoto dopo la normalizzazione: non valutabile.")
            return CTCCandidateScore(feasible=False, viterbi_logprob=float("-inf"), **base)
        path = ctc_viterbi(lp, targets, self._blank)
        if not np.isfinite(path.logprob):
            warnings.append("Frase troppo lunga per la durata del tratto: allineamento impossibile.")
            return CTCCandidateScore(feasible=False, viterbi_logprob=float("-inf"), **base)
        sec_per_frame = n_samples / n_frames / TARGET_SR
        spans = path.token_spans(len(targets))
        aligned = []
        for w, (a, b) in zip(words, word_token_ranges):
            frames = [t for k in range(a, b) for t in range(*spans[k])]
            aligned.append(WordAlignment(
                word=w, start=spans[a][0] * sec_per_frame, end=spans[b - 1][1] * sec_per_frame,
                mean_logprob=float(np.mean(path.frame_logprobs[frames])), n_frames=len(frames),
            ))
        return CTCCandidateScore(feasible=True, viterbi_logprob=path.logprob, words=aligned, **base)

    def score_detailed(self, audio: np.ndarray, sr: int, candidates: list[str]) -> list[CTCCandidateScore]:
        lp, n = self.emissions(audio, sr)
        return [self._score_one(lp, n, c) for c in candidates]

    def score(self, audio: np.ndarray, sr: int, candidates: list[str]) -> list[float]:
        return [r.score for r in self.score_detailed(audio, sr, candidates)]
