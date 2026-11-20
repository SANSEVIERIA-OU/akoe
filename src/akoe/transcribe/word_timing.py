# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Word-level helpers shared by the transformers ASR backends (Cohere
Transcribe, Qwen3-ASR, Nemotron): grouping generated tokens into words,
word confidence from token log-probabilities, word times from an external
aligner.

- `token_word_groups`: tokenizer-agnostic (SentencePiece "▁", byte-level BPE
  "Ġ", multi-token UTF-8 characters): token k covers the characters that
  decoding the first k tokens adds to the final text.
- Word confidence = probability of the word's tokens given the audio and the
  preceding text (product of token probabilities, teacher-forced); raw and
  uncalibrated, as in `whisper_backend` (calibration: self-assessment, M5).
- `WordTimer`: anything with `word_times(audio, sr, words)`; implemented by
  `ctc_aligner.CTCTranscriptAligner` (Italian CTC model) and
  `qwen3_backend.Qwen3ForcedAligner`. Words the aligner cannot time get the
  gap between their timed neighbours, never an invented precise time.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from akoe.models import WordHypothesis

Times = list[tuple[float, float] | None]


@runtime_checkable
class WordTimer(Protocol):
    name: str

    def word_times(self, audio: np.ndarray, sr: int, words: list[str]) -> Times:
        """(start, end) in seconds from the start of `audio` for each word, or
        None for a word that cannot be timed."""
        ...


def resolve_timer(timer: WordTimer | Callable[[], WordTimer] | None) -> WordTimer | None:
    """Accept a timer or a zero-argument factory (lazy, e.g. a cached provider method)."""
    if timer is None or isinstance(timer, WordTimer):
        return timer
    return timer()


def _common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def token_word_groups(decode: Callable[[list[int]], str], ids: Sequence[int]) -> list[tuple[str, list[int]]]:
    """Split the decoded text of `ids` into whitespace words and give each word
    the positions (indices into `ids`) of the tokens that produce it.

    A token producing no visible character (a lone "▁", part of a multi-byte
    character) goes to the word of the next visible token; a token spanning
    two words goes to the first one (the second then has no tokens and no
    confidence)."""
    ids = list(ids)
    text = decode(ids)
    ends = [0]
    for k in range(1, len(ids) + 1):
        ends.append(max(ends[-1], _common_prefix(decode(ids[:k]), text)))
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        j = i
        while j < len(text) and not text[j].isspace():
            j += 1
        spans.append((i, j))
        i = j
    owner = [-1] * len(text)
    for w, (a, b) in enumerate(spans):
        for c in range(a, b):
            owner[c] = w
    groups: list[list[int]] = [[] for _ in spans]
    pending: list[int] = []
    for k in range(len(ids)):
        visible = next((c for c in range(ends[k], ends[k + 1]) if owner[c] >= 0), None)
        if visible is None:
            pending.append(k)
            continue
        groups[owner[visible]].extend(pending + [k])
        pending = []
    if pending and groups:
        groups[-1].extend(pending)
    return [(text[a:b], g) for (a, b), g in zip(spans, groups)]


def fill_missing_times(times: Times, seg_dur: float) -> list[tuple[float, float]]:
    """Clip times to [0, seg_dur]; a word without a time spans the gap between
    the end of the previous timed word and the start of the next one."""
    clipped: Times = []
    for t in times:
        if t is None:
            clipped.append(None)
            continue
        s = float(min(max(t[0], 0.0), seg_dur))
        clipped.append((s, float(min(max(t[1], s), seg_dur))))
    out: list[tuple[float, float]] = []
    for i, t in enumerate(clipped):
        if t is not None:
            out.append(t)
            continue
        prev_end = out[-1][1] if out else 0.0
        nxt = next((x[0] for x in clipped[i + 1:] if x is not None), seg_dur)
        out.append((prev_end, max(prev_end, nxt)))
    return out


def build_words(
    groups: list[tuple[str, list[int]]],
    token_logprobs: Sequence[float],
    times: Times | None,
    *,
    offset: float,
    seg_dur: float,
    origin: str,
) -> list[WordHypothesis]:
    """`WordHypothesis` list from token groups, per-token log-probabilities
    (same indexing as the token ids) and optional aligner times (relative to
    the segment). Without times every word spans the whole segment, as in
    `whisper_backend` when no alignment is available."""
    if times is None or len(times) != len(groups):
        spans = [(0.0, seg_dur)] * len(groups)
    else:
        spans = fill_missing_times(times, seg_dur)
    words = []
    for (text, toks), (s, e) in zip(groups, spans):
        conf = float(np.exp(sum(token_logprobs[k] for k in toks))) if toks else None
        words.append(WordHypothesis(text=text, start=offset + s, end=offset + e, confidence=conf, origin=origin))
    return words


def timer_provenance(timer: Any) -> dict[str, Any] | None:
    """Short description of the aligner for the backend's provenance."""
    if timer is None:
        return None
    prov = getattr(timer, "provenance", None)
    out: dict[str, Any] = {"name": getattr(timer, "name", type(timer).__name__)}
    if prov is not None:
        out.update(model_id=prov.model_id, revision=prov.revision, weights=prov.weights, license=prov.license)
    return out
