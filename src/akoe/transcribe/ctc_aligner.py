# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Acoustic aligner for untimed transcriptions (issue #7).

Implements the `TranscriptAligner` hook of `speaker/dataset.py` with the CTC
forced alignment of `ctc_scorer.py`: the whole plain-text transcription of a
recording is force-aligned to the emissions of an Italian CTC model (Viterbi
on the CTC trellis), then cut into timed spans that become `DatasetItem`s.

Spans
-----
The text is kept as the operator wrote it (case, punctuation, apostrophes);
only the alignment uses the model's alphabet. A span ends after a token with
sentence punctuation (``. ! ? …``), at a pause longer than `pause_s` between
aligned words, or before it would exceed `max_span_s` (Whisper sees at most
30 s). Tokens the model cannot score (digits, symbols) stay in the text of
the span they fall in and get no time of their own.

Score
-----
Each span carries a CTC score: the log-probability of the forced path minus
that of the unconstrained best path over the span's frames (0 = exactly what
the model would decode freely, always <= 0), divided by the larger of the
number of target characters and the number of characters the free path emits
there. Per-character rather than per-frame normalization (as in
`ctc_scorer`), because CTC paths are mostly blank: averaging over frames
dilutes a wrong text placed over a long stretch of other speech. Spans where
the model recognizes almost nothing also score low, which is intended: their
alignment cannot be checked. `SubjectDataset.align_pending(..., min_score=...)`
flags spans below the threshold as possibly misaligned. `DEFAULT_MIN_SCORE`
(-1.0) was checked on real Italian clips with a small CTC model (correct spans
mostly above -0.9, a shifted text mostly below -1.4), to be confirmed on case
material.
On poor audio a wrong text can still "align" (Loakes 2022): the score is a
warning, not a proof of correctness, and aligned items always need the
transcriber's verification before use.

Long recordings
---------------
Emissions are computed in windows (`window_s`, with `context_s` of audio on
each side, frames on the wav2vec2 grid of 320 samples); the Viterbi pass runs
on the whole recording. The trellis needs about `frames x (2 x characters + 1)`
bytes; above `max_trellis_bytes` the recording is refused with a message
(split it into shorter recordings).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from akoe.neural import TARGET_SR, ModelProvenance, prepare_audio
from akoe.speaker.dataset import AlignedSpan
from akoe.transcribe.ctc_scorer import CTCForcedAlignmentScorer, ctc_viterbi

DEFAULT_MIN_SCORE = -1.0
FRAME_STRIDE = 320  # wav2vec2 / XLS-R feature encoder: 20 ms at 16 kHz
RECEPTIVE_FIELD = 400
_SENTENCE_END = re.compile(r"[.!?…]+[\"'»”)\]]*$")


@dataclass
class _Token:
    text: str
    words: list[int] = field(default_factory=list)  # indices into the normalized word list
    start: float | None = None
    end: float | None = None


class CTCTranscriptAligner:
    """`TranscriptAligner` over CTC forced alignment (see module docstring)."""

    def __init__(
        self,
        scorer: CTCForcedAlignmentScorer | None = None,
        *,
        pause_s: float = 0.5,
        max_span_s: float = 15.0,
        min_span_s: float = 0.5,
        pad_s: float = 0.15,
        window_s: float = 30.0,
        context_s: float = 1.0,
        min_score: float = DEFAULT_MIN_SCORE,
        max_trellis_bytes: float = 600e6,
    ) -> None:
        self.scorer = scorer or CTCForcedAlignmentScorer()
        self.pause_s = pause_s
        self.max_span_s = max_span_s
        self.min_span_s = min_span_s
        self.pad_s = pad_s
        self.window_s = window_s
        self.context_s = context_s
        self.min_score = min_score
        self.max_trellis_bytes = max_trellis_bytes
        base = self.scorer.name.removesuffix("/ctc-forced-alignment")
        self.name = f"{base}/ctc-aligner"
        self.last_report: dict[str, Any] = {}

    @property
    def provenance(self) -> ModelProvenance | None:
        return self.scorer.provenance

    def params(self) -> dict[str, Any]:
        return {
            "method": "ctc_forced_alignment_viterbi",
            "pause_s": self.pause_s, "max_span_s": self.max_span_s, "min_span_s": self.min_span_s,
            "pad_s": self.pad_s, "window_s": self.window_s, "context_s": self.context_s,
            "min_score": self.min_score,
            "score": "(forced - free path log-probability) / max(target chars, free-path chars), <= 0",
        }

    # ------------------------------------------------------------- emissions
    def _emissions(self, a16: np.ndarray) -> tuple[np.ndarray, float]:
        """(T x C) log-probabilities over the whole audio and seconds per frame."""
        n = len(a16)
        win = int(self.window_s * TARGET_SR) // FRAME_STRIDE * FRAME_STRIDE
        ctx = int(self.context_s * TARGET_SR) // FRAME_STRIDE * FRAME_STRIDE
        if n <= win + 2 * ctx:
            lp, n_samples = self.scorer.emissions(a16, TARGET_SR)
            return lp, n_samples / max(lp.shape[0], 1) / TARGET_SR
        total = (n - RECEPTIVE_FIELD) // FRAME_STRIDE + 1
        parts: list[np.ndarray] = []
        for ws in range(0, n, win):
            we = min(n, ws + win)
            cs = max(0, ws - ctx)
            ce = min(n, we + ctx)
            lp, _ = self.scorer.emissions(a16[cs:ce], TARGET_SR)
            first = (ws - cs) // FRAME_STRIDE
            want = min(total, (we + FRAME_STRIDE - 1) // FRAME_STRIDE) - ws // FRAME_STRIDE
            parts.append(lp[first:first + max(0, want)])
        lp = np.concatenate(parts, axis=0)[:total]
        return lp, FRAME_STRIDE / TARGET_SR

    # ------------------------------------------------------------- alignment
    def _targets(self, tokens: list[_Token], warnings: list[str]) -> tuple[list[str], list[int], list[tuple[int, int]]]:
        """Normalize each whitespace token to the model's alphabet (filling
        `tok.words`) and build the CTC target sequence with the word ranges."""
        words: list[str] = []
        for tok in tokens:
            ws, w = self.scorer.normalize_text(tok.text)
            warnings.extend(x for x in w if x not in warnings)
            for word in ws:
                tok.words.append(len(words))
                words.append(word)
        vocab, delim = self.scorer._vocab, self.scorer._delim
        targets: list[int] = []
        ranges: list[tuple[int, int]] = []
        for i, w in enumerate(words):
            if i:
                targets.append(vocab[delim])
            a = len(targets)
            targets.extend(vocab[c] for c in w)
            ranges.append((a, len(targets)))
        return words, targets, ranges

    def _force_align(self, a16: np.ndarray, targets: list[int], warnings: list[str]) -> tuple[np.ndarray, float, Any] | None:
        """Emissions and Viterbi path, or None (reason appended to `warnings`)."""
        duration = len(a16) / TARGET_SR
        est_frames = len(a16) / FRAME_STRIDE
        trellis = est_frames * (2 * len(targets) + 1)
        if trellis > self.max_trellis_bytes:
            warnings.append(
                f"Registrazione troppo lunga per l'allineamento in un solo passaggio ({duration:.0f} s, "
                f"{len(targets)} caratteri): dividerla in registrazioni più brevi."
            )
            return None
        lp, spf = self._emissions(a16)
        path = ctc_viterbi(lp, targets, self.scorer._blank)
        if not np.isfinite(path.logprob):
            warnings.append("Testo troppo lungo per la durata dell'audio: allineamento impossibile.")
            return None
        return lp, spf, path

    def word_times(self, audio: np.ndarray, sr: int, words: list[str]) -> list[tuple[float, float] | None]:
        """Start and end (seconds from the start of `audio`) of each given
        whitespace word, by CTC forced alignment of the whole word sequence.

        A word with nothing the model can score (digits, symbols) gets None,
        and so does every word when the alignment is impossible (reason, in
        Italian, in `last_report["warnings"]`). Used by ASR backends whose
        model gives no timestamps (Cohere Transcribe)."""
        warnings: list[str] = []
        self.last_report = {"aligner": self.name, "warnings": warnings}
        tokens = [_Token(w) for w in words]
        norm, targets, ranges = self._targets(tokens, warnings)
        if not norm:
            warnings.append("Trascrizione senza parole allineabili.")
            return [None] * len(words)
        aligned = self._force_align(prepare_audio(audio, sr), targets, warnings)
        if aligned is None:
            return [None] * len(words)
        lp, spf, path = aligned
        spans = path.token_spans(len(targets))
        out: list[tuple[float, float] | None] = []
        for tok in tokens:
            if not tok.words:
                out.append(None)
                continue
            a, b = ranges[tok.words[0]][0], ranges[tok.words[-1]][1]
            out.append((round(spans[a][0] * spf, 3), round(spans[b - 1][1] * spf, 3)))
        free = lp.max(axis=1)
        n_free = int(np.count_nonzero(lp.argmax(axis=1) != self.scorer._blank))
        overall = float((path.frame_logprobs - free).sum() / max(len(targets), n_free, 1))
        self.last_report.update(overall_score=round(overall, 4), n_words=len(norm), n_frames=int(lp.shape[0]),
                                seconds_per_frame=spf)
        return out

    def align(self, audio: np.ndarray, sr: int, text: str) -> list[AlignedSpan]:
        """Timed spans of `text` in `audio`; [] when alignment is impossible
        (the reason, in Italian, is in `last_report["warnings"]`)."""
        warnings: list[str] = []
        self.last_report = {"aligner": self.name, "warnings": warnings}
        tokens = [_Token(t) for t in text.split()]
        words, targets, ranges = self._targets(tokens, warnings)
        if not words:
            warnings.append("Trascrizione senza parole allineabili.")
            return []

        a16 = prepare_audio(audio, sr)
        duration = len(a16) / TARGET_SR
        aligned = self._force_align(a16, targets, warnings)
        if aligned is None:
            return []
        lp, spf, path = aligned
        free = lp.max(axis=1)
        free_ids = lp.argmax(axis=1)
        gap = path.frame_logprobs - free  # <= 0 per frame
        spans = path.token_spans(len(targets))
        word_frames = [(spans[a][0], spans[b - 1][1]) for a, b in ranges]
        for tok in tokens:
            if tok.words:
                tok.start = word_frames[tok.words[0]][0] * spf
                tok.end = word_frames[tok.words[-1]][1] * spf

        groups = self._group(tokens)
        out: list[AlignedSpan] = []
        timed_groups = [(g, [t for t in g if t.start is not None]) for g in groups]
        for k, (group, timed) in enumerate(timed_groups):
            if not timed:  # only unscorable tokens: glue them to the previous span
                if out:
                    out[-1] = out[-1].model_copy(
                        update={"text": out[-1].text + " " + " ".join(t.text for t in group)})
                continue
            first_w = timed[0].words[0]
            last_w = timed[-1].words[-1]
            f0, f1 = word_frames[first_w][0], word_frames[last_w][1]
            n_tok = sum(ranges[w][1] - ranges[w][0] for t in timed for w in t.words)
            n_free = int(np.count_nonzero(free_ids[f0:f1] != self.scorer._blank))
            score = float(gap[f0:f1].sum() / max(n_tok, n_free, 1)) if f1 > f0 else float("-inf")
            start, end = timed[0].start, timed[-1].end
            prev_end = out[-1].end if out else 0.0
            nxt = next((tg[0].start for _, tg in timed_groups[k + 1:] if tg), duration)
            out.append(AlignedSpan(
                start=round(max(prev_end, start - self.pad_s, 0.0), 3),
                end=round(min(nxt, end + self.pad_s, duration), 3),
                text=" ".join(t.text for t in group),
                score=round(score, 4),
            ))
        if out and timed_groups and not timed_groups[0][1]:
            lead = " ".join(t.text for t in timed_groups[0][0])
            out[0] = out[0].model_copy(update={"text": lead + " " + out[0].text})
        n_free_all = int(np.count_nonzero(free_ids != self.scorer._blank))
        overall = float(gap.sum() / max(len(targets), n_free_all, 1))
        self.last_report.update(
            overall_score=round(overall, 4), n_words=len(words), n_spans=len(out),
            n_frames=int(lp.shape[0]), seconds_per_frame=spf,
            n_flagged=sum(1 for s in out if s.score is not None and s.score < self.min_score),
        )
        if overall < self.min_score:
            warnings.append(
                f"Allineamento complessivamente scadente (punteggio {overall:.2f} < {self.min_score:.2f}): "
                "verificare che la trascrizione corrisponda all'audio."
            )
        return out

    def _group(self, tokens: list[_Token]) -> list[list[_Token]]:
        groups: list[list[_Token]] = []
        cur: list[_Token] = []
        for i, tok in enumerate(tokens):
            cur.append(tok)
            timed = [t for t in cur if t.start is not None]
            nxt = next((t for t in tokens[i + 1:] if t.start is not None), None)
            if nxt is None or not timed:
                continue
            dur = timed[-1].end - timed[0].start  # type: ignore[operator]
            sentence_end = bool(_SENTENCE_END.search(tok.text)) and dur >= self.min_span_s
            pause = tok.start is not None and nxt.start - tok.end > self.pause_s  # type: ignore[operator]
            too_long = nxt.end - timed[0].start > self.max_span_s  # type: ignore[operator]
            if sentence_end or pause or too_long:
                groups.append(cur)
                cur = []
        if cur:
            groups.append(cur)
        return groups
