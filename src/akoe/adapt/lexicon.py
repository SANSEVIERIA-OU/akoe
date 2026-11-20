# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Case lexicon and controlled contextual biasing (IMPLEMENTATION_PLAN M4, issue #26).

Names, places and jargon of the case are proposed from previous *verified*
transcripts and used only after the operator confirms them. Biasing makes the
recogniser insert them where they are not said (MASTER_PLAN §1.4, §3.4: without
a filter, errors on the other words went from 1.9% to 21.2%, Liu et al. 2025),
so it is used with three guarantees (MASTER_PLAN §4.5):

1. controlled boosting: :class:`LexiconBiasProcessor` boosts lexicon token
   sequences through a prefix trie, but only for tokens the model already finds
   plausible (confidence gate), with a cap on lexicon terms per hypothesis and,
   in beam search, withdrawal of the bonus of terms left unfinished. It is
   inspired by trie-based biasing with filtering, not a reimplementation of a
   specific paper;
2. the lexicon is versioned by SHA-256 (:meth:`CaseLexicon.save`);
3. decoding with and without the lexicon (:class:`DualHypothesis`): when the two
   differ both are kept and the segment is flagged ``LEXICON_DEPENDENT``;

plus the mandatory insertion test (:func:`measure_insertion_rate`): how often
lexicon terms appear in hypotheses of audio where they are absent.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, Field

from akoe.models import Alternative, ArtifactRef, DatasetItem, DatasetSplit, SegmentFlag
from akoe.utils import sha256_file

try:  # transformers is an optional (ml) dependency; the lexicon itself is not
    from transformers import LogitsProcessor as _LogitsProcessorBase
except ImportError:  # pragma: no cover
    _LogitsProcessorBase = object  # type: ignore[assignment,misc]

T = TypeVar("T")


class LexiconError(Exception):
    """Raised with an operator-facing (Italian) message."""


# --------------------------------------------------------------------------
# text normalisation and term matching
# --------------------------------------------------------------------------

_APOSTROPHES = "'’`´"
_SENTENCE_END = set(".!?…;:\n")


def normalize_text(text: str) -> str:
    """Casefold, NFC, punctuation and apostrophes to spaces (accents are kept:
    in Italian they distinguish words)."""
    text = unicodedata.normalize("NFC", text).casefold()
    text = re.sub(rf"[{_APOSTROPHES}]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def contains_term(text: str, term: str, *, normalized: bool = False) -> bool:
    t = text if normalized else normalize_text(text)
    n = normalize_text(term)
    return bool(n) and f" {n} " in f" {t} "


def count_term(text: str, term: str) -> int:
    t = f" {normalize_text(text)} "
    n = normalize_text(term)
    if not n:
        return 0
    return len(re.findall(rf"(?<= ){re.escape(n)}(?= )", t))


def find_terms(text: str, terms: Iterable[str]) -> list[str]:
    """Terms (deduplicated by normalised form) that occur in ``text``."""
    t = normalize_text(text)
    out: dict[str, str] = {}
    for term in terms:
        key = normalize_text(term)
        if key and key not in out and contains_term(t, term, normalized=True):
            out[key] = term
    return sorted(out.values(), key=normalize_text)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level Levenshtein distance / reference length (normalised text)."""
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


# Italian function words and frequent sentence openers that are capitalised in
# transcripts without being names.
ITALIAN_STOPWORDS = frozenset(
    """
    a ad al allo alla ai agli alle anche ancora che chi ci cioè come con contro così cosa da dal dallo
    dalla dai dagli dalle del dello della dei degli delle di dove dopo e ed è era ero fa già gli ha
    hai hanno ho i il in io la le lei li lo loro lui ma me mi mia mie miei mio molto ne nei negli nel
    nello nella nelle no noi non nostra nostro o per perché però più poi qua quale quando quanto quel
    quella quelle quelli quello questa queste questi questo qui se sei senza si sì sia siamo siete
    sono sopra sotto su sua sue sui suo suoi sul sulla te ti tra tu tua tuo tutto tutti un una uno
    va vabbè vai vi voi vostra vostro ok okay allora boh eh ah oh mah beh dai senti guarda ciao
    buongiorno buonasera grazie prego sì signore signora dottore avvocato
    """.split()
)

_WORD_RE = re.compile(r"[^\W_]+(?:[-][^\W_]+)*", re.UNICODE)


def _tokens_with_boundaries(text: str) -> list[tuple[str, bool]]:
    """Words of ``text`` with a flag telling whether each starts a sentence."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(rf"[{_APOSTROPHES}]", " ", text)
    out: list[tuple[str, bool]] = []
    sentence_start = True
    pos = 0
    for m in _WORD_RE.finditer(text):
        gap = text[pos : m.start()]
        if any(c in _SENTENCE_END for c in gap):
            sentence_start = True
        out.append((m.group(0), sentence_start))
        sentence_start = False
        pos = m.end()
    return out


# --------------------------------------------------------------------------
# lexicon model
# --------------------------------------------------------------------------


class TermCategory(str, Enum):
    NAME = "name"
    PLACE = "place"
    JARGON = "jargon"
    OTHER = "other"


class TermStatus(str, Enum):
    PROPOSED = "proposed"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class LexiconEntry(BaseModel):
    term: str
    category: TermCategory = TermCategory.OTHER
    status: TermStatus = TermStatus.PROPOSED
    count: int = 0  # occurrences in the verified transcripts used
    source_items: list[str] = Field(default_factory=list)
    variants: list[str] = Field(default_factory=list)  # other spellings, confirmed with the term
    decided_by: str | None = None
    decided_at: datetime | None = None
    note: str | None = None

    @property
    def key(self) -> str:
        return normalize_text(self.term)


class CaseLexicon(BaseModel):
    """Lexicon of a case. Only operator-confirmed entries are used for biasing."""

    lexicon_id: str
    case_id: str | None = None
    version: int = 1
    entries: list[LexiconEntry] = Field(default_factory=list)
    source_items: list[str] = Field(default_factory=list)

    # -- queries -------------------------------------------------------------

    def get(self, term: str) -> LexiconEntry | None:
        key = normalize_text(term)
        return next((e for e in self.entries if e.key == key), None)

    def pending(self) -> list[LexiconEntry]:
        return [e for e in self.entries if e.status == TermStatus.PROPOSED]

    def confirmed(self) -> list[LexiconEntry]:
        return [e for e in self.entries if e.status == TermStatus.CONFIRMED]

    def terms(self) -> list[str]:
        """Surface forms of the confirmed entries (with variants), deduplicated."""
        out: dict[str, str] = {}
        for e in self.confirmed():
            for form in [e.term, *e.variants]:
                out.setdefault(form.strip(), form.strip())
        return sorted(v for v in out.values() if v)

    # -- operator decisions --------------------------------------------------

    def _require(self, term: str) -> LexiconEntry:
        entry = self.get(term)
        if entry is None:
            raise LexiconError(f"Termine non presente nel lessico: {term!r}.")
        return entry

    def confirm(
        self,
        term: str,
        operator: str,
        *,
        category: TermCategory | None = None,
        variants: Sequence[str] | None = None,
        note: str | None = None,
    ) -> LexiconEntry:
        if not operator.strip():
            raise LexiconError("La conferma di un termine richiede l'identificativo dell'operatore.")
        entry = self._require(term)
        entry.status = TermStatus.CONFIRMED
        if category is not None:
            entry.category = category
        if variants is not None:
            entry.variants = sorted({v.strip() for v in variants if v.strip()})
        entry.decided_by = operator
        entry.decided_at = datetime.now(timezone.utc)
        entry.note = note if note is not None else entry.note
        self.version += 1
        return entry

    def reject(self, term: str, operator: str, *, note: str | None = None) -> LexiconEntry:
        if not operator.strip():
            raise LexiconError("Il rifiuto di un termine richiede l'identificativo dell'operatore.")
        entry = self._require(term)
        entry.status = TermStatus.REJECTED
        entry.decided_by = operator
        entry.decided_at = datetime.now(timezone.utc)
        entry.note = note if note is not None else entry.note
        self.version += 1
        return entry

    def add_term(
        self,
        term: str,
        operator: str,
        *,
        category: TermCategory = TermCategory.OTHER,
        variants: Sequence[str] = (),
        note: str | None = None,
    ) -> LexiconEntry:
        """Term typed by the operator: confirmed directly."""
        if not normalize_text(term):
            raise LexiconError("Termine vuoto.")
        if self.get(term) is None:
            self.entries.append(LexiconEntry(term=term.strip(), category=category))
        return self.confirm(term, operator, category=category, variants=list(variants), note=note)

    # -- persistence ---------------------------------------------------------

    def canonical_json(self) -> bytes:
        data = self.model_dump(mode="json")
        return (json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json()).hexdigest()

    def save(self, path: Path, root: Path | None = None) -> ArtifactRef:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.canonical_json())
        return ArtifactRef.from_file(path, kind="lexicon", root=root)

    @classmethod
    def load(cls, path: Path, expected_sha256: str | None = None) -> "CaseLexicon":
        if expected_sha256 is not None and sha256_file(path) != expected_sha256:
            raise LexiconError(
                f"Il lessico {path} non corrisponde all'impronta SHA-256 registrata: caricamento rifiutato."
            )
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def propose_lexicon(
    items: Sequence[DatasetItem],
    *,
    lexicon_id: str,
    case_id: str | None = None,
    existing: CaseLexicon | None = None,
    known_words: Iterable[str] | None = None,
    min_count: int = 1,
    min_jargon_count: int = 2,
    min_chars: int = 3,
    max_words: int = 3,
    include_holdout: bool = False,
) -> CaseLexicon:
    """Propose lexicon terms from verified transcripts; the operator then confirms.

    Candidates:
    - runs of up to ``max_words`` capitalised words not at the start of a
      sentence (names, places) -> ``NAME`` (the operator can change it to ``PLACE``);
    - alphanumeric codes (plates, numbers mixed with letters) -> ``OTHER``;
    - with ``known_words`` (a common-vocabulary list), lowercase words outside
      it seen at least ``min_jargon_count`` times -> ``JARGON``.

    Unverified items are never used; holdout items are excluded by default so
    the self-assessment does not see a lexicon built from its own answers.
    Decisions already in ``existing`` are kept (rejected terms stay rejected).
    """
    eligible = [
        i for i in items if i.verified and (include_holdout or i.split != DatasetSplit.HOLDOUT)
    ]
    known = {normalize_text(w) for w in known_words} if known_words is not None else None
    candidates: dict[str, tuple[str, TermCategory]] = {}

    def add(term: str, category: TermCategory) -> None:
        key = normalize_text(term)
        if len(key.replace(" ", "")) < min_chars or key in ITALIAN_STOPWORDS:
            return
        candidates.setdefault(key, (term, category))

    for item in eligible:
        tokens = _tokens_with_boundaries(item.text)
        run: list[str] = []
        for word, sentence_start in tokens + [("", True)]:
            has_digit = any(c.isdigit() for c in word)
            is_cap = bool(word) and word[0].isupper() and not sentence_start and not has_digit
            if is_cap and word.casefold() not in ITALIAN_STOPWORDS and len(run) < max_words:
                run.append(word)
                continue
            if run:
                add(" ".join(run), TermCategory.NAME)
                run = []
            if is_cap and word.casefold() not in ITALIAN_STOPWORDS:
                run = [word]
        for word, _ in tokens:
            if any(c.isdigit() for c in word) and any(c.isalpha() for c in word):
                add(word, TermCategory.OTHER)
            elif known is not None and word.islower() and normalize_text(word) not in known:
                add(word, TermCategory.JARGON)

    lexicon = (
        existing.model_copy(deep=True)
        if existing is not None
        else CaseLexicon(lexicon_id=lexicon_id, case_id=case_id)
    )
    for key, (term, category) in sorted(candidates.items()):
        hits = [(i.item_id, count_term(i.text, term)) for i in eligible]
        count = sum(c for _, c in hits)
        sources = sorted(item_id for item_id, c in hits if c)
        needed = min_jargon_count if category == TermCategory.JARGON else min_count
        if count < needed:
            continue
        entry = lexicon.get(term)
        if entry is None:
            lexicon.entries.append(
                LexiconEntry(term=term, category=category, count=count, source_items=sources)
            )
        else:
            entry.count = count
            entry.source_items = sources
    lexicon.entries.sort(key=lambda e: (-e.count, e.key))
    lexicon.source_items = sorted({*lexicon.source_items, *(i.item_id for i in eligible)})
    return lexicon


# --------------------------------------------------------------------------
# token trie and biasing
# --------------------------------------------------------------------------


class BiasConfig(BaseModel):
    """Defaults are conservative; the insertion test measures their effect."""

    boost: float = Field(default=2.0, ge=0.0)  # added to the score of a plausible lexicon token
    # A term may start only if its first token is within `start_margin` (log-prob)
    # of the model's best token and among its `start_top_k` best tokens.
    start_margin: float = Field(default=3.0, ge=0.0)
    start_top_k: int = Field(default=10, ge=1)
    # A started term continues to be boosted only for plausible continuations.
    continuation_margin: float = Field(default=6.0, ge=0.0)
    # Cap on lexicon terms per hypothesis; once reached, no new term is boosted.
    max_terms_per_decode: int | None = Field(default=3, ge=0)
    min_term_chars: int = Field(default=3, ge=1)
    add_capitalized_variants: bool = True


class TokenTrie:
    """Prefix trie over token-id sequences; nodes are integers, 0 is the root."""

    def __init__(self, sequences: dict[tuple[int, ...], str]) -> None:
        self.children: list[dict[int, int]] = [{}]
        self.terminal: list[str | None] = [None]
        self.sequences = dict(sequences)
        for seq, term in sorted(sequences.items()):
            node = 0
            for tok in seq:
                nxt = self.children[node].get(tok)
                if nxt is None:
                    nxt = len(self.children)
                    self.children[node][tok] = nxt
                    self.children.append({})
                    self.terminal.append(None)
                node = nxt
            self.terminal[node] = term
        self.max_depth = max((len(s) for s in sequences), default=0)

    def walk(self, tokens: Sequence[int]) -> list[int] | None:
        """Nodes visited by ``tokens`` from the root, or None if it falls off."""
        node = 0
        path = []
        for tok in tokens:
            node = self.children[node].get(tok, -1)
            if node < 0:
                return None
            path.append(node)
        return path


def term_token_sequences(
    tokenizer: Any,
    terms: Iterable[str],
    *,
    min_term_chars: int = 3,
    add_capitalized_variants: bool = True,
) -> tuple[dict[tuple[int, ...], str], list[str]]:
    """Token sequences of each term as a whole word (leading space, as Whisper
    emits words). Returns ``(sequences -> term, skipped_terms)``; terms shorter
    than ``min_term_chars`` are skipped (short terms cause insertions)."""
    sequences: dict[tuple[int, ...], str] = {}
    skipped: list[str] = []
    for term in terms:
        term = " ".join(term.split())
        if len(normalize_text(term).replace(" ", "")) < min_term_chars:
            skipped.append(term)
            continue
        forms = {term}
        if add_capitalized_variants and term[:1].islower():
            forms.add(term[:1].upper() + term[1:])
        for form in sorted(forms):
            ids = tuple(tokenizer(" " + form, add_special_tokens=False).input_ids)
            if ids:
                sequences.setdefault(ids, term)
    return sequences, skipped


class LexiconBiasProcessor(_LogitsProcessorBase):
    """Hugging Face ``LogitsProcessor`` that boosts lexicon token sequences in a
    controlled way (prefix trie + filters).

    At each step, for every hypothesis row:

    - if the generated text ends inside a lexicon term (proper prefix in the
      trie), its continuations get ``boost``, but only those within
      ``continuation_margin`` of the best token;
    - otherwise, if fewer than ``max_terms_per_decode`` terms were already
      produced, first tokens of terms get ``boost``, but only those within
      ``start_margin`` of the best token and among the ``start_top_k`` best:
      the lexicon re-ranks what the model already hears, it does not create it;
    - with ``cumulative_scores`` (beam search, where scores are summed along the
      beam), the bonus given to a term left unfinished is withdrawn from every
      token that does not continue it.

    The processor is stateless with respect to beam reordering: bonuses are
    recorded by (batch item, token prefix). Use one instance per ``generate``
    call or call :meth:`reset`.
    """

    def __init__(
        self,
        sequences: dict[tuple[int, ...], str],
        config: BiasConfig | None = None,
        *,
        num_beams: int = 1,
        begin_token_id: int | None = None,
        cumulative_scores: bool | None = None,
    ) -> None:
        if not sequences:
            raise LexiconError("Nessun termine del lessico utilizzabile per il biasing.")
        self.config = config or BiasConfig()
        self.trie = TokenTrie(sequences)
        self.num_beams = max(1, num_beams)
        self.begin_token_id = begin_token_id
        self.cumulative = num_beams > 1 if cumulative_scores is None else cumulative_scores
        self._root_tokens = sorted(self.trie.children[0])
        self._root_tensor = None
        self.reset()

    @classmethod
    def from_tokenizer(
        cls, tokenizer: Any, terms: Iterable[str], config: BiasConfig | None = None, **kwargs: Any
    ) -> "LexiconBiasProcessor":
        config = config or BiasConfig()
        seqs, _ = term_token_sequences(
            tokenizer,
            terms,
            min_term_chars=config.min_term_chars,
            add_capitalized_variants=config.add_capitalized_variants,
        )
        return cls(seqs, config, **kwargs)

    def reset(self) -> None:
        self._awarded: dict[tuple[int, int, int], float] = {}
        self._counts: dict[tuple[int, int], int] = {}
        self._last_len = 0
        self.stats = {"start_boosts": 0, "continuation_boosts": 0, "withdrawals": 0, "capped_rows": 0}

    # -- helpers -------------------------------------------------------------

    def _text_part(self, seq: list[int]) -> list[int]:
        if self.begin_token_id is None:
            return seq
        for i in range(len(seq) - 1, -1, -1):
            if seq[i] == self.begin_token_id:
                return seq[i + 1 :]
        return seq

    def _completed(self, text: list[int]) -> int:
        """Number of positions of ``text`` where a lexicon term ends (cached)."""
        n = len(text)
        if n == 0:
            return 0
        key = (n, hash(tuple(text)))
        hit = self._counts.get(key)
        if hit is not None:
            return hit
        prev_key = (n - 1, hash(tuple(text[:-1])))
        prev = self._counts.get(prev_key)
        if prev is None:
            prev = self._completed(text[:-1]) if n <= 512 else 0
        ends = any(tuple(text[-k:]) in self.trie.sequences for k in range(1, min(n, self.trie.max_depth) + 1))
        count = prev + int(ends)
        if len(self._counts) > 200_000:
            self._counts.clear()
        self._counts[key] = count
        return count

    def _active_match(self, text: list[int]) -> tuple[int, int] | None:
        """(node, match length) of the longest suffix of ``text`` that is a proper
        prefix of a lexicon term (a node with children)."""
        for k in range(min(len(text), self.trie.max_depth - 1), 0, -1):
            path = self.trie.walk(text[-k:])
            if path is not None and self.trie.children[path[-1]]:
                return path[-1], k
        return None

    def _award_key(self, batch: int, seq: list[int]) -> tuple[int, int, int]:
        return (batch, len(seq), hash(tuple(seq)))

    def _pending_bonus(self, batch: int, seq: list[int], k: int) -> float:
        """Bonus awarded to the last ``k`` tokens after the last completed term."""
        text_suffix = seq[len(seq) - k :]
        path = self.trie.walk(text_suffix) or []
        last_terminal = max((i for i, node in enumerate(path) if self.trie.terminal[node]), default=-1)
        total = 0.0
        for i in range(last_terminal + 1, k):
            total += self._awarded.get(self._award_key(batch, seq[: len(seq) - k + i + 1]), 0.0)
        return total

    # -- main ----------------------------------------------------------------

    def __call__(self, input_ids, scores):  # type: ignore[override]
        import torch

        if input_ids.shape[1] < self._last_len:  # a new generate call reused this instance
            self.reset()
        self._last_len = input_ids.shape[1]
        cfg = self.config
        logp = torch.log_softmax(scores.float(), dim=-1)
        best = logp.max(dim=-1).values
        k = min(cfg.start_top_k, logp.shape[-1])
        kth = logp.topk(k, dim=-1).values[:, -1]
        bonus = torch.zeros_like(logp)
        root = self._root_tensor
        if root is None or root.device != scores.device:
            root = self._root_tensor = torch.tensor(self._root_tokens, dtype=torch.long, device=scores.device)

        for row, seq in enumerate(input_ids.tolist()):
            batch = row // self.num_beams
            text = self._text_part(seq)
            match = self._active_match(text)
            capped = cfg.max_terms_per_decode is not None and self._completed(text) >= cfg.max_terms_per_decode
            boosted: list[int] = []
            if match is not None:
                node, length = match
                child_ids = torch.tensor(sorted(self.trie.children[node]), dtype=torch.long, device=scores.device)
                if self.cumulative:
                    pending = self._pending_bonus(batch, seq, length)
                    if pending > 0:
                        bonus[row] -= pending
                        bonus[row, child_ids] += pending
                        self.stats["withdrawals"] += 1
                if not capped:
                    ok = logp[row, child_ids] >= best[row] - cfg.continuation_margin
                    boosted = child_ids[ok].tolist()
                    self.stats["continuation_boosts"] += len(boosted)
            elif not capped:
                threshold = torch.maximum(best[row] - cfg.start_margin, kth[row])
                ok = logp[row, root] >= threshold
                boosted = root[ok].tolist()
                self.stats["start_boosts"] += len(boosted)
            if capped:
                self.stats["capped_rows"] += 1
            if boosted and cfg.boost > 0:
                bonus[row, boosted] += cfg.boost
                for tok in boosted:
                    self._awarded[self._award_key(batch, seq + [tok])] = cfg.boost
        return scores + bonus.to(scores.dtype)


# --------------------------------------------------------------------------
# with / without lexicon
# --------------------------------------------------------------------------


def _resolve_terms(terms: CaseLexicon | Sequence[str]) -> tuple[list[str], str]:
    if isinstance(terms, CaseLexicon):
        resolved = terms.terms()
        if not resolved:
            raise LexiconError(
                "Il lessico non contiene termini confermati dall'operatore: il biasing non è applicabile."
            )
        return resolved, terms.sha256()
    resolved = sorted({" ".join(t.split()) for t in terms if t.strip()})
    digest = hashlib.sha256("\n".join(resolved).encode("utf-8")).hexdigest()
    return resolved, digest


class DualHypothesis(BaseModel):
    """Decoding of one segment with and without the case lexicon."""

    without_lexicon: str
    with_lexicon: str
    lexicon_sha256: str | None = None
    terms_introduced: list[str] = Field(default_factory=list)  # in "with" only
    terms_removed: list[str] = Field(default_factory=list)  # in "without" only

    @property
    def differs(self) -> bool:
        return normalize_text(self.with_lexicon) != normalize_text(self.without_lexicon)

    def alternatives(self, origin: str, audio_variant: str = "original") -> list[Alternative]:
        """Both readings when they differ. The reading without lexicon comes
        first (conservative: the lexicon can insert words that were not said)."""
        plain = Alternative(text=self.without_lexicon, origin=origin, audio_variant=audio_variant)
        if not self.differs:
            return [plain]
        tag = (self.lexicon_sha256 or "")[:12]
        biased = Alternative(
            text=self.with_lexicon, origin=f"{origin}+lexicon:{tag}", audio_variant=audio_variant
        )
        return [plain, biased]

    def flags(self) -> list[SegmentFlag]:
        return [SegmentFlag.LEXICON_DEPENDENT] if self.differs else []


def dual_decode(
    decode: Callable[[bool], str], terms: CaseLexicon | Sequence[str]
) -> DualHypothesis:
    """Model-agnostic: ``decode(use_lexicon)`` returns the text of one segment."""
    resolved, digest = _resolve_terms(terms)
    plain = decode(False)
    biased = decode(True)
    with_terms = set(find_terms(biased, resolved))
    without_terms = set(find_terms(plain, resolved))
    return DualHypothesis(
        without_lexicon=plain,
        with_lexicon=biased,
        lexicon_sha256=digest,
        terms_introduced=sorted(with_terms - without_terms),
        terms_removed=sorted(without_terms - with_terms),
    )


class WhisperLexiconDecoder:
    """Decodes Hugging Face Whisper (base or adapted, e.g. a PeftModel) with and
    without the lexicon.

    With greedy decoding (``num_beams=1``) a term, once started, cannot be
    withdrawn: if the model does not follow it, a fragment remains (observed
    with whisper-tiny and unfiltered biasing: "Cin" + "co"). The confidence gate
    is what prevents this; beam search (``num_beams >= 2``) also enables the
    withdrawal of unfinished terms.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        terms: CaseLexicon | Sequence[str],
        config: BiasConfig | None = None,
        *,
        language: str = "it",
        task: str = "transcribe",
        num_beams: int = 1,
        max_new_tokens: int = 128,
    ) -> None:
        self.model = model
        self.processor = processor
        self.config = config or BiasConfig()
        self.terms, self.lexicon_sha256 = _resolve_terms(terms)
        self.sequences, self.skipped_terms = term_token_sequences(
            processor.tokenizer,
            self.terms,
            min_term_chars=self.config.min_term_chars,
            add_capitalized_variants=self.config.add_capitalized_variants,
        )
        if not self.sequences:
            raise LexiconError("Nessun termine del lessico utilizzabile (tutti troppo corti).")
        self.language = language
        self.task = task
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.last_stats: dict[str, int] = {}

    def _generation_config(self) -> Any:
        model = self.model
        gc = getattr(model, "generation_config", None)
        if gc is None and hasattr(model, "get_base_model"):
            gc = model.get_base_model().generation_config
        return gc

    def make_processor(self) -> LexiconBiasProcessor:
        gc = self._generation_config()
        return LexiconBiasProcessor(
            self.sequences,
            self.config,
            num_beams=self.num_beams,
            begin_token_id=getattr(gc, "decoder_start_token_id", None),
        )

    def decode(self, audio: Any, sr: int, *, use_lexicon: bool) -> str:
        import torch
        from transformers import LogitsProcessorList

        param = next(self.model.parameters())
        feats = self.processor.feature_extractor(audio, sampling_rate=sr, return_tensors="pt").input_features
        feats = feats.to(device=param.device, dtype=param.dtype)
        kwargs: dict[str, Any] = dict(
            language=self.language,
            task=self.task,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
        )
        proc = None
        if use_lexicon:
            proc = self.make_processor()
            kwargs["logits_processor"] = LogitsProcessorList([proc])
        with torch.no_grad():
            out = self.model.generate(input_features=feats, **kwargs)
        sequences = out.sequences if hasattr(out, "sequences") else out
        if proc is not None:
            self.last_stats = dict(proc.stats)
        return self.processor.tokenizer.batch_decode(sequences, skip_special_tokens=True)[0].strip()

    def decode_both(self, audio: Any, sr: int) -> DualHypothesis:
        return dual_decode(lambda use: self.decode(audio, sr, use_lexicon=use), self.terms)


# --------------------------------------------------------------------------
# insertion test
# --------------------------------------------------------------------------


class ItemInsertion(BaseModel):
    item_id: str
    inserted: list[str]
    hypothesis: str


class InsertionStats(BaseModel):
    opportunities: int  # (item, lexicon term absent from its reference) pairs
    insertions: int  # of those, the term appears in the hypothesis
    insertion_rate: float
    items_with_insertion: int
    item_insertion_rate: float  # over items with at least one absent term
    per_term: dict[str, int] = Field(default_factory=dict)
    mean_wer: float | None = None
    details: list[ItemInsertion] = Field(default_factory=list)


class InsertionReport(BaseModel):
    """Insertion test of the lexicon (self-assessment, M5)."""

    lexicon_sha256: str
    n_items: int
    n_items_with_absent_terms: int
    n_terms: int
    with_lexicon: InsertionStats
    without_lexicon: InsertionStats | None = None

    @property
    def excess_insertion_rate(self) -> float | None:
        """Insertion rate attributable to the lexicon (with minus without)."""
        if self.without_lexicon is None:
            return None
        return self.with_lexicon.insertion_rate - self.without_lexicon.insertion_rate


def _default_item_id(item: Any, index: int) -> str:
    return str(getattr(item, "item_id", index))


def _insertion_stats(
    hyps: list[str], refs: list[str], ids: list[str], terms: list[str]
) -> InsertionStats:
    opportunities = insertions = items_hit = items_with_absent = 0
    per_term: dict[str, int] = {}
    details: list[ItemInsertion] = []
    wers: list[float] = []
    for hyp, ref, item_id in zip(hyps, refs, ids):
        ref_n = normalize_text(ref)
        hyp_n = normalize_text(hyp)
        absent = [t for t in terms if not contains_term(ref_n, t, normalized=True)]
        if absent:
            items_with_absent += 1
        opportunities += len(absent)
        inserted = [t for t in absent if contains_term(hyp_n, t, normalized=True)]
        insertions += len(inserted)
        for t in inserted:
            per_term[t] = per_term.get(t, 0) + 1
        if inserted:
            items_hit += 1
            details.append(ItemInsertion(item_id=item_id, inserted=inserted, hypothesis=hyp))
        wers.append(word_error_rate(ref, hyp))
    return InsertionStats(
        opportunities=opportunities,
        insertions=insertions,
        insertion_rate=insertions / opportunities if opportunities else 0.0,
        items_with_insertion=items_hit,
        item_insertion_rate=items_hit / items_with_absent if items_with_absent else 0.0,
        per_term=dict(sorted(per_term.items())),
        mean_wer=sum(wers) / len(wers) if wers else None,
        details=details,
    )


def measure_insertion_rate(
    decode: Callable[[T], str],
    items: Sequence[T],
    terms: CaseLexicon | Sequence[str],
    *,
    baseline_decode: Callable[[T], str] | None = None,
    reference: Callable[[T], str] = lambda item: item.text,  # type: ignore[attr-defined]
    item_id: Callable[[T, int], str] = _default_item_id,
) -> InsertionReport:
    """Measure how often lexicon terms appear where they are not said.

    For every item and every lexicon term absent from the item's reference text,
    an insertion is counted when the term appears in ``decode(item)``. Use items
    whose audio does not contain the terms (e.g. the self-assessment holdout).
    ``baseline_decode`` (the same model without lexicon) gives the insertion
    rate the lexicon is not responsible for. ``mean_wer`` is over all words, so
    the damage on the other words is visible too.
    """
    resolved, digest = _resolve_terms(terms)
    # one entry per normalised form (capitalisation variants are the same term)
    unique = list({normalize_text(t): t for t in resolved}.values())
    refs = [reference(it) for it in items]
    ids = [item_id(it, i) for i, it in enumerate(items)]
    with_stats = _insertion_stats([decode(it) for it in items], refs, ids, unique)
    without_stats = (
        _insertion_stats([baseline_decode(it) for it in items], refs, ids, unique)
        if baseline_decode is not None
        else None
    )
    n_absent = sum(
        1 for r in refs if any(not contains_term(normalize_text(r), t, normalized=True) for t in unique)
    )
    return InsertionReport(
        lexicon_sha256=digest,
        n_items=len(items),
        n_items_with_absent_terms=n_absent,
        n_terms=len(unique),
        with_lexicon=with_stats,
        without_lexicon=without_stats,
    )
