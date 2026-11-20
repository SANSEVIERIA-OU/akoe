"""Per-case self-assessment (IMPLEMENTATION_PLAN M5, issue #28).

The product measures itself on each case: held-out transcribed material of
the subject (DatasetItems with split HOLDOUT) is optionally degraded with the
case's channel simulation (`degrade(audio, sr, rng)`, supplied by the
caller), then passed through the same backends used on the case. Every
metric is reported overall and per recording condition
(`RecordingCondition.key()`), so the operator sees where the subject model
holds and where it does not.

Only the protocols in `forensicaudio.interfaces` are used. The result is a
`SelfAssessment` (written as `self_assessment.json`) and a short Italian
summary rendered from it. When nothing can be computed, the assessment says
why instead of being silently absent.

Metric keys in `overall` and in each `ConditionAssessment.metrics` are
`<metric>@<system>` (e.g. `wer@generic`, `voiceprint_eer@ecapa`), except
the shared counters `n_items`, `n_ref_words` and the marker `degraded`
(1.0 if the channel simulation was applied).
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf

from forensicaudio.assess.metrics import (
    DEFAULT_NORMALIZATION,
    CandidateTrial,
    EditCounts,
    LexiconInsertionResult,
    NormalizationConfig,
    ReliabilityBin,
    candidate_separation,
    char_error_counts,
    cosine_similarity,
    equal_error_rate,
    expected_calibration_error,
    label_hypothesis_words,
    lexicon_insertion_rate,
    oracle_wer,
    reliability_diagram,
    unintelligible_quality,
    word_error_counts,
    words,
)
from forensicaudio.config import StageKind
from forensicaudio.interfaces import ASRBackend, CandidateScorer, VoiceprintBackend
from forensicaudio.models import (
    ArtifactRef,
    ConditionAssessment,
    DatasetItem,
    DatasetSplit,
    SegmentHypotheses,
    SelfAssessment,
)
from forensicaudio.transcribe.combine import (
    DEFAULT_COMBINE,
    CombineConfig,
    HypothesisSource,
    combine_hypotheses,
    join_nbest,
)
from forensicaudio.transcribe.format import DEFAULT_FORMAT, FormatConfig, is_unintelligible
from forensicaudio.utils import sha256_file

DegradeFn = Callable[[np.ndarray, int, np.random.Generator], np.ndarray]
AudioLoader = Callable[[DatasetItem], tuple[np.ndarray, int]]
AlterFn = Callable[[str, np.random.Generator], list[str]]

ROVER_LABEL = "rover"
SELF_ASSESSMENT_KIND = "self_assessment"
SUMMARY_KIND = "self_assessment_summary"

LIMITATION_HOLDOUT = (
    "Il materiale messo da parte non è identico ai soliloqui in auto (sforzo vocale, contesto): "
    "l'autovalutazione è una stima su materiale simile, non una garanzia."
)
NOT_COMPUTABLE = "Autovalutazione non calcolabile"


@dataclass
class HoldoutSample:
    """A held-out item with its audio already loaded (mono float32)."""

    item: DatasetItem
    audio: np.ndarray
    sr: int


@dataclass(frozen=True)
class SelfCheckConfig:
    n_best: int = 5
    seed: int = 0
    normalization: NormalizationConfig = DEFAULT_NORMALIZATION
    ece_bins: int = 10
    ece_strategy: Literal["uniform", "quantile"] = "uniform"
    unintelligible: FormatConfig = DEFAULT_FORMAT
    # An item "deserves" the unintelligible label when its WER is at least this.
    unintelligible_wer_threshold: float = 0.5
    n_altered: int = 3
    require_verified: bool = True
    verify_hashes: bool = True
    combine: bool = True
    combine_config: CombineConfig = DEFAULT_COMBINE
    n_bootstrap: int = 1000
    min_items: int = 20
    min_words: int = 200


@dataclass
class ItemResult:
    item_id: str
    condition_key: str
    system: str
    reference: str
    hypothesis: str
    nbest: list[str]
    counts: EditCounts
    char_counts: EditCounts
    word_confidences: list[float | None]
    word_correct: list[bool | None]
    unintelligible: bool

    @property
    def wer(self) -> float | None:
        return self.counts.rate


@dataclass
class SelfCheckResult:
    assessment: SelfAssessment
    summary: str
    computable: bool
    items: list[ItemResult] = field(default_factory=list)
    reliability: dict[str, list[ReliabilityBin]] = field(default_factory=dict)
    lexicon: dict[str, LexiconInsertionResult] = field(default_factory=dict)
    candidate_trials: list[CandidateTrial] = field(default_factory=list)


# --------------------------------------------------------------------------
# Audio and randomness
# --------------------------------------------------------------------------


class HashMismatch(Exception):
    pass


class FileAudioLoader:
    """Loads `item.audio` (relative to `root`), slices [start, end], mixes to
    mono float32, and checks the file's SHA-256 against the ArtifactRef."""

    def __init__(self, root: Path | None = None, verify: bool = True) -> None:
        self.root = root
        self.verify = verify
        self._digests: dict[Path, str] = {}  # each file is hashed once

    def __call__(self, item: DatasetItem) -> tuple[np.ndarray, int]:
        path = item.audio.resolve(self.root)
        if self.verify:
            if path not in self._digests:
                self._digests[path] = sha256_file(path)
            if self._digests[path] != item.audio.sha256:
                raise HashMismatch(str(path))
        sr = sf.info(str(path)).samplerate
        start = max(0, int(round(item.start * sr)))
        stop = int(round(item.end * sr)) if item.end > item.start else None
        audio, sr = sf.read(str(path), start=start, stop=stop, dtype="float32", always_2d=True)
        return audio.mean(axis=1).astype(np.float32), int(sr)


def item_rng(seed: int, item_id: str, salt: str = "") -> np.random.Generator:
    """Deterministic per-item generator, independent of item order."""
    h = int.from_bytes(hashlib.sha256(f"{salt}:{item_id}".encode()).digest()[:8], "big")
    return np.random.default_rng([seed, h])


def make_altered_phrases(
    text: str, vocabulary: Sequence[str], rng: np.random.Generator, n: int = 3
) -> list[str]:
    """Up to `n` distinct altered versions of `text`: one word substituted,
    deleted, inserted, or two adjacent words swapped."""
    toks = text.split()
    vocab = sorted(set(vocabulary))
    target = " ".join(words(text))
    out: list[str] = []
    for _ in range(20 * max(n, 1)):
        if len(out) >= n or not toks:
            break
        t = list(toks)
        op = rng.integers(4)
        i = int(rng.integers(len(t)))
        if op == 0 and vocab:
            choices = [v for v in vocab if v.lower() != t[i].lower()]
            if not choices:
                continue
            t[i] = choices[int(rng.integers(len(choices)))]
        elif op == 1 and len(t) > 1:
            del t[i]
        elif op == 2 and vocab:
            t.insert(int(rng.integers(len(t) + 1)), vocab[int(rng.integers(len(vocab)))])
        elif op == 3 and len(t) > 1:
            j = min(i, len(t) - 2)
            t[j], t[j + 1] = t[j + 1], t[j]
        else:
            continue
        cand = " ".join(t)
        norm = " ".join(words(cand))
        if norm and norm != target and cand not in out:
            out.append(cand)
    return out


# --------------------------------------------------------------------------
# Preparation
# --------------------------------------------------------------------------


@dataclass
class _Prepared:
    samples: list[HoldoutSample]
    notes: list[str]


def _prepare(
    items: Sequence[DatasetItem | HoldoutSample],
    loader: AudioLoader,
    cfg: SelfCheckConfig,
) -> _Prepared:
    samples: list[HoldoutSample] = []
    excluded = {"duplicate": 0, "train": 0, "unverified": 0, "empty_text": 0, "hash": 0, "missing": 0}
    no_split = 0
    seen: set[str] = set()
    for x in items:
        item = x.item if isinstance(x, HoldoutSample) else x
        if item.item_id in seen:
            excluded["duplicate"] += 1
            continue
        seen.add(item.item_id)
        if item.split == DatasetSplit.TRAIN:
            excluded["train"] += 1
            continue
        if cfg.require_verified and not item.verified:
            excluded["unverified"] += 1
            continue
        if not words(item.text, cfg.normalization):
            excluded["empty_text"] += 1
            continue
        if isinstance(x, HoldoutSample):
            sample = x
        else:
            try:
                audio, sr = loader(item)
            except HashMismatch:
                excluded["hash"] += 1
                continue
            except (FileNotFoundError, sf.LibsndfileError, RuntimeError):
                excluded["missing"] += 1
                continue
            sample = HoldoutSample(item, audio, sr)
        no_split += item.split is None
        samples.append(sample)

    notes = []
    msgs = {
        "duplicate": "{n} elementi esclusi perché il loro identificativo è ripetuto.",
        "train": "{n} elementi esclusi perché usati per l'addestramento (split TRAIN).",
        "unverified": "{n} elementi esclusi perché il testo non è verificato.",
        "empty_text": "{n} elementi esclusi perché senza testo di riferimento.",
        "hash": "{n} elementi esclusi perché l'audio non corrisponde allo SHA-256 registrato.",
        "missing": "{n} elementi esclusi perché l'audio non è leggibile.",
    }
    for k, n in excluded.items():
        if n:
            notes.append(msgs[k].format(n=n))
    if no_split:
        notes.append(
            f"{no_split} elementi senza split dichiarato: non è verificabile che non siano "
            "stati usati per l'addestramento."
        )
    return _Prepared(samples, notes)


def _systems(asr: Mapping[str, ASRBackend] | Sequence[ASRBackend] | None) -> list[tuple[str, ASRBackend]]:
    if asr is None:
        return []
    if isinstance(asr, Mapping):
        return list(asr.items())
    out: list[tuple[str, ASRBackend]] = []
    used: set[str] = set()
    for b in asr:
        label, k = b.name, 2
        while label in used:
            label, k = f"{b.name}#{k}", k + 1
        used.add(label)
        out.append((label, b))
    return out


def _merge(segs: Sequence[SegmentHypotheses], dur: float) -> SegmentHypotheses:
    ordered = sorted(segs, key=lambda s: (s.start, s.end))
    return SegmentHypotheses(
        start=0.0,
        end=dur,
        audio_provenance=StageKind.NEURAL,
        hypotheses=join_nbest(ordered),
        words=[w for s in ordered for w in s.words],
    )


# --------------------------------------------------------------------------
# Metrics per group
# --------------------------------------------------------------------------


def _bootstrap_ci(results: Sequence[ItemResult], n_boot: int, seed: int) -> tuple[float, float] | None:
    if n_boot <= 0 or len(results) < 2:
        return None
    err = np.array([r.counts.errors for r in results], dtype=np.float64)
    ref = np.array([r.counts.n_ref for r in results], dtype=np.float64)
    rng = np.random.default_rng([seed, 1])
    idx = rng.integers(0, len(results), size=(n_boot, len(results)))
    den = ref[idx].sum(axis=1)
    ok = den > 0
    if not ok.any():
        return None
    rates = err[idx].sum(axis=1)[ok] / den[ok]
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


def _asr_metrics(
    results: Sequence[ItemResult],
    lexicon: Sequence[str],
    cfg: SelfCheckConfig,
) -> tuple[dict[str, float], list[ReliabilityBin], LexiconInsertionResult | None]:
    m: dict[str, float] = {}
    wc = sum((r.counts for r in results), EditCounts())
    cc = sum((r.char_counts for r in results), EditCounts())
    if wc.rate is not None:
        m["wer"] = wc.rate
        m["sub_rate"] = wc.substitutions / wc.n_ref
        m["del_rate"] = wc.deletions / wc.n_ref
        m["ins_rate"] = wc.insertions / wc.n_ref
    if cc.rate is not None:
        m["cer"] = cc.rate
    ci = _bootstrap_ci(results, cfg.n_bootstrap, cfg.seed)
    if ci is not None:
        m["wer_ci_low"], m["wer_ci_high"] = ci

    refs = [r.reference for r in results]
    orc = oracle_wer(refs, [r.nbest for r in results], cfg.normalization)
    if orc.oracle_wer is not None:
        m["oracle_wer"] = orc.oracle_wer
    if orc.exact_coverage is not None:
        m["nbest_coverage"] = orc.exact_coverage
    m["nbest_mean_size"] = float(np.mean([len(r.nbest) for r in results])) if results else 0.0

    confs = [c for r in results for c in r.word_confidences]
    labels = [k for r in results for k in r.word_correct]
    pairs = [(c, k) for c, k in zip(confs, labels) if c is not None and k is not None]
    bins: list[ReliabilityBin] = []
    if pairs:
        cs, ks = [p[0] for p in pairs], [p[1] for p in pairs]
        ece = expected_calibration_error(cs, ks, cfg.ece_bins, cfg.ece_strategy)
        if ece is not None:
            m["ece"] = ece
        m["n_conf_words"] = float(len(pairs))
        bins = reliability_diagram(cs, ks, cfg.ece_bins, cfg.ece_strategy)

    uq = unintelligible_quality(
        [r.unintelligible for r in results],
        [r.wer if r.wer is not None else 0.0 for r in results],
        cfg.unintelligible_wer_threshold,
    )
    for k, v in (("unintelligible_rate", uq.flag_rate), ("unintelligible_precision", uq.precision),
                 ("unintelligible_recall", uq.recall)):
        if v is not None:
            m[k] = v

    lex = None
    if lexicon:
        lex = lexicon_insertion_rate(refs, [r.hypothesis for r in results], lexicon, cfg.normalization)
        if lex.rate is not None:
            m["lexicon_insertion_rate"] = lex.rate
        if lex.item_rate is not None:
            m["lexicon_item_rate"] = lex.item_rate
        m["lexicon_opportunities"] = float(lex.n_opportunities)
    return m, bins, lex


def _suffix(metrics: Mapping[str, float], label: str) -> dict[str, float]:
    return {f"{k}@{label}": float(v) for k, v in metrics.items()}


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def _not_computable(
    case_id: str, models: list[str], snr_db: float | None, reasons: list[str]
) -> SelfCheckResult:
    sa = SelfAssessment(
        case_id=case_id,
        models=models,
        snr_db=snr_db,
        overall={"n_items": 0.0},
        limitations=reasons + [LIMITATION_HOLDOUT],
    )
    return SelfCheckResult(assessment=sa, summary=render_summary_it(sa), computable=False)


def run_selfcheck(
    case_id: str,
    items: Sequence[DatasetItem | HoldoutSample],
    asr: Mapping[str, ASRBackend] | Sequence[ASRBackend] | None = None,
    *,
    scorer: CandidateScorer | None = None,
    voiceprint: VoiceprintBackend | None = None,
    enrollment: Sequence[DatasetItem | HoldoutSample] = (),
    impostors: Sequence[DatasetItem | HoldoutSample] = (),
    degrade: DegradeFn | None = None,
    snr_db: float | None = None,
    lexicon: Sequence[str] = (),
    alter: AlterFn | None = None,
    root: Path | None = None,
    audio_loader: AudioLoader | None = None,
    config: SelfCheckConfig | None = None,
) -> SelfCheckResult:
    """Run the self-assessment of one case.

    items: held-out subject material with verified text (split TRAIN is
        always excluded); DatasetItems are loaded with `audio_loader`
        (default: `FileAudioLoader(root)`, which checks SHA-256).
    asr: backends to assess; a mapping gives explicit labels (e.g.
        {"generic": ..., "adapted": ...}), otherwise labels are backend names.
        With two or more backends the ROVER combination is assessed too.
    scorer: candidate-phrase scorer, assessed on true vs altered phrases.
    voiceprint, enrollment, impostors: target trials are held-out items
        against the centroid of `enrollment` (not degraded); non-target
        trials are `impostors` (other speakers, degraded like the items).
    degrade: channel/codec simulation applied to items and impostors, with a
        per-item deterministic generator; `snr_db` is recorded as given.
    """
    cfg = config or SelfCheckConfig()
    systems = _systems(asr)
    models = list(dict.fromkeys([b.name for _, b in systems]
                                + ([scorer.name] if scorer else [])
                                + ([voiceprint.name] if voiceprint else [])))
    loader = audio_loader or FileAudioLoader(root, verify=cfg.verify_hashes)

    if not items:
        return _not_computable(case_id, models, snr_db, [
            f"{NOT_COMPUTABLE}: nessun materiale trascritto del soggetto messo da parte "
            "per l'autovalutazione."
        ])
    prep = _prepare(items, loader, cfg)
    if not prep.samples:
        return _not_computable(case_id, models, snr_db, [
            f"{NOT_COMPUTABLE}: nessun elemento utilizzabile tra quelli messi da parte."
        ] + prep.notes)
    if not systems and scorer is None and voiceprint is None:
        return _not_computable(case_id, models, snr_db, [
            f"{NOT_COMPUTABLE}: nessun trascrittore, punteggio di candidate o impronta vocale fornito."
        ] + prep.notes)

    limitations: list[str] = list(prep.notes)

    def degraded(s: HoldoutSample, salt: str) -> HoldoutSample:
        if degrade is None:
            return s
        out = degrade(s.audio, s.sr, item_rng(cfg.seed, s.item.item_id, salt))
        return HoldoutSample(s.item, np.asarray(out, dtype=np.float32).ravel(), s.sr)

    samples = [degraded(s, "degrade") for s in prep.samples]
    cond_of = {s.item.item_id: s.item.condition.key() for s in samples}

    # -- transcription -----------------------------------------------------
    outputs: dict[str, list[list[SegmentHypotheses]]] = {}
    for label, backend in systems:
        try:
            outs = []
            for s in samples:
                dur = len(s.audio) / s.sr
                outs.append(backend.transcribe(s.audio, s.sr, segments=[(0.0, dur)], n_best=cfg.n_best))
            outputs[label] = outs
        except NotImplementedError as e:
            limitations.append(f"Trascrittore {label} non disponibile: {e or 'non implementato'}.")
    asr_labels = list(outputs)
    if cfg.combine and len(asr_labels) >= 2:
        outputs[ROVER_LABEL] = [
            combine_hypotheses(
                [HypothesisSource(segments=outputs[lab][i], origin=lab) for lab in asr_labels],
                cfg.combine_config,
            )
            for i in range(len(samples))
        ]

    results: list[ItemResult] = []
    for label, outs in outputs.items():
        for s, segs in zip(samples, outs):
            merged = _merge(segs, len(s.audio) / s.sr)
            hyp = merged.hypotheses[0].text if merged.hypotheses else ""
            wtexts = [w.text for w in merged.words]
            results.append(ItemResult(
                item_id=s.item.item_id,
                condition_key=cond_of[s.item.item_id],
                system=label,
                reference=s.item.text,
                hypothesis=hyp,
                nbest=[a.text for a in merged.hypotheses],
                counts=word_error_counts(s.item.text, hyp, cfg.normalization),
                char_counts=char_error_counts(s.item.text, hyp, cfg.normalization),
                word_confidences=[w.confidence for w in merged.words],
                word_correct=label_hypothesis_words(s.item.text, wtexts, cfg.normalization),
                unintelligible=is_unintelligible(merged, cfg.unintelligible),
            ))
    if systems and not outputs:
        limitations.append("Errori per parola non calcolati: nessun trascrittore disponibile.")

    # -- candidate phrases -------------------------------------------------
    trials: list[tuple[str, CandidateTrial]] = []
    if scorer is not None:
        vocab = [w for s in samples for w in s.item.text.split()]
        try:
            for s in samples:
                rng = item_rng(cfg.seed, s.item.item_id, "candidates")
                altered = alter(s.item.text, rng) if alter else make_altered_phrases(
                    s.item.text, vocab, rng, cfg.n_altered)
                if not altered:
                    continue
                cands = [s.item.text] + list(altered)
                order = rng.permutation(len(cands))
                scores = scorer.score(s.audio, s.sr, [cands[k] for k in order])
                by_cand = {int(k): float(sc) for k, sc in zip(order, scores)}
                trials.append((s.item.item_id, CandidateTrial(
                    by_cand[0], tuple(by_cand[k] for k in range(1, len(cands))))))
        except NotImplementedError as e:
            limitations.append(
                f"Punteggio delle candidate {scorer.name} non disponibile: {e or 'non implementato'}."
            )
            trials = []

    # -- voiceprint --------------------------------------------------------
    target: list[tuple[str, float]] = []
    nontarget: list[float] = []
    if voiceprint is not None:
        # Enrollment and impostors need no reference text.
        enr = _load_any(enrollment, loader)
        imp = [degraded(s, "impostor") for s in _load_any(impostors, loader)]
        if not enr:
            limitations.append("Impronta vocale non valutata: manca materiale di riferimento del soggetto.")
        elif not imp:
            limitations.append(
                "Impronta vocale non valutata: manca materiale di altri parlanti per le prove di confronto."
            )
        else:
            try:
                embs = [np.asarray(voiceprint.embed(s.audio, s.sr), dtype=np.float64) for s in enr]
                centroid = np.mean([e / (np.linalg.norm(e) or 1.0) for e in embs], axis=0)
                enr_ids = {s.item.item_id for s in enr}
                overlap = 0
                for s in samples:
                    if s.item.item_id in enr_ids:
                        overlap += 1
                        continue
                    score = cosine_similarity(voiceprint.embed(s.audio, s.sr), centroid)
                    target.append((s.item.item_id, score))
                nontarget = [cosine_similarity(voiceprint.embed(s.audio, s.sr), centroid) for s in imp]
                if overlap:
                    limitations.append(
                        f"{overlap} elementi esclusi dalle prove dell'impronta perché usati "
                        "anche come riferimento."
                    )
            except NotImplementedError as e:
                limitations.append(
                    f"Impronta vocale {voiceprint.name} non disponibile: {e or 'non implementata'}."
                )
                target, nontarget = [], []

    # -- assemble ----------------------------------------------------------
    reliability: dict[str, list[ReliabilityBin]] = {}
    lex_results: dict[str, LexiconInsertionResult] = {}

    def group_metrics(item_ids: set[str], *, keep_details: bool) -> dict[str, float]:
        group = [s for s in samples if s.item.item_id in item_ids]
        m: dict[str, float] = {
            "n_items": float(len(group)),
            "n_ref_words": float(sum(len(words(s.item.text, cfg.normalization)) for s in group)),
            "degraded": 1.0 if degrade is not None else 0.0,
        }
        for label in outputs:
            rs = [r for r in results if r.system == label and r.item_id in item_ids]
            if not rs:
                continue
            am, bins, lex = _asr_metrics(rs, lexicon, cfg)
            m.update(_suffix(am, label))
            if keep_details:
                reliability[label] = bins
                if lex is not None:
                    lex_results[label] = lex
        if scorer is not None:
            ts = [t for iid, t in trials if iid in item_ids]
            sep = candidate_separation(ts)
            if sep.n_trials:
                cm = {"candidate_top1": sep.top1_rate, "candidate_tie_rate": sep.tie_rate,
                      "candidate_auc": sep.pairwise_auc, "candidate_margin": sep.mean_margin,
                      "n_candidate_trials": float(sep.n_trials)}
                m.update(_suffix({k: v for k, v in cm.items() if v is not None}, scorer.name))
        if voiceprint is not None and nontarget:
            tg = [sc for iid, sc in target if iid in item_ids]
            if tg:
                e = equal_error_rate(tg, nontarget)
                m.update(_suffix({"voiceprint_eer": e.eer, "voiceprint_threshold": e.threshold,
                                  "n_target_trials": float(e.n_target),
                                  "n_nontarget_trials": float(e.n_nontarget)}, voiceprint.name))
        return m

    all_ids = {s.item.item_id for s in samples}
    overall = group_metrics(all_ids, keep_details=True)
    by_condition = []
    for key in sorted(set(cond_of.values())):
        ids = {iid for iid, k in cond_of.items() if k == key}
        by_condition.append(ConditionAssessment(condition_key=key, n_items=len(ids),
                                                metrics=group_metrics(ids, keep_details=False)))

    if degrade is None:
        limitations.append(
            "Il materiale non è stato degradato con la simulazione del canale del caso: "
            "la stima è probabilmente ottimistica."
        )
    if len(samples) < cfg.min_items or overall["n_ref_words"] < cfg.min_words:
        limitations.append(
            f"Stima basata su pochi dati ({len(samples)} elementi, {int(overall['n_ref_words'])} parole): "
            "l'incertezza è alta."
        )
    limitations.append(LIMITATION_HOLDOUT)

    sa = SelfAssessment(
        case_id=case_id,
        models=models,
        snr_db=snr_db,
        overall=overall,
        by_condition=by_condition,
        limitations=limitations,
    )
    return SelfCheckResult(
        assessment=sa,
        summary=render_summary_it(sa),
        computable=True,
        items=results,
        reliability=reliability,
        lexicon=lex_results,
        candidate_trials=[t for _, t in trials],
    )


def _load_any(items: Sequence[DatasetItem | HoldoutSample], loader: AudioLoader) -> list[HoldoutSample]:
    """Load audio without text requirements (enrollment, impostors); unreadable
    or tampered files are skipped."""
    out = []
    for x in items:
        if isinstance(x, HoldoutSample):
            out.append(x)
            continue
        try:
            audio, sr = loader(x)
        except (HashMismatch, FileNotFoundError, sf.LibsndfileError, RuntimeError):
            continue
        out.append(HoldoutSample(x, audio, sr))
    return out


# --------------------------------------------------------------------------
# Italian summary
# --------------------------------------------------------------------------

_CHANNEL_IT = {"car": "auto", "phone": "telefono", "room": "stanza", "other": "altro canale",
               "unknown": "canale sconosciuto"}
_SOURCE_IT = {"local": "registrazione locale", "transmitted": "registrazione trasmessa",
              "other": "altra fonte"}
_EFFORT_IT = {"normal": "voce normale", "low": "voce bassa", "raised": "voce alzata",
              "whispered": "voce sussurrata", "unknown": "sforzo vocale sconosciuto"}


def describe_condition_it(key: str) -> str:
    """Italian description of a `RecordingCondition.key()`."""
    parts = key.split("/")
    if len(parts) != 4:
        return key
    ch, src, codec, effort = parts
    codec_it = "codec sconosciuto" if codec == "unknown-codec" else f"codec {codec}"
    return ", ".join(
        [_CHANNEL_IT.get(ch, ch), _SOURCE_IT.get(src, src), codec_it, _EFFORT_IT.get(effort, effort)]
    )


def _pct(x: float) -> int:
    return int(round(100 * x))


def _num(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def _systems_in(metrics: Mapping[str, float], prefix: str) -> list[str]:
    return [k.split("@", 1)[1] for k in metrics if k.startswith(prefix + "@")]


def _wer_phrase(wer: float) -> str:
    if wer <= 1.0:
        return f"circa {_pct(wer)} parole su 100 sono sbagliate"
    return f"gli errori superano le parole (circa {_pct(wer)} errori ogni 100 parole)"


def render_summary_it(sa: SelfAssessment) -> str:
    """Short Italian summary to attach to the transcript."""
    o = sa.overall
    lines: list[str] = []
    if not o.get("n_items"):
        lines.append(f"Autovalutazione del caso {sa.case_id}: non calcolata.")
        lines += [f"- {x}" for x in sa.limitations]
        return "\n".join(lines)

    head = (f"Autovalutazione del caso {sa.case_id} su {int(o['n_items'])} tratti messi da parte "
            f"({int(o.get('n_ref_words', 0))} parole)")
    if o.get("degraded"):
        head += ", degradati con la simulazione del canale del caso"
        if sa.snr_db is not None:
            head += f" (rapporto segnale/rumore stimato {_num(sa.snr_db, 0)} dB)"
    lines.append(head + ". È una stima, non una garanzia.")

    for label in _systems_in(o, "wer"):
        wer = o[f"wer@{label}"]
        line = f"- {label}: su materiale simile, in queste condizioni, {_wer_phrase(wer)}"
        lo, hi = o.get(f"wer_ci_low@{label}"), o.get(f"wer_ci_high@{label}")
        if lo is not None and hi is not None:
            line += f" (intervallo indicativo {_pct(lo)}-{_pct(hi)})"
        lines.append(line + ".")
        orc, cov = o.get(f"oracle_wer@{label}"), o.get(f"nbest_coverage@{label}")
        if orc is not None and o.get(f"nbest_mean_size@{label}", 1.0) > 1.0:
            extra = ("  Scegliendo la migliore tra le alternative proposte si scende a circa "
                     f"{_pct(orc)} su 100")
            if cov is not None:
                extra += f"; la frase giusta compare tra le alternative in {_pct(cov)} casi su 100"
            lines.append(extra + ".")
        ece = o.get(f"ece@{label}")
        if ece is not None:
            lines.append(f"  La confidenza dichiarata si discosta in media di circa {_pct(ece)} punti su 100 "
                         "dall'accuratezza osservata.")
        lex = o.get(f"lexicon_insertion_rate@{label}")
        if lex is not None:
            lines.append("  Termini del lessico scritti dove non erano stati pronunciati: "
                         f"{_pct(lex)} casi su 100.")

    if len(sa.by_condition) > 1 or (sa.by_condition and _systems_in(o, "wer")):
        lines.append("Per condizione:")
        for c in sa.by_condition:
            parts = [f"{lab} circa {_pct(c.metrics[f'wer@{lab}'])} su 100"
                     for lab in _systems_in(c.metrics, "wer")]
            detail = "; ".join(parts) if parts else "errori per parola non calcolati"
            n_tr = f"{c.n_items} {'tratto' if c.n_items == 1 else 'tratti'}"
            lines.append(f"- {describe_condition_it(c.condition_key)} ({n_tr}): {detail}.")

    for label in _systems_in(o, "candidate_top1"):
        line = (f"Frasi candidate ({label}): la frase giusta ottiene il punteggio più alto in "
                f"{_pct(o[f'candidate_top1@{label}'])} casi su 100")
        auc = o.get(f"candidate_auc@{label}")
        if auc is not None:
            line += f" (AUC {_num(auc)})"
        lines.append(line + ".")
    for label in _systems_in(o, "voiceprint_eer"):
        lines.append(f"Impronta vocale ({label}): tasso di errore a pari errore circa "
                     f"{_pct(o[f'voiceprint_eer@{label}'])} su 100. È un punteggio di compatibilità, "
                     "mai un'identificazione.")

    if sa.limitations:
        lines.append("Limiti:")
        lines += [f"- {x}" for x in sa.limitations]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_self_assessment(sa: SelfAssessment, path: Path, *, root: Path | None = None) -> ArtifactRef:
    """Write `self_assessment.json` and return its ArtifactRef."""
    _atomic_write(path, sa.model_dump_json(indent=2))
    return ArtifactRef.from_file(path, kind=SELF_ASSESSMENT_KIND, root=root)


def write_summary(sa: SelfAssessment, path: Path, *, root: Path | None = None) -> ArtifactRef:
    """Write the Italian summary (plain text) and return its ArtifactRef."""
    _atomic_write(path, render_summary_it(sa) + "\n")
    return ArtifactRef.from_file(path, kind=SUMMARY_KIND, root=root)
