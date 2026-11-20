import numpy as np
import pytest
import soundfile as sf

from forensicaudio.assess.selfcheck import (
    LIMITATION_HOLDOUT,
    FileAudioLoader,
    HoldoutSample,
    SelfCheckConfig,
    describe_condition_it,
    item_rng,
    make_altered_phrases,
    render_summary_it,
    run_selfcheck,
    write_self_assessment,
    write_summary,
)
from forensicaudio.config import Channel, RecordingSource, StageKind, VocalEffort
from forensicaudio.interfaces import ASRBackend
from forensicaudio.models import (
    Alternative,
    ArtifactRef,
    DatasetItem,
    DatasetSplit,
    RecordingCondition,
    SegmentHypotheses,
    SelfAssessment,
    WordHypothesis,
)
from tests.fakes import FakeScorer, FakeVoiceprint

SR = 16000
CAR = RecordingCondition(channel=Channel.CAR, source=RecordingSource.LOCAL, vocal_effort=VocalEffort.NORMAL)
PHONE = RecordingCondition(channel=Channel.PHONE, source=RecordingSource.OTHER, codec="amr-nb")
REFS = {
    1000: ("vado a casa domani", CAR),
    1100: ("ci vediamo al porto", CAR),
    1200: ("non lo so proprio", PHONE),
    1300: ("chiama Moretti stasera", PHONE),
}


class ScriptedASR:
    """Returns a fixed text per item, recognized by the audio length."""

    def __init__(self, name, texts, conf=0.9, n_best=None):
        self.name = name
        self.texts = texts  # length -> text
        self.conf = conf
        self.extra = n_best or {}  # length -> extra alternatives

    def transcribe(self, audio, sr, *, segments=None, n_best=5):
        text = self.texts[len(audio)]
        start, end = (segments or [(0.0, len(audio) / sr)])[0]
        toks = text.split()
        step = (end - start) / max(len(toks), 1)
        return [SegmentHypotheses(
            start=start, end=end, audio_provenance=StageKind.NEURAL,
            hypotheses=[Alternative(text=text, origin=self.name)]
            + [Alternative(text=t, origin=self.name) for t in self.extra.get(len(audio), ())][: n_best - 1],
            words=[WordHypothesis(text=w, start=start + i * step, end=start + (i + 1) * step,
                                  confidence=self.conf, origin=self.name) for i, w in enumerate(toks)],
        )]


class BrokenASR:
    name = "broken"

    def transcribe(self, audio, sr, *, segments=None, n_best=5):
        raise NotImplementedError("pesi non installati")


def tone(freq, n, seed=0, sr=SR):
    t = np.arange(n) / sr
    rng = np.random.default_rng(seed)
    return (0.5 * np.sin(2 * np.pi * freq * t) + 0.01 * rng.normal(size=n)).astype(np.float32)


def sample(n, text=None, cond=None, item_id=None, split=DatasetSplit.HOLDOUT, verified=True, freq=300.0):
    ref_text, ref_cond = REFS.get(n, ("", CAR))
    item = DatasetItem(
        item_id=item_id or f"it{n}", recording_id="rec", audio=ArtifactRef(path="x.wav", sha256="0" * 64, kind="audio"),
        start=0.0, end=n / SR, text=text if text is not None else ref_text, condition=cond or ref_cond,
        verified=verified, split=split,
    )
    return HoldoutSample(item, tone(freq, n, seed=n), SR)


def holdout():
    return [sample(n) for n in REFS]


def perfect():
    return ScriptedASR("generic", {n: t for n, (t, _) in REFS.items()})


def biased():
    texts = {n: t for n, (t, _) in REFS.items()}
    texts[1000] = "vado da Moretti domani"
    texts[1200] = "non lo so"
    return ScriptedASR("adapted", texts, conf=0.6)


FAST = SelfCheckConfig(n_bootstrap=200)


def test_fakes_satisfy_protocol():
    assert isinstance(perfect(), ASRBackend) and isinstance(BrokenASR(), ASRBackend)


# -- not computable ------------------------------------------------------------


def test_not_computable_without_material():
    r = run_selfcheck("case-1", [], [perfect()])
    assert not r.computable
    sa = r.assessment
    assert sa.overall == {"n_items": 0.0} and sa.by_condition == []
    assert "nessun materiale trascritto" in sa.limitations[0]
    assert LIMITATION_HOLDOUT in sa.limitations
    assert "non calcolata" in r.summary and "nessun materiale trascritto" in r.summary


def test_not_computable_when_all_items_excluded():
    items = [sample(1000, split=DatasetSplit.TRAIN), sample(1100, verified=False), sample(1200, text="  ...  ")]
    r = run_selfcheck("case-1", items, [perfect()])
    assert not r.computable
    text = " ".join(r.assessment.limitations)
    assert "addestramento" in text and "non è verificato" in text and "senza testo" in text


def test_duplicate_item_ids_are_excluded_and_declared():
    r = run_selfcheck("c", holdout() + [sample(1000)], [perfect()], config=FAST)
    assert r.assessment.overall["n_items"] == 4
    assert any("identificativo è ripetuto" in x for x in r.assessment.limitations)


def test_not_computable_without_any_backend():
    r = run_selfcheck("case-1", holdout(), None)
    assert not r.computable and "nessun trascrittore" in r.assessment.limitations[0]


# -- WER per condition ---------------------------------------------------------


def test_metrics_overall_and_by_condition():
    r = run_selfcheck("case-1", holdout(), {"generic": perfect(), "adapted": biased()}, lexicon=["Moretti"],
                      config=FAST)
    assert r.computable
    o = r.assessment.overall
    assert o["n_items"] == 4 and o["n_ref_words"] == 15
    assert o["wer@generic"] == 0.0 and o["cer@generic"] == 0.0
    # adapted: 1 substitution (a->da) + 1 substitution (casa->Moretti) + 1 deletion (proprio) over 15 words
    assert o["wer@adapted"] == pytest.approx(3 / 15)
    assert o["del_rate@adapted"] == pytest.approx(1 / 15)
    assert o["wer_ci_low@adapted"] <= o["wer@adapted"] <= o["wer_ci_high@adapted"]
    # ROVER is assessed when there are two transcribers
    assert "wer@rover" in o
    # lexicon bias test: Moretti inserted in 1 of the 3 items where it was not said
    assert o["lexicon_insertion_rate@adapted"] == pytest.approx(1 / 3)
    assert o["lexicon_insertion_rate@generic"] == 0.0
    # calibration: generic says 0.9 and is always right
    assert o["ece@generic"] == pytest.approx(0.1)

    by = {c.condition_key: c for c in r.assessment.by_condition}
    assert set(by) == {CAR.key(), PHONE.key()}
    assert by[CAR.key()].n_items == 2
    assert by[CAR.key()].metrics["wer@adapted"] == pytest.approx(2 / 8)
    assert by[PHONE.key()].metrics["wer@adapted"] == pytest.approx(1 / 7)
    assert r.assessment.models == ["generic", "adapted"]
    assert set(r.reliability) == {"generic", "adapted", "rover"}
    assert r.lexicon["adapted"].per_term == {"Moretti": (1, 3)}


def test_oracle_and_coverage_use_nbest():
    asr = ScriptedASR("nb", {1000: "vado a cassa domani", 1100: "ci vediamo al porto"},
                      n_best={1000: ("vado a casa domani",)})
    r = run_selfcheck("c", [sample(1000), sample(1100)], [asr], config=FAST)
    o = r.assessment.overall
    assert o["wer@nb"] == pytest.approx(1 / 8)
    assert o["oracle_wer@nb"] == 0.0 and o["nbest_coverage@nb"] == 1.0
    assert "alternative" in r.summary


def test_unintelligible_rate():
    mumble = ScriptedASR("mumble", {1000: "boh", 1100: "ci vediamo al porto"}, conf=0.1)
    r = run_selfcheck("c", [sample(1000), sample(1100)], [mumble], config=FAST)
    o = r.assessment.overall
    assert o["unintelligible_rate@mumble"] == 1.0
    assert o["unintelligible_precision@mumble"] == pytest.approx(0.5)
    assert o["unintelligible_recall@mumble"] == 1.0


def test_unavailable_backend_is_reported_not_fatal():
    r = run_selfcheck("c", holdout(), [perfect(), BrokenASR()], config=FAST)
    assert r.computable
    assert "wer@generic" in r.assessment.overall and "wer@rover" not in r.assessment.overall
    assert any("broken" in x and "pesi non installati" in x for x in r.assessment.limitations)


# -- degradation and reproducibility -------------------------------------------


def test_degrade_is_applied_deterministically():
    seen: dict[int, float] = {}

    def degrade(audio, sr, rng):
        seen[len(audio)] = float(rng.random())
        return audio + 0.001 * rng.normal(size=audio.shape).astype(np.float32)

    r1 = run_selfcheck("c", holdout(), [perfect()], degrade=degrade, snr_db=5.0, config=FAST)
    first = dict(seen)
    r2 = run_selfcheck("c", list(reversed(holdout())), [perfect()], degrade=degrade, snr_db=5.0, config=FAST)
    assert seen == first  # same generator per item, whatever the order
    assert r1.assessment.overall == r2.assessment.overall
    assert r1.assessment.overall["degraded"] == 1.0 and r1.assessment.snr_db == 5.0
    assert "simulazione del canale" in r1.summary and "5 dB" in r1.summary
    assert not any("non è stato degradato" in x for x in r1.assessment.limitations)


def test_without_degrade_the_estimate_is_declared_optimistic():
    r = run_selfcheck("c", holdout(), [perfect()], config=FAST)
    assert r.assessment.overall["degraded"] == 0.0
    assert any("ottimistica" in x for x in r.assessment.limitations)
    assert any("pochi dati" in x for x in r.assessment.limitations)
    assert r.assessment.limitations[-1] == LIMITATION_HOLDOUT


# -- candidates and voiceprint -------------------------------------------------


class TextOracleScorer:
    """Knows the true text of each item (by audio length): a perfect scorer."""

    name = "oracle-scorer"

    def score(self, audio, sr, candidates):
        truth = REFS[len(audio)][0]
        return [1.0 if c == truth else 0.0 for c in candidates]


def test_candidate_separation_metrics():
    r = run_selfcheck("c", holdout(), scorer=TextOracleScorer(), config=FAST)
    o = r.assessment.overall
    assert o["candidate_top1@oracle-scorer"] == 1.0 and o["candidate_auc@oracle-scorer"] == 1.0
    assert o["n_candidate_trials@oracle-scorer"] == 4
    assert "Frasi candidate" in r.summary
    # a scorer blind to word order cannot separate swapped phrases: ties lower the AUC
    r2 = run_selfcheck("c", holdout(), scorer=FakeScorer("vado a casa domani"), config=FAST)
    assert r2.assessment.overall["candidate_auc@fake-scorer"] < 1.0


def test_voiceprint_eer():
    enrollment = [sample(900, text="", item_id="enr1"), sample(950, text="", item_id="enr2")]
    impostors = [sample(1000 + k, text="", item_id=f"imp{k}", freq=3000.0) for k in (7, 8, 9)]
    r = run_selfcheck("c", holdout(), voiceprint=FakeVoiceprint(), enrollment=enrollment, impostors=impostors,
                      config=FAST)
    o = r.assessment.overall
    assert o["voiceprint_eer@fake-voiceprint"] == 0.0
    assert o["n_target_trials@fake-voiceprint"] == 4 and o["n_nontarget_trials@fake-voiceprint"] == 3
    assert "mai un'identificazione" in r.summary
    by = r.assessment.by_condition[0].metrics
    assert "voiceprint_eer@fake-voiceprint" in by


def test_voiceprint_without_impostors_is_declared():
    r = run_selfcheck("c", holdout(), [perfect()], voiceprint=FakeVoiceprint(),
                      enrollment=[sample(900, text="", item_id="enr")], config=FAST)
    assert not any(k.startswith("voiceprint_eer") for k in r.assessment.overall)
    assert any("altri parlanti" in x for x in r.assessment.limitations)


# -- loading from files --------------------------------------------------------


def test_file_loader_slices_and_checks_hash(tmp_path):
    rec = tmp_path / "evidence" / "rec.wav"
    rec.parent.mkdir()
    sf.write(rec, np.zeros((2 * SR, 2), dtype=np.float32), SR)  # stereo: mixed to mono
    ok_ref = ArtifactRef.from_file(rec, kind="audio", root=tmp_path)
    item = DatasetItem(item_id="a", recording_id="rec", audio=ok_ref, start=0.5, end=1.5, text="vado a casa",
                       verified=True, split=DatasetSplit.HOLDOUT)
    audio, sr = FileAudioLoader(tmp_path)(item)
    assert sr == SR and audio.shape == (SR,) and audio.dtype == np.float32

    bad = item.model_copy(update={"item_id": "b", "audio": ok_ref.model_copy(update={"sha256": "f" * 64})})
    asr = ScriptedASR("g", {SR: "vado a casa"})
    r = run_selfcheck("c", [item, bad], [asr], root=tmp_path, config=FAST)
    assert r.assessment.overall["n_items"] == 1 and r.assessment.overall["wer@g"] == 0.0
    assert any("SHA-256" in x for x in r.assessment.limitations)


# -- summary, writing, helpers -------------------------------------------------


def test_summary_wording():
    r = run_selfcheck("case-7", holdout(), {"adapted": biased()}, config=FAST)
    s = r.summary
    assert "su materiale simile, in queste condizioni, circa 20 parole su 100 sono sbagliate" in s
    assert "Per condizione:" in s and "auto, registrazione locale" in s and "codec amr-nb" in s
    assert "non è identico ai soliloqui in auto" in s
    assert render_summary_it(r.assessment) == s


def test_summary_when_errors_exceed_words():
    sa = SelfAssessment(case_id="c", overall={"n_items": 1.0, "n_ref_words": 2.0, "wer@x": 1.5})
    assert "gli errori superano le parole" in render_summary_it(sa)


def test_write_assessment_and_summary(tmp_path):
    r = run_selfcheck("c", holdout(), [perfect()], config=FAST)
    ref = write_self_assessment(r.assessment, tmp_path / "outputs" / "self_assessment.json", root=tmp_path)
    assert ref.path == "outputs/self_assessment.json" and ref.kind == "self_assessment" and ref.verify(tmp_path)
    back = SelfAssessment.model_validate_json((tmp_path / ref.path).read_text())
    assert back == r.assessment
    sref = write_summary(r.assessment, tmp_path / "outputs" / "self_assessment.txt", root=tmp_path)
    assert sref.verify(tmp_path) and "parole su 100" in (tmp_path / sref.path).read_text()


def test_altered_phrases_are_distinct_and_reproducible():
    vocab = "vado a casa domani porto mare".split()
    a = make_altered_phrases("vado a casa domani", vocab, item_rng(0, "x"), n=3)
    b = make_altered_phrases("vado a casa domani", vocab, item_rng(0, "x"), n=3)
    assert a == b and len(a) == 3 and len(set(a)) == 3
    assert "vado a casa domani" not in a


def test_describe_condition_it():
    assert describe_condition_it(CAR.key()) == "auto, registrazione locale, codec sconosciuto, voce normale"
    assert describe_condition_it("strana") == "strana"
