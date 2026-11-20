import math
from types import SimpleNamespace

import pytest

from forensicaudio.adapt.lexicon import (
    BiasConfig,
    CaseLexicon,
    DualHypothesis,
    LexiconError,
    TermCategory,
    TermStatus,
    dual_decode,
    find_terms,
    measure_insertion_rate,
    normalize_text,
    propose_lexicon,
    word_error_rate,
)
from forensicaudio.models import ArtifactRef, DatasetItem, DatasetSplit, SegmentFlag

REF = ArtifactRef(path="/x.wav", sha256="0" * 64, kind="evidence")


def _item(item_id, text, verified=True, split=DatasetSplit.TRAIN):
    return DatasetItem(item_id=item_id, recording_id="r", audio=REF, start=0, end=1, text=text,
                       verified=verified, split=split)


ITEMS = [
    _item("a", "Ieri ho visto Mario Rossi a Porta Nuova. Poi siamo andati da Brambilla."),
    _item("b", "Domani chiamo Brambilla per la roba, targa AB123CD."),
    _item("c", "Porta la roba domani."),
    _item("u", "Ho parlato con Giovanni Esposito.", verified=False),
    _item("h", "C'era anche Ferraris.", split=DatasetSplit.HOLDOUT),
]


# -- text helpers -------------------------------------------------------------


def test_normalisation_and_matching():
    assert normalize_text("Dell'Aquila, PERCHÉ?") == "dell aquila perché"
    assert find_terms("Siamo all'Aquila con Mario Rossi.", ["aquila", "Mario Rossi", "Rossi", "Roma"]) == [
        "aquila", "Mario Rossi", "Rossi"]
    assert find_terms("grossi guai", ["Rossi"]) == []  # whole words only
    assert word_error_rate("vado a casa", "vado a cassa") == pytest.approx(1 / 3)
    assert word_error_rate("", "") == 0.0


# -- lexicon building and operator decisions ----------------------------------


def test_propose_uses_only_verified_training_transcripts():
    lex = propose_lexicon(ITEMS, lexicon_id="lex1", case_id="c1")
    terms = {e.term: e for e in lex.entries}
    assert set(terms) == {"Mario Rossi", "Porta Nuova", "Brambilla", "AB123CD"}
    assert terms["Brambilla"].count == 2 and terms["Brambilla"].source_items == ["a", "b"]
    assert terms["Mario Rossi"].category == TermCategory.NAME
    assert terms["AB123CD"].category == TermCategory.OTHER
    assert all(e.status == TermStatus.PROPOSED for e in lex.entries)
    assert lex.source_items == ["a", "b", "c"]  # not the unverified nor the holdout item
    assert lex.terms() == []  # nothing confirmed yet
    with_holdout = propose_lexicon(ITEMS, lexicon_id="lex1", include_holdout=True)
    assert with_holdout.get("Ferraris") is not None
    assert with_holdout.get("Giovanni Esposito") is None


def test_jargon_needs_a_vocabulary_and_repetition():
    known = {"porta", "domani", "chiamo", "targa", "ieri", "visto", "siamo", "andati", "ho"}
    lex = propose_lexicon(ITEMS, lexicon_id="l", known_words=known)
    roba = lex.get("roba")
    assert roba is not None and roba.category == TermCategory.JARGON and roba.count == 2


def test_operator_confirmation_flow(tmp_path):
    lex = propose_lexicon(ITEMS, lexicon_id="lex1")
    v0 = lex.version
    lex.confirm("Brambilla", "op-17", variants=["Brambila"])
    lex.confirm("mario rossi", "op-17")
    lex.reject("AB123CD", "op-17", note="targa: non serve")
    lex.add_term("Rho", "op-17", category=TermCategory.PLACE)
    assert lex.version == v0 + 4
    assert lex.terms() == ["Brambila", "Brambilla", "Mario Rossi", "Rho"]
    assert lex.get("Brambilla").decided_by == "op-17" and lex.get("Brambilla").decided_at is not None
    assert [e.term for e in lex.pending()] == ["Porta Nuova"]
    with pytest.raises(LexiconError, match="operatore"):
        lex.confirm("Porta Nuova", " ")
    with pytest.raises(LexiconError, match="non presente"):
        lex.confirm("Inesistente", "op-17")
    # re-proposing keeps decisions (a rejected term stays rejected)
    again = propose_lexicon(ITEMS + [_item("d", "Visto Brambilla.")], lexicon_id="lex1", existing=lex)
    assert again.get("AB123CD").status == TermStatus.REJECTED
    assert again.get("Brambilla").status == TermStatus.CONFIRMED and again.get("Brambilla").count == 3
    assert lex.get("Brambilla").count == 2  # the original is not modified

    ref = lex.save(tmp_path / "profiles" / "lexicon.json", root=tmp_path)
    assert ref.kind == "lexicon" and ref.sha256 == lex.sha256()
    assert CaseLexicon.load(ref.resolve(tmp_path), expected_sha256=ref.sha256) == lex
    with pytest.raises(LexiconError, match="impronta SHA-256"):
        CaseLexicon.load(ref.resolve(tmp_path), expected_sha256="0" * 64)


# -- dual hypotheses and insertion test (model-agnostic) ----------------------


def _confirmed(*terms):
    lex = CaseLexicon(lexicon_id="l")
    for t in terms:
        lex.add_term(t, "op")
    return lex


def test_dual_decode_keeps_both_when_they_differ():
    lex = _confirmed("Brambilla")
    same = dual_decode(lambda use: "vado a casa", lex)
    assert not same.differs and same.flags() == []
    assert [a.text for a in same.alternatives("whisper-small/adapted:ab")] == ["vado a casa"]

    diff = dual_decode(lambda use: "chiamo Brambilla" if use else "chiamo bramilla", lex)
    assert diff.differs and diff.terms_introduced == ["Brambilla"] and diff.terms_removed == []
    assert diff.flags() == [SegmentFlag.LEXICON_DEPENDENT]
    alts = diff.alternatives("whisper-small/adapted:ab")
    assert [a.text for a in alts] == ["chiamo bramilla", "chiamo Brambilla"]
    assert alts[0].origin == "whisper-small/adapted:ab"
    assert alts[1].origin == f"whisper-small/adapted:ab+lexicon:{lex.sha256()[:12]}"
    assert DualHypothesis.model_validate_json(diff.model_dump_json()) == diff


def test_unconfirmed_lexicon_cannot_bias():
    lex = propose_lexicon(ITEMS, lexicon_id="l")
    with pytest.raises(LexiconError, match="confermati"):
        dual_decode(lambda use: "x", lex)


def test_insertion_rate_measures_terms_where_absent():
    items = [SimpleNamespace(item_id=f"i{k}", text=t)
             for k, t in enumerate(["vado a casa", "sono qui", "ciao Rossi", "ciao mondo"])]
    lex = _confirmed("Rossi", "Brambilla", "rossi")  # case variants count once

    def biased(item):
        return item.text + " Rossi" if item.item_id in ("i0", "i1") else item.text

    report = measure_insertion_rate(biased, items, lex, baseline_decode=lambda it: it.text)
    assert report.n_terms == 2 and report.n_items == 4 and report.n_items_with_absent_terms == 4
    w = report.with_lexicon
    assert w.opportunities == 7 and w.insertions == 2
    assert w.insertion_rate == pytest.approx(2 / 7)
    assert w.items_with_insertion == 2 and w.item_insertion_rate == pytest.approx(0.5)
    assert w.per_term == {"Rossi": 2}
    assert [d.item_id for d in w.details] == ["i0", "i1"]
    assert w.mean_wer > 0
    assert report.without_lexicon.insertions == 0 and report.without_lexicon.mean_wer == 0
    assert report.excess_insertion_rate == pytest.approx(2 / 7)
    assert report.lexicon_sha256 == lex.sha256()


# -- token trie and logits processor (needs transformers) ---------------------

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from forensicaudio.adapt.lexicon import (  # noqa: E402
    LexiconBiasProcessor,
    TokenTrie,
    WhisperLexiconDecoder,
    term_token_sequences,
)
from tests.tiny_whisper import make_model, make_processor  # noqa: E402


@pytest.fixture(scope="module")
def tok():
    return make_processor().tokenizer


def _id(tok, token):
    return tok.convert_tokens_to_ids(token)


def test_term_sequences_and_trie(tok):
    seqs, skipped = term_token_sequences(tok, ["Brambilla", "Rossi", "casa", "a"])
    bram, billa, rossi = _id(tok, "ĠBram"), _id(tok, "billa"), _id(tok, "ĠRossi")
    assert seqs[(bram, billa)] == "Brambilla" and seqs[(rossi,)] == "Rossi"
    assert skipped == ["a"]
    assert sorted(t for t, term in seqs.items() if term == "casa") == sorted(
        [tuple(tok(" casa", add_special_tokens=False).input_ids),
         tuple(tok(" Casa", add_special_tokens=False).input_ids)])
    trie = TokenTrie(seqs)
    path = trie.walk([bram, billa])
    assert path is not None and trie.terminal[path[-1]] == "Brambilla"
    assert trie.walk([billa]) is None


def _scores(tok, values: dict[int, float], base: float = 0.0, rows: int = 1):
    s = torch.full((rows, len(tok)), base)
    for i, v in values.items():
        s[:, i] = v
    return s


def test_start_gate_boosts_only_plausible_terms(tok):
    sot = _id(tok, "<|startoftranscript|>")
    bram, rossi, ciao = _id(tok, "ĠBram"), _id(tok, "ĠRossi"), _id(tok, "Ġciao")
    proc = LexiconBiasProcessor.from_tokenizer(tok, ["Rossi", "Brambilla"], BiasConfig(boost=2.0, start_margin=3.0),
                                               begin_token_id=sot)
    ids = torch.tensor([[sot, ciao]])
    scores = _scores(tok, {ciao: 10.0, rossi: 8.5, bram: 1.0}, base=-5.0)
    out = proc(ids, scores.clone())
    assert out[0, rossi] == pytest.approx(10.5)  # within the margin: boosted
    assert out[0, bram] == pytest.approx(1.0)  # implausible: untouched
    assert out[0, ciao] == pytest.approx(10.0)
    assert proc.stats["start_boosts"] == 1


def test_continuation_and_cap(tok):
    sot = _id(tok, "<|startoftranscript|>")
    bram, billa, rossi, ciao = _id(tok, "ĠBram"), _id(tok, "billa"), _id(tok, "ĠRossi"), _id(tok, "Ġciao")
    cfg = BiasConfig(boost=2.0, start_margin=3.0, continuation_margin=6.0, max_terms_per_decode=2)
    proc = LexiconBiasProcessor.from_tokenizer(tok, ["Rossi", "Brambilla"], cfg, begin_token_id=sot)
    # inside "Brambilla": continuation boosted even if 5 nats below the best
    out = proc(torch.tensor([[sot, ciao, bram]]), _scores(tok, {ciao: 10.0, billa: 5.0, rossi: 9.0}))
    assert out[0, billa] == pytest.approx(7.0)
    assert out[0, rossi] == pytest.approx(9.0)  # no new term while one is in progress
    # two terms already produced: cap reached, nothing boosted
    proc.reset()
    out = proc(torch.tensor([[sot, rossi, ciao, bram, billa]]), _scores(tok, {ciao: 10.0, rossi: 9.0}))
    assert out[0, rossi] == pytest.approx(9.0) and proc.stats["capped_rows"] == 1
    # terms in the prompt (before <|startoftranscript|>) do not count
    proc.reset()
    out = proc(torch.tensor([[rossi, rossi, sot, ciao]]), _scores(tok, {ciao: 10.0, rossi: 9.0}))
    assert out[0, rossi] == pytest.approx(11.0)


def test_unfinished_term_bonus_is_withdrawn_in_beam_search(tok):
    sot = _id(tok, "<|startoftranscript|>")
    bram, billa, ciao = _id(tok, "ĠBram"), _id(tok, "billa"), _id(tok, "Ġciao")
    proc = LexiconBiasProcessor.from_tokenizer(tok, ["Brambilla"], BiasConfig(boost=2.0), begin_token_id=sot,
                                               num_beams=2, cumulative_scores=True)
    step1 = torch.log_softmax(_scores(tok, {ciao: 10.0, bram: 9.0}, rows=2), -1)
    out1 = proc(torch.tensor([[sot], [sot]]), step1.clone())
    assert out1[0, bram] - step1[0, bram] == pytest.approx(2.0)
    # beam 0 took "ĠBram": its other continuations lose the 2.0 again, "billa" keeps it (+2 more)
    step2 = torch.log_softmax(_scores(tok, {ciao: 10.0, billa: 9.0}, rows=2), -1)
    out2 = proc(torch.tensor([[sot, bram], [sot, ciao]]), step2.clone())
    assert out2[0, ciao] - step2[0, ciao] == pytest.approx(-2.0)
    assert out2[0, billa] - step2[0, billa] == pytest.approx(2.0)
    assert out2[1, ciao] - step2[1, ciao] == pytest.approx(0.0)
    assert proc.stats["withdrawals"] == 1


def test_suppressed_tokens_stay_suppressed(tok):
    rossi, ciao = _id(tok, "ĠRossi"), _id(tok, "Ġciao")
    proc = LexiconBiasProcessor.from_tokenizer(tok, ["Rossi"], BiasConfig(start_margin=100.0))
    scores = _scores(tok, {ciao: 1.0, rossi: -math.inf})
    out = proc(torch.tensor([[ciao]]), scores)
    assert out[0, rossi] == -math.inf


def test_whisper_decoder_with_and_without_lexicon():
    processor = make_processor()
    model = make_model(processor.tokenizer, seed=3).eval()
    tok = processor.tokenizer
    import numpy as np

    audio = np.zeros(16000, dtype=np.float32)
    plain = WhisperLexiconDecoder(model, processor, ["Rossi"], max_new_tokens=4)
    feats = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
    prefix = torch.tensor([tok.prefix_tokens])
    with torch.no_grad():
        logits = model(input_features=feats, decoder_input_ids=prefix).logits[0, -1]
    first = int(logits.argmax())
    # make " Rossi" just below the model's first choice: only the lexicon can promote it
    rossi = _id(tok, "ĠRossi")
    factor = 0.99 if logits[first] > 0 else 1.01
    with torch.no_grad():
        model.proj_out.weight[rossi] = model.proj_out.weight[first] * factor
        assert model(input_features=feats, decoder_input_ids=prefix).logits[0, -1].argmax() == first
    dual = plain.decode_both(audio, 16000)
    assert dual.differs and dual.with_lexicon.startswith("Rossi")
    assert "Rossi" in dual.terms_introduced and dual.flags() == [SegmentFlag.LEXICON_DEPENDENT]
    assert plain.last_stats["start_boosts"] >= 1
    # beam search path runs too (cumulative scores, withdrawal enabled)
    beam = WhisperLexiconDecoder(model, processor, ["Rossi"], max_new_tokens=4, num_beams=2)
    assert beam.make_processor().cumulative is True
    assert isinstance(beam.decode(audio, 16000, use_lexicon=True), str)
