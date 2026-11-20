# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Case workflows end to end with fake backends (engine.py)."""
import json
import stat

import numpy as np
import pytest

from akoe import engine
from akoe.adapt.artifact import load_adapter_artifact, verify_adapter
from akoe.adapt.dataset import TrainingManifest
from akoe.adapt.lexicon import CaseLexicon
from akoe.adapt.train import TrainConfig
from akoe.case import Case
from akoe.config import Channel, RecordingSource, StageKind
from akoe.models import DatasetSplit, HypothesisTranscript, ProcessingLog, RecordingCondition, SegmentFlag, SelfAssessment
from akoe.speaker.dataset import SubjectDataset
from akoe.speaker.profile import LORO_SCHEME, load_calibration
from akoe.vehicle.profile import NoiseSource
from tests.engine_synth import make_evidence, make_subject_folder, make_vehicle_measurements
from tests.fakes import FakeASR, FakeProvider

CODECS = ["g711_ulaw", "g711_alaw"]


def _log(case, result) -> ProcessingLog:
    assert result.log.verify(case.root)
    return ProcessingLog.model_validate_json(result.log.resolve(case.root).read_text())


def _stage_names(log):
    return [s.name for s in log.stages]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One case taken through every workflow (module scope: built once)."""
    tmp = tmp_path_factory.mktemp("engine")
    case = Case.create(tmp / "case", "caso-integrazione")
    folder = make_subject_folder(tmp / "soggetto")
    ev_paths = make_evidence(tmp / "reperti")
    meas = make_vehicle_measurements(tmp / "misure")
    prov = FakeProvider()
    ev = case.import_evidence(ev_paths["evidence"], RecordingCondition(channel=Channel.CAR,
                                                                        source=RecordingSource.LOCAL),
                              pair_id="evento1")
    tx = case.import_evidence(ev_paths["transmitted"], RecordingCondition(
        channel=Channel.CAR, source=RecordingSource.TRANSMITTED, codec="g711_ulaw"), pair_id="evento1")
    lex = CaseLexicon(lexicon_id="lessico-caso", case_id=case.manifest.case_id)
    lex.add_term("Brambilla", "operatore1")
    lex_path = tmp / "lessico.json"
    lex.save(lex_path)

    r = {}
    r["vehicle"] = engine.build_vehicle_profile(case, "auto1", evidence_id=ev.evidence_id)
    r["vehicle_measured"] = engine.build_vehicle_profile(
        case, "auto2", impulse_responses=[meas["ir"]],
        noise_recordings=[NoiseSource(path=meas["noise"], label="minimo")])
    r["subject"] = engine.build_subject_model(case, "soggetto1", folder=folder, holdout_fraction=0.3,
                                              certify_all_by="operatore1", provider=prov)
    r["clean"] = engine.clean(case, ev.evidence_id, subject_id="soggetto1", vehicle_id="auto1",
                              reference=ev_paths["reference"])
    r["transcribe"] = engine.transcribe(case, pair_id="evento1", lexicon=lex_path, subject_id="soggetto1",
                                        provider=prov)
    r["candidates"] = engine.score_candidates(case, ev.evidence_id, 1.0, 3.0, ["vado a casa", "vado a cassa"],
                                              {"fake-ctc": 2.0}, provider=prov)
    r["adapter"] = engine.train_subject_adapter(case, "soggetto1", vehicle_id="auto1", codecs=CODECS,
                                                train_config=TrainConfig(base_model="fake/whisper-tiny"),
                                                provider=prov)
    r["assess"] = engine.self_assess(case, "soggetto1", vehicle_id="auto1", adapter="latest", codecs=CODECS,
                                     lexicon=lex_path, provider=prov)
    r["transcribe_adapted"] = engine.transcribe(case, ev.evidence_id, adapter="latest", subject_id="soggetto1",
                                                provider=prov)
    return {"case": case, "ev": ev, "tx": tx, "results": r, "paths": ev_paths, "lex": lex_path, "prov": prov}


def test_every_workflow_writes_a_verified_log_recorded_in_the_manifest(built):
    case = Case.open(built["case"].root)
    results = built["results"]
    assert len(case.manifest.logs) == len(results)
    for name, res in results.items():
        log = _log(case, res)
        assert log.workflow == res.workflow and log.tier.value == "edge" and log.finished_at is not None
        assert any(r.path == res.log.path and r.sha256 == res.log.sha256 for r in case.manifest.logs)
        for ref in log.artifacts + log.inputs:
            assert ref.verify(case.root if not ref.path.startswith("/") else None), (name, ref.path)
        assert not res.log.resolve(case.root).stat().st_mode & stat.S_IWUSR  # logs are read-only
    assert case.verify() == []  # evidence untouched by every workflow


def test_vehicle_profiles_estimated_and_measured(built):
    case, res = built["case"], built["results"]
    est = res["vehicle"]
    assert est.summary["estimated"] is True and est.summary["noise_labels"] == ["estimated-nonspeech"]
    log = _log(case, est)
    assert "vehicle_profile_estimated" in _stage_names(log)
    assert log.inputs[0].sha256 == built["ev"].artifact.sha256
    profile, ref = engine.load_vehicle_profile(case, "auto1")
    assert ref == est.artifacts["vehicle_profile"] and profile.noise_estimate.snr_db is not None
    measured = res["vehicle_measured"]
    assert measured.summary["estimated"] is False and measured.summary["n_impulse_responses"] == 1
    assert "vehicle_profile_measured" in _stage_names(_log(case, measured))


def test_subject_model_aligns_splits_profiles_and_calibrates_without_holdout(built):
    case, res = built["case"], built["results"]["subject"]
    log = _log(case, res)
    names = _stage_names(log)
    for stage in ("subject_import", "transcript_alignment", "split_by_recording", "speaker_profile",
                  "score_calibration"):
        assert stage in names
    ds = SubjectDataset.load_or_create(case, "soggetto1")
    assert len(ds.recordings) == 7 and ds.summary()["pending_alignment_recordings"] == 0
    aligned = [it for it in ds.items if "-a" in it.item_id]
    assert aligned and all(it.text for it in aligned)
    assert len(ds.holdout_recordings()) == 2
    state = engine.load_subject_model(case, "soggetto1")
    assert state.profile is not None and state.voiceprint_backend == "fake-voiceprint"
    cal = load_calibration(case, state.calibration)
    assert cal.scheme == LORO_SCHEME and cal.holdout_used is False
    material = json.loads(cal.material)
    assert material["holdout_used"] is False and material["n_groups"] >= 2
    assert any("leave-one-recording-out" in line for line in res.summary["summary_it"])
    assert res.summary["calibration"]["performed"] is True
    report = json.loads(res.artifacts["report"].resolve(case.root).read_text())
    assert report["calibration"]["holdout_used"] is False
    # only verified training items enter the profile
    holdout_items = {it.item_id for it in ds.items if it.split == DatasetSplit.HOLDOUT}
    build = json.loads(log.stages[names.index("speaker_profile")].model_dump_json())
    assert build["params"]["n_eligible"] > 0
    for cp_ref in [a for a in log.artifacts if a.kind == "voiceprint_centroid"]:
        data = json.loads(cp_ref.resolve(case.root).read_text())
        assert not set(data["item_ids"]) & holdout_items


def test_listening_variants_are_separate_labelled_artifacts(built):
    case, res, ev = built["case"], built["results"]["clean"], built["ev"]
    variants = res.summary["variants"]
    assert {"bands", "denoise_offline", "reference_cancel", "nmf_listening"} <= set(variants)
    index = engine.listening_index(case, ev.evidence_id)
    assert index["variants"][0]["variant"] == "original" and index["variants"][0]["is_original"]
    shas = set()
    for entry in index["variants"][1:]:
        assert entry["audio_provenance"] == StageKind.DETERMINISTIC.value
        assert "non è l'audio originale" in entry["label_it"]
        rec = json.loads((case.root / entry["record"]["path"]).read_text())
        assert rec["intended_use"] == "human_listening" and rec["evidence_sha256"] == ev.artifact.sha256
        shas.add(entry["audio"]["sha256"])
    assert len(shas) == len(index["variants"]) - 1 and ev.artifact.sha256 not in shas
    log = _log(case, res)
    assert "vad_robust" in _stage_names(log)
    assert any(r.kind == "nmf_dictionaries" for r in log.artifacts)
    harmonics = "engine_harmonics" in variants or "listening:engine_harmonics_skipped" in _stage_names(log)
    assert harmonics


def test_transcript_from_the_local_recording_with_lexicon_readings_and_scores(built):
    case, res, ev = built["case"], built["results"]["transcribe"], built["ev"]
    assert res.summary["evidence_id"] == ev.evidence_id  # local first
    ref = res.artifacts["transcript"]
    assert ref.verify(case.root) and "/versions/" in ref.path
    tr = HypothesisTranscript.model_validate_json(ref.resolve(case.root).read_text())
    assert tr.is_ai_hypothesis and tr.evidence_id == ev.evidence_id
    assert tr.segments and all(s.audio_provenance == StageKind.DETERMINISTIC for s in tr.segments)
    origins = {a.origin for s in tr.segments for a in s.hypotheses}
    assert "fake-asr/generic" in origins and any("+lexicon:" in o for o in origins)
    assert all(SegmentFlag.LEXICON_DEPENDENT in s.flags for s in tr.segments)
    # the consensus is the reading without lexicon (lexicon weight 0)
    assert all("Brambilla" not in s.hypotheses[0].text for s in tr.segments)
    assert all(s.speaker_score is not None and s.speaker_condition for s in tr.segments)
    log = _log(case, res)
    for stage in ("vad_robust", "asr:generic", "asr_lexicon:generic", "combine", "speaker_scores",
                  "hallucination_guard", "format"):
        assert stage in _stage_names(log)
    assert any("Coppia evento1" in n for n in log.notes)


def test_candidates_reported_per_scorer(built):
    case, res = built["case"], built["results"]["candidates"]
    comp = res.summary["comparison"]
    assert [r["scorer"] for r in comp["reports"]] == ["fake-ctc", "fake-whisper"]
    assert comp["reports"][0]["threshold"] == 2.0 and comp["reports"][1]["threshold"] is None
    assert comp["outcome"] == "some_compatible"
    assert res.artifacts["candidate_comparison"].verify(case.root)
    names = _stage_names(_log(case, res))
    assert "candidate_scoring:fake-ctc" in names and "candidate_scoring:fake-whisper" in names


def test_adapter_trained_on_verified_training_items_with_channel_and_codecs(built):
    case, res = built["case"], built["results"]["adapter"]
    art = load_adapter_artifact(res.artifacts["adapter"].resolve(case.root))
    verify_adapter(art, case.root)
    manifest = TrainingManifest.read(res.artifacts["train_manifest"].resolve(case.root))
    ds = SubjectDataset.load_or_create(case, "soggetto1")
    by_id = {it.item_id: it for it in ds.items}
    assert manifest.sources and all(by_id[s.item_id].verified and by_id[s.item_id].split == DatasetSplit.TRAIN
                                    for s in manifest.sources)
    augmented = [e for e in manifest.examples if not e.clean]
    assert augmented and manifest.augment_name == "vehicle-auto1+codecs"
    assert all(e.augmentation["noise"] is not None for e in augmented)
    assert {(e.augmentation.get("codec") or {}).get("codec") for e in augmented} <= {None, "none", *CODECS}
    assert engine.load_subject_model(case, "soggetto1").adapter == res.artifacts["adapter"]
    assert "adapter_training" in _stage_names(_log(case, res))


def test_self_assessment_per_condition_with_calibration_statement(built):
    case, res = built["case"], built["results"]["assess"]
    sa = SelfAssessment.model_validate_json(res.artifacts["self_assessment"].resolve(case.root).read_text())
    o = sa.overall
    for key in ("wer@generic", "wer@adapted", "wer@generic+lexicon", "candidate_top1@fake-ctc",
                "candidate_top1@fake-whisper", "voiceprint_eer@fake-voiceprint", "candidate_threshold@fake-ctc"):
        assert key in o, key
    assert o["degraded"] == 1.0 and sa.by_condition
    assert "leave-one-recording-out" in sa.limitations[0] and "non è stata usata" in sa.limitations[0]
    summary = res.artifacts["summary"].resolve(case.root).read_text()
    assert "leave-one-recording-out" in summary and "Per condizione" in summary
    thresholds = json.loads(res.artifacts["candidate_thresholds"].resolve(case.root).read_text())
    assert set(thresholds["scorers"]) == {"fake-ctc", "fake-whisper"}
    log = _log(case, res)
    assert "adapter_verification" in _stage_names(log) and "self_assessment" in _stage_names(log)
    found = engine.read_self_assessment(case, "soggetto1")
    assert found is not None and found[0] == sa


def test_adapted_transcription_adds_the_adapted_source(built):
    case, res, ev = built["case"], built["results"]["transcribe_adapted"], built["ev"]
    assert res.summary["adapter"]
    tr, _ = engine.read_transcript(case, ev.evidence_id)
    origins = {a.origin for s in tr.segments for a in s.hypotheses}
    assert any(o.startswith("fake-asr/adapted:") for o in origins)
    # every transcript version is kept, read-only: earlier logs still verify
    versions = case.root / "outputs" / ev.evidence_id / "versions"
    assert len([p for p in versions.iterdir() if p.name.startswith("transcript-")]) >= 2
    assert built["results"]["transcribe"].artifacts["transcript"].verify(case.root)


def test_thresholds_from_self_assessment(built):
    case = built["case"]
    res = engine.score_candidates(case, built["ev"].evidence_id, 4.5, 6.5, ["vado a casa", "poi esco"],
                                  thresholds_from="soggetto1", provider=built["prov"])
    reports = res.summary["comparison"]["reports"]
    assert all(r["threshold"] is not None for r in reports)
    assert res.summary["thresholds_source"] == "self_assessment:soggetto1"


class _LegacyASR:
    """A backend that ignores provenance (declares NEURAL on original audio)."""

    name = "legacy"

    def transcribe(self, audio, sr, *, segments=None, n_best=5):
        return FakeASR("ciao").transcribe(audio, sr, segments=segments)


def test_provenance_is_the_one_of_the_audio_given(built):
    case, ev = built["case"], built["ev"]
    res = engine.transcribe(case, ev.evidence_id, backends=[_LegacyASR()])
    tr, _ = engine.read_transcript(case, ev.evidence_id)
    assert all(s.audio_provenance == StageKind.DETERMINISTIC for s in tr.segments)
    assert any("provenienza" in n for n in res.notes)


def test_tampered_inputs_are_refused(tmp_path, built):
    case = Case.create(tmp_path / "c2", "c2")
    item = case.import_evidence(built["paths"]["evidence"])
    path = case.path_of(item)
    path.chmod(0o644)
    data = bytearray(path.read_bytes())
    data[-10] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(engine.EngineError, match="SHA-256"):
        engine.transcribe(case, item.evidence_id, backends=[FakeASR()])
    with pytest.raises(engine.EngineError):
        engine.transcribe(case, item.evidence_id, pair_id="x", backends=[FakeASR()])


def test_tampered_adapter_is_refused(built):
    case = built["case"]
    art_ref = built["results"]["adapter"].artifacts["adapter"]
    art = load_adapter_artifact(art_ref.resolve(case.root))
    weights = art.weights.resolve(case.root).parent / "adapter_model.safetensors"
    original = weights.read_bytes()
    try:
        weights.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        with pytest.raises(engine.EngineError, match="Adattatore non utilizzabile"):
            engine.resolve_adapter(case, "latest", "soggetto1")
    finally:
        weights.write_bytes(original)
    assert engine.resolve_adapter(case, "latest", "soggetto1").artifact.adapter_id == art.adapter_id


def test_subject_model_without_verified_items_is_not_built(tmp_path):
    case = Case.create(tmp_path / "c3", "c3")
    folder = make_subject_folder(tmp_path / "s", n_car=2, n_phone=1)
    res = engine.build_subject_model(case, "s1", folder=folder, provider=FakeProvider(), compute_timbre=False)
    assert res.summary["profile_version"] is None and "profile" not in res.artifacts
    assert any("nessun segmento verificato" in n for n in res.notes)
    with pytest.raises(engine.EngineError):
        engine.train_subject_adapter(case, "s1", provider=FakeProvider())


def test_audio_is_never_merged_with_neural_audio(built, tmp_path):
    """Investigative mode records the (unavailable) neural cleaning as skipped,
    in its own folder, and never touches the deterministic variants."""
    from akoe.config import Mode

    case, ev = built["case"], built["ev"]
    before = engine.listening_index(case, ev.evidence_id)
    res = engine.clean(case, ev.evidence_id, variants=["denoise_offline"], mode=Mode.BOTH)
    log = _log(case, res)
    assert "listening:neural_skipped" in _stage_names(log)
    after = engine.listening_index(case, ev.evidence_id)
    assert [v["variant"] for v in after["variants"]] == ["original", "denoise_offline"]
    assert all(v["audio_provenance"] == "deterministic" for v in after["variants"])
    assert before["variants"][0] == after["variants"][0]


def test_channel_augment_keeps_length_and_rate(built):
    case = built["case"]
    aug, snr, ref = engine.channel_augment_for(case, "auto1", codecs=CODECS, codec_probability=1.0)
    x = np.random.default_rng(0).normal(0, 0.1, 8000).astype(np.float32)
    y, params = aug(x, 8000, np.random.default_rng(1))
    assert len(y) == len(x) and params["codec"]["codec"] in CODECS and ref is not None
    y2, _ = aug(x, 8000, np.random.default_rng(1))
    assert np.array_equal(y, y2)


def test_default_backends_factories_can_nest_without_deadlock():
    """The aligner factory needs the CTC scorer from the same cache."""
    import threading

    from tests.fakes import uniform_ctc_scorer

    class Offline(engine.DefaultBackends):
        def ctc_scorer(self):
            return self._get(("ctc", "fake"), uniform_ctc_scorer)

    backends = Offline()
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("aligner", backends.aligner()), daemon=True)
    t.start()
    t.join(timeout=10)
    assert not t.is_alive() and out["aligner"].scorer is backends.ctc_scorer()
    assert backends.aligner() is out["aligner"]  # cached
