# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""End-to-end case on real Italian speech with real models (slow, offline).

Assets (never committed; the test is skipped when they are missing):
- SpeechBrain's ``example-it.wav`` (Common Voice clip, apache-2.0 repo) from
  the Hugging Face cache: it becomes the speech inside a synthetic in-car
  evidence recording (engine harmonics + road noise);
- MInDS-14 it-IT clips (CC-BY-4.0) extracted under ``/tmp/whisper_test_audio``
  with their plain-text transcriptions: they stand in for the subject's
  reference recordings (different callers, so the voiceprint numbers are not
  meaningful; the point is to run the real models through every workflow).

Models from the local cache only: openai/whisper-tiny, the small Italian CTC
model facebook/wav2vec2-base-10k-voxpopuli-ft-it (cc-by-nc-4.0, tests only),
SpeechBrain ECAPA-TDNN.

Run: HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -m slow tests/test_engine_slow.py -s
"""
import json
import os
import resource
import shutil
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
pytestmark = pytest.mark.slow

MINDS = Path(os.environ.get("AKOE_TEST_MINDS_DIR", "/tmp/whisper_test_audio"))
CTC = "facebook/wav2vec2-base-10k-voxpopuli-ft-it"
IT_TEXT = "Buongiorno a tutti e benvenuti a bordo"


def _example_it() -> Path:
    from huggingface_hub import hf_hub_download

    from tests.real_audio import IT_REPO, IT_REVISION

    try:
        return Path(hf_hub_download(IT_REPO, "example-it.wav", revision=IT_REVISION, local_files_only=True))
    except Exception:
        pytest.skip("example-it.wav non presente nella cache locale")


def _other_speakers() -> list[Path]:
    """Clips of speakers other than the MInDS callers: the Italian clip and the
    two English examples shipped with the ECAPA model (local cache)."""
    from huggingface_hub import hf_hub_download

    from tests.real_audio import EN_REPO, EN_REVISION

    out = [_example_it()]
    for name in ("example1.wav", "example2.flac"):
        try:
            out.append(Path(hf_hub_download(EN_REPO, name, revision=EN_REVISION, local_files_only=True)))
        except Exception:
            pass
    return out


def _peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def test_real_italian_case_end_to_end(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if not (MINDS / "minds_meta.json").exists():
        pytest.skip("clip MInDS-14 it-IT non presenti")
    import soundfile as sf

    from akoe import engine
    from akoe.adapt.train import TrainConfig
    from akoe.case import Case
    from akoe.config import Channel, RecordingSource, StageKind
    from akoe.models import HypothesisTranscript, ProcessingLog, RecordingCondition, SelfAssessment
    from akoe.speaker.voiceprint_backends import EcapaVoiceprint
    from akoe.vehicle.audio import resample
    from tests import m6_synth

    try:
        backends = engine.DefaultBackends(whisper_model="openai/whisper-tiny", ctc_model=CTC, local_files_only=True)
        backends.whisper("openai/whisper-tiny").load()
        backends.ctc_scorer().load()
        vp = EcapaVoiceprint(local_files_only=True)
        vp.load()
    except Exception as e:  # models not in the local cache
        pytest.skip(f"modelli non disponibili offline: {e}")

    # -- material ------------------------------------------------------------
    subj = tmp_path / "soggetto"
    subj.mkdir()
    meta = json.loads((MINDS / "minds_meta.json").read_text())
    index = []
    for m in meta:
        shutil.copy(MINDS / m["file"], subj / m["file"])
        (subj / m["file"]).with_suffix(".txt").write_text(m["text"])
        index.append({"audio": m["file"], "transcript": Path(m["file"]).with_suffix(".txt").name,
                      "channel": "phone", "source": "other", "vocal_effort": "normal"})
    (subj / "recordings.json").write_text(json.dumps({"recordings": index}))

    speech, sr = sf.read(_example_it(), dtype="float32")
    speech = resample(speech, sr, 16000) if sr != 16000 else speech
    dur = 10.0
    n = int(dur * 16000)
    t0 = 2.0
    clean_track = np.zeros(n, dtype=np.float32)
    clean_track[int(t0 * 16000):int(t0 * 16000) + len(speech)] = speech
    engine_noise, _ = m6_synth.engine_noise(dur, seed=1, level=0.02)
    road = m6_synth.colored_noise(dur, seed=2, level=0.01)
    evidence = (clean_track + engine_noise + road).astype(np.float32)
    ev_path = tmp_path / "auto_locale.wav"
    sf.write(ev_path, evidence / max(1.0, float(np.abs(evidence).max()) / 0.9), 16000)

    case = Case.create(tmp_path / "caso", "caso-reale")
    ev = case.import_evidence(ev_path, RecordingCondition(channel=Channel.CAR, source=RecordingSource.LOCAL))

    # -- workflows -------------------------------------------------------------
    out = {}
    out["vehicle"] = engine.build_vehicle_profile(case, "auto1", evidence_id=ev.evidence_id)
    out["subject"] = engine.build_subject_model(
        case, "s1", folder=subj, holdout_fraction=0.25, certify_all_by="test", voiceprint=vp,
        compute_timbre=False, other_speaker_files=_other_speakers(), provider=backends)
    out["clean"] = engine.clean(case, ev.evidence_id, variants=["bands", "denoise_offline", "engine_harmonics"])
    out["transcribe"] = engine.transcribe(case, ev.evidence_id, provider=backends, n_best=3)
    out["candidates"] = engine.score_candidates(
        case, ev.evidence_id, t0 - 0.2, t0 + len(speech) / 16000 + 0.2,
        [IT_TEXT, "Buonasera a tutti e benvenuti a casa", "Arrivederci e grazie della cena"], provider=backends)
    out["adapter"] = engine.train_subject_adapter(
        case, "s1", vehicle_id="auto1", codecs=["g711_ulaw", "g711_alaw"], provider=backends,
        train_config=TrainConfig(base_model="openai/whisper-tiny", max_steps=2, batch_size=2, lora_r=16,
                                 lora_alpha=32, num_threads=2))
    out["assess"] = engine.self_assess(case, "s1", vehicle_id="auto1", adapter="latest", voiceprint=vp,
                                       codecs=["g711_ulaw"], provider=backends,
                                       config=__import__("akoe.assess.selfcheck", fromlist=["x"])
                                       .SelfCheckConfig(n_best=3, n_bootstrap=200))
    out["transcribe_adapted"] = engine.transcribe(case, ev.evidence_id, adapter="latest", subject_id="s1",
                                                  voiceprint=vp, provider=backends, n_best=3)

    # -- checks ------------------------------------------------------------------
    for res in out.values():
        assert res.log.verify(case.root)
    assert case.verify() == []

    # issue #7 on real audio: plain-text transcripts became timed items
    align = next(s for s in ProcessingLog.model_validate_json(
        out["subject"].log.resolve(case.root).read_text()).stages if s.name == "transcript_alignment")
    assert align.params["n_items"] >= len(meta) and align.model_version.startswith(CTC)
    cal = out["subject"].summary["calibration"]
    assert cal["scheme"] == "leave-one-recording-out" and cal["holdout_used"] is False

    tr = HypothesisTranscript.model_validate_json(out["transcribe"].artifacts["transcript"]
                                                  .resolve(case.root).read_text())
    assert tr.segments and all(s.audio_provenance == StageKind.DETERMINISTIC for s in tr.segments)
    assert "whisper-tiny/generic" in tr.models
    asr_stage = next(s for s in ProcessingLog.model_validate_json(
        out["transcribe"].log.resolve(case.root).read_text()).stages if s.name == "asr:generic")
    assert asr_stage.model_version.startswith("openai/whisper-tiny@169d4a43") and asr_stage.model_sha256

    comp = out["candidates"].summary["comparison"]
    assert len(comp["reports"]) == 2  # CTC and Whisper decoder, reported separately
    ranks = {r["scorer"]: next(e["rank"] for e in r["entries"] if e["candidate"] == IT_TEXT)
             for r in comp["reports"]}
    assert min(ranks.values()) == 1

    tra = HypothesisTranscript.model_validate_json(out["transcribe_adapted"].artifacts["transcript"]
                                                   .resolve(case.root).read_text())
    assert any("/adapted:" in m for m in tra.models)
    sa = SelfAssessment.model_validate_json(out["assess"].artifacts["self_assessment"]
                                            .resolve(case.root).read_text())
    assert "wer@generic" in sa.overall and "wer@adapted" in sa.overall
    assert "leave-one-recording-out" in sa.limitations[0]

    print("\n--- transcript (generic):", [s.hypotheses[0].text for s in tr.segments])
    print("--- transcript (adapted):", [s.hypotheses[0].text for s in tra.segments])
    print("--- candidate ranks of the true phrase:", ranks, comp["outcome"])
    print("--- calibration:", {k: cal[k] for k in ("n_same", "n_other", "eer", "reliable")})
    print("--- alignment flagged:", len(align.params["flagged_items"]), "of", align.params["n_items"])
    print("--- self-assessment:\n" + out["assess"].summary["summary_it"])
    print(f"--- peak RSS: {_peak_mb():.0f} MB")
    assert _peak_mb() < 4096
