# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Engine workflows through the API (jobs, transcript, self-assessment,
labelled listening audio, synchronous candidate scoring), with fakes."""
import time
from urllib.parse import unquote

import pytest
from fastapi.testclient import TestClient

from akoe.api.server import create_app
from tests.engine_synth import make_evidence, make_subject_folder
from tests.fakes import FakeProvider

H = {"Authorization": "Bearer t0k"}


def _wait(client, job, timeout=120.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = client.get(f"/jobs/{job['job_id']}", headers=H).json()
        if state["status"] in ("done", "failed"):
            return state
        time.sleep(0.05)
    raise AssertionError(f"job {job['kind']} did not finish")


def _job(client, kind, **params):
    r = client.post("/cases/c1/jobs", json={"kind": kind, "params": params}, headers=H)
    assert r.status_code == 202, r.text
    state = _wait(client, r.json())
    assert state["status"] == "done", state.get("error") or state.get("message")
    return state["result"]


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api")
    imports = tmp / "import"
    folder = make_subject_folder(imports / "soggetto")
    ev = make_evidence(tmp / "reperti")
    app = create_app(tmp / "cases", token="t0k", backends=FakeProvider(), import_roots=[imports])
    client = TestClient(app)
    assert client.post("/cases", json={"case_id": "c1"}, headers=H).status_code == 201
    with ev["evidence"].open("rb") as f:
        item = client.post("/cases/c1/evidence", headers=H, files={"file": ("locale.wav", f, "audio/wav")},
                           data={"channel": "car", "source": "local", "pair_id": "e1"}).json()
    with ev["reference"].open("rb") as f:
        ref = client.post("/cases/c1/uploads", headers=H, files={"file": ("brano radio.wav", f, "audio/wav")})
    assert ref.status_code == 201 and ref.json()["path"].startswith("working/uploads/")
    return {"client": client, "item": item, "folder": folder, "reference": ref.json(), "tmp": tmp}


def test_health_lists_the_workflow_jobs(api):
    jobs = api["client"].get("/health", headers=H).json()["jobs"]
    for kind in ("build_subject_model", "build_vehicle_profile", "clean", "transcribe", "score_candidates",
                 "train_subject_adapter", "self_assess", "verify"):
        assert kind in jobs


def test_paths_outside_the_case_or_import_roots_are_refused(api):
    client = api["client"]
    for bad in ("../../../etc/passwd", "/etc/passwd", str(api["tmp"] / "reperti")):
        r = client.post("/cases/c1/jobs", headers=H, json={"kind": "build_subject_model",
                                                            "params": {"subject_id": "s1", "folder": bad}})
        assert r.status_code == 400, bad
    r = client.post("/cases/c1/jobs", headers=H, json={"kind": "transcribe", "params": {"evidence_id": "../x"}})
    assert r.status_code == 400
    r = client.post("/cases/c1/jobs", headers=H, json={"kind": "clean", "params": {"evidence_id": "e", "rm": 1}})
    assert r.status_code == 400 and "sconosciuti" in r.json()["detail"]


def test_workflows_as_jobs_and_their_outputs(api):
    client, eid = api["client"], api["item"]["evidence_id"]
    assert client.get(f"/cases/c1/evidence/{eid}/transcript", headers=H).status_code == 404

    veh = _job(client, "build_vehicle_profile", vehicle_id="auto1", evidence_id=eid)
    assert veh["summary"]["estimated"] is True
    subj = _job(client, "build_subject_model", subject_id="s1", folder=str(api["folder"]), holdout_fraction=0.3,
                certify_all_by="operatore1", compute_timbre=False)
    assert subj["summary"]["calibration"]["holdout_used"] is False
    model = client.get("/cases/c1/subjects/s1/model", headers=H).json()
    assert model["state"]["calibration_scheme"] == "leave-one-recording-out" and model["report"]["summary_it"]

    _job(client, "clean", evidence_id=eid, subject_id="s1", vehicle_id="auto1", reference=api["reference"]["path"],
         variants=["bands", "reference_cancel"])
    listing = client.get(f"/cases/c1/evidence/{eid}/listening", headers=H).json()
    assert [v["variant"] for v in listing["variants"]] == ["original", "bands", "reference_cancel"]
    audio = client.get(f"/cases/c1/evidence/{eid}/listening/bands/audio", headers=H)
    assert audio.status_code == 200 and audio.content[:4] == b"RIFF"
    assert audio.headers["x-audio-variant"] == "bands"
    assert audio.headers["x-audio-provenance"] == "deterministic"
    assert "non è l'audio originale" in unquote(audio.headers["x-audio-label"])
    orig = client.get(f"/cases/c1/evidence/{eid}/listening/original/audio", headers=H)
    assert orig.headers["x-audio-variant"] == "original" and orig.headers["x-content-sha256"] == \
        api["item"]["artifact"]["sha256"]
    assert client.get(f"/cases/c1/evidence/{eid}/listening/../../case.json/audio", headers=H).status_code == 404
    assert client.get(f"/cases/c1/evidence/{eid}/listening/neural/audio", headers=H).status_code == 404

    tr = _job(client, "transcribe", pair_id="e1", subject_id="s1")
    got = client.get(f"/cases/c1/evidence/{eid}/transcript", headers=H)
    assert got.status_code == 200 and got.json()["is_ai_hypothesis"] is True
    assert got.headers["x-content-sha256"] == tr["artifacts"]["transcript"]["sha256"]
    assert all(s["audio_provenance"] == "deterministic" for s in got.json()["segments"])

    sync = client.post(f"/cases/c1/evidence/{eid}/candidates", headers=H,
                       json={"start": 1.0, "end": 3.0, "candidates": ["vado a casa", "vado a cassa"],
                             "thresholds": {"fake-ctc": 2.0}, "timeout_s": 60})
    assert sync.status_code == 200, sync.text
    reports = sync.json()["summary"]["comparison"]["reports"]
    assert [r["scorer"] for r in reports] == ["fake-ctc", "fake-whisper"]
    bad = client.post(f"/cases/c1/evidence/{eid}/candidates", headers=H,
                      json={"start": 3.0, "end": 1.0, "candidates": ["a", "b"]})
    assert bad.status_code == 422 and "Tratto non valido" in bad.json()["detail"]
    slow = client.post(f"/cases/c1/evidence/{eid}/candidates", headers=H,
                       json={"start": 1.0, "end": 3.0, "candidates": ["vado a casa"], "timeout_s": 0})
    assert slow.status_code in (200, 202)

    adapter = _job(client, "train_subject_adapter", subject_id="s1", vehicle_id="auto1",
                   train_config={"base_model": "fake/whisper-tiny"}, codecs=["g711_ulaw"])
    assert adapter["summary"]["adapter_id"].startswith("s1-lora-")
    assess = _job(client, "self_assess", subject_id="s1", vehicle_id="auto1", adapter="latest",
                  codecs=["g711_ulaw"])
    assert assess["summary"]["computable"] is True
    sa = client.get("/cases/c1/subjects/s1/self-assessment", headers=H)
    assert sa.status_code == 200 and "leave-one-recording-out" in sa.json()["summary_it"]
    assert "wer@adapted" in sa.json()["assessment"]["overall"]

    logs = client.get("/cases/c1/logs", headers=H).json()
    assert {"build_vehicle_profile", "build_subject_model", "clean", "transcribe", "score_candidates",
            "train_subject_adapter", "self_assess"} <= {p["path"].split("/")[1].rsplit("-", 2)[0] for p in logs}
    assert client.get("/cases/c1/subjects/nessuno/self-assessment", headers=H).status_code == 404
    assert client.get("/cases/c1/subjects/..%2F..%2Fetc/model", headers=H).status_code in (400, 404)


def test_failed_job_reports_the_reason(api):
    client = api["client"]
    r = client.post("/cases/c1/jobs", headers=H, json={"kind": "self_assess", "params": {"subject_id": "ignoto"}})
    state = _wait(client, r.json())
    assert state["status"] == "failed" and state["error"]
