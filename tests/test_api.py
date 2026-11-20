import time

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from forensicaudio.api.server import create_app


def _client(tmp_path):
    app = create_app(tmp_path / "cases", token="t0k")
    return TestClient(app), {"Authorization": "Bearer t0k"}


def _wav(path, seconds=1.0, sr=16000):
    t = np.arange(int(seconds * sr)) / sr
    sf.write(path, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)
    return path


def test_requires_token(tmp_path):
    client, _ = _client(tmp_path)
    assert client.get("/health").status_code == 401
    assert client.get("/health", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_case_import_spectrogram_and_verify_job(tmp_path):
    client, h = _client(tmp_path)
    assert client.post("/cases", json={"case_id": "c1"}, headers=h).status_code == 201
    assert client.post("/cases", json={"case_id": "../evil"}, headers=h).status_code == 400
    assert client.get("/cases", headers=h).json() == ["c1"]

    wav = _wav(tmp_path / "Registrazione locale.wav")
    with wav.open("rb") as f:
        r = client.post("/cases/c1/evidence", headers=h,
                        files={"file": ("Registrazione locale.wav", f, "audio/wav")},
                        data={"channel": "car", "source": "local", "pair_id": "evt-1"})
    assert r.status_code == 201, r.text
    item = r.json()
    assert item["original_name"] == "Registrazione locale.wav"
    assert item["condition"]["channel"] == "car" and item["pair_id"] == "evt-1"

    audio = client.get(f"/cases/c1/evidence/{item['evidence_id']}/audio", headers=h)
    assert audio.status_code == 200 and audio.content[:4] == b"RIFF"

    spec = client.get(f"/cases/c1/evidence/{item['evidence_id']}/spectrogram?n_fft=256&hop=64", headers=h).json()
    assert spec["params"]["n_fft"] == 256 and spec["evidence_sha256"] == item["artifact"]["sha256"]
    assert spec["shape"][0] == 129

    job = client.post("/cases/c1/jobs", json={"kind": "verify"}, headers=h).json()
    for _ in range(50):
        state = client.get(f"/jobs/{job['job_id']}", headers=h).json()
        if state["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    assert state["status"] == "done" and state["result"]["ok"] is True
    assert client.post("/cases/c1/jobs", json={"kind": "nope"}, headers=h).status_code == 400
