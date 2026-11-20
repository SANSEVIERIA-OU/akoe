# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""CLI commands mirroring the engine workflows (fake backends)."""
import json

import pytest
from typer.testing import CliRunner

from akoe import engine
from akoe.cli import app
from tests.engine_synth import make_evidence, make_subject_folder
from tests.fakes import FakeProvider


@pytest.fixture()
def fake_provider():
    engine.set_backend_provider(FakeProvider())
    yield
    engine.set_backend_provider(None)


def _ok(result):
    assert result.exit_code == 0, result.output
    return result.output


def test_cli_case_workflow(tmp_path, fake_provider):
    runner = CliRunner()
    root = tmp_path / "caso"
    folder = make_subject_folder(tmp_path / "soggetto", n_car=3, n_phone=1)
    ev = make_evidence(tmp_path / "reperti")
    _ok(runner.invoke(app, ["case", "init", str(root), "--case-id", "caso-cli"]))
    out = _ok(runner.invoke(app, ["case", "import", str(root), str(ev["evidence"]), "--channel", "car",
                                  "--source", "local"]))
    eid = out.split()[0]

    out = _ok(runner.invoke(app, ["vehicle-profile", str(root), "--vehicle-id", "auto1", "--estimate-from", eid]))
    assert "build_vehicle_profile: completato" in out
    out = _ok(runner.invoke(app, ["subject-model", str(root), "--subject-id", "s1", "--folder", str(folder),
                                  "--holdout", "0.3", "--certify-by", "operatore1", "--no-timbre"]))
    assert "leave-one-recording-out" in out
    out = _ok(runner.invoke(app, ["clean", str(root), eid, "--variant", "denoise_offline", "--variant", "bands"]))
    assert "denoise_offline: outputs/" in out and "bands: outputs/" in out
    res = json.loads(_ok(runner.invoke(app, ["transcribe", str(root), "--evidence", eid, "--json"])))
    assert res["workflow"] == "transcribe" and res["summary"]["n_segments"] >= 1
    out = _ok(runner.invoke(app, ["candidates", str(root), eid, "--start", "1", "--end", "3",
                                  "-c", "vado a casa", "-c", "vado a cassa", "--threshold", "fake-ctc=2"]))
    assert "fake-ctc (soglia 2.0)" in out and "fake-whisper" in out
    out = _ok(runner.invoke(app, ["train-adapter", str(root), "--subject-id", "s1", "--vehicle-id", "auto1",
                                  "--base-model", "fake/whisper-tiny"]))
    assert "train_subject_adapter: completato" in out
    out = _ok(runner.invoke(app, ["self-assess", str(root), "--subject-id", "s1", "--vehicle-id", "auto1",
                                  "--adapter", "latest"]))
    assert "Autovalutazione del caso caso-cli" in out and "leave-one-recording-out" in out

    bad = runner.invoke(app, ["transcribe", str(root)])
    assert bad.exit_code == 1 and "Errore" in bad.output
