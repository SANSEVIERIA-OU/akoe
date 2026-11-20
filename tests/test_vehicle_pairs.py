# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
import json

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import butter, resample as fft_resample, sosfiltfilt

from akoe.case import Case, CaseError
from akoe.config import RecordingSource
from akoe.models import RecordingCondition
from akoe.vehicle.audio import resample, write_audio
from akoe.vehicle.codecs import apply_codec
from akoe.vehicle.pairs import PairParams, align_pair, analyze_pair, analyze_case_pair, apply_alignment
from tests.vehicle_synth import car_noise, speech_like

SR = 16000
DUR = 30.0
OFFSET = 5939 / SR  # odd number of samples: half a sample at 8 kHz
DROPOUT = (12.0, 12.25)


@pytest.fixture(scope="module")
def local():
    s, _ = speech_like(SR, DUR, seed=5, level=0.1, pause_prob=0.2)
    return s + car_noise(SR, DUR, seed=6, level=0.003)


def make_tx(local, drift_ppm=0.0, dropout=DROPOUT):
    """Transmission: 300-3400 Hz, -6 dB, optional clock drift, delay, 8 kHz G.711 A-law, a dropout."""
    tx = sosfiltfilt(butter(6, [300, 3400], btype="band", fs=SR, output="sos"), local) * 0.5
    if drift_ppm:
        tx = fft_resample(tx, int(round(len(tx) * (1 + drift_ppm * 1e-6))))
    tx = np.concatenate([np.zeros(int(round(OFFSET * SR))), tx])
    tx8, _ = apply_codec(resample(tx, SR, 8000), 8000, "g711_alaw")
    if dropout:
        tx8[int((dropout[0] + OFFSET) * 8000):int((dropout[1] + OFFSET) * 8000)] = 0
    return tx8


@pytest.fixture(scope="module")
def report(local):
    return analyze_pair(local, SR, make_tx(local), 8000)


def test_alignment_offset_and_confidence(report):
    a = report["alignment"]
    assert abs(a["offset_s"] - OFFSET) < 0.5 / 8000
    assert a["confidence"] > 0.8
    assert abs(a["coarse_offset_s"] - OFFSET) < 0.02
    assert not a["drift"]["detected"] and abs(a["drift"]["ppm"]) < 10
    assert a["n_reliable_windows"] >= 5


def test_channel_estimate(report):
    ch = report["channel"]
    assert abs(ch["active_level_difference_db"] - (-6.0)) < 1.5
    bw = ch["bandwidth_hz"]
    assert 200 <= bw["low"] <= 400 and 3200 <= bw["high"] <= 3700
    assert abs(bw["reference_gain_db"] - (-6.0)) < 0.5
    assert ch["mean_coherence_speech_band"] > 0.95
    bands = {b["center_hz"]: b for b in ch["third_octave_bands"]}
    assert abs(bands[1000.0]["gain_db_power"] + 6.0) < 0.5 and bands[1000.0]["coherence"] > 0.95
    assert bands[99.2]["gain_db_power"] < -20  # outside the transmitted band
    d = ch["dropouts"]
    assert d["n_events"] == 1
    ev = d["events"][0]
    assert abs(ev["start_s"] - DROPOUT[0]) <= 0.04 and abs(ev["end_s"] - DROPOUT[1]) <= 0.04
    assert ev["kind"] == "silence"
    assert any("perdite" in w for w in report["warnings"])


def test_report_is_json_serializable(report):
    text = json.dumps(report, allow_nan=False)
    assert json.loads(text)["version"] == 1
    assert report["params"] == PairParams().model_dump(mode="json")


def test_drift_is_detected_and_compensated(local):
    r = analyze_pair(local, SR, make_tx(local, drift_ppm=300.0), 8000)
    d = r["alignment"]["drift"]
    assert d["detected"] and abs(d["ppm"] - 300.0) < 15
    assert abs(r["alignment"]["offset_s"] - OFFSET) < 0.0005
    assert r["alignment"]["confidence"] > 0.8  # after compensation
    assert r["channel"]["mean_coherence_speech_band"] > 0.9  # spectra estimated after compensation
    assert r["channel"]["dropouts"]["n_events"] == 1
    assert any("Deriva" in w for w in r["warnings"])


def test_apply_alignment_maps_transmitted_onto_local_timeline(local):
    tx = make_tx(local, dropout=None)
    al = align_pair(local, SR, tx, 8000)
    l8 = resample(local, SR, 8000)
    y, valid = apply_alignment(tx, 8000, al, len(l8))
    assert len(y) == len(l8) and valid.mean() > 0.99
    band = butter(4, [500, 3000], btype="band", fs=8000, output="sos")
    a, b = sosfiltfilt(band, l8), sosfiltfilt(band, y)
    assert np.corrcoef(a[valid], b[valid])[0, 1] > 0.95


def test_unrelated_recordings_have_low_confidence(local):
    other, _ = speech_like(8000, DUR, seed=99, level=0.05)
    r = analyze_pair(local, SR, other, 8000)
    assert r["alignment"]["confidence"] < 0.3
    assert any("poco affidabile" in w for w in r["warnings"])


def test_case_pair_report(tmp_path, local):
    loc, tx = tmp_path / "locale.wav", tmp_path / "trasmesso.wav"
    write_audio(loc, local[: 12 * SR], SR)
    sf.write(str(tx), make_tx(local[: 12 * SR], dropout=None), 8000, subtype="ULAW")
    case = Case.create(tmp_path / "case", "c1")
    case.import_evidence(loc, RecordingCondition(source=RecordingSource.LOCAL), pair_id="evt1")
    ev_tx = case.import_evidence(tx, RecordingCondition(source=RecordingSource.TRANSMITTED), pair_id="evt1")
    report, ref = analyze_case_pair(case, "evt1")
    assert ref.verify(case.root) and ref.kind == "pair_report"
    assert report["pair_id"] == "evt1"
    assert report["transmitted"]["evidence_id"] == ev_tx.evidence_id
    assert report["transmitted"]["codec"]["codec"] == "g711_ulaw"
    assert report["transmitted"]["sha256"] == ev_tx.artifact.sha256
    assert abs(report["alignment"]["offset_s"] - OFFSET) < 0.001
    with pytest.raises(CaseError):
        analyze_case_pair(case, "missing")
