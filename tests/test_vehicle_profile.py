# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
import json
import os

import numpy as np
import pytest
from scipy.signal import welch

from akoe.case import Case
from akoe.config import Channel, RecordingSource
from akoe.models import RecordingCondition, VehicleProfile
from akoe.utils import sha256_file
from akoe.vehicle.audio import write_audio
from akoe.vehicle.profile import (
    PROFILE_FILE,
    ImpulseResponseSource,
    NoiseEstimateParams,
    NoiseSource,
    build_profile_from_manifest,
    build_vehicle_profile,
    detect_nonspeech,
    direct_path_index,
    estimate_noise,
    estimate_vehicle_profile,
    load_profile,
    track_engine_harmonics,
)
from tests.vehicle_synth import car_noise, speech_like, synthetic_ir

SR = 16000


@pytest.fixture
def measured_files(tmp_path):
    d = tmp_path / "misura"
    d.mkdir()
    ir = d / "V01_20261120_ir_guidatore_locale_01.wav"
    write_audio(ir, synthetic_ir(SR, direct_index=40), SR, subtype="FLOAT")
    ir.with_suffix(".json").write_text(json.dumps({"kind": "vehicle_ir", "deconvolution": {"method": "farina-ess"}}))
    idle = d / "V01_20261120_rumore_minimo-fin-chiusi-vent-off_locale_01.wav"
    write_audio(idle, car_noise(SR, 3.0, seed=1, level=0.01), SR)
    highway = d / "V01_20261120_rumore_autostrada-fin-chiusi-vent-on_locale_01.wav"
    write_audio(highway, car_noise(SR, 3.0, seed=2, level=0.05), SR)
    return d, ir, idle, highway


def test_direct_path_index_finds_first_strong_peak():
    h = np.zeros(200)
    h[50], h[60] = 0.8, 1.0  # a reflection stronger than the direct path
    assert direct_path_index(h, -6.0, sr=SR) == 50
    assert direct_path_index(synthetic_ir(SR, direct_index=40), sr=SR) == 40


def test_measured_profile_is_hashed_and_stored_read_only_in_case(tmp_path, measured_files):
    _, ir, idle, highway = measured_files
    case = Case.create(tmp_path / "case", "c1")
    profile = build_vehicle_profile(
        "V01",
        [ImpulseResponseSource(path=ir, position="driver", channel_path="local")],
        [NoiseSource(path=idle, label="idle", speed_kmh=0, windows_open=False, ventilation=False),
         NoiseSource(path=highway, label="highway", speed_kmh=130, windows_open=False, ventilation=True)],
        case=case,
        metadata={"operators": ["op1"]},
    )
    assert not profile.estimated
    assert len(profile.impulse_responses) == 1 and len(profile.noise_recordings) == 2
    for ref, src in [(profile.impulse_responses[0], ir), (profile.noise_recordings[0].artifact, idle)]:
        assert ref.verify(case.root)
        assert ref.sha256 == sha256_file(src)
        stored = ref.resolve(case.root)
        assert stored.parent.parent.name == "vehicle-V01" and not os.access(stored, os.W_OK)
    info = profile.ir_details[0]
    assert info.direct_path_index == 40 and info.sample_rate == SR and info.channel_path == "local"
    assert info.deconvolution["sidecar_sha256"] == sha256_file(ir.with_suffix(".json"))
    assert profile.noise_details[1].level_dbfs > profile.noise_details[0].level_dbfs
    assert profile.noise_recordings[1].speed_kmh == 130

    saved = case.dir("profiles") / "vehicle-V01" / PROFILE_FILE
    assert load_profile(saved).ir_details[0].sha256 == info.sha256
    base = VehicleProfile.model_validate_json(saved.read_text())  # still a plain VehicleProfile
    assert base.impulse_responses[0].sha256 == info.sha256

    again = build_vehicle_profile("V01", [ir], case=case)  # idempotent copy
    assert again.impulse_responses[0].path == profile.impulse_responses[0].path


def test_measured_profile_without_case_references_files_in_place(measured_files):
    _, ir, idle, _ = measured_files
    profile = build_vehicle_profile("V01", [ir], [NoiseSource(path=idle, label="idle")])
    assert profile.impulse_responses[0].path == str(ir.resolve())
    assert profile.impulse_responses[0].verify()


def test_profile_from_session_manifest(tmp_path, measured_files):
    d, ir, idle, highway = measured_files
    manifest = d / "misura.json"
    manifest.write_text(json.dumps({
        "vehicle_id": "V01",
        "vehicle": {"make_model": "Utilitaria", "year": 2019},
        "playback": {"loudspeaker": "attivo 3 pollici", "sweep_sha256": "abc"},
        "date": "2026-11-20",
        "operators": ["op1", "op2"],
        "microphone": {"position": "plafoniera", "device_model": "X"},
        "impulse_responses": [{"path": ir.name, "position": "driver", "channel_path": "local"}],
        "noise": [{"path": idle.name, "label": "idle", "speed_kmh": 0},
                  {"path": highway.name, "label": "highway", "speed_kmh": 130, "windows_open": False}],
    }))
    case = Case.create(tmp_path / "case", "c1")
    profile = build_profile_from_manifest(manifest, case=case)
    assert profile.vehicle_id == "V01"
    assert profile.measurement_metadata["manifest_sha256"] == sha256_file(manifest)
    assert profile.measurement_metadata["microphone"]["position"] == "plafoniera"
    assert [n.label for n in profile.noise_recordings] == ["idle", "highway"]
    assert profile.measurement_metadata["playback"]["sweep_sha256"] == "abc"  # extra keys kept


@pytest.fixture(scope="module")
def recording():
    dur = 20.0
    speech, intervals = speech_like(SR, dur, seed=3, level=0.1)
    noise = car_noise(SR, dur, seed=4, level=0.01, engine_f0=(30.0, 45.0), engine_level=0.01)
    return speech + noise, noise, intervals, dur


def test_noise_estimate_uses_only_nonspeech_frames(recording):
    x, noise, intervals, _ = recording
    est, noise_audio = estimate_noise(x, SR)
    speech_mask = np.zeros(len(x), dtype=bool)
    for a, b in intervals:
        speech_mask[int(a * SR):int(b * SR)] = True
    ns_mask = np.zeros(len(x), dtype=bool)
    for a, b in est.nonspeech_segments:
        ns_mask[int(a * SR):int(b * SR)] = True
    assert ns_mask.sum() > SR  # found real pauses
    assert (ns_mask & speech_mask).sum() / ns_mask.sum() < 0.01
    # Spectrum of the detected noise matches the true noise in the speech band.
    f, p = welch(noise, SR, window="hann", nperseg=512, noverlap=256)
    band = (f > 300) & (f < 3000)
    diff = np.asarray(est.psd_db)[band] - 10 * np.log10(p[band])
    assert np.all(np.abs(np.percentile(diff, [5, 95])) < 1.5)
    assert est.snr_db["p50"] > 10 and est.params == NoiseEstimateParams()
    assert len(noise_audio) >= SR and est.noise_level_dbfs < -30
    assert set(est.octave_bands_db) >= {"125", "1000", "4000"}


def test_nonspeech_detection_is_deterministic(recording):
    x = recording[0]
    a, b = detect_nonspeech(x, SR), detect_nonspeech(x, SR)
    assert np.array_equal(a["nonspeech"], b["nonspeech"]) and a["floor_db"] == b["floor_db"]


def test_engine_harmonics_tracking():
    dur = 12.0
    x = car_noise(SR, dur, seed=5, level=0.01, engine_f0=(30.0, 48.0), engine_level=0.03)
    eh = track_engine_harmonics(x, SR)
    truth = 30.0 + 18.0 * np.asarray(eh.times_s) / dur
    f0 = np.array([np.nan if v is None else v for v in eh.f0_hz])
    assert np.mean(np.abs(f0 - truth) < 1.5) > 0.8
    assert eh.summary["basis"] == "nonspeech_frames"
    assert 35 < eh.summary["f0_median_hz"] < 43
    flat = track_engine_harmonics(car_noise(SR, 6.0, seed=6, level=0.01), SR)
    assert sum(v is not None for v in flat.f0_hz) <= 1  # no engine, (almost) no detection


def test_estimated_profile_from_case_evidence(tmp_path, recording):
    x = recording[0]
    src = tmp_path / "intercettazione.wav"
    write_audio(src, x, SR)
    case = Case.create(tmp_path / "case", "c1")
    item = case.import_evidence(src, RecordingCondition(channel=Channel.CAR, source=RecordingSource.LOCAL))
    profile = estimate_vehicle_profile("V01", item, case=case)
    assert profile.estimated and profile.noise_estimate is not None
    assert profile.noise_estimate.source_sha256 == item.artifact.sha256
    assert profile.build["source"]["evidence_id"] == item.evidence_id
    noise_ref = profile.noise_recordings[0].artifact
    assert noise_ref.kind == "vehicle_noise_estimated" and noise_ref.verify(case.root)
    assert profile.noise_estimate.engine.summary["n_detected_used"] > 0
    saved_files = list((case.dir("profiles") / "vehicle-V01").glob("vehicle_profile_estimated-*.json"))
    assert len(saved_files) == 1 and not (case.dir("profiles") / "vehicle-V01" / PROFILE_FILE).exists()
    saved = load_profile(saved_files[0])
    assert saved.estimated and saved.noise_estimate.snr_db == profile.noise_estimate.snr_db
    assert case.verify() == []  # evidence untouched
    again = estimate_vehicle_profile("V01", item, case=case, track_engine=False)  # re-run reuses files
    assert again.noise_recordings[0].artifact == noise_ref.model_copy(
        update={"created_at": again.noise_recordings[0].artifact.created_at})
