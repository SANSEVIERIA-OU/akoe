# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
import json
import logging

import numpy as np
import pytest

from akoe.case import Case
from akoe.config import RecordingSource
from akoe.models import ArtifactRef, DatasetItem
from akoe.utils import sha256_file
from akoe.vehicle.audio import estimate_lag, write_audio
from akoe.vehicle.codecs import available_codecs
from akoe.vehicle.profile import NoiseSource, build_vehicle_profile, estimate_noise
from akoe.vehicle.simulate import (
    MANIFEST_FILE,
    ChannelSimConfig,
    ChannelSimulator,
    SNRDistribution,
    derive_seed,
    simulate_dataset,
    write_simulated,
)
from tests.vehicle_synth import car_noise, speech_like, synthetic_ir

SR = 16000


@pytest.fixture(scope="module")
def speech():
    return speech_like(SR, 3.0, seed=11, level=0.1)[0].astype(np.float32)


@pytest.fixture(scope="module")
def sim():
    return ChannelSimulator.from_arrays(
        SR,
        irs=[synthetic_ir(SR, direct_index=40), synthetic_ir(SR, direct_index=90, seed=3)],
        noises=[car_noise(SR, 5.0, seed=1), car_noise(SR, 1.0, seed=2, level=0.03)],
        config=ChannelSimConfig(snr=SNRDistribution(kind="uniform", low=0, high=15), gain_db=(-6, 0)),
        noise_labels=["idle", "short"],
    )


def test_same_seed_same_output_and_params(sim, speech):
    a, pa = sim.simulate(speech, SR, seed=123)
    b, pb = sim.simulate(speech, SR, seed=123)
    c, pc = sim.simulate(speech, SR, seed=124)
    assert np.array_equal(a, b) and pa == pb
    assert not np.array_equal(a, c)
    assert a.dtype == np.float32 and len(a) == len(speech)
    json.dumps(pa)  # parameters are JSON-serializable
    for key in ("seed", "ir", "noise", "gain_db", "codec_seed", "condition", "peak_limit_scale_db"):
        assert key in pa


def test_convolution_is_aligned_on_direct_path(speech):
    clicks = np.zeros(SR)
    clicks[[1000, 5000, 9000]] = 0.5
    for d in (40, 90):
        s = ChannelSimulator.from_arrays(SR, irs=[synthetic_ir(SR, direct_index=d)])
        y, p = s.simulate(clicks, SR, seed=0)
        assert p["ir"]["direct_path_index"] == d and p["noise"] is None
        for k in (1000, 5000, 9000):
            assert int(np.argmax(np.abs(y[k - 20:k + 20]))) + k - 20 == k
        y2, _ = s.simulate(speech, SR, seed=0)
        assert estimate_lag(speech, y2, 400)[0] == 0


def test_snr_and_noise_excerpt(sim, speech):
    seen_loop, snrs = set(), []
    for seed in range(30):
        _, p = sim.simulate(speech, SR, seed=seed)
        n = p["noise"]
        assert abs(n["snr_db_actual"] - n["snr_db_target"]) < 0.01
        assert 0 <= n["snr_db_target"] <= 15
        seen_loop.add((n["label"], n["looped"]))
        if not n["looped"]:
            assert 0 <= n["start_s"] <= 5.0 - 3.0
        snrs.append(n["snr_db_target"])
        assert -6 <= p["gain_db"] <= 0
    assert ("short", True) in seen_loop and ("idle", False) in seen_loop  # 1 s noise is looped
    assert np.std(snrs) > 2


def test_probabilities_and_peak_limit(speech):
    s = ChannelSimulator.from_arrays(
        SR, irs=[synthetic_ir(SR)], noises=[car_noise(SR, 4.0)],
        config=ChannelSimConfig(ir_probability=0.0, noise_probability=0.0, gain_db=(30, 30)),
    )
    y, p = s.simulate(speech, SR, seed=1)
    assert p["ir"] is None and p["noise"] is None
    assert np.max(np.abs(y)) <= 0.99 + 1e-6 and p["peak_limit_scale_db"] < 0
    with pytest.raises(ValueError):
        s.simulate(speech, 8000, seed=1)


def test_snr_distribution_from_case_estimate():
    x = speech_like(SR, 12.0, seed=2, level=0.1)[0] + car_noise(SR, 12.0, seed=3, level=0.01)
    est, _ = estimate_noise(x, SR, track_engine=False)
    dist = SNRDistribution.from_noise_estimate(est)
    assert dist.kind == "normal" and dist.mean == est.snr_db["p50"] and dist.std >= 2.0
    rng = np.random.default_rng(0)
    v = [dist.sample(rng) for _ in range(500)]
    assert abs(np.median(v) - dist.mean) < 1.5
    assert SNRDistribution(kind="choice", values=[5, 10]).sample(rng) in (5, 10)
    assert SNRDistribution(kind="fixed", mean=99).sample(rng) == 40  # clipped to max_db


def test_codec_in_channel(speech):
    if "g711_ulaw" not in available_codecs():
        pytest.skip("g711 unavailable")
    s = ChannelSimulator.from_arrays(SR, irs=[synthetic_ir(SR)], noises=[car_noise(SR, 4.0)],
                                     config=ChannelSimConfig(codec_probability=1.0, codecs=["g711_ulaw"]))
    y, p = s.simulate(speech, SR, seed=5)
    assert p["codec"]["codec"] == "g711_ulaw"
    assert p["condition"]["source"] == RecordingSource.TRANSMITTED.value
    assert p["condition"]["codec"] == "g711_ulaw" and len(y) == len(speech)
    assert s.describe()["codecs"]["active"] == ["g711_ulaw"]


def test_simulator_from_profile_verifies_hashes(tmp_path, caplog):
    ir = tmp_path / "ir.wav"
    write_audio(ir, synthetic_ir(48000, direct_index=120), 48000, subtype="FLOAT")
    nz = tmp_path / "idle.wav"
    write_audio(nz, car_noise(48000, 4.0), 48000)
    case = Case.create(tmp_path / "case", "c1")
    profile = build_vehicle_profile("V01", [ir], [NoiseSource(path=nz, label="idle")], case=case)
    s = ChannelSimulator.from_profile(profile, sr=SR, root=case.root)
    assert s.irs[0].direct_index == 40  # 120 samples at 48 kHz
    assert s.irs[0].sha256 == profile.impulse_responses[0].sha256
    assert s.describe()["noises"][0]["label"] == "idle"

    stored = profile.noise_recordings[0].artifact.resolve(case.root)
    stored.chmod(0o644)
    stored.write_bytes(stored.read_bytes()[:-2] + b"\x00\x01")
    with pytest.raises(ValueError, match="SHA-256"):
        ChannelSimulator.from_profile(profile, sr=SR, root=case.root)
    with caplog.at_level(logging.WARNING):
        empty = ChannelSimulator.from_profile(profile, sr=SR, root=case.root,
                                              config=ChannelSimConfig(noise_labels=["highway"]))
    assert empty.noises == [] and "Nessuna registrazione di rumore" in caplog.text


def _items(case, tmp_path):
    src = tmp_path / "clean.wav"
    write_audio(src, speech_like(SR, 6.0, seed=4)[0], SR)
    ev = case.import_evidence(src)
    return [
        DatasetItem(item_id=f"seg{i}", recording_id=ev.evidence_id, audio=ev.artifact,
                    start=a, end=b, text=f"frase {i}", verified=True)
        for i, (a, b) in enumerate([(0.0, 2.0), (2.0, 4.5), (4.5, 6.0)])
    ]


def test_dataset_is_reproducible_hashed_and_order_independent(tmp_path, sim):
    case = Case.create(tmp_path / "case", "c1")
    items = _items(case, tmp_path)
    m1, ref1 = simulate_dataset(sim, items, seed=7, case=case, name="run1", variants=2)
    assert len(m1.items) == 6 and ref1.verify(case.root)
    for it in m1.items:
        assert it.output.verify(case.root)
        assert it.seed == derive_seed(7, f"{it.source_item_id}#{it.variant}")
        assert it.params["n_samples"] == int(round((it.end - it.start) * SR))
        assert it.text.startswith("frase")
    m2, _ = simulate_dataset(sim, list(reversed(items)), seed=7, case=case, name="run2", variants=2)
    h1 = {i.item_id: i.output.sha256 for i in m1.items}
    h2 = {i.item_id: i.output.sha256 for i in m2.items}
    assert h1 == h2
    m3, _ = simulate_dataset(sim, items, tmp_path / "other", seed=8, root=case.root)
    assert {i.item_id: i.output.sha256 for i in m3.items} != h1
    assert m3.items[0].output.path.startswith(str(tmp_path))
    manifest = json.loads((case.dir("working") / "simulated" / "run1" / MANIFEST_FILE).read_text())
    assert manifest["master_seed"] == 7 and manifest["simulator"]["config"]["snr"]["high"] == 15


def test_dataset_rejects_tampered_input(tmp_path, sim):
    case = Case.create(tmp_path / "case", "c1")
    items = _items(case, tmp_path)
    bad = items[0].model_copy(update={"audio": ArtifactRef(path=items[0].audio.path, sha256="0" * 64,
                                                           kind="evidence")})
    with pytest.raises(ValueError, match="SHA-256"):
        simulate_dataset(sim, [bad], seed=1, case=case)


def test_write_simulated_sidecar(tmp_path, sim, speech):
    y, p = sim.simulate(speech, SR, seed=3)
    ref = write_simulated(tmp_path / "ex.wav", y, SR, p)
    side = json.loads((tmp_path / "ex.json").read_text())
    assert side["sha256"] == ref.sha256 == sha256_file(tmp_path / "ex.wav")
    assert side["params"]["seed"] == 3


def test_end_to_end_from_estimated_profile(tmp_path):
    """M3 done-when: VehicleProfile + clean segments -> reproducible simulated set (seed, hashes)."""
    from akoe.vehicle.profile import estimate_vehicle_profile

    case = Case.create(tmp_path / "case", "c1")
    rec = tmp_path / "intercettazione.wav"
    write_audio(rec, speech_like(SR, 12.0, seed=21)[0] + car_noise(SR, 12.0, seed=22, level=0.01), SR)
    profile = estimate_vehicle_profile("V01", case.import_evidence(rec), case=case, track_engine=False)
    config = ChannelSimConfig(snr=SNRDistribution.from_noise_estimate(profile.noise_estimate))
    sim = ChannelSimulator.from_profile(profile, sr=SR, root=case.root, config=config)
    assert sim.noises and not sim.irs
    items = _items(case, tmp_path)
    m1, _ = simulate_dataset(sim, items, seed=3, case=case, name="a")
    m2, _ = simulate_dataset(sim, items, seed=3, case=case, name="b")
    assert [i.output.sha256 for i in m1.items] == [i.output.sha256 for i in m2.items]
    assert all(i.params["noise"]["label"] == "estimated-nonspeech" for i in m1.items)
