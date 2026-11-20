# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
import json

import numpy as np
import pytest
from scipy.signal import butter, fftconvolve, sosfiltfilt

from akoe.utils import sha256_file
from akoe.vehicle.audio import load_audio, resample, write_audio
from akoe.vehicle.measure import (
    PARAMS_FILE,
    SweepParams,
    deconvolve,
    exponential_sweep,
    main,
    measurement_signal,
)
from tests.vehicle_synth import synthetic_ir

SR = 16000


def _band_db(h, ir, sr, lo=150, hi=6000, nfft=8192):
    f = np.fft.rfftfreq(nfft, 1 / sr)
    band = (f > lo) & (f < hi)
    return 20 * np.log10(np.abs(np.fft.rfft(ir, nfft))[band] / np.abs(np.fft.rfft(h, nfft))[band])


def test_sweep_spans_requested_band():
    p = SweepParams(sample_rate=SR, duration_s=2.0, f1_hz=100, f2_hz=6000, repetitions=1)
    x = exponential_sweep(p)
    assert len(x) == 2 * SR and np.max(np.abs(x)) <= p.amplitude + 1e-12
    assert x[0] == 0 and abs(x[-1]) < 1e-3  # fades
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    inside = spec[(f >= 80) & (f <= 6600)].sum() / spec.sum()
    assert inside > 0.99
    sig = measurement_signal(p)
    assert len(sig) == int((p.lead_s + p.duration_s + p.gap_s) * SR)
    with pytest.raises(ValueError):
        SweepParams(sample_rate=SR, f2_hz=9000)


def test_deconvolve_recovers_known_ir_without_sync():
    p = SweepParams(sample_rate=SR, duration_s=2.0, f1_hz=50, f2_hz=7000, repetitions=2, gap_s=1.0, lead_s=0.5)
    h = synthetic_ir(SR, direct_index=30)
    rng = np.random.default_rng(0)
    wet = fftconvolve(measurement_signal(p), h)
    rec = np.concatenate([np.zeros(1234), wet]) + 1e-5 * rng.standard_normal(len(wet) + 1234)
    ir, info = deconvolve(rec, SR, p, ir_length_s=0.3, pre_s=0.005)

    assert info["repetitions_found"] == 2
    assert info["direct_path_index"] == 80
    assert info["repetition_min_correlation"] > 0.999
    assert info["peak_to_noise_db"] > 40
    # The response starts `pre` samples before the direct path.
    assert int(np.argmax(np.abs(ir))) == info["direct_path_index"]
    expected = np.pad(h[30:], (80, len(ir)))[: len(ir)]
    sos = butter(4, [100, 6000], btype="band", fs=SR, output="sos")
    a, b = sosfiltfilt(sos, expected), sosfiltfilt(sos, ir)
    assert np.corrcoef(a, b)[0, 1] > 0.999
    d = _band_db(h, ir, SR)
    assert np.all(np.abs(np.percentile(d, [1, 99])) < 0.5)


def test_deconvolve_recording_at_lower_rate_than_playback():
    p = SweepParams(sample_rate=48000, duration_s=2.0, f1_hz=50, f2_hz=20000, repetitions=1, gap_s=1.0, lead_s=0.5)
    h = synthetic_ir(SR, direct_index=30)
    rec = fftconvolve(resample(measurement_signal(p), 48000, SR), h)
    ir, info = deconvolve(rec, SR, p, ir_length_s=0.3, pre_s=0.005)
    assert info["sample_rate"] == SR
    d = _band_db(h, ir, SR, hi=6500)
    assert np.all(np.abs(np.percentile(d, [1, 99])) < 1.0)


def test_cli_generate_and_deconvolve(tmp_path, capsys):
    out = tmp_path / "sweep"
    assert main(["genera", "--out", str(out), "--sr", "16000", "--durata", "1.5", "--f2", "7000",
                 "--ripetizioni", "2", "--pausa", "1", "--anticipo", "0.5"]) == 0
    meta = json.loads((out / PARAMS_FILE).read_text())
    played = out / meta["file"]
    assert sha256_file(played) == meta["sha256"]
    assert "Sweep scritto" in capsys.readouterr().out

    signal, sr = load_audio(played)
    h = synthetic_ir(sr, direct_index=20)
    rec = tmp_path / "rec_locale.wav"
    write_audio(rec, 0.5 * np.concatenate([np.zeros(500), fftconvolve(signal, h)]), sr, subtype="FLOAT")
    ir_path = tmp_path / "ir.wav"
    assert main(["deconvolvi", str(rec), "--parametri", str(out / PARAMS_FILE), "--out", str(ir_path),
                 "--durata-ir", "0.25"]) == 0
    assert "Risposta all'impulso" in capsys.readouterr().out
    sidecar = json.loads(ir_path.with_suffix(".json").read_text())
    assert sidecar["sha256"] == sha256_file(ir_path)
    assert sidecar["source_recording"]["sha256"] == sha256_file(rec)
    assert sidecar["sweep_file_sha256"] == meta["sha256"]
    assert sidecar["deconvolution"]["repetitions_found"] == 2
