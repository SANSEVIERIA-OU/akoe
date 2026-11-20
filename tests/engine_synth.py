"""Synthetic case material for the engine workflow tests (no real speech, no downloads).

- a subject folder with ``recordings.json``: car and phone recordings where the
  subject (speaker A) and another speaker (B) are labelled, plus one car
  recording with a plain-text transcription (to be aligned);
- an in-car evidence recording of A with engine harmonics, road noise and
  music played in the cabin, the original of that music (reference) and a
  transmitted (G.711) copy of the evidence;
- a synthetic cabin IR and a noise recording for a measured vehicle profile.
"""
from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import scipy.signal as ss

from tests import m6_synth
from tests.synth_voice import SPEAKER_A, SPEAKER_B, SR, bandlimit, speaker_voice, write_wav

A_TEXTS = ["vado a casa adesso", "poi passo da Brambilla", "domani si parte presto", "lascia stare la macchina"]
B_TEXTS = ["ma dove vai", "va bene ci vediamo"]


def _road_noise(n: int, seed: int, level: float = 0.02) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sos = ss.butter(2, 300, btype="low", fs=SR, output="sos")
    return (ss.sosfilt(sos, rng.normal(0, level, n)) * 3).astype(np.float32)


def _dialogue(seed: int, seconds_a: float = 2.0, seconds_b: float = 1.0) -> tuple[np.ndarray, list[dict]]:
    """A, B, A, B turns with labelled JSON segments."""
    parts, segs, t = [], [], 0.0
    for k, (spk, label, dur) in enumerate([(SPEAKER_A, "A", seconds_a), (SPEAKER_B, "B", seconds_b),
                                           (SPEAKER_A, "A", seconds_a), (SPEAKER_B, "B", seconds_b)]):
        parts.append(speaker_voice(spk, seconds=dur, seed=seed + k))
        texts = A_TEXTS if label == "A" else B_TEXTS
        segs.append({"start": round(t, 3), "end": round(t + dur, 3), "speaker": label,
                     "text": texts[(seed + k) % len(texts)]})
        t += dur
    return np.concatenate(parts), segs


def make_subject_folder(dst: Path, *, n_car: int = 4, n_phone: int = 2) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    index = []
    for k in range(n_car):
        y, segs = _dialogue(10 + 7 * k)
        y = y + _road_noise(len(y), 100 + k)
        write_wav(dst / f"car{k}.wav", y)
        (dst / f"car{k}.json").write_text(json.dumps({"segments": segs}))
        index.append({"audio": f"car{k}.wav", "transcript": f"car{k}.json", "channel": "car", "source": "local",
                      "vocal_effort": "normal", "subject_speaker": "A"})
    for k in range(n_phone):
        y, segs = _dialogue(50 + 7 * k)
        y8 = librosa.resample(bandlimit(y, 300, 3400).astype(np.float64), orig_sr=SR, target_sr=8000,
                              res_type="soxr_hq")
        write_wav(dst / f"tel{k}.wav", y8.astype(np.float32), sr=8000)
        (dst / f"tel{k}.json").write_text(json.dumps({"segments": segs}))
        index.append({"audio": f"tel{k}.wav", "transcript": f"tel{k}.json", "channel": "phone",
                      "source": "other", "codec": "amr-nb", "vocal_effort": "normal", "subject_speaker": "A"})
    y = speaker_voice(SPEAKER_A, seconds=3.0, seed=90) + _road_noise(3 * SR, 190)
    write_wav(dst / "car_text.wav", y)
    (dst / "car_text.txt").write_text("Vado a casa adesso. Poi passo da Brambilla.")
    index.append({"audio": "car_text.wav", "transcript": "car_text.txt", "channel": "car", "source": "local",
                  "vocal_effort": "normal"})
    (dst / "recordings.json").write_text(json.dumps({"recordings": index}))
    return dst


def make_evidence(dst: Path, seconds: float = 8.0) -> dict[str, Path]:
    """Evidence (local), transmitted copy and music reference."""
    from forensicaudio.vehicle.codecs import apply_codec

    dst.mkdir(parents=True, exist_ok=True)
    n = int(seconds * SR)
    voice = speaker_voice(SPEAKER_A, seconds=seconds, seed=7)
    gate = np.zeros(n, dtype=np.float32)
    for a, b in ((1.0, 3.0), (4.5, 6.5)):
        gate[int(a * SR):int(b * SR)] = 1.0
    gate = np.convolve(gate, np.hanning(321) / np.hanning(321).sum(), mode="same").astype(np.float32)
    engine, _ = m6_synth.engine_noise(seconds, seed=3, level=0.06)
    road = m6_synth.colored_noise(seconds, seed=4, level=0.01)
    music = m6_synth.music_like(seconds, seed=5, level=0.05)
    cabin_music = ss.fftconvolve(music, m6_synth.cabin_rir(seed=6))[:n]
    y = voice * gate + engine + road + 0.6 * cabin_music
    y = (y / max(1.0, np.max(np.abs(y)) / 0.9)).astype(np.float32)
    paths = {"evidence": write_wav(dst / "auto_evento1_locale.wav", y),
             "reference": write_wav(dst / "brano_radio.wav", music.astype(np.float32))}
    tx, _ = apply_codec(y, SR, "g711_ulaw", rng=0)
    tx8 = librosa.resample(tx.astype(np.float64), orig_sr=SR, target_sr=8000, res_type="soxr_hq")
    paths["transmitted"] = write_wav(dst / "auto_evento1_trasmessa.wav", tx8.astype(np.float32), sr=8000)
    return paths


def make_vehicle_measurements(dst: Path) -> dict[str, Path]:
    from tests.vehicle_synth import car_noise, synthetic_ir

    dst.mkdir(parents=True, exist_ok=True)
    return {"ir": write_wav(dst / "ir_guidatore.wav", synthetic_ir(SR).astype(np.float32), subtype="FLOAT"),
            "noise": write_wav(dst / "rumore_minimo.wav", car_noise(SR, 4.0, engine_f0=30.0, engine_level=0.02)
                               .astype(np.float32))}
