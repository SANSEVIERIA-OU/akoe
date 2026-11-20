# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Measure the neural listening enhancers (issue #59) and the VAD options
(issue #60) on car-noise mixtures built from real Italian speech.

Speech: the Common Voice clip shipped with ``speechbrain/asr-wav2vec2-
commonvoice-it`` (pinned in ``tests/real_audio.py``) and, when present, the
MInDS-14 Italian clips in ``/tmp/whisper_test_audio`` (8 kHz, upsampled).
Noise: engine harmonics plus low-frequency-tilted road noise from
``tests/m6_synth.py``; cabin reverberation from ``m6_synth.cabin_rir``.
Nothing is written but the JSON results; no audio is kept.

Enhancement metrics (per clip, condition and backend): SI-SDR and its
improvement over the noisy mixture, ESTOI (``pystoi``, extended), residual
noise attenuation in the speech-free padding, processing time and real-time
factor. VAD metrics: frame-level (10 ms) precision, recall and F1 against
the speech regions of the clean signal, for the robust (deterministic) VAD,
Silero VAD and FireRedVAD.

Run (models must already be cached when ``HF_HUB_OFFLINE=1``)::

    flock /tmp/akoe-heavy.lock env HF_HUB_OFFLINE=1 PYTHONPATH=src:. \\
        .venv/bin/python scripts/measure_neural_cleaning.py --out /tmp/akoe_neural_results.json
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

from tests import m6_synth

SR = 16000
PAD_S = 1.0
MINDS_DIR = Path("/tmp/whisper_test_audio")


def load_speech(max_minds: int) -> list[tuple[str, np.ndarray, str]]:
    from tests.real_audio import italian_example

    audio, sr, text = italian_example()
    clips = [("cv_example_it", resample_poly(audio.astype(np.float64), SR, sr) if sr != SR
              else audio.astype(np.float64), text)]
    meta = MINDS_DIR / "minds_meta.json"
    if meta.exists():
        for m in json.loads(meta.read_text())[:max_minds]:
            a, s = sf.read(MINDS_DIR / m["file"], dtype="float64")
            clips.append((m["file"].removesuffix(".wav"), resample_poly(a, SR, s), m["text"]))
    return [(n, a / (np.max(np.abs(a)) + 1e-12) * 0.3, t) for n, a, t in clips]


def car_noise(n: int, seed: int) -> np.ndarray:
    dur = n / SR + 0.01
    engine, _ = m6_synth.engine_noise(dur, seed=seed, level=0.1)
    road = m6_synth.colored_noise(dur, seed=seed + 1, level=0.05)
    return (engine + road)[:n]


def mix(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    active = speech[np.abs(speech) > 1e-4]
    g = np.sqrt(np.mean(active ** 2) / np.mean(noise ** 2) / 10 ** (snr_db / 10))
    return speech + g * noise


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    ref = ref - ref.mean()
    est = est - est.mean()
    a = np.dot(est, ref) / max(np.dot(ref, ref), 1e-20)
    t = a * ref
    return float(10 * np.log10(np.sum(t ** 2) / max(np.sum((est - t) ** 2), 1e-20)))


def estoi(ref: np.ndarray, est: np.ndarray) -> float:
    from pystoi import stoi

    return float(stoi(ref, est, SR, extended=True))


def pad_attenuation(noisy: np.ndarray, enh: np.ndarray) -> float:
    k = int(PAD_S * SR * 0.8)
    idx = np.r_[0:k, len(noisy) - k:len(noisy)]
    return float(10 * np.log10(np.mean(noisy[idx] ** 2) / max(np.mean(enh[idx] ** 2), 1e-20)))


def build_conditions(clips, snrs) -> list[dict]:
    items = []
    for ci, (name, sp, _) in enumerate(clips):
        dry = np.concatenate([np.zeros(int(PAD_S * SR)), sp, np.zeros(int(PAD_S * SR))])
        noise = car_noise(len(dry), seed=100 + ci)
        for snr in snrs:
            items.append({"clip": name, "condition": f"dry_snr{snr:g}", "ref": dry,
                          "noisy": mix(dry, noise, snr)})
        wet = fftconvolve(dry, m6_synth.cabin_rir(seed=ci))[: len(dry)]
        items.append({"clip": name, "condition": "cabin_snr5", "ref": wet, "noisy": mix(wet, noise, 5.0)})
    return items


def measure_enhancers(items, backends, wer_model=None) -> dict:
    from akoe.stages import neural_clean as nc

    rows = []
    for it in items:
        it["noisy_si_sdr"] = si_sdr(it["ref"], it["noisy"])
        it["noisy_estoi"] = estoi(it["ref"], it["noisy"])
    for name in backends:
        enh = nc.get_enhancer(name, local_files_only=True)
        t0 = time.perf_counter()
        prov = enh.load()
        load_s = time.perf_counter() - t0
        print(f"[{name}] loaded in {load_s:.1f} s", flush=True)
        for it in items:
            y, p = nc.enhance(it["noisy"], SR, enh)
            row = {"backend": name, "clip": it["clip"], "condition": it["condition"],
                   "si_sdr": si_sdr(it["ref"], y), "noisy_si_sdr": it["noisy_si_sdr"],
                   "estoi": estoi(it["ref"], y), "noisy_estoi": it["noisy_estoi"],
                   "pad_attenuation_db": pad_attenuation(it["noisy"], y),
                   "processing_s": p["processing_s"], "audio_s": p["audio_s"],
                   "rtf": p["real_time_factor"]}
            row["si_sdr_improvement"] = row["si_sdr"] - row["noisy_si_sdr"]
            row["estoi_improvement"] = row["estoi"] - row["noisy_estoi"]
            if wer_model is not None:
                it.setdefault("wer_noisy", wer_model(it["noisy"], it.get("text", "")))
                row["wer"] = wer_model(y, it.get("text", ""))
                row["wer_noisy"] = it["wer_noisy"]
            rows.append(row)
            print(f"  {it['clip']:>14} {it['condition']:>11}  SI-SDR {row['noisy_si_sdr']:6.2f} -> "
                  f"{row['si_sdr']:6.2f}  ESTOI {row['noisy_estoi']:.3f} -> {row['estoi']:.3f}  "
                  f"rtf {row['rtf']:.2f}", flush=True)
        rows.append({"backend": name, "load_s": load_s, "provenance": prov.as_params()})
        del enh, prov
        gc.collect()
    return {"rows": rows}


def vad_truth(speech: np.ndarray, n_frames: int, hop: int = 160, frame: int = 480) -> np.ndarray:
    """Clean-speech activity per 10 ms frame: energy within 40 dB of the
    loudest frame, gaps under 0.3 s filled, runs under 0.1 s dropped."""
    from scipy import ndimage

    x = np.pad(speech, (0, frame))
    e = np.array([np.sum(x[i * hop: i * hop + frame] ** 2) for i in range(n_frames)])
    act = 10 * np.log10(e + 1e-20) > 10 * np.log10(e.max() + 1e-20) - 40
    act = ndimage.binary_closing(act, np.ones(31, bool))
    lab, k = ndimage.label(act)
    for j in range(1, k + 1):
        if np.sum(lab == j) < 10:
            act[lab == j] = False
    return act


def frames_from_segments(segs, n_frames: int, hop_s: float = 0.01) -> np.ndarray:
    t = (np.arange(n_frames) + 0.5) * hop_s
    out = np.zeros(n_frames, bool)
    for a, b in segs:
        out |= (t >= a) & (t < b)
    return out


def prf(pred: np.ndarray, truth: np.ndarray) -> dict:
    tp = float(np.sum(pred & truth))
    p = tp / max(float(pred.sum()), 1.0)
    r = tp / max(float(truth.sum()), 1.0)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12),
            "false_alarm_rate": float(np.sum(pred & ~truth)) / max(float((~truth).sum()), 1.0)}


def measure_vads(clips, snrs, methods) -> dict:
    from akoe.stages import vad
    from akoe.stages.vad_neural import get_detector

    rng = np.random.default_rng(7)
    parts = []
    for _, sp, _ in clips:
        parts += [np.zeros(int(rng.uniform(1.0, 3.0) * SR)), sp]
    parts.append(np.zeros(int(1.5 * SR)))
    speech = np.concatenate(parts)
    n_frames = len(speech) // 160
    truth = vad_truth(speech, n_frames)
    noise = car_noise(len(speech), seed=5)
    detectors = {m: (None if m == "robust" else get_detector(m, local_files_only=True)
                     if m == "firered" else get_detector(m)) for m in methods}
    synth_segs = [(1.0, 2.6), (4.0, 5.2), (7.0, 9.0), (10.5, 11.3)]
    synth = m6_synth.speech_like(12.0, synth_segs, seed=1, level=0.1)
    synth_truth = frames_from_segments(synth_segs, 1200)
    rows = []
    scenarios = [(f"real_it_snr{snr:g}", mix(speech, noise, snr), truth) for snr in snrs]
    scenarios.append(("real_it_clean", speech + 1e-4 * noise, truth))
    synth_noise = car_noise(len(synth), seed=9)
    scenarios += [(f"synthetic_voice_snr{snr:g}", mix(synth, synth_noise, snr), synth_truth) for snr in snrs]
    for label, x, tr in scenarios:
        for m, det in detectors.items():
            t0 = time.perf_counter()
            segs, _ = vad.detect_regions(x, SR, method=m, detector=det)
            el = time.perf_counter() - t0
            row = {"scenario": label, "method": m, "seconds": len(x) / SR, "processing_s": el,
                   **prf(frames_from_segments(segs, len(tr)), tr)}
            rows.append(row)
            print(f"  VAD {label:>22} {m:>8}  P {row['precision']:.3f} R {row['recall']:.3f} "
                  f"F1 {row['f1']:.3f} FA {row['false_alarm_rate']:.3f}  {el:.2f}s", flush=True)
    return {"rows": rows, "speech_ratio_truth": float(truth.mean()),
            "total_seconds": len(speech) / SR}


def make_wer_model(model_id: str):
    import jiwer
    from faster_whisper import WhisperModel

    model = WhisperModel(model_id, device="cpu", compute_type="int8", local_files_only=True)

    def norm(t: str) -> str:
        return " ".join("".join(c.lower() if c.isalnum() or c.isspace() else " " for c in t).split())

    def wer(audio: np.ndarray, text: str) -> float | None:
        if not text:
            return None
        segs, _ = model.transcribe(audio.astype(np.float32), language="it", beam_size=1,
                                   condition_on_previous_text=False, vad_filter=False)
        hyp = " ".join(s.text for s in segs)
        return float(jiwer.wer(norm(text), norm(hyp) or "<vuoto>"))

    return wer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backends", nargs="*", default=["gtcrn", "deepfilternet3", "mossformer2_se_48k",
                                                      "unipase"])
    ap.add_argument("--vads", nargs="*", default=["robust", "silero", "firered"])
    ap.add_argument("--snrs", nargs="*", type=float, default=[0.0, 5.0, 10.0])
    ap.add_argument("--max-minds", type=int, default=6)
    ap.add_argument("--wer-model", default=None, help="faster-whisper model id for the content check")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    import torch

    torch.set_num_threads(args.threads)
    clips = load_speech(args.max_minds)
    print(f"{len(clips)} clips, {sum(len(a) for _, a, _ in clips) / SR:.1f} s of speech", flush=True)
    results = {"clips": [n for n, _, _ in clips], "snrs": args.snrs, "threads": args.threads}
    if args.vads:
        results["vad"] = measure_vads(clips, args.snrs, args.vads)
    if args.backends:
        items = build_conditions(clips, args.snrs)
        texts = {n: t for n, _, t in clips}
        for it in items:
            it["text"] = texts[it["clip"]]
        wer_model = make_wer_model(args.wer_model) if args.wer_model else None
        results["enhancement"] = measure_enhancers(items, args.backends, wer_model)
    args.out.write_text(json.dumps(results, indent=1, default=str))
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
