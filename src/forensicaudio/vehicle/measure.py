"""Cabin impulse response measurement with an exponential sine sweep.

Method: Farina, "Simultaneous measurement of impulse response and distortion
with a swept-sine technique" (AES 108th Convention, 2000). The sweep

    x(t) = A sin( 2 pi f1 T / L * (exp(t L / T) - 1) ),   L = ln(f2 / f1)

is played at the driver's head position and recorded by the covert device.
Convolving the recording with the inverse filter (the time-reversed sweep with
a +6 dB/octave amplitude envelope) yields the linear impulse response; the
harmonic distortion products land before it in time and are discarded.

Operator procedure: docs/runbook_misura_veicolo.md. Command line:

    python -m forensicaudio.vehicle.measure genera --out DIR
    python -m forensicaudio.vehicle.measure deconvolvi REC.wav \
        --parametri DIR/sweep_params.json --out ir.wav
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import fftconvolve

from forensicaudio.utils import sha256_file
from forensicaudio.vehicle.audio import environment_info, load_audio, resample, to_mono, write_audio

log = logging.getLogger(__name__)

PARAMS_FILE = "sweep_params.json"


@dataclass(frozen=True)
class SweepParams:
    sample_rate: int = 48000
    duration_s: float = 10.0
    f1_hz: float = 50.0
    f2_hz: float = 20000.0
    amplitude: float = 0.5
    fade_in_s: float = 0.05
    fade_out_s: float = 0.01
    repetitions: int = 3
    gap_s: float = 3.0  # silence after each sweep: longer than reverberation + device latency
    lead_s: float = 1.0  # silence before the first sweep

    def __post_init__(self) -> None:
        if not 0 < self.f1_hz < self.f2_hz < self.sample_rate / 2:
            raise ValueError("need 0 < f1 < f2 < sample_rate / 2")
        if self.duration_s <= 0 or self.repetitions < 1 or not 0 < self.amplitude <= 1:
            raise ValueError("invalid sweep duration, repetitions or amplitude")

    @property
    def rate(self) -> float:
        """L = ln(f2 / f1)."""
        return float(np.log(self.f2_hz / self.f1_hz))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SweepParams":
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})


def exponential_sweep(p: SweepParams) -> np.ndarray:
    """One sweep (float64, no surrounding silence), with half-Hann fades."""
    n = int(round(p.duration_s * p.sample_rate))
    t = np.arange(n) / p.sample_rate
    phase = 2 * np.pi * p.f1_hz * p.duration_s / p.rate * (np.exp(t * p.rate / p.duration_s) - 1)
    x = p.amplitude * np.sin(phase)
    n_in, n_out = int(p.fade_in_s * p.sample_rate), int(p.fade_out_s * p.sample_rate)
    if n_in > 0:
        x[:n_in] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(n_in) / n_in)
    if n_out > 0:
        x[n - n_out:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(1, n_out + 1) / n_out)
    return x


def measurement_signal(p: SweepParams) -> np.ndarray:
    """The playback file: lead silence, then `repetitions` x (sweep + gap)."""
    sweep = exponential_sweep(p)
    gap = np.zeros(int(round(p.gap_s * p.sample_rate)))
    parts = [np.zeros(int(round(p.lead_s * p.sample_rate)))]
    for _ in range(p.repetitions):
        parts += [sweep, gap]
    return np.concatenate(parts)


def reference_sweep(p: SweepParams, sr: int | None = None) -> np.ndarray:
    """The sweep as the recorder would see it at `sr` (band-limited by resampling)."""
    sweep = exponential_sweep(p)
    return sweep if sr is None or sr == p.sample_rate else resample(sweep, p.sample_rate, sr)


def inverse_filter(p: SweepParams, sr: int | None = None) -> np.ndarray:
    """Farina inverse filter at `sr`, normalized for unit in-band gain."""
    sr = sr or p.sample_rate
    sweep = reference_sweep(p, sr)
    t = np.arange(len(sweep)) / sr
    inv = sweep[::-1] * np.exp(-t * p.rate / p.duration_s)
    nfft = 1 << int(np.ceil(np.log2(2 * len(sweep))))
    h = np.abs(np.fft.rfft(sweep, nfft) * np.fft.rfft(inv, nfft))
    f = np.fft.rfftfreq(nfft, 1 / sr)
    f_hi = min(p.f2_hz, 0.45 * sr)
    band = (f >= 2 * p.f1_hz) & (f <= f_hi / 2)
    return inv / float(np.median(h[band]))


def deconvolve(
    recording: np.ndarray,
    sr: int,
    p: SweepParams,
    *,
    ir_length_s: float = 0.5,
    pre_s: float = 0.02,
    repetitions: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Impulse response from a recording of `measurement_signal(p)`.

    Playback and recording need not be synchronized: each repetition's linear
    response is located at its main peak and cropped from `pre_s` before it.
    Repetitions are averaged (aligned on their peaks). Returns (ir, info).
    """
    rec = to_mono(recording)
    inv = inverse_filter(p, sr)
    full = fftconvolve(rec, inv)
    n_rep = repetitions or p.repetitions
    pre = int(round(pre_s * sr))
    length = int(round(ir_length_s * sr))
    excl = int(round((p.duration_s + p.gap_s) * sr / 2))

    work = np.abs(full)
    peaks: list[int] = []
    for _ in range(n_rep):
        k = int(np.argmax(work))
        if work[k] <= 0:
            break
        peaks.append(k)
        work[max(0, k - excl): k + excl] = 0.0
    top = max(abs(full[k]) for k in peaks) if peaks else 0.0
    if top <= 0:
        raise ValueError("no impulse response found: is the sweep in the recording?")
    kept = sorted(k for k in peaks if abs(full[k]) >= 0.25 * top)
    if len(kept) < len(peaks):
        log.warning(
            "Trovate %d ripetizioni dello sweep su %d attese: le altre sono troppo deboli.",
            len(kept), n_rep,
        )

    segs = []
    for k in kept:
        seg = full[max(0, k - pre): k - pre + length]
        if k - pre < 0:
            seg = np.concatenate([np.zeros(pre - k), seg])
        segs.append(np.pad(seg, (0, length - len(seg))))
    stack = np.vstack(segs)
    ir = stack.mean(axis=0)

    info: dict[str, Any] = {
        "method": "farina-ess",
        "sample_rate": int(sr),
        "ir_length_s": ir_length_s,
        "pre_s": pre_s,
        "direct_path_index": pre,
        "repetitions_expected": n_rep,
        "repetitions_found": len(kept),
        "peak_times_s": [round(k / sr - (len(inv) - 1) / sr, 6) for k in kept],
        "sweep": p.to_dict(),
    }
    info.update(ir_quality(ir, sr, pre=pre, repetitions=stack))
    return ir, info


def ir_quality(
    ir: np.ndarray, sr: int, *, pre: int = 0, repetitions: np.ndarray | None = None
) -> dict[str, Any]:
    """Peak-to-noise ratio (tail = last 10%) and repeatability across repetitions."""
    ir = np.asarray(ir, dtype=np.float64)
    tail = ir[int(0.9 * len(ir)):]
    peak = float(np.max(np.abs(ir))) if ir.size else 0.0
    noise = float(np.sqrt(np.mean(tail**2))) if tail.size else 0.0
    out: dict[str, Any] = {
        "peak_abs": peak,
        "peak_to_noise_db": float(20 * np.log10(peak / noise)) if noise > 0 and peak > 0 else None,
        "repetition_min_correlation": None,
    }
    if repetitions is not None and len(repetitions) > 1:
        early = repetitions[:, : pre + int(0.05 * sr)]
        c = np.corrcoef(early)
        out["repetition_min_correlation"] = float(np.min(c[np.triu_indices(len(c), 1)]))
    return out


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------


def write_measurement_signal(out_dir: Path, p: SweepParams) -> dict[str, Any]:
    """Write the playback WAV and `sweep_params.json` (parameters, hash, versions)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / (f"sweep_{p.sample_rate}Hz_{p.f1_hz:g}-{p.f2_hz:g}Hz_"
                     f"{p.duration_s:g}s_x{p.repetitions}.wav")
    digest = write_audio(wav, measurement_signal(p), p.sample_rate, subtype="PCM_24")
    meta = {
        "kind": "vehicle_sweep",
        "file": wav.name,
        "sha256": digest,
        "sweep": p.to_dict(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment_info(),
    }
    (out_dir / PARAMS_FILE).write_text(json.dumps(meta, indent=2))
    return meta


def deconvolve_file(
    recording: Path,
    params_json: Path,
    out_path: Path,
    *,
    ir_length_s: float = 0.5,
    pre_s: float = 0.02,
    repetitions: int | None = None,
) -> dict[str, Any]:
    """Deconvolve a recorded sweep file into an IR WAV (FLOAT) plus a JSON sidecar."""
    recording, out_path = Path(recording), Path(out_path)
    meta = json.loads(Path(params_json).read_text())
    p = SweepParams.from_dict(meta["sweep"])
    audio, sr = load_audio(recording)
    ir, info = deconvolve(audio, sr, p, ir_length_s=ir_length_s, pre_s=pre_s,
                          repetitions=repetitions)
    digest = write_audio(out_path, ir, sr, subtype="FLOAT")
    sidecar = {
        "kind": "vehicle_ir",
        "file": out_path.name,
        "sha256": digest,
        "source_recording": {"file": recording.name, "sha256": sha256_file(recording)},
        "sweep_file_sha256": meta.get("sha256"),
        "deconvolution": info,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment_info(),
    }
    out_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    return sidecar


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m forensicaudio.vehicle.measure",
        description="Misura della risposta all'impulso dell'abitacolo "
                    "(sweep esponenziale, metodo di Farina).",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("genera", help="genera il file di sweep da riprodurre e sweep_params.json")
    g.add_argument("--out", type=Path, required=True, help="cartella di uscita")
    g.add_argument("--sr", type=int, default=48000, help="frequenza di campionamento (Hz)")
    g.add_argument("--durata", type=float, default=10.0, help="durata di ogni sweep (s)")
    g.add_argument("--f1", type=float, default=50.0, help="frequenza iniziale (Hz)")
    g.add_argument("--f2", type=float, default=20000.0, help="frequenza finale (Hz)")
    g.add_argument("--ampiezza", type=float, default=0.5, help="ampiezza di picco (0-1]")
    g.add_argument("--ripetizioni", type=int, default=3)
    g.add_argument("--pausa", type=float, default=3.0, help="silenzio dopo ogni sweep (s)")
    g.add_argument("--anticipo", type=float, default=1.0, help="silenzio iniziale (s)")
    d = sub.add_parser("deconvolvi", help="ricava la risposta all'impulso da uno sweep registrato")
    d.add_argument("registrazione", type=Path)
    d.add_argument("--parametri", type=Path, required=True,
                   help="sweep_params.json usato in riproduzione")
    d.add_argument("--out", type=Path, required=True, help="file WAV della risposta all'impulso")
    d.add_argument("--durata-ir", type=float, default=0.5, help="lunghezza della risposta (s)")
    d.add_argument("--pre", type=float, default=0.02, help="margine prima del picco (s)")
    d.add_argument("--ripetizioni", type=int, default=None,
                   help="ripetizioni presenti (default: dai parametri)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.cmd == "genera":
        p = SweepParams(sample_rate=args.sr, duration_s=args.durata, f1_hz=args.f1, f2_hz=args.f2,
                        amplitude=args.ampiezza, repetitions=args.ripetizioni, gap_s=args.pausa,
                        lead_s=args.anticipo)
        meta = write_measurement_signal(args.out, p)
        print(f"Sweep scritto: {args.out / meta['file']}\nSHA-256: {meta['sha256']}\n"
              f"Parametri: {args.out / PARAMS_FILE}")
        return 0

    sidecar = deconvolve_file(args.registrazione, args.parametri, args.out,
                              ir_length_s=args.durata_ir, pre_s=args.pre,
                              repetitions=args.ripetizioni)
    q = sidecar["deconvolution"]
    pnr = q["peak_to_noise_db"]
    print(f"Risposta all'impulso: {args.out}\nSHA-256: {sidecar['sha256']}\n"
          f"Ripetizioni trovate: {q['repetitions_found']}/{q['repetitions_expected']}\n"
          f"Rapporto picco/rumore: {'n.d.' if pnr is None else f'{pnr:.1f} dB'}")
    if pnr is not None and pnr < 40:
        print("ATTENZIONE: rapporto picco/rumore sotto 40 dB; valutare di ripetere la misura "
              "a volume più alto o in un luogo più silenzioso.")
    rc = q["repetition_min_correlation"]
    if rc is not None and rc < 0.9:
        print(f"ATTENZIONE: ripetizioni poco coerenti (correlazione minima {rc:.2f}); "
              "controllare rumori o spostamenti durante la misura.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
