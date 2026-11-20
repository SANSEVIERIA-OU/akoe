"""Generate synthetic test audio for end-to-end pipeline validation.

Produces a fake "evidence" file (sinusoid + noise + silent gaps for VAD)
and one reference "speaker sample" (cleaner version of the same sinusoid).
No real voices — purely for plumbing tests.
"""
from pathlib import Path

import numpy as np
import soundfile as sf


def main(out_root: Path = Path("data")) -> None:
    sr = 16000
    duration_s = 6.0
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)

    # Voice-like carrier: 200 Hz fundamental, slow amplitude modulation
    voice = 0.4 * np.sin(2 * np.pi * 200 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    # Silent gaps so the VAD has something to detect
    voice[: int(0.5 * sr)] = 0.0
    voice[int(2.5 * sr) : int(3.0 * sr)] = 0.0
    voice[int(5.5 * sr) :] = 0.0

    rng = np.random.RandomState(0)
    evidence = voice + 0.08 * rng.randn(len(t))

    out_root.mkdir(parents=True, exist_ok=True)
    samples_dir = out_root / "speaker_samples"
    samples_dir.mkdir(exist_ok=True)

    sf.write(out_root / "evidence.wav", evidence.astype(np.float32), sr)
    sf.write(samples_dir / "sample_01.wav", voice.astype(np.float32), sr)

    print(f"Wrote {out_root / 'evidence.wav'} ({duration_s:.1f}s)")
    print(f"Wrote {samples_dir / 'sample_01.wav'} (1 reference sample)")


if __name__ == "__main__":
    main()
