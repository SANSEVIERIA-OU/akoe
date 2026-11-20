# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Speaker-informed cleaning for HUMAN LISTENING with supervised NMF
(M6, issue #33).

- A **speech dictionary** is learned from clean segments of the subject and
  a **noise dictionary** from the vehicle noise (non-speech portions of the
  case recording or a measured noise recording of that car).
- On the case recording the dictionaries are fixed and only the activations
  are estimated (optionally with a few free noise bases for unseen noise);
  a Wiener-like soft mask ``Vs^p / (Vs^p + Vn^p)``, floored, is applied to
  the complex STFT.

Labelling. With fixed seeds the algorithm is deterministic and
bit-reproducible, so ``KIND`` is ``StageKind.DETERMINISTIC`` (no trained
neural weights, no generative step). It is nonetheless **model-based**: the
output depends on dictionaries learned from other recordings, and the mask
can suppress parts of the voice that do not match the subject model. The
output is a **listening version for the human transcriber**, not the input
of the automatic transcriber (MASTER_PLAN §3.5.1, §4.4); the params carry
``model_based``, ``intended_use`` and an Italian label for the interface.
The bases are inspectable (:meth:`NMFDictionaries.describe`, saved with
:meth:`NMFDictionaries.save`). Issue #33 asks for a NEURAL label for this
layer: see the open question in the M6 report.

Implementation: multiplicative updates for the generalised KL divergence
(Lee & Seung 2001) on magnitude spectrograms, with basis columns normalised
to unit L1 norm. Plain NumPy, float64, seeded initialisation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from akoe.config import StageKind
from akoe.stages.common import (
    Segments,
    array_sha256,
    as_float,
    istft,
    load_mono,
    sample_mask,
    single_thread_blas,
    stft,
    to_jsonable,
    versions,
    write_mono,
)
from akoe.utils import sha256_file

KIND = StageKind.DETERMINISTIC
ALGORITHM_VERSION = "1"
INTENDED_USE = "human_listening"
LABEL_IT = ("Versione d'ascolto basata su modello (NMF con dizionari della voce del soggetto e "
            "del rumore dell'auto). Solo per l'ascolto umano: non è l'audio originale e non è "
            "l'ingresso predefinito del trascrittore automatico.")
_EPS = 1e-12


def _kl_nmf(
    v: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    *,
    n_iter: int,
    update_w_cols: slice | None,
    sparsity: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """KL-NMF multiplicative updates. ``update_w_cols`` selects the basis
    columns that are learned (None: all fixed)."""
    with single_thread_blas():
        return _kl_nmf_loop(v, w, h, n_iter=n_iter, update_w_cols=update_w_cols,
                            sparsity=sparsity)


def _kl_nmf_loop(v, w, h, *, n_iter, update_w_cols, sparsity):
    for _ in range(n_iter):
        wh = w @ h + _EPS
        # w.T @ ones == column sums of w (broadcast over frames).
        h *= (w.T @ (v / wh)) / (w.sum(axis=0)[:, None] + sparsity + _EPS)
        if update_w_cols is not None:
            wh = w @ h + _EPS
            ws = w[:, update_w_cols]
            hs = h[update_w_cols]
            ws *= ((v / wh) @ hs.T) / (hs.sum(axis=1)[None, :] + _EPS)
            norm = ws.sum(axis=0, keepdims=True) + _EPS
            w[:, update_w_cols] = ws / norm
            h[update_w_cols] *= norm.T
    return w, h


def learn_bases(
    audio: np.ndarray,
    sr: int,
    *,
    n_components: int,
    segments: Segments | None = None,
    n_fft: int = 1024,
    hop: int = 256,
    n_iter: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Learn ``n_components`` spectral bases (F x K, unit L1 columns) from the
    magnitude spectrogram of ``audio`` (restricted to ``segments`` if given)."""
    x = as_float(audio)
    if segments is not None:
        x = x[sample_mask(segments, len(x), sr)]
    v = np.abs(stft(x, n_fft, hop))
    keep = v.sum(axis=0) > 0
    v = v[:, keep]
    if v.shape[1] < n_components:
        raise ValueError("not enough material to learn the requested number of bases")
    rng = np.random.default_rng(seed)
    scale = float(v.mean())
    w = rng.random((v.shape[0], n_components)) + 0.1
    w /= w.sum(axis=0, keepdims=True)
    h = (rng.random((n_components, v.shape[1])) + 0.1) * scale * v.shape[0] / n_components
    w, h = _kl_nmf(v, w, h, n_iter=n_iter, update_w_cols=slice(None))
    wh = w @ h + _EPS
    kl = float(np.sum(v * np.log((v + _EPS) / wh) - v + wh) / v.size)
    info = {"n_components": n_components, "n_fft": n_fft, "hop": hop, "n_iter": n_iter,
            "seed": seed, "n_frames": int(v.shape[1]), "seconds": float(len(x) / sr),
            "final_kl_per_bin": kl, "training_audio_sha256": array_sha256(x),
            "segments": None if segments is None else [list(s) for s in segments]}
    return w, info


@dataclass
class NMFDictionaries:
    """Subject speech bases + vehicle noise bases, with their training record."""

    speech: np.ndarray  # (F, Ks)
    noise: np.ndarray  # (F, Kn)
    sr: int
    n_fft: int
    hop: int
    record: dict[str, Any] = field(default_factory=dict)

    def sha256(self) -> str:
        return array_sha256(np.concatenate([self.speech, self.noise], axis=1))

    def describe(self) -> dict[str, Any]:
        """Inspectable summary of each basis: peak frequency and spectral
        centroid (Hz), for display next to the bases themselves."""
        f = np.fft.rfftfreq(self.n_fft, 1.0 / self.sr)

        def stats(w: np.ndarray) -> list[dict[str, float]]:
            return [{"peak_hz": float(f[int(np.argmax(c))]),
                     "centroid_hz": float(np.sum(f * c) / (np.sum(c) + _EPS))} for c in w.T]

        return {"speech": stats(self.speech), "noise": stats(self.noise),
                "frequencies_hz": [float(v) for v in f]}

    def save(self, path: Path) -> dict[str, Any]:
        """Write an ``.npz`` with the bases and the JSON training record;
        returns path and SHA-256 for the chain of custody."""
        path = Path(path)
        meta = json.dumps(to_jsonable({"sr": self.sr, "n_fft": self.n_fft, "hop": self.hop,
                                       "record": self.record}), sort_keys=True)
        with path.open("wb") as fh:
            np.savez(fh, speech=self.speech, noise=self.noise, meta=np.array(meta))
        return {"path": str(path), "sha256": sha256_file(path), "bases_sha256": self.sha256()}

    @classmethod
    def load(cls, path: Path) -> "NMFDictionaries":
        with np.load(Path(path), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            return cls(speech=z["speech"].copy(), noise=z["noise"].copy(), sr=int(meta["sr"]),
                       n_fft=int(meta["n_fft"]), hop=int(meta["hop"]), record=meta["record"])


def learn_dictionaries(
    speech_audio: np.ndarray,
    noise_audio: np.ndarray,
    sr: int,
    *,
    speech_segments: Segments | None = None,
    noise_segments: Segments | None = None,
    n_speech: int = 32,
    n_noise: int = 16,
    n_fft: int = 1024,
    hop: int = 256,
    n_iter: int = 200,
    seed: int = 0,
) -> NMFDictionaries:
    """Learn the subject's speech bases from clean segments and the vehicle
    noise bases from noise-only material (seeds ``seed`` and ``seed + 1``)."""
    ws, rs = learn_bases(speech_audio, sr, n_components=n_speech, segments=speech_segments,
                         n_fft=n_fft, hop=hop, n_iter=n_iter, seed=seed)
    wn, rn = learn_bases(noise_audio, sr, n_components=n_noise, segments=noise_segments,
                         n_fft=n_fft, hop=hop, n_iter=n_iter, seed=seed + 1)
    record = {"algorithm": "kl_nmf_multiplicative_updates", "algorithm_version": ALGORITHM_VERSION,
              "speech": rs, "noise": rn, "versions": versions()}
    return NMFDictionaries(speech=ws, noise=wn, sr=sr, n_fft=n_fft, hop=hop, record=record)


def enhance_for_listening(
    audio: np.ndarray,
    sr: int,
    dictionaries: NMFDictionaries,
    *,
    n_iter: int = 100,
    n_free_noise: int = 0,
    mask_power: float = 2.0,
    gain_floor_db: float = -20.0,
    sparsity: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Supervised NMF masking for listening. Returns ``(audio, params)``.

    ``n_free_noise`` extra noise bases are learned on the recording itself
    (semi-supervised) to absorb noise not seen in the noise dictionary.
    """
    if dictionaries.sr != sr:
        raise ValueError("dictionaries were learned at a different sample rate")
    x = as_float(audio)
    n_fft, hop = dictionaries.n_fft, dictionaries.hop
    spec = stft(x, n_fft, hop)
    v = np.abs(spec)
    ks, kn = dictionaries.speech.shape[1], dictionaries.noise.shape[1]
    rng = np.random.default_rng(seed)
    free = rng.random((v.shape[0], n_free_noise)) + 0.1
    free /= np.maximum(free.sum(axis=0, keepdims=True), _EPS)
    w = np.concatenate([dictionaries.speech, dictionaries.noise, free], axis=1).astype(np.float64)
    k = w.shape[1]
    h = (rng.random((k, v.shape[1])) + 0.1) * float(v.mean()) * v.shape[0] / k
    w, h = _kl_nmf(v, w, h, n_iter=n_iter, sparsity=sparsity,
                   update_w_cols=slice(ks + kn, k) if n_free_noise else None)
    vs = w[:, :ks] @ h[:ks]
    vn = w[:, ks:] @ h[ks:]
    mask = vs ** mask_power / (vs ** mask_power + vn ** mask_power + _EPS)
    mask = np.maximum(mask, 10 ** (gain_floor_db / 20))
    y = istft(spec * mask, n_fft, hop, len(x))
    params = {
        "algorithm": "supervised_kl_nmf_wiener_mask",
        "algorithm_version": ALGORITHM_VERSION,
        "kind": KIND.value,
        "model_based": True,
        "intended_use": INTENDED_USE,
        "label_it": LABEL_IT,
        "sr": sr, "n_fft": n_fft, "hop": hop, "window": "hann",
        "n_iter": n_iter, "n_free_noise": n_free_noise, "mask_power": mask_power,
        "gain_floor_db": gain_floor_db, "sparsity": sparsity, "seed": seed,
        "n_speech_bases": ks, "n_noise_bases": kn,
        "dictionaries_sha256": dictionaries.sha256(),
        "dictionaries_record": dictionaries.record,
        "blas_threads": 1,
        "mean_mask_db": float(20 * np.log10(np.mean(mask))),
        "speech_energy_fraction": float(np.sum(vs ** 2)
                                        / (np.sum(vs ** 2) + np.sum(vn ** 2) + _EPS)),
        "versions": versions(),
    }
    return y, to_jsonable(params)


def run(input_path: Path, output_path: Path, dictionaries: NMFDictionaries, *,
        subtype: str = "PCM_16", **kwargs) -> dict[str, Any]:
    """File wrapper for :func:`enhance_for_listening`. Returns the params."""
    audio, sr = load_mono(input_path)
    y, params = enhance_for_listening(audio, sr, dictionaries, **kwargs)
    params.update(write_mono(output_path, y, sr, subtype=subtype))
    params["input_sha256"] = sha256_file(input_path)
    params["output_sha256"] = sha256_file(output_path)
    return params
