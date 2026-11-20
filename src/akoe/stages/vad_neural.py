# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Neural voice activity detection (M6, issue #60): Silero VAD 6.2.3 and
FireRedVAD, next to the deterministic robust VAD (:func:`vad.detect_speech`),
which stays the default in forensic mode.

Both detectors return ``(segments, params)`` like the robust VAD, with
``params["kind"] == "neural"`` and the model provenance (package version,
SHA-256 of the model files, pinned Hugging Face revision for FireRedVAD).
Use them through :func:`akoe.stages.vad.detect_regions`.

Licences (verified 2026-11-20): Silero VAD — PyPI classifier MIT and
``LICENSE`` MIT, but the README badge carries the alt text "CC BY-NC 4.0"
while showing "MIT" (inconsistency in the repository, reported); FireRedVAD
— Apache-2.0 (PyPI ``License-Expression``, repository and model card).
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from akoe.config import StageKind
from akoe.neural import BackendUnavailable, ModelProvenance, hash_files
from akoe.stages.common import as_float, to_jsonable, versions
from akoe.stages.neural_clean import fetch_hf_pinned, resample

KIND = StageKind.NEURAL
VAD_SR = 16000


@runtime_checkable
class SpeechDetector(Protocol):
    """A VAD: ``detect`` returns ``(segments in seconds, params)``."""

    name: str
    kind: StageKind

    def detect(self, audio: np.ndarray, sr: int) -> tuple[list[tuple[float, float]], dict[str, Any]]: ...


def _merge(segments: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for s, e in sorted(segments):
        s, e = max(0.0, float(s)), min(duration, float(e))
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return [(round(s, 4), round(e, 4)) for s, e in out]


def _base_params(name: str, sr: int, duration: float, segments: list[tuple[float, float]],
                 prov: ModelProvenance, settings: dict[str, Any]) -> dict[str, Any]:
    return to_jsonable({
        "algorithm": f"neural_vad_{name}", "kind": KIND.value, "method": name,
        "input_sr": sr, "model_sr": VAD_SR,
        "resampling": "none" if sr == VAD_SR else "scipy.signal.resample_poly",
        **settings,
        "model_id": prov.model_id, "model_revision": prov.revision,
        "model_weights_sha256": dict(prov.weights), "model_license": prov.license,
        "device": prov.device, "model_provenance": prov.as_params(),
        "n_segments": len(segments),
        "speech_ratio": float(sum(e - s for s, e in segments) / duration) if duration > 0 else 0.0,
        "segments": [[s, e] for s, e in segments],
        "versions": versions(),
    })


class SileroVAD:
    """Silero VAD 6.2.3 (pip ``silero-vad``; the model files ship inside the
    package, nothing is downloaded). ONNX model run by the package's own
    wrapper (single-threaded onnxruntime) and ``get_speech_timestamps``."""

    name = "silero"
    kind = KIND

    def __init__(self, *, onnx: bool = True, threshold: float = 0.5,
                 min_speech_duration_ms: int = 250, min_silence_duration_ms: int = 100,
                 speech_pad_ms: int = 30, max_speech_duration_s: float = float("inf"),
                 **_: Any) -> None:
        self.onnx = onnx
        self.settings = {"threshold": threshold, "min_speech_duration_ms": min_speech_duration_ms,
                         "min_silence_duration_ms": min_silence_duration_ms,
                         "speech_pad_ms": speech_pad_ms, "max_speech_duration_s": max_speech_duration_s}
        self._lock = threading.Lock()
        self._model: Any = None
        self._prov: ModelProvenance | None = None

    @property
    def provenance(self) -> ModelProvenance | None:
        return self._prov

    def load(self) -> ModelProvenance:
        with self._lock:
            if self._model is None:
                try:
                    import importlib.metadata as md
                    from importlib import resources

                    import silero_vad
                except ImportError as e:
                    raise BackendUnavailable(
                        "VAD neurale Silero non disponibile: manca il pacchetto silero-vad. Per "
                        "abilitarlo: pip install silero-vad==6.2.3 (con onnxruntime).") from e
                fname = "silero_vad.onnx" if self.onnx else "silero_vad.jit"
                path = Path(str(resources.files("silero_vad.data").joinpath(fname)))
                self._model = silero_vad.load_silero_vad(onnx=self.onnx)
                version = md.version("silero-vad")
                self._prov = ModelProvenance(
                    backend="silero-vad", model_id=f"silero-vad=={version}/{fname}",
                    revision=version, weights=hash_files([path]), device="cpu",
                    dtype="float32",
                    license="MIT (PyPI classifier and LICENSE; README badge alt text says CC BY-NC 4.0)",
                    extra={"package": "silero-vad", "package_version": version,
                           "runtime": "onnxruntime" if self.onnx else "torch.jit"},
                )
        return self._prov  # type: ignore[return-value]

    def detect(self, audio: np.ndarray, sr: int) -> tuple[list[tuple[float, float]], dict[str, Any]]:
        prov = self.load()
        import torch
        from silero_vad import get_speech_timestamps

        x = as_float(audio)
        duration = len(x) / sr
        x16 = resample(x, sr, VAD_SR).astype(np.float32)
        with self._lock:
            self._model.reset_states()
            ts = get_speech_timestamps(torch.from_numpy(x16), self._model, sampling_rate=VAD_SR,
                                       return_seconds=False, **self.settings)
            self._model.reset_states()
        segments = _merge([(t["start"] / VAD_SR, t["end"] / VAD_SR) for t in ts], duration)
        params = _base_params(self.name, sr, duration, segments, prov,
                              {**self.settings, "window_size_samples": 512})
        return segments, params


class FireRedVAD:
    """FireRedVAD (Xiaohongshu FireRedTeam, 2026; DFSMN on 80-dim Kaldi
    filter banks with CMVN, 100 frames/s) from pip ``fireredvad`` with the
    non-streaming VAD weights of ``FireRedTeam/FireRedVAD`` at a pinned
    revision. The checkpoint is a pickle: its SHA-256 is verified before it
    is loaded. Input is quantised to 16-bit as in the reference reader."""

    name = "firered"
    kind = KIND
    HF_REPO = "FireRedTeam/FireRedVAD"
    HF_REVISION = "7990aaccc6b7aec1e527743bd30201f2c4a03b8c"
    FILES = {
        "VAD/model.pth.tar": "63f4fb1b00a6b8607c118dd48efc18d5e40d67d99b7bf9aa7a8d61540cf23d71",
        "VAD/cmvn.ark": "c87f6f13edf0f0ec7535ddfc9cc3387d9268cb234b70182d566c5e2edf3ca473",
    }

    def __init__(self, *, local_files_only: bool = False, smooth_window_size: int = 5,
                 speech_threshold: float = 0.4, min_speech_frame: int = 20,
                 max_speech_frame: int = 2000, min_silence_frame: int = 20,
                 merge_silence_frame: int = 0, extend_speech_frame: int = 0,
                 chunk_max_frame: int = 30000, **_: Any) -> None:
        self.local_files_only = local_files_only
        self.settings = {"smooth_window_size": smooth_window_size, "speech_threshold": speech_threshold,
                         "min_speech_frame": min_speech_frame, "max_speech_frame": max_speech_frame,
                         "min_silence_frame": min_silence_frame,
                         "merge_silence_frame": merge_silence_frame,
                         "extend_speech_frame": extend_speech_frame,
                         "chunk_max_frame": chunk_max_frame}
        self._lock = threading.Lock()
        self._vad: Any = None
        self._prov: ModelProvenance | None = None

    @property
    def provenance(self) -> ModelProvenance | None:
        return self._prov

    def load(self) -> ModelProvenance:
        with self._lock:
            if self._vad is None:
                try:
                    import importlib.metadata as md

                    from fireredvad import FireRedVad, FireRedVadConfig
                except ImportError as e:
                    raise BackendUnavailable(
                        "VAD neurale FireRedVAD non disponibile: manca il pacchetto fireredvad. Per "
                        "abilitarlo: pip install fireredvad==0.0.2 (installa kaldi-native-fbank, "
                        "kaldiio, textgrid; nessun aggiornamento di torch/numpy).") from e
                paths = {f: fetch_hf_pinned(self.HF_REPO, f, self.HF_REVISION, sha,
                                            local_files_only=self.local_files_only)
                         for f, sha in self.FILES.items()}
                model_dir = paths["VAD/model.pth.tar"].parent
                if paths["VAD/cmvn.ark"].parent != model_dir:  # pragma: no cover
                    raise BackendUnavailable("FireRedVAD: modello e CMVN in cartelle diverse.")
                self._vad = FireRedVad.from_pretrained(
                    str(model_dir), FireRedVadConfig(use_gpu=False, **self.settings))
                self._prov = ModelProvenance(
                    backend="fireredvad", model_id=f"{self.HF_REPO}/VAD", revision=self.HF_REVISION,
                    weights={f: hash_files([p])[p.name] for f, p in paths.items()}, device="cpu",
                    dtype="float32", license="Apache-2.0 (model card, PyPI and repository, verified)",
                    extra={"package": "fireredvad", "package_version": md.version("fireredvad"),
                           "frame_shift_ms": 10, "features": "kaldi_fbank_80+cmvn"},
                )
        return self._prov  # type: ignore[return-value]

    def detect(self, audio: np.ndarray, sr: int) -> tuple[list[tuple[float, float]], dict[str, Any]]:
        prov = self.load()
        x = as_float(audio)
        duration = len(x) / sr
        x16 = resample(x, sr, VAD_SR)
        clipped = int(np.count_nonzero(np.abs(x16) > 1.0))
        pcm = np.round(np.clip(x16, -1.0, 1.0) * 32767.0).astype(np.int16)
        with self._lock:
            result, _probs = self._vad.detect((pcm, VAD_SR))
        segments = _merge([(float(s), float(e)) for s, e in result.get("timestamps", [])], duration)
        params = _base_params(self.name, sr, duration, segments, prov,
                              {**self.settings, "frame_shift_ms": 10, "pcm16_clipped_samples": clipped})
        return segments, params


DETECTORS: dict[str, type] = {"silero": SileroVAD, "firered": FireRedVAD}


def get_detector(name: str, **params: Any) -> SpeechDetector:
    if name not in DETECTORS:
        raise ValueError(f"VAD neurale sconosciuto: {name!r}; disponibili: {', '.join(DETECTORS)}.")
    return DETECTORS[name](**params)
