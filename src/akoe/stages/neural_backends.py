# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Neural speech enhancers for HUMAN LISTENING (M6, issue #59).

Each backend implements :class:`akoe.stages.neural_clean.NeuralEnhancer`
and is pinned: weights at a fixed revision with their SHA-256, third-party
inference code at a fixed commit with the SHA-256 of every file
(:class:`akoe.stages.neural_clean.CodeSnapshot`). Nothing is downloaded or
loaded before :meth:`load`.

=====================  =======  ========================================  ==============
backend                rate     source (licence)                          runtime
=====================  =======  ========================================  ==============
``unipase``            16 kHz   Xiaobin-Rong/unipase (code MIT; weights    torch + pinned
                                card Apache-2.0), DeWavLM-Omni + Adapter   code snapshot
                                + Vocos vocoder, 545M
``mossformer2_se_48k`` 48 kHz   ClearerVoice-Studio (Apache-2.0)          torch + pinned
                                                                           code snapshot
``gtcrn``              16 kHz   Xiaobin-Rong/gtcrn (MIT), 48K params,      onnxruntime
                                streaming ONNX of the DNS3 checkpoint
``deepfilternet3``     48 kHz   Rikorose/DeepFilterNet v0.5.6 (MIT or      onnxruntime +
                                Apache-2.0), official ONNX export          NumPy DSP
=====================  =======  ========================================  ==============

The ``clearvoice`` and ``deepfilternet`` pip packages pin ``numpy<2`` and
cannot be installed next to Akoe's stack (Python 3.13, NumPy 2): MossFormer2
runs from the pinned model code, DeepFilterNet 3 from its official ONNX
export with the libDF feature pipeline re-implemented below.
"""
from __future__ import annotations

import pickle
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

from akoe.neural import BackendUnavailable, ModelProvenance, hash_files, resolve_device
from akoe.stages.neural_clean import (
    Chunking,
    CodeSnapshot,
    PinnedFile,
    _Loaded,
    fetch_hf_pinned,
    fetch_pinned,
    missing_dependency,
)

# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _import_torch(backend: str):
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise missing_dependency(backend, "torch (extra 'ml')", e) from e
    return torch


def _onnx_session(path: Path, backend: str, threads: int = 1):
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise missing_dependency(backend, "onnxruntime", e) from e
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def _torch_load_verified(path: Path) -> tuple[Any, str]:
    """Load a checkpoint whose SHA-256 has already been verified. Tries the
    safe ``weights_only`` loader first (memory-mapped when possible)."""
    torch = _import_torch("torch")
    try:
        return torch.load(str(path), map_location="cpu", mmap=True, weights_only=True), "weights_only+mmap"
    except (pickle.UnpicklingError, RuntimeError, AttributeError, TypeError, ValueError):
        pass
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True), "weights_only"
    except (pickle.UnpicklingError, RuntimeError, AttributeError, TypeError, ValueError):
        # Full unpickling only for files matching the pinned SHA-256.
        return torch.load(str(path), map_location="cpu", weights_only=False), "pickle (SHA-256 verified)"


def _hann_periodic(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def stft_centered(x: np.ndarray, n_fft: int, hop: int, window: np.ndarray) -> np.ndarray:
    """``torch.stft(center=True, pad_mode="reflect", onesided=True)`` in NumPy:
    shape (frames, n_fft // 2 + 1)."""
    pad = n_fft // 2
    x = np.asarray(x, dtype=np.float64)
    if len(x) <= pad:
        x = np.pad(x, (0, pad + 1 - len(x)))
    xp = np.pad(x, (pad, pad), mode="reflect")
    n_frames = 1 + (len(xp) - n_fft) // hop
    frames = np.lib.stride_tricks.sliding_window_view(xp, n_fft)[::hop][:n_frames]
    return np.fft.rfft(frames * window, axis=1)


def istft_centered(spec: np.ndarray, n_fft: int, hop: int, window: np.ndarray,
                   length: int) -> np.ndarray:
    """Inverse of :func:`stft_centered` (``torch.istft(center=True)``):
    windowed overlap-add normalised by the summed squared window."""
    frames = np.fft.irfft(spec, n=n_fft, axis=1) * window
    n = n_fft + hop * (len(frames) - 1)
    y = np.zeros(n)
    wsum = np.zeros(n)
    w2 = window ** 2
    for t, fr in enumerate(frames):
        y[t * hop: t * hop + n_fft] += fr
        wsum[t * hop: t * hop + n_fft] += w2
    y /= np.where(wsum > 1e-11, wsum, 1.0)
    pad = n_fft // 2
    y = y[pad: pad + length]
    return np.pad(y, (0, length - len(y))) if len(y) < length else y


# --------------------------------------------------------------------------
# GTCRN (MIT), streaming ONNX, 16 kHz
# --------------------------------------------------------------------------


class GTCRNEnhancer:
    """GTCRN (Rong et al., ICASSP 2024): 48K parameters, sub-band grouped
    temporal convolutions and dual-path grouped RNN. Runs the official
    streaming ONNX export (DNS3 checkpoint) frame by frame on a 512-point
    STFT with a square-root Hann window, hop 256, exactly as the reference
    ``infer.py``. Causal: no look-ahead."""

    name = "gtcrn"
    native_sr = 16000
    chunking = Chunking(chunk_s=30.0, overlap_s=2.0, crossfade_s=0.5)
    REPO = "Xiaobin-Rong/gtcrn"
    COMMIT = "502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73"
    ONNX = PinnedFile(
        url=f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/stream/onnx_models/gtcrn_simple.onnx",
        sha256="b4718df6228e7bdf1a8a435cf98f838636eb2fd331acabf86ba87c5192ebcb87",
        name=f"gtcrn-{COMMIT[:12]}/gtcrn_simple.onnx",
    )
    N_FFT, HOP = 512, 256

    def __init__(self, *, device: str = "cpu", local_files_only: bool = False,
                 onnx_threads: int = 1) -> None:
        if device != "cpu":
            raise ValueError("GTCRN gira su CPU (onnxruntime); usare device='cpu'.")
        self.local_files_only = local_files_only
        self.onnx_threads = onnx_threads
        self._s = _Loaded()

    @property
    def provenance(self) -> ModelProvenance | None:
        return self._s.provenance

    def load(self) -> ModelProvenance:
        with self._s.lock:
            if self._s.provenance is None:
                path = fetch_pinned(self.ONNX, local_files_only=self.local_files_only)
                self._s.state = _onnx_session(path, self.name, self.onnx_threads)
                self._s.provenance = ModelProvenance(
                    backend=self.name, model_id=f"{self.REPO}/stream/onnx_models/gtcrn_simple.onnx",
                    revision=self.COMMIT, weights=hash_files([path]), device="cpu", dtype="float32",
                    license="MIT (repository licence, verified)",
                    extra={"checkpoint": "model_trained_on_dns3 (streaming export)",
                           "url": self.ONNX.url, "runtime": "onnxruntime"},
                )
        return self._s.provenance

    def describe(self) -> dict[str, Any]:
        return {"stft": {"n_fft": self.N_FFT, "hop": self.HOP, "window": "sqrt_hann_periodic",
                         "center": True, "pad_mode": "reflect"},
                "streaming": True, "lookahead_ms": 0.0, "onnx_threads": self.onnx_threads,
                "match_level": "none"}

    def enhance_chunk(self, audio: np.ndarray) -> np.ndarray:
        sess = self._s.state
        if sess is None:
            self.load()
            sess = self._s.state
        win = np.sqrt(_hann_periodic(self.N_FFT))
        spec = stft_centered(audio, self.N_FFT, self.HOP, win)
        conv = np.zeros((2, 1, 16, 16, 33), np.float32)
        tra = np.zeros((2, 3, 1, 1, 16), np.float32)
        inter = np.zeros((2, 1, 33, 16), np.float32)
        mix = np.zeros((1, spec.shape[1], 1, 2), np.float32)
        out = np.empty_like(spec)
        for t in range(spec.shape[0]):
            mix[0, :, 0, 0] = spec[t].real
            mix[0, :, 0, 1] = spec[t].imag
            enh, conv, tra, inter = sess.run(
                None, {"mix": mix, "conv_cache": conv, "tra_cache": tra, "inter_cache": inter})
            out[t] = enh[0, :, 0, 0] + 1j * enh[0, :, 0, 1]
        return istft_centered(out, self.N_FFT, self.HOP, win, len(audio)).astype(np.float32)


# --------------------------------------------------------------------------
# DeepFilterNet 3 (MIT or Apache-2.0), official ONNX + libDF DSP in NumPy
# --------------------------------------------------------------------------


def dfn_erb_widths(sr: int = 48000, fft_size: int = 960, nb_bands: int = 32,
                   min_nb_freqs: int = 2) -> np.ndarray:
    """ERB band widths in FFT bins, as ``libDF::erb_fb`` (f32 arithmetic)."""
    f32 = np.float32

    def freq2erb(f):
        return f32(9.265) * np.log1p(f32(f) / f32(24.7 * 9.265))

    def erb2freq(e):
        return f32(24.7 * 9.265) * (np.exp(f32(e) / f32(9.265)) - f32(1.0))

    freq_width = f32(sr) / f32(fft_size)
    erb_low, erb_high = freq2erb(f32(0.0)), freq2erb(f32(sr // 2))
    step = (erb_high - erb_low) / f32(nb_bands)
    widths = np.zeros(nb_bands, dtype=np.int64)
    prev, over = 0, 0
    for i in range(1, nb_bands + 1):
        f = erb2freq(erb_low + f32(i) * step)
        fb = int(np.floor(f / freq_width + f32(0.5)))  # Rust f32::round (half away from zero)
        nb = fb - prev - over
        if nb < min_nb_freqs:
            over = min_nb_freqs - nb
            nb = min_nb_freqs
        else:
            over = 0
        widths[i - 1] = nb
        prev = fb
    widths[-1] += 1
    too_large = int(widths.sum()) - (fft_size // 2 + 1)
    if too_large > 0:
        widths[-1] -= too_large
    return widths


def dfn_vorbis_window(fft_size: int = 960) -> np.ndarray:
    i = np.arange(fft_size, dtype=np.float64)
    s = np.sin(0.5 * np.pi * (i + 0.5) / (fft_size // 2))
    return np.sin(0.5 * np.pi * s * s).astype(np.float32)


def _exp_smooth(x: np.ndarray, alpha: float, init: np.ndarray) -> np.ndarray:
    """``s_t = (1 - alpha) * x_t + alpha * s_{t-1}`` along axis 0, ``s_{-1} = init``."""
    from scipy.signal import lfilter

    zi = (alpha * np.asarray(init, dtype=np.float64))[None, :]
    s, _ = lfilter([1.0 - alpha], [1.0, -alpha], np.asarray(x, dtype=np.float64), axis=0, zi=zi)
    return s


class DeepFilterNet3Enhancer:
    """DeepFilterNet 3 (Schröter et al., Interspeech 2023) from the official
    ONNX export of release v0.5.6 (``models/DeepFilterNet3_onnx.tar.gz``:
    encoder, ERB decoder, deep-filter decoder). The libDF pipeline is
    re-implemented in NumPy following ``libDF/src/lib.rs`` and
    ``df/enhance.py``/``df/deepfilternet3.py`` of the same release: Vorbis
    window STFT 960/480 at 48 kHz, 32 ERB bands with exponential mean
    normalisation, unit-normalised complex features of the first 96 bins,
    features shifted by the 2-frame convolution look-ahead, ERB gains on the
    whole spectrum and a 5-tap deep filter (look-ahead 2) on the first 96
    bins, overlap-add synthesis with the 480-sample delay removed.
    ``atten_lim_db`` (optional) limits the attenuation as the reference
    ``enhance(atten_lim_db=...)``."""

    name = "deepfilternet3"
    native_sr = 48000
    chunking = Chunking(chunk_s=30.0, overlap_s=4.0, crossfade_s=0.5)
    REPO = "Rikorose/DeepFilterNet"
    TAG = "v0.5.6"
    COMMIT = "978576aa8400552a4ce9730838c635aa30db5e61"
    ARCHIVE = PinnedFile(
        url=f"https://github.com/{REPO}/raw/{COMMIT}/models/DeepFilterNet3_onnx.tar.gz",
        sha256="c94d91f70911001c946e0fabb4aa9adc37045f45a03b56008cb0c8244cb63616",
        name=f"deepfilternet3-{TAG}/DeepFilterNet3_onnx.tar.gz",
    )
    MEMBERS = ("tmp/export/enc.onnx", "tmp/export/erb_dec.onnx", "tmp/export/df_dec.onnx",
               "tmp/export/config.ini")
    FFT, HOP, NB_ERB, NB_DF, DF_ORDER, DF_LOOKAHEAD, CONV_LOOKAHEAD = 960, 480, 32, 96, 5, 2, 2
    ALPHA = 0.99  # libDF calc_norm_alpha(48000, 480, tau=1), rounded to 3 decimals

    def __init__(self, *, device: str = "cpu", local_files_only: bool = False,
                 atten_lim_db: float | None = None, onnx_threads: int = 1) -> None:
        if device != "cpu":
            raise ValueError("DeepFilterNet 3 (ONNX) gira su CPU; usare device='cpu'.")
        self.local_files_only = local_files_only
        self.atten_lim_db = atten_lim_db
        self.onnx_threads = onnx_threads
        self._s = _Loaded()
        self._erb = dfn_erb_widths()
        self._win = dfn_vorbis_window(self.FFT)

    @property
    def provenance(self) -> ModelProvenance | None:
        return self._s.provenance

    def _extract(self, archive: Path) -> dict[str, Path]:
        out_dir = archive.parent
        paths = {Path(m).name: out_dir / Path(m).name for m in self.MEMBERS}
        if all(p.exists() for p in paths.values()):
            return paths
        with tarfile.open(archive, "r:gz") as tf:
            for m in self.MEMBERS:  # named members only, never extractall
                fh = tf.extractfile(m)
                if fh is None:
                    raise BackendUnavailable(f"Archivio DeepFilterNet3 incompleto: manca {m}")
                tmp = paths[Path(m).name].with_suffix(".part")
                tmp.write_bytes(fh.read())
                tmp.replace(paths[Path(m).name])
        return paths

    def load(self) -> ModelProvenance:
        with self._s.lock:
            if self._s.provenance is None:
                archive = fetch_pinned(self.ARCHIVE, local_files_only=self.local_files_only)
                files = self._extract(archive)
                self._s.state = {k: _onnx_session(files[f"{k}.onnx"], self.name, self.onnx_threads)
                                 for k in ("enc", "erb_dec", "df_dec")}
                self._s.provenance = ModelProvenance(
                    backend=self.name, model_id=f"{self.REPO}/models/DeepFilterNet3_onnx.tar.gz",
                    revision=f"{self.TAG}@{self.COMMIT}",
                    weights={**hash_files([archive]), **hash_files(files.values())},
                    device="cpu", dtype="float32",
                    license="MIT OR Apache-2.0 (repository, verified)",
                    extra={"url": self.ARCHIVE.url, "runtime": "onnxruntime + NumPy libDF DSP"},
                )
        return self._s.provenance

    def describe(self) -> dict[str, Any]:
        return {"sr": self.native_sr, "fft_size": self.FFT, "hop_size": self.HOP,
                "window": "vorbis", "nb_erb": self.NB_ERB, "nb_df": self.NB_DF,
                "df_order": self.DF_ORDER, "df_lookahead": self.DF_LOOKAHEAD,
                "conv_lookahead": self.CONV_LOOKAHEAD, "norm_alpha": self.ALPHA,
                "erb_widths": self._erb.tolist(), "atten_lim_db": self.atten_lim_db,
                "post_filter": False, "onnx_threads": self.onnx_threads, "match_level": "none"}

    # DSP ------------------------------------------------------------------

    def analysis(self, audio: np.ndarray) -> tuple[np.ndarray, int]:
        """Spectrum (T, 481) of ``audio`` padded as ``enhance(pad=True)``."""
        n = len(audio)
        total = n + self.FFT
        total += (-total) % self.HOP
        xp = np.concatenate([np.zeros(self.FFT - self.HOP), audio, np.zeros(total - n)])
        n_frames = total // self.HOP
        frames = np.lib.stride_tricks.sliding_window_view(xp, self.FFT)[::self.HOP][:n_frames]
        spec = np.fft.rfft(frames * self._win, axis=1) * (1.0 / (self.FFT ** 2 / (2 * self.HOP)))
        return spec.astype(np.complex64), n

    def synthesis(self, spec: np.ndarray, n: int) -> np.ndarray:
        frames = np.fft.irfft(spec.astype(np.complex128), n=self.FFT, axis=1) * self.FFT * self._win
        y = np.zeros(self.HOP * (len(frames) + 1))
        for t, fr in enumerate(frames):
            y[t * self.HOP: t * self.HOP + self.FFT] += fr
        d = self.FFT - self.HOP  # analysis padding = STFT delay
        return y[d: d + n]

    def features(self, spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bounds = np.concatenate([[0], np.cumsum(self._erb)])
        power = spec.real.astype(np.float64) ** 2 + spec.imag.astype(np.float64) ** 2
        band = np.stack([power[:, a:b].mean(axis=1) for a, b in zip(bounds[:-1], bounds[1:])], axis=1)
        erb_db = 10.0 * np.log10(band + 1e-10)
        state = _exp_smooth(erb_db, self.ALPHA, np.linspace(-60.0, -90.0, self.NB_ERB))
        feat_erb = ((erb_db - state) / 40.0).astype(np.float32)
        x = spec[:, : self.NB_DF]
        unit = _exp_smooth(np.abs(x), self.ALPHA, np.linspace(0.001, 0.0001, self.NB_DF))
        xn = x / np.sqrt(unit)
        feat_spec = np.stack([xn.real, xn.imag], axis=0).astype(np.float32)  # (2, T, 96)
        return feat_erb, feat_spec

    def enhance_chunk(self, audio: np.ndarray) -> np.ndarray:
        if self._s.state is None:
            self.load()
        s = self._s.state
        spec, n = self.analysis(np.asarray(audio, dtype=np.float64))
        t_frames = spec.shape[0]
        feat_erb, feat_spec = self.features(spec)
        la = self.CONV_LOOKAHEAD  # pad_feat: ConstantPad2d((0, 0, -la, la))
        fe = np.concatenate([feat_erb[la:], np.zeros((la, self.NB_ERB), np.float32)])[None, None]
        fs = np.concatenate([feat_spec[:, la:], np.zeros((2, la, self.NB_DF), np.float32)], axis=1)[None]
        e0, e1, e2, e3, emb, c0, _lsnr = s["enc"].run(None, {"feat_erb": fe, "feat_spec": fs})
        (m,) = s["erb_dec"].run(None, {"emb": emb, "e3": e3, "e2": e2, "e1": e1, "e0": e0})
        coefs = s["df_dec"].run(None, {"emb": emb, "c0": c0})[0]  # (1, T, 96, 10)
        gains = np.repeat(m[0, 0], self._erb, axis=1)  # (T, 481)
        spec_m = spec * gains
        c = coefs[0].reshape(t_frames, self.NB_DF, self.DF_ORDER, 2)
        c = c[..., 0] + 1j * c[..., 1]  # (T, 96, order)
        before = self.DF_ORDER - 1 - self.DF_LOOKAHEAD
        xp = np.concatenate([np.zeros((before, self.NB_DF), spec.dtype), spec[:, : self.NB_DF],
                             np.zeros((self.DF_LOOKAHEAD, self.NB_DF), spec.dtype)])
        df = np.zeros((t_frames, self.NB_DF), dtype=np.complex128)
        for k in range(self.DF_ORDER):  # frame t - 2 + k with coefficient k
            df += xp[k: k + t_frames] * c[:, :, k]
        spec_e = spec_m.astype(np.complex128)
        spec_e[:, : self.NB_DF] = df
        if self.atten_lim_db is not None and abs(self.atten_lim_db) > 0:
            lim = 10 ** (-abs(self.atten_lim_db) / 20)
            spec_e = spec * lim + spec_e * (1 - lim)
        return self.synthesis(spec_e, n).astype(np.float32)


# --------------------------------------------------------------------------
# MossFormer2_SE_48K (Apache-2.0), ClearerVoice-Studio model code
# --------------------------------------------------------------------------


class MossFormer2SE48KEnhancer:
    """MossFormer2_SE_48K (ClearerVoice-Studio, Alibaba): MossFormer2 mask
    network on Kaldi filter banks (60 mels + deltas) predicting a mask on a
    1920/384 Hamming STFT at 48 kHz. Model code from ClearerVoice-Studio at
    a pinned commit, weights from ``alibabasglab/MossFormer2_SE_48K``.
    Long inputs are processed in 4 s windows as the reference
    ``decode_one_audio_mossformer2_se_48k``. The filter banks use Kaldi
    dither 1.0 as in the reference, with a fixed seed (reproducible)."""

    name = "mossformer2_se_48k"
    native_sr = 48000
    chunking = Chunking(chunk_s=4.0, overlap_s=1.0, crossfade_s=0.5)
    HF_REPO = "alibabasglab/MossFormer2_SE_48K"
    HF_REVISION = "eff8c97925c8bec812af707814b3e5d777fd4503"
    WEIGHTS = {"last_best_checkpoint.pt":
               "03692b9f773bbd6bb43b9c5a41f96b1e28affd66e13796b7bec66ad3d8b227c6"}
    CODE = CodeSnapshot(
        name="clearvoice", repo="modelscope/ClearerVoice-Studio",
        commit="6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61", prefix="clearvoice/clearvoice/",
        license="Apache-2.0",
        files={
            "models/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "models/mossformer2_se/__init__.py":
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "models/mossformer2_se/conv_module.py":
                "cd16d84a55ed22a234d91871d685a7ae97fa3fbca3f0d94bd065ce0b4c9a1180",
            "models/mossformer2_se/fsmn.py":
                "62c376b2708ac1c59853c13cbe9acb7ea011082c39f1a42e1af0f8213a00bb76",
            "models/mossformer2_se/layer_norm.py":
                "d28fea41b3c5a2dd8bb7a1a8c1c97492fcc7eb88cc7b46283fc21e4dbf19f062",
            "models/mossformer2_se/mossformer2.py":
                "528b7d6078c8ac99806a2baefe2dc707a2d9ddc6595abe95e4084b92b397b94b",
            "models/mossformer2_se/mossformer2_block.py":
                "8b4f1714156365d9f1e5d96f750793f47d92f51c14d0afd00d43055cd78bdee7",
            "models/mossformer2_se/mossformer2_se_wrapper.py":
                "85e71167a9025a137f9ca8d314c3da4a34cd3285dc856d51352050ecc56e4dda",
        },
    )
    WIN_LEN, WIN_INC, FFT_LEN, NUM_MELS = 1920, 384, 1920, 60
    MAX_WAV_VALUE = 32768.0

    def __init__(self, *, device: str = "cpu", local_files_only: bool = False, seed: int = 0) -> None:
        self.device = device
        self.local_files_only = local_files_only
        self.seed = seed
        self._s = _Loaded()

    @property
    def provenance(self) -> ModelProvenance | None:
        return self._s.provenance

    def load(self) -> ModelProvenance:
        with self._s.lock:
            if self._s.provenance is None:
                self._load()
        return self._s.provenance  # type: ignore[return-value]

    def _load(self) -> None:
        _import_torch(self.name)
        device = resolve_device(self.device)
        try:
            import einops  # noqa: F401
            import rotary_embedding_torch  # noqa: F401
            import torchaudio  # noqa: F401
            import torchinfo  # noqa: F401
        except ImportError as e:
            raise missing_dependency(self.name, "einops rotary-embedding-torch torchinfo torchaudio",
                                     e) from e
        (fname, sha), = self.WEIGHTS.items()
        path = fetch_hf_pinned(self.HF_REPO, fname, self.HF_REVISION, sha,
                               local_files_only=self.local_files_only)
        wrapper = self.CODE.import_module("models.mossformer2_se.mossformer2_se_wrapper",
                                          local_files_only=self.local_files_only)
        model = wrapper.TestNet()
        ckpt, how = _torch_load_verified(path)
        pretrained = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        state = model.state_dict()
        missing = []
        for key in state:  # same key matching as ClearerVoice SpeechModel._load_model
            for cand in (key, key.replace("module.", ""), "module." + key):
                if cand in pretrained and state[key].shape == pretrained[cand].shape:
                    state[key] = pretrained[cand]
                    break
            else:
                missing.append(key)
        if missing:
            raise BackendUnavailable(f"MossFormer2_SE_48K: {len(missing)} pesi non caricati "
                                     f"(es. {missing[:3]}): checkpoint non compatibile.")
        model.load_state_dict(state)
        del ckpt, pretrained
        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._s.state = model
        self._s.provenance = ModelProvenance(
            backend=self.name, model_id=self.HF_REPO, revision=self.HF_REVISION,
            weights={fname: sha}, device=device, dtype="float32",
            license="Apache-2.0 (model card and repository, verified)",
            extra={"code": self.CODE.record(), "checkpoint_loader": how,
                   "n_parameters": int(sum(p.numel() for p in model.parameters()))},
        )

    def describe(self) -> dict[str, Any]:
        return {"win_type": "hamming (symmetric)", "win_len": self.WIN_LEN, "win_inc": self.WIN_INC,
                "fft_len": self.FFT_LEN, "num_mels": self.NUM_MELS, "fbank_dither": 1.0,
                "fbank_dither_seed": self.seed, "features": "kaldi_fbank+delta+delta_delta",
                "input_scale": self.MAX_WAV_VALUE, "match_level": "none"}

    def enhance_chunk(self, audio: np.ndarray) -> np.ndarray:
        if self._s.state is None:
            self.load()
        torch = _import_torch(self.name)
        import torchaudio

        model = self._s.state
        device = self._s.provenance.device  # type: ignore[union-attr]
        n = len(audio)
        if n < self.WIN_LEN:
            audio = np.pad(audio, (0, self.WIN_LEN - n))
        x = torch.from_numpy(np.asarray(audio, dtype=np.float32) * self.MAX_WAV_VALUE)
        with torch.inference_mode(), torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            fb = torchaudio.compliance.kaldi.fbank(
                x[None], dither=1.0, frame_length=self.WIN_LEN / self.native_sr * 1000,
                frame_shift=self.WIN_INC / self.native_sr * 1000, num_mel_bins=self.NUM_MELS,
                sample_frequency=self.native_sr, window_type="hamming")
            d1 = torchaudio.functional.compute_deltas(fb.T)
            d2 = torchaudio.functional.compute_deltas(d1)
            feats = torch.cat([fb, d1.T, d2.T], dim=1)[None].to(device)
            mask = model(feats)[-1].permute(2, 1, 0).cpu()
            window = torch.hamming_window(self.WIN_LEN, periodic=False)
            spec = torch.stft(x, self.FFT_LEN, self.WIN_INC, self.WIN_LEN, center=False,
                              window=window, return_complex=True)
            masked = spec * mask[..., 0]
            y = torch.istft(masked, self.FFT_LEN, self.WIN_INC, self.WIN_LEN, window=window,
                            center=False, length=len(x))
        return (y.numpy() / self.MAX_WAV_VALUE)[:n].astype(np.float32)


# --------------------------------------------------------------------------
# UniPASE (code MIT; weights card Apache-2.0)
# --------------------------------------------------------------------------


class UniPASEEnhancer:
    """UniPASE (Rong et al., IEEE TASLP 2026): DeWavLM-Omni (WavLM-Large
    fine-tuned for universal enhancement) gives a phonetic representation
    (layer 24) and an acoustic one (layer 1); the Adapter enhances the
    acoustic representation guided by the phonetic one and a Vocos vocoder
    reconstructs 16 kHz speech. Lowest content distortion (dWER) in the
    research of 2026-11-20, but generative: the output is re-synthesised.

    Listening defaults chosen for this product:

    - ``enable_plc=False``: the reference enables packet-loss concealment,
      which fills digital silences with generated content; here gaps stay
      gaps unless explicitly requested;
    - no PostNet (bandwidth extension): output at 16 kHz, resampled to the
      input rate, so nothing above 8 kHz is invented;
    - 8 s chunks with 4 s overlap as the reference ``inference_long.py``,
      switched with a 0.5 s cross-fade in the middle of each overlap;
    - output peak matched to the input peak, as the reference.
    """

    name = "unipase"
    native_sr = 16000
    chunking = Chunking(chunk_s=8.0, overlap_s=4.0, crossfade_s=0.5)
    HF_REPO = "Xiaobin-Rong/unipase"
    HF_REVISION = "f0b4d4c4411fe08fc2dddbf2d9f33260c27ac4a0"
    WEIGHTS = {
        "DeWavLM-Omni.pt": "16c94d05ff6ef2bdf08b0be2853826afda687c2d63c833382198db7682476689",
        "Adapter.pt": "3f3f9f8076e304bc740fef5496f91528ca9ea6be7f5a63deec71cdc33471c8c7",
        "Vocoder_DWO-L1.pt": "7595d48962c296bab2f7cb307f299acf5a02ec6a972d9ab7da3ed71b4a54a3d5",
    }
    _EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    CODE = CodeSnapshot(
        name="unipase", repo="Xiaobin-Rong/unipase",
        commit="857b60ad05d37a2cf6d7a89883ec9fc4fc164b45",
        license="MIT (repository); adapted parts: WavLM MIT, Vocos MIT, PASE Apache-2.0",
        files={
            "LICENSE": "4b69d8256026d257c04606e043e0c7306e024dcc918c8623b3e384d7393cf795",
            "models/wavlm/__init__.py": _EMPTY,
            "models/wavlm/WavLM.py": "64b34574953eb942b98babf51f2951222cfdbfd4111e135dbdb1196bd6f1c98b",
            "models/wavlm/modules.py": "f2cfb5e75f40baed7ddde3f80f7ebac0b79e9d25515e3d636126f7a17f0639ab",
            "models/wavlm/feature_extractor_plc.py":
                "c1ee5e5d8b7bb4883165695031bd49d29ea0a77e5b4d2dfd504a3c9cb223aa9d",
            "models/adapter/vocos/__init__.py": _EMPTY,
            "models/adapter/vocos/adapter.py":
                "a31996ac4923ff596f81015b2081951d121a112c892b9207104a4e3886ce4625",
            "models/adapter/vocos/backbone.py":
                "3d8a5ef4ae7871998c9a5e3cf6c62920f25c778b18e9ffa98f2e61c845c199d8",
            "models/adapter/vocos/conv.py":
                "4238c74d0465fe05e3c75ded5e4103635f1dec632e0f94f553f5e8a4f39e12be",
            "models/adapter/vocos/head.py":
                "1eeb5149fad0fdecd16251f5de77542571f044bbd48ffc0a66e484583d7f05fd",
            "models/vocoder/__init__.py": _EMPTY,
            "models/vocoder/vocos/__init__.py": _EMPTY,
            "models/vocoder/vocos/vocoder.py":
                "dc9c7ba6a14b9bcf4d3c925516dee129c9156d1f490745ae0155693073b2f7eb",
            "models/vocoder/vocos/backbone.py":
                "45ac4a2cceae91f416b97602aa410108122314ae143ccda0166aca48d017d069",
            "models/vocoder/vocos/conv.py":
                "50c79d1c4e46e3c0e975f6b19b9d31bb6d64ef55fc77e0a68a6f68f3a50bff3f",
            "models/vocoder/vocos/head.py":
                "b171bf6b91ed9f516f51ef6a352be3aae7a68a63877bbb0a0f7fb30bd7312a82",
        },
    )
    OUTPUT_LAYERS = [1, 24]

    def __init__(self, *, device: str = "cpu", local_files_only: bool = False,
                 enable_plc: bool = False, num_threads: int | None = None) -> None:
        self.device = device
        self.local_files_only = local_files_only
        self.enable_plc = enable_plc
        self.num_threads = num_threads
        self._s = _Loaded()

    @property
    def provenance(self) -> ModelProvenance | None:
        return self._s.provenance

    def load(self) -> ModelProvenance:
        with self._s.lock:
            if self._s.provenance is None:
                self._load()
        return self._s.provenance  # type: ignore[return-value]

    def _component(self, cls: Any, path: Path) -> tuple[Any, dict[str, Any]]:
        """``VocosAdapter/VocosVocoder.from_pretrained`` with missing weights
        treated as an error (the reference loads with ``strict=False``)."""
        ckpt, how = _torch_load_verified(path)
        model = cls(**dict(ckpt["cfg"]))
        sd = ckpt["generator"] if "generator" in ckpt else ckpt["model"]
        res = model.load_state_dict(sd, strict=False)
        if res.missing_keys:
            raise BackendUnavailable(f"UniPASE: pesi mancanti in {path.name} "
                                     f"(es. {res.missing_keys[:3]}): checkpoint non compatibile.")
        del ckpt, sd
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model, {"loader": how, "unexpected_keys": len(res.unexpected_keys)}

    def _load(self) -> None:
        torch = _import_torch(self.name)
        device = resolve_device(self.device)
        try:
            import omegaconf  # noqa: F401  (imported at module level by the adapter/vocoder code)
            import torchaudio  # noqa: F401
        except ImportError as e:
            raise missing_dependency(self.name, "omegaconf torchaudio", e) from e
        if self.num_threads:
            torch.set_num_threads(int(self.num_threads))
        paths = {f: fetch_hf_pinned(self.HF_REPO, f, self.HF_REVISION, sha,
                                    local_files_only=self.local_files_only)
                 for f, sha in self.WEIGHTS.items()}
        lof = self.local_files_only
        fe = self.CODE.import_module("models.wavlm.feature_extractor_plc", local_files_only=lof)
        adapter_mod = self.CODE.import_module("models.adapter.vocos.adapter", local_files_only=lof)
        vocoder_mod = self.CODE.import_module("models.vocoder.vocos.vocoder", local_files_only=lof)

        # Encoder: WavLM_feat built as in its __init__, loading from a memory map.
        cpt, how = _torch_load_verified(paths["DeWavLM-Omni.pt"])
        cfg = fe.WavLMConfig(cpt["cfg"])
        wavlm = fe.WavLM(cfg)
        wavlm.load_state_dict(cpt["model"])  # strict, as the reference
        del cpt
        wavlm.eval()
        for p in wavlm.parameters():
            p.requires_grad_(False)
        encoder = fe.WavLM_feat.__new__(fe.WavLM_feat)
        torch.nn.Module.__init__(encoder)
        encoder.wavlm_ckpt_path = str(paths["DeWavLM-Omni.pt"])
        encoder.cfg = cfg
        encoder.wavlm = wavlm
        encoder.output_layer = list(self.OUTPUT_LAYERS)
        adapter, a_info = self._component(adapter_mod.VocosAdapter, paths["Adapter.pt"])
        vocoder, v_info = self._component(vocoder_mod.VocosVocoder, paths["Vocoder_DWO-L1.pt"])
        modules = {"encoder": encoder.to(device), "adapter": adapter.to(device),
                   "vocoder": vocoder.to(device)}
        self._s.state = modules
        n_params = sum(p.numel() for m in modules.values() for p in m.parameters())
        self._s.provenance = ModelProvenance(
            backend=self.name, model_id=self.HF_REPO, revision=self.HF_REVISION,
            weights=dict(self.WEIGHTS), device=device, dtype="float32",
            license=("weights: apache-2.0 (model card); code: MIT (repository). Not used: "
                     "PostNet.pt, Vocoder_WavLM-L24.pt"),
            extra={"code": self.CODE.record(), "n_parameters": int(n_params),
                   "loaders": {"DeWavLM-Omni.pt": how, "Adapter.pt": a_info,
                               "Vocoder_DWO-L1.pt": v_info}},
        )

    def describe(self) -> dict[str, Any]:
        return {"enable_plc": self.enable_plc, "output_layers": list(self.OUTPUT_LAYERS),
                "postnet": "not used (no bandwidth extension)", "model_output_sr": 16000,
                "bandwidth_hz": 8000, "generative": True, "match_level": "peak",
                "num_threads": self.num_threads}

    def enhance_chunk(self, audio: np.ndarray) -> np.ndarray:
        if self._s.state is None:
            self.load()
        torch = _import_torch(self.name)
        m = self._s.state
        device = self._s.provenance.device  # type: ignore[union-attr]
        n = len(audio)
        x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None].to(device)
        with torch.inference_mode():
            noisy_a, enh_p = m["encoder"](x, sr=self.native_sr, mask=self.enable_plc)
            enh_a = m["adapter"](noisy_a, enh_p)
            y = m["vocoder"](enh_a.transpose(1, 2)).reshape(-1).cpu().numpy()
        return (np.pad(y, (0, n - len(y))) if len(y) < n else y[:n]).astype(np.float32)


BACKENDS: dict[str, type] = {
    "unipase": UniPASEEnhancer,
    "mossformer2_se_48k": MossFormer2SE48KEnhancer,
    "gtcrn": GTCRNEnhancer,
    "deepfilternet3": DeepFilterNet3Enhancer,
}


__all__ = ["BACKENDS", "DeepFilterNet3Enhancer", "GTCRNEnhancer", "MossFormer2SE48KEnhancer",
           "UniPASEEnhancer", "dfn_erb_widths", "dfn_vorbis_window", "istft_centered",
           "stft_centered"]
