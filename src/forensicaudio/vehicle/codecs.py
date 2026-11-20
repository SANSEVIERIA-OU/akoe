"""Codec augmentation (variable codec per example) and codec detection.

The covert device transmits with codecs that can change (MASTER_PLAN §4.3), so
training material is passed through a codec drawn at random per example
(IMPLEMENTATION_PLAN M3). Backends, chosen at runtime:

- numpy:      G.711 mu-law / A-law (ITU-T G.711, own implementation);
- libsndfile: GSM 06.10, MP3, Opus, Vorbis, when the installed libsndfile
              supports them (`sf.available_subtypes`);
- ffmpeg:     AMR-NB, AMR-WB, AAC (and GSM if libsndfile lacks it), only when
              the ffmpeg binary has the encoder (`ffmpeg -encoders`).

An unavailable codec is skipped with a logged warning, never an exception.
Output always has the input's sample rate and length; the codec delay is
compensated by cross-correlation so the audio stays aligned with transcripts.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import soundfile as sf

from forensicaudio.vehicle import ffmpeg
from forensicaudio.vehicle.audio import estimate_lag, pad_or_trim, resample, shift, to_mono

log = logging.getLogger(__name__)

Backend = Literal["numpy", "libsndfile", "ffmpeg"]

# Codec delays above this are not compensated (MP3 CBR at low rates: ~140 ms).
MAX_DELAY_S = 0.25
MIN_ALIGN_CORRELATION = 0.2


@dataclass(frozen=True)
class CodecSpec:
    name: str
    label: str
    backend: Backend
    sample_rates: tuple[int, ...]
    # One option is drawn per example. libsndfile: compression_level (0-1,
    # higher = smaller) and bitrate_mode; ffmpeg: bitrate_kbps.
    options: tuple[dict[str, Any], ...] = ({},)
    nominal_kbps: float | None = None
    sf_format: str | None = None
    sf_subtype: str | None = None
    ff_encoder: str | None = None
    ff_format: str | None = None  # muxer used for encoding
    ff_demuxer: str | None = None  # demuxer for decoding, when it differs from ff_format


def _kbps(*values: float) -> tuple[dict[str, Any], ...]:
    return tuple({"bitrate_kbps": v} for v in values)


CODEC_SPECS: tuple[CodecSpec, ...] = (
    CodecSpec("g711_ulaw", "G.711 mu-law", "numpy", (8000,), nominal_kbps=64.0),
    CodecSpec("g711_alaw", "G.711 A-law", "numpy", (8000,), nominal_kbps=64.0),
    CodecSpec("gsm", "GSM 06.10 full rate", "libsndfile", (8000,), nominal_kbps=13.0,
              sf_format="WAV", sf_subtype="GSM610"),
    CodecSpec("gsm", "GSM 06.10 full rate", "ffmpeg", (8000,), nominal_kbps=13.0,
              ff_encoder="libgsm", ff_format="gsm"),
    CodecSpec("mp3", "MP3", "libsndfile", (8000, 16000),
              tuple({"compression_level": c, "bitrate_mode": "CONSTANT"} for c in (0.5, 0.75, 0.9)),
              sf_format="MP3", sf_subtype="MPEG_LAYER_III"),
    # libsndfile maps compression_level linearly to the Opus bitrate:
    # 0.93 / 0.96 / 0.98 give roughly 24 / 16 / 11 kbps.
    CodecSpec("opus", "Opus", "libsndfile", (8000, 16000),
              tuple({"compression_level": c} for c in (0.93, 0.96, 0.98)),
              sf_format="OGG", sf_subtype="OPUS"),
    CodecSpec("vorbis", "Vorbis", "libsndfile", (8000, 16000),
              tuple({"compression_level": c} for c in (0.5, 0.9)),
              sf_format="OGG", sf_subtype="VORBIS"),
    CodecSpec("amr_nb", "AMR-NB", "ffmpeg", (8000,), _kbps(4.75, 5.9, 7.4, 10.2, 12.2),
              ff_encoder="libopencore_amrnb", ff_format="amr"),
    CodecSpec("amr_wb", "AMR-WB", "ffmpeg", (16000,), _kbps(6.6, 8.85, 12.65, 23.85),
              ff_encoder="libvo_amrwbenc", ff_format="amr"),
    CodecSpec("aac", "AAC-LC", "ffmpeg", (8000, 16000), _kbps(16, 24, 32),
              ff_encoder="aac", ff_format="adts", ff_demuxer="aac"),
)

# Codecs drawn by default (MASTER_PLAN §4.3). The device's transmission codecs
# vary, so the default set is broad, AAC included (decision 2026-11-20).
DEFAULT_CODECS: tuple[str, ...] = (
    "g711_ulaw", "g711_alaw", "gsm", "mp3", "opus", "vorbis", "amr_nb", "amr_wb", "aac",
)


# --------------------------------------------------------------------------
# G.711 (bit-exact with the ITU-T reference / CPython audioop)
# --------------------------------------------------------------------------

_SEG_UEND = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF])
_SEG_AEND = np.array([0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF])


def ulaw_encode(pcm16: np.ndarray) -> np.ndarray:
    v = np.asarray(pcm16, dtype=np.int32) >> 2
    mask = np.where(v < 0, 0x7F, 0xFF)
    mag = np.minimum(np.abs(v), 8159) + 33
    seg = np.searchsorted(_SEG_UEND, mag, side="left")
    segc = np.minimum(seg, 7)
    code = (segc << 4) | ((mag >> (segc + 1)) & 0xF)
    return np.where(seg >= 8, 0x7F ^ mask, code ^ mask).astype(np.uint8)


def ulaw_decode(codes: np.ndarray) -> np.ndarray:
    u = ~np.asarray(codes, dtype=np.int32) & 0xFF
    t = (((u & 0x0F) << 3) + 0x84) << ((u & 0x70) >> 4)
    return np.where(u & 0x80, 0x84 - t, t - 0x84).astype(np.int16)


def alaw_encode(pcm16: np.ndarray) -> np.ndarray:
    v = np.asarray(pcm16, dtype=np.int32) >> 3
    neg = v < 0
    mask = np.where(neg, 0x55, 0xD5)
    mag = np.where(neg, -v - 1, v)
    seg = np.searchsorted(_SEG_AEND, mag, side="left")
    segc = np.minimum(seg, 7)
    code = (segc << 4) | ((mag >> np.where(segc < 2, 1, segc)) & 0x0F)
    return np.where(seg >= 8, 0x7F ^ mask, code ^ mask).astype(np.uint8)


def alaw_decode(codes: np.ndarray) -> np.ndarray:
    a = np.asarray(codes, dtype=np.int32) ^ 0x55
    seg = (a & 0x70) >> 4
    t = (a & 0x0F) << 4
    t = np.where(seg == 0, t + 8, (t + 0x108) << np.maximum(seg - 1, 0))
    return np.where(a & 0x80, t, -t).astype(np.int16)


def _to_pcm16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)


# --------------------------------------------------------------------------
# Availability
# --------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _libsndfile_works(fmt: str, subtype: str, sr: int) -> bool:
    try:
        if subtype not in sf.available_subtypes(fmt):
            return False
        buf = io.BytesIO()
        sf.write(buf, np.zeros(sr // 10, dtype=np.float32), sr, format=fmt, subtype=subtype)
        buf.seek(0)
        sf.read(buf)
        return True
    except Exception:  # noqa: BLE001 - any failure means "not usable here"
        return False


def spec_status(spec: CodecSpec) -> tuple[bool, str]:
    """(available, reason) for one backend of a codec."""
    if spec.backend == "numpy":
        return True, "numpy"
    if spec.backend == "libsndfile":
        ok = all(_libsndfile_works(spec.sf_format, spec.sf_subtype, sr) for sr in spec.sample_rates)
        return ok, ("libsndfile" if ok else f"libsndfile senza {spec.sf_format}/{spec.sf_subtype}")
    if ffmpeg.ffmpeg_exe() is None:
        return False, "ffmpeg non trovato"
    if not ffmpeg.has_encoder(spec.ff_encoder):
        return False, f"encoder ffmpeg {spec.ff_encoder} assente"
    return True, f"ffmpeg/{spec.ff_encoder}"


@lru_cache(maxsize=None)
def available_codecs() -> dict[str, CodecSpec]:
    """First available backend for every registered codec name."""
    out: dict[str, CodecSpec] = {}
    for spec in CODEC_SPECS:
        if spec.name not in out and spec_status(spec)[0]:
            out[spec.name] = spec
    return out


def codec_report() -> dict[str, dict[str, Any]]:
    """Availability of every registered codec, for logs and the operator."""
    report: dict[str, dict[str, Any]] = {}
    for spec in CODEC_SPECS:
        ok, reason = spec_status(spec)
        entry = report.setdefault(spec.name, {"label": spec.label, "available": False, "tried": []})
        entry["tried"].append(reason)
        if ok and not entry["available"]:
            entry.update(available=True, backend=spec.backend)
    return report


def _unavailable_reason(name: str) -> str:
    reasons = [spec_status(s)[1] for s in CODEC_SPECS if s.name == name]
    return "; ".join(reasons) if reasons else "codec sconosciuto"


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------


def _encode_decode(
    spec: CodecSpec, x: np.ndarray, sr: int, option: dict[str, Any]
) -> tuple[np.ndarray, int]:
    """Run one codec on mono float64 audio at its own rate. Returns (decoded, encoded bytes)."""
    if spec.backend == "numpy":
        pcm = _to_pcm16(x)
        if spec.name == "g711_ulaw":
            return ulaw_decode(ulaw_encode(pcm)) / 32768.0, len(pcm)
        return alaw_decode(alaw_encode(pcm)) / 32768.0, len(pcm)
    if spec.backend == "libsndfile":
        buf = io.BytesIO()
        kwargs = {k: option[k] for k in ("compression_level", "bitrate_mode") if k in option}
        sf.write(buf, x.astype(np.float32), sr, format=spec.sf_format, subtype=spec.sf_subtype,
                 **kwargs)
        n_bytes = buf.tell()
        buf.seek(0)
        y, sr_dec = sf.read(buf, dtype="float64", always_2d=False)
        y = to_mono(y)
        return (y if sr_dec == sr else resample(y, sr_dec, sr)), n_bytes
    extra = []
    if "bitrate_kbps" in option:
        extra = ["-b:a", str(int(round(option["bitrate_kbps"] * 1000)))]
    return ffmpeg.encode_decode(_to_pcm16(x), sr, encoder=spec.ff_encoder, fmt=spec.ff_format,
                                demuxer=spec.ff_demuxer, extra_args=extra)


def _skipped(audio: np.ndarray, name: str, reason: str) -> tuple[np.ndarray, dict[str, Any]]:
    return np.asarray(audio, dtype=np.float32).copy(), {
        "codec": "none",
        "requested_codec": name,
        "skipped": True,
        "reason": reason,
    }


def apply_codec(
    audio: np.ndarray,
    sr: int,
    codec: str,
    *,
    rng: np.random.Generator | int | None = None,
    codec_sr: int | None = None,
    option: dict[str, Any] | None = None,
    align: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode and decode `audio` with `codec`. Returns (float32 audio, params).

    Codec sample rate and option are drawn from the codec spec with `rng`
    unless given. Output has the input's rate and length. If the codec is not
    available or fails, the input is returned unchanged with `skipped=True`.
    """
    spec = available_codecs().get(codec)
    if spec is None:
        reason = _unavailable_reason(codec)
        log.warning("Codec %s non disponibile (%s): saltato.", codec, reason)
        return _skipped(audio, codec, reason)
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    codec_sr = int(codec_sr or spec.sample_rates[int(rng.integers(len(spec.sample_rates)))])
    if option is None:
        option = spec.options[int(rng.integers(len(spec.options)))]
    option = dict(option)

    x = to_mono(audio)
    n = len(x)
    xc = resample(x, sr, codec_sr)
    clipped = int(np.sum(np.abs(xc) > 1.0))
    xc = np.clip(xc, -1.0, 1.0)
    try:
        yc, n_bytes = _encode_decode(spec, xc, codec_sr, option)
    except Exception as exc:  # noqa: BLE001 - a codec failure must never stop a dataset run
        log.warning("Codec %s fallito (%s): saltato.", codec, exc)
        return _skipped(audio, codec, f"errore: {exc}")
    y = resample(yc, codec_sr, sr)

    lag, corr = 0, 0.0
    if align:
        lag, corr = estimate_lag(x, pad_or_trim(y, max(n, len(y))), int(MAX_DELAY_S * sr))
        if corr < MIN_ALIGN_CORRELATION:
            lag = 0
        y = shift(y, lag) if lag else y
    y = pad_or_trim(y, n)

    duration = len(xc) / codec_sr
    params = {
        "codec": spec.name,
        "codec_label": spec.label,
        "backend": spec.backend,
        "codec_sample_rate": codec_sr,
        "sample_rate": int(sr),
        "n_samples": n,
        "option": option,
        "nominal_bitrate_kbps": option.get("bitrate_kbps", spec.nominal_kbps),
        # Encoded size / duration, container overhead included.
        "measured_bitrate_kbps": round(n_bytes * 8 / duration / 1000, 3) if duration > 0 else None,
        "delay_compensated_samples": int(lag),
        "alignment_correlation": round(float(corr), 4),
        "clipped_samples": clipped,
    }
    return y.astype(np.float32), params


@dataclass
class CodecAugmenter:
    """Draws one codec per example among the available ones.

    `probability` is the chance that an example gets a codec at all (the rest
    stays uncoded, so the model does not come to depend on codec artifacts;
    Balam et al. 2020). Requested codecs that are not available are dropped
    with one warning at construction.
    """

    codecs: Sequence[str] | None = None
    weights: Sequence[float] | None = None
    probability: float = 1.0
    active: list[str] = field(init=False)
    unavailable: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        requested = list(self.codecs or DEFAULT_CODECS)
        weights = list(self.weights) if self.weights is not None else [1.0] * len(requested)
        if len(weights) != len(requested):
            raise ValueError("weights must match codecs")
        avail = available_codecs()
        self.active, self._weights, self.unavailable = [], [], {}
        for name, w in zip(requested, weights):
            if name in avail:
                self.active.append(name)
                self._weights.append(float(w))
            else:
                self.unavailable[name] = _unavailable_reason(name)
        for name, reason in self.unavailable.items():
            log.warning("Codec %s non disponibile (%s): escluso dalla simulazione.", name, reason)
        if not self.active:
            log.warning("Nessun codec disponibile: la simulazione dei codec non avrà effetto.")

    def __call__(
        self, audio: np.ndarray, sr: int, rng: np.random.Generator | int | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        draw = float(rng.random())
        if not self.active or draw >= self.probability:
            return np.asarray(audio, dtype=np.float32).copy(), {"codec": "none", "skipped": False}
        p = np.asarray(self._weights) / np.sum(self._weights)
        name = self.active[int(rng.choice(len(self.active), p=p))]
        return apply_codec(audio, sr, name, rng=rng)

    def describe(self) -> dict[str, Any]:
        return {
            "requested": list(self.codecs or DEFAULT_CODECS),
            "active": list(self.active),
            "weights": list(self._weights),
            "probability": self.probability,
            "unavailable": dict(self.unavailable),
        }


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------

_SF_SUBTYPE_CODEC = {
    "ULAW": "g711_ulaw",
    "ALAW": "g711_alaw",
    "GSM610": "gsm",
    "MPEG_LAYER_III": "mp3",
    "MPEG_LAYER_II": "mp2",
    "MPEG_LAYER_I": "mp1",
    "OPUS": "opus",
    "VORBIS": "vorbis",
    "IMA_ADPCM": "ima_adpcm",
    "MS_ADPCM": "ms_adpcm",
    "G721_32": "g721",
    "G723_24": "g723_24",
    "G723_40": "g723_40",
}

_FF_CODEC = {
    "pcm_mulaw": "g711_ulaw",
    "pcm_alaw": "g711_alaw",
    "gsm": "gsm",
    "gsm_ms": "gsm",
    "amr_nb": "amr_nb",
    "amrnb": "amr_nb",
    "amr_wb": "amr_wb",
    "amrwb": "amr_wb",
    "mp3": "mp3",
    "mp3float": "mp3",
    "opus": "opus",
    "vorbis": "vorbis",
    "aac": "aac",
    "flac": "flac",
}

LOSSY = {"g711_ulaw", "g711_alaw", "gsm", "mp3", "mp2", "mp1", "opus", "vorbis", "amr_nb", "amr_wb",
         "aac", "ima_adpcm", "ms_adpcm", "g721", "g723_24", "g723_40"}


def _normalize_sf(fmt: str, subtype: str) -> str:
    if subtype in _SF_SUBTYPE_CODEC:
        return _SF_SUBTYPE_CODEC[subtype]
    if fmt == "FLAC":
        return "flac"
    if subtype.startswith("PCM_") or subtype in ("FLOAT", "DOUBLE"):
        return "pcm_" + subtype.lower().removeprefix("pcm_")
    return subtype.lower()


def detect_codec(path: Path) -> dict[str, Any]:
    """Codec of a file from its metadata: soundfile first, then `ffmpeg -i`.

    Returns {codec, lossy, container, sample_rate, channels, duration_s,
    bitrate_kbps, method, raw}; codec is None when nothing can be read.
    """
    path = Path(path)
    out: dict[str, Any] = {
        "codec": None, "lossy": None, "container": None, "sample_rate": None, "channels": None,
        "duration_s": None, "bitrate_kbps": None, "method": None, "raw": None,
    }
    try:
        info = sf.info(str(path))
        codec = _normalize_sf(info.format, info.subtype)
        out.update(
            codec=codec, container=info.format.lower(), sample_rate=int(info.samplerate),
            channels=int(info.channels), duration_s=float(info.duration), method="soundfile",
            raw=f"{info.format}/{info.subtype}",
        )
        if info.duration > 0:
            out["bitrate_kbps"] = round(path.stat().st_size * 8 / info.duration / 1000, 3)
    except (sf.LibsndfileError, RuntimeError, TypeError):
        probed = ffmpeg.probe(path)
        if probed is not None:
            out.update(
                codec=_FF_CODEC.get(probed["codec"], probed["codec"]),
                container=probed["container"], sample_rate=probed["sample_rate"],
                channels=probed["channels"], duration_s=probed["duration_s"],
                bitrate_kbps=probed["stream_bitrate_kbps"] or probed["bitrate_kbps"],
                method="ffmpeg", raw=probed["raw"],
            )
    if out["codec"] is not None:
        c = out["codec"]
        out["lossy"] = c in LOSSY or not (c.startswith("pcm_") or c == "flac")
    return out
