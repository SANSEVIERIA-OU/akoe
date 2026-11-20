# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Thin wrapper around the ffmpeg binary shipped with imageio-ffmpeg.

Used only where libsndfile cannot help: AMR/GSM/AAC encoding and decoding,
and probing files soundfile cannot open. Everything degrades gracefully when
the binary or an encoder is missing (callers check `has_encoder`).
"""
from __future__ import annotations

import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


TIMEOUT_S = 300


class FFmpegError(RuntimeError):
    pass


@lru_cache(maxsize=None)
def ffmpeg_exe() -> str | None:
    """Path of the ffmpeg binary (imageio-ffmpeg first, then PATH), or None."""
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:  # noqa: BLE001 - optional dependency
        pass
    return shutil.which("ffmpeg")


def _run(args: list[str], data: bytes | None = None) -> subprocess.CompletedProcess:
    exe = ffmpeg_exe()
    if exe is None:
        raise FFmpegError("ffmpeg not available")
    return subprocess.run(
        [exe, "-nostdin", "-hide_banner", *args],
        input=data,
        capture_output=True,
        timeout=TIMEOUT_S,
        check=False,
    )


@lru_cache(maxsize=None)
def version() -> str | None:
    if ffmpeg_exe() is None:
        return None
    try:
        out = _run(["-version"]).stdout.decode(errors="replace")
    except (OSError, subprocess.SubprocessError, FFmpegError):
        return None
    return out.splitlines()[0].strip() if out else None


@lru_cache(maxsize=None)
def encoders() -> frozenset[str]:
    """Names of the audio encoders compiled into the ffmpeg binary."""
    if ffmpeg_exe() is None:
        return frozenset()
    try:
        out = _run(["-encoders"]).stdout.decode(errors="replace")
    except (OSError, subprocess.SubprocessError, FFmpegError):
        return frozenset()
    names: set[str] = set()
    started = False
    for line in out.splitlines():
        if line.strip().startswith("------"):
            started = True
            continue
        parts = line.split()
        if started and len(parts) >= 2 and parts[0].startswith("A"):
            names.add(parts[1])
    return frozenset(names)


def has_encoder(name: str) -> bool:
    return name in encoders()


def encode_decode(
    pcm16: np.ndarray,
    sr: int,
    *,
    encoder: str,
    fmt: str,
    demuxer: str | None = None,
    extra_args: list[str] | None = None,
) -> tuple[np.ndarray, int]:
    """Encode mono int16 PCM with `encoder` into container `fmt` and decode it back
    (with `demuxer`, default `fmt`).

    Returns (decoded float64 audio at `sr`, size in bytes of the encoded stream).
    """
    raw = np.ascontiguousarray(pcm16, dtype="<i2").tobytes()
    enc = _run(
        ["-loglevel", "error", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
         "-c:a", encoder, *(extra_args or []), "-f", fmt, "pipe:1"],
        raw,
    )
    if enc.returncode != 0 or not enc.stdout:
        err = enc.stderr.decode(errors="replace")[-300:]
        raise FFmpegError(f"encoding with {encoder} failed: {err}")
    dec = _run(
        ["-loglevel", "error", "-f", demuxer or fmt, "-i", "pipe:0",
         "-f", "s16le", "-ar", str(sr), "-ac", "1", "pipe:1"],
        enc.stdout,
    )
    if dec.returncode != 0:
        raise FFmpegError(f"decoding {fmt} failed: {dec.stderr.decode(errors='replace')[-300:]}")
    out = np.frombuffer(dec.stdout, dtype="<i2").astype(np.float64) / 32768.0
    return out, len(enc.stdout)


_INPUT_RE = re.compile(r"Input #0, ([^,]+(?:,[^,]+)*?), from")
_DURATION_RE = re.compile(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)(?:.*?bitrate: (\d+) kb/s)?")
_STREAM_RE = re.compile(r"Stream #0:\d+.*?: Audio: ([^\s,]+)(?: \([^)]*\))*(.*)")


def probe(path: Path) -> dict[str, Any] | None:
    """ffprobe-like inspection of the first audio stream via `ffmpeg -i`.

    Returns {container, codec, sample_rate, channels, duration_s, bitrate_kbps,
    stream_bitrate_kbps, raw} or None when ffmpeg is missing or finds no audio.
    """
    if ffmpeg_exe() is None:
        return None
    try:
        res = _run(["-i", str(path)])
    except (OSError, subprocess.SubprocessError, FFmpegError):
        return None
    text = res.stderr.decode(errors="replace")
    stream = _STREAM_RE.search(text)
    if stream is None:
        return None
    info: dict[str, Any] = {
        "container": None,
        "codec": stream.group(1),
        "sample_rate": None,
        "channels": None,
        "duration_s": None,
        "bitrate_kbps": None,
        "stream_bitrate_kbps": None,
        "raw": stream.group(0).strip(),
    }
    m = _INPUT_RE.search(text)
    if m:
        info["container"] = m.group(1).strip()
    rest = stream.group(2)
    if m := re.search(r"(\d+) Hz", rest):
        info["sample_rate"] = int(m.group(1))
    if m := re.search(r"Hz, ([^,]+)", rest):
        layout = m.group(1).strip()
        info["channels"] = {"mono": 1, "stereo": 2}.get(layout) or (
            int(layout.split()[0]) if layout.split()[0].isdigit() else None
        )
    if m := re.search(r"(\d+) kb/s", rest):
        info["stream_bitrate_kbps"] = float(m.group(1))
    d = _DURATION_RE.search(text)
    if d:
        h, mi, s = int(d.group(1)), int(d.group(2)), float(d.group(3))
        info["duration_s"] = h * 3600 + mi * 60 + s
        if d.group(4):
            info["bitrate_kbps"] = float(d.group(4))
    return info


def decode_file(path: Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Decode any file ffmpeg understands to mono float32 at `sr` (native rate if None)."""
    if sr is None:
        info = probe(path)
        if info is None or not info.get("sample_rate"):
            raise FFmpegError(f"cannot determine the sample rate of {path}")
        sr = int(info["sample_rate"])
    res = _run(["-loglevel", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr),
                "pipe:1"])
    if res.returncode != 0:
        raise FFmpegError(f"decoding {path} failed: {res.stderr.decode(errors='replace')[-300:]}")
    return np.frombuffer(res.stdout, dtype="<f4").copy(), sr
