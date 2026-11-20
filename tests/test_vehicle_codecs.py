import logging
import subprocess

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import butter, sosfilt

from forensicaudio.vehicle import codecs, ffmpeg
from forensicaudio.vehicle.audio import estimate_lag, load_audio
from forensicaudio.vehicle.codecs import (
    CODEC_SPECS,
    CodecAugmenter,
    alaw_decode,
    alaw_encode,
    apply_codec,
    available_codecs,
    detect_codec,
    ulaw_decode,
    ulaw_encode,
)

SR = 16000
ALL_NAMES = sorted({s.name for s in CODEC_SPECS})


@pytest.fixture(scope="module")
def signal():
    rng = np.random.default_rng(0)
    n = int(1.5 * SR) + 7
    x = sosfilt(butter(4, [200, 3000], btype="band", fs=SR, output="sos"), rng.standard_normal(n))
    env = np.repeat(rng.random(n // 800 + 1) > 0.4, 800)[:n]
    return (0.2 * x / np.std(x) * env).astype(np.float32)


@pytest.fixture
def fresh_availability():
    available_codecs.cache_clear()
    yield
    available_codecs.cache_clear()


def test_g711_bit_exact_with_audioop():
    audioop = pytest.importorskip("audioop")
    pcm = np.arange(-32768, 32768).astype(np.int16)
    assert np.array_equal(ulaw_encode(pcm), np.frombuffer(audioop.lin2ulaw(pcm.tobytes(), 2), np.uint8))
    assert np.array_equal(alaw_encode(pcm), np.frombuffer(audioop.lin2alaw(pcm.tobytes(), 2), np.uint8))
    codes = np.arange(256, dtype=np.uint8)
    assert np.array_equal(ulaw_decode(codes), np.frombuffer(audioop.ulaw2lin(codes.tobytes(), 2), np.int16))
    assert np.array_equal(alaw_decode(codes), np.frombuffer(audioop.alaw2lin(codes.tobytes(), 2), np.int16))


def test_g711_roundtrip_quality():
    pcm = (8000 * np.sin(np.arange(8000) * 0.05)).astype(np.int16)
    for enc, dec in ((ulaw_encode, ulaw_decode), (alaw_encode, alaw_decode)):
        err = dec(enc(pcm)).astype(float) - pcm
        assert 10 * np.log10(np.mean(pcm.astype(float) ** 2) / np.mean(err**2)) > 30


@pytest.mark.parametrize("name", ALL_NAMES)
def test_codec_keeps_rate_length_and_alignment(name, signal):
    if name not in available_codecs():
        pytest.skip(f"{name} not available on this machine")
    y, p = apply_codec(signal, SR, name, rng=3)
    assert y.dtype == np.float32 and len(y) == len(signal)
    assert p["codec"] == name and p["sample_rate"] == SR and not p.get("skipped")
    assert p["codec_sample_rate"] in available_codecs()[name].sample_rates
    assert p["measured_bitrate_kbps"] > 0
    assert "option" in p and "nominal_bitrate_kbps" in p
    lag, corr = estimate_lag(signal, y, 200)
    assert abs(lag) <= 1 and corr > 0.3
    # Same rng seed -> same output.
    y2, p2 = apply_codec(signal, SR, name, rng=3)
    assert p2 == p and np.array_equal(y, y2)


def test_mp3_encoder_delay_is_compensated(signal):
    if "mp3" not in available_codecs():
        pytest.skip("mp3 not available")
    y, p = apply_codec(signal, SR, "mp3", codec_sr=8000,
                       option={"compression_level": 0.9, "bitrate_mode": "CONSTANT"})
    assert p["delay_compensated_samples"] != 0
    assert abs(estimate_lag(signal, y, 4000)[0]) <= 1


def test_unknown_codec_is_skipped_with_warning(signal, caplog):
    with caplog.at_level(logging.WARNING):
        y, p = apply_codec(signal, SR, "speex-imaginary")
    assert p["skipped"] and p["codec"] == "none" and p["requested_codec"] == "speex-imaginary"
    assert np.array_equal(y, signal)
    assert "non disponibile" in caplog.text


def test_missing_ffmpeg_skips_ffmpeg_codecs(signal, caplog, monkeypatch, fresh_availability):
    monkeypatch.setattr(ffmpeg, "ffmpeg_exe", lambda: None)
    assert "amr_nb" not in available_codecs()
    assert "g711_ulaw" in available_codecs()
    with caplog.at_level(logging.WARNING):
        y, p = apply_codec(signal, SR, "amr_nb")
        aug = CodecAugmenter(codecs=["amr_nb", "g711_alaw"])
    assert p["skipped"] and "ffmpeg non trovato" in p["reason"]
    assert aug.active == ["g711_alaw"] and "amr_nb" in aug.unavailable
    assert "escluso dalla simulazione" in caplog.text


def test_augmenter_draws_a_codec_per_example(signal, caplog):
    names = [n for n in ("g711_ulaw", "g711_alaw", "gsm", "opus") if n in available_codecs()]
    with caplog.at_level(logging.WARNING):
        aug = CodecAugmenter(codecs=names + ["no-such-codec"])
    assert "no-such-codec" in aug.unavailable and aug.active == names
    short = signal[: SR // 2]
    drawn = {aug(short, SR, rng=s)[1]["codec"] for s in range(20)}
    assert len(drawn) >= min(2, len(names))
    a, pa = aug(short, SR, rng=7)
    b, pb = aug(short, SR, rng=7)
    assert pa == pb and np.array_equal(a, b)
    none = CodecAugmenter(codecs=names, probability=0.0)
    assert none(short, SR, rng=1)[1]["codec"] == "none"
    assert aug.describe()["active"] == names


def test_detect_codec_from_soundfile_metadata(tmp_path, signal):
    cases = [("pcm.wav", "WAV", "PCM_16", SR, "pcm_16", False), ("ulaw.wav", "WAV", "ULAW", 8000, "g711_ulaw", True)]
    for fmt, sub, name in (("OGG", "OPUS", "opus"), ("OGG", "VORBIS", "vorbis"), ("MP3", "MPEG_LAYER_III", "mp3")):
        if sub in sf.available_subtypes(fmt):
            cases.append((f"x.{name}", fmt, sub, 16000, name, True))
    for fname, fmt, sub, sr, expected, lossy in cases:
        path = tmp_path / fname
        sf.write(str(path), signal[:sr], sr, format=fmt, subtype=sub)
        info = detect_codec(path)
        assert info["codec"] == expected, fname
        assert info["lossy"] is lossy and info["sample_rate"] == sr and info["method"] == "soundfile"
        assert info["bitrate_kbps"] > 0


def test_detect_and_load_amr_via_ffmpeg(tmp_path, signal):
    if "amr_nb" not in available_codecs():
        pytest.skip("no AMR-NB encoder")
    path = tmp_path / "tx.amr"
    pcm = (np.clip(signal[::2], -1, 1) * 32767).astype("<i2").tobytes()
    subprocess.run([ffmpeg.ffmpeg_exe(), "-nostdin", "-loglevel", "error", "-f", "s16le", "-ar", "8000",
                    "-ac", "1", "-i", "pipe:0", "-c:a", "libopencore_amrnb", "-b:a", "12200", str(path)],
                   input=pcm, check=True)
    info = detect_codec(path)
    assert info["codec"] == "amr_nb" and info["method"] == "ffmpeg"
    assert info["sample_rate"] == 8000 and info["channels"] == 1 and info["lossy"]
    audio, sr = load_audio(path, sr=SR)
    assert sr == SR and abs(len(audio) - len(signal)) < 0.05 * SR


def test_codec_report_lists_every_codec():
    report = codecs.codec_report()
    assert set(report) == set(ALL_NAMES)
    assert report["g711_ulaw"]["available"]
