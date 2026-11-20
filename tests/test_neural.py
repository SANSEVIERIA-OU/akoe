# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
import numpy as np
import pytest

from akoe.neural import (
    BackendUnavailable,
    ModelProvenance,
    hash_files,
    iter_segments,
    prepare_audio,
    resolve_device,
    resolve_hf_snapshot,
)
from akoe.utils import sha256_file


def test_backend_unavailable_is_logged_as_skipped():
    # The pipeline records NotImplementedError as "skipped" (IMPLEMENTATION_PLAN §6.5).
    assert issubclass(BackendUnavailable, NotImplementedError)


def test_provenance_digest_and_stage_log_fields(tmp_path):
    (tmp_path / "w.bin").write_bytes(b"weights")
    (tmp_path / "cfg.json").write_text("{}")
    weights = hash_files(tmp_path.iterdir(), root=tmp_path)
    assert weights["w.bin"] == sha256_file(tmp_path / "w.bin")
    p = ModelProvenance(backend="x/generic", model_id="org/x", revision="abc", weights=weights)
    q = ModelProvenance(backend="x/generic", model_id="org/x", revision="abc", weights=dict(weights))
    assert p.weights_digest() == q.weights_digest()
    fields = p.stage_log_fields()
    assert fields == {"model_version": "org/x@abc", "model_sha256": p.weights_digest(), "device": "cpu"}
    # an adapter changes the digest: generic and adapted models are distinguishable in the log
    r = q.model_copy(update={"adapter": {"weights": {"adapter_model.safetensors": "00"}}})
    assert r.weights_digest() != p.weights_digest()
    assert p.as_params()["weights"] == weights


def test_prepare_audio_resamples_and_rejects_multichannel():
    a = np.sin(np.linspace(0, 100, 8000)).astype(np.float64)
    out = prepare_audio(a, 8000)
    assert out.dtype == np.float32 and len(out) == 16000
    with pytest.raises(ValueError, match="mono"):
        prepare_audio(np.zeros((2, 100)), 16000)
    with pytest.raises(ValueError):
        prepare_audio(np.zeros(10), 0)


def test_iter_segments_clips_and_splits():
    audio = np.arange(16000 * 70, dtype=np.float32)
    segs = iter_segments(audio, 16000, [(1.0, 2.0), (5.0, 80.0)], max_seconds=30.0)
    bounds = [(s, e) for s, e, _ in segs]
    assert bounds == [(1.0, 2.0), (5.0, 35.0), (35.0, 65.0), (65.0, 70.0)]
    assert all(len(c) == round((e - s) * 16000) for s, e, c in segs)
    assert len(iter_segments(audio[:16000], 16000, None)) == 1
    with pytest.raises(ValueError, match="Segmento non valido"):
        iter_segments(audio, 16000, [(80.0, 90.0)])


def test_resolve_device_and_local_snapshot(tmp_path):
    assert resolve_device("cpu") == "cpu"
    assert resolve_hf_snapshot(str(tmp_path)) == (tmp_path, None)
