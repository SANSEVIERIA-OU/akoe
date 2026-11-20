# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
import hashlib
import json

import pytest

from akoe.adapt.artifact import (
    CHECKSUMS,
    AdapterIntegrityError,
    hash_base_weights,
    load_adapter_artifact,
    read_checksums,
    save_adapter_artifact,
    verify_adapter,
    verify_train_manifest,
    weights_dir_of,
    weights_ref,
)
from akoe.models import AdapterArtifact, AdapterTechnique
from akoe.utils import sha256_file


@pytest.fixture
def adapter(tmp_path):
    root = tmp_path / "case"
    weights = root / "adapters" / "subj-v1" / "weights"
    weights.mkdir(parents=True)
    (weights / "adapter_config.json").write_text('{"r": 32}')
    (weights / "adapter_model.safetensors").write_bytes(b"\x00\x01weights")
    (weights / "sub").mkdir()
    (weights / "sub" / "extra.json").write_text("{}")
    art = AdapterArtifact(
        adapter_id="subj-v1", base_model="openai/whisper-small", technique=AdapterTechnique.LORA,
        hyperparams={"lora_r": 32}, seed=1, train_manifest_sha256="a" * 64,
        conditions=["car/local/unknown-codec/normal"], weights=weights_ref(weights, root),
    )
    return root, weights, art


def test_checksums_cover_every_file(adapter):
    root, weights, art = adapter
    sums = read_checksums(weights / CHECKSUMS)
    assert set(sums) == {"adapter_config.json", "adapter_model.safetensors", "sub/extra.json"}
    assert sums["adapter_config.json"] == sha256_file(weights / "adapter_config.json")
    assert art.weights.path == "adapters/subj-v1/weights/SHA256SUMS"  # relative to the case root
    assert art.weights.kind == "adapter_weights"
    assert verify_adapter(art, root) == weights == weights_dir_of(art, root)


def test_artifact_round_trip_with_hash(adapter, tmp_path):
    root, _, art = adapter
    ref = save_adapter_artifact(art, root / "adapters" / "subj-v1" / "adapter.json", root)
    assert ref.kind == "adapter" and ref.path == "adapters/subj-v1/adapter.json"
    path = ref.resolve(root)
    assert load_adapter_artifact(path, expected_sha256=ref.sha256) == art
    with pytest.raises(AdapterIntegrityError, match="impronta SHA-256 attesa"):
        load_adapter_artifact(path, expected_sha256="0" * 64)
    with pytest.raises(AdapterIntegrityError, match="non trovato"):
        load_adapter_artifact(tmp_path / "missing.json")


@pytest.mark.parametrize(
    "tamper, message",
    [
        (lambda w: (w / "adapter_model.safetensors").write_bytes(b"other"), "non corrisponde per adapter_model"),
        (lambda w: (w / "adapter_config.json").unlink(), "file mancanti"),
        (lambda w: (w / "injected.bin").write_bytes(b"x"), "file non previsti"),
        (lambda w: (w / CHECKSUMS).write_text((w / CHECKSUMS).read_text()[:-5] + "x\n"), "dei pesi non corrisponde"),
        (lambda w: (w / CHECKSUMS).unlink(), "elenco delle impronte non trovato"),
    ],
)
def test_tampering_is_refused_with_italian_message(adapter, tamper, message):
    root, weights, art = adapter
    tamper(weights)
    with pytest.raises(AdapterIntegrityError, match=message) as exc:
        verify_adapter(art, root)
    assert "rifiutato" in str(exc.value) or "non trovato" in str(exc.value)


def test_regenerated_checksums_do_not_match_the_artifact(adapter):
    """Replacing weights and rewriting SHA256SUMS is caught by the artifact hash."""
    root, weights, art = adapter
    (weights / "adapter_model.safetensors").write_bytes(b"replacement")
    weights_ref(weights, root)
    with pytest.raises(AdapterIntegrityError, match="dei pesi non corrisponde"):
        verify_adapter(art, root)


def test_verify_train_manifest(adapter, tmp_path):
    _, _, art = adapter
    m = tmp_path / "train_manifest.json"
    m.write_text("{}\n")
    good = art.model_copy(update={"train_manifest_sha256": sha256_file(m)})
    verify_train_manifest(good, m)
    with pytest.raises(AdapterIntegrityError, match="manifesto"):
        verify_train_manifest(art, m)


def test_hash_base_weights_local(tmp_path):
    assert hash_base_weights(str(tmp_path)) is None
    (tmp_path / "model.safetensors").write_bytes(b"single")
    assert hash_base_weights(str(tmp_path)) == hashlib.sha256(b"single").hexdigest()
    (tmp_path / "model.safetensors").unlink()
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"one")
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"two")
    lines = "".join(
        f"{hashlib.sha256(b).hexdigest()}  model-0000{i}-of-00002.safetensors\n"
        for i, b in ((1, b"one"), (2, b"two"))
    )
    assert hash_base_weights(str(tmp_path)) == hashlib.sha256(lines.encode()).hexdigest()


def test_hash_base_weights_unknown_hub_model_is_none():
    assert hash_base_weights("nobody/not-a-cached-model-xyz") is None


def test_artifact_json_is_canonical(adapter):
    root, _, art = adapter
    p = root / "a.json"
    save_adapter_artifact(art, p)
    data = json.loads(p.read_text())
    assert list(data) == sorted(data)
