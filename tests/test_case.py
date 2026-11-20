import os

import pytest

from forensicaudio.case import CASE_SUBDIRS, Case, CaseError
from forensicaudio.config import Channel, RecordingSource
from forensicaudio.models import RecordingCondition


def _rec(path, data: bytes):
    path.write_bytes(data)
    return path


def test_create_and_open(tmp_path):
    root = tmp_path / "case1"
    Case.create(root, "case1")
    for sub in CASE_SUBDIRS:
        assert (root / sub).is_dir()
    assert Case.open(root).manifest.case_id == "case1"
    with pytest.raises(CaseError):
        Case.create(root, "case1")


def test_import_is_read_only_hashed_and_idempotent(tmp_path):
    case = Case.create(tmp_path / "c", "c")
    src = _rec(tmp_path / "Rec 01.WAV", b"RIFF....fake audio bytes")
    item = case.import_evidence(src, RecordingCondition(channel=Channel.CAR))
    dest = case.path_of(item)
    assert dest.name.startswith("ev-") and dest.suffix == ".wav"
    assert not os.access(dest, os.W_OK)
    assert item.original_name == "Rec 01.WAV"
    assert case.import_evidence(src) == item  # same bytes -> same item
    assert len(Case.open(case.root).manifest.evidence) == 1


def test_verify_detects_tampering(tmp_path):
    case = Case.create(tmp_path / "c", "c")
    item = case.import_evidence(_rec(tmp_path / "a.wav", b"original"))
    assert case.verify() == []
    dest = case.path_of(item)
    dest.chmod(0o644)
    dest.write_bytes(b"tampered")
    assert case.verify() == [item.evidence_id]


def test_pairs_prefer_local_recording(tmp_path):
    case = Case.create(tmp_path / "c", "c")
    tx = case.import_evidence(_rec(tmp_path / "tx.ogg", b"transmitted"),
                              RecordingCondition(source=RecordingSource.TRANSMITTED, codec="opus"),
                              pair_id="evt-1")
    loc = case.import_evidence(_rec(tmp_path / "loc.wav", b"local"),
                               RecordingCondition(source=RecordingSource.LOCAL), pair_id="evt-1")
    assert {i.evidence_id for i in case.pairs()["evt-1"]} == {tx.evidence_id, loc.evidence_id}
    assert case.preferred_input("evt-1") == loc
    with pytest.raises(CaseError):
        case.preferred_input("missing")
