# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
import json

import numpy as np
import pytest

from akoe.case import Case
from akoe.config import Channel, RecordingSource, VocalEffort
from akoe.models import DatasetSplit, RecordingCondition
from akoe.speaker.dataset import (
    AlignedSpan,
    DatasetError,
    QualityWarning,
    Severity,
    SubjectDataset,
    TranscriptAligner,
    TranscriptFormat,
    find_unintelligible_markers,
    holdout_target,
    parse_json_transcript,
    parse_srt,
)
from tests.synth_voice import SR, write_wav

CAR = RecordingCondition(channel=Channel.CAR, source=RecordingSource.LOCAL, vocal_effort=VocalEffort.NORMAL)
PHONE = RecordingCondition(channel=Channel.PHONE, source=RecordingSource.OTHER, codec="amr-nb")
ROOM = RecordingCondition(channel=Channel.ROOM, source=RecordingSource.OTHER)


def _noise_wav(path, seconds=4.0, seed=0, subtype="PCM_16"):
    rng = np.random.default_rng(seed)
    return write_wav(path, (rng.normal(0, 0.05, int(SR * seconds))).astype(np.float32), subtype=subtype)


def _json(path, segments, **extra):
    path.write_text(json.dumps({"schema": "akoe.transcript/v1", "segments": segments, **extra}))
    return path


@pytest.fixture()
def case(tmp_path):
    return Case.create(tmp_path / "case", "case-1")


def codes(ds, item_id=None):
    return {w.code for w in ds.warnings if item_id is None or w.item_id == item_id}


# -- parsers ----------------------------------------------------------------


def test_parse_srt_handles_tags_bom_and_malformed_blocks():
    srt = (
        "﻿1\r\n00:00:01,000 --> 00:00:02,500\r\n<i>vado a</i>\r\ncasa\r\n\r\n"
        "2\nnot a time line\nblah\n\n"
        "3\n00:00:03.2 --> 00:00:04.75\nposso venire?\n"
    )
    p = parse_srt(srt)
    assert p.timed and p.format == TranscriptFormat.SRT
    assert [(s.start, s.end, s.text) for s in p.segments] == [
        (1.0, 2.5, "vado a casa"), (3.2, 4.75, "posso venire?")
    ]
    assert len(p.warnings) == 1 and "blocco 2" in p.warnings[0]
    with pytest.raises(DatasetError):
        parse_srt("solo testo senza tempi")


def test_parse_json_schema_and_bare_list():
    p = parse_json_transcript(json.dumps({"segments": [{"start": 0, "end": 1.5, "text": " ciao  bello ", "x": 1}]}))
    assert p.segments[0].text == "ciao bello" and p.segments[0].start == 0.0
    p = parse_json_transcript(json.dumps([{"start": 0, "end": 1, "text": "a", "speaker": "S"}]))
    assert p.segments[0].speaker == "S"
    p = parse_json_transcript(json.dumps({"schema": "other/v9", "segments": []}))
    assert p.warnings
    with pytest.raises(DatasetError):
        parse_json_transcript(json.dumps({"segments": [{"text": "manca il tempo"}]}))
    with pytest.raises(DatasetError):
        parse_json_transcript("{not json")


def test_unintelligible_markers():
    assert find_unintelligible_markers("allora [incomprensibile] domani")
    assert find_unintelligible_markers("gli ho detto (inc.) va bene")
    assert find_unintelligible_markers("e poi ??? basta")
    assert find_unintelligible_markers("[voci sovrapposte] no")
    assert not find_unintelligible_markers("vado a casa, poi vediamo.")


# -- import -----------------------------------------------------------------


def test_import_srt_creates_one_item_per_segment(case, tmp_path):
    ds = SubjectDataset.load_or_create(case, "subj1")
    wav = _noise_wav(tmp_path / "car1.wav", seconds=5.0)
    srt = tmp_path / "car1.srt"
    srt.write_text("1\n00:00:00,500 --> 00:00:02,000\nvado a casa\n\n2\n00:00:02,500 --> 00:00:04,000\npoi esco\n")
    rec = ds.import_recording(case, wav, CAR, transcript_path=srt)

    assert rec.recording_id.startswith("ev-") and rec.duration == pytest.approx(5.0)
    assert rec.condition.sample_rate == SR and rec.condition.codec is None  # PCM says nothing
    assert rec.transcript is not None and rec.transcript.verify(case.root)
    assert not rec.needs_alignment
    items = ds.items_of(rec.recording_id)
    assert [it.text for it in items] == ["vado a casa", "poi esco"]
    assert all(it.audio.sha256 == rec.audio.sha256 and not it.verified for it in items)
    assert case.get(rec.recording_id).condition.channel == Channel.CAR  # evidence, read-only, hashed
    assert case.verify() == []
    # Same bytes again -> same entry, nothing duplicated.
    assert ds.import_recording(case, wav, CAR, transcript_path=srt) == rec
    assert len(ds.items) == 2


def test_codec_detected_from_header_when_not_given(case, tmp_path):
    ds = SubjectDataset.load_or_create(case, "subj1")
    wav = _noise_wav(tmp_path / "tel.wav", subtype="ULAW")
    rec = ds.import_recording(case, wav, RecordingCondition(channel=Channel.PHONE))
    assert rec.condition.codec == "g711-ulaw" and rec.audio_format == "WAV/ULAW"
    # The operator's codec always wins over the header.
    wav2 = _noise_wav(tmp_path / "tel2.wav", seed=3, subtype="ULAW")
    assert ds.import_recording(case, wav2, PHONE).condition.codec == "amr-nb"


def test_plain_text_is_one_item_needing_alignment(case, tmp_path):
    ds = SubjectDataset.load_or_create(case, "subj1")
    wav = _noise_wav(tmp_path / "room.wav", seconds=6.0)
    txt = tmp_path / "room.txt"
    txt.write_text("allora domani\nandiamo al mare\n")
    rec = ds.import_recording(case, wav, ROOM, transcript_path=txt)
    assert rec.needs_alignment and rec.transcript_format == TranscriptFormat.TEXT
    (item,) = ds.items_of(rec.recording_id)
    assert (item.start, item.end, item.text) == (0.0, pytest.approx(6.0), "allora domani andiamo al mare")
    assert "needs_alignment" in codes(ds)
    assert "too_long" not in codes(ds, item.item_id)  # whole-recording items are not judged by length
    assert ds.select_items(splits=None, verified_only=False) == []  # unaligned: not usable
    assert ds.select_items(splits=None, verified_only=False, include_unaligned=True) == [item]


def test_quality_checks(case, tmp_path):
    ds = SubjectDataset.load_or_create(case, "subj1")
    wav = _noise_wav(tmp_path / "car.wav", seconds=40.0)
    segs = [
        {"start": 0.0, "end": 2.0, "text": "tutto chiaro"},
        {"start": 2.0, "end": 2.2, "text": "eh"},
        {"start": 3.0, "end": 4.0, "text": "   "},
        {"start": 4.0, "end": 5.0, "text": "gli ho detto [incomprensibile] domani"},
        {"start": 4.5, "end": 6.0, "text": "sovrapposto [ride]"},
        {"start": 6.0, "end": 39.0, "text": "lunghissimo"},
        {"start": 39.0, "end": 41.0, "text": "oltre la fine"},
        {"start": 10.0, "end": 9.0, "text": "tempi invertiti"},
    ]
    rec = ds.import_recording(case, wav, CAR, transcript_path=_json(tmp_path / "car.json", segs))
    ids = [f"{rec.recording_id}-s{k:04d}" for k in range(len(segs))]
    assert codes(ds, ids[0]) == set()
    assert "too_short" in codes(ds, ids[1])
    assert "empty_text" in codes(ds, ids[2])
    assert "unintelligible_marker" in codes(ds, ids[3])
    assert {"overlap", "annotation"} <= codes(ds, ids[4])
    assert "too_long" in codes(ds, ids[5])
    assert "outside_audio" in codes(ds, ids[6])
    assert "invalid_times" in codes(ds, ids[7])
    assert ds.blocked_item_ids() == {ids[2], ids[6], ids[7]}
    assert all(w.message for w in ds.warnings)


def test_custom_quality_check_hook(case, tmp_path):
    ds = SubjectDataset.load_or_create(case, "subj1")

    def italianized(item, recording):
        if "andiamo" in item.text:
            yield QualityWarning(code="italianized_dialect", severity=Severity.WARNING,
                                 message="Possibile dialetto italianizzato", item_id=item.item_id,
                                 recording_id=recording.recording_id)

    ds.register_check(italianized)
    wav = _noise_wav(tmp_path / "car.wav")
    ds.import_recording(case, wav, CAR, transcript_path=_json(
        tmp_path / "car.json", [{"start": 0, "end": 1, "text": "jamme"}, {"start": 1, "end": 2, "text": "andiamo"}]))
    flagged = [w for w in ds.warnings if w.code == "italianized_dialect"]
    assert len(flagged) == 1 and flagged[0].item_id.endswith("-s0001")


def test_speaker_labels_keep_other_speakers_apart(case, tmp_path):
    ds = SubjectDataset.load_or_create(case, "subj1")
    segs = [
        {"start": 0, "end": 1, "text": "pronto", "speaker": "A"},
        {"start": 1, "end": 2, "text": "ciao sono io", "speaker": "B"},
        {"start": 2, "end": 3, "text": "dimmi", "speaker": "A"},
    ]
    wav = _noise_wav(tmp_path / "tel.wav")
    rec = ds.import_recording(case, wav, PHONE, transcript_path=_json(tmp_path / "tel.json", segs),
                              subject_speaker="A")
    assert [it.text for it in ds.items_of(rec.recording_id)] == ["pronto", "dimmi"]
    assert [(s.speaker, s.text) for s in ds.other_speaker_spans] == [("B", "ciao sono io")]

    wav2 = _noise_wav(tmp_path / "tel2.wav", seed=9)
    ds.import_recording(case, wav2, PHONE, transcript_path=_json(tmp_path / "tel2.json", segs))
    assert "multiple_speakers" in codes(ds)


def test_align_pending_with_external_aligner(case, tmp_path):
    class FakeAligner:
        name = "fake-aligner"

        def align(self, audio, sr, text):
            words = text.split()
            dur = len(audio) / sr
            return [AlignedSpan(start=0.0, end=dur / 2, text=" ".join(words[:2]), score=0.9),
                    AlignedSpan(start=dur / 2, end=dur, text=" ".join(words[2:]), score=0.1)]

    assert isinstance(FakeAligner(), TranscriptAligner)
    ds = SubjectDataset.load_or_create(case, "subj1")
    wav = _noise_wav(tmp_path / "room.wav", seconds=4.0)
    txt = tmp_path / "room.txt"
    txt.write_text("allora domani andiamo al mare")
    rec = ds.import_recording(case, wav, ROOM, transcript_path=txt)
    ds.items_of(rec.recording_id)[0].verified = True
    created = ds.align_pending(case, FakeAligner(), min_score=0.5)
    assert [it.text for it in created] == ["allora domani", "andiamo al mare"]
    assert not any(it.verified for it in created)  # verification must be repeated
    assert not ds.recording(rec.recording_id).needs_alignment
    assert ds.recording(rec.recording_id).aligned_by == "fake-aligner"
    misaligned = [w for w in ds.warnings if w.code == "possible_misalignment"]
    assert [w.item_id for w in misaligned] == [created[1].item_id]
    assert "verification_reset" in codes(ds)


def test_import_folder_with_index_and_by_stem(case, tmp_path):
    folder = tmp_path / "subject"
    folder.mkdir()
    _noise_wav(folder / "loc.wav", seed=1)
    _noise_wav(folder / "tx.wav", seed=2)
    _json(folder / "loc.json", [{"start": 0, "end": 1, "text": "uno"}])
    (folder / "recordings.json").write_text(json.dumps({"recordings": [
        {"audio": "loc.wav", "transcript": "loc.json", "channel": "car", "source": "local", "pair_id": "evt-1"},
        {"audio": "tx.wav", "channel": "car", "source": "transmitted", "codec": "opus", "pair_id": "evt-1"},
        {"audio": "missing.wav", "channel": "car"},
    ]}))
    ds = SubjectDataset.load_or_create(case, "subj1")
    recs = ds.import_folder(case, folder)
    assert [r.condition.source for r in recs] == [RecordingSource.LOCAL, RecordingSource.TRANSMITTED]
    assert {r.pair_id for r in recs} == {"evt-1"} and set(case.pairs()) == {"evt-1"}
    assert "import_failed" in codes(ds)

    other = tmp_path / "phone"
    other.mkdir()
    _noise_wav(other / "a.wav", seed=5)
    (other / "a.txt").write_text("testo")
    (other / "a.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nciao\n")
    (rec,) = ds.import_folder(case, other, PHONE)
    assert rec.transcript_format == TranscriptFormat.SRT  # timed preferred over plain text


def test_save_and_load_round_trip(case, tmp_path):
    ds = SubjectDataset.load_or_create(case, "subj1")
    ds.import_recording(case, _noise_wav(tmp_path / "a.wav"), CAR,
                        transcript_path=_json(tmp_path / "a.json", [{"start": 0, "end": 1, "text": "uno"}]))
    ds.save(case)
    back = SubjectDataset.load_or_create(case, "subj1")
    assert back == ds
    with pytest.raises(ValueError):
        SubjectDataset.load_or_create(case, "../escape")


# -- split ------------------------------------------------------------------


def _many(case, tmp_path, n_single=8, pairs=("evt-1", "evt-2"), seed0=0):
    ds = SubjectDataset.load_or_create(case, "subj1")
    k = seed0
    for i in range(n_single):
        cond = [CAR, PHONE, ROOM][i % 3]
        wav = _noise_wav(tmp_path / f"r{k}.wav", seconds=2.0, seed=100 + k)
        ds.import_recording(case, wav, cond, transcript_path=_json(
            tmp_path / f"r{k}.json", [{"start": 0, "end": 0.9, "text": "a"}, {"start": 1, "end": 1.9, "text": "b"}]))
        k += 1
    for p in pairs:
        for src in (RecordingSource.LOCAL, RecordingSource.TRANSMITTED):
            wav = _noise_wav(tmp_path / f"r{k}.wav", seconds=2.0, seed=100 + k)
            ds.import_recording(case, wav, RecordingCondition(channel=Channel.CAR, source=src), pair_id=p,
                                transcript_path=_json(tmp_path / f"r{k}.json", [{"start": 0, "end": 1, "text": "c"}]))
            k += 1
    return ds


def test_holdout_target():
    assert holdout_target(1, 0.2) == 0
    assert holdout_target(2, 0.2) == 1
    assert holdout_target(10, 0.2) == 2
    assert holdout_target(10, 0.0) == 0
    assert holdout_target(3, 0.9) == 2


def test_split_by_recording_keeps_pairs_together_and_is_deterministic(case, tmp_path):
    ds = _many(case, tmp_path)
    splits = ds.split_by_recording()  # default 20%, seed 0
    assert len(splits) == 12
    groups = {}
    for rec in ds.recordings:
        groups.setdefault(rec.pair_id or rec.recording_id, set()).add(rec.split)
    assert all(len(s) == 1 for s in groups.values())  # pairs never split
    assert sum(1 for s in groups.values() if s == {DatasetSplit.HOLDOUT}) == 2  # 20% of 10 groups
    for rec in ds.recordings:  # never by segment
        assert {it.split for it in ds.items_of(rec.recording_id)} == {rec.split}

    again = SubjectDataset.model_validate(ds.model_dump())
    again.split_by_recording(reset=True)
    assert {r.recording_id: r.split for r in again.recordings} == splits
    different = {}
    for seed in range(1, 6):
        again.split_by_recording(seed=seed, reset=True)
        different[seed] = tuple(again.holdout_recordings())
    assert len(set(different.values())) > 1


def test_split_assignments_are_frozen_when_recordings_are_added(case, tmp_path):
    ds = _many(case, tmp_path, n_single=5, pairs=())
    first = ds.split_by_recording(holdout_fraction=0.2, seed=7)
    ds2 = SubjectDataset.model_validate(ds.model_dump())
    for k in range(50, 55):
        wav = _noise_wav(tmp_path / f"r{k}.wav", seconds=2.0, seed=100 + k)
        ds2.import_recording(case, wav, CAR)
    second = ds2.split_by_recording(holdout_fraction=0.2, seed=7)
    assert all(second[rid] == s for rid, s in first.items())
    assert len(ds2.split_history) == 2 and len(ds2.split_history[1].assigned_groups) == 5
    assert len(ds2.holdout_recordings()) == 2  # 20% of 10


def test_new_pair_member_inherits_split(case, tmp_path):
    ds = SubjectDataset.load_or_create(case, "subj1")
    for k in range(4):
        ds.import_recording(case, _noise_wav(tmp_path / f"p{k}.wav", seed=k), CAR, pair_id=f"evt-{k}")
    ds.split_by_recording(holdout_fraction=0.5)
    held = ds.holdout_recordings()[0]
    pair = ds.recording(held).pair_id
    tx = ds.import_recording(case, _noise_wav(tmp_path / "tx.wav", seed=99),
                             RecordingCondition(channel=Channel.CAR, source=RecordingSource.TRANSMITTED),
                             pair_id=pair)
    ds.split_by_recording(holdout_fraction=0.5)
    assert ds.recording(tx.recording_id).split == DatasetSplit.HOLDOUT


def test_manifest_is_hashed_and_deterministic(case, tmp_path):
    ds = _many(case, tmp_path, n_single=4, pairs=("evt-1",))
    ds.split_by_recording()
    ref = ds.write_manifest(case)
    assert ref.kind == "subject_dataset_manifest" and ref.verify(case.root)
    assert ref.path.startswith("profiles/subject-subj1/manifests/dataset-")
    data = json.loads(ref.resolve(case.root).read_text())
    assert data["subject_id"] == "subj1" and data["split_history"][0]["seed"] == 0
    assert {r["split"] for r in data["recordings"]} == {"train", "holdout"}
    assert data["summary"]["split"]["holdout"]["recordings"] >= 1
    again = ds.write_manifest(case)
    assert (again.path, again.sha256) == (ref.path, ref.sha256)
    ds.items[0].verified = True
    assert ds.write_manifest(case).sha256 != ref.sha256
