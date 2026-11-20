# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Subject dataset: recordings from every context, existing transcriptions,
quality checks and a train/holdout split by recording (IMPLEMENTATION_PLAN M1,
issues #6 and #7). DETERMINISTIC.

Recordings
----------
Every recording of the subject (phone intercepts, rooms, car, interviews) is
imported read-only into the case through `Case.import_evidence` and carries a
`RecordingCondition`. When the operator leaves `codec` / `sample_rate` empty,
they are filled from the file header (only lossy or telephony codecs: a PCM
or FLAC container says nothing about the channel upstream). Local and
transmitted recordings of the same event share a `pair_id` and always land in
the same split.

Transcription formats
---------------------
- Plain text (``.txt``): the whole transcription of one recording, without
  times. It becomes ONE `DatasetItem` spanning the whole recording, and the
  recording is flagged ``needs_alignment``. Alignment needs an acoustic model
  and lives elsewhere: any object implementing `TranscriptAligner` can be
  passed to `SubjectDataset.align_pending`.
- SubRip (``.srt``): one `DatasetItem` per cue.
- JSON (``.json``), schema ``akoe.transcript/v1``::

      {
        "schema": "akoe.transcript/v1",    (optional)
        "language": "it",                            (optional)
        "segments": [
          {"start": 12.30, "end": 15.85, "text": "vado a casa", "speaker": "SOGGETTO"},
          ...
        ]
      }

  ``start`` / ``end`` are seconds from the beginning of the recording
  (required, numbers); ``text`` is required; ``speaker`` is optional. Unknown
  keys are ignored. A bare list of segments is accepted too. When speaker
  labels are present, pass ``subject_speaker``: segments of other speakers do
  not become DatasetItems and are kept as `SpeakerSpan`s (other-speaker
  material for score calibration, see `profile.py`); unlabelled segments are
  attributed to the subject.

Folder import
-------------
`SubjectDataset.import_folder` reads an optional index ``recordings.json``::

      {"recordings": [
        {"audio": "tel_01.wav", "transcript": "tel_01.srt", "channel": "phone",
         "source": "other", "vocal_effort": "normal", "codec": "amr-nb",
         "pair_id": null, "subject_speaker": "A", "notes": "..."}
      ]}

Without an index, every audio file in the folder is imported with the same
condition, and a transcription with the same stem is attached (preference
``.json`` > ``.srt`` > ``.txt``).

Split
-----
Train / holdout by recording group, never by segment: a group is a `pair_id`
or a single recording. Default 20% of the groups in holdout (at least one
when there are two or more groups), stratified by channel. The choice is
deterministic and library-independent: within each stratum groups are ordered
by SHA-256 of ``"<seed>:<group key>"``. Assignments are frozen: a later call
assigns only new recordings, so material once used as holdout never moves to
training (and vice versa) unless ``reset=True``.

Every operator-facing message is in Italian.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Literal, Protocol, runtime_checkable

import numpy as np
import soundfile as sf
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError

from akoe.case import Case
from akoe.config import Channel, RecordingSource, VocalEffort
from akoe.models import ArtifactRef, DatasetItem, DatasetSplit, EvidenceRole, RecordingCondition
from akoe.speaker._io import (
    check_id,
    load_audio,
    subject_dir,
    write_content_addressed,
    write_json_atomic,
)
from akoe.utils import sha256_file

DATASET_SCHEMA = "akoe.subject-dataset/v1"
MANIFEST_SCHEMA = "akoe.subject-dataset-manifest/v1"
TRANSCRIPT_SCHEMA = "akoe.transcript/v1"
FOLDER_INDEX = "recordings.json"
AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".opus", ".mp3", ".m4a", ".aac", ".amr", ".aif", ".aiff")


class DatasetError(Exception):
    """Raised with an operator-facing (Italian) message."""


# --------------------------------------------------------------------------
# Transcriptions
# --------------------------------------------------------------------------


class TranscriptFormat(str, Enum):
    TEXT = "text"
    SRT = "srt"
    JSON = "json"


class TranscriptSegment(BaseModel):
    start: float | None = None
    end: float | None = None
    text: str
    speaker: str | None = None


class ParsedTranscript(BaseModel):
    format: TranscriptFormat
    timed: bool
    segments: list[TranscriptSegment] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)  # Italian, operator-facing


class _JsonSegment(BaseModel):
    model_config = ConfigDict(extra="ignore")
    start: float
    end: float
    text: str
    speaker: str | None = None


class _JsonTranscript(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    schema_: str | None = Field(default=None, alias="schema")
    language: str | None = None
    segments: list[_JsonSegment]


def normalize_text(text: str) -> str:
    return " ".join(text.split())


_SRT_TIME = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)
_TAG = re.compile(r"</?[a-zA-Z][^>]*>|\{\\[^}]*\}")


def _srt_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def parse_srt(text: str) -> ParsedTranscript:
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    segments: list[TranscriptSegment] = []
    warnings: list[str] = []
    for n, block in enumerate(re.split(r"\n\s*\n", text.strip()), start=1):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        idx = next((i for i, ln in enumerate(lines) if _SRT_TIME.search(ln)), None)
        if idx is None or idx > 1:
            warnings.append(f"SRT: blocco {n} senza tempi validi, ignorato")
            continue
        m = _SRT_TIME.search(lines[idx])
        assert m is not None
        start = _srt_seconds(*m.groups()[:4])
        end = _srt_seconds(*m.groups()[4:])
        body = normalize_text(_TAG.sub("", " ".join(lines[idx + 1 :])))
        segments.append(TranscriptSegment(start=start, end=end, text=body))
    if not segments and text.strip():
        raise DatasetError("File SRT senza sottotitoli leggibili")
    return ParsedTranscript(format=TranscriptFormat.SRT, timed=True, segments=segments, warnings=warnings)


def parse_json_transcript(text: str) -> ParsedTranscript:
    try:
        raw = json.loads(text.lstrip("﻿"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"JSON della trascrizione non valido: {exc}") from exc
    if isinstance(raw, list):
        raw = {"segments": raw}
    try:
        doc = _JsonTranscript.model_validate(raw)
    except ValidationError as exc:
        raise DatasetError(
            "La trascrizione JSON non rispetta lo schema "
            f"{TRANSCRIPT_SCHEMA} (segments: start, end, text): {exc.errors()[0]['msg']}"
        ) from exc
    warnings = []
    if doc.schema_ and doc.schema_ != TRANSCRIPT_SCHEMA:
        warnings.append(f"Schema JSON dichiarato '{doc.schema_}', atteso '{TRANSCRIPT_SCHEMA}'")
    segments = [
        TranscriptSegment(start=s.start, end=s.end, text=normalize_text(s.text), speaker=s.speaker)
        for s in doc.segments
    ]
    return ParsedTranscript(format=TranscriptFormat.JSON, timed=True, segments=segments, warnings=warnings)


def parse_plain_text(text: str) -> ParsedTranscript:
    body = normalize_text(text.lstrip("﻿"))
    return ParsedTranscript(
        format=TranscriptFormat.TEXT, timed=False, segments=[TranscriptSegment(text=body)]
    )


def detect_format(path: Path) -> TranscriptFormat:
    suffix = path.suffix.lower()
    if suffix == ".srt":
        return TranscriptFormat.SRT
    if suffix == ".json":
        return TranscriptFormat.JSON
    if suffix in (".txt", ".text", ""):
        return TranscriptFormat.TEXT
    raise DatasetError(f"Formato di trascrizione non riconosciuto: {path.name}")


def read_transcript(path: Path, fmt: TranscriptFormat | None = None) -> ParsedTranscript:
    fmt = fmt or detect_format(path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")
    if fmt == TranscriptFormat.SRT:
        return parse_srt(text)
    if fmt == TranscriptFormat.JSON:
        return parse_json_transcript(text)
    return parse_plain_text(text)


# --------------------------------------------------------------------------
# Alignment hook (the acoustic aligner lives elsewhere)
# --------------------------------------------------------------------------


class AlignedSpan(BaseModel):
    start: float
    end: float
    text: str
    score: float | None = None  # aligner-specific, higher = better


@runtime_checkable
class TranscriptAligner(Protocol):
    """Cuts an untimed transcription into timed spans (e.g. CTC forced alignment)."""

    name: str

    def align(self, audio: np.ndarray, sr: int, text: str) -> list[AlignedSpan]:
        ...


# --------------------------------------------------------------------------
# Dataset records
# --------------------------------------------------------------------------


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"  # blocks the item from profile and adaptation


class QualityWarning(BaseModel):
    code: str
    severity: Severity
    message: str  # Italian, operator-facing
    recording_id: str | None = None
    item_id: str | None = None


@runtime_checkable
class QualityCheck(Protocol):
    """Custom quality check (e.g. Italianized dialect). Register it with
    `SubjectDataset.register_check`; it runs on every item at each check."""

    def __call__(self, item: DatasetItem, recording: "RecordingEntry") -> Iterable[QualityWarning]:
        ...


class QualityConfig(BaseModel):
    min_segment_s: float = 0.5
    max_segment_s: float = 30.0  # Whisper's input window
    time_tolerance_s: float = 0.05


class SpeakerSpan(BaseModel):
    """A labelled segment of someone other than the subject."""

    recording_id: str
    start: float
    end: float
    speaker: str | None = None
    text: str = ""


class RecordingEntry(BaseModel):
    recording_id: str  # = evidence_id in the case
    original_name: str
    audio: ArtifactRef
    condition: RecordingCondition
    pair_id: str | None = None
    duration: float
    sample_rate: int
    audio_format: str  # container/subtype from the header, e.g. "WAV/PCM_16"
    transcript: ArtifactRef | None = None
    transcript_format: TranscriptFormat | None = None
    subject_speaker: str | None = None
    needs_alignment: bool = False
    aligned_by: str | None = None
    split: DatasetSplit | None = None


class SplitRecord(BaseModel):
    strategy: Literal["by_recording_group"] = "by_recording_group"
    holdout_fraction: float
    seed: int
    stratify: Literal["channel"] | None = "channel"
    grouping: str = "pair_id, otherwise recording_id"
    ordering: str = "sha256('<seed>:<group key>') within stratum"
    reset: bool = False
    assigned_groups: dict[str, DatasetSplit] = Field(default_factory=dict)


class _FolderIndexEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")
    audio: str
    transcript: str | None = None
    transcript_format: TranscriptFormat | None = None
    channel: Channel = Channel.UNKNOWN
    source: RecordingSource = RecordingSource.OTHER
    vocal_effort: VocalEffort = VocalEffort.UNKNOWN
    codec: str | None = None
    sample_rate: int | None = None
    notes: str | None = None
    pair_id: str | None = None
    subject_speaker: str | None = None


class _FolderIndex(BaseModel):
    recordings: list[_FolderIndexEntry]


# libsndfile subtype -> codec name. Lossless/PCM subtypes are deliberately absent.
_SUBTYPE_CODEC = {
    "ULAW": "g711-ulaw",
    "ALAW": "g711-alaw",
    "GSM610": "gsm",
    "OPUS": "opus",
    "VORBIS": "vorbis",
    "MPEG_LAYER_III": "mp3",
    "MPEG_LAYER_II": "mp2",
    "MPEG_LAYER_I": "mp1",
    "IMA_ADPCM": "adpcm-ima",
    "MS_ADPCM": "adpcm-ms",
    "VOX_ADPCM": "adpcm-vox",
    "G721_32": "g721",
    "G723_24": "g723",
    "G723_40": "g723",
}


def probe_audio(path: Path) -> tuple[float, int, str, str | None]:
    """(duration s, sample rate, "FORMAT/SUBTYPE", detected codec or None)."""
    try:
        info = sf.info(str(path))
        return float(info.duration), int(info.samplerate), f"{info.format}/{info.subtype}", _SUBTYPE_CODEC.get(info.subtype)
    except Exception:
        pass
    try:
        import librosa

        duration = float(librosa.get_duration(path=str(path)))
        sr = int(librosa.get_samplerate(str(path)))
        return duration, sr, f"{path.suffix.lstrip('.').upper() or 'UNKNOWN'}/unknown", None
    except Exception as exc:
        raise DatasetError(f"File audio non leggibile: {path.name} ({exc})") from exc


# Unintelligible / uncertain passages, as commonly marked in Italian transcripts.
_BRACKETED = re.compile(r"[\[\(\{<]([^\]\)\}>]{0,80})[\]\)\}>]")
_UNINTELLIGIBLE_WORDS = re.compile(
    r"incompr|\binc\b|\binint|inaud|inudib|illegg|non\s+(si\s+)?(capisce|comprensibil|udibil|chiar)"
    r"|sovrapp|\?|\.\.\.|…|\bxx+\b",
    re.IGNORECASE,
)
_BARE_MARKERS = re.compile(r"\?{2,}|\*{3,}|\bx{3,}\b|…{2,}|\.{4,}", re.IGNORECASE)


def find_unintelligible_markers(text: str) -> list[str]:
    found = [m.group(0) for m in _BRACKETED.finditer(text) if _UNINTELLIGIBLE_WORDS.search(m.group(1))]
    stripped = _BRACKETED.sub(" ", text)
    found += [m.group(0) for m in _BARE_MARKERS.finditer(stripped)]
    return found


def find_annotations(text: str) -> list[str]:
    """Bracketed annotations that are not unintelligible markers (e.g. "[ride]")."""
    return [m.group(0) for m in _BRACKETED.finditer(text) if not _UNINTELLIGIBLE_WORDS.search(m.group(1))]


def _group_key(rec: RecordingEntry) -> str:
    return f"pair:{rec.pair_id}" if rec.pair_id else f"rec:{rec.recording_id}"


def _hash_order(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def holdout_target(n_groups: int, fraction: float) -> int:
    if n_groups < 2 or fraction <= 0:
        return 0
    return min(max(int(math.floor(fraction * n_groups + 0.5)), 1), n_groups - 1)


# --------------------------------------------------------------------------
# The dataset
# --------------------------------------------------------------------------


class SubjectDataset(BaseModel):
    """Working state of the subject's material (saved as dataset.json in the
    subject folder). Hashed snapshots are written with `write_manifest`."""

    schema_version: str = DATASET_SCHEMA
    subject_id: str
    recordings: list[RecordingEntry] = Field(default_factory=list)
    items: list[DatasetItem] = Field(default_factory=list)
    other_speaker_spans: list[SpeakerSpan] = Field(default_factory=list)
    # Warnings produced at import/alignment time (cannot be recomputed later).
    recorded_warnings: list[QualityWarning] = Field(default_factory=list)
    # Latest full report: recorded warnings + checks (recomputed by run_quality_checks).
    warnings: list[QualityWarning] = Field(default_factory=list)
    quality_config: QualityConfig = Field(default_factory=QualityConfig)
    split_history: list[SplitRecord] = Field(default_factory=list)

    _extra_checks: list[QualityCheck] = PrivateAttr(default_factory=list)

    # -- persistence -------------------------------------------------------

    @classmethod
    def path_for(cls, case: Case, subject_id: str) -> Path:
        return subject_dir(case, subject_id) / "dataset.json"

    @classmethod
    def load_or_create(cls, case: Case, subject_id: str) -> "SubjectDataset":
        path = cls.path_for(case, subject_id)
        if path.exists():
            return cls.model_validate_json(path.read_text(encoding="utf-8"))
        return cls(subject_id=check_id(subject_id, "identificativo del soggetto"))

    def save(self, case: Case) -> Path:
        path = self.path_for(case, self.subject_id)
        write_json_atomic(path, self.model_dump(mode="json"))
        return path

    # -- lookup ------------------------------------------------------------

    def recording(self, recording_id: str) -> RecordingEntry:
        for rec in self.recordings:
            if rec.recording_id == recording_id:
                return rec
        raise DatasetError(f"Registrazione sconosciuta: {recording_id}")

    def item(self, item_id: str) -> DatasetItem:
        for it in self.items:
            if it.item_id == item_id:
                return it
        raise DatasetError(f"Segmento sconosciuto: {item_id}")

    def items_of(self, recording_id: str) -> list[DatasetItem]:
        return [it for it in self.items if it.recording_id == recording_id]

    def pending_alignment_ids(self) -> set[str]:
        pending = {r.recording_id for r in self.recordings if r.needs_alignment}
        return {it.item_id for it in self.items if it.recording_id in pending}

    # -- import ------------------------------------------------------------

    def import_recording(
        self,
        case: Case,
        audio_path: Path,
        condition: RecordingCondition | None = None,
        *,
        pair_id: str | None = None,
        transcript_path: Path | None = None,
        transcript_format: TranscriptFormat | None = None,
        subject_speaker: str | None = None,
    ) -> RecordingEntry:
        """Import one recording (read-only, hashed) and its transcription, if any.

        Re-importing the same audio bytes returns the existing entry unchanged.
        """
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise DatasetError(f"File audio inesistente: {audio_path}")
        if pair_id is not None:
            check_id(pair_id, "identificativo di coppia")
        digest = sha256_file(audio_path)
        for rec in self.recordings:
            if rec.audio.sha256 == digest:
                self._record(
                    "already_imported", Severity.INFO,
                    f"{audio_path.name}: registrazione già importata come {rec.recording_id}, ignorata",
                    recording_id=rec.recording_id,
                )
                return rec

        duration, sample_rate, audio_format, codec = probe_audio(audio_path)
        cond = (condition or RecordingCondition()).model_copy()
        if cond.codec is None and codec is not None:
            cond.codec = codec
        if cond.sample_rate is None:
            cond.sample_rate = sample_rate

        ev = case.import_evidence(audio_path, cond, pair_id=pair_id, role=EvidenceRole.REFERENCE)
        rec = RecordingEntry(
            recording_id=ev.evidence_id,
            original_name=audio_path.name,
            audio=ev.artifact,
            condition=cond,
            pair_id=pair_id,
            duration=duration,
            sample_rate=sample_rate,
            audio_format=audio_format,
            subject_speaker=subject_speaker,
        )
        self.recordings.append(rec)
        if transcript_path is not None:
            self._attach_transcript(case, rec, Path(transcript_path), transcript_format, subject_speaker)
        self.run_quality_checks()
        return rec

    def _attach_transcript(
        self,
        case: Case,
        rec: RecordingEntry,
        path: Path,
        fmt: TranscriptFormat | None,
        subject_speaker: str | None,
    ) -> None:
        if not path.is_file():
            raise DatasetError(f"File di trascrizione inesistente: {path}")
        parsed = read_transcript(path, fmt)
        # Copy the transcription into the subject folder, hashed.
        dest_dir = subject_dir(case, self.subject_id) / "transcripts"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{rec.recording_id}{path.suffix.lower() or '.txt'}"
        dest.write_bytes(path.read_bytes())
        rec.transcript = ArtifactRef.from_file(dest, kind="subject_transcript", root=case.root)
        rec.transcript_format = parsed.format
        for msg in parsed.warnings:
            self._record("transcript_parse", Severity.WARNING, f"{path.name}: {msg}", recording_id=rec.recording_id)

        rid = rec.recording_id
        if not parsed.timed:
            rec.needs_alignment = True
            self.items.append(
                DatasetItem(
                    item_id=f"{rid}-t0000", recording_id=rid, audio=rec.audio,
                    start=0.0, end=rec.duration, text=parsed.segments[0].text,
                    condition=rec.condition, split=rec.split,
                )
            )
            return

        labels = {s.speaker for s in parsed.segments if s.speaker}
        if subject_speaker is None and len(labels) > 1:
            self._record(
                "multiple_speakers", Severity.WARNING,
                f"{path.name}: trascrizione con più parlanti ({', '.join(sorted(labels))}) ma etichetta "
                "del soggetto non indicata: tutti i segmenti sono attribuiti al soggetto",
                recording_id=rid,
            )
        if subject_speaker is not None and subject_speaker not in labels:
            self._record(
                "subject_label_missing", Severity.WARNING,
                f"{path.name}: etichetta del soggetto '{subject_speaker}' assente nella trascrizione",
                recording_id=rid,
            )
        k = 0
        for seg in parsed.segments:
            assert seg.start is not None and seg.end is not None
            if subject_speaker is not None and seg.speaker is not None and seg.speaker != subject_speaker:
                self.other_speaker_spans.append(
                    SpeakerSpan(recording_id=rid, start=seg.start, end=seg.end, speaker=seg.speaker, text=seg.text)
                )
                continue
            self.items.append(
                DatasetItem(
                    item_id=f"{rid}-s{k:04d}", recording_id=rid, audio=rec.audio,
                    start=seg.start, end=seg.end, text=seg.text,
                    condition=rec.condition, split=rec.split,
                )
            )
            k += 1

    def import_folder(
        self,
        case: Case,
        folder: Path,
        condition: RecordingCondition | None = None,
        *,
        subject_speaker: str | None = None,
    ) -> list[RecordingEntry]:
        """Import a folder of recordings (see module docstring for the optional index)."""
        folder = Path(folder)
        if not folder.is_dir():
            raise DatasetError(f"Cartella inesistente: {folder}")
        index_path = folder / FOLDER_INDEX
        jobs: list[tuple[Path, RecordingCondition, str | None, Path | None, TranscriptFormat | None, str | None]] = []
        if index_path.exists():
            try:
                index = _FolderIndex.model_validate_json(index_path.read_text(encoding="utf-8"))
            except ValidationError as exc:
                raise DatasetError(f"{FOLDER_INDEX} non valido: {exc.errors()[0]['msg']}") from exc
            for e in index.recordings:
                cond = RecordingCondition(
                    channel=e.channel, vocal_effort=e.vocal_effort, source=e.source,
                    codec=e.codec, sample_rate=e.sample_rate, notes=e.notes,
                )
                tr = folder / e.transcript if e.transcript else None
                jobs.append((folder / e.audio, cond, e.pair_id, tr, e.transcript_format, e.subject_speaker or subject_speaker))
        else:
            if condition is None:
                self._record(
                    "condition_missing", Severity.WARNING,
                    f"{folder.name}: nessuna condizione indicata e nessun {FOLDER_INDEX}: "
                    "registrazioni importate con condizione sconosciuta",
                )
            for audio in sorted(p for p in folder.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS):
                tr = next(
                    (audio.with_suffix(s) for s in (".json", ".srt", ".txt") if audio.with_suffix(s).is_file()),
                    None,
                )
                jobs.append((audio, condition or RecordingCondition(), None, tr, None, subject_speaker))

        out = []
        for audio, cond, pair_id, tr, fmt, label in jobs:
            try:
                out.append(
                    self.import_recording(
                        case, audio, cond, pair_id=pair_id, transcript_path=tr,
                        transcript_format=fmt, subject_speaker=label,
                    )
                )
            except DatasetError as exc:
                self._record("import_failed", Severity.ERROR, f"{audio.name}: importazione non riuscita: {exc}")
        self.run_quality_checks()
        return out

    # -- alignment ---------------------------------------------------------

    def align_pending(
        self,
        case: Case,
        aligner: TranscriptAligner,
        *,
        min_score: float | None = None,
        sr: int = 16000,
    ) -> list[DatasetItem]:
        """Cut untimed transcriptions into timed items with an external aligner.

        Spans whose score is below `min_score` are flagged as possibly misaligned.
        New items are always unverified: verification must be repeated on them.
        """
        created: list[DatasetItem] = []
        for rec in self.recordings:
            if not rec.needs_alignment:
                continue
            rid = rec.recording_id
            pending = self.items_of(rid)
            text = normalize_text(" ".join(it.text for it in pending))
            if not text:
                continue
            if not rec.audio.verify(case.root):
                self._record(
                    "alignment_failed", Severity.ERROR,
                    f"{rec.original_name}: audio mancante o modificato (SHA-256), allineamento non eseguito",
                    recording_id=rid,
                )
                continue
            audio = load_audio(rec.audio.resolve(case.root), sr)
            spans = aligner.align(audio, sr, text)
            if not spans:
                reasons = (getattr(aligner, "last_report", None) or {}).get("warnings") or []
                detail = f": {'; '.join(reasons)}" if reasons else ""
                self._record(
                    "alignment_failed", Severity.ERROR,
                    f"{rec.original_name}: allineamento non riuscito ({aligner.name}){detail}", recording_id=rid,
                )
                continue
            if any(it.verified for it in pending):
                self._record(
                    "verification_reset", Severity.INFO,
                    f"{rec.original_name}: trascrizione riallineata, la verifica dei segmenti va ripetuta",
                    recording_id=rid,
                )
            self.items = [it for it in self.items if it.recording_id != rid]
            for k, span in enumerate(spans):
                item = DatasetItem(
                    item_id=f"{rid}-a{k:04d}", recording_id=rid, audio=rec.audio,
                    start=float(span.start), end=float(span.end), text=normalize_text(span.text),
                    condition=rec.condition, split=rec.split,
                )
                self.items.append(item)
                created.append(item)
                if min_score is not None and span.score is not None and span.score < min_score:
                    self._record(
                        "possible_misalignment", Severity.WARNING,
                        f"Segmento {item.item_id}: possibile allineamento errato "
                        f"(punteggio {span.score:.3f} < {min_score:.3f}, {aligner.name})",
                        recording_id=rid, item_id=item.item_id,
                    )
            rec.needs_alignment = False
            rec.aligned_by = aligner.name
        self.run_quality_checks()
        return created

    # -- quality -----------------------------------------------------------

    def register_check(self, check: QualityCheck) -> None:
        """Add a custom item check (not persisted: register it again after loading)."""
        self._extra_checks.append(check)

    def _record(self, code: str, severity: Severity, message: str, *, recording_id: str | None = None,
                item_id: str | None = None) -> None:
        self.recorded_warnings.append(
            QualityWarning(code=code, severity=severity, message=message, recording_id=recording_id, item_id=item_id)
        )

    def run_quality_checks(self) -> list[QualityWarning]:
        cfg = self.quality_config
        tol = cfg.time_tolerance_s
        out: list[QualityWarning] = list(self.recorded_warnings)

        def w(code: str, sev: Severity, msg: str, rid: str | None = None, iid: str | None = None) -> None:
            out.append(QualityWarning(code=code, severity=sev, message=msg, recording_id=rid, item_id=iid))

        for rec in self.recordings:
            rid = rec.recording_id
            name = rec.original_name
            if rec.transcript is None:
                w("no_transcript", Severity.INFO,
                  f"{name}: registrazione senza trascrizione, non entra nel dataset finché non viene trascritta", rid)
            if rec.condition.channel == Channel.UNKNOWN:
                w("condition_unknown", Severity.WARNING,
                  f"{name}: canale non indicato, il profilo per condizione sarà poco informativo", rid)
            if rec.needs_alignment:
                w("needs_alignment", Severity.INFO,
                  f"{name}: trascrizione senza tempi, da allineare all'audio prima dell'uso", rid)
            items = sorted(self.items_of(rid), key=lambda it: (it.start, it.end, it.item_id))
            prev: DatasetItem | None = None
            for it in items:
                iid = it.item_id
                dur = it.end - it.start
                if not it.text.strip():
                    w("empty_text", Severity.ERROR, f"Segmento {iid}: testo vuoto", rid, iid)
                markers = find_unintelligible_markers(it.text)
                if markers:
                    w("unintelligible_marker", Severity.WARNING,
                      f"Segmento {iid}: contiene parti segnate come incomprensibili ({', '.join(markers)})", rid, iid)
                notes = find_annotations(it.text)
                if notes:
                    w("annotation", Severity.INFO,
                      f"Segmento {iid}: annotazioni tra parentesi ({', '.join(notes)}), "
                      "da escludere dal testo per l'adattamento", rid, iid)
                if dur <= 0:
                    w("invalid_times", Severity.ERROR,
                      f"Segmento {iid}: fine ({it.end:.2f} s) non successiva all'inizio ({it.start:.2f} s)", rid, iid)
                elif it.start < -tol or it.end > rec.duration + tol:
                    w("outside_audio", Severity.ERROR,
                      f"Segmento {iid}: {it.start:.2f}-{it.end:.2f} s fuori dalla durata dell'audio "
                      f"({rec.duration:.2f} s)", rid, iid)
                elif not rec.needs_alignment:
                    if dur < cfg.min_segment_s:
                        w("too_short", Severity.WARNING,
                          f"Segmento {iid}: molto breve ({dur:.2f} s < {cfg.min_segment_s:.2f} s)", rid, iid)
                    if dur > cfg.max_segment_s:
                        w("too_long", Severity.WARNING,
                          f"Segmento {iid}: molto lungo ({dur:.1f} s > {cfg.max_segment_s:.1f} s), "
                          "da suddividere per l'adattamento", rid, iid)
                if prev is not None and it.start < prev.end - tol:
                    w("overlap", Severity.WARNING,
                      f"Segmento {iid}: si sovrappone a {prev.item_id}", rid, iid)
                for check in self._extra_checks:
                    out.extend(check(it, rec))
                prev = it

        if self.split_history:
            channels = sorted({r.condition.channel.value for r in self.recordings})
            for ch in channels:
                recs = [r for r in self.recordings if r.condition.channel.value == ch]
                if not any(r.split == DatasetSplit.HOLDOUT for r in recs):
                    w("no_holdout_for_channel", Severity.INFO,
                      f"Canale '{ch}': nessuna registrazione nella quota di autovalutazione, "
                      "le prestazioni su questo canale non saranno misurate")
        self.warnings = out
        return out

    def blocked_item_ids(self) -> set[str]:
        return {w.item_id for w in self.warnings if w.severity == Severity.ERROR and w.item_id}

    # -- split -------------------------------------------------------------

    def split_by_recording(
        self,
        holdout_fraction: float = 0.2,
        seed: int = 0,
        *,
        stratify_by_channel: bool = True,
        reset: bool = False,
    ) -> dict[str, DatasetSplit]:
        """Assign every recording (and its items) to train or holdout. See module docstring."""
        if not 0.0 <= holdout_fraction < 1.0:
            raise ValueError("holdout_fraction must be in [0, 1)")
        if reset:
            for rec in self.recordings:
                rec.split = None
        groups: dict[str, list[RecordingEntry]] = {}
        for rec in self.recordings:
            groups.setdefault(_group_key(rec), []).append(rec)

        group_split: dict[str, DatasetSplit] = {}
        for key, recs in groups.items():
            existing = {r.split for r in recs if r.split is not None}
            if len(existing) > 1:
                raise DatasetError(f"Le registrazioni del gruppo {key} sono in divisioni diverse")
            if existing:
                group_split[key] = existing.pop()

        target = holdout_target(len(groups), holdout_fraction)
        already = sum(1 for s in group_split.values() if s == DatasetSplit.HOLDOUT)
        new_keys = sorted(k for k in groups if k not in group_split)
        need = max(0, min(target - already, len(new_keys)))

        def channel_of(key: str) -> str:
            return sorted(r.condition.channel.value for r in groups[key])[0]

        strata: dict[str, list[str]] = {}
        for key in new_keys:
            strata.setdefault(channel_of(key) if stratify_by_channel else "*", []).append(key)
        # Largest-remainder allocation of `need` across strata (ties: stratum name).
        exact = {s: need * len(keys) / len(new_keys) for s, keys in strata.items()} if new_keys else {}
        quota = {s: int(math.floor(x)) for s, x in exact.items()}
        rest = need - sum(quota.values())
        for s in sorted(exact, key=lambda s: (-(exact[s] - quota[s]), s))[:rest]:
            quota[s] += 1

        chosen: set[str] = set()
        for s in sorted(strata):
            ordered = sorted(strata[s], key=lambda k: _hash_order(seed, k))
            chosen.update(ordered[: quota[s]])
        assigned = {k: (DatasetSplit.HOLDOUT if k in chosen else DatasetSplit.TRAIN) for k in new_keys}
        group_split.update(assigned)

        for key, recs in groups.items():
            for rec in recs:
                rec.split = group_split[key]
        by_rec = {r.recording_id: r.split for r in self.recordings}
        for it in self.items:
            it.split = by_rec.get(it.recording_id)

        if len(groups) < 2 and holdout_fraction > 0:
            self._record(
                "holdout_impossible", Severity.WARNING,
                "Meno di due registrazioni indipendenti: nessuna quota di autovalutazione possibile",
            )
        self.split_history.append(
            SplitRecord(
                holdout_fraction=holdout_fraction, seed=seed,
                stratify="channel" if stratify_by_channel else None,
                reset=reset, assigned_groups=dict(sorted(assigned.items())),
            )
        )
        self.run_quality_checks()
        return {rid: s for rid, s in by_rec.items() if s is not None}

    @property
    def is_split(self) -> bool:
        return bool(self.recordings) and all(r.split is not None for r in self.recordings)

    def holdout_recordings(self) -> list[str]:
        return sorted(r.recording_id for r in self.recordings if r.split == DatasetSplit.HOLDOUT)

    # -- selection ---------------------------------------------------------

    def select_items(
        self,
        *,
        splits: Iterable[DatasetSplit] | None = (DatasetSplit.TRAIN,),
        verified_only: bool = True,
        include_unaligned: bool = False,
        exclude_blocked: bool = True,
    ) -> list[DatasetItem]:
        """Items usable for a purpose, sorted by item_id. `splits=None` ignores the split."""
        allowed = set(splits) if splits is not None else None
        pending = self.pending_alignment_ids()
        blocked = self.blocked_item_ids() if exclude_blocked else set()
        out = []
        for it in self.items:
            if allowed is not None and it.split not in allowed:
                continue
            if verified_only and it.verified is not True:
                continue
            if not include_unaligned and it.item_id in pending:
                continue
            if it.item_id in blocked:
                continue
            out.append(it)
        return sorted(out, key=lambda it: it.item_id)

    # -- reporting ---------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        def seconds(items: Iterable[DatasetItem]) -> float:
            return round(sum(max(0.0, it.end - it.start) for it in items), 3)

        by_condition: dict[str, dict[str, Any]] = {}
        for it in self.items:
            d = by_condition.setdefault(
                it.condition.key(), {"items": 0, "seconds": 0.0, "verified": 0, "train": 0, "holdout": 0}
            )
            d["items"] += 1
            d["seconds"] = round(d["seconds"] + max(0.0, it.end - it.start), 3)
            d["verified"] += int(it.verified)
            if it.split is not None:
                d[it.split.value] += 1
        severities: dict[str, int] = {}
        codes: dict[str, int] = {}
        for w in self.warnings:
            severities[w.severity.value] = severities.get(w.severity.value, 0) + 1
            codes[w.code] = codes.get(w.code, 0) + 1
        return {
            "recordings": len(self.recordings),
            "items": len(self.items),
            "seconds": seconds(self.items),
            "verified_items": sum(int(it.verified) for it in self.items),
            "pending_alignment_recordings": sum(int(r.needs_alignment) for r in self.recordings),
            "split": {
                s.value: {
                    "recordings": sum(1 for r in self.recordings if r.split == s),
                    "items": sum(1 for it in self.items if it.split == s),
                    "seconds": seconds(it for it in self.items if it.split == s),
                }
                for s in DatasetSplit
            },
            "by_condition": dict(sorted(by_condition.items())),
            "warnings_by_severity": dict(sorted(severities.items())),
            "warnings_by_code": dict(sorted(codes.items())),
        }

    def manifest_dict(self) -> dict[str, Any]:
        """Deterministic snapshot: sorted recordings and items, quality report, split history."""
        return {
            "schema": MANIFEST_SCHEMA,
            "subject_id": self.subject_id,
            "quality_config": self.quality_config.model_dump(mode="json"),
            "split_history": [s.model_dump(mode="json") for s in self.split_history],
            "recordings": [
                r.model_dump(mode="json") for r in sorted(self.recordings, key=lambda r: r.recording_id)
            ],
            "items": [it.model_dump(mode="json") for it in sorted(self.items, key=lambda it: it.item_id)],
            "other_speaker_spans": [
                s.model_dump(mode="json")
                for s in sorted(self.other_speaker_spans, key=lambda s: (s.recording_id, s.start, s.end))
            ],
            "warnings": [w.model_dump(mode="json") for w in self.warnings],
            "summary": self.summary(),
        }

    def write_manifest(self, case: Case) -> ArtifactRef:
        """Write the hashed manifest (content-addressed) and return its ArtifactRef."""
        self.run_quality_checks()
        return write_content_addressed(
            case, subject_dir(case, self.subject_id) / "manifests", "dataset",
            self.manifest_dict(), kind="subject_dataset_manifest",
        )
