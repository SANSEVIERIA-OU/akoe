"""CTC aligner for untimed transcriptions (issue #7), on crafted emissions."""
import numpy as np
import pytest

from forensicaudio.case import Case
from forensicaudio.config import Channel, RecordingSource
from forensicaudio.models import RecordingCondition
from forensicaudio.speaker.dataset import SubjectDataset, TranscriptAligner
from forensicaudio.transcribe.ctc_aligner import CTCTranscriptAligner
from forensicaudio.transcribe.ctc_scorer import CTCForcedAlignmentScorer
from tests.fakes import uniform_ctc_scorer
from tests.synth_voice import write_wav

LETTERS = "abcdefghijklmnopqrstuvwxyzàèéìòù'"
VOCAB = {"<pad>": 0, "|": 1, **{c: i + 2 for i, c in enumerate(LETTERS)}}
SR = 16000
SPF = 0.02
# word, start s, end s: two sentences separated by a 1.2 s pause
PLAN = [("ciao", 0.5, 0.8), ("mondo", 0.9, 1.3), ("vado", 2.5, 2.8), ("a", 2.9, 2.95), ("casa", 3.0, 3.4)]


def crafted_scorer(plan=PLAN, seconds=4.0):
    n = int(seconds / SPF)
    v = len(VOCAB)
    low = np.log(0.01 / (v - 1))
    lp = np.full((n, v), low)
    lp[:, 0] = np.log(0.99)

    def put(f, c):
        lp[f, :] = low
        lp[f, VOCAB[c]] = np.log(0.99)

    prev = None
    for word, t0, t1 in plan:
        f0, f1 = int(round(t0 / SPF)), int(round(t1 / SPF))
        if prev is not None:
            put((prev + f0) // 2, "|")
        for f, c in zip(np.linspace(f0, f1 - 1, len(word)).round().astype(int), word):
            put(int(f), c)
        prev = f1
    return CTCForcedAlignmentScorer.from_emission_fn(lambda a: lp.astype(np.float32), VOCAB, name="crafted")


def test_aligner_implements_the_dataset_hook():
    assert isinstance(CTCTranscriptAligner(uniform_ctc_scorer()), TranscriptAligner)


def test_spans_follow_sentences_and_pauses_and_keep_the_original_text():
    al = CTCTranscriptAligner(crafted_scorer())
    audio = np.zeros(int(4.0 * SR), dtype=np.float32)
    spans = al.align(audio, SR, "Ciao mondo. Vado a casa 123")
    assert [s.text for s in spans] == ["Ciao mondo.", "Vado a casa 123"]
    assert spans[0].start == pytest.approx(0.5 - al.pad_s, abs=0.05)
    assert spans[0].end == pytest.approx(1.3 + al.pad_s, abs=0.05)
    assert spans[1].start == pytest.approx(2.5 - al.pad_s, abs=0.05)
    assert spans[1].end == pytest.approx(3.4 + al.pad_s, abs=0.05)
    assert all(s.score is not None and s.score > -0.1 for s in spans)
    assert spans[0].end <= spans[1].start
    assert any("numeri" in w for w in al.last_report["warnings"])  # digits cannot be aligned


def test_wrong_text_scores_low():
    al = CTCTranscriptAligner(crafted_scorer())
    spans = al.align(np.zeros(int(4.0 * SR), dtype=np.float32), SR, "pippo pluto paperino")
    assert spans and all(s.score < al.min_score for s in spans)
    assert any("scadente" in w for w in al.last_report["warnings"])


def test_long_audio_emissions_are_windowed_on_the_wav2vec2_grid():
    al = CTCTranscriptAligner(uniform_ctc_scorer(), window_s=1.0, context_s=0.2)
    n = int(3.3 * SR)
    lp, spf = al._emissions(np.zeros(n, dtype=np.float32))
    assert lp.shape[0] == (n - 400) // 320 + 1 and spf == pytest.approx(0.02)
    spans = al.align(np.zeros(n, dtype=np.float32), SR, "uno due tre. quattro cinque")
    assert spans and spans[-1].end <= 3.3 + 1e-6


def test_too_long_is_refused_with_a_reason():
    al = CTCTranscriptAligner(uniform_ctc_scorer(), max_trellis_bytes=100)
    assert al.align(np.zeros(SR, dtype=np.float32), SR, "una frase") == []
    assert "troppo lunga" in al.last_report["warnings"][0]


def test_plain_text_transcripts_become_timed_items_and_bad_ones_are_flagged(tmp_path):
    case = Case.create(tmp_path / "case", "c1")
    ds = SubjectDataset.load_or_create(case, "s1")
    src = tmp_path / "src"
    src.mkdir()
    cond = RecordingCondition(channel=Channel.CAR, source=RecordingSource.LOCAL)
    good = write_wav(src / "good.wav", np.zeros(int(4 * SR), dtype=np.float32))
    (src / "good.txt").write_text("Ciao mondo. Vado a casa")
    bad = write_wav(src / "bad.wav", np.full(int(4 * SR), 1e-3, dtype=np.float32))
    (src / "bad.txt").write_text("pippo pluto paperino")
    ds.import_recording(case, good, cond, transcript_path=src / "good.txt")
    ds.import_recording(case, bad, cond, transcript_path=src / "bad.txt")
    assert ds.summary()["pending_alignment_recordings"] == 2
    al = CTCTranscriptAligner(crafted_scorer())
    created = ds.align_pending(case, al, min_score=al.min_score)
    assert ds.summary()["pending_alignment_recordings"] == 0
    good_items = [it for it in created if it.recording_id == ds.recordings[0].recording_id]
    assert [it.text for it in good_items] == ["Ciao mondo.", "Vado a casa"]
    assert all(not it.verified for it in created)
    flagged = {w.item_id for w in ds.warnings if w.code == "possible_misalignment"}
    bad_ids = {it.item_id for it in created if it.recording_id == ds.recordings[1].recording_id}
    assert bad_ids and bad_ids <= flagged and not (flagged & {it.item_id for it in good_items})
    assert all(r.aligned_by == al.name for r in ds.recordings)
