# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Token-to-word grouping, word confidence and CTC word times (shared by the
Cohere / Qwen3 / Nemotron backends)."""
import numpy as np
import pytest

from akoe.transcribe.ctc_aligner import CTCTranscriptAligner
from akoe.transcribe.word_timing import WordTimer, build_words, fill_missing_times, token_word_groups
from tests.test_ctc_aligner import SR, crafted_scorer


def sp_decode(vocab):
    """SentencePiece-like decoding: '▁' becomes a space, leading space stripped."""
    return lambda ids: "".join(vocab[i] for i in ids).replace("▁", " ").strip()


def test_groups_sentencepiece_tokens_into_words():
    vocab = ["▁Buon", "giorno", "▁a", "▁tutti", ".", "▁", "città"]
    groups = token_word_groups(sp_decode(vocab), [0, 1, 2, 3, 4])
    assert groups == [("Buongiorno", [0, 1]), ("a", [2]), ("tutti.", [3, 4])]
    # a lone "▁" token goes with the next visible token
    assert token_word_groups(sp_decode(vocab), [2, 5, 6]) == [("a", [0]), ("città", [1, 2])]


def test_groups_byte_level_tokens_with_split_utf8_characters():
    pieces = [b" citt", b"\xc3", b"\xa0", b" bella"]

    def decode(ids):
        return b"".join(pieces[i] for i in ids).decode("utf-8", errors="replace").strip()

    assert token_word_groups(decode, [0, 1, 2, 3]) == [("città", [0, 1, 2]), ("bella", [3])]


def test_build_words_confidence_and_times():
    groups = [("ciao", [0]), ("mondo", [1, 2]), ("!", [])]
    lps = [np.log(0.9), np.log(0.5), np.log(0.8)]
    words = build_words(groups, lps, [(0.1, 0.4), (0.5, 0.9), None], offset=10.0, seg_dur=1.0, origin="m/generic")
    assert [w.text for w in words] == ["ciao", "mondo", "!"]
    assert words[0].confidence == pytest.approx(0.9) and words[1].confidence == pytest.approx(0.4)
    assert words[2].confidence is None  # no tokens: no invented confidence
    assert (words[0].start, words[0].end) == pytest.approx((10.1, 10.4))
    assert (words[2].start, words[2].end) == pytest.approx((10.9, 11.0))  # gap after the last timed word
    untimed = build_words(groups, lps, None, offset=0.0, seg_dur=2.0, origin="m")
    assert all((w.start, w.end) == (0.0, 2.0) for w in untimed)


def test_fill_missing_times_uses_the_gap_between_neighbours():
    assert fill_missing_times([None, (0.5, 0.7), None, (1.0, 5.0)], 2.0) == [
        (0.0, 0.5), (0.5, 0.7), (0.7, 1.0), (1.0, 2.0)]


def test_ctc_aligner_gives_word_times_for_asr_words():
    al = CTCTranscriptAligner(crafted_scorer())
    assert isinstance(al, WordTimer)
    audio = np.zeros(int(4.0 * SR), dtype=np.float32)
    times = al.word_times(audio, SR, ["Ciao", "mondo.", "Vado", "a", "casa", "123"])
    assert times[-1] is None  # digits: nothing the CTC model can score
    expected = [(0.5, 0.8), (0.9, 1.3), (2.5, 2.8), (2.9, 2.95), (3.0, 3.4)]
    for (s, e), (es, ee) in zip(times[:-1], expected):
        assert s == pytest.approx(es, abs=0.05) and e == pytest.approx(ee, abs=0.05)
    assert any("numeri" in w for w in al.last_report["warnings"])


def test_ctc_word_times_when_alignment_is_impossible():
    al = CTCTranscriptAligner(crafted_scorer(plan=[], seconds=0.1))
    times = al.word_times(np.zeros(1600, dtype=np.float32), SR, ["precipitevolissimevolmente"] * 3)
    assert times == [None, None, None]
    assert al.last_report["warnings"]
