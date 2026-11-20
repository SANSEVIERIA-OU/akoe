# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Small public speech clips for the slow tests (downloaded from the HF hub at
pinned revisions; never committed). Only used by tests marked `slow`."""
from __future__ import annotations

import numpy as np

IT_REPO = "speechbrain/asr-wav2vec2-commonvoice-it"  # apache-2.0; clip from Common Voice
IT_REVISION = "70e4086640333143a4007b4ea29ec5d8affaa79e"
IT_TEXT = "Buongiorno a tutti e benvenuti a bordo"
EN_REPO = "speechbrain/spkrec-ecapa-voxceleb"
EN_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"


def _read(repo: str, filename: str, revision: str) -> tuple[np.ndarray, int]:
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, filename, revision=revision)
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def italian_example() -> tuple[np.ndarray, int, str]:
    """~4.2 s of read Italian speech and its text."""
    audio, sr = _read(IT_REPO, "example-it.wav", IT_REVISION)
    return audio, sr, IT_TEXT


def english_examples() -> list[tuple[np.ndarray, int]]:
    """Two short English clips of speakers other than the Italian one."""
    return [_read(EN_REPO, "example1.wav", EN_REVISION), _read(EN_REPO, "example2.flac", EN_REVISION)]
