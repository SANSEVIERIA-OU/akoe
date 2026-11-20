# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Real neural enhancers (issue #59) and neural VADs (issue #60) on a real
Italian clip mixed with synthetic car noise. Slow: downloads the pinned
weights on first use (UniPASE ~2.2 GB, MossFormer2 ~220 MB, the others a
few MB) and runs them on CPU. Run with ``-m slow``, ideally under
``flock /tmp/akoe-heavy.lock`` on shared machines."""
import numpy as np
import pytest

from akoe.stages import neural_clean as nc
from akoe.stages import vad
from tests import m6_synth
from tests.real_audio import italian_example

pytestmark = pytest.mark.slow
SR = 16000


def _si_sdr(ref, est):
    a = np.dot(est, ref) / np.dot(ref, ref)
    t = a * ref
    return 10 * np.log10(np.sum(t ** 2) / np.sum((est - t) ** 2))


@pytest.fixture(scope="module")
def mixture():
    from scipy.signal import resample_poly

    audio, sr, _ = italian_example()
    sp = resample_poly(audio.astype(np.float64), SR, sr) if sr != SR else audio.astype(np.float64)
    sp = sp / np.max(np.abs(sp)) * 0.3
    clean = np.concatenate([np.zeros(SR), sp, np.zeros(SR)])
    dur = len(clean) / SR + 0.01
    noise = (m6_synth.engine_noise(dur, seed=2, level=0.1)[0] + m6_synth.colored_noise(dur, seed=3, level=0.05))
    noise = noise[: len(clean)]
    g = np.sqrt(np.mean(sp ** 2) / np.mean(noise ** 2) / 10 ** (5.0 / 10))
    return clean, clean + g * noise


@pytest.mark.parametrize("name,min_gain_db", [("gtcrn", 3.0), ("deepfilternet3", 3.0),
                                               ("mossformer2_se_48k", 3.0)])
def test_mask_based_enhancers_improve_si_sdr(mixture, name, min_gain_db):
    clean, noisy = mixture
    enh = nc.get_enhancer(name)
    y, p = nc.enhance(noisy, SR, enh)
    assert len(y) == len(noisy) and np.all(np.isfinite(y))
    assert _si_sdr(clean, y) - _si_sdr(clean, noisy) >= min_gain_db
    assert p["kind"] == "neural" and p["model_weights_sha256"] and p["model_revision"]
    # Same input, same output: the listening version can be regenerated.
    y2, _ = nc.enhance(noisy, SR, enh)
    np.testing.assert_allclose(y, y2, atol=1e-6)


def test_unipase_runs_offline_on_cpu_and_stays_aligned(mixture):
    from pystoi import stoi

    clean, noisy = mixture
    enh = nc.get_enhancer("unipase")
    y, p = nc.enhance(noisy, SR, enh)
    assert len(y) == len(noisy) and np.all(np.isfinite(y))
    assert p["backend_params"]["enable_plc"] is False and p["backend_params"]["generative"] is True
    assert set(p["model_weights_sha256"]) == {"DeWavLM-Omni.pt", "Adapter.pt", "Vocoder_DWO-L1.pt"}
    assert stoi(clean, y, SR, extended=True) > stoi(clean, noisy, SR, extended=True)
    # Re-synthesised, but time-aligned with the original (needed for A/B listening and remix).
    from akoe.stages.remix import _lag

    lag, peak = _lag(y, clean, int(0.05 * SR))
    assert abs(lag) <= int(0.005 * SR) and peak > 0.5


@pytest.mark.parametrize("method", ["silero", "firered"])
def test_neural_vads_find_the_italian_speech(mixture, method):
    clean, noisy = mixture
    segs, p = vad.detect_regions(noisy, SR, method=method)
    assert p["kind"] == "neural" and p["model_weights_sha256"]
    speech = (1.0, len(clean) / SR - 1.0)
    covered = sum(max(0.0, min(e, speech[1]) - max(s, speech[0])) for s, e in segs)
    assert covered >= 0.7 * (speech[1] - speech[0])
    # No detection confined to the leading car noise (well before the speech).
    assert not [(s, e) for s, e in segs if e < speech[0] - 0.3]


def test_engine_clean_writes_a_real_neural_listening_variant(tmp_path):
    from akoe import engine
    from akoe.case import Case
    from akoe.config import Channel, Mode, RecordingSource
    from akoe.models import RecordingCondition
    from tests.engine_synth import make_evidence

    case = Case.create(tmp_path / "case", "c")
    paths = make_evidence(tmp_path / "ev", seconds=4.0)
    ev = case.import_evidence(paths["evidence"], RecordingCondition(channel=Channel.CAR,
                                                                     source=RecordingSource.LOCAL))
    res = engine.clean(case, ev.evidence_id, variants=["bands"], neural=["gtcrn"], vad="silero",
                       mode=Mode.BOTH)
    index = engine.listening_index(case, ev.evidence_id)
    assert [v["variant"] for v in index["variants"]] == ["original", "bands", "neural_gtcrn"]
    assert res.summary["variants"] == ["bands", "neural_gtcrn"]
