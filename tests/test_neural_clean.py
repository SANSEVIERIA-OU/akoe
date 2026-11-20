# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Neural cleaning for human listening (issue #59) and neural VAD options
(issue #60), with fakes: no model is downloaded or loaded."""
import hashlib
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pytest
import soundfile as sf

from akoe import engine
from akoe.case import Case
from akoe.config import Channel, Mode, RecordingSource, StageKind, Tier
from akoe.models import ProcessingLog, RecordingCondition
from akoe.neural import BackendUnavailable, ModelProvenance
from akoe.stages import denoise, neural_backends as nb, neural_clean as nc, vad
from tests import m6_synth as s
from tests.engine_synth import make_evidence
from tests.fakes import FakeProvider

SR = 16000


class FakeEnhancer:
    """Halves the signal at its own rate; records the chunks it saw."""

    def __init__(self, name: str = "gtcrn", native_sr: int = 16000,
                 chunking: nc.Chunking = nc.Chunking(2.0, 0.5, 0.25), gain: float = 0.5) -> None:
        self.name = name
        self.native_sr = native_sr
        self.chunking = chunking
        self.gain = gain
        self.calls: list[int] = []
        self.provenance: ModelProvenance | None = None

    def load(self) -> ModelProvenance:
        if self.provenance is None:
            self.provenance = ModelProvenance(backend=self.name, model_id=f"fake/{self.name}",
                                              revision="0" * 40, weights={"w.onnx": "a" * 64},
                                              license="test")
        return self.provenance

    def describe(self) -> dict:
        return {"gain": self.gain, "match_level": "none"}

    def enhance_chunk(self, audio: np.ndarray) -> np.ndarray:
        self.calls.append(len(audio))
        return audio * self.gain


class UnavailableEnhancer(FakeEnhancer):
    def load(self) -> ModelProvenance:
        raise BackendUnavailable("pesi mancanti (prova)")


class FakeDetector:
    name = "fakevad"
    kind = StageKind.NEURAL
    provenance = None

    def detect(self, audio, sr):
        segs = [(0.5, 1.5)]
        return segs, {"segments": [[0.5, 1.5]], "n_segments": 1}


def _log(case, result) -> ProcessingLog:
    assert result.log.verify(case.root)
    return ProcessingLog.model_validate_json(result.log.resolve(case.root).read_text())


# --------------------------------------------------------------------------
# Chunking, resampling, parameter record
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n,chunk,overlap,xfade", [(10, 20, 5, 2), (100, 30, 10, 6), (101, 30, 10, 10),
                                                    (1000, 160, 64, 16), (37, 16, 8, 8), (13, 8, 4, 3)])
def test_chunk_weights_sum_to_one_and_cover_the_signal(n, chunk, overlap, xfade):
    bounds = nc.chunk_bounds(n, chunk, overlap)
    assert bounds[0][0] == 0 and bounds[-1][1] == n
    assert all(e - s == min(chunk, n) for s, e in bounds)
    assert all(b[0] - a[0] > 0 and a[1] - b[0] >= min(overlap, n) for a, b in zip(bounds, bounds[1:]))
    weights, used = nc.chunk_weights(bounds, xfade)
    total = np.zeros(n)
    for (a, b), w in zip(bounds, weights):
        assert np.all(w >= 0) and np.all(w <= 1)
        total[a:b] += w
    np.testing.assert_allclose(total, 1.0, atol=1e-12)
    assert used <= xfade


def test_identity_through_chunks_is_exact():
    x = np.random.default_rng(0).standard_normal(SR * 7)
    y, info = nc.process_chunked(lambda c: c, x, SR, nc.Chunking(2.0, 0.5, 0.25))
    np.testing.assert_allclose(y, x.astype(np.float32), atol=1e-6)
    assert info["n_chunks"] >= 4 and info["crossfade_used_s"] == pytest.approx(0.25)


def test_chunking_validation():
    with pytest.raises(ValueError):
        nc.Chunking(2.0, 2.0, 0.0)
    with pytest.raises(ValueError):
        nc.Chunking(2.0, 0.5, 1.0)


def test_enhance_converts_rate_keeps_length_and_records_everything():
    x8 = 0.1 * np.random.default_rng(1).standard_normal(8000 * 5)
    enh = FakeEnhancer(native_sr=48000)
    y, p = nc.enhance(x8, 8000, enh)
    assert len(y) == len(x8)
    assert p["kind"] == "neural" and p["intended_use"] == "human_listening"
    assert p["backend"] == "gtcrn" and p["variant"] == "neural_gtcrn"
    assert p["model_id"] == "fake/gtcrn" and p["model_revision"] == "0" * 40
    assert p["model_weights_sha256"] == {"w.onnx": "a" * 64} and p["device"] == "cpu"
    assert p["model_provenance"]["backend"] == "gtcrn"
    assert p["input_sr"] == 8000 and p["model_sr"] == 48000 and p["output_sr"] == 8000
    assert "resample_poly" in p["resampling"]
    assert p["chunking"]["n_chunks"] == len(enh.calls) and max(enh.calls) == 2 * 48000
    assert "non è l'audio originale" in p["label_it"]
    assert p["processing_s"] >= 0 and p["real_time_factor"] is not None
    s.assert_loggable(p, "listening:neural_gtcrn")
    # Gain 0.5 survives the round trip through 48 kHz (band-limited noise aside).
    assert np.std(y) == pytest.approx(0.5 * np.std(x8), rel=0.05)


def test_short_inputs_are_padded_for_the_model_and_empty_input_is_refused():
    enh = FakeEnhancer(native_sr=48000)
    y, _ = nc.enhance(0.1 * np.ones(40), 8000, enh)
    assert len(y) == 40 and enh.calls == [int(nc.MIN_MODEL_INPUT_S * 48000)]
    with pytest.raises(ValueError, match="vuoto"):
        nc.enhance(np.zeros(0), SR, FakeEnhancer())


def test_level_matching_and_peak_protection():
    x = 0.5 * np.sin(2 * np.pi * 200 * np.arange(SR) / SR)
    y, p = nc.enhance(x, SR, FakeEnhancer(gain=0.25), match_level="peak")
    assert np.max(np.abs(y)) == pytest.approx(0.5, rel=1e-3) and p["level_gain"] == pytest.approx(4.0, rel=1e-3)
    y, p = nc.enhance(x, SR, FakeEnhancer(gain=4.0))
    assert np.max(np.abs(y)) <= 0.99 + 1e-9 and p["peak_protection_gain"] < 1.0
    with pytest.raises(ValueError):
        nc.enhance(x, SR, FakeEnhancer(), match_level="loud")


def test_backend_registry_and_tier_defaults():
    assert set(nc.available_backends()) == {"unipase", "mossformer2_se_48k", "gtcrn", "deepfilternet3"}
    assert nc.default_backends(Tier.EDGE) == ("gtcrn", "unipase")
    assert nc.default_backends(Tier.WORKSTATION) == ("unipase",) == nc.default_backends(Tier.CLOUD)
    assert nc.resolve_backends("default", Tier.EDGE) == ["gtcrn", "unipase"]
    assert nc.resolve_backends(["deepfilternet3", "default", "gtcrn"], Tier.CLOUD) == \
        ["deepfilternet3", "unipase", "gtcrn"]
    assert nc.resolve_backends(None, Tier.EDGE) == []
    with pytest.raises(ValueError, match="sconosciuto"):
        nc.resolve_backends(["storm"], Tier.EDGE)
    assert nc.is_neural_variant("neural_unipase") and not nc.is_neural_variant("neural_storm")
    assert not nc.is_neural_variant("bands")
    for name in nc.available_backends():
        e = nc.get_enhancer(name)  # constructing never loads or downloads
        assert e.provenance is None and e.name == name and isinstance(e, nc.NeuralEnhancer)
        assert e.chunking.chunk_s and e.native_sr in (16000, 48000)
        assert nc.label_it(name).startswith("Pulizia neurale")


def test_run_neural_needs_an_explicit_backend(tmp_path):
    src = tmp_path / "in.wav"
    sf.write(src, (0.1 * np.random.default_rng(2).standard_normal(SR * 3)).astype(np.float32), SR)
    with pytest.raises(NotImplementedError, match="nessun backend"):
        denoise.run_neural(src, tmp_path / "out.wav")
    p = denoise.run_neural(src, tmp_path / "out.wav", backend=FakeEnhancer())
    assert (tmp_path / "out.wav").exists() and p["kind"] == "neural"
    assert len(p["input_sha256"]) == 64 and len(p["output_sha256"]) == 64 and p["n_clipped_samples"] == 0


# --------------------------------------------------------------------------
# Pinned downloads and code snapshots (file:// URLs, no network)
# --------------------------------------------------------------------------


def test_fetch_pinned_verifies_sha256(tmp_path, monkeypatch):
    monkeypatch.setenv("AKOE_MODEL_DIR", str(tmp_path / "cache"))
    src = tmp_path / "weights.bin"
    src.write_bytes(b"model weights")
    sha = hashlib.sha256(b"model weights").hexdigest()
    with pytest.raises(BackendUnavailable, match="non presente in locale"):
        nc.fetch_pinned(nc.PinnedFile(src.as_uri(), sha, "m/weights.bin"), local_files_only=True)
    got = nc.fetch_pinned(nc.PinnedFile(src.as_uri(), sha, "m/weights.bin"))
    assert got.read_bytes() == b"model weights"
    with pytest.raises(RuntimeError, match="SHA-256"):
        nc.fetch_pinned(nc.PinnedFile(src.as_uri(), "0" * 64, "m/other.bin"))
    assert not (tmp_path / "cache" / "m" / "other.bin").exists()


def test_code_snapshot_imports_verified_files_under_a_private_name(tmp_path, monkeypatch):
    monkeypatch.setenv("AKOE_MODEL_DIR", str(tmp_path))
    files = {"models/pkg/__init__.py": b"", "models/pkg/core.py": b"from .helper import VALUE\n",
             "models/pkg/helper.py": b"VALUE = 42\n"}
    snap = nc.CodeSnapshot(name="toy", repo="example/toy", commit="c" * 40,
                           files={k: hashlib.sha256(v).hexdigest() for k, v in files.items()})
    for rel, data in files.items():
        p = snap.root() / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    mod = snap.import_module("models.pkg.core", local_files_only=True)
    assert mod.VALUE == 42 and mod.__name__.startswith("_akoe_ext_toy_")
    assert snap.record()["files_sha256"]["models/pkg/helper.py"] == hashlib.sha256(b"VALUE = 42\n").hexdigest()
    (snap.root() / "models/pkg/helper.py").write_bytes(b"VALUE = 666\n")
    from akoe.neural import _HASH_CACHE

    _HASH_CACHE.clear()
    with pytest.raises(RuntimeError, match="SHA-256"):
        snap.ensure(local_files_only=True)


# --------------------------------------------------------------------------
# Backend DSP with fake ONNX sessions
# --------------------------------------------------------------------------


class _Session:
    def __init__(self, fn):
        self.fn = fn

    def run(self, _outputs, feeds):
        return self.fn(feeds)


def test_gtcrn_stft_round_trip_and_frame_loop_with_identity_model():
    x = np.random.default_rng(3).standard_normal(SR * 2).astype(np.float32)
    win = np.sqrt(nb._hann_periodic(512))
    spec = nb.stft_centered(x, 512, 256, win)
    assert spec.shape == (1 + len(x) // 256, 257)
    np.testing.assert_allclose(nb.istft_centered(spec, 512, 256, win, len(x)), x, atol=1e-5)
    torch = pytest.importorskip("torch")
    ref = torch.stft(torch.from_numpy(x.astype(np.float64)), 512, 256, 512,
                     torch.from_numpy(win), return_complex=True).numpy().T
    np.testing.assert_allclose(spec, ref, atol=1e-8)

    g = nb.GTCRNEnhancer()
    g._s.provenance = ModelProvenance(backend="gtcrn", model_id="fake")
    g._s.state = _Session(lambda f: [f["mix"].copy(), f["conv_cache"], f["tra_cache"], f["inter_cache"]])
    np.testing.assert_allclose(g.enhance_chunk(x), x, atol=1e-5)


def test_deepfilternet3_dsp_matches_libdf_and_reconstructs_with_identity_filter():
    widths = nb.dfn_erb_widths()
    assert widths.sum() == 481 and len(widths) == 32
    assert widths.tolist() == [2] * 13 + [5, 5, 7, 7, 8, 10, 12, 13, 15, 18, 20, 24, 28, 31, 37, 42, 50, 56, 67]
    w = nb.dfn_vorbis_window(960).astype(np.float64)
    np.testing.assert_allclose(w[:480] ** 2 + w[480:] ** 2, 1.0, atol=1e-6)  # Princen-Bradley

    d = nb.DeepFilterNet3Enhancer()
    x = 0.1 * np.random.default_rng(4).standard_normal(48000 * 2)
    spec, n = d.analysis(x)
    np.testing.assert_allclose(d.synthesis(spec, n), x, atol=1e-5)
    fe, fs = d.features(spec)
    assert fe.shape == (spec.shape[0], 32) and fs.shape == (2, spec.shape[0], 96)

    t = spec.shape[0]

    def enc(f):
        assert f["feat_erb"].shape == (1, 1, t, 32) and f["feat_spec"].shape == (1, 2, t, 96)
        z = np.zeros((1, 64, t, 8), np.float32)
        return [z, z, z, z, np.zeros((1, t, 512), np.float32), np.zeros((1, 64, t, 96), np.float32),
                np.zeros((1, t, 1), np.float32)]

    coefs = np.zeros((1, t, 96, 5, 2), np.float32)
    coefs[..., 2, 0] = 1.0  # tap 2 is the current frame (look-ahead 2): identity filter
    d._s.provenance = ModelProvenance(backend="deepfilternet3", model_id="fake")
    d._s.state = {"enc": _Session(enc),
                  "erb_dec": _Session(lambda f: [np.ones((1, 1, t, 32), np.float32)]),
                  "df_dec": _Session(lambda f: [coefs.reshape(1, t, 96, 10)])}
    np.testing.assert_allclose(d.enhance_chunk(x), x, atol=1e-5)
    coefs[..., 2, 0] = 0.0
    coefs[..., 3, 0] = 1.0  # tap 3 = next frame: the low band comes out one hop early
    y = d.enhance_chunk(x)
    assert np.max(np.abs(y - x)) > 1e-3


# --------------------------------------------------------------------------
# VAD options
# --------------------------------------------------------------------------


def test_detect_regions_robust_is_deterministic_default():
    x = s.speech_like(6.0, [(1.0, 2.5), (3.5, 5.0)], seed=1) + s.colored_noise(6.0, seed=2, level=0.01)
    segs, p = vad.detect_regions(x, SR)
    ref, _ = vad.detect_speech(x, SR)
    assert segs == ref and p["method"] == "robust" and p["kind"] == "deterministic"
    assert vad.DEFAULT_VAD == "robust" and vad.vad_kind("robust") == StageKind.DETERMINISTIC
    assert vad.vad_kind("silero") == StageKind.NEURAL
    segs, p = vad.detect_regions(x, SR, detector=FakeDetector())
    assert segs == [(0.5, 1.5)] and p["kind"] == "neural" and p["method"] == "fakevad"
    with pytest.raises(ValueError, match="sconosciuto"):
        vad.detect_regions(x, SR, method="webrtc")


def test_neural_vad_detectors_construct_without_loading():
    from akoe.stages import vad_neural

    for name in ("silero", "firered"):
        d = vad_neural.get_detector(name)
        assert d.provenance is None and d.kind == StageKind.NEURAL
    with pytest.raises(ValueError):
        vad_neural.get_detector("webrtc")
    assert all(len(v) == 64 for v in vad_neural.FireRedVAD.FILES.values())


# --------------------------------------------------------------------------
# Engine and API wiring
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def case_ev(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("neural")
    case = Case.create(tmp / "case", "caso-neurale")
    paths = make_evidence(tmp / "reperti", seconds=5.0)
    ev = case.import_evidence(paths["evidence"], RecordingCondition(channel=Channel.CAR,
                                                                     source=RecordingSource.LOCAL))
    return case, ev


def test_clean_adds_labelled_neural_variants_beside_deterministic_ones(case_ev):
    case, ev = case_ev
    fake = FakeEnhancer()
    res = engine.clean(case, ev.evidence_id, variants=["denoise_offline"], mode=Mode.BOTH,
                       neural={"gtcrn": fake, "unipase": UnavailableEnhancer(name="unipase")})
    log = _log(case, res)
    names = [st.name for st in log.stages]
    assert "listening:denoise_offline" in names and "listening:neural_gtcrn" in names
    assert "listening:neural_unipase_skipped" in names
    st = next(x for x in log.stages if x.name == "listening:neural_gtcrn")
    assert st.kind == StageKind.NEURAL and st.model_version == f"fake/gtcrn@{'0' * 40}"
    assert st.model_sha256 and st.params["model_weights_sha256"] == {"w.onnx": "a" * 64}
    det = next(x for x in log.stages if x.name == "listening:denoise_offline")
    assert det.kind == StageKind.DETERMINISTIC

    index = engine.listening_index(case, ev.evidence_id)
    by_name = {v["variant"]: v for v in index["variants"]}
    assert list(by_name) == ["original", "denoise_offline", "neural_gtcrn"]
    assert by_name["denoise_offline"]["audio_provenance"] == "deterministic"
    assert by_name["neural_gtcrn"]["audio_provenance"] == "neural"
    assert "/listening/neural/" in "/" + by_name["neural_gtcrn"]["audio"]["path"]
    assert "/listening/neural/" not in "/" + by_name["denoise_offline"]["audio"]["path"]
    # Neural audio is computed from the original only: half of it, sample for sample.
    orig, _ = sf.read(case.path_of(ev), dtype="float64")
    neural, _ = sf.read(case.root / by_name["neural_gtcrn"]["audio"]["path"], dtype="float64")
    np.testing.assert_allclose(neural, 0.5 * orig, atol=2.0 / 32768)
    assert res.summary["variants"] == ["denoise_offline", "neural_gtcrn"]


def test_neural_cleaning_is_refused_in_forensic_mode_and_needs_known_backends(case_ev):
    case, ev = case_ev
    with pytest.raises(engine.EngineError, match="modalità investigativa"):
        engine.clean(case, ev.evidence_id, variants=["bands"], neural=["gtcrn"], mode=Mode.FORENSIC)
    with pytest.raises(engine.EngineError, match="sconosciuto"):
        engine.clean(case, ev.evidence_id, variants=["bands"], neural=["storm"], mode=Mode.BOTH)


def test_default_neural_backends_come_from_the_provider(case_ev):
    case, ev = case_ev
    made = {}

    class Provider(FakeProvider):
        def enhancer(self, name):
            made[name] = FakeEnhancer(name=name)
            return made[name]

    res = engine.clean(case, ev.evidence_id, variants=[], mode=Mode.INVESTIGATIVE, neural="default",
                       tier=Tier.EDGE, provider=Provider())
    assert set(made) == {"gtcrn", "unipase"}
    assert res.summary["variants"] == ["neural_gtcrn", "neural_unipase"]
    log = _log(case, res)
    assert log.parameters["neural"] == ["gtcrn", "unipase"]


def test_vad_choice_is_configurable_and_neural_vad_is_logged_as_neural(case_ev):
    case, ev = case_ev
    res = engine.clean(case, ev.evidence_id, variants=["denoise_offline"], vad=FakeDetector())
    log = _log(case, res)
    st = next(x for x in log.stages if x.name.startswith("vad_"))
    assert st.name == "vad_fakevad" and st.kind == StageKind.NEURAL
    assert any("VAD neurale" in n for n in log.notes)
    res = engine.clean(case, ev.evidence_id, variants=["denoise_offline"])
    st = next(x for x in _log(case, res).stages if x.name.startswith("vad_"))
    assert st.name == "vad_robust" and st.kind == StageKind.DETERMINISTIC
    with pytest.raises(engine.EngineError, match="VAD sconosciuto"):
        engine.clean(case, ev.evidence_id, variants=["denoise_offline"], vad="webrtc")

    calls = []

    class Provider(FakeProvider):
        def vad_detector(self, name, **params):
            calls.append((name, params))
            return FakeDetector()

    res = engine.transcribe(case, ev.evidence_id, vad="silero", vad_params={"threshold": 0.6},
                            provider=Provider())
    log = _log(case, res)
    st = next(x for x in log.stages if x.name.startswith("vad_"))
    assert calls == [("silero", {"threshold": 0.6})] and st.kind == StageKind.NEURAL
    assert st.params["asr_regions"] and log.parameters["vad"] == "silero"


def test_api_serves_neural_variants_with_their_provenance(tmp_path):
    from fastapi.testclient import TestClient

    from akoe.api.server import create_app
    from tests.test_api_workflows import H, _job

    class Provider(FakeProvider):
        def enhancer(self, name):
            return FakeEnhancer(name=name)

    paths = make_evidence(tmp_path / "reperti", seconds=4.0)
    client = TestClient(create_app(tmp_path / "cases", token="t0k", backends=Provider(),
                                   import_roots=[tmp_path]))
    assert client.post("/cases", json={"case_id": "c1"}, headers=H).status_code == 201
    with paths["evidence"].open("rb") as f:
        item = client.post("/cases/c1/evidence", headers=H, files={"file": ("l.wav", f, "audio/wav")},
                           data={"channel": "car", "source": "local"}).json()
    eid = item["evidence_id"]
    _job(client, "clean", evidence_id=eid, variants=["bands"], neural=["gtcrn"], mode="both", vad="robust")
    listing = client.get(f"/cases/c1/evidence/{eid}/listening", headers=H).json()
    assert [v["variant"] for v in listing["variants"]] == ["original", "bands", "neural_gtcrn"]
    audio = client.get(f"/cases/c1/evidence/{eid}/listening/neural_gtcrn/audio", headers=H)
    assert audio.status_code == 200 and audio.headers["x-audio-provenance"] == "neural"
    assert "Ipotesi neurale" in unquote(audio.headers["x-audio-label"])
    assert client.get(f"/cases/c1/evidence/{eid}/listening/neural_storm/audio", headers=H).status_code == 404
    r = client.post("/cases/c1/jobs", json={"kind": "clean", "params": {"evidence_id": eid, "neural": 3}},
                    headers=H)
    assert r.status_code in (400, 422)


def test_cli_exposes_neural_and_vad_options():
    from typer.testing import CliRunner

    from akoe.cli import app

    out = CliRunner().invoke(app, ["clean", "--help"]).output
    assert "--neural" in out and "--vad" in out
    assert "--vad" in CliRunner().invoke(app, ["transcribe", "--help"]).output


def test_new_files_carry_the_licence_header():
    root = Path(__file__).resolve().parents[1]
    for rel in ("src/akoe/stages/neural_clean.py", "src/akoe/stages/neural_backends.py",
                "src/akoe/stages/vad_neural.py", "tests/test_neural_clean.py"):
        assert (root / rel).read_text().startswith("# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1")
