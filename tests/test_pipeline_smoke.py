# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Smoke test: forensic-mode pipeline runs end-to-end on synthetic audio.

Validates plumbing only — no model quality assertions. Neural stages are
expected to be skipped (recorded in the processing log).
"""
import json
from pathlib import Path

from akoe.config import CaseConfig, Mode
from akoe.pipeline import process


def test_forensic_pipeline_end_to_end(tmp_path: Path) -> None:
    from scripts.synth_test_audio import main as synth

    data_root = tmp_path / "data"
    synth(out_root=data_root)

    cfg = CaseConfig(
        case_id="smoke_001",
        evidence_path=data_root / "evidence.wav",
        speaker_samples_dir=data_root / "speaker_samples",
        output_dir=tmp_path / "out",
        mode=Mode.FORENSIC,
    )
    log = process(cfg)

    assert (cfg.output_dir / "preprocessed.wav").exists()
    assert (cfg.output_dir / "enhanced_deterministic.wav").exists()
    assert (cfg.output_dir / "processing_log.json").exists()

    stages = [s.name for s in log.stages]
    assert "preprocess" in stages
    assert "denoise_deterministic" in stages
    assert "vad" in stages
    # ASR is not yet implemented — must be recorded as skipped, not crash.
    assert "asr_skipped" in stages

    # Every recorded output has a hash for chain of custody.
    for stage in log.stages:
        if stage.name in ("preprocess", "denoise_deterministic"):
            assert stage.input_sha256 is not None
            assert stage.output_sha256 is not None

    # processing_log.json is valid JSON and round-trips.
    parsed = json.loads((cfg.output_dir / "processing_log.json").read_text())
    assert parsed["case_id"] == "smoke_001"
    assert parsed["mode"] == "forensic"


def test_investigative_mode_skips_unimplemented_stages(tmp_path: Path) -> None:
    from scripts.synth_test_audio import main as synth

    data_root = tmp_path / "data"
    synth(out_root=data_root)

    cfg = CaseConfig(
        case_id="smoke_002",
        evidence_path=data_root / "evidence.wav",
        speaker_samples_dir=data_root / "speaker_samples",
        output_dir=tmp_path / "out",
        mode=Mode.INVESTIGATIVE,
    )
    log = process(cfg)

    stages = [s.name for s in log.stages]
    assert "preprocess" in stages
    assert "denoise_neural_skipped" in stages
    # No enhanced_neural.wav because the neural branch was skipped.
    assert "enhanced_neural" not in log.output_files


def test_tier_and_profile_are_recorded(tmp_path: Path) -> None:
    from akoe.config import RunProfile, Tier
    from scripts.synth_test_audio import main as synth

    data_root = tmp_path / "data"
    synth(out_root=data_root)
    cfg = CaseConfig(
        case_id="smoke_003",
        evidence_path=data_root / "evidence.wav",
        speaker_samples_dir=data_root / "speaker_samples",
        output_dir=tmp_path / "out",
        mode=Mode.BOTH,
        tier=Tier.WORKSTATION,
        profile=RunProfile.VEHICLE_SINGLE_SPEAKER,
    )
    log = process(cfg)
    assert log.tier == Tier.WORKSTATION and log.profile == RunProfile.VEHICLE_SINGLE_SPEAKER
    assert all(s.tier == Tier.WORKSTATION for s in log.stages)
    # Vehicle profile: no target speaker extraction, ASR input is the original audio.
    names = [s.name for s in log.stages]
    assert not any(n.startswith("extract_target_speaker") for n in names)
    asr = next(s for s in log.stages if s.name == "asr_skipped")
    assert asr.params["asr_input_variant"] == "original"
