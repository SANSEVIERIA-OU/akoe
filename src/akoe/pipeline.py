# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from akoe import models
from akoe.config import CaseConfig, Mode, RunProfile, StageKind
from akoe.stages import asr, denoise, embedding, extract, preprocess, vad
from akoe.utils import sha256_file


def _now() -> datetime:
    return datetime.now(timezone.utc)


def process(cfg: CaseConfig) -> models.ProcessingLog:
    """Run the pipeline end-to-end.

    Forensic invariant: when mode=BOTH, two audio artifacts are produced
    (enhanced_deterministic.wav, enhanced_neural.wav) and MUST NOT be merged.

    Profile vehicle-single-speaker (default): target speaker extraction is
    parked and the transcriber receives the preprocessed original audio,
    because enhancement in front of modern ASR usually hurts (MASTER_PLAN
    §3.4.1). Cleaned versions are for human listening.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    log = models.ProcessingLog(
        case_id=cfg.case_id,
        mode=cfg.mode,
        tier=cfg.tier,
        profile=cfg.profile,
        input_files={
            "evidence": str(cfg.evidence_path),
            "samples_dir": str(cfg.speaker_samples_dir),
        },
    )

    def _stage(name: str, kind: StageKind, params: dict[str, Any], started: datetime,
               finished: datetime, in_path: Path | None = None,
               out_path: Path | None = None) -> models.StageLog:
        return models.StageLog(
            name=name, kind=kind, mode=cfg.mode, params=params, tier=cfg.tier,
            input_sha256=sha256_file(in_path) if in_path and in_path.exists() else None,
            output_sha256=sha256_file(out_path) if out_path and out_path.exists() else None,
            started_at=started, finished_at=finished,
        )

    def _skipped(name: str, kind: StageKind, params: dict[str, Any], reason: str,
                 started: datetime | None = None) -> None:
        log.stages.append(_stage(f"{name}_skipped", kind, {**params, "reason": reason},
                                 started or _now(), _now()))

    def _run(name: str, kind: StageKind, in_path: Path | None, out_path: Path | None,
             params: dict[str, Any], fn: Callable[[], None]) -> bool:
        started = _now()
        try:
            fn()
        except NotImplementedError as e:
            _skipped(name, kind, params, str(e) or "not implemented", started)
            return False
        log.stages.append(_stage(name, kind, params, started, _now(), in_path, out_path))
        return True

    pre_out = cfg.output_dir / "preprocessed.wav"
    _run("preprocess", StageKind.DETERMINISTIC, cfg.evidence_path, pre_out,
         {"target_sr": 16000},
         lambda: preprocess.run(cfg.evidence_path, pre_out))

    det_out: Path | None = None
    if cfg.mode in (Mode.FORENSIC, Mode.BOTH):
        det_out = cfg.output_dir / "enhanced_deterministic.wav"
        _run("denoise_deterministic", StageKind.DETERMINISTIC, pre_out, det_out,
             {"algo": "spectral_gating_stationary"},
             lambda: denoise.run_deterministic(pre_out, det_out))
        log.output_files["enhanced_deterministic"] = str(det_out)

    neural_final: Path | None = None
    if cfg.mode in (Mode.INVESTIGATIVE, Mode.BOTH):
        neural_denoised = cfg.output_dir / "_neural_denoised.wav"
        ok = _run("denoise_neural", StageKind.NEURAL, pre_out, neural_denoised,
                  {"model": "speaker-informed (planned)", "purpose": "human listening"},
                  lambda: denoise.run_neural(pre_out, neural_denoised))
        if ok and cfg.profile == RunProfile.GENERIC:
            try:
                emb = embedding.run(cfg.speaker_samples_dir)
                neural_final = cfg.output_dir / "enhanced_neural.wav"
                _run("extract_target_speaker", StageKind.NEURAL,
                     neural_denoised, neural_final, {"model": "WeSep"},
                     lambda: extract.run(neural_denoised, neural_final, emb))
                if not neural_final.exists():
                    neural_final = None
            except NotImplementedError as e:
                _skipped("embedding", StageKind.NEURAL, {}, str(e) or "not implemented")
        elif ok:
            neural_final = neural_denoised
        if neural_final is not None:
            log.output_files["enhanced_neural"] = str(neural_final)

    if cfg.profile == RunProfile.VEHICLE_SINGLE_SPEAKER:
        asr_input: Path | None = pre_out
        asr_variant = "original"
    else:
        asr_input = neural_final or det_out
        asr_variant = "enhanced"

    if asr_input is not None and asr_input.exists():
        started = _now()
        segments = vad.run(asr_input)
        log.stages.append(_stage(
            "vad", StageKind.DETERMINISTIC,
            {"frame_ms": 30, "threshold_db": -40.0, "min_speech_s": 0.3,
             "n_segments": len(segments)},
            started, _now(), in_path=asr_input,
        ))

        try:
            transcript = asr.run(asr_input, cfg.case_id, segments)
            transcript_path = cfg.output_dir / "transcript.json"
            transcript_path.write_text(transcript.model_dump_json(indent=2))
            log.output_files["transcript"] = str(transcript_path)
        except NotImplementedError as e:
            _skipped("asr", StageKind.NEURAL,
                     {"vad_segments_available": len(segments), "asr_input_variant": asr_variant},
                     str(e) or "not implemented")

    (cfg.output_dir / "processing_log.json").write_text(log.model_dump_json(indent=2))
    return log
