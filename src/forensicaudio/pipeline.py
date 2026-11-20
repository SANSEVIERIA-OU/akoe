from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from forensicaudio import models
from forensicaudio.config import CaseConfig, Mode, StageKind
from forensicaudio.stages import asr, denoise, embedding, extract, preprocess, vad
from forensicaudio.utils import sha256_file


def _now() -> datetime:
    return datetime.now(timezone.utc)


def process(cfg: CaseConfig) -> models.ProcessingLog:
    """Run the pipeline end-to-end.

    Forensic invariant: when mode=BOTH, two audio artifacts are produced
    (enhanced_deterministic.wav, enhanced_neural.wav) and MUST NOT be merged.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    log = models.ProcessingLog(
        case_id=cfg.case_id,
        mode=cfg.mode,
        input_files={
            "evidence": str(cfg.evidence_path),
            "samples_dir": str(cfg.speaker_samples_dir),
        },
    )

    def _record(name: str, kind: StageKind, in_path: Path | None,
                out_path: Path | None, params: dict[str, Any],
                started: datetime, finished: datetime,
                model_version: str | None = None) -> None:
        log.stages.append(models.StageLog(
            name=name, kind=kind, mode=cfg.mode, params=params,
            input_sha256=sha256_file(in_path) if in_path and in_path.exists() else None,
            output_sha256=sha256_file(out_path) if out_path and out_path.exists() else None,
            model_version=model_version,
            started_at=started, finished_at=finished,
        ))

    def _run(name: str, kind: StageKind, in_path: Path | None, out_path: Path | None,
             params: dict[str, Any], fn: Callable[[], None]) -> bool:
        started = _now()
        try:
            fn()
        except NotImplementedError as e:
            log.stages.append(models.StageLog(
                name=f"{name}_skipped", kind=kind, mode=cfg.mode,
                params={**params, "reason": str(e) or "not implemented"},
                started_at=started, finished_at=_now(),
            ))
            return False
        _record(name, kind, in_path, out_path, params, started, _now())
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
                  {"model": "DeepFilterNet"},
                  lambda: denoise.run_neural(pre_out, neural_denoised))
        if ok:
            try:
                emb = embedding.run(cfg.speaker_samples_dir)
                neural_final = cfg.output_dir / "enhanced_neural.wav"
                _run("extract_target_speaker", StageKind.NEURAL,
                     neural_denoised, neural_final, {"model": "WeSep"},
                     lambda: extract.run(neural_denoised, neural_final, emb))
                if neural_final.exists():
                    log.output_files["enhanced_neural"] = str(neural_final)
                else:
                    neural_final = None
            except NotImplementedError as e:
                log.stages.append(models.StageLog(
                    name="embedding_skipped", kind=StageKind.NEURAL, mode=cfg.mode,
                    params={"reason": str(e) or "not implemented"},
                    started_at=_now(), finished_at=_now(),
                ))

    asr_input = neural_final or det_out
    if asr_input is not None and asr_input.exists():
        started = _now()
        segments = vad.run(asr_input)
        _record("vad", StageKind.DETERMINISTIC, asr_input, None,
                {"frame_ms": 30, "threshold_db": -40.0, "min_speech_s": 0.3,
                 "n_segments": len(segments)},
                started, _now())

        try:
            transcript = asr.run(asr_input, cfg.case_id, segments)
            transcript_path = cfg.output_dir / "transcript.json"
            transcript_path.write_text(transcript.model_dump_json(indent=2))
            log.output_files["transcript"] = str(transcript_path)
        except NotImplementedError as e:
            log.stages.append(models.StageLog(
                name="asr_skipped", kind=StageKind.NEURAL, mode=cfg.mode,
                params={"reason": str(e) or "not implemented",
                        "vad_segments_available": len(segments)},
                started_at=_now(), finished_at=_now(),
            ))

    (cfg.output_dir / "processing_log.json").write_text(log.model_dump_json(indent=2))
    return log
