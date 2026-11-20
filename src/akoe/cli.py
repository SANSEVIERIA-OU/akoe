# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
from pathlib import Path

import typer

from akoe import pipeline
from akoe.config import (
    CaseConfig,
    Channel,
    Mode,
    RecordingSource,
    RunProfile,
    Tier,
    VocalEffort,
)

app = typer.Typer(help="Akoe — assisted transcription adapted to the subject and the vehicle.")


@app.callback()
def _load_local_settings() -> None:
    """Read the local .env (e.g. HF_TOKEN, HF_HOME) before any command."""
    from akoe.settings import load_env

    load_env()
case_app = typer.Typer(help="Case management: layout, read-only evidence import, verification.")
app.add_typer(case_app, name="case")


@app.command()
def process(
    evidence: Path = typer.Option(..., "--evidence", "-e", help="Path to the audio evidence file."),
    samples: Path = typer.Option(..., "--speaker-samples", "-s", help="Directory with reference voice samples of the target."),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory for artifacts."),
    mode: Mode = typer.Option(Mode.FORENSIC, "--mode", "-m", help="Processing mode."),
    tier: Tier = typer.Option(Tier.EDGE, "--tier", help="Where the engine runs."),
    profile: RunProfile = typer.Option(RunProfile.VEHICLE_SINGLE_SPEAKER, "--profile", help="Pipeline shape."),
    case_id: str = typer.Option("case_001", "--case-id"),
) -> None:
    """Run the full pipeline on a single file (development entry point)."""
    cfg = CaseConfig(
        case_id=case_id,
        evidence_path=evidence,
        speaker_samples_dir=samples,
        output_dir=output,
        mode=mode,
        tier=tier,
        profile=profile,
    )
    pipeline.process(cfg)
    typer.echo(f"Done. Artifacts in {output}")


@case_app.command("init")
def case_init(
    root: Path = typer.Argument(..., help="Case directory to create."),
    case_id: str = typer.Option(..., "--case-id"),
) -> None:
    """Create the case layout."""
    from akoe.case import Case

    Case.create(root, case_id)
    typer.echo(f"Case {case_id} created at {root}")


@case_app.command("import")
def case_import(
    root: Path = typer.Argument(..., help="Case directory."),
    src: Path = typer.Argument(..., help="Recording to import (copied, hashed, made read-only)."),
    channel: Channel = typer.Option(Channel.UNKNOWN, "--channel"),
    source: RecordingSource = typer.Option(RecordingSource.OTHER, "--source"),
    effort: VocalEffort = typer.Option(VocalEffort.UNKNOWN, "--effort"),
    codec: str | None = typer.Option(None, "--codec"),
    pair_id: str | None = typer.Option(None, "--pair-id", help="Links local and transmitted recordings of the same event."),
) -> None:
    """Import a recording as evidence."""
    from akoe.case import Case
    from akoe.models import RecordingCondition

    case = Case.open(root)
    item = case.import_evidence(
        src,
        RecordingCondition(channel=channel, source=source, vocal_effort=effort, codec=codec),
        pair_id=pair_id,
    )
    typer.echo(f"{item.evidence_id}  sha256={item.artifact.sha256}")


@case_app.command("verify")
def case_verify(root: Path = typer.Argument(..., help="Case directory.")) -> None:
    """Recompute evidence hashes; exit 1 if anything changed or is missing."""
    from akoe.case import Case

    bad = Case.open(root).verify()
    if bad:
        typer.echo("MISMATCH: " + ", ".join(bad))
        raise typer.Exit(1)
    typer.echo("All evidence hashes verified.")


@app.command()
def serve(
    cases_root: Path = typer.Option(Path("cases"), "--cases-root", help="Directory holding the cases."),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address. Keep 127.0.0.1 unless behind an encrypted tunnel."),
    port: int = typer.Option(8765, "--port"),
    tier: Tier = typer.Option(Tier.EDGE, "--tier"),
    token_file: Path | None = typer.Option(None, "--token-file", help="Write the bearer token here (mode 600) instead of stdout."),
    import_root: list[Path] = typer.Option([], "--import-root", help="Cartella i cui percorsi assoluti i job possono usare (ripetibile)."),
) -> None:
    """Start the engine API for the desktop app (sidecar or remote engine)."""
    import os
    import secrets

    import uvicorn

    from akoe.api.server import create_app

    token = secrets.token_urlsafe(32)
    if token_file is not None:
        token_file.write_text(token)
        os.chmod(token_file, 0o600)
        typer.echo(f"AKOE_ENGINE_READY host={host} port={port} token_file={token_file}")
    else:
        typer.echo(f"AKOE_ENGINE_READY host={host} port={port} token={token}")
    uvicorn.run(create_app(cases_root, token=token, tier=tier, import_roots=import_root), host=host, port=port,
                log_level="warning")


@app.command(name="extract-embedding")
def extract_embedding(
    samples: Path = typer.Option(..., "--speaker-samples", "-s"),
    output: Path = typer.Option(..., "--output", "-o", help="Output .npy file."),
) -> None:
    """Compute and save a speaker embedding from reference samples."""
    import numpy as np

    from akoe.stages import embedding

    emb = embedding.run(samples)
    np.save(output, emb)
    typer.echo(f"Saved embedding to {output}")


@app.command(name="inspect-log")
def inspect_log(log_path: Path = typer.Argument(...)) -> None:
    """Pretty-print a processing_log.json."""
    typer.echo(log_path.read_text())


# --------------------------------------------------------------------------
# Case workflows (akoe.engine). Identifiers in English, help in Italian.
# --------------------------------------------------------------------------


def _open_case(root: Path):
    from akoe.case import Case, CaseError

    try:
        return Case.open(root)
    except CaseError:
        typer.echo(f"Nessun caso in {root}", err=True)
        raise typer.Exit(2)


def _report(result, as_json: bool) -> None:
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
        return
    typer.echo(f"{result.workflow}: completato. Registro: {result.log.path} (sha256 {result.log.sha256[:16]})")
    for name, ref in result.artifacts.items():
        typer.echo(f"  {name}: {ref.path}")
    lines = result.summary.get("summary_it")
    for line in lines if isinstance(lines, list) else []:
        typer.echo(f"  {line}")
    for note in result.notes:
        typer.echo(f"  Nota: {note}")


def _run(fn, *args, **kwargs):
    from akoe.engine import EngineError

    try:
        return fn(*args, **kwargs)
    except EngineError as e:
        typer.echo(f"Errore: {e}", err=True)
        raise typer.Exit(1)


def _pairs(values: list[str], what: str) -> dict[str, str]:
    out = {}
    for v in values:
        if "=" not in v:
            typer.echo(f"Formato atteso NOME=VALORE per {what}: {v}", err=True)
            raise typer.Exit(2)
        k, _, x = v.partition("=")
        out[k] = x
    return out


@app.command("subject-model")
def subject_model(
    root: Path = typer.Argument(..., help="Cartella del caso."),
    subject_id: str = typer.Option(..., "--subject-id", help="Identificativo del soggetto."),
    folder: Path | None = typer.Option(None, "--folder", help="Cartella con registrazioni e trascrizioni del soggetto (recordings.json facoltativo)."),
    channel: Channel = typer.Option(Channel.UNKNOWN, "--channel", help="Canale, se manca recordings.json."),
    source: RecordingSource = typer.Option(RecordingSource.OTHER, "--source"),
    effort: VocalEffort = typer.Option(VocalEffort.UNKNOWN, "--effort"),
    codec: str | None = typer.Option(None, "--codec"),
    subject_speaker: str | None = typer.Option(None, "--subject-speaker", help="Etichetta del soggetto nelle trascrizioni con più parlanti."),
    holdout: float = typer.Option(0.2, "--holdout", help="Quota di registrazioni messe da parte per l'autovalutazione."),
    seed: int = typer.Option(0, "--seed"),
    no_align: bool = typer.Option(False, "--no-align", help="Non allineare le trascrizioni senza tempi."),
    certify_by: str | None = typer.Option(None, "--certify-by", help="L'operatore indicato accetta le trascrizioni esistenti come certe (esclusi i segmenti dubbi)."),
    voiceprint: str | None = typer.Option(None, "--voiceprint", help="Modello dell'impronta: redimnet2-b6 (predefinito), redimnet2-b6-vox2, w2v-bert-sv, redimnet-b2, redimnet-b6, ecapa."),
    no_timbre: bool = typer.Option(False, "--no-timbre", help="Non calcolare il profilo timbrico."),
    no_calibration: bool = typer.Option(False, "--no-calibration"),
    other_speaker: list[Path] = typer.Option([], "--other-speaker", help="Clip di altri parlanti per la calibrazione (ripetibile)."),
    tier: Tier = typer.Option(Tier.EDGE, "--tier"),
    as_json: bool = typer.Option(False, "--json", help="Stampa il risultato in JSON."),
) -> None:
    """Modello del soggetto: import, allineamento, divisione, profilo e calibrazione."""
    from akoe import engine
    from akoe.models import RecordingCondition

    vp = None
    if voiceprint:
        from akoe.speaker.voiceprint_backends import default_voiceprint_backend

        vp = default_voiceprint_backend(voiceprint)
    cond = RecordingCondition(channel=channel, source=source, vocal_effort=effort, codec=codec)
    res = _run(engine.build_subject_model, _open_case(root), subject_id, folder=folder, condition=cond,
               subject_speaker=subject_speaker, holdout_fraction=holdout, seed=seed, align=not no_align,
               certify_all_by=certify_by, voiceprint=vp, compute_timbre=not no_timbre,
               calibrate=not no_calibration, other_speaker_files=other_speaker, tier=tier)
    _report(res, as_json)


@app.command("vehicle-profile")
def vehicle_profile(
    root: Path = typer.Argument(..., help="Cartella del caso."),
    vehicle_id: str = typer.Option(..., "--vehicle-id"),
    manifest: Path | None = typer.Option(None, "--manifest", help="misura.json della sessione di misura (runbook)."),
    ir: list[Path] = typer.Option([], "--ir", help="Risposta all'impulso misurata (ripetibile)."),
    noise: list[str] = typer.Option([], "--noise", help="Rumore misurato come ETICHETTA=FILE (ripetibile)."),
    estimate_from: str | None = typer.Option(None, "--estimate-from", help="Reperto da cui stimare il profilo (ripiego)."),
    tier: Tier = typer.Option(Tier.EDGE, "--tier"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Profilo del veicolo: misurato o stimato dai tratti senza voce di un reperto."""
    from akoe import engine
    from akoe.vehicle.profile import NoiseSource

    noises = [NoiseSource(label=k, path=Path(v)) for k, v in _pairs(noise, "--noise").items()]
    res = _run(engine.build_vehicle_profile, _open_case(root), vehicle_id, manifest=manifest,
               impulse_responses=ir, noise_recordings=noises, evidence_id=estimate_from, tier=tier)
    _report(res, as_json)


@app.command("clean")
def clean_cmd(
    root: Path = typer.Argument(..., help="Cartella del caso."),
    evidence_id: str = typer.Argument(..., help="Reperto."),
    variant: list[str] = typer.Option([], "--variant", help="Versione d'ascolto (ripetibile): bands, denoise_offline, engine_harmonics, reference_cancel, nmf_listening."),
    subject_id: str | None = typer.Option(None, "--subject-id"),
    vehicle_id: str | None = typer.Option(None, "--vehicle-id"),
    reference: Path | None = typer.Option(None, "--reference", help="Registrazione originale della radio/musica da cancellare."),
    neural: list[str] = typer.Option([], "--neural", help="Pulizia neurale d'ascolto (ripetibile, modalità investigative/both): unipase, mossformer2_se_48k, gtcrn, deepfilternet3, oppure 'default' per il livello."),
    vad: str = typer.Option("robust", "--vad", help="Rilevatore di voce: robust (riproducibile, predefinito), silero, firered."),
    mode: Mode = typer.Option(Mode.FORENSIC, "--mode"),
    tier: Tier = typer.Option(Tier.EDGE, "--tier"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Versioni d'ascolto etichettate per il trascrittore umano (mai unite tra loro)."""
    from akoe import engine

    res = _run(engine.clean, _open_case(root), evidence_id, variants=variant or None, subject_id=subject_id,
               vehicle_id=vehicle_id, reference=reference, neural=neural or None, vad=vad, mode=mode,
               tier=tier)
    _report(res, as_json)


@app.command("transcribe")
def transcribe_cmd(
    root: Path = typer.Argument(..., help="Cartella del caso."),
    evidence_id: str | None = typer.Option(None, "--evidence", help="Reperto da trascrivere."),
    pair_id: str | None = typer.Option(None, "--pair", help="Coppia locale/trasmessa (la locale ha la precedenza)."),
    adapter: str | None = typer.Option(None, "--adapter", help="Cartella dell'adattatore o 'latest' (con --subject-id)."),
    lexicon: Path | None = typer.Option(None, "--lexicon", help="Lessico del caso (JSON) per la lettura con/senza lessico."),
    subject_id: str | None = typer.Option(None, "--subject-id", help="Aggiunge il punteggio dell'impronta per segmento."),
    whisper_model: str | None = typer.Option(None, "--whisper-model"),
    n_best: int = typer.Option(5, "--n-best"),
    vad: str = typer.Option("robust", "--vad", help="Rilevatore di voce: robust (riproducibile, predefinito), silero, firered."),
    tier: Tier = typer.Option(Tier.EDGE, "--tier"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Ipotesi di trascrizione (sempre ipotesi AI) dall'audio originale."""
    from akoe import engine

    res = _run(engine.transcribe, _open_case(root), evidence_id, pair_id=pair_id, adapter=adapter,
               lexicon=lexicon, subject_id=subject_id, whisper_model=whisper_model, n_best=n_best, vad=vad,
               tier=tier)
    _report(res, as_json)


@app.command("candidates")
def candidates_cmd(
    root: Path = typer.Argument(..., help="Cartella del caso."),
    evidence_id: str = typer.Argument(..., help="Reperto."),
    start: float = typer.Option(..., "--start", help="Inizio del tratto (s)."),
    end: float = typer.Option(..., "--end", help="Fine del tratto (s)."),
    candidate: list[str] = typer.Option(..., "--candidate", "-c", help="Frase candidata (ripetibile, 2-5 consigliate)."),
    threshold: list[str] = typer.Option([], "--threshold", help="Soglia come METODO=VALORE (ripetibile)."),
    thresholds_from: str | None = typer.Option(None, "--thresholds-from", help="Soggetto la cui autovalutazione fornisce le soglie."),
    whisper_model: str | None = typer.Option(None, "--whisper-model"),
    tier: Tier = typer.Option(Tier.EDGE, "--tier"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """'Ha detto X o Y?': compatibilità delle frasi candidate, metodi riportati separatamente."""
    from akoe import engine

    th = {k: float(v) for k, v in _pairs(threshold, "--threshold").items()} or None
    res = _run(engine.score_candidates, _open_case(root), evidence_id, start, end, candidate, th,
               thresholds_from=thresholds_from, whisper_model=whisper_model, tier=tier)
    if as_json:
        typer.echo(res.model_dump_json(indent=2))
        return
    comp = res.summary["comparison"]
    typer.echo(comp["message"])
    for rep in comp["reports"]:
        typer.echo(f"- {rep['scorer']} (soglia {rep['threshold']}):")
        for e in rep["entries"]:
            mark = {True: "compatibile", False: "non compatibile", None: "-"}[e["compatible"]]
            typer.echo(f"    {e['rank']}. {e['candidate']!r}: {e['score']} ({mark})")
    for w in comp["warnings"]:
        typer.echo(f"  Avviso: {w}")
    typer.echo(f"Registro: {res.log.path}")


@app.command("train-adapter")
def train_adapter_cmd(
    root: Path = typer.Argument(..., help="Cartella del caso."),
    subject_id: str = typer.Option(..., "--subject-id"),
    vehicle_id: str | None = typer.Option(None, "--vehicle-id", help="Veicolo per la simulazione del canale."),
    base_model: str | None = typer.Option(None, "--base-model", help="Modello Whisper di base (predefinito secondo il livello)."),
    epochs: float = typer.Option(3.0, "--epochs"),
    max_steps: int | None = typer.Option(None, "--max-steps"),
    batch_size: int = typer.Option(4, "--batch-size"),
    lora_r: int = typer.Option(32, "--lora-r"),
    seed: int = typer.Option(1234, "--seed"),
    codec_probability: float = typer.Option(0.5, "--codec-probability"),
    tier: Tier = typer.Option(Tier.EDGE, "--tier"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Adattatore del soggetto (LoRA) su segmenti verificati, con canale del veicolo e codec variabili."""
    from akoe import engine
    from akoe.adapt.train import TrainConfig, base_model_for_tier

    cfg = TrainConfig(base_model=base_model or base_model_for_tier(tier), num_epochs=epochs, max_steps=max_steps,
                      batch_size=batch_size, lora_r=lora_r, lora_alpha=2 * lora_r, seed=seed)
    res = _run(engine.train_subject_adapter, _open_case(root), subject_id, vehicle_id=vehicle_id, train_config=cfg,
               codec_probability=codec_probability, tier=tier)
    _report(res, as_json)


@app.command("self-assess")
def self_assess_cmd(
    root: Path = typer.Argument(..., help="Cartella del caso."),
    subject_id: str = typer.Option(..., "--subject-id"),
    vehicle_id: str | None = typer.Option(None, "--vehicle-id", help="Veicolo per degradare il materiale."),
    adapter: str | None = typer.Option(None, "--adapter", help="Adattatore da confrontare con il modello generico, o 'latest'."),
    lexicon: Path | None = typer.Option(None, "--lexicon"),
    whisper_model: str | None = typer.Option(None, "--whisper-model"),
    no_degrade: bool = typer.Option(False, "--no-degrade"),
    no_candidates: bool = typer.Option(False, "--no-candidates"),
    no_voiceprint: bool = typer.Option(False, "--no-voiceprint"),
    tier: Tier = typer.Option(Tier.EDGE, "--tier"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Autovalutazione per condizione sul materiale messo da parte."""
    from akoe import engine

    res = _run(engine.self_assess, _open_case(root), subject_id, vehicle_id=vehicle_id, adapter=adapter,
               lexicon=lexicon, whisper_model=whisper_model, degrade=not no_degrade,
               include_candidates=not no_candidates, include_voiceprint=not no_voiceprint, tier=tier)
    if as_json:
        typer.echo(res.model_dump_json(indent=2))
        return
    typer.echo(res.summary.get("summary_it", ""))
    typer.echo(f"Registro: {res.log.path}")


if __name__ == "__main__":
    app()
