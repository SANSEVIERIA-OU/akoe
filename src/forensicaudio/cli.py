from pathlib import Path

import typer

from forensicaudio import pipeline
from forensicaudio.config import (
    CaseConfig,
    Channel,
    Mode,
    RecordingSource,
    RunProfile,
    Tier,
    VocalEffort,
)

app = typer.Typer(help="WHISPER — assisted transcription adapted to the subject and the vehicle.")
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
    from forensicaudio.case import Case

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
    from forensicaudio.case import Case
    from forensicaudio.models import RecordingCondition

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
    from forensicaudio.case import Case

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
) -> None:
    """Start the engine API for the desktop app (sidecar or remote engine)."""
    import os
    import secrets

    import uvicorn

    from forensicaudio.api.server import create_app

    token = secrets.token_urlsafe(32)
    if token_file is not None:
        token_file.write_text(token)
        os.chmod(token_file, 0o600)
        typer.echo(f"WHISPER_ENGINE_READY host={host} port={port} token_file={token_file}")
    else:
        typer.echo(f"WHISPER_ENGINE_READY host={host} port={port} token={token}")
    uvicorn.run(create_app(cases_root, token=token, tier=tier), host=host, port=port, log_level="warning")


@app.command(name="extract-embedding")
def extract_embedding(
    samples: Path = typer.Option(..., "--speaker-samples", "-s"),
    output: Path = typer.Option(..., "--output", "-o", help="Output .npy file."),
) -> None:
    """Compute and save a speaker embedding from reference samples."""
    import numpy as np

    from forensicaudio.stages import embedding

    emb = embedding.run(samples)
    np.save(output, emb)
    typer.echo(f"Saved embedding to {output}")


@app.command(name="inspect-log")
def inspect_log(log_path: Path = typer.Argument(...)) -> None:
    """Pretty-print a processing_log.json."""
    typer.echo(log_path.read_text())


if __name__ == "__main__":
    app()
