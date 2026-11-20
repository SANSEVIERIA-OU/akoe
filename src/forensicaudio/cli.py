from pathlib import Path

import typer

from forensicaudio import pipeline
from forensicaudio.config import CaseConfig, Mode

app = typer.Typer(help="Forensic audio enhancement and target-speaker extraction.")


@app.command()
def process(
    evidence: Path = typer.Option(..., "--evidence", "-e", help="Path to the audio evidence file."),
    samples: Path = typer.Option(..., "--speaker-samples", "-s", help="Directory with reference voice samples of the target."),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory for artifacts."),
    mode: Mode = typer.Option(Mode.FORENSIC, "--mode", "-m", help="Processing mode."),
    case_id: str = typer.Option("case_001", "--case-id"),
) -> None:
    """Run the full pipeline on a case."""
    cfg = CaseConfig(
        case_id=case_id,
        evidence_path=evidence,
        speaker_samples_dir=samples,
        output_dir=output,
        mode=mode,
    )
    pipeline.process(cfg)
    typer.echo(f"Done. Artifacts in {output}")


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
