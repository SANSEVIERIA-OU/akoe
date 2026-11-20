# Akoe — notes for contributors (human or AI)

Read `MASTER_PLAN.md` (why) and `IMPLEMENTATION_PLAN.md` (what, phase by phase) first. Roadmap: GitHub milestones M0–M8 and issues on `SANSEVIERIA-OU/akoe`.

## Run
- Tests: `.venv/bin/python -m pytest -q` (pytest finds `src/` via `pyproject.toml`; in a git worktree use the main venv at `/workspace/WHISPER/.venv`).
- CLI: `.venv/bin/python -m akoe --help`.
- ML libraries (torch CPU, transformers, peft, speechbrain, faster-whisper, jiwer, imageio-ffmpeg) are installed in the main venv. No GPU on the dev machine.

## Invariants (do not break)
1. Deterministic and neural audio are never merged into one artifact; every transcribed segment carries `audio_provenance`.
2. Machine transcripts are `HypothesisTranscript` (always AI hypotheses, with alternatives and origins). The transcription belongs to the human.
3. Every file that matters gets a SHA-256 (`ArtifactRef`); evidence is imported read-only via `Case.import_evidence`.
4. No third-party SaaS: audio only on machines the user controls.
5. No synthesis or cloning of the subject's voice.
6. The voiceprint is a score, never an identification.
7. The transcriber's input is the original audio by default; cleaned audio is for human listening, or remixed with the original only if self-assessment shows it helps.

## Conventions
- Licence: SANSEVIERIA License 1.1 (`LICENSE`, `NOTICE`). Every new source file starts with `# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1`; new third-party libraries or models go into `THIRD_PARTY_NOTICES.md` with the licence read from their card.
- Model choice: the best available in the world, constrained only to machines the user controls (owner decision 2026-11-20).
- Code and identifiers in English; user-facing docs in Italian.
- Backends implement the protocols in `src/akoe/interfaces.py`; tests use the fakes in `tests/fakes.py` and never download models.
- Tests that need real models are marked `@pytest.mark.slow` and skipped by default.
- Never commit audio, case folders, model weights or `.env` files (see `.gitignore`).
- Log every parameter of a deterministic stage so it can be reproduced (SWGDE 20-A-001-2.0).
