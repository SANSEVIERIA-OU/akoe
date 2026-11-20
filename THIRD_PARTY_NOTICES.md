# Third-party notices

Akoe's own code is under [SANSEVIERIA License 1.1](LICENSE). The components below keep their own licences. Models are downloaded at run time (pinned by revision or SHA-256 where the backend supports it) and are not stored in this repository. Licences were read from the model cards and repositories in November 2026; check them again before each release.

**Owner decision (2026-11-20).** Weights published without a licence of their own, inside or alongside a repository with a declared licence, are treated as covered by the repository's licence (e.g. ReDimNet weights under the repository's MIT licence).

## Python libraries

| Component | Use | Licence |
| --- | --- | --- |
| typer, pydantic, noisereduce, faster-whisper, fastapi | CLI, data models, stationary noise reduction, Whisper inference (CTranslate2), engine API | MIT |
| numpy, scipy, scikit-learn, threadpoolctl, soundfile, uvicorn, httpx, torch | numerics, DSP, NMF and calibration, BLAS threads, audio I/O, API server and client, neural inference | BSD-3-Clause |
| librosa | audio analysis | ISC |
| torchaudio, imageio-ffmpeg (Python package) | audio ops, ffmpeg locator | BSD-2-Clause |
| transformers, peft, accelerate, speechbrain, sentencepiece, jiwer, python-multipart | neural models, LoRA adaptation, speaker models, tokenisation, WER cross-check, uploads | Apache-2.0 |
| nemo-toolkit (optional, separate venv) | Parakeet / NeMo models | Apache-2.0 |
| ffmpeg binary shipped by imageio-ffmpeg | codec simulation and detection, run as a separate program | LGPL/GPL (per the build); not linked or distributed by Akoe |

## Speech models (downloaded, not stored)

| Model | Use | Licence | Source |
| --- | --- | --- | --- |
| OpenAI Whisper (tiny … large-v3, turbo) | transcription with alternatives, teacher-forced scoring, adaptation | MIT (repository); card of large-v3 says Apache-2.0 | https://github.com/openai/whisper |
| NVIDIA Parakeet-TDT 0.6B v3 (revision `541d1f99`) | transcription with entropy confidence | CC-BY-4.0 | https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 |
| jonatasgrosman/wav2vec2-large-xlsr-53-italian | CTC forced alignment and candidate scoring | Apache-2.0 | https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-italian |
| SpeechBrain ECAPA-TDNN (spkrec-ecapa-voxceleb) | speaker embedding (reference backend) | Apache-2.0 | https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb |
| ReDimNet (IDRnD) | speaker embedding (recommended backend) | MIT (repository licence; see owner decision above) | https://github.com/IDRnD/ReDimNet |
| CohereLabs/cohere-transcribe-03-2026 (revision `d552dadc`) | transcription (ensemble); gated on Hugging Face: requires accepting Cohere's terms and an HF token | Apache-2.0 | https://huggingface.co/CohereLabs/cohere-transcribe-03-2026 |
| Qwen/Qwen3-ASR-1.7B-hf (revision `bcd2b5b7`) | transcription (ensemble) | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf |
| Qwen3-ForcedAligner-0.6B-hf (revision `c07281df`) | word timestamps for Qwen3-ASR | Apache-2.0 | https://huggingface.co/Qwen |
| NVIDIA Nemotron 3.5 Transcribe 30B-A4B (revision `c1766764`) | transcription on the CLOUD tier (GPU ≥ 78 GB) | OpenMDW-1.1 | https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B |
| facebook/wav2vec2-base-10k-voxpopuli-ft-it | tests only | CC-BY-NC-4.0 | https://huggingface.co/facebook/wav2vec2-base-10k-voxpopuli-ft-it |

Models added later are listed here when their backend is merged, with the licence read from the card.
