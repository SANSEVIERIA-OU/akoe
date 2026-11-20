# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Meta omniASR CTC v2 candidate scorer for "did he say X or Y?" (issue #58).

Model
-----
omniASR CTC v2 (Omnilingual ASR, Meta FAIR, December 2025; Apache-2.0) is a
wav2vec 2.0 encoder with a CTC head over a 10288-entry *character* vocabulary
(written form: upper and lower case, punctuation, digits; 1600+ languages).
The official repository runs it with fairseq2, which does not support the
installed PyTorch. Here it runs with the installed stack:

- the **official checkpoint** from Meta (`dl.fbaipublicfiles.com/mms/...pt`,
  the file referenced by `cards/models/rc_models_v2.yaml` of
  facebookresearch/omnilingual-asr) is memory-mapped and read with
  `torch.load(weights_only=True)` - it holds only tensors (checked by
  disassembling its pickle) - after its SHA-256 is verified;
- its fairseq2 parameter names are mapped one to one onto the transformers
  `Wav2Vec2ForCTC` with pre-norm ("stable layer norm") layers: the
  architecture is the one of XLS-R (`large_lv60k` / `1b` / `3b` of
  `wav2vec2_ssl/config.py`: conv feature extractor with layer norm and bias,
  convolutional position encoder 128/16, pre-LN Transformer, final layer norm,
  linear CTC head). Every checkpoint tensor must find its slot and vice versa
  (strict load);
- the official SentencePiece tokenizer `omniASR_tokenizer_written_v2.model`
  gives the vocabulary; the CTC blank is index 0 (`<s>`, the default of
  `torch.nn.functional.ctc_loss` used by fairseq2);
- input: 16 kHz mono, normalized to zero mean and unit variance over the
  utterance (`layer_norm` over the waveform, as `apply_audio_normalization`
  in the omniASR pipeline), at most 40 s (`MAX_ALLOWED_AUDIO_SEC`).

The community safetensors conversion `gabrione/omniasr-ctc-v2-safetensors`
(bf16/int8/int4 of the same checkpoints, uploaded 2026-11-20) was not used:
the official fp32 file keeps the chain of custody on Meta's original bytes.

Scoring
-------
Candidates are force-aligned token by token (one token per character of the
10288 vocabulary) with the Viterbi pass of `ctc_scorer.ctc_viterbi`. Because
the model writes case and punctuation that an operator's candidate does not
specify, the emissions are **folded** before the alignment: each letter's
column is the log-sum-exp of its lower- and upper-case tokens, and the blank
column is the log-sum-exp of blank, punctuation (except the apostrophe, which
is part of Italian words) and the unused special tokens. In words: "this
letter in any case" and "no letter at this frame". The free (unconstrained)
path used for the relative score is computed on the same folded matrix, so
the score keeps the meaning of `ctc_scorer` (0 = what the model would decode
freely, always <= 0). Digits are kept: the model may write numbers in digits,
so both forms may be worth trying.
"""
from __future__ import annotations

import os
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

from akoe.neural import (
    BackendUnavailable,
    ModelProvenance,
    download_verified,
    hash_files,
    resolve_device,
)
from akoe.transcribe.ctc_scorer import CTCForcedAlignmentScorer

OMNIASR_BASE_URL = "https://dl.fbaipublicfiles.com/mms/"
OMNIASR_CODE = "facebookresearch/omnilingual-asr@81f51e224ce9e74b02cc2a3eaf21b2d91d743455"
OMNIASR_LICENSE = "apache-2.0 (LICENSE of facebookresearch/omnilingual-asr; checkpoints from its model cards)"
TOKENIZER_FILE = "omniASR_tokenizer_written_v2.model"
VOCAB_SIZE = 10288
MAX_AUDIO_SECONDS = 40.0  # omniASR pipeline: MAX_ALLOWED_AUDIO_SEC

# Architectures from omnilingual-asr `wav2vec2_asr/config.py` (1b_v2, 3b_v2,
# 300m_v2) and `wav2vec2_ssl/config.py` (encoders 1b, 3b; large_lv60k from fairseq2).
OMNIASR_MODELS: dict[str, dict[str, Any]] = {
    "omniASR_CTC_300M_v2": {"file": "omniASR-CTC-300M-v2.pt", "bytes": 1304065508,
                            "hidden_size": 1024, "layers": 24, "heads": 16, "ffn": 4096},
    "omniASR_CTC_1B_v2": {"file": "omniASR-CTC-1B-v2.pt", "bytes": 3902956068,
                          "hidden_size": 1280, "layers": 48, "heads": 16, "ffn": 5120},
    "omniASR_CTC_3B_v2": {"file": "omniASR-CTC-3B-v2.pt", "bytes": 12325920624,
                          "hidden_size": 2048, "layers": 60, "heads": 16, "ffn": 8192},
}
DEFAULT_OMNIASR_MODEL = "omniASR_CTC_1B_v2"
WORKSTATION_OMNIASR_MODEL = "omniASR_CTC_3B_v2"

# SHA-256 of the files downloaded from OMNIASR_BASE_URL on 2026-11-20 (Meta
# publishes no digest). The 3B checkpoint (12.3 GB) was not downloaded on the
# development machine: its hash must be verified and added on first install on
# a workstation; until then it is refused unless allow_unknown_weights=True.
OMNIASR_KNOWN_SHA256: dict[str, str] = {
    TOKENIZER_FILE: "8aa11a1092142ef472537476ef6e76541123e2f0d789b79f3ebd119008240b1e",
    "omniASR-CTC-1B-v2.pt": "354f981756aa8f41591ea363e45b9c4eba1ec5144c2273af82e747efbb08919c",
}


def default_cache_dir() -> Path:
    return Path(os.environ.get("AKOE_MODEL_CACHE", Path.home() / ".cache" / "akoe")) / "omniasr"


def weights_pinned(model_card: str) -> bool:
    """True when the checkpoint of `model_card` has a verified SHA-256 here."""
    spec = OMNIASR_MODELS.get(model_card)
    return spec is not None and spec["file"] in OMNIASR_KNOWN_SHA256


def is_omniasr_model(model_id: str) -> bool:
    return model_id in OMNIASR_MODELS


# --------------------------------------------------------------------------
# fairseq2 -> transformers parameter names
# --------------------------------------------------------------------------

_LAYER_MAP = {
    "self_attn_layer_norm": "layer_norm",
    "self_attn.q_proj": "attention.q_proj",
    "self_attn.k_proj": "attention.k_proj",
    "self_attn.v_proj": "attention.v_proj",
    "self_attn.output_proj": "attention.out_proj",
    "ffn_layer_norm": "final_layer_norm",
    "ffn.inner_proj": "feed_forward.intermediate_dense",
    "ffn.output_proj": "feed_forward.output_dense",
}


def fairseq2_to_hf_key(key: str, pos_conv_g: str, pos_conv_v: str) -> str:
    """Name of a fairseq2 `Wav2Vec2AsrModel` parameter in transformers
    `Wav2Vec2ForCTC` (stable layer norm). Raises KeyError for unknown names."""
    p = "encoder_frontend.feature_extractor.layers."
    if key.startswith(p):
        return "wav2vec2.feature_extractor.conv_layers." + key[len(p):]
    fixed = {
        "encoder_frontend.post_extract_layer_norm.weight": "wav2vec2.feature_projection.layer_norm.weight",
        "encoder_frontend.post_extract_layer_norm.bias": "wav2vec2.feature_projection.layer_norm.bias",
        "encoder_frontend.model_dim_proj.weight": "wav2vec2.feature_projection.projection.weight",
        "encoder_frontend.model_dim_proj.bias": "wav2vec2.feature_projection.projection.bias",
        "encoder_frontend.pos_encoder.conv.bias": "wav2vec2.encoder.pos_conv_embed.conv.bias",
        "encoder_frontend.pos_encoder.conv.weight_g": pos_conv_g,
        "encoder_frontend.pos_encoder.conv.weight_v": pos_conv_v,
        "encoder.layer_norm.weight": "wav2vec2.encoder.layer_norm.weight",
        "encoder.layer_norm.bias": "wav2vec2.encoder.layer_norm.bias",
        "final_proj.weight": "lm_head.weight",
        "final_proj.bias": "lm_head.bias",
    }
    if key in fixed:
        return fixed[key]
    if key.startswith("encoder.layers."):
        idx, rest = key[len("encoder.layers."):].split(".", 1)
        for src, dst in _LAYER_MAP.items():
            if rest.startswith(src + "."):
                return f"wav2vec2.encoder.layers.{idx}.{dst}{rest[len(src):]}"
    raise KeyError(key)


def map_state_dict(state: dict[str, Any], hf_keys: set[str]) -> dict[str, Any]:
    """Map a fairseq2 state dict onto `hf_keys`; every key on both sides must match."""
    g = next((k for k in hf_keys if k.endswith("pos_conv_embed.conv.parametrizations.weight.original0")),
             "wav2vec2.encoder.pos_conv_embed.conv.weight_g")
    v = g.replace("original0", "original1") if "original0" in g else "wav2vec2.encoder.pos_conv_embed.conv.weight_v"
    out: dict[str, Any] = {}
    for k, t in state.items():
        out[fairseq2_to_hf_key(k, g, v)] = t
    missing, extra = hf_keys - set(out), set(out) - hf_keys
    if missing or extra:
        raise RuntimeError(f"Conversione omniASR incompleta: mancano {sorted(missing)[:5]}, "
                           f"in più {sorted(extra)[:5]}")
    return out


# --------------------------------------------------------------------------
# Folding of the written-form vocabulary
# --------------------------------------------------------------------------


def build_folding(pieces: list[str], blank_id: int = 0) -> tuple[np.ndarray, dict[str, int], int, dict[str, Any]]:
    """Map every vocabulary id to a folded class.

    Returns (class index per id, {character: class} for the alignment
    alphabet, blank class, summary). Class 0 is the folded blank: the CTC
    blank, special tokens (`<...>`) and punctuation except the apostrophe.
    Upper-case letters join the class of their lower-case form.
    """
    piece_id = {p: i for i, p in enumerate(pieces)}
    cls = np.full(len(pieces), -1, dtype=np.int64)
    blank_ids: list[int] = []
    for i, p in enumerate(pieces):
        is_special = p.startswith("<") and p.endswith(">") and len(p) > 2
        is_punct = len(p) == 1 and unicodedata.category(p).startswith("P") and p != "'"
        if i == blank_id or is_special or is_punct:
            cls[i] = 0
            blank_ids.append(i)
    next_cls = 1
    alphabet: dict[str, int] = {}
    for i, p in enumerate(pieces):  # lower-case (and caseless) first
        if cls[i] >= 0:
            continue
        low = p.lower()
        if low != p and len(low) == 1 and low in piece_id:
            continue
        cls[i] = next_cls
        alphabet[p] = next_cls
        next_cls += 1
    n_case = 0
    for i, p in enumerate(pieces):
        if cls[i] < 0:
            cls[i] = cls[piece_id[p.lower()]]
            alphabet[p] = int(cls[i])
            n_case += 1
    summary = {"n_tokens": len(pieces), "n_classes": next_cls, "blank_class_tokens": len(blank_ids),
               "case_folded_tokens": n_case,
               "rule": "class = lower-case letter (upper case folded); blank class = blank + <special> + "
                       "punctuation except apostrophe; scores are log-sum-exp within a class"}
    return cls, alphabet, 0, summary


def fold_log_probs(lp: np.ndarray, cls: np.ndarray, n_classes: int) -> np.ndarray:
    """(T x V) natural-log probabilities -> (T x n_classes), log-sum-exp per class."""
    lp = np.asarray(lp, dtype=np.float64)
    m = lp.max(axis=1, keepdims=True)
    p = np.exp(lp - m)
    out = np.zeros((lp.shape[0], n_classes), dtype=np.float64)
    np.add.at(out.T, cls, p.T)
    with np.errstate(divide="ignore"):
        return (np.log(out) + m).astype(np.float32)


# --------------------------------------------------------------------------
# Scorer
# --------------------------------------------------------------------------


class OmniASRCTCScorer(CTCForcedAlignmentScorer):
    """`CandidateScorer` over omniASR CTC v2 (see module docstring).

    `dtype="bfloat16"` halves the memory on CPU at some cost in precision
    (recorded in `provenance`); the default is float32.
    """

    max_seconds = MAX_AUDIO_SECONDS

    def __init__(
        self,
        model_card: str = DEFAULT_OMNIASR_MODEL,
        *,
        device: str = "cpu",
        dtype: str = "float32",
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        allow_unknown_weights: bool = False,
    ) -> None:
        if model_card not in OMNIASR_MODELS:
            raise ValueError(f"Modello omniASR sconosciuto: {model_card!r} ({', '.join(OMNIASR_MODELS)})")
        super().__init__(model_card, device=device, local_files_only=local_files_only)
        self.model_card = model_card
        self.dtype = dtype
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
        self.allow_unknown_weights = allow_unknown_weights
        self.name = f"{model_card}/ctc-forced-alignment"
        self._delim = " "

    def _fetch(self, filename: str, expected_bytes: int | None = None) -> tuple[Path, str]:
        path = self.cache_dir / filename
        known = OMNIASR_KNOWN_SHA256.get(filename)
        if known is None and not self.allow_unknown_weights:
            raise BackendUnavailable(
                f"Lo SHA-256 di {filename} non è ancora fissato in questo progetto: il file non viene "
                "usato. Scaricarlo su una macchina controllata, verificarne l'hash e aggiungerlo a "
                "OMNIASR_KNOWN_SHA256, oppure consentire esplicitamente pesi non verificati "
                "(allow_unknown_weights=True; l'hash viene comunque registrato)."
            )
        if not path.exists():
            if self.local_files_only:
                raise BackendUnavailable(f"{filename} non è presente in {self.cache_dir} (modalità offline).")
            download_verified(OMNIASR_BASE_URL + filename, path, known, allow_unknown=self.allow_unknown_weights)
        sha = hash_files([path])[path.name]
        if known is not None and sha != known:
            raise RuntimeError(
                f"{filename} ha SHA-256 {sha}, diverso da quello verificato ({known}): non viene usato."
            )
        if expected_bytes is not None and path.stat().st_size != expected_bytes:
            raise RuntimeError(f"{filename}: dimensione inattesa ({path.stat().st_size} byte).")
        return path, sha

    def _load(self) -> None:
        try:
            import sentencepiece as spm
            import torch
            from transformers import Wav2Vec2Config, Wav2Vec2ForCTC
        except ImportError as e:  # pragma: no cover
            raise BackendUnavailable(
                "Il punteggio omniASR richiede PyTorch, transformers e sentencepiece: installare gli extra 'ml'."
            ) from e
        spec = OMNIASR_MODELS[self.model_card]
        device = resolve_device(self.device)
        tdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[self.dtype]
        tok_path, tok_sha = self._fetch(TOKENIZER_FILE)
        ckpt_path, ckpt_sha = self._fetch(spec["file"], spec["bytes"])
        sp = spm.SentencePieceProcessor(model_file=str(tok_path))
        pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
        if len(pieces) != VOCAB_SIZE:
            raise RuntimeError(f"Tokenizer omniASR inatteso: {len(pieces)} simboli invece di {VOCAB_SIZE}.")
        config = Wav2Vec2Config(
            vocab_size=VOCAB_SIZE, hidden_size=spec["hidden_size"], num_hidden_layers=spec["layers"],
            num_attention_heads=spec["heads"], intermediate_size=spec["ffn"],
            feat_extract_norm="layer", conv_bias=True, do_stable_layer_norm=True,
            num_conv_pos_embeddings=128, num_conv_pos_embedding_groups=16,
            hidden_act="gelu", feat_extract_activation="gelu", layer_norm_eps=1e-5,
            mask_time_prob=0.0, mask_feature_prob=0.0, layerdrop=0.0, hidden_dropout=0.0,
            attention_dropout=0.0, activation_dropout=0.0, feat_proj_dropout=0.0, final_dropout=0.0,
            pad_token_id=sp.pad_id(), bos_token_id=sp.bos_id(), eos_token_id=sp.eos_id(),
        )
        config._attn_implementation = "sdpa"
        raw = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=True)
        with torch.device("meta"):
            model = Wav2Vec2ForCTC(config)
        state = map_state_dict(raw["model"], set(model.state_dict().keys()))
        model.load_state_dict(state, strict=True, assign=True)
        model = model.to(device=device, dtype=tdtype).eval()
        cls, alphabet, blank_cls, fold_summary = build_folding(pieces, blank_id=0)
        n_classes = int(cls.max()) + 1

        def emit(a16: np.ndarray) -> np.ndarray:
            x = torch.from_numpy(np.ascontiguousarray(a16, dtype=np.float32))
            x = torch.nn.functional.layer_norm(x, x.shape)[None]
            with torch.inference_mode():
                logits = model(x.to(device=device, dtype=tdtype)).logits[0].float()
            lp = torch.log_softmax(logits, dim=-1).cpu().numpy()
            return fold_log_probs(lp, cls, n_classes)

        self._emit = emit
        self._vocab = alphabet
        self._blank = blank_cls
        self._delim = " "
        if " " not in alphabet:
            raise RuntimeError("Il vocabolario omniASR non contiene lo spazio tra le parole.")
        self._provenance = ModelProvenance(
            backend=self.name, model_id=f"facebook/{self.model_card}", revision=None,
            weights={spec["file"]: ckpt_sha, TOKENIZER_FILE: tok_sha}, device=device, dtype=self.dtype,
            license=OMNIASR_LICENSE,
            extra={
                "checkpoint_url": OMNIASR_BASE_URL + spec["file"], "tokenizer_url": OMNIASR_BASE_URL + TOKENIZER_FILE,
                "code_reference": OMNIASR_CODE, "runtime": "transformers Wav2Vec2ForCTC (stable layer norm), "
                "weights mapped from fairseq2 names, strict load", "attn_implementation": "sdpa",
                "architecture": {k: spec[k] for k in ("hidden_size", "layers", "heads", "ffn")},
                "vocab_size": VOCAB_SIZE, "blank_id": 0, "word_delimiter": " ",
                "input_normalization": "layer_norm over the utterance waveform (zero mean, unit variance)",
                "max_seconds": self.max_seconds, "folding": fold_summary,
            },
        )
