# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Tiny randomly initialised Whisper for fast tests (no downloads).

The tokenizer is a byte-level BPE with the Whisper special tokens in the
standard order (so language tokens follow <|startoftranscript|>) and a few
merges that turn chosen words into whole tokens.
"""
from __future__ import annotations

from pathlib import Path

DEFAULT_TOKENS = ["Ġciao", "Ġmondo", "Ġvado", "Ġcasa", "ĠRossi", "ĠBram", "billa", "Ġsono", "Ġqui"]


def make_tokenizer(tokens: list[str] = DEFAULT_TOKENS):
    from tokenizers import AddedToken
    from transformers import WhisperTokenizer
    from transformers.convert_slow_tokenizer import bytes_to_unicode
    from transformers.models.whisper.tokenization_whisper import LANGUAGES

    b2u = bytes_to_unicode()
    vocab: dict[str, int] = {}
    for b in range(256):
        vocab[b2u[b]] = len(vocab)
    merges: list[tuple[str, str]] = []
    for tok in tokens:
        cur = tok[0]
        for ch in tok[1:]:
            if cur + ch not in vocab:
                merges.append((cur, ch))
                vocab[cur + ch] = len(vocab)
            cur = cur + ch
    specials = (
        ["<|endoftext|>", "<|startoftranscript|>"]
        + [f"<|{code}|>" for code in LANGUAGES]
        + ["<|translate|>", "<|transcribe|>", "<|startoflm|>", "<|startofprev|>", "<|nospeech|>",
           "<|notimestamps|>"]
    )
    for s in specials:
        vocab[s] = len(vocab)
    tok = WhisperTokenizer(vocab=vocab, merges=merges)
    tok.add_special_tokens(
        {"additional_special_tokens": [AddedToken(s, special=True, normalized=False) for s in specials[1:]]}
    )
    tok.pad_token = tok.eos_token
    tok.set_prefix_tokens(language="italian", task="transcribe")
    return tok


def make_processor(tokens: list[str] = DEFAULT_TOKENS):
    from transformers import WhisperFeatureExtractor, WhisperProcessor

    return WhisperProcessor(feature_extractor=WhisperFeatureExtractor(feature_size=80), tokenizer=make_tokenizer(tokens))


def make_model(tokenizer, seed: int = 0, max_target_positions: int = 64):
    import torch
    from transformers import WhisperConfig, WhisperForConditionalGeneration
    from transformers.models.whisper.tokenization_whisper import LANGUAGES

    sot = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    cfg = WhisperConfig(
        vocab_size=len(tokenizer),
        num_mel_bins=80,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        max_source_positions=1500,
        max_target_positions=max_target_positions,
        decoder_start_token_id=sot,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.eos_token_id,
        begin_suppress_tokens=None,
        suppress_tokens=None,
    )
    torch.manual_seed(seed)
    model = WhisperForConditionalGeneration(cfg)
    gc = model.generation_config
    gc.lang_to_id = {f"<|{c}|>": tokenizer.convert_tokens_to_ids(f"<|{c}|>") for c in LANGUAGES}
    gc.task_to_id = {
        "transcribe": tokenizer.convert_tokens_to_ids("<|transcribe|>"),
        "translate": tokenizer.convert_tokens_to_ids("<|translate|>"),
    }
    gc.no_timestamps_token_id = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    gc.is_multilingual = True
    gc.begin_suppress_tokens = None
    gc.suppress_tokens = None
    gc.max_length = max_target_positions
    gc._from_model_config = False  # otherwise from_pretrained rebuilds it from the model config
    return model


def save_base(directory: Path, seed: int = 0):
    """Save a tiny base model + processor to ``directory`` (loadable with from_pretrained)."""
    processor = make_processor()
    model = make_model(processor.tokenizer, seed=seed)
    model.save_pretrained(str(directory))
    processor.save_pretrained(str(directory))
    return model, processor
