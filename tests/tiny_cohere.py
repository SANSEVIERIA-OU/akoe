# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Tiny randomly initialised Cohere ASR model (real transformers classes,
no downloads) saved to a folder the backend can load like a snapshot.

The tokenizer is a SentencePiece-style BPE (Metaspace "▁") with the special
tokens of the Cohere decoder prompt and a few merges that make whole
Italian words single tokens.
"""
from __future__ import annotations

from pathlib import Path

WORDS = ["ciao", "mondo", "vado", "casa", "sono", "qui", "città"]
PROMPT_SPECIALS = ["<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>", "<|it|>", "<|en|>",
                   "<|pnc|>", "<|nopnc|>", "<|noitn|>", "<|notimestamp|>", "<|nodiarize|>"]


def make_tokenizer(words: list[str] = WORDS):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab: dict[str, int] = {}
    for s in ["<unk>", "<|nospeech|>", "<pad>", "<|endoftext|>"] + PROMPT_SPECIALS:
        vocab[s] = len(vocab)
    alphabet = "▁abcdefghijklmnopqrstuvwxyzàèéìòù'.,?"
    for c in alphabet:
        vocab.setdefault(c, len(vocab))
    merges: list[tuple[str, str]] = []
    for w in words:
        cur = "▁"
        for ch in w:
            if cur + ch not in vocab:
                merges.append((cur, ch))
                vocab[cur + ch] = len(vocab)
            cur = cur + ch
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
    tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="<|endoftext|>",
        additional_special_tokens=["<|nospeech|>"] + PROMPT_SPECIALS,
    )


def make_cohere_dir(path: Path, seed: int = 0, n_mels: int = 16) -> Path:
    import torch
    from transformers import (
        CohereAsrConfig,
        CohereAsrFeatureExtractor,
        CohereAsrForConditionalGeneration,
        CohereAsrProcessor,
    )

    tok = make_tokenizer()
    cfg = CohereAsrConfig(
        encoder_config={"hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2,
                        "intermediate_size": 64, "subsampling_conv_channels": 8, "num_mel_bins": n_mels,
                        "max_position_embeddings": 512},
        vocab_size=len(tok), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        max_position_embeddings=128, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
        bos_token_id=tok.convert_tokens_to_ids("<|startoftranscript|>"),
        decoder_start_token_id=tok.convert_tokens_to_ids("<|startoftranscript|>"),
    )
    torch.manual_seed(seed)
    model = CohereAsrForConditionalGeneration(cfg).eval()
    model.generation_config.max_length = None
    proc = CohereAsrProcessor(feature_extractor=CohereAsrFeatureExtractor(feature_size=n_mels), tokenizer=tok)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    proc.save_pretrained(path)
    return path
