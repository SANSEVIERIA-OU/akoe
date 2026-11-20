"""Tiny, randomly initialized models built on the fly for fast tests (no downloads).

They exercise the real loading/decoding code of the backends; their outputs are
meaningless text, so tests check structure and invariants, not content.
"""
from __future__ import annotations

from pathlib import Path

SPECIALS = ["<|endoftext|>", "<|startoftranscript|>", "<|it|>", "<|transcribe|>", "<|notimestamps|>"]


def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def make_tiny_whisper(root: Path, name: str = "tiny-whisper", seed: int = 0) -> Path:
    """Save a 1-layer Whisper with a byte-level tokenizer (one token per byte,
    no merges) and the special tokens the backend needs. Returns the directory."""
    import torch
    from transformers import (
        GenerationConfig,
        WhisperConfig,
        WhisperFeatureExtractor,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        WhisperTokenizer,
    )

    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    vocab = {ch: i for i, ch in enumerate(_bytes_to_unicode().values())}
    tok = WhisperTokenizer(
        vocab=vocab, merges=[], unk_token="<|endoftext|>", bos_token="<|endoftext|>",
        eos_token="<|endoftext|>", pad_token="<|endoftext|>",
    )
    tok.add_special_tokens({"additional_special_tokens": SPECIALS[1:]})
    ids = {t: tok.convert_tokens_to_ids(t) for t in SPECIALS}
    assert ids["<|notimestamps|>"] == len(tok) - 1  # last: nothing after it gets suppressed
    torch.manual_seed(seed)
    cfg = WhisperConfig(
        vocab_size=len(tok), num_mel_bins=80, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32,
        d_model=16, max_source_positions=1500, max_target_positions=448,
        decoder_start_token_id=ids["<|startoftranscript|>"], pad_token_id=ids["<|endoftext|>"],
        eos_token_id=ids["<|endoftext|>"], bos_token_id=ids["<|endoftext|>"],
    )
    model = WhisperForConditionalGeneration(cfg)
    model.generation_config = GenerationConfig(
        decoder_start_token_id=ids["<|startoftranscript|>"], eos_token_id=ids["<|endoftext|>"],
        pad_token_id=ids["<|endoftext|>"], no_timestamps_token_id=ids["<|notimestamps|>"],
        alignment_heads=[[0, 0], [0, 1]], suppress_tokens=[], begin_suppress_tokens=[ids["<|endoftext|>"]],
        max_length=448,
    )
    model.save_pretrained(str(d))
    WhisperProcessor(feature_extractor=WhisperFeatureExtractor(), tokenizer=tok).save_pretrained(str(d))
    return d


def make_lora_adapter(model_dir: Path, out: Path, seed: int = 1) -> Path:
    """LoRA adapter (non-zero init, so it changes the model) saved with peft
    `save_pretrained`, as the training code of M4 does."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration

    torch.manual_seed(seed)
    base = WhisperForConditionalGeneration.from_pretrained(str(model_dir))
    cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], init_lora_weights=False)
    peft_model = get_peft_model(base, cfg)
    peft_model.save_pretrained(str(out))
    return out


def make_tiny_wav2vec2_ctc(root: Path, name: str = "tiny-w2v2-ctc", seed: int = 0) -> Path:
    """1-layer wav2vec2 CTC with an Italian character vocabulary. Its config
    carries a pad_token_id that disagrees with vocab.json on purpose (as in
    facebook/wav2vec2-base-10k-voxpopuli-ft-it): the blank must come from the vocab."""
    import json

    import torch
    from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "'": 5}
    for c in "abcdefghilmnopqrstuvzàèéìòù":
        vocab[c] = len(vocab)
    torch.manual_seed(seed)
    cfg = Wav2Vec2Config(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, conv_dim=(16, 16), conv_stride=(5, 64), conv_kernel=(10, 64),
        num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2, pad_token_id=1,
    )
    Wav2Vec2ForCTC(cfg).save_pretrained(str(d))
    Wav2Vec2FeatureExtractor(do_normalize=True).save_pretrained(str(d))
    (d / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    (d / "tokenizer_config.json").write_text(json.dumps({"pad_token": "<pad>", "word_delimiter_token": "|"}))
    return d
