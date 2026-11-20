# SPDX-License-Identifier: LicenseRef-SANSEVIERIA-1.1
"""Tiny randomly initialised Qwen3-ASR and Qwen3-ForcedAligner models (real
transformers classes, no downloads) saved to folders the backend can load.

The tokenizer is a byte-level BPE with the Qwen chat / audio special tokens;
the chat template has the same structure as the one published with
Qwen/Qwen3-ASR-1.7B-hf (Apache-2.0): system turn, user turn with the audio
placeholders, optional assistant prefill.
"""
from __future__ import annotations

from pathlib import Path

WORDS = ["Ġciao", "Ġmondo", "Ġvado", "Ġcasa", "Ġsono", "Ġqui"]
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>",
            "<asr_text>", "<timestamp>"]
CHAT_TEMPLATE = (
    "{%- set ns = namespace(system_text='') -%}{%- for m in messages -%}{%- if m.role == 'system' -%}"
    "{%- for c in m.content -%}{%- if c.type == 'text' -%}{%- set ns.system_text = ns.system_text + c.text -%}"
    "{%- endif -%}{%- endfor -%}{%- endif -%}{%- endfor -%}"
    "{%- set ns2 = namespace(user='') -%}{%- for m in messages -%}{%- if m.role == 'user' -%}"
    "{%- for c in m.content -%}{%- if c.type == 'audio' or ('audio' in c) -%}"
    "{%- set ns2.user = ns2.user + '<|audio_start|><|audio_pad|><|audio_end|>' -%}"
    "{%- endif -%}{%- endfor -%}{%- endif -%}{%- endfor -%}"
    "{{- '<|im_start|>system\n' + ns.system_text + '<|im_end|>\n' -}}"
    "{{- '<|im_start|>user\n' + ns2.user + '<|im_end|>\n' -}}"
    "{%- for m in messages -%}{%- if m.role == 'assistant' -%}{{- '<|im_start|>assistant\n' -}}"
    "{%- for c in m.content -%}{{- c.text -}}{%- endfor -%}{%- endif -%}{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}"
)
# Same as the template of Qwen/Qwen3-ForcedAligner-0.6B-hf: audio, then the words separated by timestamp pairs.
ALIGNER_TEMPLATE = (
    "{%- set ns = namespace(audio_tokens='', words=[]) -%}{%- for m in messages -%}{%- if m.content is not string -%}"
    "{%- for c in m.content -%}{%- if c.type == 'audio' or ('audio' in c) or ('audio_url' in c) -%}"
    "{%- set ns.audio_tokens = ns.audio_tokens + '<|audio_start|><|audio_pad|><|audio_end|>' -%}{%- endif -%}"
    "{%- if c.type == 'text' and (c.text is defined) -%}{%- set ns.words = ns.words + [c.text] -%}{%- endif -%}"
    "{%- endfor -%}{%- endif -%}{%- endfor -%}"
    "{{- ns.audio_tokens + ns.words | join('<timestamp><timestamp>') + '<timestamp><timestamp>' -}}"
)


def make_tokenizer(words: list[str] = WORDS):
    from transformers import Qwen2Tokenizer
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    b2u = bytes_to_unicode()
    vocab: dict[str, int] = {}
    for b in range(256):
        vocab[b2u[b]] = len(vocab)
    merges: list[tuple[str, str]] = []
    for tok in words + ["language", "ĠItalian"]:
        cur = tok[0]
        for ch in tok[1:]:
            if cur + ch not in vocab:
                merges.append((cur, ch))
                vocab[cur + ch] = len(vocab)
            cur = cur + ch
    for s in SPECIALS:
        vocab[s] = len(vocab)
    from tokenizers import AddedToken

    tok = Qwen2Tokenizer(
        vocab=vocab, merges=merges, eos_token="<|im_end|>", pad_token="<|endoftext|>",
        extra_special_tokens={"audio_token": "<|audio_pad|>", "audio_bos_token": "<|audio_start|>",
                              "audio_eos_token": "<|audio_end|>"},
    )
    tok.add_tokens([AddedToken(t, special=True, normalized=False) for t in SPECIALS], special_tokens=True)
    return tok


def _config(tok, num_labels: int | None = None):
    from transformers import Qwen3ASRConfig

    cfg = Qwen3ASRConfig(
        audio_config={"d_model": 32, "encoder_layers": 1, "encoder_attention_heads": 2, "encoder_ffn_dim": 64,
                      "downsample_hidden_size": 8, "output_dim": 32},
        text_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1, "num_attention_heads": 2,
                     "num_key_value_heads": 1, "head_dim": 16, "vocab_size": len(tok),
                     "max_position_embeddings": 4096},
        audio_token_id=tok.convert_tokens_to_ids("<|audio_pad|>"),
        timestamp_token_id=tok.convert_tokens_to_ids("<timestamp>"),
        pad_token_id=tok.convert_tokens_to_ids("<|im_end|>"),
        eos_token_id=[tok.convert_tokens_to_ids("<|endoftext|>"), tok.convert_tokens_to_ids("<|im_end|>")],
    )
    if num_labels is not None:
        cfg.num_labels = num_labels
    return cfg


def _save(model, tok, path: Path, template: str = CHAT_TEMPLATE) -> Path:
    from transformers import Qwen3ASRFeatureExtractor, Qwen3ASRProcessor

    proc = Qwen3ASRProcessor(feature_extractor=Qwen3ASRFeatureExtractor(), tokenizer=tok, chat_template=template)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    proc.save_pretrained(path)
    return path


def make_qwen3_asr_dir(path: Path, seed: int = 0) -> Path:
    import torch
    from transformers import Qwen3ASRForConditionalGeneration

    tok = make_tokenizer()
    torch.manual_seed(seed)
    model = Qwen3ASRForConditionalGeneration(_config(tok)).eval()
    model.generation_config.max_length = None
    model.generation_config.eos_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.pad_token_id
    return _save(model, tok, path)


def make_qwen3_aligner_dir(path: Path, seed: int = 0, num_labels: int = 64) -> Path:
    import torch
    from transformers import Qwen3ASRForTokenClassification

    tok = make_tokenizer()
    torch.manual_seed(seed)
    model = Qwen3ASRForTokenClassification(_config(tok, num_labels=num_labels)).eval()
    return _save(model, tok, path, ALIGNER_TEMPLATE)
