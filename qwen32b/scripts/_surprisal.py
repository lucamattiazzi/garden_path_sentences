"""MLX-native surprisal computation for `Qwen/Qwen3-32B-MLX-8bit`.

Mirrors `model_finder.surprisal.surprisal` but operates on `mx.array`
instead of `torch.Tensor`. `mlx_lm.load()` returns a `TokenizerWrapper`
that proxies the underlying HF tokenizer; we re-use the existing
`critical_token_positions` helper because that one only needs the HF
tokenizer interface.

The 8-bit quantisation adds ~0.05-0.1 nats of noise to absolute surprisal
values, well below the per-pair signals we care about (1-2 nats).
"""

from __future__ import annotations

import time

import mlx.core as mx
from mlx_lm import load as mlx_load

from model_finder.surprisal import critical_token_positions


def _hf_tokenizer(maybe_wrapped):
    """Return the underlying HF AutoTokenizer from an mlx_lm TokenizerWrapper."""
    return getattr(maybe_wrapped, "_tokenizer", maybe_wrapped)


def load_model(model_name: str):
    """Return `(mlx_model, tokenizer_wrapper)`.

    The wrapper is what every other helper in this folder takes; pass it
    around unchanged.
    """
    print(f"  loading {model_name} (MLX-LM) ...", flush=True)
    t0 = time.time()
    model, tokenizer = mlx_load(model_name)
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)
    return model, tokenizer


def surprisal(model, tokenizer, prefix: str, critical_word: str) -> float:
    """−log P(critical_word | prefix), summed over subword tokens (nats)."""
    hf_tok = _hf_tokenizer(tokenizer)
    input_ids_list, critical_indices = critical_token_positions(
        hf_tok, prefix, critical_word
    )
    if not critical_indices:
        return float("nan")

    input_ids = mx.array([input_ids_list])
    logits = model(input_ids)  # (1, seq_len, vocab) for mlx_lm Qwen3
    if logits.ndim == 3:
        logits = logits[0]  # (seq_len, vocab)
    # Numerically stable log_softmax along vocab dim
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(log_probs)  # force evaluation once, then read scalars cheaply

    total = 0.0
    for idx in critical_indices:
        if idx == 0:
            continue  # no preceding context
        token_id = input_ids_list[idx]
        total += -log_probs[idx - 1, token_id].item()
    return total
