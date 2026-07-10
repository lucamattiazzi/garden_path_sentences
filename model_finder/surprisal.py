import time

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_finder.models import ModelSpec

load_dotenv()

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def load_model(
    device: str,
    spec: ModelSpec,
):
    dtype = _DTYPES[spec.dtype]
    print(f"  loading {spec.hf_name} (dtype={spec.dtype}) ...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_name,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)
    return model, tokenizer


def critical_token_positions(
    tokenizer, prefix: str, critical_word: str
) -> tuple[list[int], list[int]]:
    """
    Tokenise (prefix + ' ' + critical_word) and return:
        input_ids           — list[int], the full sequence with special tokens
        critical_indices    — list[int], positions in input_ids that correspond
                              to the critical word's subword tokens

    Strategy:
      1. Prefer offset_mapping (fast tokenizers) — works for all major models.
      2. Fallback: tokenize prefix alone and assume prefix-property holds.
    """
    full_text = prefix + " " + critical_word
    crit_char_start = len(prefix) + 1  # after the inserted space
    crit_char_end = crit_char_start + len(critical_word)

    try:
        enc = tokenizer(
            full_text,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        input_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        critical_indices = [
            i
            for i, (a, b) in enumerate(offsets)
            if a >= crit_char_start and b <= crit_char_end and b > a
        ]
        if critical_indices:
            return input_ids, critical_indices
    except (TypeError, NotImplementedError, KeyError):
        pass  # slow tokenizer — fall through

    # Fallback: compare with prefix-only tokenization
    prefix_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=True)["input_ids"]
    if full_ids[: len(prefix_ids)] != prefix_ids:
        return full_ids, []
    critical_indices = list(range(len(prefix_ids), len(full_ids)))
    return full_ids, critical_indices


def surprisal(model, tokenizer, prefix: str, critical_word: str, device: str) -> float:
    """−log P(critical_word | prefix), summed over subword tokens (nats)."""
    input_ids_list, critical_indices = critical_token_positions(
        tokenizer, prefix, critical_word
    )
    if not critical_indices:
        return float("nan")

    input_ids = torch.tensor([input_ids_list], device=device)
    with torch.no_grad():
        logits = model(input_ids).logits[0]  # (seq_len, vocab)
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    total = 0.0
    for idx in critical_indices:
        if idx == 0:
            continue  # no preceding context
        token_id = input_ids_list[idx]
        total += -log_probs[idx - 1, token_id].item()
    return total
