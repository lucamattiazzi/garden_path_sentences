"""Detector library for the inverse-GPS screen.

Four detectors, mapping onto both this repo's machinery and the standard
paradigms in the literature:

D1  surprisal_delta      Δ-surprisal at the forced continuation:
                         S(target|trap_prefix) − S(target|control_prefix).
                         The same minimal-pair logic as the main pipeline's
                         01_baseline (and as van Schijndel & Linzen 2021 /
                         Arehalli et al. 2022 for human-style GPS).

D2  capture_score        logP(attractor|prefix) − logP(target|prefix),
                         computed on BOTH prefixes. A trap-specific pull is
                         capture(trap) ≫ capture(control). This is the
                         forced-choice/"wrong-continuation preference"
                         measure used in LM syntactic-evaluation suites
                         (BLiMP-style minimal-pair comparisons).

D3  greedy_continuation  Free continuation after the trap prefix; flags
                         whether the attractor string shows up. Analogue of
                         the main pipeline's predict steps (04-06, 09).

D4  choice_logprobs      Forced-choice comprehension probe: score answer
                         options after a Q/A prompt. LM version of the
                         Christianson et al. (2001) lingering-
                         misinterpretation paradigm, as applied to LMs by
                         Amouyal et al. (2025, ACL).

All log-probabilities are in nats and summed over subword tokens, via
model_finder.surprisal.surprisal (so tokenization handling is identical to
the rest of the repo).
"""

from __future__ import annotations

import torch

from inverse_gps.sentences import InvItem
from model_finder.surprisal import surprisal


def logprob(model, tokenizer, prefix: str, continuation: str, device: str) -> float:
    """log P(continuation | prefix), summed over subword tokens (nats)."""
    return -surprisal(model, tokenizer, prefix, continuation, device)


# ─── D1 ──────────────────────────────────────────────────────────────────────
def surprisal_delta(model, tokenizer, item: InvItem, device: str) -> dict:
    s_trap = surprisal(model, tokenizer, item.trap_prefix, item.target, device)
    s_control = surprisal(model, tokenizer, item.control_prefix, item.target, device)
    return {
        "s_trap": s_trap,
        "s_control": s_control,
        "delta": s_trap - s_control,
    }


# ─── D2 ──────────────────────────────────────────────────────────────────────
def capture_scores(model, tokenizer, item: InvItem, device: str) -> dict:
    """capture = logP(attractor) − logP(target), on trap and control prefixes.

    capture_trap > 0        model prefers the wrong continuation in the trap
    capture_trap − capture_control    trap-specificity of the pull
    """
    cap_trap = logprob(
        model, tokenizer, item.trap_prefix, item.attractor, device
    ) - logprob(model, tokenizer, item.trap_prefix, item.target, device)
    cap_control = logprob(
        model, tokenizer, item.control_prefix, item.attractor, device
    ) - logprob(model, tokenizer, item.control_prefix, item.target, device)
    return {
        "capture_trap": cap_trap,
        "capture_control": cap_control,
        "capture_specificity": cap_trap - cap_control,
    }


# ─── D3 ──────────────────────────────────────────────────────────────────────
def greedy_continuation(
    model, tokenizer, prefix: str, device: str, max_new_tokens: int = 10
) -> str:
    input_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)


def continuation_captured(continuation: str, attractor: str) -> bool:
    """True if the first word of the attractor appears in the continuation."""
    first = attractor.split()[0].lower().strip(".,;:!?")
    words = [w.lower().strip(".,;:!?") for w in continuation.split()]
    return first in words


# ─── D4 ──────────────────────────────────────────────────────────────────────
QA_PROMPT = (
    "Read the sentence and answer the question.\n"
    "Sentence: {sentence}\n"
    "Question: {question}\n"
    "Answer:"
)


def choice_logprobs(
    model, tokenizer, sentence: str, question: str, options: list[str], device: str
) -> list[float]:
    """log P(option | QA prompt) for each answer option (length-summed nats).

    Works on base models (no chat template): the prompt is a plain cloze
    frame and options are scored as continuations. Options should be
    length-matched within an item; we additionally report per-token
    normalisation downstream if needed.
    """
    prompt = QA_PROMPT.format(sentence=sentence, question=question)
    return [logprob(model, tokenizer, prompt, " " + opt, device) for opt in options]


def qa_probe(model, tokenizer, item: InvItem, device: str) -> dict:
    """Forced-choice comprehension on trap and control sentences.

    Which option is correct on the CONTROL sentence depends on the template:
    - ROLE: the control reverses the roles, so the trap sentence's
      'trap_answer' becomes the correct answer on the control.
    - IDIOM: the control keeps the literal scene (idiom noun replaced), so
      'correct_answer' stays correct on both sentences.
    """
    lp_correct_t, lp_trap_t = choice_logprobs(
        model, tokenizer, item.trap_sentence, item.question,
        [item.correct_answer, item.trap_answer], device,
    )
    lp_a_c, lp_b_c = choice_logprobs(
        model, tokenizer, item.control_sentence, item.question,
        [item.correct_answer, item.trap_answer], device,
    )
    if item.template == "ROLE":
        lp_correct_c, lp_wrong_c = lp_b_c, lp_a_c
    else:
        lp_correct_c, lp_wrong_c = lp_a_c, lp_b_c
    return {
        "trap_margin": lp_correct_t - lp_trap_t,      # >0 = answered correctly
        "control_margin": lp_correct_c - lp_wrong_c,  # >0 = answered correctly
        "trap_correct": lp_correct_t > lp_trap_t,
        "control_correct": lp_correct_c > lp_wrong_c,
    }
