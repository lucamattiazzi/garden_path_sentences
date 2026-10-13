"""Exploratory role comprehension, answer-order controls and incremental surprisal.

python -m research.behavioral --validate-only
python -m research.behavioral --model MODEL --revision COMMIT_SHA --output results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Literal

from research.audit import summarize_items
from research.pilot import PILOT_PATH, Stimulus, load_pilot

Condition = Literal["gps", "normal", "role_reversed", "no_context"]
CONDITIONS: tuple[Condition, ...] = ("gps", "normal", "role_reversed", "no_context")
Scorer = Callable[[str, str], float]


def score_question(
    item: Stimulus, question_index: int, condition: Condition, scorer: Scorer
) -> dict[str, Any]:
    question = item.questions[question_index]
    gold_actor = 1 - question.actor if condition == "role_reversed" else question.actor
    orders = []
    for order in ((0, 1), (1, 0)):
        context = "" if condition == "no_context" else f"Sentence: {getattr(item, condition)}\n"
        prompt = (
            "Choose the participant that answers the question. Answer A or B.\n"
            f"{context}Question: {question.text}\n"
            f"A. {item.actors[order[0]]}\nB. {item.actors[order[1]]}\nAnswer:"
        )
        scores = [scorer(prompt, label) for label in ("A", "B")]
        if not all(math.isfinite(s) for s in scores):
            raise ValueError(f"Non-finite answer score: {item.id}, {condition}")
        gold_index = order.index(gold_actor)
        # -S(correct) - (-S(incorrect)); do not length-normalise A/B scores.
        margin = scores[1 - gold_index] - scores[gold_index]
        orders.append(
            {
                "prompt": prompt,
                "correct_label": ("A", "B")[gold_index],
                "surprisal_a": scores[0],
                "surprisal_b": scores[1],
                "correct_log_odds": margin,
            }
        )
    margins = [r["correct_log_odds"] for r in orders]
    return {
        "id": item.id,
        "family": item.family,
        "template": item.template,
        "question": question_index,
        "condition": condition,
        "orders": orders,
        "mean_correct_log_odds": mean(margins),
        "accuracy": mean(m > 0 for m in margins),
        "both_orders_correct": all(m > 0 for m in margins),
    }


def incremental_scores(item: Stimulus, scorer: Scorer) -> dict[str, Any]:
    """Score the critical word and each spillover word at matched surface positions."""
    result: dict[str, Any] = {
        "id": item.id,
        "family": item.family,
        "template": item.template,
        "words": [],
    }
    prefixes = {c: getattr(item, c + "_prefix") for c in ("gps", "normal")}
    for index, word in enumerate([item.critical_word, *item.spillover.split()]):
        scores = {c: scorer(prefix, word) for c, prefix in prefixes.items()}
        if not all(math.isfinite(s) for s in scores.values()):
            raise ValueError(f"Non-finite incremental score: {item.id}, {word}")
        result["words"].append(
            {
                "region": "critical" if index == 0 else "spillover",
                "word": word,
                "position": index,
                **scores,
                "delta": scores["gps"] - scores["normal"],
            }
        )
        prefixes = {c: prefix + " " + word for c, prefix in prefixes.items()}
    return result


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Average questions/orders within families, then bootstrap independent families.

    A family has two lexical realisations. Answer-order variants and questions
    are repeated measurements, not independent samples. Comparisons are paired.
    """
    members = {
        c: [
            (r["id"], r["question"], r["family"], r["template"])
            for r in rows
            if r["condition"] == c
        ]
        for c in CONDITIONS
    }
    if (
        not rows
        or sum(map(len, members.values())) != len(rows)
        or any(
            len(keys) != len(set(keys)) or set(keys) != set(members["gps"])
            for keys in members.values()
        )
    ):
        raise ValueError("Conditions must have identical, unique paired item/question observations")
    result = {}
    groups = {
        "overall": rows,
        **{
            t: [r for r in rows if r["template"] == t]
            for t in sorted({r["template"] for r in rows})
        },
    }
    for group, selected in groups.items():
        by_condition: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for condition in CONDITIONS:
            by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for r in selected:
                if r["condition"] == condition:
                    by_family[r["family"]].append(r)
            by_condition[condition] = by_family
        if not selected or any(set(c) != set(by_condition["gps"]) for c in by_condition.values()):
            raise ValueError("Each condition must contain the same nonempty families")
        result[group] = {}
        for condition, families in by_condition.items():
            result[group][condition] = {
                key: summarize_items(
                    [
                        {"id": f, "value": mean(r[key] for r in rs)}
                        for f, rs in sorted(families.items())
                    ],
                    "value",
                )
                for key in ("mean_correct_log_odds", "accuracy", "both_orders_correct")
            }
        changes = [
            {
                "id": f,
                "value": mean(r["mean_correct_log_odds"] for r in by_condition["gps"][f])
                - mean(r["mean_correct_log_odds"] for r in by_condition["normal"][f]),
            }
            for f in sorted(by_condition["gps"])
        ]
        result[group]["gps_minus_normal"] = summarize_items(changes, "value")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--revision", help="Full 40-character checkpoint commit SHA")
    parser.add_argument("--split", choices=("exploration", "validation"), default="exploration")
    parser.add_argument("--dtype", choices=("bfloat16", "float32", "float16"), default="bfloat16")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    items = load_pilot()
    if args.validate_only:
        print(
            f"{len(items)} pilot items, {len({p.family for p in items})} lexical families; structural checks passed."
        )
        print("Linguistic validation and training-corpus novelty: unverified.")
        return
    if not args.model or not args.revision or not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--model and an exact 40-character --revision are required")
    if args.output is None or args.output.exists():
        parser.error("--output must specify a new file; existing runs are never overwritten")

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from model_finder.surprisal import critical_token_positions, surprisal
    from model_finder.utils import get_device

    items = [p for p in items if p.split == args.split]
    device = args.device or get_device()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, local_files_only=args.local_files_only
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=args.revision,
            dtype=getattr(torch, args.dtype),
            local_files_only=args.local_files_only,
        )
        .to(device)
        .eval()
    )
    answer_tokens: dict[str, list[int]] = {}

    def scorer(prefix: str, word: str) -> float:
        if word in ("A", "B") and prefix.endswith("Answer:"):
            ids, positions = critical_token_positions(tokenizer, prefix, word)
            if len(positions) != 1:
                raise ValueError("A/B answer labels must each be a single token in this prompt")
            prefix_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
            if ids[: positions[0]] != prefix_ids:
                raise ValueError("Answer labels must share the unchanged prompt tokenization")
            answer_tokens[word] = [ids[p] for p in positions]
        return surprisal(model, tokenizer, prefix, word, device)

    rows, predictions = [], []
    for item in items:
        predictions.append(incremental_scores(item, scorer))
        for question_index in range(len(item.questions)):
            for condition in CONDITIONS:
                rows.append(score_question(item, question_index, condition, scorer))
        print(f"{item.id}: scored", flush=True)
    root = Path(__file__).resolve().parents[1]
    sources = (
        "research/behavioral.py",
        "research/pilot.py",
        "research/audit.py",
        "model_finder/surprisal.py",
    )
    payload = {
        "schema_version": 1,
        "status": "exploratory_pilot_not_linguistically_validated",
        "metadata": {
            "model": args.model,
            "revision": args.revision,
            "dtype": args.dtype,
            "device": device,
            "split": args.split,
            "n_items": len(items),
            "n_families": len({p.family for p in items}),
            "utc": datetime.now(UTC).isoformat(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "attention_implementation": model.config._attn_implementation,
            "platform": platform.platform(),
            "answer_tokens": answer_tokens,
            "dataset_sha256": hashlib.sha256(PILOT_PATH.read_bytes()).hexdigest(),
            "source_sha256": {
                f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in sources
            },
            "bootstrap_seed": 0,
            "bootstrap_draws": 10_000,
        },
        "prediction_probes": predictions,
        "question_results": rows,
        "summary": aggregate_results(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
