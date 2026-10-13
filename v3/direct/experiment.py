"""One frozen direct-answer instrument, development families only."""

import itertools
from collections import defaultdict

import numpy as np

from v3.bridge.materials import items, sentence

DEMONSTRATIONS = """Sentence: The student admired the scientist.
Question: Who admired someone?
Answer: the student.

Sentence: The student admired the scientist.
Question: Who was admired?
Answer: the scientist.

Sentence: The child was rescued by the firefighter.
Question: Who rescued someone?
Answer: the firefighter.

Sentence: The child was rescued by the firefighter.
Question: Who was rescued?
Answer: the child.

"""


def materials() -> list[dict]:
    rows = []
    for item in items():
        if item["split"] != "development":
            continue
        for reversal, voice, role, condition in itertools.product(
            [0, 1], ["active", "passive"], ["agent", "patient"], ["simple", "no_context"]
        ):
            text, _, _ = sentence(item, reversal, voice, condition)
            first, second = item["participants"][:: 1 if reversal == 0 else -1]
            agent, patient = (first, second) if voice == "active" else (second, first)
            gold = agent if role == "agent" else patient
            question = (
                f"Who {item['event']} someone?" if role == "agent" else f"Who was {item['event']}?"
            )
            rows.append(
                dict(
                    family=item["family"],
                    split="development",
                    reversal=reversal,
                    voice=voice,
                    role=role,
                    condition=condition,
                    prompt=DEMONSTRATIONS
                    + f"Sentence: {text or '[not provided]'}\nQuestion: {question}\nAnswer: the",
                    candidates=[" " + name + "." for name in item["participants"]],
                    gold_index=item["participants"].index(gold),
                )
            )
    return rows


def decisions(rows: list[dict]) -> dict:
    pairs = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in ["family", "voice", "role", "condition"])
        if row["reversal"] in pairs[key]:
            raise ValueError("Duplicate reversal")
        pairs[key][row["reversal"]] = row["margin"]
    grouped = defaultdict(list)
    for (_, voice, role, condition), pair in pairs.items():
        if set(pair) != {0, 1}:
            raise ValueError("Missing reversal")
        grouped[voice, role, condition].append(float(min(pair.values()) > 0))
    summary = [
        dict(
            voice=v,
            role=r,
            condition=c,
            both_reversals=float(np.mean(values)),
            families=len(values),
        )
        for (v, r, c), values in sorted(grouped.items())
    ]
    simple = [r for r in summary if r["condition"] == "simple"]
    if len(simple) != 4:
        raise ValueError("All voice/role strata required")
    return dict(
        passed=all(r["both_reversals"] >= 0.75 for r in simple),
        strata=summary,
        rule="At least 75% correct in BOTH participant reversals in EACH voice/role stratum on simple development clauses",
    )
