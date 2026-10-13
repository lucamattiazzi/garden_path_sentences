"""Structural checks for pilot stimuli; these do not replace linguistic review."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PILOT_PATH = Path(__file__).parent / "materials/pilot.json"
Split = Literal["exploration", "validation"]


@dataclass(frozen=True)
class Question:
    text: str
    actor: int


@dataclass(frozen=True)
class Stimulus:
    id: str
    family: str
    template: Literal["NP_S", "NP_Z"]
    split: Split
    actors: tuple[str, str]
    gps: str
    normal: str
    role_reversed: str
    gps_prefix: str
    normal_prefix: str
    critical_word: str
    spillover: str
    questions: tuple[Question, Question]


def validate_pilot(items: list[Stimulus]) -> None:
    if not items or len({p.id for p in items}) != len(items):
        raise ValueError("Pilot must contain nonempty, unique item IDs")
    families: dict[str, Split] = {}
    for p in items:
        if p.template not in ("NP_S", "NP_Z") or p.split not in ("exploration", "validation"):
            raise ValueError(f"{p.id}: invalid template or split")
        if families.setdefault(p.family, p.split) != p.split:
            raise ValueError(f"{p.id}: lexical family crosses splits")
        if len(p.actors) != 2 or p.actors[0] == p.actors[1]:
            raise ValueError(f"{p.id}: two distinct actors required")
        if len(p.questions) != 2 or {q.actor for q in p.questions} != {0, 1}:
            raise ValueError(f"{p.id}: questions must cover both event subjects")
        if not p.critical_word or not p.spillover or any(not q.text for q in p.questions):
            raise ValueError(f"{p.id}: missing probe text")
        for condition in ("gps", "normal"):
            expected = f"{getattr(p, condition + '_prefix')} {p.critical_word} {p.spillover}."
            if getattr(p, condition) != expected:
                raise ValueError(f"{p.id}: {condition} critical/spillover alignment failed")
        if p.gps == p.normal or p.role_reversed == p.normal:
            raise ValueError(f"{p.id}: missing control manipulation")
        # The counterfactual preserves all words and exchanges the actors.
        reversed_text = (
            p.normal.lower().replace(p.actors[0], "{actor0}").replace(p.actors[1], "{actor1}")
        )
        reversed_text = reversed_text.replace("{actor0}", p.actors[1]).replace(
            "{actor1}", p.actors[0]
        )
        if reversed_text != p.role_reversed.lower():
            raise ValueError(f"{p.id}: role reversal must exchange only the actors")


def load_pilot() -> list[Stimulus]:
    data = json.loads(PILOT_PATH.read_text())
    items = [
        Stimulus(
            **dict(
                row,
                actors=tuple(row["actors"]),
                questions=tuple(Question(**q) for q in row["questions"]),
            )
        )
        for row in data["items"]
    ]
    validate_pilot(items)
    return items
