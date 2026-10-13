"""Frozen synthetic role-reversible materials; not human-validated."""

import hashlib

FRAMES = [
    ("watched", "guard", "officer", "poet", "chef", "greeted"),
    ("helped", "nurse", "doctor", "dancer", "painter", "thanked"),
    ("coached", "trainer", "mentor", "swimmer", "runner", "praised"),
    ("called", "clerk", "secretary", "farmer", "baker", "followed"),
    ("photographed", "reporter", "editor", "actor", "singer", "helped"),
    ("visited", "tourist", "traveler", "carpenter", "plumber", "thanked"),
    ("interviewed", "journalist", "host", "pilot", "sailor", "praised"),
    ("painted", "artist", "illustrator", "musician", "writer", "greeted"),
    ("trained", "coach", "instructor", "boxer", "skater", "followed"),
    ("taught", "professor", "lecturer", "waiter", "tailor", "helped"),
    ("questioned", "detective", "investigator", "driver", "gardener", "visited"),
    ("searched", "inspector", "agent", "merchant", "barber", "thanked"),
]

# Distinct predicates/nouns from the experimental role questions; balanced labels,
# voices and queried roles. No change or prompt selection after seeing outcomes.
DEMONSTRATIONS = """Sentence: The student admired the scientist.
Question: Who admired someone?
A: the student
B: the scientist
Answer: A

Sentence: The student admired the scientist.
Question: Who was admired?
A: the student
B: the scientist
Answer: B

Sentence: The child was rescued by the firefighter.
Question: Who rescued someone?
A: the child
B: the firefighter
Answer: B

Sentence: The child was rescued by the firefighter.
Question: Who was rescued?
A: the child
B: the firefighter
Answer: A

"""


def items() -> list[dict]:
    order = sorted(
        [r[0] for r in FRAMES],
        key=lambda v: hashlib.sha256(("gps-bridge:" + v).encode()).hexdigest(),
    )
    return [
        dict(
            id=verb,
            family=verb,
            split="development" if verb in order[:6] else "test",
            verb=verb,
            actor=actor,
            lexical_actor=lexical,
            participants=[first, second],
            event=event,
        )
        for verb, actor, lexical, first, second, event in FRAMES
    ]


def sentence(
    item: dict, reversal: int, voice: str, condition: str, lexical: bool = False
) -> tuple[str, int | None, int | None]:
    first, second = item["participants"][:: 1 if reversal == 0 else -1]
    suffix = (
        f"{item['event']} the {second}."
        if voice == "active"
        else f"was {item['event']} by the {second}."
    )
    if condition == "no_context":
        return "", None, None
    if condition == "simple":
        return f"The {first} {suffix}", None, None
    actor = item["lexical_actor"] if lexical else item["actor"]
    head = f"While the {actor} {item['verb']}"
    if condition == "clean":
        head += ","
    prefix = f"{head} the {first}"
    return f"{prefix} {suffix}", len(prefix), len(head) + 1


def prompt(
    item: dict,
    reversal: int,
    voice: str,
    condition: str,
    role: str,
    order: int,
    lexical: bool = False,
) -> dict:
    text, site_end, gap_start = sentence(item, reversal, voice, condition, lexical)
    first, second = item["participants"][:: 1 if reversal == 0 else -1]
    agent, patient = (first, second) if voice == "active" else (second, first)
    correct = agent if role == "agent" else patient
    options = item["participants"][:: 1 if order == 0 else -1]
    question = f"Who {item['event']} someone?" if role == "agent" else f"Who was {item['event']}?"
    before = DEMONSTRATIONS + "Sentence: "
    text = (
        before
        + (text or "[not provided]")
        + f"\nQuestion: {question}\nA: the {options[0]}\nB: the {options[1]}\nAnswer:"
    )
    return dict(
        text=text,
        gold_index=options.index(correct),
        correct=correct,
        site_end=None if site_end is None else len(before) + site_end,
        gap_start=None if condition != "gap" else len(before) + gap_start,
        sentence_start=len(before),
        question=question,
    )
