"""Fresh event families: first mention, event subject and semantic role crossed."""

import hashlib

from v3.direct.experiment import DEMONSTRATIONS

FRAMES = [
    ("watched", "guard", "officer", "surgeon", "dentist", "assisted"),
    ("helped", "nurse", "doctor", "librarian", "archivist", "telephoned"),
    ("coached", "trainer", "mentor", "violinist", "pianist", "escorted"),
    ("called", "clerk", "secretary", "engineer", "architect", "invited"),
    ("photographed", "reporter", "editor", "chemist", "physicist", "hired"),
    ("visited", "tourist", "traveler", "judge", "lawyer", "congratulated"),
    ("interviewed", "journalist", "host", "comedian", "magician", "comforted"),
    ("painted", "artist", "illustrator", "cashier", "florist", "criticized"),
    ("trained", "coach", "instructor", "mechanic", "electrician", "welcomed"),
    ("taught", "professor", "lecturer", "astronaut", "astronomer", "recognized"),
    ("questioned", "detective", "investigator", "soprano", "drummer", "introduced"),
    ("searched", "inspector", "agent", "bishop", "chaplain", "encouraged"),
]
CONSTRUCTIONS = ("active", "passive", "object_relative", "passive_relative")
FIRST_AGENT = dict(active=True, passive=False, object_relative=False, passive_relative=True)
FIRST_EVENT_SUBJECT = dict(active=True, passive=True, object_relative=False, passive_relative=False)


def items() -> list[dict]:
    order = sorted(
        [r[-1] for r in FRAMES], key=lambda s: hashlib.sha256(("gps-v4:" + s).encode()).hexdigest()
    )
    return [
        dict(
            id=event,
            family=event,
            split="development" if event in order[:6] else "confirmation",
            verb=verb,
            actor=actor,
            lexical_actor=lexical,
            participants=[first, second],
            event=event,
        )
        for verb, actor, lexical, first, second, event in FRAMES
    ]


def make_prompt(item, reversal, construction, role, condition, lexical=False):
    first, second = item["participants"][:: 1 if reversal == 0 else -1]
    actor = item["lexical_actor"] if lexical else item["actor"]
    head = f"While the {actor} {item['verb']}" + ("," if condition == "clean" else "")
    prefix = f"{head} the {first}"
    event = item["event"]
    endings = {
        "active": f" {event} the {second}.",
        "passive": f" was {event} by the {second}.",
        "object_relative": f" whom the {second} {event} smiled.",
        "passive_relative": f" by whom the {second} was {event} smiled.",
    }
    before = DEMONSTRATIONS + "Sentence: "
    question = f"Who {event} someone?" if role == "agent" else f"Who was {event}?"
    first_correct = FIRST_AGENT[construction] if role == "agent" else not FIRST_AGENT[construction]
    return dict(
        text=before + prefix + endings[construction] + f"\nQuestion: {question}\nAnswer: the",
        candidates=[" " + name + "." for name in item["participants"]],
        gold_index=reversal if first_correct else 1 - reversal,
        event_subject_index=reversal if FIRST_EVENT_SUBJECT[construction] else 1 - reversal,
        first_index=reversal,
        site_end=len(before) + len(prefix),
        gap_start=len(before) + len(head) + 1 if condition == "gap" else None,
        sentence_start=len(before),
    )
