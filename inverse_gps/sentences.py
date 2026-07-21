"""Inverse garden-path candidates: sentences hypothesized to garden-path
LLMs but NOT humans.

The main pipeline (model_finder/sentences.py) asks whether LLMs show the
human garden-path effect. This dataset asks the inverse question: are there
constructions where a statistical next-token parser is pulled toward a
wrong continuation/interpretation while the human syntactic parser sails
through? Finding both directions would be a double dissociation between
human reanalysis and LLM behaviour.

Three trap templates, each with a mechanistic hypothesis about WHY the
trap should be LLM-specific:

────────────────────────────────────────────────────────────────────────────
IDIOM   Literal use of an idiom chunk. The prefix contains a V+NP idiom
        ("spilled the beans") inside a context that forces the literal
        reading. The idiomatic continuation dominates the training
        distribution, so the model should be captured by it; a human
        reader has the literal parse fully available and the context has
        already disambiguated it (the misreading requires *ignoring* the
        context, not reanalysing the syntax).
        Trap edit: control replaces the idiom noun (or verb) with a
        non-idiomatic one, preserving the literal scene.

QUOTE   Memorized-text divergence. The prefix is a verbatim famous quote
        opening; the sentence then continues grammatically but off-script.
        LLM memorization of frequent sequences is known to distort
        surprisal away from human expectations (Oh & Schuler 2023).
        Humans recognise the quote too, but the continuation is
        semantically seamless — the predicted human cost is mild amusement,
        not reanalysis. This is the weakest "not in people" claim of the
        three; the prediction is a large *magnitude* gap.

ROLE    Argument-role reversal. Grammatically unambiguous sentences whose
        thematic roles are implausible ("The dog was bitten by the man").
        LLMs are documented to rely on plausibility over argument
        structure; the predicted failure is not a surprisal spike but a
        garden-path-style *lingering misinterpretation* (Christianson et
        al. 2001) measured with comprehension questions — except here the
        surface string is unambiguous, so humans given normal reading time
        are near ceiling. Caveat: humans do noisy-channel-correct some
        implausible sentences (Gibson et al. 2013); the human baseline
        must be measured, not assumed, for this template.

Measurement fields:
    - trap_prefix / control_prefix + target
        The target word continues BOTH prefixes grammatically; the
        LLM-garden-path effect is S(target|trap) − S(target|control),
        exactly analogous to the main pipeline's Δ-surprisal detector.
        (IDIOM and QUOTE only.)
    - attractor
        The continuation the trap is predicted to pull the model toward
        (idiom completion / quote completion). Capture score:
        logP(attractor|trap) − logP(target|trap), and the same contrast
        on the control prefix as a specificity check. (IDIOM, QUOTE.)
    - trap_sentence / control_sentence + question / correct_answer /
      trap_answer
        Forced-choice comprehension probe in the style of Christianson et
        al. (2001) as applied to LMs by Amouyal et al. (2025). (IDIOM and
        ROLE; QUOTE traps have no natural comprehension question.)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class InvItem:
    id: int
    template: str  # IDIOM | QUOTE | ROLE
    # Surprisal / capture instrumentation (IDIOM, QUOTE)
    trap_prefix: str = ""
    control_prefix: str = ""
    target: str = ""
    attractor: str = ""
    # Full sentences (all templates; ROLE control = plausible role order)
    trap_sentence: str = ""
    control_sentence: str = ""
    # Comprehension probe (IDIOM, ROLE)
    question: str = ""
    correct_answer: str = ""
    trap_answer: str = ""

    def __post_init__(self):
        assert self.template in ("IDIOM", "QUOTE", "ROLE"), self.template

        if self.template in ("IDIOM", "QUOTE"):
            assert self.trap_prefix and self.control_prefix and self.target
            assert self.attractor and self.attractor != self.target
            for name, sent, prefix in (
                ("trap", self.trap_sentence, self.trap_prefix),
                ("control", self.control_sentence, self.control_prefix),
            ):
                assert sent.startswith(prefix), (
                    f"Item {self.id}: {name}_sentence does not start with "
                    f"{name}_prefix"
                )
                rest = sent[len(prefix):].lstrip()
                assert rest.startswith(self.target), (
                    f"Item {self.id}: target '{self.target}' does not follow "
                    f"{name}_prefix (got: '{rest[:30]}...')"
                )

        if self.template in ("IDIOM", "ROLE"):
            assert self.question and self.correct_answer and self.trap_answer
            assert self.trap_sentence and self.control_sentence


# ─── IDIOM: literal idiom continuation (10 items) ────────────────────────────
_IDIOM = [
    InvItem(
        id=1, template="IDIOM",
        trap_prefix="Carrying dinner across the kitchen, the clumsy waiter spilled the beans",
        control_prefix="Carrying dinner across the kitchen, the clumsy waiter spilled the lentils",
        target="onto",
        attractor="about",
        trap_sentence="Carrying dinner across the kitchen, the clumsy waiter spilled the beans onto the tiled floor.",
        control_sentence="Carrying dinner across the kitchen, the clumsy waiter spilled the lentils onto the tiled floor.",
        question="What did the waiter do?",
        correct_answer="He dropped food on the floor.",
        trap_answer="He revealed a secret.",
    ),
    InvItem(
        id=2, template="IDIOM",
        trap_prefix="While mopping the stable, the farmer kicked the bucket",
        control_prefix="While mopping the stable, the farmer kicked the crate",
        target="over",
        attractor="at the age of",
        trap_sentence="While mopping the stable, the farmer kicked the bucket over and soaked his boots.",
        control_sentence="While mopping the stable, the farmer kicked the crate over and soaked his boots.",
        question="What happened to the farmer?",
        correct_answer="He knocked over a container.",
        trap_answer="He died.",
    ),
    InvItem(
        id=3, template="IDIOM",
        trap_prefix="Under the picnic table, the stray dog chewed the fat",
        control_prefix="Under the picnic table, the stray dog chewed the gristle",
        target="off",
        attractor="with",
        trap_sentence="Under the picnic table, the stray dog chewed the fat off a discarded steak bone.",
        control_sentence="Under the picnic table, the stray dog chewed the gristle off a discarded steak bone.",
        question="What was the dog doing?",
        correct_answer="Eating scraps of meat.",
        trap_answer="Having a friendly chat.",
    ),
    InvItem(
        id=4, template="IDIOM",
        trap_prefix="Practicing his swing in the barn, the boy hit the sack",
        control_prefix="Practicing his swing in the barn, the boy hit the crate",
        target="of",
        attractor="early",
        trap_sentence="Practicing his swing in the barn, the boy hit the sack of grain with his bat.",
        control_sentence="Practicing his swing in the barn, the boy hit the crate of grain with his bat.",
        question="What did the boy hit?",
        correct_answer="A container of grain.",
        trap_answer="He went to bed.",
    ),
    InvItem(
        id=5, template="IDIOM",
        trap_prefix="During the rehab session, the physiotherapist pulled the patient's leg",
        control_prefix="During the rehab session, the physiotherapist pulled the patient's arm",
        target="toward",
        attractor="about",
        trap_sentence="During the rehab session, the physiotherapist pulled the patient's leg toward her chest to stretch the muscle.",
        control_sentence="During the rehab session, the physiotherapist pulled the patient's arm toward her chest to stretch the muscle.",
        question="What was the physiotherapist doing?",
        correct_answer="Stretching a limb.",
        trap_answer="Teasing the patient.",
    ),
    InvItem(
        id=6, template="IDIOM",
        trap_prefix="Arriving at the clinic, the owner let the cat out of the bag",
        control_prefix="Arriving at the clinic, the owner let the rabbit out of the bag",
        target="so",
        attractor="about",
        trap_sentence="Arriving at the clinic, the owner let the cat out of the bag so the vet could examine it.",
        control_sentence="Arriving at the clinic, the owner let the rabbit out of the bag so the vet could examine it.",
        question="What happened at the clinic?",
        correct_answer="An animal was taken out of a carrier.",
        trap_answer="A secret was revealed.",
    ),
    InvItem(
        id=7, template="IDIOM",
        trap_prefix="Kneeling on the frozen pond, the fisherman broke the ice",
        control_prefix="Kneeling on the frozen pond, the fisherman cracked the ice",
        target="using",
        attractor="with a joke",
        trap_sentence="Kneeling on the frozen pond, the fisherman broke the ice using a steel auger.",
        control_sentence="Kneeling on the frozen pond, the fisherman cracked the ice using a steel auger.",
        question="What did the fisherman do?",
        correct_answer="Made a hole in frozen water.",
        trap_answer="Started a friendly conversation.",
    ),
    InvItem(
        id=8, template="IDIOM",
        trap_prefix="Loading the washing machine, the janitor threw in the towel",
        control_prefix="Loading the washing machine, the janitor threw in the apron",
        target="along",
        attractor="and gave up",
        trap_sentence="Loading the washing machine, the janitor threw in the towel along with the rags.",
        control_sentence="Loading the washing machine, the janitor threw in the apron along with the rags.",
        question="What did the janitor do?",
        correct_answer="Put laundry into a machine.",
        trap_answer="He gave up.",
    ),
    InvItem(
        id=9, template="IDIOM",
        trap_prefix="Crawling through the sewer line, the exterminator smelled a rat",
        control_prefix="Crawling through the sewer line, the exterminator smelled a raccoon",
        target="somewhere",
        attractor="and became suspicious",
        trap_sentence="Crawling through the sewer line, the exterminator smelled a rat somewhere behind the pipes.",
        control_sentence="Crawling through the sewer line, the exterminator smelled a raccoon somewhere behind the pipes.",
        question="What did the exterminator notice?",
        correct_answer="The smell of an actual rodent.",
        trap_answer="That something dishonest was going on.",
    ),
    InvItem(
        id=10, template="IDIOM",
        trap_prefix="Gripping it with chalked hands, the weightlifter raised the bar",
        control_prefix="Gripping it with chalked hands, the weightlifter raised the dumbbell",
        target="above",
        attractor="for",
        trap_sentence="Gripping it with chalked hands, the weightlifter raised the bar above his head.",
        control_sentence="Gripping it with chalked hands, the weightlifter raised the dumbbell above his head.",
        question="What did the weightlifter do?",
        correct_answer="Lifted a weight overhead.",
        trap_answer="Increased the standard for everyone.",
    ),
]


# ─── QUOTE: memorized-text divergence (10 items) ─────────────────────────────
_QUOTE = [
    InvItem(
        id=11, template="QUOTE",
        trap_prefix="To be or not to be, that is the",
        control_prefix="To sell or not to sell, that is the",
        target="dilemma",
        attractor="question",
        trap_sentence="To be or not to be, that is the dilemma every founder faces.",
        control_sentence="To sell or not to sell, that is the dilemma every founder faces.",
    ),
    InvItem(
        id=12, template="QUOTE",
        trap_prefix="In the beginning God created the heavens and the",
        control_prefix="Before anything else God created the heavens and the",
        target="oceans",
        attractor="earth",
        trap_sentence="In the beginning God created the heavens and the oceans.",
        control_sentence="Before anything else God created the heavens and the oceans.",
    ),
    InvItem(
        id=13, template="QUOTE",
        trap_prefix="I have a dream that one day this",
        control_prefix="I have a hope that one day this",
        target="company",
        attractor="nation",
        trap_sentence="I have a dream that one day this company will treat every worker fairly.",
        control_sentence="I have a hope that one day this company will treat every worker fairly.",
    ),
    InvItem(
        id=14, template="QUOTE",
        trap_prefix="The quick brown fox jumps over the lazy",
        control_prefix="The quick brown wolf jumps over the lazy",
        target="cat",
        attractor="dog",
        trap_sentence="The quick brown fox jumps over the lazy cat.",
        control_sentence="The quick brown wolf jumps over the lazy cat.",
    ),
    InvItem(
        id=15, template="QUOTE",
        trap_prefix="That's one small step for a man, one giant leap for",
        control_prefix="It was a small step for the engineer but a giant leap for",
        target="robotics",
        attractor="mankind",
        trap_sentence="That's one small step for a man, one giant leap for robotics.",
        control_sentence="It was a small step for the engineer but a giant leap for robotics.",
    ),
    InvItem(
        id=16, template="QUOTE",
        trap_prefix="Ask not what your country can do for you, ask what you can do for your",
        control_prefix="Never mind what the team can do for you, ask what you can do for your",
        target="family",
        attractor="country",
        trap_sentence="Ask not what your country can do for you, ask what you can do for your family.",
        control_sentence="Never mind what the team can do for you, ask what you can do for your family.",
    ),
    InvItem(
        id=17, template="QUOTE",
        trap_prefix="Four score and seven years ago our",
        control_prefix="Eighty-seven years ago our",
        target="grandparents",
        attractor="fathers",
        trap_sentence="Four score and seven years ago our grandparents opened this bakery.",
        control_sentence="Eighty-seven years ago our grandparents opened this bakery.",
    ),
    InvItem(
        id=18, template="QUOTE",
        trap_prefix="It was the best of times, it was the worst of",
        control_prefix="That summer brought the best of times and the worst of",
        target="luck",
        attractor="times",
        trap_sentence="It was the best of times, it was the worst of luck.",
        control_sentence="That summer brought the best of times and the worst of luck.",
    ),
    InvItem(
        id=19, template="QUOTE",
        trap_prefix="May the Force be with",
        control_prefix="May the blessing be with",
        target="them",
        attractor="you",
        trap_sentence="May the Force be with them on the battlefield.",
        control_sentence="May the blessing be with them on the battlefield.",
    ),
    InvItem(
        id=20, template="QUOTE",
        trap_prefix="Houston, we have a",
        control_prefix="Everyone, we have a",
        target="solution",
        attractor="problem",
        trap_sentence="Houston, we have a solution.",
        control_sentence="Everyone, we have a solution.",
    ),
]


# ─── ROLE: argument-role reversal (10 items) ─────────────────────────────────
_ROLE = [
    InvItem(
        id=21, template="ROLE",
        trap_sentence="The dog was bitten by the man during the fight.",
        control_sentence="The man was bitten by the dog during the fight.",
        question="Who did the biting?",
        correct_answer="The man.",
        trap_answer="The dog.",
    ),
    InvItem(
        id=22, template="ROLE",
        trap_sentence="The doctor that the patient examined seemed nervous.",
        control_sentence="The patient that the doctor examined seemed nervous.",
        question="Who performed the examination?",
        correct_answer="The patient.",
        trap_answer="The doctor.",
    ),
    InvItem(
        id=23, template="ROLE",
        trap_sentence="The jeweler robbed the thief at closing time.",
        control_sentence="The thief robbed the jeweler at closing time.",
        question="Who was robbed?",
        correct_answer="The thief.",
        trap_answer="The jeweler.",
    ),
    InvItem(
        id=24, template="ROLE",
        trap_sentence="The lifeguard was rescued by the swimmer just before noon.",
        control_sentence="The swimmer was rescued by the lifeguard just before noon.",
        question="Who did the rescuing?",
        correct_answer="The swimmer.",
        trap_answer="The lifeguard.",
    ),
    InvItem(
        id=25, template="ROLE",
        trap_sentence="The student gave the teacher detention after class.",
        control_sentence="The teacher gave the student detention after class.",
        question="Who received detention?",
        correct_answer="The teacher.",
        trap_answer="The student.",
    ),
    InvItem(
        id=26, template="ROLE",
        trap_sentence="The mouse chased the cat around the courtyard.",
        control_sentence="The cat chased the mouse around the courtyard.",
        question="Who was chased?",
        correct_answer="The cat.",
        trap_answer="The mouse.",
    ),
    InvItem(
        id=27, template="ROLE",
        trap_sentence="The customer served the waiter a bowl of soup.",
        control_sentence="The waiter served the customer a bowl of soup.",
        question="Who served the soup?",
        correct_answer="The customer.",
        trap_answer="The waiter.",
    ),
    InvItem(
        id=28, template="ROLE",
        trap_sentence="The prisoner released the guard at dawn.",
        control_sentence="The guard released the prisoner at dawn.",
        question="Who was set free?",
        correct_answer="The guard.",
        trap_answer="The prisoner.",
    ),
    InvItem(
        id=29, template="ROLE",
        trap_sentence="The baby fed the nanny a spoonful of porridge.",
        control_sentence="The nanny fed the baby a spoonful of porridge.",
        question="Who was fed?",
        correct_answer="The nanny.",
        trap_answer="The baby.",
    ),
    InvItem(
        id=30, template="ROLE",
        trap_sentence="The horse led the trainer into the stable.",
        control_sentence="The trainer led the horse into the stable.",
        question="Who was led into the stable?",
        correct_answer="The trainer.",
        trap_answer="The horse.",
    ),
]


ITEMS: list[InvItem] = _IDIOM + _QUOTE + _ROLE

INV_TEMPLATES = ["IDIOM", "QUOTE", "ROLE"]

# Templates instrumented for the surprisal/capture detectors
SURPRISAL_TEMPLATES = ["IDIOM", "QUOTE"]
# Templates instrumented for the comprehension-probe detector
QA_TEMPLATES = ["IDIOM", "ROLE"]


def by_template(template: str) -> list[InvItem]:
    return [it for it in ITEMS if it.template == template]


if __name__ == "__main__":
    print(f"Total items: {len(ITEMS)}")
    for tmpl in INV_TEMPLATES:
        print(f"  {tmpl:6}  {len(by_template(tmpl))} items")
    print()
    for it in ITEMS:
        print(f"[{it.id:2d}] {it.template:5}")
        print(f"     TRAP:    {it.trap_sentence}")
        print(f"     CONTROL: {it.control_sentence}")
        if it.target:
            print(f"     target='{it.target}'  attractor='{it.attractor}'")
        if it.question:
            print(f"     Q: {it.question}  ✓ {it.correct_answer}  ✗ {it.trap_answer}")
