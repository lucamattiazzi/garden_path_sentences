"""
50 garden-path / non-garden-path minimal pairs across 5 syntactic templates.

Each pair is designed to be NOVEL (not a canonical example from the literature
that is likely memorized during pretraining), while preserving the syntactic
template that triggers the garden-path effect.

For every pair we record:
    - id              integer identifier
    - template        one of {RR, NV, NP_S, NP_Z, COORD}
    - gps             garden-path sentence
    - normal          minimally different non-garden-path version
    - critical_word   the word whose surprisal we measure (the disambiguating
                      token in GPS; same surface form in NORMAL)
    - gps_prefix      everything in `gps` BEFORE the critical word
    - normal_prefix   everything in `normal` BEFORE the critical word

Surprisal is computed as -log P(critical_word | prefix). The garden-path
effect is the *difference*  S(GPS) - S(NORMAL)  at the critical word.

────────────────────────────────────────────────────────────────────────────
Templates
────────────────────────────────────────────────────────────────────────────
RR     Reduced Relative clause. Past-tense verb is ambiguous between simple
       past (main verb) and past participle (passive reduced relative).
       Example: "The horse raced past the barn fell."
       Disambiguator: the main verb at the end.

NV     Noun/Verb zero-derivation. A bare adjective acts as a collective noun
       ("the old" = the elderly), forcing the next word, normally read as a
       noun, to be re-analysed as the verb.
       Example: "The old man the boat."
       Disambiguator: the determiner following the re-analysed verb.

NP_S   NP-vs-S complement ambiguity. A cognition verb takes either an NP
       or a clausal complement; the auxiliary forces the clausal reading.
       Example: "The man knew the answer was wrong."
       Disambiguator: the auxiliary (was/had/were/is).

NP_Z   NP-vs-Z (zero) ambiguity in transitive/intransitive verbs in
       subordinate clauses. An optionally-transitive verb is locally
       interpreted as transitive; the matrix verb forces re-analysis.
       Example: "While the man hunted the deer ran into the woods."
       Disambiguator: the main-clause verb.

COORD  Coordination scope ambiguity. "X and Y" is initially parsed as a
       conjoined object NP; a following verb forces Y to be a new subject.
       Example: "John kissed Mary and her sister laughed."
       Disambiguator: the second verb.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Pair:
    id: int
    template: str
    gps: str
    normal: str
    critical_word: str
    gps_prefix: str
    normal_prefix: str
    # First content word AFTER the critical_word in the GPS sentence, used to
    # behaviourally test whether the model has committed to the correct parse.
    # For NV pairs this is the direct object of the re-analysed verb.
    # Empty for templates where the GPS ends at the critical_word (RR, COORD).
    verb_continuation: str = ""

    def __post_init__(self):
        # Sanity checks
        assert self.gps.startswith(self.gps_prefix), (
            f"Pair {self.id}: gps_prefix mismatch"
        )
        assert self.normal.startswith(self.normal_prefix), (
            f"Pair {self.id}: normal_prefix mismatch"
        )
        gps_rest = self.gps[len(self.gps_prefix):].lstrip()
        normal_rest = self.normal[len(self.normal_prefix):].lstrip()
        assert gps_rest.startswith(self.critical_word), (
            f"Pair {self.id}: critical word '{self.critical_word}' "
            f"does not follow gps_prefix (got: '{gps_rest[:30]}...')"
        )
        assert normal_rest.startswith(self.critical_word), (
            f"Pair {self.id}: critical word '{self.critical_word}' "
            f"does not follow normal_prefix (got: '{normal_rest[:30]}...')"
        )
        if self.template == "NV":
            assert self.verb_continuation, (
                f"Pair {self.id}: NV pair must define verb_continuation"
            )
            after_critical = gps_rest[len(self.critical_word):].lstrip()
            assert after_critical.startswith(self.verb_continuation), (
                f"Pair {self.id}: verb_continuation '{self.verb_continuation}' "
                f"does not follow critical_word (got: '{after_critical[:30]}...')"
            )


# ─── RR: Reduced Relative (15 pairs) ──────────────────────────────────────────
_RR = [
    Pair(
        id=1, template="RR",
        gps="The horse raced past the barn fell.",
        normal="The horse that was raced past the barn fell.",
        critical_word="fell",
        gps_prefix="The horse raced past the barn",
        normal_prefix="The horse that was raced past the barn",
    ),
    Pair(
        id=2, template="RR",
        gps="The boat sailed past the lighthouse sank.",
        normal="The boat that was sailed past the lighthouse sank.",
        critical_word="sank",
        gps_prefix="The boat sailed past the lighthouse",
        normal_prefix="The boat that was sailed past the lighthouse",
    ),
    Pair(
        id=3, template="RR",
        gps="The dog walked through the park bit a stranger.",
        normal="The dog that was walked through the park bit a stranger.",
        critical_word="bit",
        gps_prefix="The dog walked through the park",
        normal_prefix="The dog that was walked through the park",
    ),
    Pair(
        id=4, template="RR",
        gps="The criminal chased through the alley surrendered.",
        normal="The criminal that was chased through the alley surrendered.",
        critical_word="surrendered",
        gps_prefix="The criminal chased through the alley",
        normal_prefix="The criminal that was chased through the alley",
    ),
    Pair(
        id=5, template="RR",
        gps="The package mailed to the office arrived torn.",
        normal="The package that was mailed to the office arrived torn.",
        critical_word="arrived",
        gps_prefix="The package mailed to the office",
        normal_prefix="The package that was mailed to the office",
    ),
    Pair(
        id=6, template="RR",
        gps="The athlete trained for the marathon collapsed.",
        normal="The athlete that was trained for the marathon collapsed.",
        critical_word="collapsed",
        gps_prefix="The athlete trained for the marathon",
        normal_prefix="The athlete that was trained for the marathon",
    ),
    Pair(
        id=7, template="RR",
        gps="The boy pushed through the doorway fell.",
        normal="The boy that was pushed through the doorway fell.",
        critical_word="fell",
        gps_prefix="The boy pushed through the doorway",
        normal_prefix="The boy that was pushed through the doorway",
    ),
    Pair(
        id=8, template="RR",
        gps="The cat carried into the kitchen yowled.",
        normal="The cat that was carried into the kitchen yowled.",
        critical_word="yowled",
        gps_prefix="The cat carried into the kitchen",
        normal_prefix="The cat that was carried into the kitchen",
    ),
    Pair(
        id=9, template="RR",
        gps="The recruit taught by the sergeant resigned.",
        normal="The recruit that was taught by the sergeant resigned.",
        critical_word="resigned",
        gps_prefix="The recruit taught by the sergeant",
        normal_prefix="The recruit that was taught by the sergeant",
    ),
    Pair(
        id=10, template="RR",
        gps="The witness questioned by the detective lied.",
        normal="The witness that was questioned by the detective lied.",
        critical_word="lied",
        gps_prefix="The witness questioned by the detective",
        normal_prefix="The witness that was questioned by the detective",
    ),
    Pair(
        id=11, template="RR",
        gps="The deer hunted in the valley escaped.",
        normal="The deer that was hunted in the valley escaped.",
        critical_word="escaped",
        gps_prefix="The deer hunted in the valley",
        normal_prefix="The deer that was hunted in the valley",
    ),
    Pair(
        id=12, template="RR",
        gps="The patient examined in the ward recovered.",
        normal="The patient that was examined in the ward recovered.",
        critical_word="recovered",
        gps_prefix="The patient examined in the ward",
        normal_prefix="The patient that was examined in the ward",
    ),
    Pair(
        id=13, template="RR",
        gps="The letter delivered yesterday disappeared.",
        normal="The letter that was delivered yesterday disappeared.",
        critical_word="disappeared",
        gps_prefix="The letter delivered yesterday",
        normal_prefix="The letter that was delivered yesterday",
    ),
    Pair(
        id=14, template="RR",
        gps="The meal served at the banquet spoiled.",
        normal="The meal that was served at the banquet spoiled.",
        critical_word="spoiled",
        gps_prefix="The meal served at the banquet",
        normal_prefix="The meal that was served at the banquet",
    ),
    Pair(
        id=15, template="RR",
        gps="The ball kicked over the fence vanished.",
        normal="The ball that was kicked over the fence vanished.",
        critical_word="vanished",
        gps_prefix="The ball kicked over the fence",
        normal_prefix="The ball that was kicked over the fence",
    ),
]


# ─── NV: Noun/Verb zero-derivation (10 pairs) ─────────────────────────────────
_NV = [
    Pair(
        id=16, template="NV",
        gps="The old man the boat.",
        normal="The old people man the boat.",
        critical_word="the",
        gps_prefix="The old man",
        normal_prefix="The old people man",
        verb_continuation="boat",
    ),
    Pair(
        id=17, template="NV",
        gps="The young captain the ship.",
        normal="The young sailors captain the ship.",
        critical_word="the",
        gps_prefix="The young captain",
        normal_prefix="The young sailors captain",
        verb_continuation="ship",
    ),
    Pair(
        id=18, template="NV",
        gps="The brave guard the gate.",
        normal="The brave knights guard the gate.",
        critical_word="the",
        gps_prefix="The brave guard",
        normal_prefix="The brave knights guard",
        verb_continuation="gate",
    ),
    Pair(
        id=19, template="NV",
        gps="The wealthy bank their savings overseas.",
        normal="The wealthy families bank their savings overseas.",
        critical_word="their",
        gps_prefix="The wealthy bank",
        normal_prefix="The wealthy families bank",
        verb_continuation="savings",
    ),
    Pair(
        id=20, template="NV",
        gps="The sick nurse the children at home.",
        normal="The sick women nurse the children at home.",
        critical_word="the",
        gps_prefix="The sick nurse",
        normal_prefix="The sick women nurse",
        verb_continuation="children",
    ),
    Pair(
        id=21, template="NV",
        gps="The hungry bear the storm in silence.",
        normal="The hungry farmers bear the storm in silence.",
        critical_word="the",
        gps_prefix="The hungry bear",
        normal_prefix="The hungry farmers bear",
        verb_continuation="storm",
    ),
    Pair(
        id=22, template="NV",
        gps="The poor school their kids harshly.",
        normal="The poor parents school their kids harshly.",
        critical_word="their",
        gps_prefix="The poor school",
        normal_prefix="The poor parents school",
        verb_continuation="kids",
    ),
    Pair(
        id=23, template="NV",
        gps="The exhausted dog the trails for hours.",
        normal="The exhausted hikers dog the trails for hours.",
        critical_word="the",
        gps_prefix="The exhausted dog",
        normal_prefix="The exhausted hikers dog",
        verb_continuation="trails",
    ),
    Pair(
        id=24, template="NV",
        gps="The mighty rule the seas without mercy.",
        normal="The mighty kings rule the seas without mercy.",
        critical_word="the",
        gps_prefix="The mighty rule",
        normal_prefix="The mighty kings rule",
        verb_continuation="seas",
    ),
    Pair(
        id=25, template="NV",
        gps="The wounded fish for survival in lakes.",
        normal="The wounded soldiers fish for survival in lakes.",
        critical_word="for",
        gps_prefix="The wounded fish",
        normal_prefix="The wounded soldiers fish",
        verb_continuation="survival",
    ),
]


# ─── NP_S: NP-vs-S complement (10 pairs) ──────────────────────────────────────
_NP_S = [
    Pair(
        id=26, template="NP_S",
        gps="The man knew the answer was wrong.",
        normal="The man knew that the answer was wrong.",
        critical_word="was",
        gps_prefix="The man knew the answer",
        normal_prefix="The man knew that the answer",
    ),
    Pair(
        id=27, template="NP_S",
        gps="She believed the doctor was honest.",
        normal="She believed that the doctor was honest.",
        critical_word="was",
        gps_prefix="She believed the doctor",
        normal_prefix="She believed that the doctor",
    ),
    Pair(
        id=28, template="NP_S",
        gps="We heard the news was fake.",
        normal="We heard that the news was fake.",
        critical_word="was",
        gps_prefix="We heard the news",
        normal_prefix="We heard that the news",
    ),
    Pair(
        id=29, template="NP_S",
        gps="The detective discovered the suspect had fled.",
        normal="The detective discovered that the suspect had fled.",
        critical_word="had",
        gps_prefix="The detective discovered the suspect",
        normal_prefix="The detective discovered that the suspect",
    ),
    Pair(
        id=30, template="NP_S",
        gps="I forgot the keys were on the counter.",
        normal="I forgot that the keys were on the counter.",
        critical_word="were",
        gps_prefix="I forgot the keys",
        normal_prefix="I forgot that the keys",
    ),
    Pair(
        id=31, template="NP_S",
        gps="They realized the meeting was canceled.",
        normal="They realized that the meeting was canceled.",
        critical_word="was",
        gps_prefix="They realized the meeting",
        normal_prefix="They realized that the meeting",
    ),
    Pair(
        id=32, template="NP_S",
        gps="He suspected the witness was lying.",
        normal="He suspected that the witness was lying.",
        critical_word="was",
        gps_prefix="He suspected the witness",
        normal_prefix="He suspected that the witness",
    ),
    Pair(
        id=33, template="NP_S",
        gps="Mary remembered the gift was in the closet.",
        normal="Mary remembered that the gift was in the closet.",
        critical_word="was",
        gps_prefix="Mary remembered the gift",
        normal_prefix="Mary remembered that the gift",
    ),
    Pair(
        id=34, template="NP_S",
        gps="The reporter learned the official had resigned.",
        normal="The reporter learned that the official had resigned.",
        critical_word="had",
        gps_prefix="The reporter learned the official",
        normal_prefix="The reporter learned that the official",
    ),
    Pair(
        id=35, template="NP_S",
        gps="Tom assumed the train had departed.",
        normal="Tom assumed that the train had departed.",
        critical_word="had",
        gps_prefix="Tom assumed the train",
        normal_prefix="Tom assumed that the train",
    ),
]


# ─── NP_Z: NP/Z transitive/intransitive (10 pairs) ────────────────────────────
_NP_Z = [
    Pair(
        id=36, template="NP_Z",
        gps="While the man hunted the deer ran into the woods.",
        normal="While the man hunted, the deer ran into the woods.",
        critical_word="ran",
        gps_prefix="While the man hunted the deer",
        normal_prefix="While the man hunted, the deer",
    ),
    Pair(
        id=37, template="NP_Z",
        gps="Before the boy washed the dog escaped the yard.",
        normal="Before the boy washed, the dog escaped the yard.",
        critical_word="escaped",
        gps_prefix="Before the boy washed the dog",
        normal_prefix="Before the boy washed, the dog",
    ),
    Pair(
        id=38, template="NP_Z",
        gps="After Mary returned the books seemed misplaced.",
        normal="After Mary returned, the books seemed misplaced.",
        critical_word="seemed",
        gps_prefix="After Mary returned the books",
        normal_prefix="After Mary returned, the books",
    ),
    Pair(
        id=39, template="NP_Z",
        gps="When the chef cooked the pasta boiled over.",
        normal="When the chef cooked, the pasta boiled over.",
        critical_word="boiled",
        gps_prefix="When the chef cooked the pasta",
        normal_prefix="When the chef cooked, the pasta",
    ),
    Pair(
        id=40, template="NP_Z",
        gps="While the soldiers fought the battle raged on.",
        normal="While the soldiers fought, the battle raged on.",
        critical_word="raged",
        gps_prefix="While the soldiers fought the battle",
        normal_prefix="While the soldiers fought, the battle",
    ),
    Pair(
        id=41, template="NP_Z",
        gps="Before the baby ate the food fell off the table.",
        normal="Before the baby ate, the food fell off the table.",
        critical_word="fell",
        gps_prefix="Before the baby ate the food",
        normal_prefix="Before the baby ate, the food",
    ),
    Pair(
        id=42, template="NP_Z",
        gps="After the student read the book gathered dust.",
        normal="After the student read, the book gathered dust.",
        critical_word="gathered",
        gps_prefix="After the student read the book",
        normal_prefix="After the student read, the book",
    ),
    Pair(
        id=43, template="NP_Z",
        gps="While the singer sang the song echoed loudly.",
        normal="While the singer sang, the song echoed loudly.",
        critical_word="echoed",
        gps_prefix="While the singer sang the song",
        normal_prefix="While the singer sang, the song",
    ),
    Pair(
        id=44, template="NP_Z",
        gps="Before the driver moved the car broke down.",
        normal="Before the driver moved, the car broke down.",
        critical_word="broke",
        gps_prefix="Before the driver moved the car",
        normal_prefix="Before the driver moved, the car",
    ),
    Pair(
        id=45, template="NP_Z",
        gps="When the artist painted the canvas dried quickly.",
        normal="When the artist painted, the canvas dried quickly.",
        critical_word="dried",
        gps_prefix="When the artist painted the canvas",
        normal_prefix="When the artist painted, the canvas",
    ),
]


# ─── COORD: Coordination scope ambiguity (5 pairs) ────────────────────────────
_COORD = [
    Pair(
        id=46, template="COORD",
        gps="John kissed Mary and her sister laughed.",
        normal="John kissed Mary, and her sister laughed.",
        critical_word="laughed",
        gps_prefix="John kissed Mary and her sister",
        normal_prefix="John kissed Mary, and her sister",
    ),
    Pair(
        id=47, template="COORD",
        gps="Sarah called Tom and his brother answered.",
        normal="Sarah called Tom, and his brother answered.",
        critical_word="answered",
        gps_prefix="Sarah called Tom and his brother",
        normal_prefix="Sarah called Tom, and his brother",
    ),
    Pair(
        id=48, template="COORD",
        gps="We invited Linda and her husband declined.",
        normal="We invited Linda, and her husband declined.",
        critical_word="declined",
        gps_prefix="We invited Linda and her husband",
        normal_prefix="We invited Linda, and her husband",
    ),
    Pair(
        id=49, template="COORD",
        gps="The chef served Paul and his wife complained.",
        normal="The chef served Paul, and his wife complained.",
        critical_word="complained",
        gps_prefix="The chef served Paul and his wife",
        normal_prefix="The chef served Paul, and his wife",
    ),
    Pair(
        id=50, template="COORD",
        gps="The teacher praised Anna and her classmate smiled.",
        normal="The teacher praised Anna, and her classmate smiled.",
        critical_word="smiled",
        gps_prefix="The teacher praised Anna and her classmate",
        normal_prefix="The teacher praised Anna, and her classmate",
    ),
]


PAIRS: list[Pair] = _RR + _NV + _NP_S + _NP_Z + _COORD

TEMPLATES = ["RR", "NV", "NP_S", "NP_Z", "COORD"]


def by_template(template: str) -> list[Pair]:
    return [p for p in PAIRS if p.template == template]


if __name__ == "__main__":
    print(f"Total pairs: {len(PAIRS)}")
    print()
    for tmpl in TEMPLATES:
        bucket = by_template(tmpl)
        print(f"  {tmpl:6}  {len(bucket)} pairs")
    print()
    for p in PAIRS:
        print(f"[{p.id:2d}] {p.template:5}  crit='{p.critical_word}'")
        print(f"     GPS:    {p.gps}")
        print(f"     NORMAL: {p.normal}")
