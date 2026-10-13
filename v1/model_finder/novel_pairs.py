"""10 novel NV garden-path pairs for the memorisation-vs-parsing diagnostic.

These are additional hand-authored NV items, historically labelled "novel".
Neither corpus absence nor human garden-path behaviour has been verified.
Several verb-object collocations are frequent, and the inserted subject nouns
in the normal condition can provide additional semantic cues. Full-minus-
stripped differences are descriptive; they cannot identify memorisation or
syntactic reanalysis. See research/PROTOCOL.md before using these as evidence.

Shared by qwen4b, qwen14b and qwen32b (steps 08-11).
"""

from model_finder.sentences import Pair

NOVEL_NV_PAIRS: list[Pair] = [
    Pair(
        id=51, template="NV",
        gps="The poor butter the toast.",
        normal="The poor servants butter the toast.",
        critical_word="the",
        gps_prefix="The poor butter",
        normal_prefix="The poor servants butter",
        verb_continuation="toast",
    ),
    Pair(
        id=52, template="NV",
        gps="The weary shoulder the burden.",
        normal="The weary porters shoulder the burden.",
        critical_word="the",
        gps_prefix="The weary shoulder",
        normal_prefix="The weary porters shoulder",
        verb_continuation="burden",
    ),
    Pair(
        id=53, template="NV",
        gps="The bold head the rebellion.",
        normal="The bold generals head the rebellion.",
        critical_word="the",
        gps_prefix="The bold head",
        normal_prefix="The bold generals head",
        verb_continuation="rebellion",
    ),
    Pair(
        id=54, template="NV",
        gps="The elderly witness the robbery.",
        normal="The elderly pedestrians witness the robbery.",
        critical_word="the",
        gps_prefix="The elderly witness",
        normal_prefix="The elderly pedestrians witness",
        verb_continuation="robbery",
    ),
    Pair(
        id=55, template="NV",
        gps="The elite fashion the crown.",
        normal="The elite jewelers fashion the crown.",
        critical_word="the",
        gps_prefix="The elite fashion",
        normal_prefix="The elite jewelers fashion",
        verb_continuation="crown",
    ),
    Pair(
        id=56, template="NV",
        gps="The strong mother the child.",
        normal="The strong women mother the child.",
        critical_word="the",
        gps_prefix="The strong mother",
        normal_prefix="The strong women mother",
        verb_continuation="child",
    ),
    Pair(
        id=57, template="NV",
        gps="The wealthy father the newborn.",
        normal="The wealthy men father the newborn.",
        critical_word="the",
        gps_prefix="The wealthy father",
        normal_prefix="The wealthy men father",
        verb_continuation="newborn",
    ),
    Pair(
        id=58, template="NV",
        gps="The homeless stomach the criticism.",
        normal="The homeless refugees stomach the criticism.",
        critical_word="the",
        gps_prefix="The homeless stomach",
        normal_prefix="The homeless refugees stomach",
        verb_continuation="criticism",
    ),
    Pair(
        id=59, template="NV",
        gps="The brave crew the expedition.",
        normal="The brave sailors crew the expedition.",
        critical_word="the",
        gps_prefix="The brave crew",
        normal_prefix="The brave sailors crew",
        verb_continuation="expedition",
    ),
    Pair(
        id=60, template="NV",
        gps="The rich partner the newcomer.",
        normal="The rich investors partner the newcomer.",
        critical_word="the",
        gps_prefix="The rich partner",
        normal_prefix="The rich investors partner",
        verb_continuation="newcomer",
    ),
]
