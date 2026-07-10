"""10 novel NV garden-path pairs for the memorisation-vs-parsing diagnostic.

The original 10 NV pairs in `model_finder/sentences.py` include several
linguistics-textbook canonicals — most notably pair 16 *"the old man the
boat"* — whose apparent comprehension is likely training-set recall rather
than reanalysis. These pairs provide GPS *sentences* that do not appear in
the linguistics literature.

What "novel" does and does not mean here:
- NOVEL: the full `[Det Adj V-amb Det N]` garden-path frame (e.g. "The weary
  shoulder the burden") is unattested as a textbook example, so the model
  cannot have memorised the sentence-level parse.
- NOT novel: several verb+object collocations are themselves frequent
  idioms ("shoulder the burden", "stomach the criticism", "head the
  rebellion"). This is deliberate for the `full − stripped` diagnostic: it
  gives the stripped condition a strong lexical baseline, so any *extra*
  benefit of the full GPS prefix must come from parsing the frame, not from
  the verb-object bigram itself.

Design constraints per pair:
- adjective is substantively usable (`The [Adj] [people]` reading works)
- V-amb has a transitive zero-derivation verb form
- both parses (verb reading and noun-phrase reading) are grammatically
  legal English
- every adjective is used only once across the 10 pairs

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
