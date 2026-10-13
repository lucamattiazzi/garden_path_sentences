import unittest

from v4.analyze import contrasts


class ContrastTests(unittest.TestCase):
    def test_subject_bias_and_role_selectivity_are_distinct(self):
        rows = []
        for construction, subject, agent in [
            ("active", 1, 1),
            ("passive", 1, -1),
            ("object_relative", -1, -1),
            ("passive_relative", -1, 1),
        ]:
            for role, role_sign in [("agent", 1), ("patient", -1)]:
                for reversal in [0, 1]:
                    for direction, direction_sign in [("clean_to_gap", 1), ("gap_to_clean", -1)]:
                        for kind, amplitude in [
                            ("cross", 1),
                            ("lexical", 0),
                            ("random", 0),
                            ("identity", 0),
                        ]:
                            rows.append(
                                dict(
                                    family="f",
                                    construction=construction,
                                    role=role,
                                    reversal=reversal,
                                    direction=direction,
                                    kind=kind,
                                    first_effect=amplitude
                                    * direction_sign
                                    * (2 * subject + 3 * agent * role_sign),
                                )
                            )
        values = {(r["kind"], r["contrast"]): r["mean"] for r in contrasts(rows)}
        self.assertEqual(values["cross", "event_subject"], 2)
        self.assertEqual(values["cross", "role_selective"], 3)
        self.assertEqual(values["cross", "first_mention"], 0)
        self.assertEqual(values["cross", "semantic_agent"], 0)
        self.assertEqual(values["cross_minus_random", "role_selective"], 3)
