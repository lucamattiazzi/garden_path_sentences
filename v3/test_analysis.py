import unittest

from v3.analyze import family_values


class AnalysisTests(unittest.TestCase):
    def test_direction_alignment_and_seed_averaging_precede_family_aggregation(self):
        rows = []
        for item, effect in [("a", 2), ("b", 6)]:
            for seed in range(3):
                for verb, mult in [("optional", 1), ("intransitive", 0)]:
                    for direction, sign in [("forward", 1), ("reverse", -1)]:
                        rows.append(
                            dict(
                                split="test",
                                stage="next",
                                contrast="comma_aligned",
                                kind="random",
                                family="one",
                                id=item,
                                seed=seed,
                                verb_class=verb,
                                direction=direction,
                                effects={"margin": mult * sign * effect},
                            )
                        )
        self.assertEqual(family_values(rows, "random"), {"one": 4})
        with self.assertRaises(ValueError):
            family_values(rows[:-1], "random")


if __name__ == "__main__":
    unittest.main()
