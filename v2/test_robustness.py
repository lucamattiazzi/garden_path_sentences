import json
import unittest
from pathlib import Path

from v2.robustness import analyze, candidate_components, layer_stability


class RobustnessTests(unittest.TestCase):
    def test_saved_decomposition_reproduces_original_estimate_and_interval(self):
        source = Path(__file__).parent / "results/garden-followup-20261008-02/garden-followup.json"
        original = next(
            row
            for row in json.loads(source.read_text())["summaries"]
            if row["split"] == "test"
            and row["kind"] == "cross_condition"
            and row["direction"] == "value"
        )
        data = analyze()
        total = data["garden_component_summary"]["total"]
        self.assertAlmostEqual(total["mean"], original["mean"], places=12)
        self.assertEqual(total["ci95"], original["ci95"])
        self.assertEqual(
            sum(data["garden_layer_stability"]["bootstrap_selection_counts"].values()), 10000
        )

    def test_candidate_decomposition_aligns_both_directions_and_subtracts_object(self):
        rows = []
        for recipient, sign in [("no_comma", 1), ("comma", -1)]:
            for verb, multiplier in [("optional_transitive", 2), ("intransitive", 1)]:
                for candidate, effect in [("main", 3), ("object", -2)]:
                    rows.append(
                        dict(
                            family="a",
                            split="test",
                            layer=4,
                            kind="cross_condition",
                            random_seed=None,
                            recipient=recipient,
                            verb_class=verb,
                            candidate=candidate,
                            effect=sign * multiplier * effect,
                        )
                    )
        result = candidate_components(rows, 4)[0]
        self.assertEqual(result["main"], 3)
        self.assertEqual(result["object"], 2)
        self.assertEqual(result["total"], 5)
        with self.assertRaises(ValueError):
            candidate_components(rows[:-1], 4)

    def test_layer_selection_resamples_development_families_and_uses_earliest_tie(self):
        rows = [
            dict(split="development", family=f, layer=layer, value=v)
            for f in ["a", "b"]
            for layer, v in [(0, 1), (1, 2), (2, 2)]
        ]
        result = layer_stability(rows)
        self.assertEqual(result["bootstrap_selection_counts"], {"1": 10000})
        self.assertEqual(result["within_95_percent_peak"], [1, 2])
        with self.assertRaises(ValueError):
            layer_stability([dict(r, split="test") for r in rows])


if __name__ == "__main__":
    unittest.main()
