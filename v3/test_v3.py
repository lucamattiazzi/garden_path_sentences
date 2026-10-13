import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from v2.test_v2 import TransformerIntegrationTests
from v3.experiment import conditions, run_experiment, summarize_effects
from v3.measure import TokenMeasurements


class MaterialsTests(unittest.TestCase):
    def test_published_items_are_complete_and_splits_group_repeated_verbs(self):
        items = json.loads(Path("v3/materials/npz.json").read_text())["items"]
        self.assertEqual(len(items), 24)
        self.assertEqual(len({r["id"] for r in items}), 24)
        self.assertEqual(
            {r["family"] for r in items if r["split"] == "development"}
            & {r["family"] for r in items if r["split"] == "test"},
            set(),
        )
        for split in ["development", "test"]:
            group = [r for r in items if r["split"] == split]
            self.assertEqual(len(group), 12)
            self.assertEqual(len({r["family"] for r in group}), 6)
        for r in items:
            self.assertEqual(len(r["continuation"].split()), 3)
            self.assertNotEqual(r["actor"], r["other_actor"])
            for verb in ["optional", "intransitive"]:
                c = conditions(r, verb)
                self.assertEqual(c["comma"].replace(",", ""), c["native"])
                self.assertEqual(c["gap"], c["native"])


class TokenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TransformerIntegrationTests.setUpClass()
        source = TransformerIntegrationTests.measure
        cls.measure = TokenMeasurements(source.model, source.tokenizer)

    def test_identity_and_per_token_sum_match_existing_score(self):
        prefix = "While the guard watched the prisoner"
        for suffix in ["", "was the guard"]:
            baseline, states, pos = self.measure.evaluate(prefix, suffix, capture=True)
            for layer, state in enumerate(states):
                patched, _, _ = self.measure.evaluate(prefix, suffix, patch=(layer, pos, state))
                self.assertEqual(baseline["margin"], patched["margin"])
                self.assertEqual(baseline["token_logp"], patched["token_logp"])
            if suffix:
                self.assertAlmostEqual(
                    sum(baseline["token_logp"]), self.measure.score(prefix, suffix), places=6
                )
                self.assertEqual(len(baseline["word_logp"]), 3)

    def test_position_gap_aligns_noncomma_tokens_and_changes_no_token_ids(self):
        native = "While the guard watched the prisoner"
        comma = "While the guard watched, the prisoner"
        self.measure.validate_alignment(native, comma)
        normal, _, _ = self.measure.evaluate(native)
        gap, _, _ = self.measure.evaluate(native, gap=True)
        self.assertEqual(normal["input_ids"], gap["input_ids"])
        self.assertEqual(gap["position_ids"][-1], normal["position_ids"][-1] + 1)
        self.assertEqual(gap["position_ids"][:4], normal["position_ids"][:4])

    def test_final_block_transfers_next_token_logits_but_not_future_capture(self):
        native = "While the guard watched the prisoner"
        comma = "While the guard watched, the prisoner"
        a, _, pos = self.measure.evaluate(native)
        b, states, _ = self.measure.evaluate(comma, capture=True)
        patched, _, _ = self.measure.evaluate(native, patch=(2, pos, states[2]))
        self.assertAlmostEqual(patched["margin"], b["margin"], places=5)
        _, full, _ = self.measure.evaluate(comma, "was the guard", capture=True)
        for first, second in zip(states, full, strict=True):
            torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)

    def test_hooks_removed_on_failure(self):
        with self.assertRaises(IndexError):
            self.measure.evaluate("the guard", patch=(0, 999, torch.zeros(24)))
        self.assertFalse(self.measure.layers[0]._forward_hooks)

    def test_end_to_end_tiny_run_logs_controls_and_complete_factorials(self):
        items = [
            dict(
                id=split,
                family=split,
                split=split,
                actor="guard",
                other_actor="prisoner",
                prefixes={
                    "optional": "While the guard watched the prisoner",
                    "intransitive": "While the guard was the prisoner",
                },
                continuation="was the guard",
            )
            for split in ["development", "test"]
        ]
        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with tempfile.TemporaryDirectory() as folder, patch("v3.experiment.LAYER", 1):
                result = run_experiment(self.measure, Path(folder), items)
                rows = [
                    json.loads(line)
                    for line in (Path(folder) / "observations.jsonl").read_text().splitlines()
                ]
            self.assertEqual(result["interventions"], 288)
            self.assertEqual(result["numerical_checks"]["maximum_identity_error"], 0)
            self.assertEqual(rows[0]["record"], "frozen_selection")
            self.assertTrue(all(r["families"] == 1 for r in result["summaries"]))
        finally:
            torch.set_num_threads(old_threads)


class AggregationTests(unittest.TestCase):
    def test_random_seeds_and_items_do_not_become_independent_families(self):
        rows = []
        for family in ["one", "two"]:
            for item in ["a", "b"]:
                for kind in ["cross", "identity", "lexical", "random"]:
                    for seed in range(3) if kind == "random" else [None]:
                        for verb in ["optional", "intransitive"]:
                            for direction, sign in [("forward", 1), ("reverse", -1)]:
                                value = sign * (3 if verb == "optional" and kind == "cross" else 0)
                                rows.append(
                                    dict(
                                        split="test",
                                        family=family,
                                        id=family + item,
                                        contrast="comma_aligned",
                                        stage="next",
                                        kind=kind,
                                        seed=seed,
                                        verb_class=verb,
                                        direction=direction,
                                        effects={
                                            "margin": value,
                                            "was_logp": value,
                                            "comma_logp": 0,
                                        },
                                    )
                                )
        summary = summarize_effects(rows)
        row = next(r for r in summary if r["kind"] == "cross" and r["metric"] == "margin")
        self.assertEqual(row["mean"], 3)
        self.assertEqual(row["families"], 2)
        self.assertEqual(row["ci95"], [3, 3])
        with self.assertRaises(ValueError):
            summarize_effects(rows[:-1])


if __name__ == "__main__":
    unittest.main()
