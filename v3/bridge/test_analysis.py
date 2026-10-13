import json
import shutil
import tempfile
import unittest
from pathlib import Path

from v3.bridge.analyze import aggregate_behavior, aggregate_patches, audit_precision


class AnalysisTests(unittest.TestCase):
    def test_pairing_does_not_credit_a_fixed_answer_label(self):
        rows = [
            dict(
                split="test",
                condition="simple",
                voice="active",
                family="f",
                id="x",
                reversal=0,
                role="agent",
                order=i,
                correct_margin=1 if i == 0 else -1,
            )
            for i in [0, 1]
        ]
        values = aggregate_behavior(rows)
        self.assertEqual(values["test", "simple", "active", "accuracy"]["f"], 0.5)
        self.assertEqual(values["test", "simple", "active", "both_orders"]["f"], 0)
        with self.assertRaises(ValueError):
            aggregate_behavior(rows[:1])

    def test_seeds_and_directions_are_averaged_before_family_inference(self):
        rows = []
        for direction, sign in [("clean_to_gap", 1), ("gap_to_clean", -1)]:
            for kind, values in [("cross", [4]), ("lexical", [1]), ("random", [0, 1, 2])]:
                for seed, value in enumerate(values):
                    rows.append(
                        dict(
                            split="test",
                            family="f",
                            id="x",
                            reversal=0,
                            role="agent",
                            order=0,
                            direction=direction,
                            kind=kind,
                            seed=seed,
                            qa_effect=sign * value,
                            next_effect=sign * value,
                        )
                    )
        values = aggregate_patches(rows)
        self.assertEqual(values["test", "average", "cross_minus_random", "qa_effect"]["f"], 3)
        self.assertEqual(values["test", "average", "cross_minus_lexical", "qa_effect"]["f"], 3)
        with self.assertRaises(ValueError):
            aggregate_patches(rows[:-1])


class PrecisionAuditTests(unittest.TestCase):
    def test_rejects_a_changed_prompt_in_precision_diagnostic(self):
        root = Path(__file__).parent / "results"
        parent = root / "bridge-v3-20261008-02"
        original = root / "bridge-precision-20261008-01"
        self.assertFalse(audit_precision(parent, original)["gate"]["passed"])
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "diagnostic"
            shutil.copytree(original, target)
            path = target / "observations.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["prompt"] += " changed"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with self.assertRaises(ValueError):
                audit_precision(parent, target)
