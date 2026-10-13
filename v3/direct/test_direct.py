import unittest

import numpy as np

from v3.direct.experiment import decisions, materials
from v3.direct.measure import sequence_logps


class DirectTests(unittest.TestCase):
    def test_no_context_gold_reverses_without_changing_prompt(self):
        rows = materials()
        self.assertEqual(len(rows), 96)
        self.assertEqual({r["split"] for r in rows}, {"development"})
        for row in rows:
            self.assertNotIn("\nA:", row["prompt"])
            if row["condition"] != "no_context" or row["reversal"]:
                continue
            other = next(
                r
                for r in rows
                if all(r[k] == row[k] for k in ["family", "voice", "role", "condition"])
                and r["reversal"] == 1
            )
            self.assertEqual(row["prompt"], other["prompt"])
            self.assertNotEqual(row["gold_index"], other["gold_index"])

    def test_constant_noun_fails_but_role_tracking_passes(self):
        rows = materials()
        for row in rows:
            row["margin"] = 1 if row["gold_index"] == 0 else -1
        self.assertFalse(decisions(rows)["passed"])
        for row in rows:
            row["margin"] = 1
        self.assertTrue(decisions(rows)["passed"])
        with self.assertRaises(ValueError):
            decisions(rows[:-1])

    def test_teacher_forcing_scores_only_suffix_with_correct_shift(self):
        import torch

        logits = torch.tensor([[0.0, 1.0, 2.0], [2.0, 0.0, 1.0], [1.0, 2.0, 0.0]])
        ids = torch.tensor([0, 2, 1])
        actual = sequence_logps(logits, ids, 1)
        expected = [
            float(torch.log_softmax(logits[0], -1)[2]),
            float(torch.log_softmax(logits[1], -1)[1]),
        ]
        np.testing.assert_allclose(actual, expected)
        self.assertEqual(len(sequence_logps(logits, ids, 2)), 1)
        with self.assertRaises(ValueError):
            sequence_logps(logits, ids, 0)


class ModelIntegrationTests(unittest.TestCase):
    def test_actual_causal_model_scores_independently_of_candidate_order(self):
        from v3.bridge.test_bridge import MeasurementTests
        from v3.direct.measure import DirectMeasurements

        MeasurementTests.setUpClass()
        tiny = MeasurementTests.measure
        measure = DirectMeasurements(tiny.model, tiny.tokenizer)
        candidates = [" the guard", " the prisoner"]
        a = measure.score("the guard Answer:", candidates)
        b = measure.score("the guard Answer:", candidates[::-1])
        self.assertEqual(a["candidates"], b["candidates"][::-1])
        self.assertTrue(
            all(len(c["token_logp"]) == len(c["token_ids"]) == 2 for c in a["candidates"])
        )
