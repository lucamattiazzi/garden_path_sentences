import unittest
from dataclasses import replace
from unittest.mock import Mock

from research.behavioral import aggregate_results, incremental_scores, score_question
from research.pilot import load_pilot, validate_pilot


class PilotTests(unittest.TestCase):
    def test_pilot_has_disjoint_lexical_families_and_balanced_roles(self):
        items = load_pilot()
        validate_pilot(items)
        self.assertEqual(len(items), 24)
        exploration = {p.family for p in items if p.split == "exploration"}
        validation = {p.family for p in items if p.split == "validation"}
        self.assertFalse(exploration & validation)
        for item in items:
            self.assertEqual({q.actor for q in item.questions}, {0, 1})

    def test_rejects_a_family_crossing_splits(self):
        items = load_pilot()
        items[1] = replace(items[1], split="validation")
        with self.assertRaisesRegex(ValueError, "family"):
            validate_pilot(items)

    def test_rejects_missing_disambiguator_and_duplicate_ids(self):
        items = load_pilot()
        with self.assertRaises(ValueError):
            validate_pilot([replace(items[0], gps="A different sentence.")])
        with self.assertRaises(ValueError):
            validate_pilot([items[0], items[0]])


class BehaviouralScoringTests(unittest.TestCase):
    def test_margin_is_correct_minus_incorrect_log_probability_in_both_orders(self):
        scorer = Mock(side_effect=[1.0, 3.0, 5.0, 1.0])
        row = score_question(load_pilot()[0], 0, "gps", scorer)
        self.assertEqual(row["mean_correct_log_odds"], 3.0)
        self.assertEqual(row["accuracy"], 1.0)
        self.assertTrue(row["both_orders_correct"])

    def test_always_a_cannot_pass_the_order_robust_measure(self):
        row = score_question(load_pilot()[0], 0, "gps", Mock(side_effect=[1.0, 3.0, 1.0, 3.0]))
        self.assertEqual(row["mean_correct_log_odds"], 0.0)
        self.assertEqual(row["accuracy"], 0.5)
        self.assertFalse(row["both_orders_correct"])

    def test_counterfactual_reverses_gold_actor_and_retains_candidate_set(self):
        item = load_pilot()[0]
        scorer = Mock(return_value=1.0)
        row = score_question(item, 0, "role_reversed", scorer)
        self.assertEqual(row["orders"][0]["correct_label"], "B")
        prompt = scorer.call_args_list[0].args[0]
        self.assertIn(item.role_reversed, prompt)
        self.assertIn("A. " + item.actors[0], prompt)
        self.assertIn("B. " + item.actors[1], prompt)

    def test_rejects_nonfinite_scores_and_ties_are_not_successes(self):
        item = load_pilot()[0]
        with self.assertRaises(ValueError):
            score_question(item, 0, "gps", Mock(return_value=float("nan")))
        row = score_question(item, 0, "gps", Mock(return_value=1.0))
        self.assertFalse(row["both_orders_correct"])
        self.assertEqual(row["accuracy"], 0.0)

    def test_no_context_prompt_omits_the_sentence(self):
        scorer = Mock(return_value=1.0)
        item = load_pilot()[0]
        score_question(item, 0, "no_context", scorer)
        self.assertNotIn(item.gps, scorer.call_args.args[0])

    def test_aggregation_resamples_families_not_answer_orders(self):
        rows = []
        for family, item, value in (("a", 1, 0.0), ("a", 2, 0.0), ("b", 3, 6.0)):
            for question in (0, 1):
                for condition, cost in (
                    ("normal", 0.0),
                    ("gps", value),
                    ("role_reversed", 0.0),
                    ("no_context", 0.0),
                ):
                    rows.append(
                        {
                            "id": item,
                            "family": family,
                            "template": "NP_S",
                            "question": question,
                            "condition": condition,
                            "mean_correct_log_odds": cost,
                            "accuracy": 0.5,
                            "both_orders_correct": False,
                        }
                    )
        summary = aggregate_results(rows)
        self.assertEqual(summary["overall"]["gps_minus_normal"]["n"], 2)
        self.assertEqual(summary["overall"]["gps_minus_normal"]["mean"], 3.0)
        with self.assertRaisesRegex(ValueError, "paired"):
            aggregate_results(rows[:-1])

    def test_incremental_scoring_never_exposes_future_words(self):
        item = load_pilot()[0]
        scorer = Mock(return_value=2.0)
        result = incremental_scores(item, scorer)
        self.assertEqual(scorer.call_args_list[0].args, (item.gps_prefix, item.critical_word))
        self.assertEqual(
            scorer.call_args_list[2].args,
            (item.gps_prefix + " " + item.critical_word, item.spillover.split()[0]),
        )
        self.assertEqual(len(result["words"]), 1 + len(item.spillover.split()))


if __name__ == "__main__":
    unittest.main()
