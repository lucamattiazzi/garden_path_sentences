import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from v4.experiment import competence, components, run_experiment
from v4.materials import items, make_prompt


class MaterialTests(unittest.TestCase):
    def test_factorial_mapping_and_common_prefix(self):
        item = items()[0]
        for reversal in [0, 1]:
            first = reversal
            for construction, first_is_agent, first_is_subject in [
                ("active", True, True),
                ("passive", False, True),
                ("object_relative", False, False),
                ("passive_relative", True, False),
            ]:
                a = make_prompt(item, reversal, construction, "agent", "clean")
                b = make_prompt(item, reversal, construction, "patient", "clean")
                self.assertEqual(a["gold_index"], first if first_is_agent else 1 - first)
                self.assertNotEqual(a["gold_index"], b["gold_index"])
                self.assertEqual(a["event_subject_index"], first if first_is_subject else 1 - first)
                prefix = a["text"][: a["site_end"]]
                if construction == "active":
                    reference = prefix
                self.assertEqual(prefix, reference)
        self.assertEqual(len({i["family"] for i in items()}), 12)
        self.assertEqual(sum(i["split"] == "development" for i in items()), 6)

    def test_components_distinguish_common_bias_and_question_interaction(self):
        self.assertEqual(components(2.0, 2.0), {"common_first": 2.0, "question_difference": 0.0})
        self.assertEqual(components(2.0, -2.0), {"common_first": 0.0, "question_difference": 2.0})


class GateTests(unittest.TestCase):
    def test_failed_gate_never_uses_confirmation_or_patches(self):
        class Fixed:
            model = torch.nn.Linear(2, 2)
            layers = [None] * 48

            def evaluate(self, p, **kwargs):
                return dict(
                    answer_logp=[-1.0, -2.0], next_margin=0.0, input_ids=[1], position_ids=[0]
                ), torch.zeros(2)

        material = [dict(items()[0], split="development"), dict(items()[1], split="confirmation")]
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("v4.experiment.items", return_value=material),
        ):
            r = run_experiment(Fixed(), Path(folder), 32)
        self.assertFalse(r["gate"]["passed"])
        self.assertEqual(r["baselines"], 16)
        self.assertEqual(r["interventions"], 0)

    def test_every_construction_and_role_must_pass(self):
        rows = []
        for construction in ["active", "passive", "object_relative", "passive_relative"]:
            for role in ["agent", "patient"]:
                for reversal in [0, 1]:
                    rows.append(
                        dict(
                            construction=construction,
                            role=role,
                            family="f",
                            reversal=reversal,
                            margin=1.0,
                        )
                    )
        self.assertTrue(competence(rows)["passed"])
        rows[-1]["margin"] = -1
        self.assertFalse(competence(rows)["passed"])


class IntegrationTests(unittest.TestCase):
    def test_tiny_success_branch_alignment_identity_and_counts(self):
        from v3.bridge.test_bridge import MeasurementTests
        from v4.measure import BridgeMeasurements

        MeasurementTests.setUpClass()
        m = MeasurementTests.measure
        measure = BridgeMeasurements(m.model, m.tokenizer)
        material = [
            dict(
                id=s,
                family=s,
                split=s,
                verb="watched",
                actor="guard",
                lexical_actor="prisoner",
                participants=["guard", "prisoner"],
                event="watched",
            )
            for s in ["development", "confirmation"]
        ]
        previous = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with (
                tempfile.TemporaryDirectory() as folder,
                patch("v4.experiment.items", return_value=material),
                patch("v4.experiment.competence", return_value={"passed": True}),
            ):
                r = run_experiment(measure, Path(folder), 1)
            self.assertEqual(r["baselines"], 48)
            self.assertEqual(r["interventions"], 128)
            self.assertEqual(r["maximum_identity_error"], 0)
        finally:
            torch.set_num_threads(previous)


class PaddingTests(unittest.TestCase):
    def test_unequal_answers_use_equal_forward_shapes_without_scoring_padding(self):
        from v3.bridge.test_bridge import MeasurementTests
        from v4.measure import BridgeMeasurements

        MeasurementTests.setUpClass()
        m = MeasurementTests.measure
        measure = BridgeMeasurements(m.model, m.tokenizer)
        p = make_prompt(items()[0], 0, "active", "agent", "clean")
        p["candidates"] = [" guard.", " prisoner guard."]
        shapes = []
        h = m.model.register_forward_pre_hook(
            lambda _m, _a, kwargs: shapes.append(tuple(kwargs["input_ids"].shape)), with_kwargs=True
        )
        previous = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            score, _ = measure.evaluate(p, layer=1, capture=True)
        finally:
            h.remove()
            torch.set_num_threads(previous)
        self.assertEqual(shapes[0], shapes[1])
        self.assertNotEqual(
            len(score["answer_tokens"][0]["token_ids"]), len(score["answer_tokens"][1]["token_ids"])
        )
