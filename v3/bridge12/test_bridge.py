import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from v3.bridge12.experiment import make_prompt, run_experiment
from v3.bridge12.measure import BridgeMeasurements


class BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from v3.bridge.test_bridge import MeasurementTests

        MeasurementTests.setUpClass()
        tiny = MeasurementTests.measure
        cls.measure = BridgeMeasurements(tiny.model, tiny.tokenizer)
        cls.item = dict(
            id="f",
            family="f",
            split="development",
            verb="watched",
            actor="guard",
            lexical_actor="prisoner",
            participants=["guard", "prisoner"],
            event="watched",
        )

    def test_identity_and_future_answer_invariance(self):
        p = make_prompt(self.item, 0, "agent", "gap")
        scores, state = self.measure.evaluate(p, layer=1, capture=True)
        patched, _ = self.measure.evaluate(p, layer=1, donor=state)
        self.assertEqual(scores, patched)
        swapped = dict(p, candidates=p["candidates"][::-1])
        other, other_state = self.measure.evaluate(swapped, layer=1, capture=True)
        torch.testing.assert_close(state, other_state)
        self.assertEqual(scores["next_margin"], other["next_margin"])
        self.assertEqual(scores["answer_logp"], other["answer_logp"][::-1])

    def test_gate_failure_never_touches_test_or_interventions(self):
        class Fixed:
            model = torch.nn.Linear(2, 2)
            layers = [None] * 48

            def evaluate(self, p, **kwargs):
                return dict(
                    answer_logp=[-1.0, -2.0], next_margin=0.0, input_ids=[1], position_ids=[0]
                ), torch.zeros(2)

        with (
            tempfile.TemporaryDirectory() as folder,
            patch(
                "v3.bridge12.experiment.items",
                return_value=[self.item, dict(self.item, id="test", family="test", split="test")],
            ),
        ):
            result = run_experiment(Fixed(), Path(folder))
        self.assertFalse(result["gate"]["passed"])
        self.assertEqual(result["baselines"], 16)
        self.assertEqual(result["interventions"], 0)

    def test_full_tiny_pipeline_has_exact_identity(self):
        material = [self.item, dict(self.item, id="test", family="test", split="test")]
        old = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with (
                tempfile.TemporaryDirectory() as folder,
                patch("v3.bridge12.experiment.items", return_value=material),
                patch("v3.bridge12.experiment.LAYER", 1),
                patch("v3.bridge12.experiment.competence", return_value={"passed": True}),
            ):
                result = run_experiment(self.measure, Path(folder))
            self.assertEqual(result["baselines"], 32)
            self.assertEqual(result["interventions"], 96)
            self.assertEqual(result["maximum_identity_error"], 0)
        finally:
            torch.set_num_threads(old)
