import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from model_finder.sentences import by_template
from model_finder.surprisal import critical_token_positions

ROOT = Path(__file__).resolve().parents[1]
ABLATION_PATH = ROOT / "qwen4b" / "scripts" / "_ablation.py"
SPEC = importlib.util.spec_from_file_location("gps_ablation", ABLATION_PATH)
assert SPEC and SPEC.loader
ABLATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ABLATION)
HeadsAblated = ABLATION.HeadsAblated


class OffsetTokenizer:
    def __init__(self, offsets):
        self.offsets = offsets

    def __call__(self, _text, **kwargs):
        if kwargs.get("return_offsets_mapping"):
            return {"input_ids": [0, 1, 2], "offset_mapping": self.offsets}
        return {"input_ids": [0, 1, 2]}


class TinyModel:
    def __init__(self):
        layers = []
        for _ in range(2):
            projection = torch.nn.Linear(4, 3, bias=False)
            layers.append(SimpleNamespace(self_attn=SimpleNamespace(o_proj=projection)))
        self.model = SimpleNamespace(layers=layers)


class CriticalTokenTests(unittest.TestCase):
    def test_accepts_leading_space_in_word_offset(self):
        tokenizer = OffsetTokenizer([(0, 0), (0, 5), (5, 11)])
        ids, indices = critical_token_positions(tokenizer, "hello", "world")
        self.assertEqual(ids, [0, 1, 2])
        self.assertEqual(indices, [2])

    def test_rejects_token_that_consumes_prefix_text(self):
        tokenizer = OffsetTokenizer([(0, 0), (0, 4), (4, 11)])
        _, indices = critical_token_positions(tokenizer, "hello", "world")
        self.assertEqual(indices, [])


class HeadsAblatedTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyModel()
        self.layers = self.model.model.layers
        self.original = [layer.self_attn.o_proj.weight.detach().clone() for layer in self.layers]

    def assert_weights_unchanged(self):
        for layer, original in zip(self.layers, self.original, strict=True):
            self.assertTrue(torch.equal(layer.self_attn.o_proj.weight, original))

    def test_duplicate_target_is_restored(self):
        with HeadsAblated(self.model, 2, [(0, 1), (0, 1)]):
            self.assertTrue(torch.count_nonzero(self.layers[0].self_attn.o_proj.weight[:, 2:]).item() == 0)
        self.assert_weights_unchanged()

    def test_invalid_target_does_not_partially_mutate_model(self):
        with (
            self.assertRaises(ValueError),
            HeadsAblated(self.model, 2, [(0, 0), (5, 0)]),
        ):
            pass
        self.assert_weights_unchanged()

    def test_body_exception_still_restores_model(self):
        with (
            self.assertRaisesRegex(RuntimeError, "boom"),
            HeadsAblated(self.model, 2, [(1, 0)]),
        ):
            raise RuntimeError("boom")
        self.assert_weights_unchanged()


class MeasurementTests(unittest.TestCase):
    def test_invalid_scores_abort_instead_of_changing_the_paired_sample(self):
        pair = by_template("NV")[0]
        for measure in (
            ABLATION.measure_effect,
            ABLATION.measure_verb_continuation,
            ABLATION.measure_lexical_baseline,
        ):
            for invalid in (float("nan"), float("inf"), float("-inf")):
                with (
                    self.subTest(measure=measure.__name__, invalid=invalid),
                    patch.object(ABLATION, "surprisal", side_effect=[2.0, invalid] + [1.0] * 8),
                    self.assertRaisesRegex(ValueError, "Non-finite surprisal"),
                ):
                    measure(None, None, [pair], "cpu")

    def test_invalid_control_aborts_the_measurement(self):
        with (
            patch.object(ABLATION, "surprisal", side_effect=[2.0, 1.0, float("nan")] + [1.0] * 7),
            self.assertRaisesRegex(ValueError, "Non-finite surprisal"),
        ):
            ABLATION.measure_effect(None, None, by_template("NV")[:1], "cpu")

    def test_valid_paired_means_preserve_the_difference(self):
        with patch.object(ABLATION, "surprisal", side_effect=[4.0, 1.0, 2.0, 3.0] + [1.0] * 8):
            summary, rows = ABLATION.measure_effect(None, None, by_template("NV")[:2], "cpu")
        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["mean_delta"], 1.0)
        self.assertEqual(summary["mean_s_gps"] - summary["mean_s_normal"], 1.0)


if __name__ == "__main__":
    unittest.main()
