import json
import unittest

import torch

from v3.scale.measure import ScaleMeasurements
from v3.scale.run import next_size, validate_loading


class ScaleTests(unittest.TestCase):
    def test_selection_stops_at_first_pass_or_after_12b(self):
        self.assertEqual(next_size([]), "4b")
        self.assertEqual(next_size([False]), "12b")
        self.assertIsNone(next_size([True]))
        self.assertIsNone(next_size([False, True]))
        self.assertIsNone(next_size([False, False]))

    def test_missing_checkpoint_weights_abort(self):
        validate_loading(
            {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}
        )
        with self.assertRaises(ValueError):
            validate_loading({"missing_keys": ["lm_head.weight"]})

    def test_loading_metadata_is_json_serializable(self):
        info = {
            "missing_keys": set(),
            "unexpected_keys": set(),
            "mismatched_keys": set(),
            "error_msgs": [],
        }
        self.assertEqual(json.loads(json.dumps(validate_loading(info))), {k: [] for k in info})

    def test_text_scores_through_multimodal_wrapper(self):
        from transformers import Gemma3Config, Gemma3ForConditionalGeneration, SiglipVisionConfig

        from v3.bridge.test_bridge import MeasurementTests

        MeasurementTests.setUpClass()
        tiny = MeasurementTests.measure
        config = Gemma3Config(
            text_config=tiny.model.config.to_dict(),
            vision_config=SiglipVisionConfig(
                hidden_size=24,
                intermediate_size=48,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=16,
                patch_size=4,
            ).to_dict(),
        )
        config._attn_implementation = "eager"
        model = Gemma3ForConditionalGeneration(config).eval()
        measure = ScaleMeasurements(model, tiny.tokenizer)
        result = measure.score("the guard Answer:", [" the guard", " the prisoner"])
        self.assertEqual(len(measure.layers), 3)
        self.assertEqual(len(result["candidates"]), 2)
        self.assertTrue(all(torch.isfinite(torch.tensor(c["logp"])) for c in result["candidates"]))
