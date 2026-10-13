import unittest

import torch

from v2.experiment import Measurements, family_interval, patch_state, token_position
from v2.materials import garden_paths


class MaterialsTests(unittest.TestCase):
    def test_comma_is_the_only_clean_difference(self):
        for item in garden_paths():
            self.assertEqual(item["clean"].replace(",", ""), item["ambiguous"])
            self.assertNotIn(item["critical"], item["ambiguous"].split())


class MeasurementTests(unittest.TestCase):
    def test_patch_changes_only_requested_position_and_does_not_mutate(self):
        original = torch.zeros(1, 4, 3)
        donor = torch.ones(3)
        patched = patch_state((original, "cache"), 2, donor)
        self.assertEqual(patched[1], "cache")
        self.assertEqual(original.sum().item(), 0)
        self.assertEqual(patched[0].sum().item(), 3)
        self.assertTrue(torch.equal(patched[0][0, 2], donor))
        self.assertTrue(torch.equal(patch_state(original, 0, donor)[0, 0], donor))

    def test_alignment_requires_exact_word_end_and_rejects_missing_span(self):
        offsets = [(0, 0), (0, 3), (3, 6), (6, 9)]
        self.assertEqual(token_position(offsets, 4, 9), 3)
        with self.assertRaises(ValueError):
            token_position(offsets, 10, 15)
        with self.assertRaises(ValueError):
            token_position(offsets, 4, 8)

    def test_family_resampling_does_not_count_repeated_items_as_families(self):
        result = family_interval([("a", 0), ("a", 0), ("b", 6)])
        self.assertEqual(result["mean"], 3)
        self.assertEqual(result["families"], 2)
        with self.assertRaises(ValueError):
            family_interval([("a", float("nan"))])


class TransformerIntegrationTests(unittest.TestCase):
    """Random tiny Gemma verifies hooks/scoring, not linguistic competence."""

    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import Gemma3ForCausalLM, Gemma3TextConfig, PreTrainedTokenizerFast

        words = ["[UNK]", "While", "the", "guard", "watched", ",", "prisoner", "was"]
        backend = Tokenizer(
            WordLevel(dict(zip(words, range(len(words)), strict=True)), unk_token="[UNK]")
        )
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
        config = Gemma3TextConfig(
            vocab_size=len(words),
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=12,
            sliding_window=32,
            sliding_window_pattern=2,
        )
        config._attn_implementation = "eager"
        torch.manual_seed(0)
        cls.measure = Measurements(Gemma3ForCausalLM(config).eval(), tokenizer)

    def test_self_patch_is_identity_and_last_layer_copies_readout(self):
        target = "While the guard watched the prisoner"
        donor = "While the guard watched, the prisoner"
        captured, pos = self.measure.capture(target)
        donor_states, _ = self.measure.capture(donor)
        baseline = self.measure.score(target, "was")
        for layer, state in enumerate(captured):
            self.assertAlmostEqual(
                self.measure.score(target, "was", (layer, pos, state)), baseline, places=6
            )
        patched = self.measure.score(target, "was", (2, pos, donor_states[2]))
        self.assertAlmostEqual(patched, self.measure.score(donor, "was"), places=6)

    def test_hooks_are_removed_when_forward_fails(self):
        with self.assertRaises(IndexError):
            self.measure.score("the guard", "was", (0, 999, torch.zeros(24)))
        self.assertFalse(self.measure.layers[0]._forward_hooks)

    def test_capture_uses_the_scoring_shape_without_reading_future_positions(self):
        prefix = "While the guard watched the prisoner"
        short, pos = self.measure.capture(prefix)
        full, full_pos = self.measure.capture(prefix, continuation="was")
        self.assertEqual(pos, full_pos)
        for a, b in zip(short, full, strict=True):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        self.assertEqual(
            self.measure.score(prefix, "was", (1, pos, full[1])), self.measure.score(prefix, "was")
        )


if __name__ == "__main__":
    unittest.main()
