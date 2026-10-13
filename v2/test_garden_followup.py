import tempfile
import unittest
from pathlib import Path

import torch

from v2.garden_followup import (
    _intervention_interactions,
    factorial_interaction,
    garden_materials,
    matched_state,
    run_garden_followup,
    select_layer,
)


class FakeMeasurements:
    """Deterministic hook stand-in; gives no evidence about language models."""

    def __init__(self):
        self.layers = [None] * 4
        self.model = torch.nn.Linear(4, 4)
        self.calls = []

    def encode(self, text):
        return {"inputs": {"input_ids": torch.zeros(1, len(text.split()), dtype=torch.long)}}

    def capture(self, prefix, continuation=""):
        state = torch.tensor(
            [float("," in prefix), float(len(prefix)), sum(map(ord, prefix)) / 100.0, 1.0]
        )
        return [state + i for i in range(4)], len(prefix.split()) - 1

    def score(self, prefix, word, patch=None):
        states, _ = self.capture(prefix)
        state = states[0] if patch is None else patch[2] - patch[0]
        self.calls.append((prefix, None if patch is None else patch[0]))
        return float(state.sum()) * (1 if word.startswith("was") else -0.5)


class GardenFollowupTests(unittest.TestCase):
    def test_materials_cross_punctuation_and_verb_class_with_fresh_splits(self):
        items = garden_materials()
        self.assertEqual(len(items), 12)
        self.assertEqual(len({r["family"] for r in items}), 12)
        groups = {s: [r for r in items if r["split"] == s] for s in ("development", "test")}
        self.assertEqual([len(x) for x in groups.values()], [6, 6])
        for item in items:
            for verb_class in ("optional_transitive", "intransitive"):
                prefixes = item["prefixes"][verb_class]
                self.assertEqual(prefixes["comma"].replace(",", ""), prefixes["no_comma"])
                for comma in ("comma", "no_comma"):
                    self.assertEqual(
                        item["lexical_prefixes"][verb_class][comma].replace(
                            item["other_actor"], item["actor"]
                        ),
                        prefixes[comma],
                    )
            self.assertEqual(set(item["continuations"]), {"main", "object"})
        old = {"washed", "dressed", "hunted", "watched", "visited", "called", "trained", "read"}
        self.assertFalse(old & {r["family"] for r in items})

    def test_difference_in_differences_removes_shared_punctuation_effect(self):
        values = {
            "optional_transitive": {"comma": 9.0, "no_comma": 3.0},
            "intransitive": {"comma": 7.0, "no_comma": 3.0},
        }
        self.assertEqual(factorial_interaction(values), 2.0)

    def test_bidirectional_interaction_keeps_direction_signs(self):
        effects = {
            "optional_transitive": {"no_comma": (3.0, -1.0), "comma": (-2.0, 1.0)},
            "intransitive": {"no_comma": (1.0, 0.0), "comma": (-1.0, 0.0)},
        }
        rows = [
            {
                "split": "development",
                "family": "example",
                "layer": 1,
                "kind": "cross_condition",
                "random_seed": None,
                "verb_class": verb_class,
                "recipient": recipient,
                "candidate": candidate,
                "effect": value,
            }
            for verb_class, directions in effects.items()
            for recipient, values in directions.items()
            for candidate, value in zip(("main", "object"), values, strict=True)
        ]
        result = _intervention_interactions(rows)[0]
        self.assertEqual(result["to_no_comma"], 3.0)
        self.assertEqual(result["to_comma"], 2.0)
        self.assertEqual(result["value"], 2.5)

    def test_norm_matching_records_bfloat16_rounding(self):
        target = torch.tensor([20.0, 1.0, -4.0])
        difference = torch.tensor([0.0, 3.0, 4.0])
        actual, info = matched_state(target, difference, 2.0, torch.bfloat16)
        self.assertEqual(info["requested_l2"], 2.0)
        self.assertAlmostEqual(info["actual_l2"], float(torch.linalg.vector_norm(actual - target)))
        self.assertTrue(torch.equal(actual, actual.to(torch.bfloat16).float()))
        with self.assertRaises(ValueError):
            matched_state(target, torch.zeros(3), 2.0, torch.float32)

    def test_layer_selection_cannot_read_test_or_final_block(self):
        rows = [
            {"split": "development", "layer": 0, "value": 1.0},
            {"split": "development", "layer": 1, "value": 3.0},
        ]
        self.assertEqual(select_layer(rows, 3), 1)
        with self.assertRaises(ValueError):
            select_layer(rows + [{"split": "test", "layer": 1, "value": 10}], 3)
        with self.assertRaises(ValueError):
            select_layer(rows + [{"split": "development", "layer": 2, "value": 10}], 3)

    def test_complete_flow_only_intervenes_on_selected_block_in_test(self):
        measure = FakeMeasurements()
        with tempfile.TemporaryDirectory() as folder:
            result = run_garden_followup(measure, Path(folder))
            self.assertTrue((Path(folder) / "garden-followup-observations.jsonl").is_file())
        chosen = result["selection"]["layer"]
        rows = result["interventions"]
        self.assertEqual({r["layer"] for r in rows if r["split"] == "test"}, {chosen})
        self.assertNotIn(3, {r["layer"] for r in rows})
        self.assertEqual(
            {r["kind"] for r in rows if r["split"] == "test"},
            {"cross_condition", "identity", "lexical", "random"},
        )
        self.assertEqual({r["random_seed"] for r in rows if r["kind"] == "random"}, {0, 1, 2})
        self.assertTrue(all(abs(r["effect"]) < 1e-5 for r in rows if r["kind"] == "identity"))
        self.assertEqual(result["exclusions"], [])


class TinyTransformerTests(unittest.TestCase):
    def test_complete_multi_token_assay_with_real_hooks(self):
        """Random tiny Gemma checks the implementation, not any linguistic claim."""
        import re

        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import Gemma3ForCausalLM, Gemma3TextConfig, PreTrainedTokenizerFast

        from v2.experiment import Measurements

        texts = []
        for item in garden_materials():
            for key in ("prefixes", "lexical_prefixes"):
                texts.extend(
                    text for conditions in item[key].values() for text in conditions.values()
                )
            texts.extend(item["continuations"].values())
        words = ["[UNK]"] + sorted(set(re.findall(r"\w+|[^\w\s]", " ".join(texts))))
        backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
        config = Gemma3TextConfig(
            vocab_size=len(words),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            sliding_window=32,
            sliding_window_pattern=2,
        )
        config._attn_implementation = "eager"
        torch.manual_seed(10)
        measure = Measurements(Gemma3ForCausalLM(config).eval(), tokenizer)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as folder:
                result = run_garden_followup(measure, Path(folder))
        finally:
            torch.set_num_threads(threads)
        identities = [r["effect"] for r in result["interventions"] if r["kind"] == "identity"]
        self.assertLess(max(map(abs, identities)), 1e-5)
        self.assertTrue(all(r["completion_tokens"] > 1 for r in result["baselines"]))


if __name__ == "__main__":
    unittest.main()
