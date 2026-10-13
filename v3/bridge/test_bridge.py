import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from v3.bridge.experiment import behavioral_summary, gate, run_experiment
from v3.bridge.materials import items, prompt, sentence
from v3.bridge.measure import QuestionMeasurements


class MaterialsTests(unittest.TestCase):
    def test_reversal_and_voice_flip_role_with_same_participant_inventory(self):
        for item in items():
            for role in ["agent", "patient"]:
                base = prompt(item, 0, "active", "clean", role, 0)
                reverse = prompt(item, 1, "active", "clean", role, 0)
                passive = prompt(item, 0, "passive", "clean", role, 0)
                self.assertNotEqual(base["correct"], reverse["correct"])
                self.assertEqual(reverse["correct"], passive["correct"])
                self.assertNotEqual(
                    base["gold_index"], prompt(item, 0, "active", "clean", role, 1)["gold_index"]
                )
            for voice in ["active", "passive"]:
                plain, _, _ = sentence(item, 0, voice, "native")
                clean, _, _ = sentence(item, 0, voice, "clean")
                self.assertEqual(clean.replace(",", ""), plain)
                self.assertEqual(sentence(item, 0, voice, "gap")[0], plain)
        dev = {r["family"] for r in items() if r["split"] == "development"}
        test = {r["family"] for r in items() if r["split"] == "test"}
        self.assertEqual(len(dev), 6)
        self.assertEqual(len(test), 6)
        self.assertFalse(dev & test)

    def test_no_context_prompts_are_identical_under_counterfactual_gold_labels(self):
        item = items()[0]
        a = prompt(item, 0, "active", "no_context", "agent", 0)
        b = prompt(item, 0, "passive", "no_context", "agent", 0)
        self.assertEqual(a["text"], b["text"])
        self.assertNotEqual(a["gold_index"], b["gold_index"])


class MeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import Gemma3ForCausalLM, Gemma3TextConfig, PreTrainedTokenizerFast

        words = [
            "[UNK]",
            "While",
            "the",
            "guard",
            "watched",
            ",",
            "prisoner",
            "was",
            "A",
            "B",
            "Question",
            ":",
            "who",
            "Answer",
        ]
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
            sliding_window=512,
            sliding_window_pattern=2,
        )
        config._attn_implementation = "eager"
        torch.manual_seed(0)
        cls.measure = QuestionMeasurements(Gemma3ForCausalLM(config).eval(), tokenizer)

    def test_same_forward_identity_and_future_question_cannot_change_earlier_state(self):
        prefix = "While the guard watched the prisoner"
        text = prefix + " was the guard Question: who Answer:"
        base, state, site = self.measure.evaluate(text, len(prefix), None, layer=1, capture=True)
        changed, _, _ = self.measure.evaluate(text, len(prefix), None, layer=1, patch=state)
        self.assertEqual(base["answer_logp"], changed["answer_logp"])
        self.assertEqual(base["next_margin"], changed["next_margin"])
        _, other, _ = self.measure.evaluate(
            prefix + " was the prisoner Question: guard Answer:",
            len(prefix),
            None,
            layer=1,
            capture=True,
        )
        torch.testing.assert_close(state, other, rtol=1e-5, atol=1e-6)
        self.assertEqual(site, 5)

    def test_unknown_answer_labels_and_malformed_site_fail(self):
        text = "the guard Answer:"
        with self.assertRaises(ValueError):
            self.measure.evaluate(text, 2, None, layer=1)
        with self.assertRaises(IndexError):
            self.measure.evaluate(text, len("the guard"), None, layer=999, capture=True)
        self.assertTrue(all(not layer._forward_hooks for layer in self.measure.layers))

    def test_complete_tiny_bridge_preserves_alignment_and_identity(self):
        material = [
            dict(
                id=split,
                family=split,
                split=split,
                verb="watched",
                actor="guard",
                lexical_actor="prisoner",
                participants=["guard", "prisoner"],
                event="watched",
            )
            for split in ["development", "test"]
        ]
        previous = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with (
                tempfile.TemporaryDirectory() as folder,
                patch("v3.bridge.experiment.items", return_value=material),
                patch("v3.bridge.experiment.LAYER", 1),
                patch("v3.bridge.experiment.gate", return_value={"passed": True}),
            ):
                result = run_experiment(self.measure, Path(folder))
            self.assertEqual(result["status"], "completed_bridge")
            self.assertEqual(result["baseline_count"], 160)
            self.assertEqual(result["intervention_count"], 192)
            self.assertEqual(result["maximum_identity_error"], 0)
        finally:
            torch.set_num_threads(previous)


class GateTests(unittest.TestCase):
    def test_failed_gate_leaves_test_items_and_interventions_unrun(self):
        class AlwaysA:
            layers = [None] * 26
            model = torch.nn.Linear(4, 4)

            def evaluate(self, text, site_end, gap_start, layer=17, capture=False):
                scores = dict(
                    answer_logp=[-0.1, -3],
                    answer_mass=0.95,
                    next_margin=0,
                    input_ids=[1],
                    position_ids=[0],
                )
                return scores, torch.zeros(4) if capture else None, 0

        selected = [dict(items()[0], split="development"), dict(items()[1], split="test")]
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("v3.bridge.experiment.items", return_value=selected),
        ):
            result = run_experiment(AlwaysA(), Path(folder))
        self.assertEqual(result["status"], "instrument_gate_failed")
        self.assertEqual(result["baseline_count"], 80)
        self.assertEqual(result["intervention_count"], 0)
        self.assertTrue(all(r["split"] == "development" for r in result["behavioral"]))

    def rows(self, always_a=False):
        rows = []
        for voice in ["active", "passive"]:
            for role in ["agent", "patient"]:
                for order in [0, 1]:
                    correct = (order == 0) if always_a else True
                    rows.append(
                        dict(
                            record="baseline",
                            id="x",
                            family="x",
                            split="development",
                            reversal=0,
                            voice=voice,
                            condition="simple",
                            role=role,
                            order=order,
                            correct_margin=1 if correct else -1,
                            answer_mass=1.0,
                        )
                    )
        return rows

    def test_gate_requires_both_orders_in_each_voice(self):
        self.assertTrue(gate(self.rows())["passed"])
        self.assertFalse(gate(self.rows(always_a=True))["passed"])
        with self.assertRaises(ValueError):
            behavioral_summary(self.rows()[:-1])


if __name__ == "__main__":
    unittest.main()
