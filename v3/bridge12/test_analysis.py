import json
import shutil
import tempfile
import unittest
from pathlib import Path

from v3.bridge12.analyze import audit, validate_score


class ScoreAuditTests(unittest.TestCase):
    def test_score_audit_uses_full_sequence_and_correct_gold_direction(self):
        score = dict(
            answer_logp=[-2.0, -5.0],
            answer_tokens=[dict(token_logp=[-1.0, -1.0]), dict(token_logp=[-2.0, -3.0])],
            next_logp=[-4.0, -6.0],
            next_margin=2.0,
        )
        self.assertEqual(validate_score(score, 1), -3.0)
        score["answer_tokens"][0]["token_logp"][0] = 0.0
        with self.assertRaises(ValueError):
            validate_score(score, 1)


class ChronologyAuditTests(unittest.TestCase):
    def test_saved_run_passes_but_intervention_before_gate_fails(self):
        original = Path(__file__).parent / "results/bridge12-20261009-01"
        self.assertTrue(audit(original)[3]["checks"]["passed"])
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "run"
            shutil.copytree(original, target)
            path = target / "observations.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            index = next(i for i, r in enumerate(rows) if r["record"] == "intervention")
            row = rows.pop(index)
            rows.insert(1, row)
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            with self.assertRaises(ValueError):
                audit(target)
