import json
import shutil
import tempfile
import unittest
from pathlib import Path

from v3.direct.analyze import audit, summarize


class AnalysisTests(unittest.TestCase):
    def test_family_pairs_are_required_and_not_counted_as_independent_successes(self):
        rows = [
            dict(
                condition="simple",
                voice="active",
                role="agent",
                family="f",
                reversal=i,
                margin=1 if i == 0 else -1,
            )
            for i in [0, 1]
        ]
        result = summarize(rows)[0]
        self.assertEqual(result["accuracy"]["mean"], 0.5)
        self.assertEqual(result["both_reversals"]["mean"], 0)
        self.assertEqual(result["margin"]["mean"], 0)
        with self.assertRaises(ValueError):
            summarize(rows[:1])
        with self.assertRaises(ValueError):
            summarize(rows + rows)


class AuditTests(unittest.TestCase):
    def test_saved_run_passes_audit_but_altered_score_fails(self):
        original = Path(__file__).parent / "results/direct-v3-20261008-01"
        self.assertTrue(audit(original)[3]["checks"]["passed"])
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "run"
            shutil.copytree(original, target)
            path = target / "observations.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["scores"]["candidates"][0]["token_logp"][0] += 1
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            with self.assertRaises(ValueError):
                audit(target)
