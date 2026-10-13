import json
import tempfile
import unittest
from pathlib import Path

from research.audit import audit_model, paired_change, summarize_items


class AuditTests(unittest.TestCase):
    def test_reports_nonfinite_and_changed_baselines_without_editing_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp)
            baseline = {"id": 1, "surprisal_gps": 2.0, "surprisal_normal": 1.0, "delta": 1.0}
            changed = dict(baseline, surprisal_normal=float("nan"), delta=float("nan"))
            for step, key, row in (
                ("01_baseline", "pair_results", baseline),
                ("02_layer_sweep", "baseline_pair_results", changed),
            ):
                (logs / f"{step}.json").write_text(json.dumps({"results": {key: [row]}}))
            before = {p.name: p.read_bytes() for p in logs.iterdir()}
            result = audit_model(logs)
            self.assertEqual(len(result["nonfinite"]), 2)
            self.assertEqual(result["baseline_changes"][0]["id"], 1)
            self.assertEqual(before, {p.name: p.read_bytes() for p in logs.iterdir()})

    def test_leave_one_out_exposes_a_dominant_item(self):
        result = summarize_items([{"id": 1, "value": 1.0}, {"id": 2, "value": 9.0}], "value")
        self.assertEqual(result["mean"], 5.0)
        self.assertEqual(
            result["leave_one_out"],
            [{"omitted_id": 1, "mean": 9.0}, {"omitted_id": 2, "mean": 1.0}],
        )

    def test_paired_comparison_joins_ids_and_rejects_different_samples(self):
        first = [{"id": 1, "vc_delta": 1.0}, {"id": 2, "vc_delta": 8.0}]
        second = [{"id": 2, "vc_delta": 10.0}, {"id": 1, "vc_delta": 3.0}]
        result = paired_change(first, second)
        self.assertEqual(result["mean"], 2.0)
        self.assertEqual(result["ci95"], [2.0, 2.0])
        with self.assertRaises(ValueError):
            paired_change(first, second[:1])

    def test_nonfinite_item_cannot_disappear_from_a_new_comparison(self):
        with self.assertRaises(ValueError):
            summarize_items([{"id": 1, "value": float("nan")}], "value")


if __name__ == "__main__":
    unittest.main()
