import unittest

from v3.bridge.precision import select_rows


class PrecisionTests(unittest.TestCase):
    def test_diagnostic_uses_only_development_simple_and_no_context(self):
        rows = [
            dict(record="baseline", split=s, condition=c)
            for s in ["development", "test"]
            for c in ["simple", "no_context", "clean"]
        ]
        selected = select_rows(rows)
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(r["split"] == "development" for r in selected))
        self.assertEqual({r["condition"] for r in selected}, {"simple", "no_context"})
