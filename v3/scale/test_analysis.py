import unittest

from v3.scale.analyze import check_selection


class SelectionAuditTests(unittest.TestCase):
    def test_only_next_prespecified_model_can_follow_failure(self):
        self.assertEqual(check_selection(["4b"], [True]), "4b")
        self.assertEqual(check_selection(["4b", "12b"], [False, True]), "12b")
        self.assertIsNone(check_selection(["4b", "12b"], [False, False]))
        with self.assertRaises(ValueError):
            check_selection(["12b"], [True])
        with self.assertRaises(ValueError):
            check_selection(["4b", "12b"], [True, True])
