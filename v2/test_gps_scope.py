"""The remaining V2 entry point must not launch independent role experiments."""

import tempfile
import unittest
from pathlib import Path

from v2.followup_run import SOURCE_FILES, start_run


class ScopeTests(unittest.TestCase):
    def test_roles_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            with self.assertRaises(ValueError):
                start_run(out, "roles")
            self.assertFalse(out.exists())

    def test_upload_allowlist_contains_only_existing_gps_sources(self):
        root = Path(__file__).parent
        for name in SOURCE_FILES:
            self.assertTrue((root / name).is_file(), name)
            self.assertNotIn("roles", name.lower())
            self.assertNotIn("ud_validation", name)
