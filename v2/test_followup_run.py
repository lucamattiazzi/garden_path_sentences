import hashlib
import tempfile
import unittest
from pathlib import Path

from v2.followup_run import source_bundle, start_run


class ProvenanceTests(unittest.TestCase):
    def test_snapshot_is_allowlisted_and_hashes_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "program.py").write_text("print('hello')\n")
            (root / ".env").write_text("DO_NOT_UPLOAD=private")
            snapshot, hashes = source_bundle(root, ("program.py",))
            self.assertEqual(set(snapshot), {"program.py"})
            self.assertEqual(
                hashes["program.py"], hashlib.sha256(snapshot["program.py"].encode()).hexdigest()
            )
            with self.assertRaises(FileNotFoundError):
                source_bundle(root, ("missing.py",))

    def test_invalid_track_does_not_create_output_and_existing_run_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run"
            with self.assertRaises(ValueError):
                start_run(target, "unknown")
            self.assertFalse(target.exists())
            start_run(target, "garden")
            (target / "existing.json").write_text("saved")
            with self.assertRaises(FileExistsError):
                start_run(target, "garden")
            self.assertEqual((target / "existing.json").read_text(), "saved")


if __name__ == "__main__":
    unittest.main()
