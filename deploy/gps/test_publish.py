import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("publish", Path(__file__).with_name("publish.py"))
publish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish)


class StagingTests(unittest.TestCase):
    def test_exact_file_set_and_hashes_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "index.html").write_text("public fixture")
            manifest = {"files": {"index.html": publish.digest(root / "index.html")}}
            m = root / "data/manifest.json"
            m.write_text(json.dumps(manifest))
            expected = publish.digest(m)
            publish.verify_bundle(root, expected)
            (root / "extra.txt").write_text("unlisted")
            with self.assertRaises(ValueError):
                publish.verify_bundle(root, expected)
            (root / "extra.txt").unlink()
            (root / "index.html").write_text("changed")
            with self.assertRaises(ValueError):
                publish.verify_bundle(root, expected)

    def test_symlinks_and_changed_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            m = root / "data/manifest.json"
            m.write_text('{"files":{}}')
            expected = publish.digest(m)
            (root / "link").symlink_to(m)
            with self.assertRaises(ValueError):
                publish.verify_bundle(root, expected)
            (root / "link").unlink()
            m.write_text('{"files":{},"changed":true}')
            with self.assertRaises(ValueError):
                publish.verify_bundle(root, expected)


if __name__ == "__main__":
    unittest.main()
