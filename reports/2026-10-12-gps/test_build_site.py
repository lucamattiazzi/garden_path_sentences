import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).with_name("build_site.py")
spec = importlib.util.spec_from_file_location("build_site", MODULE)
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


class PublicBundleTests(unittest.TestCase):
    def test_historical_snapshot_uses_bundle_relative_navigation(self):
        page = build.render(
            "# Historical report\n\nText.", "Snapshot", base="../../", require_levels=False
        )
        self.assertIn('href="../../styles.css"', page)
        self.assertIn('href="../../index.html"', page)
        self.assertIn('href="../../data/evidence.json"', page)

    def test_local_links_and_anchors_must_resolve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "index.html").write_text(
                '<a href="#intro">Start</a><h2 id="intro">Intro</h2><img src="plot.svg">'
            )
            (root / "plot.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
            build.validate_public_tree(root)
            (root / "plot.svg").unlink()
            with self.assertRaises(ValueError):
                build.validate_public_tree(root)

    def test_no_symlinks_private_files_or_local_absolute_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "index.html"
            p.write_text('<a href="/Users/example/private.txt">Bad</a>')
            with self.assertRaises(ValueError):
                build.validate_public_tree(root)
            p.write_text("<p>Clean</p>")
            (root / ".env").write_text("fixture")
            with self.assertRaises(ValueError):
                build.validate_public_tree(root)
            (root / ".env").unlink()
            (root / "link.txt").symlink_to(p)
            with self.assertRaises(ValueError):
                build.validate_public_tree(root)

    def test_parent_link_stays_inside_public_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "child").mkdir()
            (root / "index.html").write_text('<a href="child/">Details</a>')
            (root / "child/index.html").write_text('<a href="../">Back</a>')
            build.validate_public_tree(root)
            (root / "child/index.html").write_text('<a href="../../outside.txt">Bad</a>')
            with self.assertRaises(ValueError):
                build.validate_public_tree(root)


if __name__ == "__main__":
    unittest.main()
