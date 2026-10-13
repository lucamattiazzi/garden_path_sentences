"""Verify the public HTTPS report against local build hashes using normal DNS."""

import concurrent.futures
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "site/mech-interp/garden-path-sentences"
BASE = "https://research.grokked.it/mech-interp/garden-path-sentences/"


def fetch(url):
    with urlopen(url, timeout=30) as response:
        if response.status != 200:
            raise ValueError(f"HTTP {response.status}: {url}")
        return response.read()


def check_file(item):
    name, expected = item
    actual = hashlib.sha256(fetch(BASE + name)).hexdigest()
    if actual != expected:
        raise ValueError("Public hash mismatch: " + name)
    return name


def main():
    manifest_bytes = (BUNDLE / "data/manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    files = dict(manifest["files"])
    files["data/manifest.json"] = hashlib.sha256(manifest_bytes).hexdigest()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        checked = sorted(pool.map(check_file, files.items()))
    root = fetch("https://research.grokked.it/").decode()
    category = fetch("https://research.grokked.it/mech-interp/").decode()
    if "mech-interp/" not in root or "garden-path-sentences/" not in category:
        raise ValueError("Report not discoverable from the shared indexes")
    result = dict(
        verified_at=datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - macOS Python 3.9
        url=BASE,
        transport="HTTPS using normal DNS on deployment workstation; no --resolve override",
        manifest_sha256=files["data/manifest.json"],
        public_files_matched=checked,
        indexes_verified=True,
    )
    (Path(__file__).parent / "public-verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
