"""Run on the UTM host to promote only the verified GPS static report."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_bundle(root, expected_manifest):
    root = root.resolve()
    manifest_path = root / "data/manifest.json"
    if digest(manifest_path) != expected_manifest:
        raise ValueError("Manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("Symlink in public bundle")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]) | {"data/manifest.json"}:
        raise ValueError("Unexpected public files")
    for name, expected in manifest["files"].items():
        target = (root / name).resolve()
        if not target.is_relative_to(root) or digest(target) != expected:
            raise ValueError("File mismatch: " + name)
    return manifest


def shared_state(apps):
    config = ["caddy/Caddyfile", "caddy-compose.yml", "migration-images.yml"]
    # Includes any existing custom browse template, without copying or exposing its contents.
    template = "caddy/config/research-browse.html"
    if (apps / template).is_file():
        config.append(template)
    containers = subprocess.check_output(
        ["docker", "ps", "-a", "--format", "{{.Names}} {{.ID}} {{.Image}}"]
    ).decode()
    return dict(
        containers=sorted(containers.splitlines()),
        config_hashes={name: digest(apps / name) for name in config},
    )


def publish(release, manifest_hash, previous):
    if not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z-gps", release):
        raise ValueError("Invalid release")
    apps = Path("/home/deploy/apps")
    slug = Path("mech-interp/garden-path-sentences")
    stage = apps / "research-staging" / release / slug
    target = apps / "research" / slug
    backup = apps / "research-deploy-backups" / release / slug
    verify_bundle(stage, manifest_hash)
    if target.exists():
        if previous == "absent" or digest(target / "data/manifest.json") != previous:
            raise ValueError("Live report changed or belongs to an unreviewed release")
    elif previous != "absent":
        raise ValueError("Expected previous release missing")
    if backup.exists():
        raise ValueError("Backup already exists")
    before = shared_state(apps)
    target.parent.mkdir(parents=True, exist_ok=True)
    backup.parent.mkdir(parents=True, exist_ok=True)
    had_previous = target.exists()
    if had_previous:
        target.rename(backup)
    try:
        stage.rename(target)
        verify_bundle(target, manifest_hash)
        if shared_state(apps) != before:
            raise ValueError("Shared configuration or container identity changed")
    except Exception:
        if target.exists():
            target.rename(stage)
        if had_previous:
            backup.rename(target)
        raise
    result = dict(
        published=True,
        release=release,
        url="https://research.grokked.it/mech-interp/garden-path-sentences/",
        manifest_sha256=manifest_hash,
        index_sha256=digest(target / "index.html"),
        backup=str(backup) if had_previous else None,
        previous="preserved in backup" if had_previous else "no previous report",
        shared_configuration_unchanged=True,
        containers_unchanged=True,
    )
    (backup.parent / "deployment.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("release")
    parser.add_argument("manifest_hash")
    parser.add_argument("--previous", default="absent")
    args = parser.parse_args()
    publish(args.release, args.manifest_hash, args.previous)
