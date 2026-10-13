"""Bounded FP32 follow-up; identical development prompts, no held-out inference."""

from pathlib import Path

import modal

from v3.bridge.modal_app import ROOT, cache, image, results
from v3.bridge.precision import EXTRA_SOURCES

app = modal.App("garden-path-v3-bridge-precision")
for name in EXTRA_SOURCES:
    image = image.add_local_file(ROOT / name, "/app/" + name)


@app.function(
    image=image,
    gpu="A10G",
    cpu=2,
    timeout=600,
    retries=0,
    max_containers=1,
    volumes={"/cache/hf": cache, "/results": results},
    secrets=[modal.Secret.from_name("huggingface")],
)
def execute() -> dict[str, bytes]:
    from v3.bridge.precision import run

    parent = Path("/results/bridge-v3-20261008-02")
    destination = Path("/results/bridge-precision-20261008-01")
    try:
        run(parent, destination)
    finally:
        cache.commit()
        results.commit()
    return {p.name: p.read_bytes() for p in destination.iterdir() if p.is_file()}


@app.local_entrypoint()
def main():
    files = execute.remote()
    destination = ROOT / "v3/bridge/results/bridge-precision-20261008-01"
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        (destination / name).write_bytes(content)
    print(f"Saved {len(files)} diagnostic files to {destination}")
