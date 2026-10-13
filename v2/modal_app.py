"""Run one bounded A10G pilot, uploading only the v2 Python source allowlist."""

from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
app = modal.App("garden-path-v2")
cache = modal.Volume.from_name("gps-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("gps-v2-results", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.10.0", "transformers==5.9.0", "numpy==2.4.3", "sentencepiece==0.2.1")
    .env({"HF_HOME": "/cache/hf", "PYTHONPATH": "/app"})
)
for name in ("__init__.py", "materials.py", "experiment.py", "modal_app.py"):
    image = image.add_local_file(ROOT / name, f"/app/v2/{name}")


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    retries=0,
    max_containers=1,
    volumes={"/cache/hf": cache, "/results": results},
    secrets=[modal.Secret.from_name("huggingface")],
)
def pilot(run_id: str, revision: str = "") -> dict[str, bytes]:
    from huggingface_hub import HfApi

    from v2.experiment import run

    revision = revision or HfApi().model_info("google/gemma-3-1b-pt").sha
    destination = Path("/results") / run_id
    try:
        run(str(destination), revision)
    finally:
        cache.commit()
        results.commit()
    return {p.name: p.read_bytes() for p in destination.iterdir() if p.is_file()}


@app.function(image=image, volumes={"/results": results}, timeout=120)
def collect(run_id: str) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in (Path("/results") / run_id).iterdir() if p.is_file()}


def save(run_id: str, files: dict[str, bytes]) -> None:
    destination = ROOT / "results" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        path = destination / name
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError(f"Refusing to overwrite different result: {path}")
        path.write_bytes(data)
    print(f"Saved {len(files)} artifacts to {destination}")


@app.local_entrypoint()
def main(run_id: str = "", fetch: bool = False, revision: str = "") -> None:
    import datetime
    import re

    run_id = run_id or datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", run_id):
        raise ValueError("run-id must contain only letters, numbers, underscores and hyphens")
    save(run_id, collect.remote(run_id) if fetch else pilot.remote(run_id, revision))
