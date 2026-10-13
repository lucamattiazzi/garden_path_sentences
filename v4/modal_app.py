"""Run/fetch direct participant answers on Modal; upload only scientific sources and frozen data."""

from pathlib import Path

import modal

from v4.run import SOURCES

ROOT = Path(__file__).resolve().parents[1] if modal.is_local() else Path("/app")
app = modal.App("garden-path-v4")
cache = modal.Volume.from_name("gps-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("gps-v4-results", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.10.0", "transformers==5.9.0", "numpy==2.4.3", "sentencepiece==0.2.1")
    .env(
        {
            "HF_HOME": "/cache/hf",
            "PYTHONPATH": "/app",
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
        }
    )
)
for name in SOURCES:
    image = image.add_local_file(ROOT / name, "/app/" + name)


@app.function(
    image=image,
    gpu="A100-80GB",
    memory=81920,
    cpu=2,
    timeout=1800,
    retries=0,
    max_containers=1,
    volumes={"/cache/hf": cache, "/results": results},
    secrets=[modal.Secret.from_name("huggingface")],
)
def execute12(run_id: str) -> dict[str, bytes]:
    from v4.run import run

    destination = Path("/results") / run_id
    try:
        run(destination, "12b")
    finally:
        cache.commit()
        results.commit()
    return {p.name: p.read_bytes() for p in destination.iterdir() if p.is_file()}


@app.function(
    image=image,
    gpu="H200",
    memory=196608,
    cpu=2,
    timeout=1800,
    retries=0,
    max_containers=1,
    volumes={"/cache/hf": cache, "/results": results},
    secrets=[modal.Secret.from_name("huggingface")],
)
def execute27(run_id: str) -> dict[str, bytes]:
    from v4.run import run

    destination = Path("/results") / run_id
    try:
        run(destination, "27b")
    finally:
        cache.commit()
        results.commit()
    return {p.name: p.read_bytes() for p in destination.iterdir() if p.is_file()}


@app.function(image=image, volumes={"/results": results}, timeout=120)
def collect(run_id: str) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in (Path("/results") / run_id).iterdir() if p.is_file()}


@app.local_entrypoint()
def main(run_id: str, size: str = "12b", fetch: bool = False) -> None:
    import re

    if not re.fullmatch(r"[a-zA-Z0-9_-]+", run_id):
        raise ValueError("Unsafe run ID")
    execute = {"12b": execute12, "27b": execute27}[size]
    files = collect.remote(run_id) if fetch else execute.remote(run_id)
    destination = ROOT / "v4/results" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        path = destination / name
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(path)
        path.write_bytes(content)
    print(f"Saved {len(files)} files to {destination}")
