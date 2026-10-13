"""Bounded, sequential PT scoring and IT interpretation of GPS examples."""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
app = modal.App("gps-title-case-study")
cache = modal.Volume.from_name("gps-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("gps-title-results", create_if_missing=True)
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
for name in ("__init__.py", "probe.py", "modal_app.py", "PROTOCOL.md"):
    image = image.add_local_file(ROOT / name, "/app/gps_title/" + name)


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=2,
    memory=81920,
    timeout=1200,
    retries=0,
    max_containers=1,
    volumes={"/cache/hf": cache, "/results": results},
    secrets=[modal.Secret.from_name("huggingface")],
)
def execute(run_id: str):
    from gps_title.probe import run

    out = Path("/results") / run_id
    try:
        run(out)
    finally:
        cache.commit()
        results.commit()
    return {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}


@app.function(image=image, volumes={"/results": results}, timeout=120)
def collect(run_id: str):
    return {p.name: p.read_bytes() for p in (Path("/results") / run_id).iterdir() if p.is_file()}


@app.local_entrypoint()
def main(run_id: str, fetch: bool = False):
    import re

    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid run ID")
    files = collect.remote(run_id) if fetch else execute.remote(run_id)
    out = ROOT / "results" / run_id
    out.mkdir(parents=True, exist_ok=True)
    for name, contents in files.items():
        p = out / name
        if p.exists() and p.read_bytes() != contents:
            raise FileExistsError(p)
        p.write_bytes(contents)
    print("Saved", out)
