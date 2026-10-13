"""Bounded independent Gemma 3 follow-ups; only allowlisted source/data are uploaded."""

from pathlib import Path

import modal

from v2.followup_run import SOURCE_FILES

ROOT = Path(__file__).resolve().parent
app = modal.App("garden-path-v2-followup")
cache = modal.Volume.from_name("gps-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("gps-v2-results", create_if_missing=True)
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
for name in SOURCE_FILES:
    image = image.add_local_file(ROOT / name, f"/app/v2/{name}")


@app.function(
    image=image,
    gpu="A10G",
    cpu=2,
    timeout=1200,
    retries=0,
    max_containers=1,
    volumes={"/cache/hf": cache, "/results": results},
    secrets=[modal.Secret.from_name("huggingface")],
)
def execute(run_id: str, track: str) -> dict[str, bytes]:
    from v2.followup_run import run_followup

    destination = Path("/results") / run_id
    try:
        run_followup(destination, track)
    finally:
        cache.commit()
        results.commit()
    return {p.name: p.read_bytes() for p in destination.iterdir() if p.is_file()}


@app.function(image=image, volumes={"/results": results}, timeout=120)
def collect(run_id: str) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in (Path("/results") / run_id).iterdir() if p.is_file()}


@app.local_entrypoint()
def main(track: str = "garden", run_id: str = "", fetch: bool = False) -> None:
    import datetime
    import re

    if track != "garden":
        raise ValueError("Only the garden track is in scope")
    run_id = run_id or f"{track}-" + datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", run_id):
        raise ValueError("Unsafe run ID")
    files = collect.remote(run_id) if fetch else execute.remote(run_id, track)
    destination = ROOT / "results" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        path = destination / name
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f"Refusing to replace different result {path}")
        path.write_bytes(content)
    print(f"Saved {len(files)} files to {destination}")
