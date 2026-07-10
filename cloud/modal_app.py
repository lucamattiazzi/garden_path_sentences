"""Modal runner for the Gemma-3-27B GPS pipeline.

One command spins up a single A100-80GB, runs the automatable core
(`cloud/run_all.sh gemma3_27b`), and writes the result JSONs straight back
into your local `gemma3_27b/logs/`. Model weights are cached in a Modal Volume,
so a second run skips the ~54 GB download.

The 27B is the only size that needs this — 1B/4B/12B run on your Mac.

─────────────────────────────────────────────────────────────────────────────
Setup (once)
─────────────────────────────────────────────────────────────────────────────
    pip install -U modal
    modal setup                                     # browser auth
    # accept the license at https://huggingface.co/google/gemma-3-27b-pt
    # then store a token (of the account that accepted it) as a Modal secret:
    modal secret create huggingface HF_TOKEN=hf_xxxxxxxx

Run
    modal run --detach cloud/modal_app.py            # → gemma3_27b/logs/*.json
    modal run cloud/modal_app.py --model-dir gemma3_12b   # any size, if you like

--detach keeps the remote run alive even if this terminal disconnects (laptop
sleep, network drop). Step outputs are written to the `gps-results` Volume as
they complete, so nothing is lost either way: a re-run skips finished steps,
and you can pull whatever is on the Volume at any time with

    modal run cloud/modal_app.py::fetch              # → gemma3_27b/logs/*.json

Cost: a few minutes of A100 time (~$1-3, often inside Modal's monthly free
credit). Nothing is billed once the function returns — no VM to shut down.
"""

from __future__ import annotations

import pathlib

import modal

REPO = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = "gemma3_27b"

# Same dependency set as pyproject.toml, minus the macOS-only mlx-lm. torch's
# default wheel is CUDA-enabled, which is what we want on the A100.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.3.0",
        "transformers==5.9.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "accelerate>=0.30.0",
        "sentencepiece>=0.2.0",
        "protobuf>=4.25.0",
        "python-dotenv>=1.0.0",
        "huggingface_hub",
    )
    .env(
        {
            "HF_HOME": "/cache/hf",        # weights cached on the Volume below
            "PYTHONPATH": "/app",          # make `import model_finder` resolve
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        }
    )
    .add_local_dir(
        str(REPO),
        remote_path="/app",
        # keep the image small: skip vcs, the local venv, caches, big artifacts
        ignore=[
            ".git",
            ".venv",
            "**/__pycache__",
            "post_it_assets",
            "**/*.pyc",
        ],
    )
)

app = modal.App("gps-gemma3")
hf_cache = modal.Volume.from_name("gps-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("gps-results", create_if_missing=True)

RESULTS_ROOT = "/results"


def _collect(model_dir: str) -> dict[str, str]:
    import glob
    import os

    out: dict[str, str] = {}
    for p in sorted(glob.glob(f"{RESULTS_ROOT}/{model_dir}/*")):
        if p.endswith((".json", ".log")):
            out[os.path.basename(p)] = pathlib.Path(p).read_text()
    return out


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/cache/hf": hf_cache, RESULTS_ROOT: results_vol},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 60 * 3,  # 3h ceiling; the core pipeline is far under an hour
)
def run(model_dir: str = DEFAULT_MODEL_DIR, null_samples: int = 60) -> dict[str, str]:
    import os
    import shutil
    import subprocess

    # Point {model_dir}/logs at the Volume so every step's JSON lands on
    # persistent storage the moment it is written — a disconnect or crash
    # mid-pipeline loses at most the step in flight, and SKIP_DONE=1 makes
    # the next run resume from the first missing output.
    results_dir = pathlib.Path(RESULTS_ROOT) / model_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    logs = pathlib.Path("/app") / model_dir / "logs"
    if logs.is_dir() and not logs.is_symlink():
        shutil.rmtree(logs)
    if not logs.is_symlink():
        logs.symlink_to(results_dir)

    os.chdir("/app")
    env = dict(os.environ, PYTHON_RUN="python", SKIP_DONE="1")  # deps live in the image, no uv
    try:
        subprocess.run(
            ["bash", "cloud/run_all.sh", model_dir, str(null_samples)],
            check=True,
            env=env,
        )
    finally:
        hf_cache.commit()  # persist the downloaded weights for next time
        results_vol.commit()

    return _collect(model_dir)


@app.function(image=image, volumes={RESULTS_ROOT: results_vol}, timeout=300)
def collect(model_dir: str = DEFAULT_MODEL_DIR) -> dict[str, str]:
    return _collect(model_dir)


def _write_local(results: dict[str, str], model_dir: str) -> None:
    dest = REPO / model_dir / "logs"
    dest.mkdir(parents=True, exist_ok=True)
    for name, text in results.items():
        (dest / name).write_text(text)
    print(f"\nwrote {len(results)} files to {dest}/")


@app.local_entrypoint()
def main(model_dir: str = DEFAULT_MODEL_DIR, null_samples: int = 60) -> None:
    _write_local(run.remote(model_dir, null_samples), model_dir)


@app.local_entrypoint()
def fetch(model_dir: str = DEFAULT_MODEL_DIR) -> None:
    """Pull whatever results are on the Volume (e.g. after a detached run)."""
    _write_local(collect.remote(model_dir), model_dir)
