# Running the Gemma-3 GPS pipeline in the cloud

The 27B in bf16 is ~54 GB of weights plus activations — it does not fit
comfortably in the 64 GB of an M1 Max next to the patching caches. Run it on a
single cloud GPU. `1b/4b/12b` run fine locally; only `27b` needs this.

The code is backend-agnostic: `model_finder.utils.get_device()` returns `cuda`
when a GPU is present, so **no code changes are needed** — same scripts, same
`cloud/run_all.sh`.

## 0. Before anything: Gemma is gated

1. Accept the license on <https://huggingface.co/google/gemma-3-27b-pt>.
2. Create a read token at <https://huggingface.co/settings/tokens>.
3. Export it as `HF_TOKEN` wherever you run (the code reads `.env` / the env).

## 1. Pick a machine

| model | dtype | weights | GPU that fits |
|-------|-------|---------|---------------|
| gemma3-1b  | bf16 | ~2 GB  | anything / local |
| gemma3-4b  | bf16 | ~8 GB  | anything / local |
| gemma3-12b | bf16 | ~24 GB | 1×24 GB (3090/4090/A10) |
| gemma3-27b | bf16 | ~54 GB | **1×80 GB (A100/H100)** |

Single GPU only — the pipeline uses `.to(device)`, not `device_map="auto"`.
Any provider works: RunPod, Lambda, vast.ai, or a GCP/AWS GPU VM.

## 2·0. Serverless with Modal (simplest for a one-off — recommended)

No VM to start or remember to stop; billed per second, usually inside the free
monthly credit. See the header of `cloud/modal_app.py` for the full setup, in
short:

```bash
pip install -U modal && modal setup
modal secret create huggingface HF_TOKEN=hf_xxx     # token that accepted the license
modal run cloud/modal_app.py                        # → gemma3_27b/logs/*.json land locally
```

Weights are cached on a Modal Volume, so re-runs skip the ~54 GB download. The
DIY paths below are for when you'd rather manage the machine yourself.

## 2a. With Docker (any GPU host)

```bash
docker build -f cloud/Dockerfile -t gps .
docker run --gpus all -e HF_TOKEN=hf_xxx \
  -v "$PWD/gemma3_27b/logs:/app/gemma3_27b/logs" \
  gps bash cloud/run_all.sh gemma3_27b
```

The `-v` mount makes the results land back on the host in
`gemma3_27b/logs/*.json`.

## 2b. Bare VM, no Docker

```bash
git clone <your-repo> && cd garden_path_sentences
curl -LsSf https://astral.sh/uv/install.sh | sh
export HF_TOKEN=hf_xxx
uv sync --no-dev
bash cloud/run_all.sh gemma3_27b
```

## 3. Confirm 27B before the full run (optional but recommended)

The 27B was **not** in the original `model_finder` sweep. Check it clears the
"positive Δ on all 5 templates" bar first (a few minutes):

```bash
uv run python gemma3_27b/scripts/01_baseline.py
# look at gemma3_27b/logs/01_baseline.json → per_template all > 0 ?
```

## 4. Bring the results home

Everything is JSON under `<model>/logs/`. Copy those back and reuse your local
plotting — the schemas are identical to the qwen4b/qwen14b logs, so the same
charts and cross-model comparison just work.

```bash
scp -r user@host:/app/gemma3_27b/logs ./gemma3_27b/
```

## Notes / gotchas

- **bf16 is mandatory** for Gemma (fp16 → NaN logits); already pinned in every
  `gemma3_*/scripts/_common.py`.
- **Step 10** (attention patterns) needs eager attention — already handled by
  `load_model_eager`, which passes `attn_implementation="eager"`.
- **Multimodal**: 4b/12b/27b ship a vision tower; only the language model is
  exercised, but the weights still download. `model_finder.arch` resolves the
  nested `language_model` layout transparently.
- **03 / 04 / 12** are candidate-driven; `run_all.sh` now picks their layers/heads
  automatically from the 02/03/05 sweeps (`pick_candidates.py`, same criteria a
  human would use). Set `AUTO_CANDIDATES=0` to skip them and choose by hand.
