"""Bounded pilot: incremental NP/Z garden-path patching."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from v2.materials import garden_paths


def family_interval(values: list[tuple[str, float]]) -> dict:
    groups = defaultdict(list)
    for family, value in values:
        if not np.isfinite(value):
            raise ValueError("Nonfinite measurement")
        groups[family].append(value)
    means = np.array([np.mean(x) for x in groups.values()])
    if not len(means):
        raise ValueError("No observations")
    draws = np.random.default_rng(0).choice(means, (10000, len(means))).mean(axis=1)
    return {
        "mean": float(means.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "families": len(means),
    }


def token_position(offsets: list, start: int, end: int) -> int:
    hits = [i for i, (a, b) in enumerate(offsets) if b > start and a < end]
    if not hits or offsets[hits[-1]][1] != end or offsets[hits[0]][0] > start:
        raise ValueError("Token offsets do not cover the requested span exactly")
    return hits[-1]


def patch_state(output, position: int, donor: torch.Tensor):
    state = output[0] if isinstance(output, tuple) else output
    changed = state.clone()
    changed[:, position, :] = donor.to(device=state.device, dtype=state.dtype)
    return (changed, *output[1:]) if isinstance(output, tuple) else changed


class Measurements:
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer
        self.layers = model.model.layers
        self.device = next(model.parameters()).device

    def encode(self, text: str) -> dict:
        data = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        offsets = data.pop("offset_mapping")[0].tolist()
        return {"inputs": {k: v.to(self.device) for k, v in data.items()}, "offsets": offsets}

    def capture(self, prefix: str, continuation: str = "") -> tuple[list[torch.Tensor], int]:
        data = self.encode(prefix)
        position = token_position(data["offsets"], prefix.rfind(" ") + 1, len(prefix))
        if continuation:
            full = self.encode(prefix + " " + continuation)
            ids = data["inputs"]["input_ids"]
            if not torch.equal(full["inputs"]["input_ids"][:, : ids.shape[1]], ids):
                raise ValueError("Continuation changes prefix tokenization")
            # Match scoring GEMM shapes in BF16. The causal mask prevents the
            # recorded prefix position from seeing these teacher-forced tokens.
            data = full
        captured = []

        def hook(_module, _inputs, output):
            state = output[0] if isinstance(output, tuple) else output
            captured.append(state[0, position].detach().float().cpu())

        handles = [layer.register_forward_hook(hook) for layer in self.layers]
        try:
            with torch.inference_mode():
                self.model(**data["inputs"], use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        return captured, position

    def score(self, prefix: str, word: str, patch: tuple | None = None) -> float:
        base = self.encode(prefix)["inputs"]["input_ids"][0]
        data = self.encode(prefix + " " + word)
        ids = data["inputs"]["input_ids"][0]
        if not torch.equal(ids[: len(base)], base) or len(ids) <= len(base):
            raise ValueError("Continuation changes prefix tokenization")
        handle = None
        if patch is not None:
            layer, position, donor = patch
            handle = self.layers[layer].register_forward_hook(
                lambda _m, _i, output: patch_state(output, position, donor)
            )
        try:
            with torch.inference_mode():
                logits = self.model(**data["inputs"], use_cache=False).logits[0].float()
                indices = torch.arange(len(base), len(ids), device=self.device)
                value = (
                    logits[indices - 1].log_softmax(-1).gather(1, ids[indices, None]).sum().item()
                )
        finally:
            if handle is not None:
                handle.remove()
        if not np.isfinite(value):
            raise ValueError("Nonfinite log probability")
        return value


def run_garden_paths(measure: Measurements) -> dict:
    rows = []
    for item in garden_paths():
        for site in ("pre_disambiguation", "post_disambiguation"):
            suffix = "" if site == "pre_disambiguation" else " " + item["critical"]
            word = item["critical"] if not suffix else item["spillover"]
            prefixes = {
                c: item[c] + suffix
                for c in ("clean", "ambiguous", "lexical_clean", "lexical_ambiguous")
            }
            states = {c: measure.capture(p, continuation=word) for c, p in prefixes.items()}
            baseline = {c: measure.score(prefixes[c], word) for c in ("clean", "ambiguous")}
            for recipient, donor in (("ambiguous", "clean"), ("clean", "ambiguous")):
                for layer in range(len(measure.layers)):
                    target_states, pos = states[recipient]
                    value = measure.score(
                        prefixes[recipient], word, (layer, pos, states[donor][0][layer])
                    )
                    lexical = states["lexical_" + recipient][0][layer]
                    control = measure.score(prefixes[recipient], word, (layer, pos, lexical))
                    self_score = measure.score(
                        prefixes[recipient], word, (layer, pos, target_states[layer])
                    )
                    if abs(self_score - baseline[recipient]) > 1e-4:
                        raise ValueError(
                            f"Self patch failed: {item['id']} {site} {recipient} L{layer}: {self_score - baseline[recipient]:.8g}"
                        )
                    rows.append(
                        {
                            "family": item["family"],
                            "site": site,
                            "layer": layer,
                            "recipient": recipient,
                            "donor": donor,
                            "word": word,
                            "prefix": prefixes[recipient],
                            "position": pos,
                            "baseline": baseline[recipient],
                            "donor_baseline": baseline[donor],
                            "patched": value,
                            "lexical_control": control,
                            "self_patch": self_score,
                            "effect": value - baseline[recipient],
                            "lexical_effect": control - baseline[recipient],
                            "patch_l2": float(
                                torch.linalg.vector_norm(
                                    states[donor][0][layer] - target_states[layer]
                                )
                            ),
                            "lexical_l2": float(
                                torch.linalg.vector_norm(lexical - target_states[layer])
                            ),
                        }
                    )
        print(f"Completed NP/Z family {item['family']}", flush=True)
    summaries = []
    for site in ("pre_disambiguation", "post_disambiguation"):
        for recipient in ("clean", "ambiguous"):
            for layer in range(len(measure.layers)):
                subset = [
                    r
                    for r in rows
                    if (r["site"], r["recipient"], r["layer"]) == (site, recipient, layer)
                ]
                summaries.append(
                    {
                        "site": site,
                        "recipient": recipient,
                        "layer": layer,
                        **{
                            name: family_interval([(r["family"], r[name]) for r in subset])
                            for name in ("effect", "lexical_effect")
                        },
                    }
                )
    return {"materials": garden_paths(), "rows": rows, "summaries": summaries}


def run(output: str, revision: str) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("An exact checkpoint SHA is required")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    torch.manual_seed(0)
    np.random.seed(0)
    model_id = "google/gemma-3-1b-pt"
    source = Path(__file__).parent
    metadata = {
        "model": model_id,
        "revision": revision,
        "seed": 0,
        "dtype": "bfloat16",
        "attention": "eager",
        "layer_index": "zero-based, post-block",
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "numpy", "huggingface_hub")
        },
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(),
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")
        },
        "materials_sha256": hashlib.sha256(
            json.dumps(garden_paths(), sort_keys=True).encode()
        ).hexdigest(),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out / "source.json").write_text(
        json.dumps({p.name: p.read_text() for p in source.glob("*.py")}, indent=2)
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=torch.bfloat16, attn_implementation="eager"
        )
        .to("cuda")
        .eval()
    )
    measure = Measurements(model, tokenizer)
    result = run_garden_paths(measure)
    (out / "garden-paths.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    metadata["elapsed_seconds"] = time.time() - start
    metadata["complete"] = True
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
