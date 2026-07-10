"""Build the gemma3 HTML report.

Reads the structured JSON results in gemma3_*/logs/ and emits report/dist/
containing index.html, styles.css, app.js (copied from report/assets/),
data.json (the trimmed per-model payload) and data.js (the same payload as a
window global, so the page works when opened via file://).

Usage: uv run python report/build_report.py
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = Path(__file__).resolve().parent
ASSETS = REPORT_DIR / "assets"
DIST = REPORT_DIR / "dist"

MODELS = ["gemma3_1b", "gemma3_4b", "gemma3_12b", "gemma3_27b"]


def load_step(logs: Path, name: str):
    path = logs / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)["results"]


def extract_model(model_dir: Path):
    logs = model_dir / "logs"

    s01 = load_step(logs, "01_baseline")
    s02 = load_step(logs, "02_layer_sweep")

    out = {"meta": None, "steps_present": []}

    if s02 is not None:
        out["meta"] = {
            "model": s02["model"],
            "n_layers": s02["n_layers"],
            "n_heads": s02["n_heads"],
        }

    if s01 is not None:
        out["s01"] = {
            "summary": s01["summary"],
            "per_template": s01["per_template"],
            "pairs": [
                {"id": p["id"], "template": p["template"], "delta": p["delta"]}
                for p in s01["pair_results"]
            ],
        }

    if s02 is not None:
        out["s02"] = {
            "per_layer": [
                {
                    "layer": l["layer"],
                    "effect_drop": l["effect_drop"],
                    "control_inflation": l["control_inflation"],
                }
                for l in s02["per_layer"]
            ]
        }

    sweeps = []
    for path in sorted(logs.glob("03_head_sweep_L*.json")):
        res = json.load(open(path))["results"]
        sweeps.append(
            {
                "layer": res["layer"],
                "per_head": [
                    {
                        "head": h["head"],
                        "effect_drop": h["effect_drop"],
                        "control_inflation": h["control_inflation"],
                    }
                    for h in res["per_head"]
                ],
            }
        )
    if sweeps:
        out["s03"] = sorted(sweeps, key=lambda s: s["layer"])

    s04 = load_step(logs, "04_predict")
    if s04 is not None:
        out["s04"] = {
            "conditions": [
                {
                    "label": c["label"],
                    "mean_vc_delta": c["summary"]["mean_vc_delta"],
                    "mean_s_vc_gps": c["summary"]["mean_s_vc_gps"],
                    "mean_s_control": c["summary"]["mean_s_control"],
                }
                for c in s04["conditions"]
            ]
        }

    vc_sweeps = []
    for path in sorted(logs.glob("06_predict_head_sweep_L*.json")):
        res = json.load(open(path))["results"]
        vc_sweeps.append(
            {
                "layer": res["layer"],
                "per_head": [
                    {
                        "head": h["head"],
                        "vc_gps_inflation": h["vc_gps_inflation"],
                        "vc_delta_inflation": h["vc_delta_inflation"],
                        "control_inflation": h["control_inflation"],
                    }
                    for h in res["per_head"]
                ],
            }
        )
    if vc_sweeps:
        out["s06"] = sorted(vc_sweeps, key=lambda s: s["layer"])

    s05 = load_step(logs, "05_predict_layer_sweep")
    if s05 is not None:
        out["s05"] = {
            "baseline_summary": s05["baseline_summary"],
            "per_layer": [
                {
                    "layer": l["layer"],
                    "vc_delta_inflation": l["vc_delta_inflation"],
                    "vc_gps_inflation": l["vc_gps_inflation"],
                    "control_inflation": l["control_inflation"],
                }
                for l in s05["per_layer"]
            ],
        }

    for step, key in [("07_lexical_baseline", "s07"), ("08_novel_lexical_baseline", "s08")]:
        res = load_step(logs, step)
        if res is not None:
            out[key] = {
                "label": res["label"],
                "summary": res["summary"],
                "ci95_full_minus_stripped": res["ci95_full_minus_stripped"],
            }

    s09 = load_step(logs, "09_novel_predict")
    if s09 is not None:
        out["s09"] = {
            cond: {"summary": s09[cond]["summary"], "ci95": s09[cond]["ci95"]}
            for cond in ("canonical", "novel")
        }

    s10 = load_step(logs, "10_attention_patterns")
    if s10 is not None:
        out["s10"] = {
            "mean_attn_gps": s10["mean_attn_gps"],
            "mean_attn_diff": s10["mean_attn_diff"],
            "top_heads_by_diff": s10["top_heads_by_diff"][:10],
        }

    s11 = load_step(logs, "11_patching")
    if s11 is not None:
        out["s11"] = {
            "mean_s_gps": s11["mean_s_gps"],
            "mean_s_norm": s11["mean_s_norm"],
            "per_layer": [
                {
                    "layer": l["layer"],
                    "recovery_v_amb": l["recovery_v_amb"],
                    "recovery_critical": l["recovery_critical"],
                    "recovery_both": l["recovery_both"],
                }
                for l in s11["per_layer"]
            ],
        }

    s12 = load_step(logs, "12_mean_ablation")
    if s12 is not None:
        out["s12"] = {
            "head_sets": [
                {"label": e["label"], "zero": e["zero"], "mean": e["mean"]}
                for e in s12["head_sets"]
            ],
            "layers": [
                {"label": e["label"], "zero": e["zero"], "mean": e["mean"]}
                for e in s12["layers"]
            ],
        }

    s13 = load_step(logs, "13_null_distribution")
    if s13 is not None:
        out["s13"] = {
            "n_samples": s13["n_samples"],
            "samples": s13["samples"],
            "percentiles": s13["percentiles"],
        }

    s14 = load_step(logs, "14_null_distribution_vc")
    if s14 is not None:
        out["s14"] = {
            "n_samples": s14["n_samples"],
            "vc_gps_inflations": [s["vc_gps_inflation"] for s in s14["samples"]],
            "percentiles": s14["percentiles"],
            "max_of_k": s14["max_of_k"],
        }

    step_names = {
        "s01": "01", "s02": "02", "s03": "03", "s04": "04", "s05": "05",
        "s06": "06", "s07": "07", "s08": "08", "s09": "09", "s10": "10",
        "s11": "11", "s12": "12", "s13": "13", "s14": "14",
    }
    out["steps_present"] = [step_names[k] for k in step_names if k in out]
    return out


def main():
    data = {"models": {}, "model_order": MODELS}
    for name in MODELS:
        model_dir = ROOT / name
        if not model_dir.exists():
            print(f"WARNING: {model_dir} does not exist, skipping", file=sys.stderr)
            continue
        data["models"][name] = extract_model(model_dir)

    DIST.mkdir(parents=True, exist_ok=True)
    for asset in ("index.html", "styles.css", "app.js"):
        shutil.copy(ASSETS / asset, DIST / asset)

    payload = json.dumps(data, indent=1)
    (DIST / "data.json").write_text(payload)
    (DIST / "data.js").write_text("window.REPORT_DATA = " + json.dumps(data) + ";\n")

    print(f"dist written to {DIST}\n")
    print(f"{'model':<12} {'steps present':<40}")
    for name in MODELS:
        m = data["models"].get(name)
        steps = ", ".join(m["steps_present"]) if m and m["steps_present"] else "(none)"
        print(f"{name:<12} {steps}")


if __name__ == "__main__":
    main()
