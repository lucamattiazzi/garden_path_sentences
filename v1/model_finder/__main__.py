import json
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from model_finder.models import MODELS, ModelSpec
from model_finder.sentences import PAIRS
from model_finder.surprisal import load_model, surprisal
from model_finder.utils import aggregate, cleanup, get_device

load_dotenv()

RESULTS_PATH = Path("./model_finder_results.json")


def evaluate_model(
    device: str,
    spec: ModelSpec,
) -> dict:
    model, tokenizer = load_model(device, spec)

    pair_results = []
    for pair in PAIRS:
        s_gps = surprisal(model, tokenizer, pair.gps_prefix, pair.critical_word, device)
        s_norm = surprisal(
            model, tokenizer, pair.normal_prefix, pair.critical_word, device
        )
        pair_results.append(
            {
                "id": pair.id,
                "template": pair.template,
                "critical_word": pair.critical_word,
                "surprisal_gps": s_gps,
                "surprisal_normal": s_norm,
                "delta": s_gps - s_norm,
            }
        )

    del model
    del tokenizer
    cleanup(device)

    return aggregate(spec, pair_results)


def load_existing_results() -> dict[str, dict]:
    """Read previous run, keyed by hf_name. Successful entries are kept and
    skipped on the next run; entries with "error" are retried."""
    if not RESULTS_PATH.exists():
        return {}
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    return {r["model"]["hf_name"]: r for r in data.get("results", [])}


def main():
    device = get_device()
    cached = load_existing_results()
    successful_names = {name for name, r in cached.items() if "error" not in r}
    print(
        f"Loaded {len(cached)} previous result(s) — "
        f"{len(successful_names)} successful (skip), "
        f"{len(cached) - len(successful_names)} errored (retry)"
    )

    all_results: list[dict] = []
    for idx, model in enumerate(MODELS):
        if model.hf_name in successful_names:
            print(f"{idx} - {model.hf_name}:{model.approx_params}  → cached, skip")
            all_results.append(cached[model.hf_name])
            continue

        try:
            print(f"{idx} - {model.hf_name}:{model.approx_params}")
            model_results = evaluate_model(device, model)
            all_results.append(model_results)
            if "error" in model_results:
                print(f"  → {model_results['error']}")
            else:
                print(
                    f"  → Δ̄ = {model_results['mean_delta']:+.3f}   "
                    f"acc = {model_results['accuracy']:.0%}   "
                    f"p = {model_results['wilcoxon_p']:.4g}"
                )
        except Exception as e:                                  # noqa: BLE001
            print(f"  → FAILED: {type(e).__name__}: {e}")
            all_results.append({"model": asdict(model), "error": f"{type(e).__name__}: {e}"})
            cleanup(device)
        print()

        # Persist incrementally so a crash mid-run doesn't lose progress
        with open(RESULTS_PATH, "w") as f:
            json.dump({"results": all_results}, f, indent=2)

    # print_ranking(all_results)
    # print_per_model_templates(all_results)
    # print(f"\nFull results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
