"""Audit and plot the complete fixed GPS title case study, without inference."""

import hashlib
import json
from pathlib import Path

from gps_title.probe import CASES, word_scores

ROOT = Path(__file__).resolve().parents[1]
RUN = Path(__file__).parent / "results/title-20261013-01"
OUT = ROOT / "reports/2026-10-12-gps"


def analyze():
    data = json.loads((RUN / "results.json").read_text())
    meta = json.loads((RUN / "metadata.json").read_text())
    snapshots = json.loads((RUN / "source.json").read_text())
    assert meta["complete"] and set(data["surprisal"]) == set(CASES) == set(data["interpretations"])
    for name, sha in meta["source_sha256"].items():
        assert hashlib.sha256(snapshots[name].encode()).hexdigest() == sha, name
    for name, row in data["surprisal"].items():
        assert row["text"] == CASES[name]
        rebuilt = word_scores(row["text"], row["offsets"], row["token_logps"])
        for before, after in zip(row["words"], rebuilt, strict=True):
            assert all(before[k] == v for k, v in after.items())
        assert abs(sum(w["surprisal_nats"] for w in rebuilt) + sum(row["token_logps"][1:])) < 1e-6
    for name, row in data["interpretations"].items():
        assert row["text"] == CASES[name] and row["generated_tokens"] < row["max_new_tokens"]
    legacy = []
    for size in ["1b", "4b", "12b", "27b"]:
        path = ROOT / f"v1/gemma3_{size}/logs/01_baseline.json"
        row = next(
            r for r in json.loads(path.read_text())["results"]["pair_results"] if r["id"] == 16
        )
        legacy.append(
            dict(
                model="Gemma 3 " + size.upper(),
                **row,
                source=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    result = dict(
        date="2026-10-13",
        metadata=meta,
        **data,
        legacy_old_man=legacy,
        limitations=[
            "Exploratory fixed cases, not independent linguistic families.",
            "PT surprisal and IT interpretation are different checkpoints and tasks.",
            "Final punctuation belongs to the last plotted word.",
            "Raw pretrained logits include special tokens. The exact lowercase title predicts start_of_image most strongly after models.",
            "IT fails even the old/man examples; its prompted parse is not a validated comprehension endpoint.",
            "No activation patching, circuit discovery or test of necessity/sufficiency performed.",
        ],
    )
    (OUT / "title-case-study.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    (RUN / "verification.json").write_text(
        json.dumps(
            dict(
                complete=True,
                cases=6,
                source_hashes=True,
                word_token_sums=True,
                responses_not_truncated=True,
                new_inference=True,
            ),
            indent=2,
        )
        + "\n"
    )
    return result


def plot(data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"svg.fonttype": "path", "font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(10, 4.5))
    rows = data["legacy_old_man"]
    y = np.arange(len(rows))
    ax.barh(
        y - 0.18,
        [r["surprisal_gps"] for r in rows],
        height=0.34,
        label="The old man → the",
        color="#2467a0",
    )
    ax.barh(
        y + 0.18,
        [r["surprisal_normal"] for r in rows],
        height=0.34,
        label="The old people man → the",
        color="#80918a",
    )
    ax.set_yticks(y, [r["model"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Surprisal di «the» (nat)")
    ax.set_title("V1 · il token disambiguante originale", loc="left", weight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.03,
        0.015,
        "Un singolo esempio storico per modello; non è una curva di generalizzazione o scaling.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "figures/figure-old-man.svg")
    fig.savefig(OUT / ".build/figure-old-man.png", dpi=120)
    plt.close(fig)
    names = ["title_exact", "title_article", "title_verb_cue", "title_noun_cue"]
    labels = [
        "Titolo esatto, senza articolo",
        "Titolo con «The» e punto",
        "Indizio verbale: «actually models»",
        "Lettura nominale: «models affects»",
    ]
    values = [
        next(w["surprisal_nats"] for w in data["surprisal"][name]["words"] if w["word"] == "human")
        for name in names
    ]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.barh(range(4), values, color=["#80918a", "#2467a0", "#3c7955", "#be7d43"])
    ax.set_yticks(range(4), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 17)
    for i, v in enumerate(values):
        ax.text(v + 0.15, i, f"{v:.3f}", va="center")
    ax.set_xlabel("Surprisal di «human» (nat)")
    ax.set_title("Titolo · Gemma 3 12B pretrained, FP32", loc="left", weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.03,
        0.015,
        "Casi esplorativi con contesti diversi. Nessuna patch, nessun intervallo o effetto sintattico puro.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "figures/figure-title-human.svg")
    fig.savefig(OUT / ".build/figure-title-human.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    plot(analyze())
    print("Verified six cases, source hashes and all word/token sums; rendered two figures")
