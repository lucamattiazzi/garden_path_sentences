"""Render six GPS figures from audited aggregate evidence, without inference."""

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "evidence.json").read_text())
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
(HERE / ".build").mkdir(exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "svg.fonttype": "path",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
    }
)


def point_color(label):
    if "Vantaggio" in label:
        return "#3c7955"
    if "lessicale" in label:
        return "#be7d43"
    if "casuale" in label:
        return "#80918a"
    return "#2467a0"


for item in DATA["figures"]:
    panels = item["panels"]
    if item["id"] == "figure-v4-gate":
        values = np.array([[r["correct"] for r in p["rows"]] for p in panels]).T
        fig, ax = plt.subplots(figsize=(9.5, 6.5))
        ax.imshow(values, cmap="RdYlGn", vmin=0, vmax=6, aspect="auto", alpha=0.65)
        names = {
            "active": "Attiva",
            "passive": "Passiva",
            "object_relative": "Relativa oggetto",
            "passive_relative": "Relativa passiva",
            "agent": "agente",
            "patient": "paziente",
        }
        labels = [" · ".join(names[s] for s in r["label"].split("/")) for r in panels[0]["rows"]]
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.set_xticks([0, 1], ["Gemma 3 12B", "Gemma 3 27B"])
        for row in range(len(labels)):
            for col in range(2):
                ax.text(
                    col,
                    row,
                    f"{values[row, col]}/6",
                    ha="center",
                    va="center",
                    fontsize=15,
                    weight="bold",
                )
        ax.set_title(
            "V4 originale · il gate si ferma sulle relative passive",
            loc="left",
            pad=22,
            weight="bold",
        )
        fig.text(
            0.02,
            0.025,
            "Famiglie corrette in entrambe le inversioni. Soglia: almeno 5/6 in OGNI cella. Nessun patch V4 eseguito.",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.055, 1, 1))
    else:
        count = len(panels)
        height = sum(max(2.25, len(p["rows"]) * 0.45 + 1.15) for p in panels) + 0.65
        fig, axes = plt.subplots(count, 1, figsize=(10, height), squeeze=False)
        shared = item["id"] in ["figure-v3-interaction", "figure-bridge-roles"]
        if shared:
            low = min(r["ci95"][0] for p in panels for r in p["rows"])
            high = max(r["ci95"][1] for p in panels for r in p["rows"])
            span = max(high - low, 0.01)
            bounds = (min(low, 0) - span * 0.1, max(high, 0) + span * 0.2)
        for ax, panel in zip(axes[:, 0], panels, strict=True):
            rows = panel["rows"]
            for i, row in enumerate(rows):
                mean = row["mean"]
                lo, hi = row["ci95"]
                ax.errorbar(
                    mean,
                    i,
                    xerr=[[mean - lo], [hi - mean]],
                    fmt="o",
                    markersize=7,
                    color=point_color(row["label"]),
                    elinewidth=2,
                    capsize=4,
                )
                ax.annotate(
                    f"{mean:+.3f}",
                    (hi, i),
                    xytext=(8, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=9,
                    color="#23312e",
                )
            ax.axvline(0, color="#89918d", lw=0.8, ls="--")
            ax.set_yticks(np.arange(len(rows)), [r["label"] for r in rows])
            ax.set_ylim(len(rows) - 0.45, -0.6)
            ax.set_title(panel["label"], loc="left", fontsize=12, pad=10)
            ax.grid(axis="x", alpha=0.12)
            if shared:
                ax.set_xlim(*bounds)
            else:
                low = min(0, min(r["ci95"][0] for r in rows))
                high = max(0, max(r["ci95"][1] for r in rows))
                span = max(high - low, 0.01)
                ax.set_xlim(low - span * 0.1, high + span * 0.25)
            xlabel = (
                "Interazione: effetto opzionale meno intransitivo (nat)"
                if item["id"] == "figure-v3-interaction"
                else "Variazione orientata del margine (nat)"
            )
            ax.set_xlabel(xlabel, fontsize=10)
        fig.suptitle(item["title"], x=0.035, ha="left", weight="bold", fontsize=14)
        caption = "Punti: medie di sei famiglie. Barre: IC 95% bootstrap descrittivi."
        if count > 1 and not shared:
            caption += " Scale orizzontali diverse fra pannelli."
        fig.text(0.035, 0.014, "\n".join(textwrap.wrap(caption, 110)), fontsize=9)
        fig.tight_layout(rect=(0, 0.045, 1, 0.955), h_pad=2.2)
    fig.savefig(OUT / (item["id"] + ".svg"))
    fig.savefig(HERE / ".build" / (item["id"] + ".png"), dpi=110)
    plt.close(fig)
print("Rendered six figures from frozen aggregate evidence")
