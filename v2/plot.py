"""Plot saved results without loading a model: uv run --no-project --with matplotlib ..."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run", type=Path)
args = parser.parse_args()
gps = json.loads((args.run / "garden-paths.json").read_text())["summaries"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for axis, site in zip(axes, ("pre_disambiguation", "post_disambiguation"), strict=True):
    for recipient, label in (("ambiguous", "Clean → ambiguous"), ("clean", "Ambiguous → clean")):
        rows = [r for r in gps if r["site"] == site and r["recipient"] == recipient]
        x = [r["layer"] for r in rows]
        (line,) = axis.plot(x, [r["effect"]["mean"] for r in rows], label=label)
        axis.plot(
            x,
            [r["lexical_effect"]["mean"] for r in rows],
            linestyle="--",
            color=line.get_color(),
            alpha=0.65,
            label=f"Lexical control ({recipient})",
        )
    axis.axhline(0, color="gray", linestyle=":")
    axis.axvline(max(x), color="gray", linestyle=":")
    axis.set(title=site.replace("_", " ").capitalize(), ylabel="Patch effect, nats")
    axis.legend(fontsize=7)
for axis in axes.flat:
    axis.set_xlabel("Decoder block (zero-based)")
    axis.spines[["top", "right"]].set_visible(False)
fig.suptitle(
    "Gemma 3 1B · exploratory pilot\nGarden-path prediction; final-block patch = readout control",
    fontsize=12,
)
fig.savefig(args.run / "curves.png", dpi=180)
fig.savefig(args.run / "curves.svg")
