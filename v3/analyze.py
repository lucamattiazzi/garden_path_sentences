"""Audit raw V3 measurements and build an offline Italian report without inference.

uv run --no-project --with numpy --with matplotlib --with markdown python v3/analyze.py v3/results/<run-id>
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CONTRASTS = ["comma_native", "comma_aligned", "position_only"]
NAMES = {
    "comma_native": "Virgola / originale",
    "comma_aligned": "Virgola / posizioni allineate",
    "position_only": "Solo posizioni",
}


def family_values(rows: list[dict], kind: str, contrast: str = "comma_aligned") -> dict[str, float]:
    cells = defaultdict(dict)
    for r in rows:
        if (r["split"], r["stage"], r["contrast"], r["kind"]) != ("test", "next", contrast, kind):
            continue
        key = (r["family"], r["id"], r["seed"])
        cell = (r["verb_class"], r["direction"])
        if cell in cells[key]:
            raise ValueError("Duplicate cell")
        cells[key][cell] = r["effects"]["margin"]
    per_item = defaultdict(list)
    for (family, item, _seed), values in cells.items():
        if set(values) != {
            (v, d) for v in ["optional", "intransitive"] for d in ["forward", "reverse"]
        }:
            raise ValueError("Incomplete factorial")
        value = (
            values["optional", "forward"]
            - values["intransitive", "forward"]
            - values["optional", "reverse"]
            + values["intransitive", "reverse"]
        ) / 2
        per_item[family, item].append(value)
    per_family = defaultdict(list)
    for (family, _item), values in per_item.items():
        if len(values) != (3 if kind == "random" else 1):
            raise ValueError("Missing seeds")
        per_family[family].append(float(np.mean(values)))
    return {family: float(np.mean(values)) for family, values in sorted(per_family.items())}


def audit(directory: Path) -> tuple[dict, dict, dict]:
    metadata = json.loads((directory / "metadata.json").read_text())
    if not metadata["complete"]:
        raise ValueError("Incomplete run")
    data = json.loads((directory / "results.json").read_text())
    snapshot = json.loads((directory / "source.json").read_text())
    for name, digest in metadata["source_sha256"].items():
        if hashlib.sha256(snapshot[name].encode()).hexdigest() != digest:
            raise ValueError("Source snapshot mismatch")
    raw = [json.loads(line) for line in (directory / "observations.jsonl").read_text().splitlines()]
    if raw[0]["record"] != "frozen_selection" or raw[0]["layer"] != 17:
        raise ValueError("Missing initial fixed-layer declaration")
    baselines = [r for r in raw if r["record"] == "baseline"]
    interventions = [r for r in raw if r["record"] == "intervention"]
    if len(baselines) != 288 or len(interventions) != 3456:
        raise ValueError("Unexpected run size")
    if baselines != data["baselines"]:
        raise ValueError("Baseline artifact mismatch")
    lookup = {
        (r["id"], r["verb_class"], r["stage"], r["condition"]): r["scores"] for r in baselines
    }
    for r in interventions:
        base = lookup[r["id"], r["verb_class"], r["stage"], r["recipient"]]
        changed = r["scores"]
        for metric, effect in r["effects"].items():
            if metric == "three_words":
                delta = sum(changed["token_logp"]) - sum(base["token_logp"])
            elif metric.startswith("word_"):
                index = int(metric.split("_")[1])
                delta = changed["word_logp"][index] - base["word_logp"][index]
            else:
                delta = changed[metric] - base[metric]
            if abs(delta - effect) > 1e-9:
                raise ValueError("Raw effect mismatch")
    families = {}
    for contrast in CONTRASTS:
        families[contrast] = {
            kind: family_values(interventions, kind, contrast)
            for kind in ["cross", "lexical", "random"]
        }
        for kind, values in families[contrast].items():
            row = select(data, contrast, kind)
            if not np.isclose(np.mean(list(values.values())), row["mean"], atol=1e-12):
                raise ValueError("Independent family aggregation mismatch")
    for split in ["development", "test"]:
        rows = {
            r["contrast"]: r["mean"]
            for r in data["baseline_summaries"]
            if r["split"] == split and r["stage"] == "next" and r["metric"] == "margin"
        }
        if not np.isclose(
            rows["comma_native"], rows["comma_aligned"] + rows["position_only"], atol=1e-12
        ):
            raise ValueError("Baseline contrast decomposition failed")
    summary = dict(
        source_snapshot_verified=True,
        raw_effects_verified=len(interventions),
        family_values=families,
        maximum_norm_mismatch_by_contrast={
            contrast: max(
                abs(row["actual_l2"] / row["requested_l2"] - 1)
                for row in interventions
                if row["contrast"] == contrast and row.get("requested_l2", 0) > 0
            )
            for contrast in CONTRASTS
        },
        raw_sha256=hashlib.sha256((directory / "observations.jsonl").read_bytes()).hexdigest(),
        results_sha256=hashlib.sha256((directory / "results.json").read_bytes()).hexdigest(),
    )
    return metadata, data, summary


def select(
    data: dict,
    contrast: str,
    kind: str,
    stage: str = "next",
    metric: str = "margin",
    direction: str = "average",
) -> dict:
    return next(
        r
        for r in data["summaries"]
        if (r["split"], r["contrast"], r["stage"], r["kind"], r["metric"], r["direction"])
        == ("test", contrast, stage, kind, metric, direction)
    )


def table(headers: list[str], rows: list[list]) -> str:
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
        + ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    )


def estimate(row: dict) -> str:
    return f"{row['mean']:.3f} [{row['ci95'][0]:.3f}, {row['ci95'][1]:.3f}]"


def render(metadata: dict, data: dict, checks: dict, directory: Path) -> str:
    primary = select(data, "comma_aligned", "cross")
    advantages = [select(data, "comma_aligned", "cross_minus_" + k) for k in ["lexical", "random"]]
    positive = all(r["ci95"][0] > 0 for r in [primary, *advantages])
    decision = (
        "Il segnale supera entrambi i controlli anche nel contrasto con posizioni allineate. "
        "Questo sostiene un effetto sulle preferenze di continuazione oltre questo particolare controllo posizionale."
        if positive
        else "Il criterio esplorativo completo non è soddisfatto: almeno uno degli intervalli "
        "dell’effetto o dei vantaggi sui controlli include zero o valori negativi. Prima di aumentare "
        "la dimensione del modello serve capire questa instabilità."
    )
    preferences = []
    for verb in ["optional", "intransitive"]:
        for condition in ["native", "gap", "comma"]:
            values = [
                r["scores"]["margin"]
                for r in data["baselines"]
                if (r["split"], r["stage"], r["verb_class"], r["condition"])
                == ("test", "next", verb, condition)
            ]
            preferences.append(
                [
                    verb,
                    condition,
                    f"{np.mean(values):+.3f}",
                    f"{sum(v > 0 for v in values)}/{len(values)}",
                ]
            )
    trajectory = [
        [
            f"Parola {i + 1}",
            estimate(select(data, "comma_aligned", "cross", "trajectory", f"word_{i}")),
            estimate(
                select(data, "comma_aligned", "cross_minus_lexical", "trajectory", f"word_{i}")
            ),
            estimate(
                select(data, "comma_aligned", "cross_minus_random", "trajectory", f"word_{i}")
            ),
        ]
        for i in range(3)
    ]
    return (
        "\n\n".join(
            [
                "# V3 — Preferenze sintattiche, un token alla volta",
                f"Run `{directory.name}` · Gemma 3 1B pretrained · layer 17 fissato prima del run · "
                f"{metadata['elapsed_seconds']:.1f} secondi nella funzione GPU su {metadata['gpu']}.",
                "**Stato: esperimento completato.** 24 materiali pubblicati, 12 famiglie di verbi, "
                "12 frasi di sviluppo e 12 di test. Le frasi sono due per famiglia: il test ha sei unità di ricampionamento, "
                "non dodici famiglie indipendenti. I risultati sono esplorativi.",
                "## Perché questa versione",
                "V2 confrontava continuazioni di lunghezze diverse. Qui il confronto principale è tra "
                "la parola `was` e una virgola, ciascuna verificata come singolo token. Salviamo le due "
                "log-probabilità, la loro differenza e gli effetti delle patch separatamente. "
                "Nessun punteggio di stringhe lunghe entra nella misura principale.",
                "I materiali provengono da [Hanna e Mueller (2025)](https://aclanthology.org/2025.naacl-long.164/) "
                "e dal [repository degli autori](https://github.com/hannamw/GP-mechanisms). "
                "Le varianti e le combinazioni usate qui sono adattamenti: non abbiamo raccolto giudizi umani sul nostro test.",
                "## Risultato principale sul test",
                decision,
                table(
                    ["Misura: interazione opzionale − intransitivo", "Media [IC95%], nats"],
                    [
                        ["Patch, posizioni allineate", estimate(primary)],
                        ["Vantaggio sul controllo lessicale", estimate(advantages[0])],
                        ["Vantaggio sulle direzioni casuali", estimate(advantages[1])],
                    ],
                ),
                "Una differenza positiva indica che la patch sposta `was` rispetto alla virgola più nel contesto "
                "con verbo opzionalmente transitivo che nel controllo intransitivo. Le due direzioni della patch "
                "sono allineate nel segno prima della media. Gli intervalli ricampionano intere famiglie, 10.000 volte, "
                "e non coprono l’incertezza delle scelte linguistiche o tutti i confronti del progetto.",
                "### Sviluppo e test, riportati separatamente",
                table(
                    ["Split", "Patch allineata", "Vantaggio lessicale", "Vantaggio casuale"],
                    [
                        [
                            split,
                            *[
                                estimate(
                                    next(
                                        row
                                        for row in data["summaries"]
                                        if (
                                            row["split"],
                                            row["contrast"],
                                            row["stage"],
                                            row["metric"],
                                            row["direction"],
                                            row["kind"],
                                        )
                                        == (
                                            split,
                                            "comma_aligned",
                                            "next",
                                            "margin",
                                            "average",
                                            kind,
                                        )
                                    )
                                )
                                for kind in ["cross", "cross_minus_lexical", "cross_minus_random"]
                            ],
                        ]
                        for split in ["development", "test"]
                    ],
                ),
                "## Che cosa controlliamo sulle posizioni",
                "In “gap” i token sono identici al prefisso senza virgola, ma i position ID dal secondo gruppo nominale "
                "in poi aumentano di uno. Corrispondono così ai position ID degli stessi token nella frase con virgola. "
                "Non viene inserito un token nascosto: è una perturbazione artificiale della codifica posizionale.",
                "Questo controllo non riproduce il contributo del token virgola all’attenzione e non isola perfettamente "
                "la sintassi dalla punteggiatura. Esclude soltanto una spiegazione basata interamente su questo specifico "
                "spostamento dei position ID, se il segnale resta presente.",
                table(
                    [
                        "Contrasto",
                        "Interazione di base",
                        "Patch",
                        "Controllo lessicale",
                        "Controllo casuale",
                    ],
                    [
                        [
                            NAMES[c],
                            estimate(
                                next(
                                    r
                                    for r in data["baseline_summaries"]
                                    if (r["split"], r["stage"], r["metric"], r["contrast"])
                                    == ("test", "next", "margin", c)
                                )
                            ),
                            *[estimate(select(data, c, k)) for k in ["cross", "lexical", "random"]],
                        ]
                        for c in CONTRASTS
                    ],
                ),
                "Le tre interazioni di base si scompongono algebricamente: virgola/originale = "
                "virgola/allineato + solo posizioni. Le patch agiscono in un sistema non lineare e non devono soddisfare la stessa somma.",
                "## Quale token cambia?",
                table(
                    ["Componente, posizioni allineate", "Media [IC95%], nats"],
                    [
                        [
                            "Effetto su log P(was)",
                            estimate(select(data, "comma_aligned", "cross", metric="was_logp")),
                        ],
                        [
                            "Effetto su log P(virgola)",
                            estimate(select(data, "comma_aligned", "cross", metric="comma_logp")),
                        ],
                        ["Differenza was − virgola", estimate(primary)],
                    ],
                ),
                "Il contributo della virgola va sottratto: la tabella riporta il suo segno originale. "
                "L’interazione misura una differenza tra classi di verbo, non un cambiamento assoluto identico in ogni frase.",
                "## Prima della patch: was o virgola?",
                table(
                    [
                        "Classe del verbo",
                        "Condizione",
                        "Margine medio, nats",
                        "Frasi con P(was) > P(virgola)",
                    ],
                    preferences,
                ),
                "Un margine negativo rende la virgola più probabile di `was`. Questo non dimostra un parse sbagliato: "
                "le due possibilità non esauriscono le continuazioni né le interpretazioni sintattiche.",
                "## Dove cade l’effetto nella continuazione effettiva?",
                "La seconda misura usa le prime tre parole della proposizione principale originale. "
                "Si interviene sempre sul nome prima della disambiguazione e si salvano i punteggi di ogni token. "
                "Questa analisi secondaria riguarda parole effettive come “received unwelcome attention”, "
                "non necessariamente `was`. Non è la dimostrazione di un passaggio da interpretazione errata a corretta.",
                table(
                    [
                        "Posizione",
                        "Patch, nats [IC95%]",
                        "Vantaggio lessicale",
                        "Vantaggio casuale",
                    ],
                    trajectory,
                ),
                "## Stabilità tra famiglie e tra direzioni",
                table(
                    ["Famiglia di test", "Patch", "Lessicale", "Casuale"],
                    [
                        [
                            family,
                            *[
                                f"{checks['family_values']['comma_aligned'][k][family]:+.3f}"
                                for k in ["cross", "lexical", "random"]
                            ],
                        ]
                        for family in checks["family_values"]["comma_aligned"]["cross"]
                    ],
                ),
                table(
                    ["Direzione, posizioni allineate", "Interazione patch, nats [IC95%]"],
                    [
                        [
                            direction,
                            estimate(select(data, "comma_aligned", "cross", direction=direction)),
                        ]
                        for direction in ["forward", "reverse"]
                    ],
                ),
                "## Verifiche e limiti",
                f"- Identità: errore massimo {data['numerical_checks']['maximum_identity_error']:.8g} nats.\n"
                f"- Scarto relativo delle norme dopo arrotondamento BF16: mediana {data['numerical_checks']['median_norm_mismatch']:.3%}, massimo {data['numerical_checks']['maximum_norm_mismatch']:.3%}.\n"
                f"- Nel contrasto a posizioni allineate lo scarto massimo è {checks['maximum_norm_mismatch_by_contrast']['comma_aligned']:.3%}; le norme non sono esattamente identiche.\n"
                "- Verificati 3.456 effetti grezzi, 288 baseline, le aggregazioni per famiglia e gli hash dello snapshot.\n"
                "- Nessuna selezione di layer sui dati V3 e nessuna esclusione di frasi in base ai risultati.\n"
                "- Layer e checkpoint sono fissi; il confronto non verifica una legge di scala.\n"
                "- I verbi di controllo e alcuni nomi ricorrono tra split; sono disgiunte le famiglie del verbo opzionale.\n"
                "- Le nuove frasi sono pubbliche e potrebbero essere state viste in addestramento.\n"
                "- La virgola è un indizio sintattico ma anche un token con proprie regolarità; patchare l’intero residuo non identifica una feature di sintassi.\n"
                "- Non abbiamo addestrato nuove sonde dei ruoli né dimostrato rianalisi online.",
                "## Prossima decisione",
                "Prima di cercare feature SAE o scalare a modelli grandi, verificare il significato linguistico del "
                "contrasto su nuovi materiali e collegarlo a un compito comportamentale sui ruoli. "
                "La generalizzazione delle sonde di V2 resta un problema separato. Dopo questa ispezione, "
                "i dati V3 non sono più una conferma intatta per un protocollo modificato.",
                "## Riproduzione",
                f"Checkpoint `{metadata['revision']}`. Protocollo e sorgenti originali sono in `source.json`; "
                "misure grezze in `observations.jsonl`, riepiloghi in `results.json` e audit in `audit.json`.",
                "```bash\nuv run --no-project --with numpy --with matplotlib --with markdown python v3/analyze.py "
                f"v3/results/{directory.name}\n```",
            ]
        )
        + "\n"
    )


def plot(data: dict, directory: Path) -> bytes:
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for offset, kind, color in [
        (-0.2, "cross", "#1e695c"),
        (0, "lexical", "#ad4c31"),
        (0.2, "random", "#888888"),
    ]:
        rows = [select(data, c, kind) for c in CONTRASTS]
        means = np.array([r["mean"] for r in rows])
        cis = np.array([r["ci95"] for r in rows])
        axes[0, 0].errorbar(
            np.arange(3) + offset,
            means,
            yerr=np.maximum(0, np.array([means - cis[:, 0], cis[:, 1] - means])),
            fmt="o",
            capsize=3,
            label=kind,
            color=color,
        )
    axes[0, 0].set(
        title="Interazione delle patch sul test",
        xticks=np.arange(3),
        xticklabels=["Virgola", "Allineata", "Solo posizioni"],
        ylabel="Nats: was − virgola",
    )
    axes[0, 0].legend(fontsize=8)
    for verb, color in [("optional", "#1e695c"), ("intransitive", "#ad4c31")]:
        values = []
        for condition in ["native", "gap", "comma"]:
            values.append(
                np.mean(
                    [
                        r["scores"]["margin"]
                        for r in data["baselines"]
                        if (r["split"], r["stage"], r["verb_class"], r["condition"])
                        == ("test", "next", verb, condition)
                    ]
                )
            )
        axes[0, 1].plot(range(3), values, "o-", label=verb, color=color)
    axes[0, 1].set(
        title="Preferenza prima della patch",
        xticks=range(3),
        xticklabels=["Originale", "Gap posizionale", "Virgola"],
        ylabel="Log P(was) − log P(virgola)",
    )
    axes[0, 1].legend(fontsize=8)
    for kind, color in [("cross", "#1e695c"), ("lexical", "#ad4c31"), ("random", "#888888")]:
        rows = [select(data, "comma_aligned", kind, "trajectory", f"word_{i}") for i in range(3)]
        axes[1, 0].plot(range(1, 4), [r["mean"] for r in rows], "o-", label=kind, color=color)
    axes[1, 0].set(
        title="Effetto sulle tre parole effettive",
        xticks=range(1, 4),
        xlabel="Parola nella continuazione",
        ylabel="Interazione media, nats",
    )
    axes[1, 0].legend(fontsize=8)
    rows = [select(data, "comma_aligned", k) for k in ["cross_minus_lexical", "cross_minus_random"]]
    means = np.array([r["mean"] for r in rows])
    cis = np.array([r["ci95"] for r in rows])
    axes[1, 1].errorbar(
        means,
        range(2),
        xerr=np.maximum(0, np.array([means - cis[:, 0], cis[:, 1] - means])),
        fmt="o",
        capsize=4,
        color="#1e695c",
    )
    axes[1, 1].set(
        title="Vantaggi appaiati, posizioni allineate",
        yticks=range(2),
        yticklabels=["Su lessicale", "Su casuale"],
        xlabel="Nats, IC95% per famiglia",
    )
    axes[1, 1].axvline(0, color="gray", ls=":")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes.flat[:3]:
        axis.axhline(0, color="gray", ls=":", linewidth=0.8)
    fig.suptitle("V3 · Gemma 3 1B · layer 17 fisso · sei famiglie di test", fontsize=14)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="svg")
    fig.savefig(directory / "report.png", dpi=180)
    plt.close(fig)
    return buffer.getvalue()


def main(directory: Path) -> None:
    import markdown

    metadata, data, checks = audit(directory)
    report = render(metadata, data, checks, directory)
    (directory / "audit.json").write_text(json.dumps(checks, indent=2, allow_nan=False) + "\n")
    (directory / "report.md").write_text(report)
    svg = plot(data, directory)
    (directory / "report.svg").write_bytes(svg)
    css = (Path(__file__).resolve().parents[1] / "v2/html_assets/style.css").read_text()
    body = markdown.markdown(report, extensions=["tables", "fenced_code"])
    body = body.replace("<table>", '<div class="table-scroll" tabindex="0"><table>').replace(
        "</table>", "</table></div>"
    )
    html = (
        '<!doctype html><html lang="it"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V3 — Risultati NP/Z</title><style>'
        + css
        + 'main{padding-top:40px!important;padding-bottom:50px!important}h1{font:42px Georgia}h2{margin-top:40px}</style><body><main class="wrap">'
    )
    html += (
        '<figure class="chart-card"><div class="chart"><img alt="Confronto di patch e controlli, preferenze prima della patch, traiettoria per parola e vantaggi appaiati" src="data:image/svg+xml;base64,'
        + base64.b64encode(svg).decode()
        + '"></div><figcaption>Effetti esplorativi; intervalli per famiglia. I grafici scorrono orizzontalmente su mobile.</figcaption></figure>'
    )
    (directory / "report.html").write_text(html + body + "</main></body></html>")
    print(f"Audited run and wrote reports in {directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory)
