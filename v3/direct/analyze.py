"""Independent audit and offline report for direct participant answers."""

import argparse
import base64
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def interval(values: list[float]) -> dict:
    draws = np.random.default_rng(0).choice(values, (10000, len(values))).mean(axis=1)
    return dict(mean=float(np.mean(values)), ci95=np.quantile(draws, [0.025, 0.975]).tolist())


def summarize(rows: list[dict]) -> list[dict]:
    cells = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in ["condition", "voice", "role", "family"])
        if row["reversal"] in cells[key]:
            raise ValueError("Duplicate reversal")
        cells[key][row["reversal"]] = row["margin"]
    groups = defaultdict(list)
    for (condition, voice, role, family), values in sorted(cells.items()):
        if set(values) != {0, 1}:
            raise ValueError("Missing reversal")
        margins = list(values.values())
        groups[condition, voice, role].append(
            dict(
                family=family,
                margin=float(np.mean(margins)),
                accuracy=float(np.mean(np.array(margins) > 0)),
                both_reversals=float(min(margins) > 0),
            )
        )
    return [
        dict(
            condition=c,
            voice=v,
            role=r,
            families=fs,
            **{m: interval([f[m] for f in fs]) for m in ["margin", "accuracy", "both_reversals"]},
        )
        for (c, v, r), fs in sorted(groups.items())
    ]


def audit(directory: Path) -> tuple[dict, dict, list[dict], dict]:
    meta, result, material, source = [
        json.loads((directory / name).read_text())
        for name in ["metadata.json", "results.json", "materials.json", "source.json"]
    ]
    rows = [
        json.loads(line) for line in (directory / "observations.jsonl").read_text().splitlines()
    ]
    if (
        not meta["complete"]
        or len(rows) != 96
        or result["baselines"] != 96
        or result["interventions"]
        or result["test_inference"]
    ):
        raise ValueError("Run scope or completion mismatch")
    for name, digest in meta["source_sha256"].items():
        if hashlib.sha256(source[name].encode()).hexdigest() != digest:
            raise ValueError("Source mismatch")
    for row, original in zip(rows, material, strict=True):
        if any(row[k] != v for k, v in original.items()) or row["split"] != "development":
            raise ValueError("Material changed or test data used")
        scores = row["scores"]["candidates"]
        if len(scores) != 2:
            raise ValueError("Two candidates required")
        for score in scores:
            if (
                not score["token_ids"]
                or len(score["token_ids"]) != len(score["token_logp"])
                or abs(sum(score["token_logp"]) - score["logp"]) > 1e-9
            ):
                raise ValueError("Sequence score mismatch")
        gold = row["gold_index"]
        if abs(scores[gold]["logp"] - scores[1 - gold]["logp"] - row["margin"]) > 1e-9:
            raise ValueError("Gold margin mismatch")
    summary = summarize(rows)
    if len(summary) != 8 or any(len(r["families"]) != 6 for r in summary):
        raise ValueError("Incomplete strata")
    gate = all(r["both_reversals"]["mean"] >= 0.75 for r in summary if r["condition"] == "simple")
    if gate != result["gate"]["passed"]:
        raise ValueError("Gate mismatch")
    for item in result["gate"]["strata"]:
        actual = next(
            r for r in summary if all(r[k] == item[k] for k in ["condition", "voice", "role"])
        )
        if actual["both_reversals"]["mean"] != item["both_reversals"]:
            raise ValueError("Pair aggregation mismatch")
    no_context = defaultdict(list)
    for row in rows:
        if row["condition"] == "no_context":
            no_context[row["family"], row["voice"], row["role"]].append(row)
    for a, b in no_context.values():
        if (
            a["prompt"] != b["prompt"]
            or a["gold_index"] == b["gold_index"]
            or a["scores"] != b["scores"]
            or abs(a["margin"] + b["margin"]) > 1e-9
        ):
            raise ValueError("No-context counterfactual check failed")
    checks = dict(
        passed=True,
        sources=len(source),
        baselines=len(rows),
        independent_gate=True,
        no_context_counterfactuals=True,
        test_inference=False,
    )
    return meta, result, rows, dict(checks=checks, summary=summary)


def report(directory: Path) -> None:
    import markdown
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta, result, rows, audit_data = audit(directory)
    summary = audit_data["summary"]
    simple = [r for r in summary if r["condition"] == "simple"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    x = np.arange(4)
    for ax, metric, title in zip(
        axes,
        ["accuracy", "both_reversals"],
        ["Accuratezza per frase", "Corrette entrambe le inversioni"],
        strict=True,
    ):
        for index, condition in enumerate(["simple", "no_context"]):
            selected = [r for r in summary if r["condition"] == condition]
            ax.bar(
                x + (index - 0.5) * 0.34,
                [r[metric]["mean"] * 100 for r in selected],
                width=0.34,
                label=condition,
                color=["#1e695c", "#aaaaaa"][index],
            )
        ax.set(
            title=title,
            xticks=x,
            xticklabels=[
                "Attiva\nagente",
                "Attiva\npaziente",
                "Passiva\nagente",
                "Passiva\npaziente",
            ],
            ylim=(0, 105),
            ylabel="%",
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8)
    axes[1].axhline(75, color="#ad4c31", ls=":", label="Soglia")
    axes[1].legend(fontsize=8)
    fig.suptitle("Risposte dirette sui nomi · Gemma 3 1B FP32 · sei famiglie di sviluppo")
    fig.savefig(directory / "report.svg")
    fig.savefig(directory / "report.png", dpi=180)
    plt.close(fig)
    passed = result["gate"]["passed"]
    text = "# V3 · Rispondere direttamente con il nome\n\n"
    text += (
        "**Il nuovo formato "
        + ("supera" if passed else "non supera")
        + " il controllo di comprensione prefissato.** Abbiamo eliminato le etichette A/B e valutato risposte complete con i nomi dei partecipanti. Questa misura riguarda soltanto lo sviluppo, non le famiglie di test.\n\n"
    )
    text += "## Risultati sulle frasi semplici\n\n| Voce | Ruolo richiesto | Accuratezza | Entrambe le inversioni | Coppie riuscite |\n|---|---|---:|---:|---:|\n"
    for row in simple:
        text += f"| {row['voice']} | {row['role']} | {row['accuracy']['mean']:.1%} | {row['both_reversals']['mean']:.1%} | {sum(f['both_reversals'] for f in row['families']):.0f}/6 |\n"
    text += "\nLa soglia era 75% di coppie corrette **in ciascuna combinazione di voce e ruolo**: almeno cinque famiglie su sei. Una coppia contiene la frase originale e quella con i partecipanti scambiati. Preferire sempre lo stesso nome non basta.\n\n"
    text += "## Che cosa misuriamo\n\nQuattro esempi fissi mostrano risposte espresse con nomi. Il prompt termina con `Answer: the`; confrontiamo, per esempio, ` poet.` e ` chef.` sommando i logaritmi delle probabilità di tutti i token aggiunti, incluso il punto. I candidati non sono elencati nel prompt. Questa è accuratezza di scelta fra due continuazioni, non accuratezza di generazione libera.\n\n"
    counts = sorted({len(c["token_ids"]) for r in rows for c in r["scores"]["candidates"]})
    unequal = sum(
        len(r["scores"]["candidates"][0]["token_ids"])
        != len(r["scores"]["candidates"][1]["token_ids"])
        for r in rows
    )
    mass = np.median(
        [
            sum(np.exp(c["logp"]) for c in r["scores"]["candidates"])
            for r in rows
            if r["condition"] == "simple"
        ]
    )
    text += f"Token per risposta completa osservati: {counts}. Coppie di candidati di lunghezza diversa: {unequal}/96. Mediana della probabilità congiunta dei due candidati, condizionata al prefisso comune `Answer: the`, sulle frasi semplici: {mass:.1%}. Nessuna normalizzazione per lunghezza né esclusione di item.\n\n"
    text += "Senza frase, i prompt sono identici nelle due inversioni, mentre la risposta corretta cambia: il controllo deve fallire almeno una delle due. L’audit verifica identità dei punteggi e inversione esatta del margine.\n\n"
    text += "## Dettaglio e incertezza\n\n| Condizione | Voce | Ruolo | Margine medio [IC95%], nats | Entrambe le inversioni [IC95%] |\n|---|---|---|---:|---:|\n"
    for row in summary:
        m, b = row["margin"], row["both_reversals"]
        text += f"| {row['condition']} | {row['voice']} | {row['role']} | {m['mean']:.3f} [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}] | {b['mean']:.1%} [{b['ci95'][0]:.1%}, {b['ci95'][1]:.1%}] |\n"
    text += "\nIntervalli bootstrap descrittivi su sei famiglie, 10.000 ricampionamenti; inversioni e domande non sono repliche indipendenti. I risultati per famiglia sono in `audit.json`. La media del margine può nascondere errori: il gate usa entrambe le risposte corrette.\n\n"
    pairs = defaultdict(dict)
    for row in rows:
        if row["condition"] == "simple":
            scores = [c["logp"] for c in row["scores"]["candidates"]]
            pairs[row["voice"], row["family"], row["reversal"]][row["role"]] = (
                None if scores[0] == scores[1] else int(scores[1] > scores[0])
            )
    text += "## Diagnosi descrittiva degli errori\n\n"
    for voice in ["active", "passive"]:
        selected = [p for (v, _, _), p in pairs.items() if v == voice]
        same = sum(p["agent"] is not None and p["agent"] == p["patient"] for p in selected)
        text += f"Nelle frasi {'attive' if voice == 'active' else 'passive'}, il modello preferisce lo stesso nome nelle domande sull'agente e sul paziente in **{same}/{len(selected)}** casi. Le due risposte corrette dovrebbero invece essere diverse.\n\n"
    text += "Questa verifica post-hoc descrive una scarsa sensibilità al ruolo richiesto, soprattutto nelle passive; non identifica il meccanismo interno dell'errore e non cambia la soglia del gate.\n\n"
    text += "## Interpretazione\n\n"
    text += (
        "Il gate passato autorizza a progettare un nuovo esperimento causale con questo endpoint; non dimostra ancora che la patch di V3 modifichi le risposte. Occorre congelare separatamente i controlli del nuovo ponte prima di interrogare il test.\n\n"
        if passed
        else "Il passaggio ai nomi non ha validato lo strumento secondo il criterio stabilito. Non eseguiamo patch né inferenza sul test. Dopo due formati falliti, altre varianti di prompt sullo stesso piccolo checkpoint rischiano di diventare una ricerca opportunistica: la priorità è una verifica di competenza su un modello più capace, con un nuovo protocollo di sviluppo.\n\n"
    )
    text += "Il confronto con A/B non è un confronto sperimentale isolato: cambiano le dimostrazioni, spariscono le opzioni e cambia la metrica di robustezza (inversione dei partecipanti invece dell’ordine delle opzioni). Questo seguito è stato scelto dopo aver visto i risultati precedenti. Le frasi sono sintetiche e prive di validazione umana. Il fallimento non prova assenza di rappresentazioni dei ruoli; il successo potrebbe ancora usare regole superficiali. Gli effetti predittivi già osservati in V3 rimangono distinti da queste domande di comprensione.\n\n"
    text += f"## Riproducibilità\n\nModello `{meta['model']}`, revisione `{meta['revision']}`, FP32, TF32 disabilitato, attenzione eager. 96 osservazioni (192 forward), nessuna patch e nessuna inferenza sul test. Funzione GPU su {meta['gpu']}: {meta['elapsed_seconds']:.1f} secondi, escluso provisioning. Audit dei {len(meta['source_sha256'])} sorgenti, somme dei token, margini, coppie e ricalcolo del gate: verifiche superate.\n\nProtocollo: [PROTOCOL.md](../../PROTOCOL.md). Dati e snapshot nella stessa cartella: `observations.jsonl`, `materials.json`, `source.json`, `metadata.json`, `results.json`, `audit.json`.\n"
    (directory / "audit.json").write_text(json.dumps(audit_data, indent=2, allow_nan=False))
    (directory / "report.md").write_text(text)
    css = (Path(__file__).resolve().parents[2] / "v2/html_assets/style.css").read_text()
    body = markdown.markdown(text, extensions=["tables"])
    body = body.replace("<table>", '<div class="table-scroll" tabindex="0"><table>').replace(
        "</table>", "</table></div>"
    )
    encoded = base64.b64encode((directory / "report.svg").read_bytes()).decode()
    chart = (
        '<figure class="chart-card"><div class="chart"><img alt="Accuratezza e robustezza alle inversioni, con e senza frase" src="data:image/svg+xml;base64,'
        + encoded
        + '"></div></figure>'
    )
    html = (
        '<!doctype html><html lang="it"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V3 — Risposte dirette sui nomi</title><style>'
        + css
        + 'main{padding-top:40px!important;padding-bottom:50px!important}h1{font:42px Georgia}h2{margin-top:40px}</style><body><main class="wrap">'
    )
    (directory / "report.html").write_text(
        html + body.replace("<h2>", chart + "<h2>", 1) + "</main></body></html>"
    )
    print(json.dumps(audit_data["checks"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    report(parser.parse_args().directory)
