"""Independently audit the behavioral bridge and generate an offline report."""

import argparse
import base64
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def aggregate_behavior(rows):
    pairs = defaultdict(dict)
    for row in rows:
        key = tuple(
            row[k] for k in ["split", "condition", "voice", "family", "id", "reversal", "role"]
        )
        if row["order"] in pairs[key]:
            raise ValueError("Duplicate order")
        pairs[key][row["order"]] = row["correct_margin"]
    families = defaultdict(lambda: defaultdict(list))
    for (split, condition, voice, family, _, _, _), orders in pairs.items():
        if set(orders) != {0, 1}:
            raise ValueError("Missing order")
        values = list(orders.values())
        for metric, value in [
            ("margin", np.mean(values)),
            ("accuracy", np.mean(np.array(values) > 0)),
            ("both_orders", float(min(values) > 0)),
        ]:
            for domain in [voice, "all"]:
                families[split, condition, domain, metric][family].append(float(value))
    return {
        key: {f: float(np.mean(v)) for f, v in sorted(fs.items())} for key, fs in families.items()
    }


def aggregate_patches(rows):
    cells = defaultdict(dict)
    for row in rows:
        key = tuple(
            row[k] for k in ["split", "family", "id", "reversal", "role", "order", "direction"]
        )
        for metric in ["qa_effect", "next_effect"]:
            label = (row["kind"], row["seed"])
            if label in cells[key, metric]:
                raise ValueError("Duplicate intervention")
            sign = 1 if row["direction"] == "clean_to_gap" else -1
            cells[key, metric][label] = sign * row[metric]
    families = defaultdict(lambda: defaultdict(list))
    for (key, metric), values in cells.items():
        split, family, _, _, _, _, direction = key
        means = {}
        for kind in ["cross", "lexical", "random", "identity"]:
            group = [value for (k, _), value in values.items() if k == kind]
            if kind == "identity" and not group:
                continue
            if len(group) != (3 if kind == "random" else 1):
                raise ValueError("Missing control or seed")
            means[kind] = float(np.mean(group))
        for kind in ["lexical", "random"]:
            means["cross_minus_" + kind] = means["cross"] - means[kind]
        for kind, value in means.items():
            for domain in [direction, "average"]:
                families[split, domain, kind, metric][family].append(value)
    return {
        key: {f: float(np.mean(v)) for f, v in sorted(fs.items())} for key, fs in families.items()
    }


def interval(values):
    array = np.array(list(values.values()))
    draws = np.random.default_rng(0).choice(array, (10000, len(array))).mean(axis=1)
    return dict(
        mean=float(array.mean()),
        ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
        families=len(array),
    )


def audit(directory):
    meta = json.loads((directory / "metadata.json").read_text())
    data = json.loads((directory / "results.json").read_text())
    if not meta["complete"]:
        raise ValueError("Incomplete run")
    sources = json.loads((directory / "source.json").read_text())
    for name, digest in meta["source_sha256"].items():
        if hashlib.sha256(sources[name].encode()).hexdigest() != digest:
            raise ValueError("Source hash mismatch")
    raw = [json.loads(line) for line in (directory / "observations.jsonl").read_text().splitlines()]
    if raw[0] != dict(
        record="frozen_protocol", layer=17, patch_voice="passive", gate_before_test=True
    ):
        raise ValueError("Missing initial protocol")
    baselines = [r for r in raw if r["record"] == "baseline"]
    patches = [r for r in raw if r["record"] == "intervention"]
    gates = [i for i, r in enumerate(raw) if r["record"] == "development_gate"]
    if len(gates) != 1 or {k: v for k, v in raw[gates[0]].items() if k != "record"} != data["gate"]:
        raise ValueError("Gate record mismatch")
    passed = data["gate"]["passed"]
    for i, row in enumerate(raw):
        if (row.get("split") == "test" or row["record"] == "intervention") and (
            not passed or i < gates[0]
        ):
            raise ValueError("Gate chronology violated")
    expected = (960, 1152) if passed else (480, 0)
    if (len(baselines), len(patches)) != expected or (
        data["baseline_count"],
        data["intervention_count"],
    ) != expected:
        raise ValueError("Wrong measurement count")
    materials = json.loads((directory / "materials.json").read_text())
    if materials != data["materials"] or len(materials) != 12:
        raise ValueError("Material snapshot mismatch")
    lookup = {}
    fields = ["id", "reversal", "voice", "condition", "role", "order"]
    for row in baselines:
        key = tuple(row[k] for k in fields)
        if key in lookup:
            raise ValueError("Duplicate baseline")
        lookup[key] = row
        gold = row["gold_index"]
        logs = row["scores"]["answer_logp"]
        if abs(logs[gold] - logs[1 - gold] - row["correct_margin"]) > 1e-9:
            raise ValueError("Raw correct margin mismatch")
        if abs(sum(np.exp(logs)) - row["answer_mass"]) > 1e-6:
            raise ValueError("Answer mass mismatch")
    behavior = aggregate_behavior(baselines)
    effects = aggregate_patches(patches)
    for table, keys, computed in [
        (data["behavioral"], ["split", "condition", "voice", "metric"], behavior),
        (data["patches"], ["split", "direction", "kind", "metric"], effects),
    ]:
        if len(table) != len(computed):
            raise ValueError("Aggregate row count mismatch")
        for row in table:
            independently = interval(computed[tuple(row[k] for k in keys)])
            for field in ["mean", "ci95", "families"]:
                if not np.allclose(row[field], independently[field], atol=1e-10, rtol=0):
                    raise ValueError("Independent aggregate mismatch")
    rates = {
        v: interval(behavior["development", "simple", v, "both_orders"])["mean"]
        for v in ["active", "passive"]
    }
    mass = float(
        np.median(
            [
                r["answer_mass"]
                for r in baselines
                if r["split"] == "development" and r["condition"] == "simple"
            ]
        )
    )
    if (
        passed != (min(rates.values()) >= 0.75 and mass >= 0.1)
        or rates != data["gate"]["both_orders_correct_by_voice"]
        or mass != data["gate"]["median_answer_mass"]
    ):
        raise ValueError("Gate recomputation failed")
    identity = []
    norm_errors = []
    for row in patches:
        base = lookup[tuple(row[k] for k in fields)]
        gold = base["gold_index"]
        logs = row["scores"]["answer_logp"]
        margin = logs[gold] - logs[1 - gold]
        if (
            abs(margin - row["correct_margin"]) > 1e-9
            or abs(margin - base["correct_margin"] - row["qa_effect"]) > 1e-9
            or abs(
                row["scores"]["next_margin"] - base["scores"]["next_margin"] - row["next_effect"]
            )
            > 1e-9
        ):
            raise ValueError("Intervention raw effect mismatch")
        if row["kind"] == "identity":
            identity.append(max(abs(row["qa_effect"]), abs(row["next_effect"])))
        if row.get("requested_l2", 0) > 0:
            norm_errors.append(abs(row["actual_l2"] / row["requested_l2"] - 1))
    if identity and max(identity) > 1e-4:
        raise ValueError("Identity check failed")
    checks = dict(
        passed=True,
        source_files=len(sources),
        baselines=len(baselines),
        interventions=len(patches),
        independent_aggregation=True,
        gate_chronology=True,
        maximum_identity_error=max(identity, default=None),
        maximum_relative_norm_error=max(norm_errors, default=None),
    )
    return meta, data, checks, baselines


def select(data, split, condition, voice, metric):
    return next(
        r
        for r in data["behavioral"]
        if (r["split"], r["condition"], r["voice"], r["metric"])
        == (split, condition, voice, metric)
    )


def fmt(row):
    return f"{row['mean']:.3f} [{row['ci95'][0]:.3f}, {row['ci95'][1]:.3f}]"


def render(meta, data, checks, baselines):
    passed = data["gate"]["passed"]
    text = "# V3 · Dalla previsione alle domande sui ruoli\n\n"
    text += (
        "Il controllo di comprensione è passato. Possiamo esaminare gli interventi sul test, entro i limiti di questo compito sintetico.\n\n"
        if passed
        else "**Il controllo di comprensione è fallito.** Questo formato di domande, con Gemma 3 1B pretrained, non supera il criterio fissato prima della misura. Non abbiamo eseguito patch causali né inferenza sulle famiglie di test.\n\n"
    )
    text += "## Il controllo prima dell'esperimento\n\n"
    text += "Il modello legge quattro esempi, una frase e una domanda sull'agente o sul paziente dell'evento principale. Valutiamo le probabilità delle etichette A/B; non sono risposte generate liberamente. Scambiamo sia i partecipanti sia l'ordine delle opzioni.\n\n"
    text += "Per procedere servivano almeno **75% di domande corrette in entrambi gli ordini**, separatamente per attiva e passiva, su frasi semplici di sviluppo, e una mediana di P(A)+P(B) almeno del 10%. Una risposta corretta in un solo ordine non basta.\n\n"
    text += "| Frasi semplici di sviluppo | Accuratezza per ordine | Corrette in entrambi gli ordini |\n|---|---:|---:|\n"
    for voice, label in [("active", "Attive"), ("passive", "Passive")]:
        a = select(data, "development", "simple", voice, "accuracy")["mean"]
        b = select(data, "development", "simple", voice, "both_orders")["mean"]
        text += f"| {label} | {a:.1%} | {b:.1%} |\n"
    text += f"\nMediana P(A)+P(B): **{data['gate']['median_answer_mass']:.1%}**. Sei famiglie di sviluppo; 24 domande distinte per voce, ciascuna presentata in due ordini.\n\n"
    simple = [r for r in baselines if r["split"] == "development" and r["condition"] == "simple"]
    ties = sum(r["correct_margin"] == 0 for r in simple)
    half_credit = np.mean(
        [(r["correct_margin"] > 0) + 0.5 * (r["correct_margin"] == 0) for r in simple]
    )
    text += f"**Pareggi numerici:** {ties}/{len(simple)} sulle frasi semplici. Il criterio congelato li conta come errori. Assegnando invece mezzo punto ai pareggi, l'accuratezza descrittiva complessiva è {half_credit:.1%}: il 25% rigoroso non va letto come prova di prestazioni sotto il caso. Questa analisi aggiuntiva non cambia il gate.\n\n"
    text += "## Condizioni misurate\n\n| Split | Condizione | Voce | Accuratezza | Entrambi gli ordini | Margine corretto, nats |\n|---|---|---|---:|---:|---:|\n"
    for split in ["development", "test"] if passed else ["development"]:
        for condition in ["simple", "no_context", "native", "gap", "clean"]:
            for voice in ["active", "passive"]:
                a = select(data, split, condition, voice, "accuracy")["mean"]
                b = select(data, split, condition, voice, "both_orders")["mean"]
                m = select(data, split, condition, voice, "margin")
                text += f"| {split} | {condition} | {voice} | {a:.1%} | {b:.1%} | {fmt(m)} |\n"
    text += "\n`simple`: solo frase principale; `no_context`: frase assente; `native`: subordinata senza virgola; `gap`: stessa frase con un salto negli ID posizionali; `clean`: virgola. Gli intervalli sono bootstrap descrittivi per famiglia (10.000 campioni), non repliche indipendenti per ordine/domanda.\n\n"
    if passed:
        text += "## Interventi sulle famiglie di test\n\nPatch del residuo dopo il blocco 17, sull'ultimo token del secondo nome, prima di `was`; solo frasi passive. Posizioni allineate, entrambe le direzioni con segno coerente.\n\n| Endpoint | Patch / confronto | Effetto medio [IC95%], nats |\n|---|---|---:|\n"
        for row in data["patches"]:
            if row["split"] == "test" and row["direction"] == "average":
                text += f"| {row['metric']} | {row['kind']} | {fmt(row)} |\n"
        text += "\n`qa_effect` misura il cambiamento nel margine risposta corretta/errata; `next_effect` quello di log P(was) − log P(virgola), nello stesso forward pass. Gli effetti sono differenze grezze del segnale della virgola: manca qui il controllo intransitivo fattoriale di V3. Un effetto QA non dimostra una rappresentazione astratta dei ruoli; un effetto nullo può dipendere dalla ricostruzione o ridondanza dell'informazione.\n\n"
    text += "## Cosa cambia rispetto a V3\n\n"
    text += (
        "Questa misura aggiunge un endpoint comportamentale sullo stesso stato manipolato. L'interpretazione va basata sul confronto con i controlli e sull'incertezza delle sei famiglie di test.\n\n"
        if passed
        else "Il risultato circoscrive **lo strumento di misura**, non prova che il modello ignori i ruoli. Il test causale programmato resta ineseguito. Prima di riprenderlo occorre validare un altro readout o un modello più capace, usando lo sviluppo e conservando il test per la conferma. Non abbassiamo la soglia dopo avere visto il risultato.\n\n"
    )
    text += "Gli effetti predittivi già misurati in V3 restano validi nel loro dominio. Questo seguito non autorizza a trasformarli in una dimostrazione di rianalisi sintattica. Le nuove frasi sono sintetiche, senza validazione umana; il contesto few-shot cambia rispetto ai prefissi nudi di V3. I predicati principali possono ricorrere nei due split. Cambiare voce e ordine esclude scorciatoie basate su un nome fisso, ma non regole superficiali sulla sequenza.\n\n"
    text += "## Riproducibilità\n\n"
    text += f"Modello `{meta['model']}`, revisione `{meta['revision']}`, BF16, attenzione eager, {meta['gpu']}. Funzione GPU: {meta['elapsed_seconds']:.1f} secondi. Baseline: {checks['baselines']}; interventi: {checks['interventions']}. Stato scientifico: `{data['status']}`. `complete=true` indica il completamento del ramo previsto dal protocollo.\n\n"
    text += f"Audit superato: {checks['source_files']} sorgenti verificati, punteggi grezzi e aggregazioni ricalcolati indipendentemente, controllo della sequenza sviluppo → gate → eventuale test. Errore massimo identità: {checks['maximum_identity_error']}.\n\n"
    text += "File nella stessa cartella: `metadata.json`, `source.json`, `materials.json`, `observations.jsonl`, `results.json`, `audit.json`, `report.svg` e `report.png`. Protocollo: `../../PROTOCOL.md`.\n"
    return text


def plot(data, directory, diagnostic=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = ["simple", "no_context", "native", "gap", "clean"]
    fig, grid = plt.subplots(
        2 if diagnostic else 1, 2, figsize=(12, 8 if diagnostic else 4.5), constrained_layout=True
    )
    axes = np.array(grid).reshape(-1)

    for voice, color in [("active", "#1e695c"), ("passive", "#ad4c31")]:
        for axis, metric in zip(axes[:2], ["accuracy", "both_orders"], strict=True):
            ys = [select(data, "development", c, voice, metric)["mean"] * 100 for c in conditions]
            axis.plot(range(5), ys, "o-", color=color, label=voice)
            axis.set(xticks=range(5), xticklabels=conditions, ylim=(-3, 103), ylabel="Percentuale")
            axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Accuratezza stretta (pareggi = errori)")
    axes[1].set_title("Corrette in entrambi gli ordini")
    axes[1].axhline(75, ls=":", color="gray", label="Soglia gate (solo simple)")
    if diagnostic:
        for axis, metric, title in zip(
            axes[2:],
            ["accuracy", "both_orders"],
            ["Frasi semplici: accuratezza stretta", "Frasi semplici: entrambi gli ordini"],
            strict=True,
        ):
            for index, (source, label, color) in enumerate(
                [(data, "BF16", "#888888"), (diagnostic, "FP32 (post-hoc)", "#1e695c")]
            ):
                values = [
                    select(source, "development", "simple", voice, metric)["mean"] * 100
                    for voice in ["active", "passive"]
                ]
                axis.bar(
                    np.arange(2) + (index - 0.5) * 0.3, values, width=0.3, color=color, label=label
                )
            axis.set(
                title=title,
                xticks=range(2),
                xticklabels=["Attive", "Passive"],
                ylim=(0, 103),
                ylabel="Percentuale",
            )
            axis.spines[["top", "right"]].set_visible(False)
            if metric == "both_orders":
                axis.axhline(75, ls=":", color="gray", label="Soglia gate")
    for axis in axes:
        axis.legend(fontsize=8)
    fig.suptitle("Gemma 3 1B · domande sui ruoli · sei famiglie di sviluppo")
    fig.savefig(directory / "report.svg")
    fig.savefig(directory / "report.png", dpi=180)
    plt.close(fig)


def audit_precision(parent, directory):
    meta = json.loads((directory / "metadata.json").read_text())
    data = json.loads((directory / "results.json").read_text())
    snapshot = json.loads((directory / "source.json").read_text())
    original_bytes = (parent / "observations.jsonl").read_bytes()
    if (
        not meta["complete"]
        or hashlib.sha256(original_bytes).hexdigest() != meta["parent_observations_sha256"]
    ):
        raise ValueError("Precision provenance mismatch")
    for name, digest in meta["source_sha256"].items():
        if hashlib.sha256(snapshot[name].encode()).hexdigest() != digest:
            raise ValueError("Precision source mismatch")
    originals = [
        r
        for r in map(json.loads, original_bytes.splitlines())
        if r.get("record") == "baseline"
        and r["split"] == "development"
        and r["condition"] in ["simple", "no_context"]
    ]
    rows = [
        json.loads(line) for line in (directory / "observations.jsonl").read_text().splitlines()
    ]
    if (
        len(rows) != 192
        or data["baseline_count"] != 192
        or data["test_inference"]
        or data["interventions"]
    ):
        raise ValueError("Precision scope violated")
    for original, row in zip(originals, rows, strict=True):
        for field in [
            "prompt",
            "id",
            "family",
            "split",
            "condition",
            "voice",
            "reversal",
            "role",
            "order",
            "gold_index",
        ]:
            if original[field] != row[field]:
                raise ValueError("Precision prompt or label changed")
        if original["scores"]["input_ids"] != row["scores"]["input_ids"]:
            raise ValueError("Precision tokenization changed")
        logs = row["scores"]["answer_logp"]
        gold = row["gold_index"]
        if (
            abs(logs[gold] - logs[1 - gold] - row["correct_margin"]) > 1e-9
            or abs(sum(np.exp(logs)) - row["answer_mass"]) > 1e-6
        ):
            raise ValueError("Precision score mismatch")
    computed = aggregate_behavior(rows)
    for row in data["behavioral"]:
        check = interval(computed[tuple(row[k] for k in ["split", "condition", "voice", "metric"])])
        if not np.allclose(
            [row["mean"], *row["ci95"]], [check["mean"], *check["ci95"]], atol=1e-10, rtol=0
        ):
            raise ValueError("Precision aggregation mismatch")
    rates = {
        v: np.mean(list(computed["development", "simple", v, "both_orders"].values()))
        for v in ["active", "passive"]
    }
    simple = [r for r in rows if r["condition"] == "simple"]
    mass = float(np.median([r["answer_mass"] for r in simple]))
    if (
        data["gate"]["passed"] != (min(rates.values()) >= 0.75 and mass >= 0.1)
        or rates != data["gate"]["both_orders_correct_by_voice"]
    ):
        raise ValueError("Precision gate mismatch")
    diagnostic = dict(
        passed=True,
        gate=data["gate"],
        simple_ties=sum(r["correct_margin"] == 0 for r in simple),
        elapsed_seconds=meta["elapsed_seconds"],
        development_only=True,
        baselines=len(rows),
        source_files=len(snapshot),
    )
    (directory / "audit.json").write_text(json.dumps(diagnostic, indent=2))
    return dict(diagnostic, behavioral=data["behavioral"])


def precision_text(diagnostic):
    text = "## Verifica numerica aggiuntiva: FP32\n\nDopo aver visto i pareggi BF16, abbiamo ripetuto **solo** le 96 frasi semplici e le 96 misure senza contesto di sviluppo, con pesi e calcolo FP32 e TF32 disabilitato. Identici prompt, checkpoint, etichette e ordine. È una diagnosi post-hoc, separata dall'esperimento congelato.\n\n"
    text += "| Frasi semplici FP32 | Accuratezza | Entrambi gli ordini |\n|---|---:|---:|\n"
    for voice in ["active", "passive"]:
        a = select(diagnostic, "development", "simple", voice, "accuracy")["mean"]
        b = select(diagnostic, "development", "simple", voice, "both_orders")["mean"]
        text += f"| {voice} | {a:.1%} | {b:.1%} |\n"
    text += f"\nPareggi FP32 sulle frasi semplici: {diagnostic['simple_ties']}/96. Il gate resta **{'superato' if diagnostic['gate']['passed'] else 'fallito'}**. La precisione modifica i punteggi e spiega molti pareggi, ma questo controllo non sostiene una comprensione stabile nel formato scelto. Nessuna patch e nessun dato di test in questa diagnosi. Funzione GPU: {diagnostic['elapsed_seconds']:.1f} secondi.\n\n"
    return text


def main(directory):
    import markdown

    meta, data, checks, baselines = audit(directory)
    report = render(meta, data, checks, baselines)
    diagnostic = None
    precision_dir = directory.parent / "bridge-precision-20261008-01"
    if directory.name == "bridge-v3-20261008-02" and precision_dir.exists():
        diagnostic = audit_precision(directory, precision_dir)
        report = report.replace(
            "## Cosa cambia rispetto a V3",
            precision_text(diagnostic) + "## Cosa cambia rispetto a V3",
        )
        checks["precision_diagnostic_audited"] = True
    (directory / "audit.json").write_text(json.dumps(checks, indent=2))
    (directory / "report.md").write_text(report)
    plot(data, directory, diagnostic)
    css = (Path(__file__).resolve().parents[2] / "v2/html_assets/style.css").read_text()
    body = markdown.markdown(report, extensions=["tables"])
    body = body.replace("<table>", '<div class="table-scroll" tabindex="0"><table>').replace(
        "</table>", "</table></div>"
    )
    encoded = base64.b64encode((directory / "report.svg").read_bytes()).decode()
    html = (
        '<!doctype html><html lang="it"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V3 — Domande sui ruoli</title><style>'
        + css
        + 'main{padding-top:40px!important;padding-bottom:50px!important}h1{font:42px Georgia}h2{margin-top:40px}</style><body><main class="wrap">'
    )
    chart = (
        '<figure class="chart-card"><div class="chart"><img alt="Accuratezza delle domande sui ruoli e robustezza allo scambio delle opzioni" src="data:image/svg+xml;base64,'
        + encoded
        + '"></div><figcaption>Misure di sviluppo; la soglia si applica alle frasi semplici.</figcaption></figure>'
    )
    (directory / "report.html").write_text(
        html + body.replace("<h2>", chart + "<h2>", 1) + "</main></body></html>"
    )
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory)
