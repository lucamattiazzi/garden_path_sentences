"""Audit both endpoints and build the 12B causal-bridge report without inference."""

import argparse
import base64
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v3.bridge.analyze import aggregate_patches, interval
from v3.direct.analyze import summarize


def validate_score(score: dict, gold: int) -> float:
    for logp, details in zip(score["answer_logp"], score["answer_tokens"], strict=True):
        if abs(sum(details["token_logp"]) - logp) > 1e-9:
            raise ValueError("Answer token sum mismatch")
    if (
        score["next_logp"] is not None
        and abs(score["next_logp"][0] - score["next_logp"][1] - score["next_margin"]) > 1e-9
    ):
        raise ValueError("Next margin mismatch")
    return score["answer_logp"][gold] - score["answer_logp"][1 - gold]


def audit(directory: Path):
    meta, result, source = [
        json.loads((directory / name).read_text())
        for name in ["metadata.json", "results.json", "source.json"]
    ]
    if not meta["complete"] or meta["fixed_layer"] != 32 or meta["layers"] != 48:
        raise ValueError("Incomplete or wrong layer run")
    for name, digest in meta["source_sha256"].items():
        if hashlib.sha256(source[name].encode()).hexdigest() != digest:
            raise ValueError("Source mismatch")
    raw = [json.loads(line) for line in (directory / "observations.jsonl").read_text().splitlines()]
    if raw[0] != dict(record="frozen_protocol", layer=32, gate_before_test=True, voice="passive"):
        raise ValueError("Missing initial declaration")
    bases = [r for r in raw if r["record"] == "baseline"]
    patches = [r for r in raw if r["record"] == "intervention"]
    gates = [i for i, r in enumerate(raw) if r["record"] == "development_gate"]
    if len(gates) != 1:
        raise ValueError("Expected exactly one gate decision")
    gate = {k: v for k, v in raw[gates[0]].items() if k != "record"}
    if gate != result["gate"]:
        raise ValueError("Gate artifact mismatch")
    for i, r in enumerate(raw):
        if (r.get("split") == "test" or r["record"] == "intervention") and (
            not gate["passed"] or i < gates[0]
        ):
            raise ValueError("Test or patch before competence gate")
    expected = (192, 576) if gate["passed"] else (96, 0)
    if (len(bases), len(patches)) != expected or (
        result["baselines"],
        result["interventions"],
    ) != expected:
        raise ValueError("Wrong observation counts")
    lookup = {}
    for row in bases:
        key = tuple(row[k] for k in ["id", "reversal", "role", "condition"])
        if key in lookup:
            raise ValueError("Duplicate baseline")
        lookup[key] = row
        if abs(validate_score(row["scores"], row["prompt"]["gold_index"]) - row["margin"]) > 1e-9:
            raise ValueError("Baseline margin mismatch")
    behavior = {
        split: summarize([dict(r, voice="passive") for r in bases if r["split"] == split])
        for split in sorted({r["split"] for r in bases})
    }
    rates = {
        r["condition"] + "/" + r["role"]: r["both_reversals"]["mean"]
        for r in behavior["development"]
        if r["condition"] in ["simple", "clean"]
    }
    if rates != gate["both_reversal_rates"] or gate["passed"] != (
        len(rates) == 4 and min(rates.values()) >= 0.75
    ):
        raise ValueError("Independent gate failed")
    cells = defaultdict(set)
    identity, norm_errors = [], []
    for row in patches:
        key = tuple(row[k] for k in ["id", "reversal", "role", "condition"])
        base = lookup[key]
        margin = validate_score(row["scores"], base["prompt"]["gold_index"])
        if (
            abs(margin - row["margin"]) > 1e-9
            or abs(margin - base["margin"] - row["qa_effect"]) > 1e-9
            or abs(
                row["scores"]["next_margin"] - base["scores"]["next_margin"] - row["next_effect"]
            )
            > 1e-9
        ):
            raise ValueError("Raw intervention delta mismatch")
        label = (row["kind"], row["seed"])
        if label in cells[key]:
            raise ValueError("Duplicate intervention")
        cells[key].add(label)
        if row["kind"] == "identity":
            identity.append(max(abs(row["qa_effect"]), abs(row["next_effect"])))
        if row.get("requested_l2", 0) > 0:
            norm_errors.append(abs(row["actual_l2"] / row["requested_l2"] - 1))
    required = {
        ("cross", None),
        ("identity", None),
        ("lexical", None),
        *{("random", i) for i in range(3)},
    }
    if any(labels != required for labels in cells.values()):
        raise ValueError("Incomplete intervention controls")
    if identity and max(identity) > 1e-4:
        raise ValueError("Nonzero identity effect")
    families = aggregate_patches([dict(r, order=0) for r in patches])
    effects = [
        dict(
            split=k[0],
            direction=k[1],
            kind=k[2],
            metric=k[3],
            family_values=values,
            **interval(values),
        )
        for k, values in sorted(families.items())
    ]
    role_effects = []
    for role in ["agent", "patient"]:
        grouped = aggregate_patches(
            [dict(r, order=0) for r in patches if r["split"] == "test" and r["role"] == role]
        )
        for key, values in sorted(grouped.items()):
            if key[1] == "average" and key[3] == "qa_effect":
                role_effects.append(
                    dict(role=role, kind=key[2], family_values=values, **interval(values))
                )
    cross_test = [r for r in patches if r["split"] == "test" and r["kind"] == "cross"]
    choice_flips = sum(
        (r["margin"] > 0)
        != (lookup[tuple(r[k] for k in ["id", "reversal", "role", "condition"])]["margin"] > 0)
        for r in cross_test
    )
    checks = dict(
        passed=True,
        source_files=len(source),
        baselines=len(bases),
        interventions=len(patches),
        gate_chronology=True,
        independent_gate=True,
        raw_effects_checked=True,
        maximum_identity_error=max(identity, default=None),
        maximum_relative_norm_error=max(norm_errors, default=None),
    )
    return (
        meta,
        result,
        bases,
        dict(
            checks=checks,
            behavior=behavior,
            effects=effects,
            exploratory_role_effects=role_effects,
            cross_test_observations=len(cross_test),
            cross_test_choice_flips=choice_flips,
        ),
    )


def format_interval(row):
    return f"{row['mean']:.3f} [{row['ci95'][0]:.3f}, {row['ci95'][1]:.3f}]"


def build(directory: Path):
    import markdown
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta, result, bases, data = audit(directory)
    passed = result["gate"]["passed"]
    split = "test" if passed else "development"

    def effect(kind, metric, direction="average"):
        return next(
            r
            for r in data["effects"]
            if (r["split"], r["kind"], r["metric"], r["direction"])
            == (split, kind, metric, direction)
        )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    conditions = ["simple", "native", "gap", "clean"]
    for role, color in [("agent", "#1e695c"), ("patient", "#ad4c31")]:
        rows = [
            next(r for r in data["behavior"][split] if r["condition"] == c and r["role"] == role)
            for c in conditions
        ]
        axes[0, 0].plot(
            range(4), [r["accuracy"]["mean"] * 100 for r in rows], "o-", color=color, label=role
        )
        axes[0, 1].plot(
            range(4), [r["margin"]["mean"] for r in rows], "o-", color=color, label=role
        )
    axes[0, 0].set(title=f"Accuratezza prima delle patch · {split}", ylabel="%", ylim=(0, 105))
    axes[0, 1].set(title="Margine risposta corretta − errata", ylabel="Nats")
    for ax in axes[0]:
        ax.set(xticks=range(4), xticklabels=conditions)
        ax.legend(fontsize=8)
    for ax, metric, title in zip(
        axes[1],
        ["next_effect", "qa_effect"],
        ["Effetto sulla previsione was / virgola", "Effetto sulla risposta finale"],
        strict=True,
    ):
        if passed:
            kinds = ["cross", "lexical", "random", "cross_minus_lexical", "cross_minus_random"]
            rows = [effect(k, metric) for k in kinds]
            means = np.array([r["mean"] for r in rows])
            ci = np.array([r["ci95"] for r in rows])
            ax.errorbar(
                means,
                range(5),
                xerr=np.maximum(0, np.array([means - ci[:, 0], ci[:, 1] - means])),
                fmt="o",
                capsize=4,
                color="#1e695c",
            )
            ax.set(
                yticks=range(5),
                yticklabels=[
                    "Cross",
                    "Lessicale",
                    "Casuale",
                    "Cross − lessicale",
                    "Cross − casuale",
                ],
            )
            ax.axvline(0, color="gray", ls=":")
        else:
            ax.text(
                0.5, 0.5, "Gate fallito: patch non eseguite", ha="center", transform=ax.transAxes
            )
        ax.set(title=title, xlabel="Nats, IC95% per famiglia")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Gemma 3 12B · FP32 · blocco 32 fisso · ponte tra previsione e risposta")
    fig.savefig(directory / "report.svg")
    fig.savefig(directory / "report.png", dpi=180)
    plt.close(fig)
    if passed:
        role_rows = [
            next(
                r
                for r in data["exploratory_role_effects"]
                if r["role"] == role and r["kind"] == "cross"
            )
            for role in ["agent", "patient"]
        ]
        means = np.array([r["mean"] for r in role_rows])
        cis = np.array([r["ci95"] for r in role_rows])
        role_fig, role_ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
        role_ax.errorbar(
            means,
            [1, 0],
            xerr=np.maximum(0, np.array([means - cis[:, 0], cis[:, 1] - means])),
            fmt="o",
            capsize=4,
            color="#1e695c",
        )
        role_ax.axvline(0, color="gray", ls=":")
        role_ax.set(
            yticks=[1, 0],
            yticklabels=["Agente", "Paziente"],
            xlabel="Effetto sul margine corretto, nats [IC95%]",
            title="Scomposizione esplorativa per ruolo · test · patch cross",
            ylim=(-0.5, 1.5),
        )
        role_ax.spines[["top", "right"]].set_visible(False)
        role_fig.savefig(directory / "report-roles.svg")
        role_fig.savefig(directory / "report-roles.png", dpi=180)
        plt.close(role_fig)
    text = "# V3 · Il ponte causale su Gemma 3 12B\n\n"
    text += (
        "**Esperimento completato sul test tenuto da parte.** Il modello ha superato il controllo di comprensione sia sulle frasi semplici sia su quelle con virgola dello sviluppo.\n\n"
        if passed
        else "**Il nuovo controllo contestuale è fallito.** Nessuna patch e nessuna inferenza sul test.\n\n"
    )
    text += "Il 12B era stato selezionato dopo il fallimento di 1B e 4B sulle stesse domande semplici. Qui fissiamo un secondo protocollo e chiediamo se modificare uno stato prima di `was` cambi sia la previsione immediata sia la risposta successiva su agente/paziente.\n\n"
    if passed:
        text += f"Sul test, l'effetto medio cross sulla previsione è **{format_interval(effect('cross', 'next_effect'))} nats**; sulla risposta finale è **{format_interval(effect('cross', 'qa_effect'))} nats**. Quest'ultimo non è un effetto nullo: aggrega spostamenti opposti per agente e paziente, senza cambiare il candidato scelto nei {data['cross_test_observations']} interventi cross.\n\n"
    text += "## Comportamento prima delle patch\n\n| Split | Condizione | Ruolo | Accuratezza | Entrambe le inversioni | Margine corretto [IC95%], nats |\n|---|---|---|---:|---:|---:|\n"
    for domain, rows in data["behavior"].items():
        for row in rows:
            text += f"| {domain} | {row['condition']} | {row['role']} | {row['accuracy']['mean']:.1%} | {row['both_reversals']['mean']:.1%} | {format_interval(row['margin'])} |\n"
    text += "\n`simple`: sola principale; `native`: subordinata senza virgola; `gap`: stessi token senza virgola, con ID posizionali allineati; `clean`: virgola prima del secondo nome. Frasi passive, due inversioni dei partecipanti, due domande per frase. La misura classifica due risposte complete, non generazioni libere.\n\n"
    text += "## Preferenza prima della disambiguazione\n\n| Split | Condizione | Log P(was) − log P(virgola), media [IC95%] |\n|---|---|---:|\n"
    for domain in data["behavior"]:
        for condition in ["native", "gap", "clean"]:
            values = defaultdict(list)
            for row in bases:
                if row["split"] == domain and row["condition"] == condition:
                    values[row["family"]].append(row["scores"]["next_margin"])
            estimate = interval(
                {key: float(np.mean(value)) for key, value in sorted(values.items())}
            )
            text += f"| {domain} | {condition} | {format_interval(estimate)} |\n"
    text += "\nQuesta preferenza locale precede il verbo; l'accuratezza QA precedente riguarda invece la frase completata. Non sono due misure intercambiabili di comprensione.\n\n"
    if passed:
        text += "## Interventi sulle sei famiglie di test\n\n| Endpoint | Intervento / confronto | Media [IC95%], nats |\n|---|---|---:|\n"
        for metric in ["next_effect", "qa_effect"]:
            for kind in [
                "cross",
                "identity",
                "lexical",
                "random",
                "cross_minus_lexical",
                "cross_minus_random",
            ]:
                text += f"| {metric} | {kind} | {format_interval(effect(kind, metric))} |\n"
        text += "\nEffetti orientati come clean − gap: la direzione inversa è negata. Positivo indica un aumento dell’endpoint nel trasferimento verso lo stato con virgola; non implica automaticamente un miglioramento della comprensione. `next_effect` è log P(was) − log P(virgola); `qa_effect` è il margine risposta corretta − errata. Ogni effetto deriva dagli stessi forward pass.\n\n"
        text += "### Le due direzioni, senza nascondere asimmetrie\n\n| Direzione (segno allineato) | Previsione: cross | Risposta: cross |\n|---|---:|---:|\n"
        for direction in ["clean_to_gap", "gap_to_clean"]:
            text += f"| {direction} | {format_interval(effect('cross', 'next_effect', direction))} | {format_interval(effect('cross', 'qa_effect', direction))} |\n"
        text += "\nI tre seed casuali sono mediati prima delle coppie; ruoli, inversioni e direzioni prima delle famiglie. Intervalli bootstrap descrittivi: 10.000 ricampionamenti di sei famiglie. Non trattiamo tutte le patch come repliche indipendenti. Valori per famiglia, singole direzioni e sviluppo sono in `audit.json`.\n\n"
    if passed:
        text += "## Diagnosi esplorativa: la media nasconde due effetti opposti\n\n| Ruolo | Effetto cross sulla risposta [IC95%] | Cross − lessicale | Cross − casuale |\n|---|---:|---:|---:|\n"
        for role in ["agent", "patient"]:
            rows = [
                next(
                    r
                    for r in data["exploratory_role_effects"]
                    if r["role"] == role and r["kind"] == kind
                )
                for kind in ["cross", "cross_minus_lexical", "cross_minus_random"]
            ]
            text += f"| {role} | " + " | ".join(format_interval(r) for r in rows) + " |\n"
        text += f"\nScelte finali cambiate: **{data['cross_test_choice_flips']}/{data['cross_test_observations']} interventi cross sul test**. Le probabilità relative cambiano anche quando il candidato preferito resta lo stesso. Questa scomposizione per ruolo è descrittiva e post-hoc; non sostituisce l'endpoint primario medio.\n\n"
        text += "Il calo del margine per l'agente e l'aumento per il paziente sono compatibili con uno spostamento verso il soggetto grammaticale della passiva (il paziente), comune alle due domande. I controlli non consentono di attribuirlo a un trasferimento selettivo del ruolo richiesto. Servono interventi che distinguano preferenza per un partecipante e informazione sul ruolo, prima di parlare di un circuito della comprensione.\n\n"
    text += "## Che cosa questo esperimento può stabilire\n\nLa patch sostituisce il residuo completo dopo il blocco 32, sull’ultimo token del secondo nome. Il blocco è stato fissato per profondità relativa rispetto al 17 del modello 1B, senza cercare il migliore sul 12B. Questa scelta non implica omologia tra i due modelli. I controlli lessicale e casuale hanno la stessa norma della perturbazione cross; il controllo identità deve riprodurre i punteggi originali.\n\n"
    text += "Un effetto sulla previsione non basta a dimostrare rianalisi o comprensione dei ruoli. Qui misuriamo anche uno spostamento dei margini QA, ma nessun cambio delle scelte finali; i controlli e gli effetti opposti per ruolo richiedono un’interpretazione più limitata di un trasferimento della comprensione. Un intervallo QA che include zero non prova assenza di un circuito: il layer può essere inadeguato, l’informazione ridondante o ricostruita dai token successivi.\n\n"
    text += "Questa è una differenza del segnale della virgola, **non** l’interazione opzionale-meno-intransitivo di V3: qui manca il controllo intransitivo fattoriale. Materiali sintetici, senza validazione umana; predicati principali possono ricorrere tra split. Le nuove misure non vanno confrontate in ampiezza direttamente con V3 1B, che usa altri materiali e contesti.\n\n"
    checks = data["checks"]
    text += f"## Riproducibilità\n\n{checks['baselines']} baseline, {checks['interventions']} interventi; modello `{meta['model']}`, revisione `{meta['revision']}`, FP32, attenzione eager, TF32 disabilitato. GPU: {meta['gpu']}; funzione: {meta['elapsed_seconds']:.1f} secondi; picco memoria: {meta['peak_gpu_bytes'] / 2**30:.1f} GiB. Tempi escluso provisioning.\n\n"
    text += f"Audit superato: {checks['source_files']} sorgenti, gate ricalcolato, ordine sviluppo → gate → test, somme token e differenze grezze ricostruite, completezza dei controlli. Errore massimo identità: {checks['maximum_identity_error']}; errore relativo massimo delle norme: {checks['maximum_relative_norm_error']}.\n\n"
    text += "[Protocollo congelato](../../PROTOCOL.md). Dati nella stessa cartella: `metadata.json`, `source.json`, `materials.json`, `observations.jsonl`, `results.json`, `audit.json`.\n"
    (directory / "audit.json").write_text(json.dumps(data, indent=2, allow_nan=False))
    (directory / "report.md").write_text(text)
    root = Path(__file__).resolve().parents[2]
    css = (root / "v2/html_assets/style.css").read_text()
    body = markdown.markdown(text, extensions=["tables"])
    body = body.replace("<table>", '<div class="table-scroll" tabindex="0"><table>').replace(
        "</table>", "</table></div>"
    )
    if passed:
        role_encoded = base64.b64encode((directory / "report-roles.svg").read_bytes()).decode()
        role_chart = (
            '<figure class="chart-card"><div class="chart"><img alt="Effetti QA opposti per agente e paziente, scomposizione esplorativa" src="data:image/svg+xml;base64,'
            + role_encoded
            + '"></div></figure>'
        )
        heading = "<h2>Diagnosi esplorativa: la media nasconde due effetti opposti</h2>"
        body = body.replace(heading, heading + role_chart)
    encoded = base64.b64encode((directory / "report.svg").read_bytes()).decode()
    chart = (
        '<figure class="chart-card"><div class="chart"><img alt="Comportamento iniziale e confronto dei due endpoint causali con i controlli" src="data:image/svg+xml;base64,'
        + encoded
        + '"></div></figure>'
    )
    html = (
        '<!doctype html><html lang="it"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V3 — Ponte causale 12B</title><style>'
        + css
        + 'main{padding-top:40px!important;padding-bottom:50px!important}h1{font:42px Georgia}h2{margin-top:40px}</style><body><main class="wrap">'
    )
    (directory / "report.html").write_text(
        html + body.replace("<h2>", chart + "<h2>", 1) + "</main></body></html>"
    )
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    build(parser.parse_args().directory)
