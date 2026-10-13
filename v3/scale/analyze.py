"""Compare the fixed larger-model validation series with the saved 1B baseline."""

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v3.direct.analyze import audit as audit_run
from v3.scale.run import MODELS, next_size


def check_selection(sizes: list[str], outcomes: list[bool]) -> str | None:
    previous = []
    for size, passed in zip(sizes, outcomes, strict=True):
        if size != next_size(previous):
            raise ValueError("Selection order violated")
        previous.append(passed)
    return next((size for size, passed in zip(sizes, outcomes, strict=True) if passed), None)


def build(paths: list[Path], output: Path) -> None:
    import markdown
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(__file__).resolve().parents[2]
    baseline = root / "v3/direct/results/direct-v3-20261008-01"
    original_material = json.loads((baseline / "materials.json").read_text())
    models, summaries, rows_by_model, sizes, outcomes = [], [], [], [], []
    for directory in [baseline, *paths]:
        meta, result, rows, audited = audit_run(directory)
        if json.loads((directory / "materials.json").read_text()) != original_material:
            raise ValueError("Prompt or label changed across models")
        size = meta["model"].split("-")[-2]
        if directory != baseline:
            if MODELS[size] != (meta["model"], meta["revision"]):
                raise ValueError("Unplanned checkpoint")
            if any(meta["loading_info"].values()):
                raise ValueError("Checkpoint loading errors")
            sizes.append(size)
            outcomes.append(result["gate"]["passed"])
        models.append(
            dict(size=size, run_id=directory.name, metadata=meta, result=result, audit=audited)
        )
        summaries.append([r for r in audited["summary"] if r["condition"] == "simple"])
        rows_by_model.append(rows)
    selected = check_selection(sizes, outcomes)
    output.mkdir(parents=True, exist_ok=True)
    comparison = dict(
        selected=selected,
        development_only=True,
        identical_materials=True,
        selection_sequence_checked=True,
        models=models,
    )
    # Keep the failed metadata-save attempt visible, without counting it as another checkpoint.
    failed = output.parent / "scale-4b-20261009-01"
    if failed.exists() and any(m["size"] == "4b" for m in models):
        repeat = paths[sizes.index("4b")]
        first = (failed / "observations.jsonl").read_bytes()
        second = (repeat / "observations.jsonl").read_bytes()
        comparison["metadata_failure_attempt"] = dict(
            run_id=failed.name,
            metadata_complete=json.loads((failed / "metadata.json").read_text())["complete"],
            raw_observations_identical=first == second,
            raw_sha256=hashlib.sha256(first).hexdigest(),
        )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    width = 0.8 / len(models)
    colors = ["#aaaaaa", "#1e695c", "#ad4c31"]
    for index, (model, summary) in enumerate(zip(models, summaries, strict=True)):
        for ax, metric in zip(axes, ["accuracy", "both_reversals"], strict=True):
            values = [r[metric]["mean"] * 100 for r in summary]
            ax.bar(
                np.arange(4) + (index - (len(models) - 1) / 2) * width,
                values,
                width=width,
                label=model["size"].upper(),
                color=colors[index],
            )
    for ax, title in zip(
        axes, ["Accuratezza per frase", "Corrette entrambe le inversioni"], strict=True
    ):
        ax.set(
            title=title,
            xticks=range(4),
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
        ax.legend()
    axes[1].axhline(75, color="#333333", ls=":")
    fig.suptitle("Gemma 3 pretrained · FP32 · stessi prompt · sei famiglie di sviluppo")
    fig.savefig(output / "report.svg")
    fig.savefig(output / "report.png", dpi=180)
    plt.close(fig)
    text = "# V3 · Il controllo di comprensione su modelli più grandi\n\n"
    text += (
        f"**Gemma 3 {selected.upper()} supera il criterio prefissato.** È il primo checkpoint della sequenza pianificata a riuscirci; la selezione si ferma qui.\n\n"
        if selected
        else "**Nessun checkpoint completato supera il criterio prefissato.** I risultati vanno valutati per ciascun ruolo e voce, non solo sulla media complessiva.\n\n"
    )
    text += "## Risultati sulle stesse frasi semplici\n\n| Modello | Attiva: agente | Attiva: paziente | Passiva: agente | Passiva: paziente | Gate |\n|---|---:|---:|---:|---:|---|\n"
    for model, summary in zip(models, summaries, strict=True):
        text += (
            "| "
            + model["size"].upper()
            + " | "
            + " | ".join(f"{r['both_reversals']['mean']:.1%}" for r in summary)
            + " | "
            + ("passato" if model["result"]["gate"]["passed"] else "fallito")
            + " |\n"
        )
    text += "\nLa tabella misura **entrambe le inversioni corrette**, con sei coppie per cella. Soglia congelata: almeno 75% in tutte e quattro le celle (quindi almeno 5/6 coppie). Le percentuali per singola frase sono nel grafico e nei dati.\n\n"
    text += "## Disegno e controlli\n\nAbbiamo riusato esattamente i 96 prompt, le etichette, le quattro dimostrazioni e il punteggio del precedente test diretto sui nomi: nessuna opzione A/B, nessuna ricerca di prompt, nessuna normalizzazione per lunghezza. Sequenza fissata prima della misura: 4B, poi 12B soltanto se 4B fallisce; arresto al primo successo. Il precedente 1B è riportato come riferimento, non è stato rieseguito. Tutto resta sullo sviluppo.\n\n"
    text += "Le risposte complete sono valutate mediante la somma dei log-punteggi dei token dopo `Answer: the`. La verifica senza frase usa prompt identici e risposte corrette opposte quando scambiamo i partecipanti; controlliamo punteggi identici e margini opposti. La scelta tra due candidati non equivale a generazione libera.\n\n"
    text += "| Modello | Accuratezza complessiva sulle frasi semplici | Token per candidato | Coppie con lunghezze diverse | Mediana massa dei due candidati |\n|---|---:|---|---:|---:|\n"
    for model, rows in zip(models, rows_by_model, strict=True):
        simple = [r for r in rows if r["condition"] == "simple"]
        accuracy = np.mean([r["margin"] > 0 for r in simple])
        counts = sorted({len(c["token_ids"]) for r in rows for c in r["scores"]["candidates"]})
        unequal = sum(
            len(r["scores"]["candidates"][0]["token_ids"])
            != len(r["scores"]["candidates"][1]["token_ids"])
            for r in rows
        )
        mass = np.median(
            [sum(np.exp(c["logp"]) for c in r["scores"]["candidates"]) for r in simple]
        )
        text += (
            f"| {model['size'].upper()} | {accuracy:.1%} | {counts} | {unequal}/96 | {mass:.1%} |\n"
        )
    text += "\nLa massa è condizionata al prefisso comune, non è una probabilità di risposta libera. I dettagli per famiglia e gli intervalli bootstrap descrittivi (10.000 ricampionamenti) sono in `comparison.json`; le inversioni non sono osservazioni indipendenti.\n\n"
    text += "## Interpretazione e limite del passo successivo\n\n"
    text += (
        "Il controllo identifica un candidato utilizzabile per progettare il ponte causale. Non abbiamo ancora misurato se la virgola cambia la comprensione o se una patch prima della disambiguazione modifica le risposte. Il prossimo protocollo deve includere entrambe le direzioni, controlli lessicali/casuali e posizioni allineate, congelati prima di interrogare il test.\n\n"
        if selected
        else "La procedura non ha validato un modello per il ponte causale. Aumentare i parametri non garantisce che il readout sia affidabile in tutti i ruoli. Non abbassiamo la soglia e non selezioniamo le sole celle facili. Questa fase non dimostra assenza di rappresentazioni sintattiche.\n\n"
    )
    text += "Sono dati di sviluppo riutilizzati dopo gli esperimenti precedenti: la selezione non è conferma indipendente. I checkpoint differiscono anche per architettura e addestramento; non attribuiamo causalmente le differenze al solo numero di parametri. Le frasi sintetiche non hanno validazione umana. Nessun risultato qui prova rianalisi sintattica o identifica un circuito dei ruoli. I risultati predittivi di V3 rimangono distinti.\n\n"
    text += "## Esecuzione e audit\n\n| Modello | Run | GPU | Secondi nella funzione | Picco memoria GPU |\n|---|---|---|---:|---:|\n"
    for model in models:
        meta = model["metadata"]
        peak = meta.get("peak_gpu_bytes")
        text += f"| {model['size'].upper()} | `{model['run_id']}` | {meta['gpu']} | {meta['elapsed_seconds']:.1f} | {f'{peak / 2**30:.1f} GiB' if peak else 'non registrato'} |\n"
    text += "\nDurate comprensive del caricamento, escluso provisioning; non sono una fattura. FP32, TF32 disabilitato, attenzione eager. Modelli grandi caricati con il wrapper multimodale completo, senza input immagine; nessun peso mancante o inatteso. Sorgenti e revisioni sono salvati nei metadati. Audit indipendente di punteggi, gate, ordine di selezione e identità dei materiali superato.\n\n"
    if "metadata_failure_attempt" in comparison:
        same = comparison["metadata_failure_attempt"]["raw_observations_identical"]
        text += f"Il tentativo `scale-4b-20261009-01` aveva completato le misure ma fallito il salvataggio JSON dei metadati: rimane archiviato come incompleto. Il problema è stato corretto e il 4B rieseguito senza cambiare il protocollo. Punteggi grezzi identici tra i due tentativi: **{same}**. Il tentativo incompleto non è una replica scientifica aggiuntiva.\n\n"
    text += "Protocollo: [PROTOCOL.md](../../PROTOCOL.md). Riferimenti: [Gemma 3 4B pretrained](https://huggingface.co/google/gemma-3-4b-pt), [Gemma 3 12B pretrained](https://huggingface.co/google/gemma-3-12b-pt), [documentazione Transformers](https://huggingface.co/docs/transformers/model_doc/gemma3).\n"
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2, allow_nan=False))
    (output / "report.md").write_text(text)
    css = (root / "v2/html_assets/style.css").read_text()
    body = markdown.markdown(text, extensions=["tables"])
    body = body.replace("<table>", '<div class="table-scroll" tabindex="0"><table>').replace(
        "</table>", "</table></div>"
    )
    encoded = base64.b64encode((output / "report.svg").read_bytes()).decode()
    chart = (
        '<figure class="chart-card"><div class="chart"><img alt="Confronto dei checkpoint sulle stesse domande di sviluppo" src="data:image/svg+xml;base64,'
        + encoded
        + '"></div></figure>'
    )
    html = (
        '<!doctype html><html lang="it"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V3 — Modelli più grandi</title><style>'
        + css
        + 'main{padding-top:40px!important;padding-bottom:50px!important}h1{font:42px Georgia}h2{margin-top:40px}</style><body><main class="wrap">'
    )
    (output / "report.html").write_text(
        html + body.replace("<h2>", chart + "<h2>", 1) + "</main></body></html>"
    )
    print(json.dumps(dict(selected=selected, audited=[m["run_id"] for m in models]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build(args.runs, args.output)
