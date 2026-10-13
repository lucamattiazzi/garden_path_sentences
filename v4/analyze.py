"""Recompute V4 scores and contrasts from saved raw measurements; no inference."""

import argparse
import base64
import hashlib
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v3.bridge.analyze import interval
from v3.bridge12.analyze import audit as audit_legacy
from v3.bridge12.analyze import validate_score

CONSTRUCTIONS = ["active", "passive", "object_relative", "passive_relative"]
WEIGHTS = dict(
    first_mention=[1, 1, 1, 1],
    event_subject=[1, 1, -1, -1],
    semantic_agent=[1, -1, -1, 1],
    construction_interaction=[1, -1, 1, -1],
    role_selective=[1, -1, -1, 1],
)


def contrasts(rows):
    cells = defaultdict(list)
    for r in rows:
        sign = 1 if r["direction"] == "clean_to_gap" else -1
        cells[r["family"], r["construction"], r["role"], r["kind"]].append(sign * r["first_effect"])
    families = sorted({r["family"] for r in rows})
    estimates = defaultdict(dict)
    for family in families:
        for kind in ["cross", "lexical", "random", "identity"]:
            common, difference = [], []
            for construction in CONSTRUCTIONS:
                a, p = [
                    float(np.mean(cells[family, construction, role, kind]))
                    for role in ["agent", "patient"]
                ]
                common.append((a + p) / 2)
                difference.append((a - p) / 2)
            for name, signs in WEIGHTS.items():
                values = difference if name == "role_selective" else common
                estimates[kind, name][family] = float(np.mean(np.array(values) * signs))
        for control in ["lexical", "random"]:
            for name in WEIGHTS:
                estimates["cross_minus_" + control, name][family] = (
                    estimates["cross", name][family] - estimates[control, name][family]
                )
    return [
        dict(kind=k, contrast=c, family_values=v, **interval(v))
        for (k, c), v in sorted(estimates.items())
    ]


def audit(folder):
    meta, result, sources = [
        json.loads((folder / name).read_text())
        for name in ["metadata.json", "results.json", "source.json"]
    ]
    assert meta["complete"] and meta["fixed_layer"] == round(17 / 25 * (meta["layers"] - 1))
    assert all(
        hashlib.sha256(sources[n].encode()).hexdigest() == h
        for n, h in meta["source_sha256"].items()
    )
    rows = [json.loads(line) for line in (folder / "observations.jsonl").read_text().splitlines()]
    bases = [r for r in rows if r["record"] == "baseline"]
    patches = [r for r in rows if r["record"] == "intervention"]
    gate_indices = [i for i, r in enumerate(rows) if r["record"] == "development_gate"]
    assert len(gate_indices) == 1
    gate = gate_indices[0]
    assert rows[0] == dict(
        record="frozen_protocol", layer=meta["fixed_layer"], gate_before_confirmation=True
    )
    materials = json.loads((folder / "materials.json").read_text())
    development = {r["id"] for r in materials if r["split"] == "development"}
    confirmation = {r["id"] for r in materials if r["split"] == "confirmation"}
    assert len(development) == len(confirmation) == 6 and not development & confirmation
    assert len({r["family"] for r in materials}) == 12
    pairs = defaultdict(dict)
    lookup = {}
    for r in bases:
        key = tuple(r[k] for k in ["id", "construction", "reversal", "role", "condition"])
        assert key not in lookup
        lookup[key] = r
        assert r["id"] in (development if r["split"] == "development" else confirmation)
        assert r["family"] == r["id"]
        assert r["construction"] in CONSTRUCTIONS and r["role"] in ["agent", "patient"]
        assert r["reversal"] in [0, 1]
        assert r["condition"] in (["clean"] if r["split"] == "development" else ["clean", "gap"])
        first_agent = r["construction"] in ["active", "passive_relative"]
        first_correct = first_agent if r["role"] == "agent" else not first_agent
        assert r["prompt"]["first_index"] == r["reversal"]
        assert r["prompt"]["gold_index"] == (r["reversal"] if first_correct else 1 - r["reversal"])
        assert abs(validate_score(r["scores"], r["prompt"]["gold_index"]) - r["margin"]) < 1e-9
        if r["split"] == "development":
            pairs[r["construction"], r["role"], r["family"]][r["reversal"]] = r["margin"]
    rates = defaultdict(list)
    for (c, role, _), reversals in pairs.items():
        assert set(reversals) == {0, 1}
        rates[c + "/" + role].append(min(reversals.values()) > 0)
    assert len(rates) == 8 and all(len(v) == 6 for v in rates.values())
    rates = {k: float(np.mean(v)) for k, v in rates.items()}
    passed = min(rates.values()) >= 0.75
    assert result["gate"] == dict(passed=passed, both_reversal_rates=rates)
    assert {k: v for k, v in rows[gate].items() if k != "record"} == result["gate"]
    for i, r in enumerate(rows):
        if r.get("split") == "confirmation" or r["record"] == "intervention":
            assert passed and i > gate
    expected = (288, 768) if passed else (96, 0)
    assert (len(bases), len(patches)) == expected == (result["baselines"], result["interventions"])
    cells = defaultdict(set)
    norm_errors, identities, flips = [], [], 0
    for r in patches:
        key = tuple(r[k] for k in ["id", "construction", "reversal", "role", "condition"])
        b = lookup[key]
        assert r["kind"] not in cells[key]
        cells[key].add(r["kind"])
        assert r["direction"] == ("clean_to_gap" if r["condition"] == "gap" else "gap_to_clean")
        margin = validate_score(r["scores"], b["prompt"]["gold_index"])
        first = b["prompt"]["first_index"]
        effect = (r["scores"]["answer_logp"][first] - r["scores"]["answer_logp"][1 - first]) - (
            b["scores"]["answer_logp"][first] - b["scores"]["answer_logp"][1 - first]
        )
        assert abs(margin - r["margin"]) < 1e-9
        assert abs(margin - b["margin"] - r["qa_effect"]) < 1e-9
        assert abs(effect - r["first_effect"]) < 1e-9
        assert (
            abs(r["scores"]["next_margin"] - b["scores"]["next_margin"] - r["next_effect"]) < 1e-9
        )
        if r["kind"] == "identity":
            identities.append(max(abs(r["qa_effect"]), abs(effect), abs(r["next_effect"])))
        if r.get("requested_l2", 0) > 0:
            norm_errors.append(abs(r["actual_l2"] / r["requested_l2"] - 1))
        if r["kind"] == "cross":
            flips += (r["margin"] > 0) != (b["margin"] > 0)
    assert all(v == {"cross", "identity", "lexical", "random"} for v in cells.values())
    assert max(identities, default=0) <= 1e-4
    assert max(norm_errors, default=0) <= 1e-4
    accuracy = defaultdict(list)
    for r in bases:
        accuracy["/".join(r[k] for k in ["split", "condition", "construction", "role"])].append(
            r["margin"] > 0
        )
    return dict(
        run=folder.name,
        metadata=meta,
        results=result,
        effects=contrasts(patches),
        accuracy={k: float(np.mean(v)) for k, v in accuracy.items()},
        checks=dict(
            passed=True,
            raw_score_sums=True,
            independent_gate=True,
            source_hashes=True,
            gate_chronology=True,
            baselines=len(bases),
            interventions=len(patches),
            maximum_identity_error=max(identities, default=None),
            maximum_relative_norm_error=max(norm_errors, default=None),
        ),
        cross_choice_flips=int(flips),
        cross_observations=sum(r["kind"] == "cross" for r in patches),
    )


def legacy(folder):
    _, _, _, data = audit_legacy(folder)
    values = {(r["role"], r["kind"]): r["family_values"] for r in data["exploratory_role_effects"]}
    result = []
    for kind in ["cross", "lexical", "random", "cross_minus_lexical", "cross_minus_random"]:
        a, p = values["agent", kind], values["patient", kind]
        for name, sign in [("common_first", 1), ("question_difference", -1)]:
            v = {f: (-a[f] + sign * p[f]) / 2 for f in a}
            result.append(dict(kind=kind, component=name, family_values=v, **interval(v)))
    return dict(
        source=str(folder),
        observations_sha256=hashlib.sha256(
            (folder / "observations.jsonl").read_bytes()
        ).hexdigest(),
        post_hoc=True,
        effects=result,
    )


def build(folders, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [audit(f) for f in folders]
    if len(runs) == 2:
        assert runs[0]["metadata"]["model"] == "google/gemma-3-12b-pt"
        assert not runs[0]["results"]["gate"]["passed"]
        assert runs[1]["metadata"]["model"] == "google/gemma-3-27b-pt"
    old = legacy(Path("v3/bridge12/results/bridge12-20261009-01"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(
        json.dumps(dict(runs=runs, legacy=old), indent=2, allow_nan=False)
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    labels = list(runs[0]["results"]["gate"]["both_reversal_rates"])
    y = np.arange(len(labels))
    for i, run in enumerate(runs):
        rates = run["results"]["gate"]["both_reversal_rates"]
        axes[0].barh(
            y + (i - (len(runs) - 1) / 2) * 0.32,
            [rates[k] for k in labels],
            height=0.3,
            label=run["metadata"]["model"].split("/")[-1],
        )
    axes[0].set_yticks(y, labels)
    axes[0].axvline(0.75, color="black", linestyle="--")
    axes[0].set_xlim(0, 1.03)
    axes[0].set_title("V4: sviluppo, entrambe le inversioni corrette")
    axes[0].legend(fontsize=8)
    selected = [r for r in old["effects"] if r["component"] == "common_first"]
    for i, r in enumerate(selected):
        axes[1].errorbar(
            r["mean"],
            i,
            xerr=[[r["mean"] - r["ci95"][0]], [r["ci95"][1] - r["mean"]]],
            fmt="o",
            color="#267569",
        )
    axes[1].set_yticks(range(len(selected)), [r["kind"] for r in selected])
    axes[1].axvline(0, color="gray", linewidth=0.7)
    axes[1].set_title("V3 post-hoc: preferenza comune primo partecipante")
    axes[1].set_xlabel("Log-probabilità; IC bootstrap su sei famiglie")
    fig.savefig(output / "overview.png", dpi=160)
    fig.savefig(output / "overview.svg")
    plt.close(fig)
    pictures = ["overview.png"]
    for run in runs:
        if not run["effects"]:
            continue
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        for i, kind in enumerate(["cross", "cross_minus_lexical", "cross_minus_random"]):
            selected = [
                next(r for r in run["effects"] if r["kind"] == kind and r["contrast"] == c)
                for c in WEIGHTS
            ]
            means = np.array([r["mean"] for r in selected])
            limits = np.array([r["ci95"] for r in selected])
            ax.errorbar(
                means,
                np.arange(5) + (i - 1) * 0.18,
                xerr=[means - limits[:, 0], limits[:, 1] - means],
                fmt="o",
                label=kind,
            )
        ax.set_yticks(range(5), list(WEIGHTS))
        ax.axvline(0, color="gray")
        ax.legend()
        ax.set_title("V4 conferma: contrasti pianificati, sei famiglie")
        ax.set_xlabel("Variazione log-probabilità; IC 95% non corretti per molteplicità")
        name = run["run"] + "-effects.png"
        fig.savefig(output / name, dpi=160)
        fig.savefig(output / name.replace(".png", ".svg"))
        plt.close(fig)
        pictures.append(name)

    def ci(r):
        return f"{r['mean']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"

    body = "<h1>V4 · Chi è il primo partecipante?</h1><p>Una prova più severa per distinguere posizione nella frase, soggetto del verbo interrogato e ruolo semantico.</p>"
    if all(not r["results"]["gate"]["passed"] for r in runs):
        body += "<p><strong>Il test V4 non supera la validazione comportamentale.</strong> Il fallimento riguarda l’agente nelle relative passive anche aumentando la scala. Nessuna inferenza sulle famiglie di conferma e nessuna patch V4 sono state eseguite. Non è evidenza che i modelli non rappresentino i ruoli: è un limite di questa combinazione di costruzione e domanda.</p>"
    body += "<p>Il primo tentativo 12B si è interrotto dopo 12 misure di sviluppo su un controllo numerico del prefisso. Le risposte di diversa lunghezza sono ora portate alla stessa dimensione con padding mascherato; si valutano soltanto i token reali e la tolleranza resta 1e-5. Tentativo originale e sorgenti sono conservati. Nessuna scelta scientifica è cambiata.</p>"
    for run in runs:
        result = run["results"]
        status = (
            "Soglia superata; conferma eseguita."
            if result["gate"]["passed"]
            else "Soglia fallita; conferma e interventi non eseguiti."
        )
        body += f"<h2>{html.escape(run['metadata']['model'])}</h2><p><strong>{status}</strong> {result['baselines']} misure di base, {result['interventions']} interventi. Layer {run['metadata']['fixed_layer']}; FP32; {run['metadata']['elapsed_seconds']:.1f} secondi inclusi caricamento e misura.</p>"
        if result["gate"]["passed"]:
            body += f"<p>Risposta preferita cambiata in {run['cross_choice_flips']}/{run['cross_observations']} patch cross-condition. Identità: errore massimo {run['checks']['maximum_identity_error']:.2g}.</p><div class='table'><table><tr><th>Contrasto</th><th>Patch virgola</th><th>Meno lessicale</th><th>Meno casuale</th></tr>"
            for c in WEIGHTS:
                body += (
                    "<tr><td>"
                    + c
                    + "</td>"
                    + "".join(
                        "<td>"
                        + ci(
                            next(r for r in run["effects"] if r["kind"] == k and r["contrast"] == c)
                        )
                        + "</td>"
                        for k in ["cross", "cross_minus_lexical", "cross_minus_random"]
                    )
                    + "</tr>"
                )
            body += "</table></div>"
    body += "<h2>Prima di V4: il bias era specifico?</h2><p>Non emerge una specificità: nei dati V3, la preferenza comune per il primo partecipante è simile per patch virgola, lessicale e casuale. Le differenze appaiate hanno intervalli che includono zero: non è una dimostrazione di equivalenza, ma manca evidenza di un vantaggio specifico della patch. Questa è una rianalisi esplorativa degli stessi sei gruppi di test, non nuova conferma. La dipendenza dalla domanda resta distinta dal bias comune.</p>"
    for name in pictures:
        encoded = base64.b64encode((output / name).read_bytes()).decode()
        body += f"<img alt='{html.escape(name)}' src='data:image/png;base64,{encoded}'>"
    body += '<h2>Come leggere il controllo</h2><p>Il prefisso termina sempre sul primo partecipante. Le continuazioni attiva, passiva, relativa oggetto e relativa passiva incrociano due proprietà: essere agente dell’evento ed essere soggetto del verbo interrogato. Le domande chiedono chi agisce e chi subisce. Entrambe le inversioni dei nomi devono essere corrette in almeno cinque famiglie su sei, in ciascuna delle otto combinazioni costruzione/domanda.</p><p>Per ogni costruzione, C è la media delle variazioni di preferenza per il primo partecipante nelle due domande; D è metà della loro differenza. I contrasti confrontano C con le assegnazioni di soggetto/agente; role_selective combina D con il ruolo corretto. Le due direzioni di patch sono orientate con segni opposti.</p><h2>Limiti</h2><p>Nelle relative, il primo partecipante resta soggetto della frase principale (“smiled”): separiamo il soggetto dell’evento interrogato, non ogni nozione di soggetto. Le relative sono più complesse e non validate con persone. Dodici famiglie sintetiche, sei per split; intervalli su sei famiglie e una sola direzione casuale per item/inversione/direzione. Nessuna correzione per confronti multipli. Il confronto predittivo was/virgola non è la continuazione effettiva delle relative. Un effetto dipendente dalla domanda non dimostra un circuito astratto dei ruoli.</p><p>Protocollo: <a href="../../PROTOCOL.md">PROTOCOL.md</a>. Numeri e verifiche: <a href="audit.json">audit.json</a>.</p>'
    body += "<h2>Cosa cambia e come proseguire</h2><p>Resta l’evidenza V3 che intervenire su un singolo stato cambia fortemente una preferenza predittiva locale. La preferenza comune per il primo partecipante non identifica una rappresentazione sintattica specifica: i controlli possono produrla anch’essi. Gli spostamenti opposti fra domande restano un risultato distinto, senza cambi della risposta preferita nel precedente test.</p><p>Prima di altre GPU per patching, serve validare lo strumento: costruzioni meno innaturali, controllo senza clausola garden-path e domande su evento/soggetto formulate in modo non ambiguo. Questo sarebbe nuovo sviluppo; le soglie andrebbero congelate di nuovo e la conferma dovrebbe restare separata. Aumentare ancora la scala, da solo, non ha risolto il problema osservato.</p>"
    body += "<h2>Un errore concreto</h2><blockquote>While the nurse helped, the librarian by whom the archivist was telephoned smiled.<br>Who telephoned someone?</blockquote><p>La risposta corretta è <em>the librarian</em>: è la persona che ha telefonato all’archivista. Entrambi i modelli assegnano più probabilità a <em>the archivist</em> fra i due candidati. È una scelta misurata mediante probabilità delle continuazioni, non una risposta generata liberamente.</p>"
    document = (
        '<!doctype html><html lang="it"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Garden-path V4</title><style>body{font:17px/1.6 system-ui;max-width:1100px;margin:auto;padding:28px;color:#243b38;background:#fafaf7}h1{font-size:clamp(30px,5vw,52px);line-height:1.15}h2{margin-top:2em}img{width:100%;height:auto;margin:24px 0}.table{overflow-x:auto}table{border-collapse:collapse;min-width:690px}th,td{text-align:left;padding:10px;border-bottom:1px solid #ccc}a{color:#176c60}@media(prefers-color-scheme:dark){body{color:#e2e9e7;background:#172321}a{color:#8ed5c6}}</style>'
        + body
        + "</html>"
    )
    (output / "report.html").write_text(document)
    print(
        json.dumps(
            [dict(run=r["run"], results=r["results"], effects=r["effects"]) for r in runs], indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build(args.runs, args.output)
