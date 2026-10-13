"""Build the explicitly public GPS report bundle; never copy the repository."""

import hashlib
import html
import json
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import markdown

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "site/mech-interp/garden-path-sentences"
REPORT = ROOT / "reports/Garden path evidenze e limiti.md"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.ids = [], set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for key in ["href", "src"]:
            if key in attrs:
                self.links.append(attrs[key])


def validate_public_tree(root):
    root = root.resolve()
    for path in root.rglob("*"):
        if path.is_symlink() or any(part.startswith(".") for part in path.relative_to(root).parts):
            raise ValueError(f"Private file or symlink: {path.name}")
        if not path.is_file() or path.suffix != ".html":
            continue
        text = path.read_text()
        if re.search(r"/Users/|/home/deploy/|file://|localhost:", text):
            raise ValueError(f"Local reference in {path.name}")
        parsed = Links()
        parsed.feed(text)
        for link in parsed.links:
            url = urlsplit(link)
            if url.scheme in ["https", "http", "mailto", "data"]:
                continue
            if url.scheme or url.netloc:
                raise ValueError(f"Unsupported URL: {link}")
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            if target.is_dir():
                target /= "index.html"
            if not target.is_relative_to(root) or not target.is_file():
                raise ValueError(f"Missing or out-of-bundle link: {link}")
            if url.fragment and target.suffix == ".html":
                destination = Links()
                destination.feed(target.read_text())
                if unquote(url.fragment) not in destination.ids:
                    raise ValueError(f"Missing anchor: {link}")


def render(text, title, base="", require_levels=True):
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc"])
    headings = re.findall(r'<h2 id="([^"]+)">(Livello [123][^<]+)</h2>', body)
    if require_levels and len(headings) != 3:
        raise ValueError("Exactly three complexity levels required")
    nav = "".join(
        f'<a href="#{anchor}">{html.escape(label.split(" — ")[0])}</a>'
        for anchor, label in headings
    )
    body = body.replace("<table>", '<div class="table" tabindex="0">\n<table>').replace(
        "</table>", "</table>\n</div>"
    )
    body = re.sub(
        r'<img ([^>]*?)src="([^"]+\.svg)"([^>]*?)>',
        lambda m: (
            f'<a class="figure" href="{m[2]}" aria-label="Apri il grafico a tutta pagina"><img {m[1]}src="{m[2]}"{m[3]}></a>'
        ),
        body,
    )
    return f"""<!doctype html>
<html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark"><meta name="description" content="Garden-path sentences nei modelli linguistici: esperimenti, interventi causali e limiti delle prove. Report italiano a tre livelli.">
<title>{html.escape(title)} · grokked.it research</title><link rel="stylesheet" href="{base}styles.css">
<script defer data-domain="research.grokked.it" src="https://check.grokked.it/js/script.js"></script>
</head><body><a class="skip" href="#report">Vai al contenuto</a>
<header><a href="https://research.grokked.it/">research.grokked.it</a><span> / </span><a href="https://research.grokked.it/mech-interp/">mech-interp</a><span class="date">Aggiornato il 13 ottobre 2026</span></header>
<nav aria-label="Livelli di lettura">{nav}<a href="{base}index.html">Report completo</a><a href="{base}report.md">Scarica il testo</a><a href="{base}data/evidence.json">Dati dei grafici</a></nav>
<main id="report">{body}</main>
<footer><p>Report GPS · Luca Mattiazzi · <a href="https://grokked.it/">grokked.it</a></p>
<p><a href="{base}data/manifest.json">Manifest dei file pubblicati</a> · <a href="{base}report.md">Versione Markdown</a> · <a href="{base}data/evidence.json">Evidenze aggregate</a></p></footer></body></html>"""


def build():
    text = REPORT.read_text()
    if any(token in text for token in ["@@", "SEGNAPOSTO", "TODO", "PENDING"]):
        raise ValueError("Unresolved placeholder")
    if OUT.exists():
        # This is only our own generated bundle, never the shared site or repository.
        shutil.rmtree(OUT)
    (OUT / "figures").mkdir(parents=True)
    (OUT / "data").mkdir()
    (OUT / "report.md").write_text(text)
    (OUT / "index.html").write_text(render(text, "Garden path: evidenze e limiti"))
    shutil.copyfile(HERE / "styles.css", OUT / "styles.css")
    shutil.copyfile(HERE / "evidence.json", OUT / "data/evidence.json")
    shutil.copyfile(HERE / "title-case-study.json", OUT / "data/title-case-study.json")
    for p in sorted((HERE / "figures").glob("*.svg")):
        shutil.copyfile(p, OUT / "figures" / p.name)
    protocols = {
        "protocol-v2.txt": "v2/GARDEN_PROTOCOL.md",
        "protocol-v3.txt": "v3/PROTOCOL.md",
        "protocol-bridge12.txt": "v3/bridge12/PROTOCOL.md",
        "protocol-v4.txt": "v4/PROTOCOL.md",
        "protocol-title.txt": "gps_title/PROTOCOL.md",
    }
    for target, source in protocols.items():
        shutil.copyfile(ROOT / source, OUT / "data" / target)
    snapshots = {
        "v3-1b": ("v3/results/npz-v3-20261008-01", "protocol-v3.txt", "8 ottobre 2026"),
        "v3-12b": (
            "v3/bridge12/results/bridge12-20261009-01",
            "protocol-bridge12.txt",
            "9 ottobre 2026",
        ),
    }
    for slug, (folder, protocol, date) in snapshots.items():
        source = ROOT / folder
        destination = OUT / "esperimenti" / slug
        destination.mkdir(parents=True)
        original = (source / "report.md").read_text()
        original = original.replace("../../PROTOCOL.md", "../../data/" + protocol)
        if slug == "v3-12b":
            original = original.replace(
                "l'accuratezza QA precedente riguarda invece la frase completata",
                "l'accuratezza QA riportata nella tabella precedente riguarda la frase completata",
            )
        notice = (
            f"> Scheda del run del {date}, conservata come resoconto sperimentale. "
            "Il [report GPS aggiornato](../../index.html) distingue effetti predittivi, "
            "risposte finali e rianalisi, e include le verifiche successive. "
            "Questi risultati non identificano da soli un meccanismo di reinterpretazione.\n\n"
        )
        if slug == "v3-12b":
            notice += (
                "> Aggiornamento interpretativo del 12 ottobre: nella scomposizione successiva, "
                "la componente comune verso il primo partecipante non mostra un vantaggio "
                "specifico sui controlli lessicale e casuale. Gli intervalli sono ampi; "
                "questo non dimostra equivalenza. I valori originali del run sono conservati.\n\n"
            )
        figures = []
        for name in ["report.svg", "report-roles.svg"]:
            if (source / name).exists():
                shutil.copyfile(source / name, destination / name)
                figures.append(f"\n\n![Grafico del run {slug}]({name})")
        snapshot = notice + original + "".join(figures)
        (destination / "index.html").write_text(
            render(snapshot, f"Scheda {slug}", base="../../", require_levels=False)
        )
        (destination / "report.md").write_text(snapshot)
    # Only the report, its figures, authored protocols and aggregate data are public.
    files = {
        str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(OUT.rglob("*"))
        if p.is_file()
    }
    manifest = dict(
        schema=1,
        publication_date="2026-10-13",
        scope="GPS only: original critical-token surprisal, causal follow-ups, auxiliary GPS-context gates, and exploratory title case study.",
        word_count=len(text.split()),
        levels=3,
        files=files,
        sources={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                REPORT,
                HERE / "evidence.json",
                HERE / "plot_results.py",
                Path(__file__),
                HERE / "styles.css",
                HERE / "title-case-study.json",
                ROOT / "gps_title/analyze.py",
                ROOT / "gps_title/results/title-20261013-01/results.json",
                ROOT / "gps_title/results/title-20261013-01/metadata.json",
            ]
            + [ROOT / source for source in protocols.values()]
            + [ROOT / folder / "report.md" for folder, _, _ in snapshots.values()]
        },
    )
    (OUT / "data/manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    validate_public_tree(OUT)
    print(
        json.dumps(
            dict(output=str(OUT), files=len(files) + 1, words=manifest["word_count"], levels=3)
        )
    )


if __name__ == "__main__":
    build()
