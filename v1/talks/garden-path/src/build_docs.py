"""Build printable, offline talk materials from the reviewed text sources."""
import html
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'output'
SLIDES = json.loads((ROOT / 'src/content.json').read_text())
CSS = '''html{max-width:78ch;margin:auto;font:20px/1.65 Arial,sans-serif;background:white;color:#222}body{margin:2rem 1.2rem}h1,h2,h3{color:black;line-height:1.2}h1{font-size:2.2rem}h2{margin-top:2rem;font-size:1.4rem}a{color:#00e;text-decoration:none}nav{font-size:.85rem;margin-bottom:2rem}table{border-collapse:collapse;width:100%;font-size:.8rem}th,td{text-align:left;vertical-align:top;border-bottom:1px solid #bbb;padding:.65rem .4rem}p{margin:.9rem 0}small,.meta{color:#555;font-size:.85rem}.cue{color:#00e}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:16px/1.5 "Courier New",monospace}section{scroll-margin-top:2rem}img{max-width:100%;height:auto;border:1px solid #ddd}@media print{html{font-size:11pt;max-width:none}body{margin:0}nav{display:none}h2,h3{break-after:avoid}tr,img{break-inside:avoid}a{color:#222}}'''
NAV = '<nav><a href="index.html">Materiali</a> / <a href="garden-path-talk.pptx">PowerPoint</a> / <a href="canovaccio.html">Canovaccio</a> / <a href="copione-20-minuti.html">Copione</a> / <a href="ripasso-tecnico.html">Ripasso</a> / <a href="domande.html">Domande</a></nav>'


def page(name, title, body):
    (OUT / name).write_text(f'<!doctype html><html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{CSS}</style></head><body>{NAV}<h1>{html.escape(title)}</h1>{body}</body></html>')


def render_text(text):
    parts = []
    for block in text.split('\n\n'):
        escaped = html.escape(block.strip())
        escaped = re.sub(r'https://[^\s&lt;]+', lambda m: f'<a href="{m[0]}">{m[0]}</a>', escaped)
        if re.match(r'^\d+\. ', block) and '\n' not in block.strip():
            parts.append(f'<h2>{escaped}</h2>')
        else:
            parts.append(f'<p>{escaped.replace(chr(10), "<br>")}</p>')
    return ''.join(parts)


OUT.mkdir(exist_ok=True, parents=True)
words = sum(len(s['script'].split()) for s in SLIDES[:14])
assert sum(s['seconds'] for s in SLIDES) == 1200
intro = f'''<p>14 slide principali e 2 di backup. Copione in italiano di circa {words:,} parole, con esempi originali in inglese. Destinatari: sviluppatori con nozioni di base sugli LLM.</p><p>La scansione nominale somma a 20 minuti: circa 130 parole al minuto più brevi pause. La durata reale dipende dal ritmo e dalle interazioni. Fai una prima prova cronometrata e adatta il testo. Le domande finali sono escluse dai 20 minuti.</p>'''
timing = '''<h2>Formato dell’intervento</h2><ul><li><strong>Slot da 20 minuti senza domande:</strong> percorso principale, slide 1–14.</li><li><strong>Slot da 20 minuti con Q&amp;A:</strong> punta a 17 minuti di esposizione; comprimi architettura, dataset e audit, mantenendo i limiti centrali.</li><li><strong>Slot da 25 minuti:</strong> 20 minuti di esposizione e 5 di domande.</li><li><strong>Slot da 30 minuti:</strong> 20 minuti, 4 minuti di approfondimenti con le due slide backup e 6 minuti di domande.</li><li><strong>30 minuti tutti parlati:</strong> usa i moduli di estensione nel canovaccio e prova il ritmo prima del talk.</li></ul>'''
outline = '<table><thead><tr><th>Tempo</th><th>Slide</th><th>Da ricordare</th></tr></thead><tbody>'
for i, s in enumerate(SLIDES[:14], 1):
    outline += f'<tr><td>{s["time"]}</td><td>{i:02} · {html.escape(s["title"])}</td><td>{html.escape(s["cue"])}</td></tr>'
outline += '</tbody></table>'
extensions = '''MODULI PER ESTENDERE FINO A CIRCA 30 MINUTI DI ESPOSIZIONE

A. Dopo la slide 2, +2 minuti: fai leggere “While the man hunted the deer ran into the woods.” Chiedi al pubblico che ruolo avesse assegnato inizialmente a the deer. Mostra la differenza introdotta dalla virgola dopo hunted. Aspetta le risposte e spiega che il cervo è soggetto di ran nella lettura finale. È una dimostrazione informale, non una raccolta di dati. Se non c’è interazione, questo modulo durerà meno.

B. Dopo la slide 6, +2 minuti: usa la slide backup 15. Confronta NP/S e NP/Z con la relativa ridotta. Spiega oggetto contro proposizione, oggetto contro nessun oggetto, e perché cambiare costruzione cambia anche i confondenti. Non leggere cinque esempi in rapida sequenza: fermati su due.

C. Dopo la slide 10, +2 minuti: esempio didattico del recupero. S_GPS=6, S_normale=4, S_patch=5: (6−5)/(6−4)=50%. Chiedi che cosa significherebbe S_patch=3: 150%, previsione più favorevole del normale, non “comprensione al 150%”. Se S_patch=7, il recupero è −50%. Sono numeri illustrativi, non risultati del progetto.

D. Dopo la slide 11, +2 minuti: usa la slide backup 16. Commenta media, intervallo e leave-one-out. Distingui variabilità fra item da variabilità fra run. Spiega perché scegliere un item da eliminare dopo aver visto il risultato falserebbe la domanda. In un seguito serve un criterio linguistico stabilito prima.

E. Dopo la slide 13, +2 minuti: proponi un modello fittizio che risponde sempre A. Con due ordini bilanciati ottiene il 50% sulle presentazioni ma lo 0% di robustezza all’ordine. Poi proponi una scorciatoia diversa, basata sulla posizione dei nomi: invertire i nomi non la elimina necessariamente. Chiedi quale controllo aggiungere. Chiudi distinguendo QA finale e lettura incrementale.

Questi moduli comprendono esempi e pause guidate: la durata è una stima da verificare nelle prove, non dieci minuti aggiuntivi di testo da leggere.
'''
shortcuts = '''PROVE E RECUPERO DEL TEMPO

Prima prova: leggi il copione integralmente e segna i tempi reali a fine slide. Non accelerare per rispettare il cronometro.
Seconda prova: usa soltanto il canovaccio. Memorizza l’apertura, la distinzione previsione/comprensione e la chiusura.
Terza prova: simula tre domande fra quelle allegate, rispondendo entro 30–45 secondi ciascuna.

Checkpoint: slide 4 finita a 05:00; slide 8 a 10:20; slide 11 a 15:00; conclusione a 20:00.
Se sei in ritardo di un minuto, riduci i dettagli sui cinque template e leggi solo due righe della tabella del patching.
Se sei in ritardo di due o tre minuti, compatta il controllo del 100% in una frase e descrivi un solo controllo del pilot. Conserva sempre la distinzione fra risultato misurato e interpretazione.
Se hai tempo in più, usa un modulo di estensione, senza improvvisare nuove conclusioni.

Le slide 15–16 sono backup dopo la chiusura. Durante la versione da 20 minuti fermati alla 14. La slide 2 mostra soltanto la frase: lascia leggere il pubblico prima di spiegarla oralmente.

Il talk funziona interamente offline. La demo è la frase letta insieme al pubblico. Evita una nuova inferenza live, che non è necessaria per raccontare i log storici.
'''
canovaccio = 'GARDEN-PATH SENTENCES — CANOVACCIO E SCALETTA\n\n20 minuti di esposizione, Q&A escluso.\n\n'
for i,s in enumerate(SLIDES[:14],1):
    canovaccio += f'{i:02}. {s["time"]} — {s["title"]}\n{s["cue"]}\nPassaggio: {s["transition"]}\n\n'
canovaccio += extensions + '\n' + shortcuts
(OUT/'canovaccio.txt').write_text(canovaccio)
page('canovaccio.html','Canovaccio e scaletta',intro+timing+outline+render_text(extensions)+render_text(shortcuts))
speech = 'GARDEN-PATH SENTENCES — COPIONE PER 20 MINUTI\n\nTesto modificabile. I tempi sono obiettivi di prova, non tempi misurati. Le indicazioni fra parentesi quadre non vanno lette.\n\n'
body = intro+'<p><a href="copione-20-minuti.txt">Apri il testo modificabile (.txt)</a>. Le frasi di passaggio sono già incluse nel copione; non ripeterle dalle note.</p>'
for i,s in enumerate(SLIDES[:14],1):
    speech += f'{i:02}. {s["time"]} — {s["title"]}\n\n{s["script"]}\n\n'
    body += f'<section id="s{i}"><h2>{i:02}. {html.escape(s["title"])}</h2><p class="meta">{s["time"]}</p><p>{html.escape(s["script"])}</p></section>'
(OUT/'copione-20-minuti.txt').write_text(speech)
page('copione-20-minuti.html','Copione per 20 minuti',body)
for source,name,title in [('ripasso.txt','ripasso-tecnico','Ripasso tecnico'),('domande.txt','domande','Domande plausibili e risposte')]:
    text=(ROOT/'src'/source).read_text()
    (OUT/f'{name}.txt').write_text(text)
    page(f'{name}.html',title,render_text(text))
preview=OUT/'anteprima'
preview.mkdir(exist_ok=True)
slides_html=''
for i,s in enumerate(SLIDES,1):
    src=ROOT/'.build'/f'slide-{i:02}.png'
    if src.exists():
        shutil.copy2(src,preview/src.name)
        slides_html+=f'<section><h2>{i:02}. {html.escape(s["title"])}</h2><img loading="lazy" src="anteprima/{src.name}" alt="Slide {i}: {html.escape(s["title"])}"></section>'
page('anteprima.html','Anteprima delle slide',slides_html)
page('index.html','Garden-path sentences',intro+'''<ul><li><a href="garden-path-talk.pptx">Presentazione PowerPoint</a> — testo, tabella e grafico modificabili; copione nelle note.</li><li><a href="anteprima.html">Anteprima di tutte le slide</a></li><li><a href="canovaccio.html">Canovaccio e scaletta</a> — tempi, passaggi, tagli ed estensioni a 30 minuti. <a href="canovaccio.txt">Testo</a>.</li><li><a href="copione-20-minuti.html">Copione completo</a> — circa 2.500 parole. <a href="copione-20-minuti.txt">Testo da modificare</a>.</li><li><a href="ripasso-tecnico.html">Ripasso tecnico</a> — concetti, formule, implementazione e limiti. <a href="ripasso-tecnico.txt">Testo</a>.</li><li><a href="domande.html">24 domande plausibili con risposte</a>. <a href="domande.txt">Testo</a>.</li></ul>'''+timing+'''<h2>Uso dei materiali</h2><p>Apri i file HTML nel browser per studiare e stampare. Modifica i .txt con qualsiasi editor o le note direttamente in PowerPoint. Le modifiche manuali ai file esportati non si sincronizzano con gli altri formati. I sorgenti del pacchetto sono nella cartella src.</p><h2>Contenuti e stile</h2><p>Materiali basati sul codice, sui log salvati e sul protocollo locale consultati il 1 ottobre 2026. I risultati storici e il pilot ancora da eseguire sono distinti. Il talk segue le interpretazioni riviste nell’audit, più prudenti del <a href="https://grokked.it/posts/garden_path_sentences/">post originale</a>.</p><p>Stile ispirato al tema chiaro di <a href="https://grokked.it/">grokked.it</a>: bianco, nero e blu #0000EE, composizione tipografica essenziale. Arial è un’alternativa portabile al font dichiarato dal sito, Sarabun.</p>''')
print(f'Generated talk materials: {words} words, 1200 seconds, {len(SLIDES)} slides.')
