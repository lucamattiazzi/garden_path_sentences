# Pubblicazione del report GPS

URL: <https://research.grokked.it/mech-interp/garden-path-sentences/>.
Report italiano a tre livelli, aggiornato al 13 ottobre 2026: V1–V3 e ponte
causale 12B; V4 originale compare soltanto come tentativo fermato al gate.
Le analisi generali dei ruoli V5–V9 non fanno parte di questa pubblicazione.
Nessuna nuova inferenza viene eseguita dalla build. Il caso sul titolo usa
un run GPU separato completato il 13 ottobre in `gps_title/`.

Rilascio verificato: `20261013T111953Z-gps`, 26 file, 8.216 parole e otto grafici principali.
Manifest SHA-256: `cb817d7000814dc609087c312bdccaf42199ef5a1b71bd4bc416ba02573c4564`.
Backup precedente: `/home/deploy/apps/research-deploy-backups/20261013T111953Z-gps/mech-interp/garden-path-sentences`.
Ricevute: `deployment-20261013.json`, `public-verification-20261013.json`, `verification-20261013.json`.
Verificati tutti i file via HTTPS/DNS ordinario e le tre pagine su desktop/mobile.
Configurazione Caddy, container e report `crittografie` invariati. Le ricevute senza
suffisso di data conservano parte della cronologia del primo rilascio; per lo stato
corrente usare i file datati sopra.

## Build e verifiche locali

Dalla root del repository:

```bash
uv run --no-project --with matplotlib python reports/2026-10-12-gps/plot_results.py
uv run --no-project --with matplotlib python -m gps_title.analyze
uv run --no-project --with markdown python reports/2026-10-12-gps/build_site.py
uv run --no-project --with markdown python -m unittest discover -s reports/2026-10-12-gps -p test_build_site.py
python3 -m unittest discover -s deploy/gps -p test_publish.py
node deploy/gps/check-ui.mjs
```

Il controllo UI richiede Playwright nel runtime Codex e Google Chrome sul Mac;
nel caso il comando `node` non sia nel PATH, usare il binario del runtime:
`~/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node`.
Gli screenshot e le verifiche browser sono in `reports/2026-10-12-gps/.build/`.
Il report sorgente è `reports/Garden path evidenze e limiti.md`.
I sei grafici V3/V4 derivano da `reports/2026-10-12-gps/evidence.json`;
il file contiene aggregati, riferimenti ai campi originali e hash delle fonti.

L'output pubblico è esclusivamente `site/mech-interp/garden-path-sentences/`:
HTML, Markdown, CSS, otto SVG principali, cinque protocolli e dati con manifest SHA-256.
I due nuovi grafici provengono da `gps_title/analyze.py` e il caso sul titolo
include punteggi token per token e le sei risposte qualitative complete.
Le due schede V3 includono avvertenze interpretative aggiornate; i resoconti
originali nel repository restano intatti. Nessun corpus integrale, log privato,
credenziale, configurazione server o symlink viene incluso.

## Pubblicazione

Prima di ogni rilascio rileggere `~/.agents/SERVER.md`, verificare l'indirizzo
UTM e controllare che lo slug live appartenga a questo report.
Il server e l'hosting Caddy esistono già: non modificare configurazioni,
Compose, immagini, template condivisi o gli altri report. Non serve un reload.

1. Scegliere un identificativo UTC `YYYYMMDDTHHMMSSZ-gps`.
2. Copiare soltanto la build in
   `/home/deploy/apps/research-staging/<release>/mech-interp/garden-path-sentences/`.
3. Copiare `publish.py` in `/home/deploy/apps/research-staging/<release>/publish.py`,
   esterno alla cartella pubblicabile. Impostare directory 755 e file 644.
4. Calcolare SHA-256 di `data/manifest.json`. Per un aggiornamento, leggere
   anche l'hash del manifest live e usarlo come `--previous`; alla prima
   pubblicazione usare `--previous absent`.
5. Eseguire sulla VM:

```text
python3 /home/deploy/apps/research-staging/<release>/publish.py <release> <manifest-sha256> --previous <hash-precedente-o-absent>
```

Lo script verifica manifest, insieme esatto dei file e hash, poi promuove
solo la directory di questo report. Se presente, conserva la versione precedente
in `/home/deploy/apps/research-deploy-backups/<release>/mech-interp/garden-path-sentences/`.
Verifica che configurazione condivisa e identità dei container non siano cambiate.
Un errore durante la promozione ripristina automaticamente la versione precedente.
La ricevuta è `research-deploy-backups/<release>/mech-interp/deployment.json`;
copiarla anche nel progetto come `deployment.json`.

Dopo la promozione verificare HTTPS usando DNS pubblico, percorso indice →
`mech-interp` → report, tutte le risorse e i download, e uguaglianza degli hash:

```bash
node deploy/gps/check-ui.mjs https://research.grokked.it/mech-interp/garden-path-sentences/
python3 deploy/gps/verify_live.py
```

Confrontare inoltre l'impronta degli altri report e della configurazione condivisa
registrata prima e dopo. I file di verifica locali non sono pubblicati.

## Rollback

Verificare prima l'hash della versione live. Spostare solo la cartella live GPS
in un nuovo percorso fuori dalla root pubblica, poi riportare il backup della
specifica release nella destinazione GPS. Alla prima pubblicazione non esiste
una versione precedente: il rollback consiste nello spostare il nuovo report
fuori dalla root pubblica. Non cancellare dati e non ripristinare Caddy o la
categoria condivisa. Verificare nuovamente indice, HTTPS e gli altri report.
