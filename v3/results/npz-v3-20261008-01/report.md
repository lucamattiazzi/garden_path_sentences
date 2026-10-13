# V3 — Preferenze sintattiche, un token alla volta

Run `npz-v3-20261008-01` · Gemma 3 1B pretrained · layer 17 fissato prima del run · 133.7 secondi nella funzione GPU su NVIDIA A10.

**Stato: esperimento completato.** 24 materiali pubblicati, 12 famiglie di verbi, 12 frasi di sviluppo e 12 di test. Le frasi sono due per famiglia: il test ha sei unità di ricampionamento, non dodici famiglie indipendenti. I risultati sono esplorativi.

## Perché questa versione

V2 confrontava continuazioni di lunghezze diverse. Qui il confronto principale è tra la parola `was` e una virgola, ciascuna verificata come singolo token. Salviamo le due log-probabilità, la loro differenza e gli effetti delle patch separatamente. Nessun punteggio di stringhe lunghe entra nella misura principale.

I materiali provengono da [Hanna e Mueller (2025)](https://aclanthology.org/2025.naacl-long.164/) e dal [repository degli autori](https://github.com/hannamw/GP-mechanisms). Le varianti e le combinazioni usate qui sono adattamenti: non abbiamo raccolto giudizi umani sul nostro test.

## Risultato principale sul test

Il segnale supera entrambi i controlli anche nel contrasto con posizioni allineate. Questo sostiene un effetto sulle preferenze di continuazione oltre questo particolare controllo posizionale.

| Misura: interazione opzionale − intransitivo | Media [IC95%], nats |
|---|---|
| Patch, posizioni allineate | 6.599 [4.065, 8.734] |
| Vantaggio sul controllo lessicale | 6.223 [3.706, 8.148] |
| Vantaggio sulle direzioni casuali | 6.564 [3.957, 8.786] |

Una differenza positiva indica che la patch sposta `was` rispetto alla virgola più nel contesto con verbo opzionalmente transitivo che nel controllo intransitivo. Le due direzioni della patch sono allineate nel segno prima della media. Gli intervalli ricampionano intere famiglie, 10.000 volte, e non coprono l’incertezza delle scelte linguistiche o tutti i confronti del progetto.

### Sviluppo e test, riportati separatamente

| Split | Patch allineata | Vantaggio lessicale | Vantaggio casuale |
|---|---|---|---|
| development | 4.911 [3.206, 6.518] | 4.940 [3.206, 6.617] | 4.757 [2.650, 6.623] |
| test | 6.599 [4.065, 8.734] | 6.223 [3.706, 8.148] | 6.564 [3.957, 8.786] |

## Che cosa controlliamo sulle posizioni

In “gap” i token sono identici al prefisso senza virgola, ma i position ID dal secondo gruppo nominale in poi aumentano di uno. Corrispondono così ai position ID degli stessi token nella frase con virgola. Non viene inserito un token nascosto: è una perturbazione artificiale della codifica posizionale.

Questo controllo non riproduce il contributo del token virgola all’attenzione e non isola perfettamente la sintassi dalla punteggiatura. Esclude soltanto una spiegazione basata interamente su questo specifico spostamento dei position ID, se il segnale resta presente.

| Contrasto | Interazione di base | Patch | Controllo lessicale | Controllo casuale |
|---|---|---|---|---|
| Virgola / originale | 6.479 [3.583, 9.042] | 6.437 [3.531, 9.008] | 0.339 [-0.089, 0.734] | 0.374 [0.245, 0.519] |
| Virgola / posizioni allineate | 6.719 [4.214, 8.844] | 6.599 [4.065, 8.734] | 0.376 [-0.141, 0.849] | 0.035 [-0.224, 0.335] |
| Solo posizioni | -0.240 [-1.016, 0.510] | -0.141 [-0.898, 0.648] | -0.016 [-0.073, 0.044] | 0.106 [0.062, 0.174] |

Le tre interazioni di base si scompongono algebricamente: virgola/originale = virgola/allineato + solo posizioni. Le patch agiscono in un sistema non lineare e non devono soddisfare la stessa somma.

## Quale token cambia?

| Componente, posizioni allineate | Media [IC95%], nats |
|---|---|
| Effetto su log P(was) | 4.291 [2.529, 5.911] |
| Effetto su log P(virgola) | -2.308 [-3.250, -1.291] |
| Differenza was − virgola | 6.599 [4.065, 8.734] |

Il contributo della virgola va sottratto: la tabella riporta il suo segno originale. L’interazione misura una differenza tra classi di verbo, non un cambiamento assoluto identico in ogni frase.

## Prima della patch: was o virgola?

| Classe del verbo | Condizione | Margine medio, nats | Frasi con P(was) > P(virgola) |
|---|---|---|---|
| optional | native | -5.729 | 0/12 |
| optional | gap | -4.958 | 1/12 |
| optional | comma | +3.229 | 12/12 |
| intransitive | native | +1.161 | 10/12 |
| intransitive | gap | +2.172 | 10/12 |
| intransitive | comma | +3.641 | 12/12 |

Un margine negativo rende la virgola più probabile di `was`. Questo non dimostra un parse sbagliato: le due possibilità non esauriscono le continuazioni né le interpretazioni sintattiche.

## Dove cade l’effetto nella continuazione effettiva?

La seconda misura usa le prime tre parole della proposizione principale originale. Si interviene sempre sul nome prima della disambiguazione e si salvano i punteggi di ogni token. Questa analisi secondaria riguarda parole effettive come “received unwelcome attention”, non necessariamente `was`. Non è la dimostrazione di un passaggio da interpretazione errata a corretta.

| Posizione | Patch, nats [IC95%] | Vantaggio lessicale | Vantaggio casuale |
|---|---|---|---|
| Parola 1 | 3.815 [1.974, 5.605] | 3.106 [1.581, 4.446] | 4.009 [2.186, 5.790] |
| Parola 2 | 0.027 [-0.024, 0.098] | 0.093 [0.022, 0.169] | 0.068 [0.012, 0.139] |
| Parola 3 | -0.018 [-0.057, 0.021] | -0.008 [-0.033, 0.011] | -0.021 [-0.063, 0.020] |

## Stabilità tra famiglie e tra direzioni

| Famiglia di test | Patch | Lessicale | Casuale |
|---|---|---|---|
| began | +8.156 | +0.953 | +0.651 |
| called | +4.938 | -0.562 | -0.161 |
| left | +1.266 | +0.766 | +0.380 |
| lost | +9.734 | +1.078 | -0.172 |
| read | +6.219 | -0.359 | -0.359 |
| signed | +9.281 | +0.383 | -0.130 |

| Direzione, posizioni allineate | Interazione patch, nats [IC95%] |
|---|---|
| forward | 6.776 [4.240, 8.948] |
| reverse | 6.422 [3.911, 8.542] |

## Verifiche e limiti

- Identità: errore massimo 0 nats.
- Scarto relativo delle norme dopo arrotondamento BF16: mediana 0.035%, massimo 6.958%.
- Nel contrasto a posizioni allineate lo scarto massimo è 2.491%; le norme non sono esattamente identiche.
- Verificati 3.456 effetti grezzi, 288 baseline, le aggregazioni per famiglia e gli hash dello snapshot.
- Nessuna selezione di layer sui dati V3 e nessuna esclusione di frasi in base ai risultati.
- Layer e checkpoint sono fissi; il confronto non verifica una legge di scala.
- I verbi di controllo e alcuni nomi ricorrono tra split; sono disgiunte le famiglie del verbo opzionale.
- Le nuove frasi sono pubbliche e potrebbero essere state viste in addestramento.
- La virgola è un indizio sintattico ma anche un token con proprie regolarità; patchare l’intero residuo non identifica una feature di sintassi.
- Non abbiamo addestrato nuove sonde dei ruoli né dimostrato rianalisi online.

## Prossima decisione

Prima di cercare feature SAE o scalare a modelli grandi, verificare il significato linguistico del contrasto su nuovi materiali e collegarlo a un compito comportamentale sui ruoli. La generalizzazione delle sonde di V2 resta un problema separato. Dopo questa ispezione, i dati V3 non sono più una conferma intatta per un protocollo modificato.

## Riproduzione

Checkpoint `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29`. Protocollo e sorgenti originali sono in `source.json`; misure grezze in `observations.jsonl`, riepiloghi in `results.json` e audit in `audit.json`.

```bash
uv run --no-project --with numpy --with matplotlib --with markdown python v3/analyze.py v3/results/npz-v3-20261008-01
```
