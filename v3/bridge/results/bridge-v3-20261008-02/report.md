# V3 · Dalla previsione alle domande sui ruoli

**Il controllo di comprensione è fallito.** Questo formato di domande, con Gemma 3 1B pretrained, non supera il criterio fissato prima della misura. Non abbiamo eseguito patch causali né inferenza sulle famiglie di test.

## Il controllo prima dell'esperimento

Il modello legge quattro esempi, una frase e una domanda sull'agente o sul paziente dell'evento principale. Valutiamo le probabilità delle etichette A/B; non sono risposte generate liberamente. Scambiamo sia i partecipanti sia l'ordine delle opzioni.

Per procedere servivano almeno **75% di domande corrette in entrambi gli ordini**, separatamente per attiva e passiva, su frasi semplici di sviluppo, e una mediana di P(A)+P(B) almeno del 10%. Una risposta corretta in un solo ordine non basta.

| Frasi semplici di sviluppo | Accuratezza per ordine | Corrette in entrambi gli ordini |
|---|---:|---:|
| Attive | 25.0% | 8.3% |
| Passive | 25.0% | 0.0% |

Mediana P(A)+P(B): **84.5%**. Sei famiglie di sviluppo; 24 domande distinte per voce, ciascuna presentata in due ordini.

**Pareggi numerici:** 46/96 sulle frasi semplici. Il criterio congelato li conta come errori. Assegnando invece mezzo punto ai pareggi, l'accuratezza descrittiva complessiva è 49.0%: il 25% rigoroso non va letto come prova di prestazioni sotto il caso. Questa analisi aggiuntiva non cambia il gate.

## Condizioni misurate

| Split | Condizione | Voce | Accuratezza | Entrambi gli ordini | Margine corretto, nats |
|---|---|---|---:|---:|---:|
| development | simple | active | 25.0% | 8.3% | -0.005 [-0.013, 0.005] |
| development | simple | passive | 25.0% | 0.0% | -0.005 [-0.023, 0.010] |
| development | no_context | active | 25.0% | 0.0% | 0.000 [0.000, 0.000] |
| development | no_context | passive | 25.0% | 0.0% | 0.000 [0.000, 0.000] |
| development | native | active | 29.2% | 4.2% | 0.008 [-0.010, 0.029] |
| development | native | passive | 27.1% | 0.0% | -0.008 [-0.023, 0.008] |
| development | gap | active | 35.4% | 4.2% | 0.013 [-0.008, 0.036] |
| development | gap | passive | 31.2% | 12.5% | 0.005 [-0.005, 0.018] |
| development | clean | active | 33.3% | 0.0% | 0.005 [-0.005, 0.013] |
| development | clean | passive | 22.9% | 4.2% | -0.005 [-0.013, 0.005] |

`simple`: solo frase principale; `no_context`: frase assente; `native`: subordinata senza virgola; `gap`: stessa frase con un salto negli ID posizionali; `clean`: virgola. Gli intervalli sono bootstrap descrittivi per famiglia (10.000 campioni), non repliche indipendenti per ordine/domanda.

## Verifica numerica aggiuntiva: FP32

Dopo aver visto i pareggi BF16, abbiamo ripetuto **solo** le 96 frasi semplici e le 96 misure senza contesto di sviluppo, con pesi e calcolo FP32 e TF32 disabilitato. Identici prompt, checkpoint, etichette e ordine. È una diagnosi post-hoc, separata dall'esperimento congelato.

| Frasi semplici FP32 | Accuratezza | Entrambi gli ordini |
|---|---:|---:|
| active | 47.9% | 33.3% |
| passive | 50.0% | 33.3% |

Pareggi FP32 sulle frasi semplici: 0/96. Il gate resta **fallito**. La precisione modifica i punteggi e spiega molti pareggi, ma questo controllo non sostiene una comprensione stabile nel formato scelto. Nessuna patch e nessun dato di test in questa diagnosi. Funzione GPU: 14.3 secondi.

## Cosa cambia rispetto a V3

Il risultato circoscrive **lo strumento di misura**, non prova che il modello ignori i ruoli. Il test causale programmato resta ineseguito. Prima di riprenderlo occorre validare un altro readout o un modello più capace, usando lo sviluppo e conservando il test per la conferma. Non abbassiamo la soglia dopo avere visto il risultato.

Gli effetti predittivi già misurati in V3 restano validi nel loro dominio. Questo seguito non autorizza a trasformarli in una dimostrazione di rianalisi sintattica. Le nuove frasi sono sintetiche, senza validazione umana; il contesto few-shot cambia rispetto ai prefissi nudi di V3. I predicati principali possono ricorrere nei due split. Cambiare voce e ordine esclude scorciatoie basate su un nome fisso, ma non regole superficiali sulla sequenza.

## Riproducibilità

Modello `google/gemma-3-1b-pt`, revisione `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29`, BF16, attenzione eager, NVIDIA A10. Funzione GPU: 22.2 secondi. Baseline: 480; interventi: 0. Stato scientifico: `instrument_gate_failed`. `complete=true` indica il completamento del ramo previsto dal protocollo.

Audit superato: 12 sorgenti verificati, punteggi grezzi e aggregazioni ricalcolati indipendentemente, controllo della sequenza sviluppo → gate → eventuale test. Errore massimo identità: None.

File nella stessa cartella: `metadata.json`, `source.json`, `materials.json`, `observations.jsonl`, `results.json`, `audit.json`, `report.svg` e `report.png`. Protocollo: `../../PROTOCOL.md`.
