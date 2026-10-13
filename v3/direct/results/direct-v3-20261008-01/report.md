# V3 · Rispondere direttamente con il nome

**Il nuovo formato non supera il controllo di comprensione prefissato.** Abbiamo eliminato le etichette A/B e valutato risposte complete con i nomi dei partecipanti. Questa misura riguarda soltanto lo sviluppo, non le famiglie di test.

## Risultati sulle frasi semplici

| Voce | Ruolo richiesto | Accuratezza | Entrambe le inversioni | Coppie riuscite |
|---|---|---:|---:|---:|
| active | agent | 75.0% | 50.0% | 3/6 |
| active | patient | 75.0% | 50.0% | 3/6 |
| passive | agent | 33.3% | 0.0% | 0/6 |
| passive | patient | 83.3% | 66.7% | 4/6 |

La soglia era 75% di coppie corrette **in ciascuna combinazione di voce e ruolo**: almeno cinque famiglie su sei. Una coppia contiene la frase originale e quella con i partecipanti scambiati. Preferire sempre lo stesso nome non basta.

## Che cosa misuriamo

Quattro esempi fissi mostrano risposte espresse con nomi. Il prompt termina con `Answer: the`; confrontiamo, per esempio, ` poet.` e ` chef.` sommando i logaritmi delle probabilità di tutti i token aggiunti, incluso il punto. I candidati non sono elencati nel prompt. Questa è accuratezza di scelta fra due continuazioni, non accuratezza di generazione libera.

Token per risposta completa osservati: [2]. Coppie di candidati di lunghezza diversa: 0/96. Mediana della probabilità congiunta dei due candidati, condizionata al prefisso comune `Answer: the`, sulle frasi semplici: 95.9%. Nessuna normalizzazione per lunghezza né esclusione di item.

Senza frase, i prompt sono identici nelle due inversioni, mentre la risposta corretta cambia: il controllo deve fallire almeno una delle due. L’audit verifica identità dei punteggi e inversione esatta del margine.

## Dettaglio e incertezza

| Condizione | Voce | Ruolo | Margine medio [IC95%], nats | Entrambe le inversioni [IC95%] |
|---|---|---|---:|---:|
| no_context | active | agent | 0.000 [0.000, 0.000] | 0.0% [0.0%, 0.0%] |
| no_context | active | patient | 0.000 [0.000, 0.000] | 0.0% [0.0%, 0.0%] |
| no_context | passive | agent | 0.000 [0.000, 0.000] | 0.0% [0.0%, 0.0%] |
| no_context | passive | patient | 0.000 [0.000, 0.000] | 0.0% [0.0%, 0.0%] |
| simple | active | agent | 0.705 [0.152, 1.282] | 50.0% [16.7%, 83.3%] |
| simple | active | patient | 0.778 [0.481, 1.185] | 50.0% [16.7%, 83.3%] |
| simple | passive | agent | -0.227 [-0.749, 0.199] | 0.0% [0.0%, 0.0%] |
| simple | passive | patient | 0.791 [0.508, 1.146] | 66.7% [33.3%, 100.0%] |

Intervalli bootstrap descrittivi su sei famiglie, 10.000 ricampionamenti; inversioni e domande non sono repliche indipendenti. I risultati per famiglia sono in `audit.json`. La media del margine può nascondere errori: il gate usa entrambe le risposte corrette.

## Diagnosi descrittiva degli errori

Nelle frasi attive, il modello preferisce lo stesso nome nelle domande sull'agente e sul paziente in **6/12** casi. Le due risposte corrette dovrebbero invece essere diverse.

Nelle frasi passive, il modello preferisce lo stesso nome nelle domande sull'agente e sul paziente in **10/12** casi. Le due risposte corrette dovrebbero invece essere diverse.

Questa verifica post-hoc descrive una scarsa sensibilità al ruolo richiesto, soprattutto nelle passive; non identifica il meccanismo interno dell'errore e non cambia la soglia del gate.

## Interpretazione

Il passaggio ai nomi non ha validato lo strumento secondo il criterio stabilito. Non eseguiamo patch né inferenza sul test. Dopo due formati falliti, altre varianti di prompt sullo stesso piccolo checkpoint rischiano di diventare una ricerca opportunistica: la priorità è una verifica di competenza su un modello più capace, con un nuovo protocollo di sviluppo.

Il confronto con A/B non è un confronto sperimentale isolato: cambiano le dimostrazioni, spariscono le opzioni e cambia la metrica di robustezza (inversione dei partecipanti invece dell’ordine delle opzioni). Questo seguito è stato scelto dopo aver visto i risultati precedenti. Le frasi sono sintetiche e prive di validazione umana. Il fallimento non prova assenza di rappresentazioni dei ruoli; il successo potrebbe ancora usare regole superficiali. Gli effetti predittivi già osservati in V3 rimangono distinti da queste domande di comprensione.

## Riproducibilità

Modello `google/gemma-3-1b-pt`, revisione `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29`, FP32, TF32 disabilitato, attenzione eager. 96 osservazioni (192 forward), nessuna patch e nessuna inferenza sul test. Funzione GPU su NVIDIA A10G: 15.2 secondi, escluso provisioning. Audit dei 12 sorgenti, somme dei token, margini, coppie e ricalcolo del gate: verifiche superate.

Protocollo: [PROTOCOL.md](../../PROTOCOL.md). Dati e snapshot nella stessa cartella: `observations.jsonl`, `materials.json`, `source.json`, `metadata.json`, `results.json`, `audit.json`.
