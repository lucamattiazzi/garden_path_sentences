# V3 · Il controllo di comprensione su modelli più grandi

**Gemma 3 12B supera il criterio prefissato.** È il primo checkpoint della sequenza pianificata a riuscirci; la selezione si ferma qui.

## Risultati sulle stesse frasi semplici

| Modello | Attiva: agente | Attiva: paziente | Passiva: agente | Passiva: paziente | Gate |
|---|---:|---:|---:|---:|---|
| 1B | 50.0% | 50.0% | 0.0% | 66.7% | fallito |
| 4B | 100.0% | 50.0% | 33.3% | 100.0% | fallito |
| 12B | 100.0% | 100.0% | 100.0% | 100.0% | passato |

La tabella misura **entrambe le inversioni corrette**, con sei coppie per cella. Soglia congelata: almeno 75% in tutte e quattro le celle (quindi almeno 5/6 coppie). Le percentuali per singola frase sono nel grafico e nei dati.

## Disegno e controlli

Abbiamo riusato esattamente i 96 prompt, le etichette, le quattro dimostrazioni e il punteggio del precedente test diretto sui nomi: nessuna opzione A/B, nessuna ricerca di prompt, nessuna normalizzazione per lunghezza. Sequenza fissata prima della misura: 4B, poi 12B soltanto se 4B fallisce; arresto al primo successo. Il precedente 1B è riportato come riferimento, non è stato rieseguito. Tutto resta sullo sviluppo.

Le risposte complete sono valutate mediante la somma dei log-punteggi dei token dopo `Answer: the`. La verifica senza frase usa prompt identici e risposte corrette opposte quando scambiamo i partecipanti; controlliamo punteggi identici e margini opposti. La scelta tra due candidati non equivale a generazione libera.

| Modello | Accuratezza complessiva sulle frasi semplici | Token per candidato | Coppie con lunghezze diverse | Mediana massa dei due candidati |
|---|---:|---|---:|---:|
| 1B | 66.7% | [2] | 0/96 | 95.9% |
| 4B | 85.4% | [2] | 0/96 | 98.2% |
| 12B | 100.0% | [2] | 0/96 | 95.3% |

La massa è condizionata al prefisso comune, non è una probabilità di risposta libera. I dettagli per famiglia e gli intervalli bootstrap descrittivi (10.000 ricampionamenti) sono in `comparison.json`; le inversioni non sono osservazioni indipendenti.

## Interpretazione e limite del passo successivo

Il controllo identifica un candidato utilizzabile per progettare il ponte causale. Non abbiamo ancora misurato se la virgola cambia la comprensione o se una patch prima della disambiguazione modifica le risposte. Il prossimo protocollo deve includere entrambe le direzioni, controlli lessicali/casuali e posizioni allineate, congelati prima di interrogare il test.

Sono dati di sviluppo riutilizzati dopo gli esperimenti precedenti: la selezione non è conferma indipendente. I checkpoint differiscono anche per architettura e addestramento; non attribuiamo causalmente le differenze al solo numero di parametri. Le frasi sintetiche non hanno validazione umana. Nessun risultato qui prova rianalisi sintattica o identifica un circuito dei ruoli. I risultati predittivi di V3 rimangono distinti.

## Esecuzione e audit

| Modello | Run | GPU | Secondi nella funzione | Picco memoria GPU |
|---|---|---|---:|---:|
| 1B | `direct-v3-20261008-01` | NVIDIA A10G | 15.2 | non registrato |
| 4B | `scale-4b-20261009-02` | NVIDIA A10 | 29.9 | 16.3 GiB |
| 12B | `scale-12b-20261009-01` | NVIDIA A100-SXM4-80GB | 63.1 | 45.8 GiB |

Durate comprensive del caricamento, escluso provisioning; non sono una fattura. FP32, TF32 disabilitato, attenzione eager. Modelli grandi caricati con il wrapper multimodale completo, senza input immagine; nessun peso mancante o inatteso. Sorgenti e revisioni sono salvati nei metadati. Audit indipendente di punteggi, gate, ordine di selezione e identità dei materiali superato.

Il tentativo `scale-4b-20261009-01` aveva completato le misure ma fallito il salvataggio JSON dei metadati: rimane archiviato come incompleto. Il problema è stato corretto e il 4B rieseguito senza cambiare il protocollo. Punteggi grezzi identici tra i due tentativi: **True**. Il tentativo incompleto non è una replica scientifica aggiuntiva.

Protocollo: [PROTOCOL.md](../../PROTOCOL.md). Riferimenti: [Gemma 3 4B pretrained](https://huggingface.co/google/gemma-3-4b-pt), [Gemma 3 12B pretrained](https://huggingface.co/google/gemma-3-12b-pt), [documentazione Transformers](https://huggingface.co/docs/transformers/model_doc/gemma3).
