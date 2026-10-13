# Garden path: perimetro, significato delle misure e domanda ancora aperta

## Qual è l'ultimo punto della ricerca ancora centrato sui garden path?

### Takeaway
L'ultimo esperimento causale completato direttamente nel filone GPS è **V3 / bridge12**, eseguito il 9 ottobre 2026. V4 originale conserva un disegno GPS, ma si arresta alla verifica dello strumento: va raccontata come chiusura metodologica, non come ulteriore risultato causale sui garden path.

### Cited Findings
- V3 misura il contrasto predittivo `log P(was) − log P(,)` su 24 adattamenti NP/Z da materiali pubblicati, con 12 famiglie verbali divise 6 sviluppo / 6 test. L'effetto causale primario sottrae l'effetto della virgola con verbi intransitivi a quello con verbi opzionalmente transitivi, allinea gli ID di posizione e mantiene fisso il blocco 17 di Gemma 3 1B. — [Protocollo V3](../../v3/PROTOCOL.md)
- V3 / bridge12 trasferisce l'indagine a Gemma 3 12B pretrained e a 12 famiglie sintetiche, 6 sviluppo / 6 test. Modello e blocco 32 sono fissati prima delle misure; il blocco deriva da una regola di profondità relativa, non da una selezione del massimo sul 12B. Il secondo controllo di competenza riguarda le domande in frasi semplici e con virgola. — [Protocollo bridge12](../../v3/bridge12/PROTOCOL.md)
- Sul test bridge12, l'effetto cross predittivo è 8,069 nats [5,338; 10,582], mentre il margine QA corretto-meno-errato cambia di −0,074 [−0,132; −0,021]. Non è corretto chiamare nullo il secondo effetto. Nessuna delle 48 patch cross cambia la risposta preferita. La scomposizione agente −0,463 / paziente +0,315 è post-hoc. — [Report bridge12](../../v3/bridge12/results/bridge12-20261009-01/report.md)
- V4 originale mantiene il prefisso `While the guard watched[,] ...`, le condizioni virgola/gap e patch bidirezionali sulla prima persona della principale. Incrocia prima menzione, soggetto dell'evento e agente semantico mediante attive, passive e due relative. Il contrasto `was`/virgola diventa solo accessorio e controfattuale per le relative. — [Protocollo V4](../../v4/PROTOCOL.md)
- In V4, 12B e 27B falliscono entrambi il controllo di competenza sulle domande agente della relativa passiva: 0/6 famiglie corrette in entrambe le inversioni. Ognuno completa 96 baseline di sviluppo; nessuna inferenza di conferma e nessuna patch vengono eseguite. — [Risultati V4](../../v4/README.md)
- La successiva analisi esplorativa degli stessi dati bridge12 trova uno spostamento comune verso il primo partecipante di 0,389 per la patch della virgola, 0,366 per il controllo lessicale e 0,392 per quello casuale. I confronti appaiati hanno intervalli ampi che includono zero. È una rianalisi degli stessi dati, non una replica. — [Risultati V4 e rianalisi V3](../../v4/README.md)

### Inferences
- Titolo/conclusione consigliata: **«Garden-path sentences: possiamo spostare le aspettative del modello, ma non abbiamo ancora isolato il meccanismo della reinterpretazione»**. È più fedele delle formule “abbiamo trovato il layer della sintassi” o “il modello non rianalizza”.
- Confine editoriale suggerito: risultati V1–V3 e bridge12 nel corpo; breve epilogo V4 originale sul gate fallito e sulla rianalisi non specifica dei margini. Le validazioni V4 senza prefisso ambiguo, V5 e V6–V9 possono essere riconosciute in una sola nota come filone separato, senza presentarne i risultati come progresso GPS.
- Il numero della versione non è il criterio: V4 nasce ancora da una domanda GPS, ma la sua inferenza effettiva non raggiunge mai il confronto ambiguo/disambiguato. Per evitare ambiguità indicare entrambi: “ultimo esperimento causale GPS completato: bridge12; ultimo tentativo GPS: V4 originale, fermato al gate”.

### Gaps
- Non esiste in questi protocolli una misura causale completata della revisione *dopo* la parola disambiguante.
- Né la provenienza pubblicata di V3 né la generazione sintetica di bridge12 equivale a una nuova validazione umana delle frasi esatte usate.

## Che cosa sono i garden path e che cosa misurano davvero gli esperimenti?

### Takeaway
Un garden path riguarda il rapporto tra un'interpretazione iniziale plausibile e informazione successiva che la rende inadeguata. Preferire una continuazione, rispondere a una domanda dopo la frase e dimostrare un meccanismo di reinterpretazione sono tre osservazioni diverse.

### Cited Findings
- Hanna e Mueller studiano ambiguità temporanee NP/Z, NP/S e MV/RR; il contrasto virgola/`was` è la loro operazionalizzazione predittiva per NP/Z. Avvertono che due token non coprono tutte le continuazioni possibili. In Pythia-70m e Gemma-2-2B identificano feature sintattiche insieme a euristiche, con indizi di entrambe le letture. Le feature predittive spiegano poco della QA successiva; l'effetto non è esattamente zero e il lavoro non esclude definitivamente circuiti di rianalisi. Questi risultati appartengono ai loro modelli e metodi, non sono risultati del nostro Gemma 3. — [Hanna & Mueller 2025, §§4–7 e appendice H](https://aclanthology.org/2025.naacl-long.164.pdf)
- L'esempio effettivo derivabile dai materiali bridge12 è `While the guard watched the poet was greeted by the chef.` La variante non ambigua inserisce una virgola dopo `watched`. La domanda agente è `Who greeted someone?`, la domanda paziente `Who was greeted?`: riguardano l'evento `greeted`. — [Materiali e generatore](../../v3/bridge/materials.py); [Costruzione del prompt bridge12](../../v3/bridge12/experiment.py)
- In bridge12 si sostituisce il residuo completo sull'ultimo token del secondo sintagma nominale, prima di `was`. Le risposte candidate sono stringhe complete, valutate mediante probabilità teacher-forced; non sono risposte generate liberamente. Il protocollo controlla l'invarianza degli stati precedenti rispetto ai candidati futuri. — [Protocollo bridge12](../../v3/bridge12/PROTOCOL.md)
- Il `gap` lascia invariati i token senza virgola ma sposta di uno gli ID di posizione dalla seconda NP in poi. È un controllo artificiale sulla codifica posizionale; non ricrea l'attenzione al token virgola e non elimina ogni differenza tra le condizioni. — [Protocollo V3](../../v3/PROTOCOL.md)
- I risultati dipendono dalla misura e dalla corruzione scelte: Zhang e Nanda mostrano che scelte diverse nell'activation patching possono dare localizzazioni diverse. I loro risultati motivano controlli e definizioni precise dell'endpoint, non invalidano automaticamente ogni uso di rumore casuale come riferimento. — [Zhang & Nanda, ICLR 2024](https://arxiv.org/html/2309.16042v2)
- Heimersheim e Nanda distinguono patch clean→corrupt e corrupt→clean: sufficienza e necessità condizionate all'intervento non sono equivalenti. Le proprietà identificate dipendono dalle differenze tra i prompt; componenti ridondanti o interazioni possono occultare effetti individuali. — [Heimersheim & Nanda 2024](https://arxiv.org/html/2404.15255v1)

### Inferences
- Spiegazione principiante sull'esempio bridge12: leggendo fino a `the poet`, è plausibile trattare il poeta come oggetto di “guardava”; l'arrivo di `was greeted` richiede che il poeta sia soggetto della principale. La virgola prima del poeta anticipa il confine. Questo è un esempio didattico di struttura, non una nuova misura di difficoltà negli esseri umani.
- Dire “il modello credeva che…” è una scorciatoia narrativa da qualificare: il dato è una preferenza relativa tra due continuazioni, non l'osservazione diretta di un'unica analisi sintattica discreta.
- L'esito bridge12 è una dissociazione: grande spostamento di un contrasto predittivo, piccolo spostamento medio del margine QA con segni opposti per domanda, nessun cambio di risposta. Non è “effetto presente contro effetto assente” e non è una misura comune espressa su due scale intercambiabili.
- La QA bridge12 interroga chi saluta/è salutato nella principale. Non chiede direttamente se il poeta sia stato interpretato come oggetto del verbo precedente `watched`, né dimostra che quell'aggancio iniziale sia stato abbandonato. Una risposta corretta dopo l'intera frase può essere ottenuta leggendo la principale, indipendentemente dal destino dell'aspettativa iniziale.
- Una patch dello stato *prima* di `was` interviene sull'informazione disponibile prima della disambiguazione. Non isola il calcolo che incorpora `was`, e per causalità autoregressiva quel token passato non può aggiornarsi retroattivamente quando arriva la parola futura. Per studiare l'aggiornamento occorre osservare anche posizioni successive che possono accedere alla nuova evidenza.
- Sostituire l'intero residuo trasferisce un insieme di informazioni. Un effetto non identifica una singola feature “sintassi”; il blocco efficace è un punto dove l'intervento ha influenza, non necessariamente dove nasce o risiede esclusivamente l'informazione.
- Un controllo casuale con stessa norma controlla parte dell'ampiezza della perturbazione, non equivale a una lettura sintattica alternativa né costruisce automaticamente una distribuzione nulla di tutte le perturbazioni possibili.
- Il dato QA stabile non prova assenza di comprensione: l'informazione può essere ricostruita dai token successivi, distribuita altrove, non necessaria per la domanda scelta oppure lasciata intatta dall'intervento specifico. Si tratta di spiegazioni compatibili, non di meccanismi già dimostrati.
- L'analisi V3 opzionale-meno-intransitivo è più specifica del semplice contrasto comma/gap del bridge12. Le ampiezze 6,599 e 8,069 non costituiscono una curva di scaling: cambiano modello, precisione, materiali, prompt ed estimand.

### Gaps
- Nessun dato qui mostra che il modello adotti una sola interpretazione alla volta.
- Nessun dato dimostra un processo cognitivamente equivalente alla rilettura umana.
- Manca un readout validato che distingua direttamente aggancio del sintagma nominale al verbo subordinato e nuova funzione nella principale, evitando di confondere “non implica che” con “implica che non”.

## Quale prossimo esperimento resterebbe davvero centrato sui GPS?

### Takeaway
Il passo successivo sensato è seguire la disambiguazione nella stessa famiglia di frasi e intervenire nei punti in cui l'informazione nuova può modificare l'interpretazione. Non serve prima dimostrare un codice generale per agente e paziente.

### Cited Findings
- V3 già separa la previsione del primo disambiguatore dalle due parole successive: l'interazione allineata della continuazione è 3,815 [1,974; 5,605] nats sulla prima parola, 0,027 e −0,018 sulle successive. Questo localizza l'effetto predittivo misurato ma non osserva direttamente il cambiamento d'analisi. — [Risultati V3](../../v3/README.md)
- Bridge12 prescrive soglia di competenza prima del test, due direzioni, identità, controllo lessicale e tre perturbazioni casuali appaiate; analizza le famiglie come unità di ricampionamento. Questi vincoli sono una base da conservare per evitare di ottenere un risultato solo cambiando il compito a posteriori. — [Protocollo bridge12](../../v3/bridge12/PROTOCOL.md)
- V4 dichiara i gate falliti come endpoint pubblicabili, senza conferma né patch. Il rapporto deve preservare la distinzione tra “non testato perché lo strumento fallisce” e “testato senza effetto”. — [Protocollo V4](../../v4/PROTOCOL.md); [Risultati V4](../../v4/README.md)

### Inferences
Proposta di ricerca, non protocollo già eseguito:

1. **Materiali nuovi e verificati.** Coppie ambigue/non ambigue con stesso contenuto finale, varianti con virgola e controllo verbale. Separare sviluppo e conferma per famiglie; far controllare naturalezza e interpretazioni delle frasi esatte. Non dichiarare nuovi test le famiglie già viste.
2. **Tre momenti della frase.** Prima del disambiguatore; al disambiguatore; una o più posizioni successive predefinite. Nel linguaggio tecnico distinguere la posizione dello stato catturato dal token la cui probabilità viene misurata.
3. **Comportamento prima del meccanismo.** Verificare che il modello presenti il contrasto predittivo atteso, che comprenda le versioni non ambigue e che il readout finale distingua le conseguenze della reinterpretazione. Se manca uno dei presupposti, fermare il filone causale e riportarlo.
4. **Domande pertinenti all'aggancio iniziale.** Testare conseguenze della lettura oggetto versus nuovo soggetto, mantenendo anche una domanda di controllo sulla principale. Una risposta yes/no che pretende di inferire la negazione di un evento soltanto perché non è menzionato sarebbe un cattivo strumento: usare domande/parafrasi con gold linguisticamente difendibile.
5. **Interventi prima e dopo l'evidenza.** Confrontare patch a siti/layer prefissati, entrambe le direzioni, allineamento documentato, identità e perturbazioni di riferimento. Se vengono scelti siti sullo sviluppo, congelarli prima della conferma. Un effetto dopo `was` sulla domanda pertinente sarebbe un passo verso la dinamica di reinterpretazione; non prova da solo un algoritmo umano.
6. **Criterio di successo limitato.** Il risultato ricercato è che una modifica di stati nel momento dell'aggiornamento trasferisca in modo selettivo una conseguenza dell'interpretazione, oltre i controlli, su nuove famiglie. Non basta aumentare `P(was)` né cambiare il nome preferito in una frase non ambigua.

Per il rapporto a tre livelli, la stessa conclusione può essere spiegata con tre profondità senza cambiare contenuto epistemico:
- **Livello 1:** una frase mostrata a tappe, il ruolo della virgola, cosa significa “spostare un'aspettativa”, ciò che ancora non sappiamo.
- **Livello 2:** confronto dei disegni V1/V2/V3, perché servono gap e controlli intransitivi, perché una domanda finale deve essere validata, esito bridge12 e arresto V4.
- **Livello 3:** formule degli estimand, ordine temporale e siti di patch, differenze di precisione e checkpoint, split di famiglie, bootstrap e suoi limiti, controlli numerici, cronologia gate→test, limiti di generalizzazione e fonti dei dati.

### Gaps
- Non sono stati fissati in queste note nuovi materiali, layer, numerosità o criteri numerici: richiedono un protocollo distinto e nessuna nuova inferenza è stata lanciata.
- Il candidato esperimento non consente già di promettere un risultato positivo. Un fallimento ben controllato sarebbe comunque informativo sulla combinazione specifica di readout, siti e modello.
