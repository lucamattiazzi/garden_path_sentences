# L’ultimo stato del filone garden-path: V3 e limite della V4 originale

## 1. Qual è il confine corretto del report?

### Takeaway

L’ultimo **risultato causale completato ancora direttamente collegato ai garden-path** è il ponte V3 su Gemma 3 12B, eseguito il 9 ottobre 2026. L’ultimo **tentativo di estensione ancora inserito nel contesto garden-path** è la V4 originale, anch’essa del 9 ottobre: mantiene la subordinata iniziale e il contrasto con/senza virgola, ma si arresta sul controllo di competenza, senza arrivare a nuove patch o al test di conferma. Il report dovrebbe chiudere qui, escludendo le successive V4 validation/paraphrase/targeted e V5–V9 come risultati GPS. — [Protocollo bridge12](../../v3/bridge12/PROTOCOL.md); [Protocollo V4 originale](../../v4/PROTOCOL.md); [Risultati V4](../../v4/README.md)

### Cited Findings

- V3 1B misura un’interazione fattoriale su preferenze di continuazione NP/Z, con materiali pubblicati adattati; bridge12 conserva un contesto ambiguo seguito da una principale passiva e misura previsione e risposta finale negli stessi forward. I due protocolli non hanno però lo stesso estimando: bridge12 non ha il controllo intransitivo. Non vanno presentati come due repliche dello stesso identico effetto né come legge di scala. — [V3 protocollo](../../v3/PROTOCOL.md); [bridge12 protocollo](../../v3/bridge12/PROTOCOL.md)
- V4 originale costruisce attive, passive, relative oggettive e relative passive dopo una subordinata del tipo «While the guard watched[,] ...». L’intento è separare primo partecipante, soggetto dell’evento interrogato e agente semantico. La principale delle relative mantiene come soggetto il primo partecipante: il disegno non separa primo menzionato da ogni possibile significato di soggetto grammaticale. — [V4 protocollo](../../v4/PROTOCOL.md); [Generatore materiali V4](../../v4/materials.py)
- Prima di nuove misure causali, ciascun modello deve ottenere entrambe le inversioni corrette in almeno cinque famiglie su sei, in ciascuna delle otto celle costruzione × domanda. Il 12B fallisce, quindi secondo la sequenza fissata viene provato il 27B; anche questo fallisce. Entrambi totalizzano zero famiglie su sei nelle domande sull’agente della relativa passiva. Nessun dato di conferma né intervento è raccolto. — [Audit V4, runs](../../v4/results/comparison-20261009/audit.json)
- **Non confondere 0/6 coppie corrette con 0% di singole risposte corrette**: nella cella critica il 27B ha 25% di accuratezza per singola risposta, ma nessuna famiglia con entrambe le inversioni corrette. Il 12B ha invece 0% anche per singola risposta. — [Audit V4, accuracy e gate](../../v4/results/comparison-20261009/audit.json)
- La V4 contiene anche una rianalisi post-hoc dei dati già raccolti nel ponte 12B. È rilevante al report GPS perché corregge l’interpretazione dello stesso esperimento, non perché aggiunga una nuova replica. Non va nascosta solo per mantenere il confine di versione. — [Audit V4, legacy](../../v4/results/comparison-20261009/audit.json)

### Inferences

- Formula editoriale consigliata: **«Stato del filone garden-path: il segnale predittivo è causalmente modificabile; il collegamento con la reinterpretazione finale resta da dimostrare.»**
- Titolo o sottotitolo non devono dire «abbiamo trovato il circuito della rianalisi» o «il layer della sintassi». Il risultato forte è più circoscritto ma reale.
- L’estensione V4 originale va presentata come ultimo tentativo e limite operativo, non come conferma negativa dell’esistenza di ruoli interni. La derivazione verso l’analisi logica generale inizia quando si semplificano e validano quelle costruzioni separatamente dai garden-path.

### Gaps

- Non esiste qui una dimostrazione causale completa di passaggio da interpretazione iniziale errata a interpretazione corretta dopo il disambiguatore.
- Nessuna validazione umana raccolta per gli esatti adattamenti V3 o per i nuovi materiali sintetici bridge12/V4. Le frasi di provenienza pubblicata non convalidano automaticamente le trasformazioni locali.

## 2. Che cosa dimostra esattamente V3 1B?

### Takeaway

Su sei famiglie di test, una patch del residuo prima della disambiguazione cambia la preferenza fra `was` e virgola con un’interazione opzionale meno intransitivo di **6,599 nats [4,065; 8,734]**, anche quando i position ID comuni sono allineati. L’effetto supera i controlli lessicali e casuali e si concentra sulla prima parola della continuazione effettiva. Questo sostiene causalità sulla previsione locale, non una misura diretta del parse o della sua riparazione. — [Report V3](../../v3/results/npz-v3-20261008-01/report.md)

### Cited Findings

**Disegno e campione.** 24 materiali provenienti dal repository di Hanna e Mueller, a commit fissato, organizzati in 12 famiglie del verbo opzionale, due materiali per famiglia. Sei famiglie sviluppo e sei test; la separazione riguarda il verbo opzionale, non ogni nome o verbo di controllo. Gemma 3 1B pretrained, revisione `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29`, BF16, blocco 17 zero-based fissato da V2; nessuna ricerca del layer nei dati V3. Una sola posizione: ultimo token del nome prima del verbo disambiguante. — [Protocollo V3](../../v3/PROTOCOL.md); [Materiali effettivi](../../v3/materials/npz.json)

**Endpoint.** `log P(was) − log P(virgola)`, con ciascuna alternativa verificata come esattamente un token aggiunto. Positivo significa preferenza relativa per `was`; negativo per virgola. Non è una probabilità di due parse esaustivi. La quantità non va chiamata accuratezza sintattica. — [Protocollo V3](../../v3/PROTOCOL.md)

**Controllo posizionale.** `native` conserva il prefisso originario senza virgola. `comma` inserisce la virgola prima del secondo gruppo nominale. `gap` conserva gli stessi token di native, ma incrementa di uno i position ID dal secondo gruppo nominale in avanti. Una verifica controlla che, togliendo il token virgola dalla condizione comma, token e posizioni coincidano con gap. Non viene aggiunto un token vuoto o invisibile. Questa manipolazione artificiale controlla un preciso scostamento posizionale, non il contributo della virgola all’attenzione né tutti i suoi correlati lessicali. — [Protocollo V3](../../v3/PROTOCOL.md)

**Baseline di test: dodici materiali, sei famiglie per cella.** — [Risultati V3, baselines](../../v3/results/npz-v3-20261008-01/results.json)

| Classe | Condizione | Margine medio | Materiali che preferiscono was |
|---|---|---:|---:|
| Opzionale | native | −5,729 | 0/12 |
| Opzionale | gap | −4,958 | 1/12 |
| Opzionale | comma | +3,229 | 12/12 |
| Intransitivo | native | +1,161 | 10/12 |
| Intransitivo | gap | +2,172 | 10/12 |
| Intransitivo | comma | +3,641 | 12/12 |

**Interazione opzionale meno intransitivo**, media di entrambe le direzioni dopo inversione del segno della direzione reverse. — [Risultati V3, summaries](../../v3/results/npz-v3-20261008-01/results.json)

| Contrasto | Patch cross | Lessicale | Casuale |
|---|---:|---:|---:|
| Virgola / native | 6,437 [3,531;9,008] | 0,339 [−0,089;0,734] | 0,374 [0,245;0,519] |
| Virgola / gap | 6,599 [4,065;8,734] | 0,376 [−0,141;0,849] | 0,035 [−0,224;0,335] |
| Gap / native | −0,141 [−0,898;0,648] | −0,016 [−0,073;0,044] | 0,106 [0,062;0,174] |

- Nel contrasto allineato il vantaggio appaiato sul lessicale è **6,223 [3,706;8,148]**, sul casuale **6,564 [3,957;8,786]**. Tutte e sei le famiglie hanno una media cross positiva. Le due direzioni danno **6,776 [4,240;8,948]** e **6,422 [3,911;8,542]**, con segno già allineato. — [Report V3](../../v3/results/npz-v3-20261008-01/report.md); [Audit valori per famiglia](../../v3/results/npz-v3-20261008-01/audit.json)
- La variazione è distribuita fra aumento di log P(was), **4,291 [2,529;5,911]**, e diminuzione di log P(virgola), **−2,308 [−3,250;−1,291]**. La differenza del secondo va sottratta. — [Risultati V3, summaries](../../v3/results/npz-v3-20261008-01/results.json)
- La traiettoria secondaria usa le prime tre parole **effettive** della continuazione pubblicata, non necessariamente `was`. Effetti cross: prima parola **3,815 [1,974;5,605]**, seconda **0,027 [−0,024;0,098]**, terza **−0,018 [−0,057;0,021]**. Ogni parola successiva è condizionata alle precedenti tramite teacher forcing; il risultato non prova l’assenza di ogni effetto successivo, né mostra da solo una riparazione del parse. — [Report V3](../../v3/results/npz-v3-20261008-01/report.md)
- 288 baseline e 3.456 interventi. Identità: errore zero. Controlli lessicale e tre direzioni gaussiane abbinati alla norma cross, ma conversione BF16 rende il matching approssimativo: massimo scarto relativo 2,491% nel contrasto allineato, 6,958% complessivo. — [Audit V3](../../v3/results/npz-v3-20261008-01/audit.json); [Risultati numerical_checks](../../v3/results/npz-v3-20261008-01/results.json)
- Intervalli descrittivi bootstrap percentile: sei famiglie, 10.000 ricampionamenti, seed zero. Non trattare migliaia di patch o dodici frasi come migliaia/dodici unità linguistiche indipendenti; nessuna correzione globale per molteplici confronti. — [Protocollo V3](../../v3/PROTOCOL.md)

### Inferences

- Il pattern opzionale × virgola è coerente con una sensibilità alla struttura del prefisso; il confronto intransitivo rende l’interpretazione più forte di una semplice sensibilità generale alla punteggiatura.
- Il controllo gap impedisce di spiegare **tutto** l’effetto con quel preciso offset dei position ID; non elimina ogni confondente legato al token virgola.
- «Il layer 17 conta per questa misura» è sostenibile. «Il modello fa sintassi soltanto lì» non lo è, perché il residuo contiene informazione accumulata e lo studio non esegue una mappatura esaustiva del circuito.

### Gaps

- Il risultato non distingue esplicitamente una rappresentazione del parse da una combinazione di correlati predittivi del prefisso.
- Materiali pubblici potenzialmente presenti nel training; provenienza documentata non significa test incontaminato.
- Incertezza sui sei cluster non include scelta iniziale della metrica, del checkpoint, del layer o dei materiali.

## 3. Che cosa aggiunge il ponte 12B e perché V4 si ferma?

### Takeaway

Il ponte 12B mette nello stesso esperimento la previsione immediata prima del disambiguatore e la successiva risposta sull’evento. La previsione cambia molto; il margine QA cambia poco in media ma in modo non nullo, con effetti opposti per agente e paziente e **zero scelte invertite su 48 patch cross di test**. La successiva V4 originale non supera la competenza necessaria a separare le interpretazioni del segnale QA. — [Report bridge12](../../v3/bridge12/results/bridge12-20261009-01/report.md); [Risultati V4](../../v4/README.md)

### Cited Findings

**Selezione dello strumento.** Le verifiche di sviluppo su domande dirette danno frazioni di famiglie con entrambe le inversioni corrette: 1B attiva-agente 3/6, attiva-paziente 3/6, passiva-agente 0/6, passiva-paziente 4/6; 4B rispettivamente 6/6, 3/6, 2/6, 6/6; 12B 6/6 in tutte le quattro celle. Il 12B è il primo a superare il gate prefissato (almeno 5/6 per cella); non è una prova che la sola taglia causi il miglioramento, perché i checkpoint differiscono anche per architettura/addestramento. Il 1B è un risultato preesistente; la sequenza nuova era 4B poi 12B se necessario. — [Confronto modelli](../../v3/scale/results/comparison-20261009/report.md)

**Secondo gate contestuale.** La buona prestazione sulle frasi semplici non bastava: bridge12 richiede almeno 5/6 famiglie con entrambe le inversioni corrette per ciascun ruolo sia su frasi semplici sia su quelle con virgola. Passa con 5/6 per comma/agente e 6/6 nelle altre tre celle. Si usa Gemma 3 12B pretrained, revisione `295efb63d01a7017928f273a94ebb86105c9526f`, FP32, no TF32, attenzione eager; blocco 32 zero-based scelto per profondità relativa dal 17 del 1B, non ottimizzato su questo test né dichiarato omologo. — [Protocollo bridge12](../../v3/bridge12/PROTOCOL.md); [Gate salvato](../../v3/bridge12/results/bridge12-20261009-01/results.json)

**Materiali.** Dodici frame sintetici: sei sviluppo e sei test, due ordini dei partecipanti, domande agente e paziente, principale passiva. Esempio illustrativo del formato generato: «While the guard watched[,] the poet was greeted by the chef.» Per il predicato greeted, chef è l’agente e poet il paziente. Quattro dimostrazioni fisse, risposte candidate complete nome + punto dopo `Answer: the`; non generazione libera. I frame riusano alcuni predicati principali fra split; nessuna validazione umana. — [Generatore](../../v3/bridge/materials.py); [Prompt bridge12](../../v3/bridge12/experiment.py)

**Intervento.** Ultimo token del secondo gruppo nominale, prima di `was`; sostituzione del residuo completo, direzioni clean→gap e gap→clean. Identità, sostituzione dell’attore della subordinata con norma abbinata, tre direzioni gaussiane della stessa norma. Si controllano invarianti del prefisso rispetto alle future risposte candidate e allineamento token/posizioni rimuovendo la virgola. Gli stessi forward producono entrambe le misure. — [Protocollo](../../v3/bridge12/PROTOCOL.md); [Implementazione](../../v3/bridge12/experiment.py)

**Effetti sul test, segno orientato come clean meno gap.** — [Audit bridge12, effects](../../v3/bridge12/results/bridge12-20261009-01/audit.json)

| Endpoint | Cross | Lessicale | Casuale |
|---|---:|---:|---:|
| log P(was) − log P(virgola) | 8,069 [5,338;10,582] | 0,117 [−1,025;1,441] | 0,524 [0,044;0,934] |
| Margine QA corretto − errato | −0,074 [−0,132;−0,021] | 0,104 [0,012;0,190] | 0,163 [0,070;0,245] |

- Vantaggi predittivi cross−lessicale **7,952 [5,567;9,895]**, cross−casuale **7,544 [5,137;9,708]**. Differenze QA cross−lessicale **−0,178 [−0,278;−0,067]**, cross−casuale **−0,237 [−0,306;−0,165]**. Non appiattire tutto in «QA invariata»: i margini variano, le scelte no. — [Audit bridge12](../../v3/bridge12/results/bridge12-20261009-01/audit.json)
- Due direzioni orientate: clean→gap previsione **8,126 [5,326;10,697]**, QA **−0,100 [−0,145;−0,052]**; gap→clean previsione **8,011 [5,350;10,467]**, QA **−0,047 [−0,117;0,019]**. La media non deve nascondere questa asimmetria. — [Audit bridge12, effects](../../v3/bridge12/results/bridge12-20261009-01/audit.json)
- Analisi per ruolo **post-hoc**: cross QA agente **−0,463 [−0,622;−0,326]**, paziente **+0,315 [0,189;0,421]**. Nelle passive, il primo partecipante è paziente e soggetto grammaticale della principale: il pattern è compatibile con un aumento della preferenza per quel partecipante, comune a entrambe le domande. «Compatibile» non significa meccanismo identificato. — [Audit exploratory_role_effects](../../v3/bridge12/results/bridge12-20261009-01/audit.json)
- Rianalisi post-hoc V4 dello stesso bridge12: componente comune verso il primo partecipante cross **0,389 [0,272;0,515]**, lessicale **0,366 [−0,159;1,047]**, casuale **0,392 [0,209;0,582]**. Vantaggi appaiati cross−lessicale **0,023 [−0,697;0,586]**, cross−casuale **−0,003 [−0,234;0,215]**. Nessuna evidenza di vantaggio specifico del comma per quella componente comune; gli intervalli ampi non dimostrano equivalenza. — [Audit V4, legacy](../../v4/results/comparison-20261009/audit.json)
- Sul test la preferenza predittiva di baseline vale native **−5,517**, gap **−4,298**, comma **+3,660**. Eppure le risposte finali gap sono corrette in 12/12 per ciascun ruolo; native agente 12/12, paziente 11/12; comma agente 10/12, paziente 12/12. Preferire virgola prima di `was` può coesistere con identificare correttamente i ruoli dopo la frase completa. È una dissociazione fra misure, non ancora una dimostrazione diretta del processo di riparazione. — [Report bridge12](../../v3/bridge12/results/bridge12-20261009-01/report.md)
- Totali bridge12: 192 baseline, 576 interventi, sei famiglie di test. Identità errore zero, massimo errore relativo norm matching **8,56×10⁻⁷**. Fonte, cronologia gate→test, somme token ed effetti grezzi verificati. — [Audit bridge12, checks](../../v3/bridge12/results/bridge12-20261009-01/audit.json)
- V4 completata: 96 osservazioni di sviluppo per ciascun modello, **zero conferma e zero patch**. Il tentativo 12B iniziale interrotto da una verifica numerica è conservato; il run completo usa padding mascherato delle risposte per rendere uguale la forma dei forward, senza cambiare soglia o protocollo scientifico. Non contare il tentativo interrotto come ulteriore replica. — [V4 README](../../v4/README.md); [Amendamento protocollo](../../v4/PROTOCOL.md)

### Inferences

- Il contributo più interessante del ponte è separare tre oggetti: aspettativa sul prossimo token, margine delle risposte dopo la frase e scelta finale. Una stessa patch può muovere il primo moltissimo, il secondo modestamente, il terzo per nulla.
- Zero inversioni delle scelte non prova assenza di informazione sintattica nel sito patchato: i margini possono essere robusti, l’informazione ridondante, o ricostruita dai token successivi. Ma non consente di rivendicare che la patch abbia trasferito una reinterpretazione completa.
- Per proseguire restando sui GPS: fissare un nuovo test con misure prima e dopo lo specifico disambiguatore, confronto non ambiguo e intransitivo, outcome comportamentale qualificato prima delle patch, e separazione di sviluppo/conferma. Non ricominciare scegliendo liberamente layer o formulazione sullo stesso test già visto.

### Gaps

- Nessuna nuova inferenza svolta per questo report del 12 ottobre. Ho rieseguito i due audit V3 e bridge12 sui raw; gli oggetti ricostruiti sono esattamente uguali agli audit archiviati. Questo è controllo riproducibilità, non replica GPU.
- Il salto da effetti predittivi a circuito della rianalisi resta aperto. L’ultimo tentativo utile all’obiettivo GPS si arresta onestamente perché lo strumento comportamentale V4 non passa la soglia, non perché abbia dimostrato inesistenza del meccanismo.

Il file pubblico aggregato [evidence.json](../../reports/2026-10-12-gps/evidence.json) contiene sei figure, tabelle aggiuntive, valori a piena precisione, localizzatori delle fonti e hash, senza prompt grezzi, corpus, log o segreti. Le figure sono `figure-v3-interaction`, `figure-v3-trajectory`, `figure-bridge-endpoints`, `figure-bridge-controls`, `figure-bridge-roles`, `figure-v4-gate`.
