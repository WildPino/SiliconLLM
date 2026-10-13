# Progetto della chat locale

Architettura scelta il 13 ottobre 2026. L'implementazione del
[goal](GOAL.md) è avviata. I test della finestra e del protocollo usano fixture
separate; la verifica sul modello reale attende il runtime CPU approvato.

## Esperienza dell'utente

Una finestra desktop con elenco delle conversazioni a sinistra, selettore del
modello e messaggi al centro, campo di composizione in basso. Le statistiche
rimangono compatte sotto la risposta, con dettagli espandibili.
Il testo cresce senza animazioni decorative, evidenzia il codice e permette
di copiarlo. Invio spedisce il messaggio; Maiusc e Invio aggiungono una riga.
Lo stato distingue caricamento del modello, preparazione del prompt,
generazione, interruzione e errore.

Il catalogo mostra nome, revisione, limiti di contesto e stato di disponibilità.
Un modello non disponibile non avvia download né sostituzioni automatiche.
La selezione di un modello diverso apre una nuova conversazione, per evitare
di alterare implicitamente il significato dei messaggi precedenti.
Un solo invio può essere attivo: un secondo riceve lo stato occupato, senza
accumulare una coda di richieste o avviare altra inferenza.

## Architettura proposta

Finestra C++ con Qt Widgets → adattatore → processo del runtime.

La scelta è un'app compilata con controlli desktop, senza WebView, Qt WebEngine,
server HTTP o runtime JavaScript. C++ gestisce interfaccia, cronologia e
comunicazione. Il runtime scientifico mantiene il proprio processo e build.

- Qt Core, Gui e Widgets, compilati con CMake. Nessun ciclo di rendering continuo
  o animazione decorativa. Aggiornare soltanto testo e zone che cambiano.
- Adattatore con I/O guidato dagli eventi; Qt offre QProcess per avviare e
  comunicare con processi esterni. Lettura e decodifica devono restare separate
  dal rendering se le misure mostrano contesa nel thread della finestra.
- Tokenizer e template devono restare identici al riferimento. Riutilizzare
  un helper Python limitato a queste operazioni se necessario; una migrazione
  nativa richiede parità prima di sostituirlo. Non caricare librerie di training
  per il solo tokenizer. Il runtime finale può avere ulteriori dipendenze
  approvate dalla ricerca: la scelta della UI non ne cambia la qualificazione.
- Un processo di inferenza persistente: caricare il modello una volta evita
  di pagare il caricamento a ogni messaggio.
- Comunicazione binaria per token e stato tra runtime e adattatore; nessun
  server HTTP o serializzazione JSON inseriti nei kernel di calcolo.
- Eventi progressivi locali verso l'interfaccia, con cancellazione separata
  dalla pipe delle richieste; nessun SSE o WebSocket.
- Persistenza delle conversazioni in una directory dati dell'utente, fuori
  dal repository, con scritture atomiche a fine turno o dopo un'interruzione.
  Il testo non viene salvato a ogni token.

Nessuna porta di rete in ascolto. I percorsi degli artefatti provengono dalla
configurazione del catalogo, non da comandi o percorsi forniti dal modello.

Directory proposte: `apps/local-chat/src/`, `apps/local-chat/resources/`,
`apps/local-chat/adapters/`, `tests/local_chat/` e `docs/local-chat/`.
Il nome dell'applicazione rimane quello del progetto.

Qt Widgets offre layout e controlli desktop maturi, riducendo il lavoro rispetto
a costruire manualmente la UI con Win32. Tauri conserva processi WebView:
la scelta nativa elimina quel livello, ma il vantaggio CPU è un'ipotesi da
misurare, non una garanzia. La dimensione dell'installer non misura il costo
durante la generazione. Documentazione:
[Qt Widgets](https://doc.qt.io/qt-6/qtwidgets-index.html),
[QProcess](https://doc.qt.io/qt-6/qprocess.html),
[processi Tauri](https://v2.tauri.app/concept/process-model/).

## Evidenze del motore alla base del progetto

Base esaminata: commit `ecd0ebbdd988dd31256dc1204361e517a64d760a`.
Le osservazioni identificano percorsi diversi; non dichiarano quale sarà il
runtime vincente alla chiusura della ricerca.

| Percorso | Comportamento osservato | Conseguenza |
| --- | --- | --- |
| `benchmarks/phase60/engine.c` | Seleziona diversi entry point tramite macro di compilazione; il percorso generale espone OpenMP e `--threads`. | Il nome del sorgente non identifica da solo il protocollo o il parallelismo. |
| `benchmarks/native_expert_scaling/original_engine_chat_main.c` | Processo persistente, affinità Windows sul processore logico 0, stato riutilizzato tra richieste; risposta scritta dopo tutto il decode. | La modalità `--stream` non è ancora streaming progressivo. Lo stato va isolato tra chat. |
| `benchmarks/native_expert_scaling/chatbot_compact_entry.c` | Imposta sei thread OpenMP; accumula gli ID e li invia a fine generazione. | Non ereditare un conteggio thread fisso senza qualificare questo profilo. |
| `benchmarks/native_expert_scaling/chatbot_hybrid_engine_chat.py` | Lettore dedicato bloccante sulle pipe e coda di risposte; protocollo con ID e logits finali. | È un riferimento da verificare, non un client generico da copiare integralmente. |
| `benchmarks/native_expert_scaling/full_width_native_chat.py` | Percorso recente con componenti Python, affinità e thread espliciti. | L'adattatore finale deve seguire il risultato approvato della ricerca; il progetto UI non presume che ogni risultato sia già un eseguibile C autonomo. |

Il wrapper originale include inoltre `original_wide_prefix.c`, non presente
nel percorso relativo esaminato: ricostruire il build richiede gli artefatti
e le ricevute corretti. Il worktree non contiene i file non tracciati, i pesi
o gli ambienti locali della cartella di ricerca.

## Streaming reale e cancellazione

Il contratto espone capacità dichiarate: risposta progressiva, cancellazione,
riuso dello stato, limite di contesto e modalità di generazione.
L'adattatore non inventa capacità mancanti.

Per lo streaming nativo, proporre un protocollo versionato e separato:
inizio richiesta, ID prodotti, completamento con contatori e tempi, errore.
Conservare il protocollo storico e i suoi consumer di ricerca.
Emettere il primo ID appena disponibile; raggruppare gli altri soltanto per
un intervallo breve, inizialmente 30–50 ms da misurare. La finestra può aggiornare
il testo ogni 50–100 ms senza attendere il completamento della risposta.
La decodifica incrementale deve conservare correttamente Unicode, byte parziali
e token speciali. Il risultato finale deve coincidere con la decodifica canonica.

Stop va implementato fuori dal canale delle richieste bloccante: un comando
su stdin non può essere letto da un runtime che sta ancora facendo decode.
Preferenza per un segnale cooperativo controllato tra passi di generazione;
se non disponibile, arresto e riavvio del solo processo posseduto dal servizio.
Dopo lo Stop, validare o ricostruire lo stato dal transcript effettivo.
Non basta interrompere la visualizzazione lasciando il motore a calcolare.

I limiti di contesto devono essere controllati prima della richiesta. Nessuna
troncatura silenziosa e nessun riassunto automatico nella prima versione.

## Protezione delle prestazioni CPU

C non dispone automaticamente di code più efficienti: contano scheduling del
sistema operativo, thread runnable, cache condivise, banda RAM e trasferimenti.
Anche una finestra nativa può competere con il motore. Un processo ponte in attesa di
I/O non equivale a un core impegnato continuamente; evitare il polling attivo.

La lettura deve drenare stdout e stderr anche quando il rendering rallenta.
Su Windows una pipe piena può bloccare la scrittura del processo produttore:
[operazioni sulle pipe](https://learn.microsoft.com/en-us/windows/win32/ipc/anonymous-pipe-operations).
Prevedere buffer limitati in byte, aggregazione degli aggiornamenti e una
politica esplicita per client troppo lento: interrompere la richiesta e
ricostruire lo stato, senza scartare ID o crescere senza limite in memoria.

Per OpenMP, l'attesa attiva può consumare cicli anche fuori dal lavoro utile.
`OMP_WAIT_POLICY=PASSIVE` è un suggerimento al runtime, non una garanzia e
non va imposto senza confronto:
[specifica OpenMP](https://www.openmp.org/spec-html/5.0/openmpse55.html).

Politiche iniziali:

- Una sola inferenza, nessun pool di worker che carichi più copie del modello.
- Nessuna tokenizzazione parallela al decode di un'altra richiesta.
- Nessun watcher, compilatore o ricaricamento automatico nel percorso distribuito.
- Memoria campionata al massimo una volta al secondo; contatori aggiornati
  insieme agli eventi già prodotti. Nessun profiling pesante in uso normale.
- Markdown e highlighting raggruppati durante la risposta, completati a fine
  turno. Evitare la rielaborazione completa del transcript per ogni ID.
- Affinità, numero di thread e priorità derivati dal runtime qualificato.
  Non riservare un core per principio, non usare priorità realtime e non
  modificare la configurazione della ricerca per far sembrare economica la UI.
- Nessun dump di logits, tracce degli esperti o log per token nel percorso
  prodotto, salvo strumenti diagnostici esplicitamente attivati.

## Statistiche e verifica dell'overhead

Separare durata di caricamento, prefill, decode e tempo complessivo.
Il tempo al primo token parte dall'invio e finisce al primo contenuto inserito
nel widget di testo; la presentazione fisica del frame non è misurata.
eventuali tempi interni del motore hanno etichette diverse.
La velocità decode usa il contatore dei token e la finestra dichiarata dal
runtime; la velocità complessiva include preparazione e trasporto.
Non contare parole o eventi di visualizzazione come token. Un tempo non disponibile
rimane non disponibile. La memoria del processo non è equivalente alla
dimensione dei pesi; riportare la metrica Windows utilizzata.

Confronto proposto sullo stesso artefatto, binario, prompt, limite di output,
politica di generazione, thread e affinità:

1. Percorso diretto senza UI.
2. Adattatore senza finestra, con un consumer di riferimento.
3. Adattatore con finestra aperta, testo e statistiche attivi.
4. Variante progressiva contro variante completa, se entrambe disponibili.

Riscaldamento separato dal caricamento a freddo. Almeno sette ripetizioni per
caso, ordine alternato, prompt corti e lunghi e risposte brevi e lunghe.
Riportare mediana, dispersione, tempi al primo token, CPU e memoria, senza
attivare campionamento diverso tra i casi confrontati. Il budget del goal
riguarda la mediana decode; latenza totale e variazioni restano visibili.
Se le differenze non si risolvono rispetto alla variabilità osservata,
aumentare le ripetizioni prima di dichiarare il gate superato.

Verifiche mirate: parità degli ID con il riferimento deterministico, primo
contenuto ricevuto prima della fine della generazione, Unicode, EOS, Stop,
disconnessione, processo terminato, limiti di contesto, ripristino della
cronologia e isolamento dello stato tra conversazioni.

## Fasi di sviluppo

1. Confermare o correggere goal, perimetro e budget di overhead.
2. Costruire la finestra e il contratto usando un adattatore simulato,
   esplicitamente segnalato come demo di sviluppo; nessuna statistica simulata
   presentata come misura.
3. Qualificare il runtime e l'artefatto scelti dalla ricerca, collegare tokenizer,
   template e catalogo, verificare conversazioni complete.
4. Aggiungere streaming reale e Stop, se richiedono estensioni al consumer.
5. Misurare overhead, correggere le regressioni e preparare avvio e documentazione.
6. Integrare con il ramo principale aggiornato e ripetere le verifiche pertinenti.

Il 16 ottobre è il vincolo di disponibilità indicato per il piano settimanale:
conviene anticipare il lavoro indipendente dalla ricerca. Il rilascio completo
rimane subordinato a un runtime utile e approvato, senza abbassarne i gate.

## Branch e integrazione

`feature/local-chat` nasce dal commit di ricerca esaminato in un worktree
separato. È una dipendenza deliberata: il branch include gli antenati della
ricerca, quindi non va unito al ramo principale prima della ricerca stessa.

Nei riferimenti locali il ramo principale è `SirProjects` e il riferimento
remoto predefinito è `origin/SirProjects`; `main` non risulta presente.
Non rinominare il ramo principale nell'ambito di questo progetto.

Percorso previsto:

1. Ricerca positiva → revisione e merge nel ramo principale.
2. Incorporare il ramo principale aggiornato in `feature/local-chat`.
3. Risolvere soltanto le integrazioni necessarie, qualificare il runtime e
   verificare che il diff del branch contenga il progetto UI previsto.
4. Revisione e merge della chat nel ramo principale.

Preferire un merge ordinario della ricerca per conservarne l'ascendenza.
Se la ricerca viene integrata con squash o cronologia riscritta, ricostruire
il branch UI sopra il ramo principale e trasferire soltanto i commit UI,
prima della revisione finale: il merge semplice potrebbe ripresentare
modifiche della ricerca.

Commit piccoli e separati per contratto, finestra, adattatore, streaming e avvio.
Non modificare risultati, ricevute o manifest della ricerca. Eventuali
estensioni native additive vanno concordate con il runtime finale e verificate
contro il consumer storico. Nessuna firma o attribuzione automatica nei
branch, commit o messaggi; usare l'identità Git del proprietario.
