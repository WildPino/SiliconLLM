Lavora nel repository D:\_THINGS\Progetti\SiliconLLM, sul branch
research/native-expert-scaling.

OBIETTIVO FINALE

Scoprire, implementare e validare un procedimento riproducibile che,
partendo da un LLM già addestrato, permetta di adattarne la conoscenza
e la capacità alla nostra architettura ed eseguirlo nel nostro engine C,
con qualità conservata entro limiti espliciti e misurati.

Non cerchiamo soltanto di far funzionare un singolo checkpoint.
Cerchiamo un metodo applicabile a più modelli, possibilmente appartenenti
a famiglie e dimensioni diverse, compresi modelli dell’ordine di 10B
e, quando le risorse lo consentono, 100B parametri.

Il risultato deve comprendere:

- un procedimento documentato, implementato e riproducibile;
- strumenti di analisi, trasformazione, eventuale adattamento e export;
- criteri per stabilire su quali modelli il procedimento è applicabile;
- evidenze di qualità, costo di conversione e prestazioni del risultato;
- limiti, fallimenti e varianti richieste dalle diverse famiglie.

La generalità è un obiettivo da dimostrare, non da dichiarare.
Una soluzione specifica per un modello è un primo risultato utile:
deve poi essere distinta dalle parti realmente trasferibili del metodo.

TESI ARCHITETTURALE DA PRESERVARE

Il vantaggio ricercato è separare la capacità totale dal costo per token:
un nucleo computazionale compatto e riutilizzabile, con grande capacità
condizionale consultata selettivamente.

Aumentare la dimensione totale del modello non deve comportare
necessariamente un aumento proporzionale del calcolo e del traffico
memoria per token.

Parti dalla struttura di engine.c e dalle evidenze già raccolte:
SSM/SWA, esperti, precisioni differenziate, LUT e organizzazione
della memoria. Questi sono il patrimonio iniziale della ricerca,
non una dimostrazione che la conversione sia già possibile.

Le estensioni del motore sono ammesse quando servono questa tesi.
Limitarsi a implementare un runtime generico per il donor, conservandone
un costo attivo incompatibile con l’obiettivo, non conclude la ricerca.

Gli esperti letti dalla DRAM hanno un costo reale. Non assumere
residenza in cache, località o banda ideale senza evidenze.

L’obiettivo prestazionale è >=50 token/s end-to-end, con 100 token/s
come traguardo migliorativo, sull’hardware e nei contesti dichiarati.
Qualità e velocità devono essere verificate sul medesimo artefatto.
Non basta produrre token: deve rimanere un modello utile.

CHE COSA DEVE SCOPRIRE LA RICERCA

Devi trovare quali operazioni permettono di passare dal modello
pretrained alla geometria eseguibile conveniente per il nostro engine.

Puoi esplorare trasformazioni dei pesi, partizionamento, routing,
fattorizzazioni, cambi di rappresentazione, precisione, sparsità,
riuso, sostituzione di operatori, distillazione, adattamento congiunto
e altre idee motivate dalle evidenze.

Non è obbligatorio provare indiscriminatamente ogni tecnica.
Scegli quelle che possono risolvere il vincolo emerso e cambiare
una decisione verso il goal.

Il procedimento potrà contenere trasformazioni esatte, approssimazioni
e addestramento. Dichiara quali passaggi appartengono a ciascuna categoria,
quanta qualità perdono o recuperano e quali risorse richiedono.

Non presumere che passaggi validi separatamente si compongano bene.
La qualità del procedimento completo deve essere verificata.

La verifica dello scaling degli esperti è un’attività abilitante:
serve a capire quale architettura target possa ospitare capacità utile.
Non sostituisce la costruzione del percorso dal pretrained al target.

Un piccolo student veloce, esperti duplicati o pesi sintetici non
dimostrano di aver trasferito la capacità di un grande donor.
Distingui dimensione del donor, parametri distinti del risultato,
parametri attivi e capacità effettivamente conservata.

STRATEGIA DI RICERCA

Mantieni separate ma collegate due domande:

1. Quale geometria target offre capacità utile a basso costo per token?
2. Come trasferire in quella geometria un modello già addestrato?

Ogni esperimento deve contribuire ad almeno una delle due.
Evita di perfezionare indefinitamente il target senza un percorso
credibile di trasferimento, o di completare un porting senza un
vantaggio architetturale verificabile.

Scegli autonomamente esperimenti, metodi e ordine delle attività.
Le intuizioni devono emergere anche dai risultati inattesi.
Puoi cambiare ipotesi e procedimento; non cambiare silenziosamente
il risultato che stiamo cercando.

Costruisci inizialmente un percorso end-to-end su un caso trattabile.
Poi verifica quali parti si trasferiscono a un’altra famiglia o scala,
riutilizzando gli strumenti e rendendo esplicite le parti specifiche.

Non serve che tutti i modelli seguano identici passaggi:
può emergere una procedura comune con varianti selezionate attraverso
un’analisi iniziale del donor. Anche questa selezione deve diventare
documentata e riproducibile.

Prima di un’attività sostanziale chiarisci:

- quale incertezza risolve;
- quale evidenza precedente riutilizza;
- quale decisione cambierà in base al risultato;
- costo previsto e criterio di arresto.

RIGORE E LIMITI

Confronta il modello adattato con il donor su dati held-out,
comportamento generativo e task pertinenti, oltre alle metriche locali.
Fissa i criteri decisionali prima di osservare i risultati.

Misura costo del routing, calcolo, memoria, latenza e risorse necessarie
all’adattamento. Riporta anche i costi che crescono con i parametri totali,
pur mantenendo fisso il numero di esperti attivi.

Distingui misure, deduzioni, ipotesi, proposte e risultati invalidi.
Non estendere automaticamente una conclusione a un’altra famiglia
o dimensione. Non promettere compatibilità universale:
individuare i prerequisiti e i casi non supportati è parte del metodo.

Non ripetere strade già esplorate senza specificare la nuova variabile
o l’incertezza residua. Non trasformare un fallimento locale in una
confutazione generale.

DOCUMENTAZIONE E CONTINUITÀ

Il punto di ingresso operativo è:
D:\_THINGS\Progetti\SiliconLLM\docs\research\NATIVE\_EXPERT\_SCALING\_20260925\INDEX.md

Leggi all’inizio questo indice, PRIOR\_EVIDENCE.md nella stessa cartella,
e i riferimenti necessari. Verifica lo stato reale di codice,
checkpoint, risultati e processi.

Mantieni INDEX.md breve e aggiornato con:

- goal e vincoli;
- stato delle due domande di ricerca;
- ultima evidenza decisiva;
- esperimenti svolti e relativi rimandi;
- ipotesi aperte e strade chiuse;
- prossimo passo e punto esatto di ripresa.

Crea e mantieni nella stessa cartella METHOD.md:
è il documento del procedimento in costruzione, distinto dal diario
degli esperimenti. Deve indicare:

- modelli di partenza e prerequisiti;
- analisi iniziale e scelta del percorso;
- passaggi attualmente disponibili;
- strumenti, input, output e costi;
- verifiche intermedie ed end-to-end;
- parti validate, provvisorie o mancanti;
- varianti per famiglia e limiti di applicabilità.

Non scrivere come acquisito un passaggio ancora ipotetico.
Un lettore deve capire subito cosa può già riprodurre e cosa resta
ricerca aperta.

Ogni esperimento deve avere un record identificabile con configurazione,
revisioni e hash rilevanti, dati, comando, risultati, costo, limiti
e decisione. Conserva anche fallimenti e interruzioni.
Collega i dettagli senza duplicarli in tutti gli indici.

La linea donor-adaptation resta in pausa come percorso operativo
precedente, ma le sue evidenze sono riutilizzabili. Se riattivarne
una parte serve al nuovo metodo, documenta perché: non riprenderla
automaticamente dal vecchio “prossimo passo”.

OPERATIVITÀ

Procedi autonomamente entro questo perimetro.
Comunica prima di avviare lavoro su T4 motivo, budget e criterio
di arresto. Chiedi chiarimenti per cambi sostanziali del goal
o nuove risorse/autorizzazioni.

Evita polling continuo e interferenze con i benchmark.
Delega soltanto quando il risparmio netto è reale.
Conserva il lavoro esistente e fai commit coerenti, senza firme
o attribuzioni a modelli/assistenti.

Prima di terminare una sessione aggiorna il punto di ripresa.
Il progresso si misura in parti del procedimento rese reali,
riproducibili e validate, non nel numero di esperimenti prodotti.