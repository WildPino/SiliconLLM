# BRIEF H4 — Train the aggressive rank

**Stato:** `PRE-REGISTERED / NOT RUN`

## 1. Domanda scientifica

Sul donor reale congelato `Qwen/Qwen2.5-1.5B`, il training direttamente nel formato ternario low-rank q/o a frazione aggressiva `r/D = 1/32` produce un miglioramento congiunto di score e ranking rispetto alla propria inizializzazione ternaria step 0?

H4 non chiede se il modello risultante sia già un modello 10B utile, né se raggiunga una velocità specifica. Chiede se il training nel formato riesca a recuperare segnale a `rank=48`, la stessa frazione di rank usata dalla shape sintetica A10B-R128.

## 2. Motivazione e anti-duplicazione

Il goal generale resta eseguire un modello pretrained di circa 10B parametri su `engine.c` ad almeno 50 tok/s, con almeno 100 tok/s come risultato eccellente.

Le misure esistenti delimitano la domanda, ma non la risolvono:

- E40 A10B-R128 usa `D=4096`, `rank=128`, quindi `r/D=1/32`, e misura circa 113–130 tok/s con FFN live. È però un artefatto sintetico/noise: misura una shape di esecuzione, non qualità pretrained.
- E65 misura sul donor reale una compressione q/o post-hoc fp32. A `D=1536`, QO-48 realizza `r/D=1/32` e ottiene BPB `2.473430292224694`, free `4/160`, teacher-forced `33/160`, mean rank `3476.46875`, median rank `30.5` e rank ≤5 `55/160`.
- E65 misura inoltre QO-96 post-hoc fp32 a BPB `2.0972750188750418`, free `1/160`, teacher-forced `42/160`, mean rank `1172.9625`, median rank `8` e rank ≤5 `69/160`; QO-192 post-hoc fp32 ottiene BPB `1.8563784310382423`, free `4/160` e teacher-forced `56/160`.
- Il controllo dense di E65 ottiene BPB `0.7675949641196624` e free/teacher-forced `160/160`.
- H0 ha mostrato, esclusivamente a rank 512, che l'inizializzazione ternaria e il modello addestrato sono oggetti diversi: BPB `2.812238`, free `1/160`, teacher-forced `28/160` all'inizializzazione; BPB `0.810022`, free `15/160`, teacher-forced `115/160` dopo 1000 step cumulativi. H0 non stabilisce alcun trasferimento in larghezza o alla frazione `1/32`.

H4 colma quindi una cella non ancora misurata: training nel formato ternario alla frazione aggressiva `1/32` sul donor reale. Non ripete E65, perché E65 valuta fattori fp32 redensified post-hoc; non ripete H0, perché H0 usa rank 512.

## 3. Oggetto sperimentale congelato

- Donor: `Qwen/Qwen2.5-1.5B`.
- Revisione: `8faed761d45a263340a0528343f099c05c9a4323`.
- Layer: tutti i 28 layer.
- Proiezioni addestrabili: soltanto `q_proj` e `o_proj`.
- Rank: `48`.
- Dimensione del modello: `D=1536`.
- Frazione di rank: `r/D=1/32`.
- Parametrizzazione: `ternarize(A) * diag(s) * ternarize(B)` con straight-through estimator (STE).
- Parametri addestrabili: `8,260,224`.
- Tutto il resto rimane frozen in fp32.
- Inizializzazione: la stessa factorization activation-weighted balanced usata da H0/E22.
- Calibrazione della factorization: `32 × 512`, seed `42424`.
- Training stream: lo stesso di H0, seed `90011`.
- Valutazione held-out congelata: `24 × 512`, seed `1234`.
- SHA-256 degli ID di valutazione: `a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65`.
- Ranking: gli stessi 5 prompt di E6, per 160 posizioni.

L'inizializzazione step 0 di H4 con fattori ternari **non** coincide con QO-48 fp32 redensified di E65. Prima di qualunque uso di T4 deve essere valutata localmente e congelata in `h4_eval_init.json`. Le metriche di quel file costituiscono gli anchor `init` usati dalla banda `TRAINING_HELPS`; non vengono sostituite dalle metriche QO-48 di E65.

## 4. Controlli piantati e invarianti

Il controllo dense intact è un planted control. Deve essere valutato nel percorso CPU locale prima del training insieme all'inizializzazione H4.

Prima di T4 devono inoltre essere congelati:

- la preregistrazione presente;
- l'artefatto di factorization rank 48;
- il risultato dei self-test;
- la valutazione intact;
- `h4_eval_init.json`;
- il pacchetto di training;
- il manifest che identifica payload, configurazione, dati e anchor.

I self-test devono verificare l'apparato prima dell'esecuzione, ma questa preregistrazione non introduce soglie numeriche ulteriori rispetto a quelle dichiarate nelle bande terminali.

## 5. Protocollo

### 5.1 Fase CPU locale obbligatoria

L'ordine è:

1. factorize;
2. self-test;
3. valutazione dense intact;
4. valutazione H4 step 0;
5. pack;
6. congelamento di anchor e manifest.

Non si usa T4 finché preregistrazione, anchor step 0 e manifest non sono congelati.

### 5.2 Stage A

Il protocollo scientifico ammette una sola esecuzione su una sola T4. Questo brief non avvia
né autorizza implicitamente l'uso della risorsa: prima del launch va comunicato esplicitamente
all'utente che la T4 sta per essere usata.

- seed `1717`;
- trainer e schedule H0 invariati;
- massimo `4000` step;
- batch size `2`;
- gradient accumulation `8`;
- learning rate `2e-4`;
- OneCycle con `pct_start=0.05`;
- checkpoint ogni `250` step;
- wall cap `2.8 h`;
- checkpoint disponibile al raggiungimento del cap.

Le metriche teacher-forced calcolate su GPU sono soltanto segnali di avanzamento e non sono gate. L'adjudication è eseguita dalla valutazione CPU fp32 del checkpoint terminale prodotto dallo stop effettivo di Stage A. Non è consentita la selezione retrospettiva del miglior checkpoint.

## 6. Metriche da riportare

Per l'inizializzazione step 0 e per il risultato terminale devono essere riportate almeno:

- BPB;
- free exact-match su 160 posizioni;
- teacher-forced exact-match su 160 posizioni;
- mean rank sulle 160 posizioni;
- median rank sulle 160 posizioni;
- numero di posizioni con rank ≤5 su 160.

Il risultato terminale deve riportare separatamente tutti e quattro i booleani definiti sotto. Non esiste un `combined PASS` per H4.

## 7. Bande terminali preregistrate

### 7.1 `TRAINING_HELPS`

`TRAINING_HELPS = true` se e solo se, rispetto agli anchor congelati in `h4_eval_init.json`, sono vere contemporaneamente tutte le condizioni:

- BPB terminale `<` BPB init;
- teacher-forced terminale `>` teacher-forced init;
- mean rank terminale `<` mean rank init;
- rank ≤5 terminale `>` rank ≤5 init.

### 7.2 `BEATS_QO96`

`BEATS_QO96 = true` se e solo se sono vere contemporaneamente:

- BPB terminale `< 2.0972750188750418`;
- teacher-forced terminale `> 42`;
- free terminale `> 1`.

### 7.3 `BEATS_QO192`

`BEATS_QO192 = true` se e solo se sono vere contemporaneamente:

- BPB terminale `< 1.8563784310382423`;
- teacher-forced terminale `> 56`;
- free terminale `> 4`.

### 7.4 `GENERATOR_PARTIAL`

`GENERATOR_PARTIAL = true` se e solo se free terminale `> 14`.

La banda storica resta:

- free `≤14`: `AT-FLOOR`;
- free `≥15`: `PARTIAL`.

Le quattro bande sono indipendenti e devono essere esposte come quattro booleani distinti, insieme a tutte le metriche; non vengono aggregate in un verdetto unico.

## 8. Predizioni preregistrate

Queste sono predizioni falsificabili, non risultati né soglie aggiuntive:

- **P1.** Lo step 0 ternario rank 48 sarà peggiore di QO-48 fp32 post-hoc in BPB e non supererà teacher-forced `33/160`. La banda BPB prevista è volutamente larga, `3.2–5.5`, per non fingere un esponente non misurato.
- **P2.** Stage A produrrà `TRAINING_HELPS = true`.
- **P3.** Stage A probabilmente non produrrà `BEATS_QO96 = true`.
- **P4.** Stage A probabilmente non produrrà `GENERATOR_PARTIAL = true`.

P2 e P3 possono entrambe essere vere: il training può migliorare congiuntamente il proprio punto di partenza ternario senza recuperare abbastanza qualità da superare il riferimento post-hoc QO-96. `TRAINING_HELPS` misura direzione del segnale rispetto all'init H4; `BEATS_QO96` misura il superamento di un riferimento esterno più forte.

## 9. Stop e rerun policy

- Non è autorizzato alcun secondo seed o seconda sessione automatica.
- Se `TRAINING_HELPS = false`, la route viene chiusa.
- Se `TRAINING_HELPS = true` ma `BEATS_QO96 = false`, il risultato viene registrato come weak signal e non autorizza automaticamente altro compute.
- Se `BEATS_QO96 = true` oppure `GENERATOR_PARTIAL = true`, qualunque Stage B richiede un nuovo brief separato.
- Un failure puramente operativo può essere riparato soltanto mantenendo invariato il payload scientifico e documentando l'esecuzione fallita come `VOID_OPERATIONAL`.
- Un repair operativo non autorizza modifiche a seed, dati, modello, factorization, trainer, schedule, budget, metriche o bande.
- Non è consentito scegliere il best checkpoint. L'adjudication usa il checkpoint terminale definito dal protocollo Stage A.

## 10. Limiti e divieti di inferenza

H4 non autorizza:

- alcun claim di qualità a 10B;
- alcun claim di tok/s misurati;
- alcun width transfer o scale transfer;
- alcun export verso `engine.c`;
- alcun claim su FFN carve, router o one-byte;
- alcun confronto causale con H0 oltre al fatto osservativo che training e post-hoc sono oggetti differenti;
- la riapertura di H2I;
- modifiche o adjudication di `G-E63d`.

La relazione con A10B-R128 è soltanto geometrica: entrambe usano `r/D=1/32`. La misura di velocità sintetica E40 non trasferisce qualità a H4, e un eventuale risultato di qualità H4 non trasferisce velocità o qualità alla scala 10B.

## 11. Output attesi

Prima di T4:

- artefatto di factorization H4 rank 48;
- esito dei self-test;
- valutazione dense intact;
- `h4_eval_init.json` con tutti gli anchor step 0;
- pacchetto di training;
- manifest congelato del payload scientifico.

Dopo Stage A:

- checkpoint terminale;
- valutazione CPU fp32 terminale con tutte le metriche elencate;
- i quattro booleani `TRAINING_HELPS`, `BEATS_QO96`, `BEATS_QO192` e `GENERATOR_PARTIAL` riportati separatamente;
- record operativo sufficiente a distinguere un risultato scientifico da un eventuale `VOID_OPERATIONAL`.

Nessuno di questi output, da solo o in combinazione, costituisce un `combined PASS` o autorizza automaticamente Stage B.

---

**Stato finale della preregistrazione:** `PRE-REGISTERED / NOT RUN`
