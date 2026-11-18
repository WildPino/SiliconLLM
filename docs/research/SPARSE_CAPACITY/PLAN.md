# Sparse Capacity: piano e gate

**Ramo:** `research/sparse-capacity`, aperto il 2026-11-17 da `SirProjects`.

**Questo file e la fonte unica del piano.**
- Ogni protocollo, risultato e commit del ramo dice a quale fase e a quale domanda serve.
- Il registro corto sta in [`INDEX.md`](INDEX.md); la letteratura letta in [`LITERATURE.md`](LITERATURE.md).

## 1. Le domande

| Sigla | Domanda | Perche conta |
|---|---|---|
| **D-0** | A parametri attivi e token fissi, una banca di esperti piu grande abbassa il BPB su questa architettura? | Se no, le altre due non hanno senso |
| **D-M** | Su un PC consumer si puo addestrare una banca di micro-esperti **piu grande della VRAM**, tenuta nella RAM dell'host, con la stessa qualita dell'addestramento solo-GPU e a un ritmo utilizzabile? | E il metodo: il contributo minimo del programma |
| **D-R** | A nucleo fisso, una banca piu grande migliora **solo la conoscenza** o anche il **ragionamento**? | E la domanda forte. La letteratura sui MoE classici dice "solo la conoscenza"; con micro-esperti non e misurata |

**Il PC consumer di D-M e questo:**
- GPU: RTX 3060 da 12 GB;
- CPU: Ryzen 5 3600X, AVX2 senza AVX-512 ne AMX;
- RAM: 80 GB DDR4-3200;
- in piu, la quota Kaggle T4.

## 2. Cosa vale come successo

- **Risultato minimo (D-M):** le fasi 3 e 4 passano. Ne esce un metodo misurato e rifacibile che allena su questo PC una
  banca oltre la VRAM, entro 2σ dall'addestramento solo-GPU, ad almeno il 50% del suo ritmo.
- **Risultato pieno (D-R):** in piu, a nucleo fisso il ragionamento sale con la banca oltre il rumore, misurato separato
  dalla conoscenza.
- **Non e un obiettivo:** un chatbot, l'instruction tuning, l'allineamento.

## 3. Vincoli

- **Il modello e originale.**
  - Nessun peso di modelli pre-addestrati esterni: ne come inizializzazione, ne come insegnante, ne come donor.
  - Nessun testo generato da LLM esterni.
  - Solo corpora pubblici scritti da persone, e il tokenizer si allena qui.
- **Le regole di ricerca del repo valgono tutte:**
  - protocollo pushato prima del run, CONFIG stampato;
  - controllo banale per ogni metrica primaria, letteratura letta prima;
  - revisore indipendente oltre 10 h di GPU o per una metrica nuova;
  - dopo un FAIL una sola raccomandazione;
  - protocollo di circa 3 pagine, risultato di circa 2.
- **Un valore non finito interrompe il run,** senza saltarlo in silenzio.
- **Gli artefatti grandi** non vanno in git: si registrano percorso e SHA-256.

## 4. Cosa sappiamo prima di misurare

- **Probe-4:** un MoE E32 x h128, top-8, batte il denso a 22,5 M parametri totali. Un seme solo, 4k passi
  (`docs/SCALEUP_ARCHITECTURE.md` §3.5).
- **Rung-1:** D256 N96 L8 swa@5, E32 x h128 top-8, addestrato per 183.104 passi, BPB P62 0,9277. Il confronto con la sua
  versione densa (`stage_D`) non e mai stato letto (`docs/PHASE64_RUNG1_RUN_RECORD.md` §5).
- **Fase 64.1:** a queste taglie la velocita d'inferenza ha un margine di 12-130 volte; il muro e l'addestramento
  (`docs/PHASE64_BUDGET.md` §4).
- **Letteratura** ([`LITERATURE.md`](LITERATURE.md)):
  - "GPU per il nucleo, esperti in RAM" e gia fatto per il fine-tuning su server (KTransformers-FineTune, OOM, ES-MoE);
  - OOM misura lo streaming dei pesi sulla GPU 26-60 volte piu veloce del calcolo degli esperti sulla CPU;
  - a tavolino la CPU conviene solo sotto circa 5 token per esperto per passo;
  - nei MoE classici la banca porta conoscenza piu che ragionamento.

## 5. Le fasi

Solo la prossima fase e dettagliata.
- **A fine fase:** un risultato di al massimo 2 pagine, con una sola raccomandazione; l'utente decide il passaggio alla
  successiva.
- **Se una fase fallisce,** il suo "se va male" e scritto qui prima.

### Fase 0: l'avvio (IN CORSO)

- **Consegna:**
  - questo piano, l'indice e la nota di letteratura, pushati;
  - l'audit di `benchmarks/phase64/mve/` contro la quantizzazione di `benchmarks/phase60/engine.c` (scala per riga, 63
    livelli), che serve alla fase 2.
- **Gate:** nessuno.

### Fase 1: le prove economiche (DETTAGLIATA)

Sono due prove indipendenti, e ognuna puo chiudere una strada.

**1a. Il gate 1 del rung-1 (D-0, solo CPU).**
- **Domanda:** sul testo P62 code-val, il rung-1 con la banca (`stage_E`, `stage_F`) batte la sua versione densa
  (`stage_D`)?
- **Limite noto:** la versione densa e il punto prima della divisione in esperti, quindi ha visto meno token. Il
  risultato e un indizio su D-0, non la risposta.
- **Il protocollo registra prima della lettura:** il confronto, la soglia (in unita di σ_seed ≈ 0,005 BPB) e il
  controllo banale.
- **Dati:** i pesi sono in `D:/_ktmp/mainrun_final/`. Il testo P62 sta su E:, oggi scollegato, e si legge quando torna.
- **Esito:**
  - se la banca e peggio del denso, e una bandiera rossa su D-0: se ne parla prima della fase 2;
  - altrimenti la fase 2 parte.

**1b. Magazzino contro officina (D-M, CPU e GPU, macchina scarica).**
- **Domanda:** sul 3600X con la 3060, sotto quanti token per esperto per passo (r*) conviene calcolare un esperto sulla
  CPU (forward, backward e aggiornamento) invece di portarne i pesi sulla GPU e riportarne il gradiente?
- **Bracci:**
  - esperti h128, h8 e h1 (stile PEER) a D = 256;
  - r da 1 a 128;
  - CPU a 6 thread: fp32, e il kernel LUT ternario nel forward;
  - PCIe misurata nei due versi.
- **Predizione a tavolino:** r* ≈ 5 con PCIe a 25 GB/s, ≈ 11 a 12 GB/s. La stima della CPU (200 GFLOP/s) non e misurata.
- **Controllo banale:** banda pura di RAM e di PCIe. Un tempo misurato sotto la banda vuol dire una misura sbagliata.
- **Esito:**
  - se r* ≥ 2 nel regime che si allenerebbe davvero (micro-esperti, passi da 16k token o piu), l'angolo CPU esiste e la
    fase 3 lo usa: posto scelto per esperto, a seconda del suo r;
  - se la CPU perde anche a r = 1, il metodo e lo streaming, gia pubblicato da OOM: D-M diventa ingegneria, e si decide
    se il programma continua solo per D-R.
- **Costo:** ore, a macchina scarica, con il via dell'utente.

### Fase 2: la prova piccola su E (PROTOCOLLO: [`F2_BANK_PROTOCOL.md`](F2_BANK_PROTOCOL.md))

Il protocollo sostituisce i bracci e la base qui sotto (base m1 invece del denso, h8, fino a m64); il perche sta nel suo §8.


- **Domande:** D-0 fatta bene, e D-R.
- **Il nucleo e fisso,** con la forma del rung-1: D256 N96 L8 swa@5.
- **Bracci:** solo GPU, banche che entrano nella 3060. Il denso e E ∈ {32, 256, 2048}, a pari parametri attivi e pari
  token, con almeno 10⁵ token per esperto.
- **Misure:**
  - BPB su testo tenuto da parte;
  - una sonda di conoscenza e una di ragionamento, separate, con compiti sintetici scelti dalla letteratura (per
    esempio la rubrica e il cammino minimo di Mixture of Parrots);
  - il controllo banale di ciascuna.
- **Gate, registrati nel protocollo:**
  - **D-0 PASS:** il BPB scende con E in modo monotono, e il guadagno al punto piu grande e almeno 4σ;
  - **D-R VIVA:** la sonda di ragionamento sale con E di almeno 2σ, a nucleo fisso.
- **Budget:** fissato nel protocollo, con il revisore (regola 6). La precisione (ternaria o bf16) la decide il
  protocollo, dopo l'audit della fase 0.
- **Se va male:**
  - D-0 FAIL: il programma si chiude con un rapporto;
  - D-0 PASS e D-R piatta: "non mostrato" a questa scala; si va avanti con il metodo, la domanda resta aperta e le
    sonde K e R restano letture nelle fasi 3 e 4;
  - D-R viva: le fasi 3 e 4 la misurano di nuovo a banche piu grandi.

### Fase 3: il metodo (ABBOZZO)

- Il trainer tiene la banca nella RAM, nella forma decisa dalla 1b.
- **Gate:**
  - equivalenza con l'addestramento solo-GPU su un braccio della fase 2: BPB entro 2σ, curve sovrapposte;
  - ritmo almeno al 50% di quello solo-GPU, a pari parametri attivi.

### Fase 4: oltre la VRAM (ABBOZZO)

- Almeno una banca che non entra nella GPU (banca piu stato ≥ 20 GB), con gli stessi parametri attivi e la stessa soglia
  di token per esperto.
- **Gate:** il BPB scende ancora con log E, senza saturazione precoce, e la sonda di ragionamento e riportata.

### Fase 5: nel motore (ABBOZZO)

- Export nel motore C, parita dei logit con PyTorch, tok/s a due taglie di banca.

## 6. Regole contro la deriva

- **Ogni lavoro nomina la fase e la domanda** (D-0, D-M o D-R). Un lavoro che non serve a nessuna delle tre non si fa
  senza il via dell'utente.
- **Ogni risultato ha una riga in `INDEX.md`,** di al massimo 300 caratteri, che dice cosa cambia per D-0, D-M e D-R.
- **Due celle di fila su una deviazione** vogliono dire fermarsi e proporre il ritorno al piano.
- **Fuori programma:**
  - chatbot e instruction tuning;
  - donor e distillazione;
  - modifiche a `engine.c` prima della fase 5;
  - il livello di recall;
  - le alternative alla backpropagation.
- **Il piano cambia solo a fine fase,** con il via dell'utente, e la modifica si registra qui sotto.

## 7. Dal charter iniziale (2026-11-12, non nel repo)

Questo piano parte da un charter iniziale. Cosa ne toglie, e perche:

| Elemento del charter | Qui | Perche |
|---|---|---|
| Fase A con 11 bracci, 4-6 settimane | 4 punti (denso, E32, E256, E2048) nella fase 2 | Prima le prove economiche; la fase 2 si dimensiona sull'esito della fase 1 |
| H2 granularita (h8 contro h128) | rinviata dopo la fase 2 | Krajewski 2024: oltre G = 16 il costo del routing domina; si decide con l'esito della 1b |
| H3 ternario contro bf16 | la precisione e decisa nel protocollo della fase 2 | Serve prima l'audit della fase 0 |
| H5 router a chiavi prodotto contro scansione piena | rinviata | Non decide nessuna delle tre domande alla fase 2 |
| P1 "si spostano le attivazioni, non i pesi" | sostituita dalla misura 1b | Smentita a tavolino e da OOM nel regime di addestramento ([`LITERATURE.md`](LITERATURE.md) §3) |
| Metrica unica, BPB | BPB piu le sonde di conoscenza e di ragionamento | Il BPB premia anche la conoscenza: da solo non risponde a D-R |

## Registro delle modifiche al piano

| Data | Modifica |
|---|---|
| 2026-11-17 | Piano iniziale, fasi 0-5 |
| 2026-11-18 | Dopo la 1b (GRIGIO, via dato): il metodo e lo streaming, il programma continua per D-R. La granularita (H2, §7) si decide nel protocollo della fase 2, non dopo: h8. |
| 2026-11-18 | Via alla fase 2 con una modifica: una D-R piatta alla fase 2 non chiude la domanda sul ragionamento. A quella scala (nucleo 4,6 M, 60 M token) vale "non mostrato"; le sonde K e R restano letture nelle fasi 3 e 4. |
