# F4: oltre la VRAM, il protocollo

2026-11-28. Fase 4 del [piano](PLAN.md), domanda **D-M** e gate di fase su log E. Codice in
`benchmarks/sparse_capacity/`: `sc4_model.py` (router e calcolo sparso), `sc4_host.py` (banca sull'host),
`csrc/adamw_tern.c` e `sc4_adamw.py` (il kernel), `sc4_train.py` (il trainer). Il verdetto lo calcola
`sc4_verdict.py` con le regole del §4, scritte qui prima dei run.

**Stato: REGISTRATO** il 2026-11-28, dopo il revisore (§9). I numeri di §5 sono stime contese: la cella di ritmo a
macchina libera e il primo passo dei run.

**Le domande.**
- **D-M:** su questo PC si puo allenare una banca di micro-esperti che non entra nella GPU, con la stessa matematica
  dell'addestramento solo-GPU, a un ritmo utilizzabile?
- **Gate di fase:** la banca piu grande abbassa ancora il BPB, a parita di router e di token?

**Riga sulla tesi.** La banca e di 1,61 G parametri: 25,8 GB sull'host, contando i master, i due momenti di AdamW e
il gradiente. Un PASS di D-M e il risultato minimo del piano (§2): un metodo misurato che allena oltre la VRAM, entro
il rumore del solo-GPU e ad almeno meta del suo ritmo. Il costo sull'host cresce linearmente con la banca (§5), quindi
dice anche quanto costerebbe a 40 GB. Non dice niente sul ragionamento (D-R, §6).

## 1. Cosa si allena

| Braccio | E per strato | Banca (8 strati) | Master e AdamW | Micro-batch | Token | Token per esperto |
|---|---|---|---|---|---|---|
| e2k | 2.048 | 101 M | GPU | 16 | 240 M | 15,0 M |
| e8k | 8.192 | 403 M | GPU | 8, accumulo 2 | 240 M | 3,75 M |
| e32k | 32.768 | 1.611 M | host | 16 | 240 M | 0,94 M |
| e2k-h | 2.048 | 101 M | host | 16 | i primi 32,8 M di e2k (4.000 passi) | (equivalenza, G0) |

- **Uguale per tutti i bracci:**
  - il nucleo della fase 2 (rung-1 S0: D256, N96, H8, L8, SWA allo strato 5);
  - esperti h8, k = 128 attivi, hidden attivo 1.024;
  - pesi ternari e attivazioni del motore come nella fase 2 (F2 §3);
  - la ricetta della fase 2: AdamW (0,9, 0,95), weight decay 0,1, lr 3e-3, 200 passi di warmup e coseno fino al 10%
    su 29.296 passi, clip 1;
  - seed 0 e stessi batch.
- **Il router a chiavi prodotto** (§7):
  - q = Wq x con dk = 128; due insiemi di sottochiavi, n1 × n2 = E;
  - top-k esatto, softmax sui k scelti per k (la formula del gating della fase 2);
  - la perdita di bilanciamento della fase 2, calcolata esatta (peso 0,01).
- **Il calcolo degli esperti** usa solo le coppie (token, esperto) scelte, raggruppate per esperto: e la matematica
  della fase 2 (§3).
- **Il flusso dei dati:** 80% codice, 4% K, 12% P con nome (rubriche da 64 e 512), 4% rubrica esplicita; cifre
  separate da spazi. Sono letture (§6).
- **Il metodo (e32k, e2k-h):**
  - master fp32, gradiente e momenti di AdamW nella RAM dell'host;
  - il kernel fuso fa AdamW (con il coefficiente del clip) e ricalcola i ternari impacchettati: uguale bit a bit ad
    AdamW di torch, codici uguali a `w_q`;
  - alla GPU vanno i codici a 2 bit e le scale per riga (0,43 byte per peso) una volta a passo; restano li per il
    forward e il backward;
  - il backward manda alla RAM, strato per strato, il gradiente straight-through g/s in fp32, con la sua norma per il
    clip;
  - l'aggiornamento dell'host gira strato per strato in un thread: lo strato l arriva sulla GPU mentre il forward del
    passo dopo lavora sugli strati prima di l. Stesse operazioni nello stesso ordine (§3).

## 2. Perche quattro bracci

- **Il gate su log E si legge a parita di router e di token** (PLAN, fase 4). e2k ed e8k entrano nella GPU con lo
  stesso router e gli stessi token di e32k: tre punti a passi di 4 volte, come nella fase 2. Con due punti si vede solo
  una differenza, non la saturazione.
- **e2k-h** e il metodo a una taglia che entra anche nella GPU: e2k-h contro e2k misura l'equivalenza su 4.000 passi
  (G0), con il codice di e32k.
- e2k ed e8k sono anche il riferimento solo-GPU del ritmo (G1).

## 3. Controlli fatti prima della registrazione

| Controllo | Esito |
|---|---|
| Kernel: AdamW fuso con il coefficiente del clip, contro clip e AdamW fused di torch | uguale bit a bit; 20/20 |
| Router: top-k a chiavi prodotto contro il top-k su tutti gli E (E = 128, 2.048, 32.768) | identico |
| Router: pesi e perdita di bilanciamento contro le formule della fase 2 sulla softmax piena | ≤ 3e-7 relativo |
| Calcolo sparso contro quello denso della fase 2: somme intere di gate e up | uguali bit a bit |
| Lo stesso: forward, gradienti di x e dei pesi | ≤ 4e-7 relativo |
| Determinismo: due forward e backward, router compreso | uguali bit a bit; 29/29 |
| Host contro solo-GPU dallo stesso init, gli stessi batch e lr 3e-3 dal primo passo: E = 2.048; clip forzato (0,05); E = 32.768 sullo strato 0 | §3.1 |
| Aggiornamento in pipeline contro quello sincrono, 30 passi; ripresa da stato salvato | uguali bit a bit (perdite, master, momenti, nucleo) |
| Dry a dimensioni vere (e32k): picco GPU; checkpoint da 18 GB salvato e riletto, master e momenti compresi | 10,1 GB; 2 min, identico |

**3.1. L'equivalenza a 20 passi.** Le due strade differiscono solo per gli arrotondamenti: la scala per riga del kernel
somma |w| in un altro ordine rispetto a torch (fino a 2,7e-7 relativo), e AdamW gira sulla CPU invece che su CUDA.
- Il metro e un gemello nullo: la strada solo-GPU con tutte le scale moltiplicate per (1 + 2,4e-7).
- Regola: l'host resta entro 3 volte lo scostamento del gemello nullo, sulla perdita e sulla norma del gradiente; un
  negativo piantato (uno strato dell'host mai aggiornato) sta fuori di almeno 10 volte.
- **Esiti** (`--equiv`, 13/13), scostamento massimo della perdita:

  | Caso | Passi | Gemello nullo | Host | Negativo piantato (primi 6 passi) |
  |---|---|---|---|---|
  | E = 2.048, lr 3e-3 | 20 | 7,4e-3 | 7,3e-3 | 2,7e-2 (host 9,6e-4) |
  | E = 2.048, clip 0,05 (coefficiente ≤ 0,09 a ogni passo) | 12 | 2,6e-3 | 1,4e-3 | 2,7e-2 (host 3,9e-4) |
  | E = 32.768 sullo strato 0 | 10 | 8,8e-4 | 1,1e-3 | 1,6e-1 (host 1,1e-3) |

  Le norme del gradiente stanno entro 0,1-0,4 volte la soglia. Al passo 0 (solo il forward) l'host si scosta di
  3,1e-5 contro 9,5e-7 del nullo, ma i codici sono identici: il nullo cambia tutte le scale dello stesso fattore, che
  la quantizzazione A2 (per massimo) assorbe. Con un ulp diverso riga per riga, come nel kernel, il nullo da 1,1e-5. Il
  nullo uniforme e quindi il metro piu severo.
- **Correzione della regola, dichiarata.** Nella prima prova a lr 3e-3 il negativo piantato si leggeva sul massimo dei
  20 passi. Anche il gemello nullo deriva di 7,3e-3 (l'host di 7,3e-3, il piantato di 2,7e-2): a quel lr
  l'arrotondamento cresce in fretta. Un aggiornamento saltato si vede invece subito, quindi il negativo piantato si
  legge sui primi 6 passi. La regola sull'host (entro 3 volte il nullo su tutti i passi) non cambia.
- La fase 3 aveva gia mostrato l'equivalenza della banca in streaming su 7.324 passi (ΔBPB −0,0005), ma con un altro
  codice; per questo c'e G0.

## 4. Gate (registrati prima dei run)

- **G0, equivalenza lunga.** e2k-h contro e2k, stesso seed, programma e batch: il BPB delle curve ai passi 2.000 e
  4.000 differisce al massimo di 0,010, la tolleranza delle curve della fase 3.
- **G1, ritmo.** Token/s di e32k diviso per i token/s del piu veloce fra e2k ed e8k: almeno 0,5.
  - Si misura nella cella di ritmo (`sc4_train.py --time`): 200 passi dopo 20 di riscaldamento, i batch fatti dentro
    il cronometro come nel run, macchina libera, codice registrato. Il file riporta il carico della CPU e gli altri
    processi Python.
  - **Accanto, letture:** il ritmo contro m1 della fase 2, misurato nella stessa cella; il costo dell'host esposto (il
    passo di e32k meno lo stesso passo con l'aggiornamento dell'host saltato); il ritmo a meta run, da un checkpoint e
    dalle finestre del run (§9, obiezioni 1 e 5).
  - **Lettura del ritmo proiettato** (aggiunta il 2026-11-28, prima dei run): il ritmo se il calcolo sparso costasse
    quanto il banco denso di m1. E il passo di m1, piu quanto il routing sparso aggiunge alla scansione del router di
    m1, piu il costo dell'host esposto, a confronto con m1. Serve a sapere quanto pesera l'host quando il calcolo
    sparso avra un kernel suo.
    - Il routing sparso (router a chiavi prodotto, perdita di bilanciamento e raggruppamento per esperto) contro la
      scansione di m1 si misura nella cella `--time-parts`. Forward, backward e il forward di nuovo (checkpointing),
      per strato, alle forme del run. Dry: 82 ms contro 3,5 ms per strato, cioe 0,63 s a passo.
    - Si riporta anche il limite con tutto l'aggiornamento dell'host esposto: con un forward piu corto la pipeline
      nasconde meno.
  - La cella gira prima dei run lunghi: se G1 fallisce, i run non partono.
- **D-M PASS = G0 e G1.** Conduzione: G0 si legge dopo e2k; se fallisce, e32k ed e8k non partono e si scrive il
  risultato.
- **G2, log E (gate di fase).** Sul val della fase 2 (2 M token):
  - il BPB scende in modo monotono, e2k > e8k > e32k;
  - l'ultimo passo, BPB(e8k) − BPB(e32k), vale almeno 0,005, il pavimento σ della fase 2 (R1);
  - controllo banale: lo stesso guadagno sulla meta B del val, alla temperatura scelta sulla meta A, almeno 0,005;
  - accanto, la top-1 e l'uso degli esperti.
  - **Cosa puo dire un seed solo.** Con σ fra i seed di 0,0019 (fase 2), la differenza fra due bracci ha σ ≈ 0,0027.
    Con guadagno vero zero, G2 passa per caso circa 3 volte su 100. Con guadagno vero 0,005 passa una volta su due,
    con 0,010 quasi sempre (97%).
- **Se G2 fallisce, la lettura delle curve decide la raccomandazione (registrata).** Si confronta il distacco
  e8k − e32k nelle curve a fine run e al passo 14.000:
  - se e cresciuto di almeno 0,002, si raccomandano piu token;
  - altrimenti il guadagno si chiude a questo nucleo.
- **Regole di arresto (run VOID, nessun verdetto su G2):**
  - un valore non finito;
  - un picco: dal passo 2.000, la media della perdita su 500 passi supera di 0,10 la migliore media precedente;
  - una banca morta: a una curva dal passo 2.000, exp(entropia del carico)/E sotto 0,5 in uno strato (misurato su
    262 k token di val; m64 della fase 2 aveva 0,91).
  - **Rimedio registrato:** dopo un picco o una banca morta si rilancia con la BatchNorm sulla query (il rimedio di
    PKM, §7), senza altri cambi.
- **Predizioni:**
  - G0 PASS, entro 0,005;
  - G1 PASS, con un ritmo fra 0,6 e 0,8 contro e2k. Contro m1, fra 0,35 e 0,45;
  - G2 incerto. Nella fase 2, m16 → m64 (gli stessi token per esperto di e8k → e32k) ha guadagnato 0,004, ma il
    distacco cresceva ancora a fine run, e qui il nucleo vede 4 volte i token. Stima 0,003-0,012. Non e una predizione
    contro lo scopo: con un FAIL cambia la decisione sulla fase 5 (piu token, o la chiusura del guadagno a questo
    nucleo).

## 5. Costi

- **Stime dalla cella di ritmo in dry, contese** (sulla CPU girava un altro lavoro, carico 54-85%):

  | Braccio | Passo | Token/s | Note |
  |---|---|---|---|
  | m1 (fase 2) | 3,32 s | 2.470 | riferimento denso, E = 128 |
  | e2k | 6,10-6,75 s | 1.214-1.343 | picco GPU 11,0 GB |
  | e8k | ~9,5 s | ~860 | dal dry; picco 10,8 GB |
  | e32k | 9,15 s (min 8,01, max 9,25) | 895 | picco 10,1 GB; aggiornamento dell'host 2,47 s, nascosto in parte dalla pipeline (senza pipeline: 11,4 s). Misura con la CPU al 5-14% |

  - Dove va il tempo in piu di e2k su m1: il calcolo sparso costa 142 ms per strato (forward e backward) contro 20 ms
    del banco denso di m1. E limitato dalla memoria: ogni coppia porta con se il vettore del suo token, e i tensori
    per coppia sono di 1 M × 256 (circa 1 GB ciascuno).
  - AdamW dell'host a macchina libera: 1,31 ns per parametro (`a58aa5b`), 2,1 s a passo. Cresce linearmente con la banca.
- **Il budget**, da queste misure contese, che sono un tetto: circa 50 h per e2k, 77 per e8k, 75 per e32k e 7 per
  e2k-h. Si aggiungono 15 curve da circa 7 minuti per braccio e i checkpoint (e32k: 18 GB, 2 minuti ogni 3 ore). In
  tutto **~220 h di GPU**, circa 9 giorni. La cella di ritmo a macchina libera dara i numeri veri.
- **RAM dell'host per e32k:** 25,8 GB (6,4 GB di gradiente in memoria bloccata).
- **Ordine** (`sc4_chain.py`): le celle di ritmo (m1, e2k, e8k, e32k, e32k senza host) e la cella del routing a
  macchina libera; il verdetto del ritmo; poi e2k-h, e2k, il verdetto di G0, e32k, e8k. I run lunghi non sono benchmark: possono convivere con lavori sulla
  CPU, a costo di allungarsi.

## 6. Letture (non gate)

- P62, K, P con nome (64 e 512), rubrica esplicita, uso degli esperti e KL dall'uniforme, norma del gradiente, curve
  ogni 2.000 passi (BPB, uso e sonde), tempi del metodo e token/s per finestra.
- **Ripiego per D-R** (PLAN, fase 4): la regola della rubrica esplicita (margine di almeno 0,10 e 4 errori standard
  sul controllo piu forte) si applica al run di e32k. Se non passa, dopo la fase 4 si apre una fase D-R con un nucleo
  piu grande.
- **Se l'esplicita passa in e32k, la regola su P con nome** (registrata il 2026-11-28, prima dei run; lettura, non gate
  di D-M):
  - per ciascuna rubrica con nome (64 e 512), sulle letture finali;
  - P di e32k sta sopra il suo controllo piu forte (il massimo fra copia, regola piu frequente e funzioni mai viste)
    di almeno 0,10 e di almeno 4 errori standard binomiali;
  - e P di e32k sta sopra P di e2k di almeno 0,05 e di almeno 4 errori standard della differenza,
    √(p₁(1−p₁)/n + p₂(1−p₂)/n). Con n = 256 (P64) servono circa 0,16, con n = 2.048 (P512) circa 0,06;
  - **segnale D-R registrato** se le due condizioni valgono in almeno una rubrica. Si riportano tutte e due le
    rubriche ed e8k accanto. Se l'esplicita non passa, la regola non si legge.

## 7. Letteratura e modi noti di fallire

In [`LITERATURE.md`](LITERATURE.md), aggiunte del 2026-11-27 e del 2026-11-28.
- **Chiavi prodotto** (PKM, arXiv 1907.05242; PEER, arXiv 2407.04153; Memory Layers at Scale, arXiv 2412.09764). La
  ricerca e esatta (PKM sez. 3.1).
  - **Uso delle chiavi che crolla** con memorie grandi. In PKM senza BatchNorm l'uso e 98,97% a 65k chiavi e 25,8% a
    1M (tab. 2). Qui E = 32.768, c'e la perdita di bilanciamento della fase 2 e c'e la regola della banca morta (§4).
  - **Instabilita** con memorie grandi su nuclei piccoli (Memory Layers sez. 3.1.4, che usa la qk-norm "when needed"):
    regola del picco (§4).
- **Pesi quantizzati nel trasferimento:** il precedente piu vicino e il float8 all-gather di FSDP2. Qui la matematica
  non cambia (aggiunte del 2026-11-27).

## 8. Cosa toglie al design

| Elemento | Qui | Perche |
|---|---|---|
| Molti piu token per esperto della fase 2 (PLAN) | e32k ne ha 0,94 M, quanti m64 nella fase 2; e8k 3,75 M; e2k 15 M | I 20 GB fissano E; quattro volte m64 a e32k vorrebbe 960 M token, 4 volte questo budget. Alla scala della tesi la domanda resta aperta: le curve dicono se il distacco cresce ancora (§4) |
| Il ritmo contro il passo di m1 (proposto alla fine della fase 3) | m1 e una lettura; G1 usa lo stesso codice sulla GPU | Nella fase 3 m1 e il braccio in streaming erano lo stesso modello. Qui no: una banca di 32 k esperti chiede un calcolo sparso su qualsiasi GPU (in denso costerebbe da 16 a 256 volte i FLOP). Quel costo e del modello, non del metodo. Contro m1 il ritmo si riporta, con il costo dell'host esposto (§9, obiezione 1) |
| Un kernel fuso di raccolta e GEMM per il calcolo sparso (come ScatterMoE o MegaBlocks, non letti) | calcolo sparso in PyTorch | Su questa macchina non ci sono ne Triton ne nvcc. `torch.sparse` non serve: SpMM non e deterministico, SDDMM e lento (48 ms per strato) |
| Un gemello solo-GPU di e32k | non esiste: non entra | Equivalenza a 2.048 su 4.000 passi (G0) e a 32.768 su uno strato (§3) |
| Due seed per braccio | un seed | Costo: e8k s1 costerebbe ~77 h. σ dalla fase 2; potere statistico dichiarato in §4 |
| Teste multiple, BatchNorm sulla query, qk-norm (PKM, PEER, Memory Layers) | una testa, top-128, perdita di bilanciamento | Tiene la formula del gating della fase 2; la BatchNorm e il rimedio registrato (§4) |
| lr piu alto per i valori (PKM) | lr unico | La ricetta della fase 2 |
| La sonda T (legame) nel flusso | tolta | Le letture scelte sono K, esplicita e P con nome |
| Gradiente in bf16 sul bus, Adagrad per riga | no | Cambiano la matematica; solo se G1 non basta |
| Stesso micro-batch per tutti | e8k a 8 con accumulo 2 | A 16 il picco e 15,5 GB su 12. La cross-entropy media e la stessa; la perdita di bilanciamento si calcola su ogni mezzo batch (peso 0,01) |

## 9. Revisore indipendente (regola 6): obiezioni e risposte

Revisione del 2026-11-28, sul protocollo, sul piano e sul codice.
1. **Il riferimento di G1 era cambiato senza dirlo** (la fase 3 aveva proposto m1; contro m1 la stima e 0,34).
   Accolta in parte: il cambio e dichiarato in §8 con il motivo. Il ritmo contro m1 si misura nella stessa cella e si
   riporta; si riporta anche il costo dell'host esposto, cosi un denominatore lento non nasconde l'host. Il profilo
   del calcolo sparso e in §5.
2. **D-M non aveva un gate di qualita per questo codice** (la fase 3 provava un altro codice, 20 passi sono pochi).
   Accolta: G0, e2k-h contro e2k su 4.000 passi (~7 h); D-M = G0 e G1.
3. **G2 non decide la saturazione con un seed; il ramo FAIL non aveva una regola.** Accolta in parte: verdetto
   separato (D-M non dipende da G2), potere statistico dichiarato, regola numerica sulle curve. Il secondo seed non si
   aggiunge (costo, §8).
4. **L'equivalenza a 20 passi era debole** (lr basso, clip mai attivo, niente norme, pipeline provata su un passo).
   Accolta: lr 3e-3 dal primo passo, clip forzato, norme del gradiente, E = 32.768 su uno strato, pipeline contro
   sincrono su 30 passi (§3).
5. **La cella di ritmo non e il run** (batch fatti prima del cronometro, solo il warmup, nessuna dispersione).
   Accolta: batch dentro il cronometro, min, mediana e max del passo, carico della CPU; il ritmo a meta run e dalle
   finestre del run e una lettura.
6. **Il router a chiavi prodotto non e il router della fase 2, e i suoi modi di fallire erano solo letture.** Accolta:
   regole di arresto per la banca morta e per i picchi, con un rimedio registrato. "Gating identico" corretto in "la
   formula del gating della fase 2".
7. **Minori.** Perdita di bilanciamento con l'accumulo: dichiarata in §8. Il verdetto controlla registrazione, push e
   passi completati. Le letture ripartono da flussi nuovi a ogni curva, quindi non dipendono da una ripresa. Il
   conteggio dell'uso in addestramento conta due volte (checkpointing): l'uso si legge solo in eval.
