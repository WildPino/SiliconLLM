# F4: oltre la VRAM, il protocollo

2026-11-28. Fase 4 del [piano](PLAN.md), domanda **D-M**. Codice in `benchmarks/sparse_capacity/`: `sc4_model.py`
(router e calcolo sparso), `sc4_host.py` (banca sull'host), `csrc/adamw_tern.c` e `sc4_adamw.py` (il kernel),
`sc4_train.py` (il trainer). Il verdetto lo calcola `sc4_verdict.py` con le regole del §4, scritte qui prima dei run.

**Stato: bozza.** Mancano i tempi della cella di ritmo a macchina libera (§5) e il revisore (§8).

**La domanda.** Su questo PC si puo allenare una banca di micro-esperti che non entra nella GPU, con la stessa
matematica dell'addestramento solo-GPU, a un ritmo utilizzabile? E la banca piu grande abbassa ancora il BPB, a parita
di router e di token?

**Riga sulla tesi.** Qui la banca e di 1,61 G parametri: 25,8 GB sull'host, contando i master, i due momenti di AdamW
e il gradiente. Un PASS e il risultato minimo del piano (§2): un metodo misurato che allena oltre la VRAM, entro il
rumore del solo-GPU e ad almeno meta del suo ritmo. Il costo sull'host cresce linearmente con la banca (§5), quindi un
PASS dice anche quanto costerebbe a 40 GB. Non dice niente sul ragionamento (D-R, §6).

## 1. Cosa si allena

| Braccio | E per strato | Banca (8 strati) | Dove vivono i master e AdamW | Micro-batch | Token per esperto |
|---|---|---|---|---|---|
| e2k | 2.048 | 101 M | GPU | 16 | 15,0 M |
| e8k | 8.192 | 403 M | GPU | 8, accumulo 2 | 3,75 M |
| e32k | 32.768 | 1.611 M | host | 16 | 0,94 M |

- **Uguale per i tre bracci:**
  - il nucleo della fase 2 (rung-1 S0: D256, N96, H8, L8, SWA allo strato 5);
  - esperti h8, k = 128 attivi, hidden attivo 1.024;
  - pesi ternari e attivazioni del motore come nella fase 2 (F2 §3);
  - la ricetta della fase 2: AdamW (0,9, 0,95), weight decay 0,1, lr 3e-3, 200 passi di warmup e coseno fino al
    10%, clip 1;
  - 240 M token (29.296 passi da 8.192 token), seed 0, stessi batch.
- **Il router a chiavi prodotto** (§7), uguale nei tre bracci:
  - q = Wq x, con dk = 128;
  - due insiemi di sottochiavi, n1 × n2 = E;
  - top-k esatto, softmax sui k scelti per k (il gating della fase 2), perdita di bilanciamento della fase 2 calcolata
    esatta (peso 0,01).
- **Il calcolo degli esperti** usa solo le coppie (token, esperto) scelte, raggruppate per esperto. E la matematica
  della fase 2 (selftest, §3).
- **Il flusso dei dati:** 80% codice, 4% K (la rubrica della fase 2), 12% P con nome (rubriche da 64 e 512), 4%
  rubrica esplicita. Le cifre sono separate da spazi. Sono letture (§6).
- **e32k, il metodo:**
  - master fp32, gradiente e momenti di AdamW stanno nella RAM dell'host;
  - il kernel fuso fa AdamW e ricalcola i ternari impacchettati: uguale bit a bit ad AdamW di torch, e codici uguali a
    `w_q`;
  - alla GPU vanno i codici a 2 bit piu le scale per riga, 0,43 byte per peso, una volta a passo. La banca impacchettata
    resta sulla GPU (0,69 GB) per il forward e il backward;
  - il backward manda alla RAM, strato per strato, il gradiente straight-through g/s in fp32, con la sua norma per il
    clip;
  - l'aggiornamento dell'host gira strato per strato in un thread: lo strato l arriva sulla GPU mentre il forward del
    passo dopo lavora gia sugli strati prima di l. Stesse operazioni, stesso ordine.

## 2. Perche tre bracci

- **Il gate su log E si legge a parita di router e di token** (PLAN, fase 4). La serie della fase 2 (scansione piena,
  60 M token) non e confrontabile.
- e2k ed e8k entrano nella GPU con lo stesso router e gli stessi token di e32k. Tre punti a passi di 4 volte, come
  nella fase 2: servono per dire "scende ancora, senza saturazione precoce". Con due punti si vede solo una differenza.
- e2k ed e8k sono anche il riferimento solo-GPU del ritmo (§4, G1).

## 3. Controlli fatti prima della registrazione

| Controllo | Esito |
|---|---|
| Kernel: AdamW fuso con il coefficiente del clip, contro clip e AdamW fused di torch | uguale bit a bit; 20/20 |
| Router: top-k a chiavi prodotto contro il top-k su tutti gli E (E = 128, 2.048, 32.768) | identico |
| Router: pesi e perdita di bilanciamento contro le formule della fase 2 sulla softmax piena | ≤ 3e-7 relativo |
| Calcolo sparso contro quello denso della fase 2 (E = 256, 2.048): somme intere di gate e up | uguali bit a bit |
| Lo stesso: forward, gradienti di x e dei pesi gate, up e down | ≤ 4e-7 relativo |
| Determinismo: due forward e backward, router compreso | uguali bit a bit; 29/29 |
| e32k contro solo-GPU, E = 2.048, 20 passi dallo stesso init con gli stessi batch | §3.1 |
| Aggiornamento in pipeline contro quello sincrono; ripresa da checkpoint | uguali bit a bit |
| Dry a dimensioni vere (e32k): picco GPU, checkpoint da 18 GB salvato e riletto | 10,1 GB; 2 min, identico |

**3.1. L'equivalenza.** Le due strade differiscono solo per gli arrotondamenti. La scala per riga del kernel somma |w|
in un altro ordine rispetto a torch (fino a 2,7e-7 relativo), e AdamW gira sulla CPU invece che su CUDA.
- Il metro e un gemello nullo: la strada solo-GPU con tutte le scale moltiplicate per (1 + 2,4e-7).
- In 20 passi la perdita dell'host si scosta da quella solo-GPU al massimo di 1,9e-4; il gemello nullo di 2,4e-4.
- Il negativo piantato (uno strato dell'host mai aggiornato) si scosta di 6,1e-2.
- La fase 3 aveva gia mostrato l'equivalenza della banca in streaming su 7.324 passi (ΔBPB −0,0005).

## 4. Gate (registrati prima dei run)

- **G1, ritmo.** Token/s di e32k diviso per i token/s del piu veloce fra e2k ed e8k, almeno 0,5.
  - Si misura nella cella di ritmo (`sc4_train.py --time`): 200 passi dopo 20 di riscaldamento, per ogni braccio,
    con la macchina libera e il codice registrato.
  - La cella gira prima dei run lunghi: se G1 fallisce, i run non partono.
  - Lettura accanto: il ritmo contro m1 della fase 2 (2.450 token/s).
- **G2, log E.** Sul val della fase 2 (2 M token):
  - il BPB scende in modo monotono, e2k > e8k > e32k;
  - l'ultimo passo, BPB(e8k) − BPB(e32k), vale almeno 0,005: il pavimento σ della fase 2 (R1), che e 2,6 volte la σ
    misurata fra i seed (0,0019);
  - controllo banale: lo stesso guadagno sulla meta B del val, alla temperatura scelta sulla meta A, almeno 0,005;
  - accanto, la top-1.
- **Un valore non finito** ferma il run.
- **Fase 4 PASS = G1 e G2.**
- **Predizioni:**
  - G1 PASS, con un ritmo fra 0,6 e 0,9: la pipeline nasconde quasi tutto AdamW dell'host dietro il forward;
  - G2 incerto. Nella fase 2, m16 → m64 (gli stessi token per esperto di e8k → e32k) ha guadagnato 0,004, ma il
    distacco cresceva ancora a fine run, e qui il nucleo vede 4 volte i token. Stima 0,003-0,012. Non e una predizione
    contro lo scopo: con un FAIL cambia la decisione sulla fase 5 (prima piu token, o la chiusura del guadagno a
    questo nucleo).
- **Se va male:**
  - G1 FAIL: il metodo e troppo lento a 20 GB. Pagina con le leve rimaste (gradiente in bf16 sul bus, Adagrad per
    riga), nessun run lungo;
  - G2 FAIL con G1 PASS: il metodo funziona, ma a 240 M token la banca oltre la VRAM non compra BPB. La pagina legge le
    curve (§6): se il distacco e8k − e32k sta ancora crescendo, raccomanda piu token; se e piatto, chiude il guadagno a
    questo nucleo.

## 5. Costi

- **Stime dalla cella di ritmo in dry, contese** (sulla CPU girava un altro lavoro al 90%; 25 passi dopo 5):

  | Braccio | Passo | Token/s | Note |
  |---|---|---|---|
  | e2k | 6,75 s | 1.214 | picco GPU 11,0 GB |
  | e8k | ~9,5 s | ~860 | dal dry; picco 10,8 GB |
  | e32k | 9,76 s | 839 | picco GPU 10,1 GB; aggiornamento dell'host 2,66 s a passo, nascosto in parte dalla pipeline (senza pipeline: 11,4 s) |

  - Ritmo stimato di e32k contro e2k: 0,69.
  - AdamW dell'host a macchina libera: 1,31 ns per parametro (`a58aa5b`), 2,1 s a passo. Cresce linearmente con la banca.
- **Il budget**, da queste misure contese, che sono un tetto: circa 55 h per e2k, 77 per e8k e 79 per e32k. Si
  aggiungono 15 curve da circa 6 minuti per braccio e i checkpoint (e32k: 18 GB, 2 minuti ogni 3 ore). In tutto
  **~215 h di GPU**, circa 9 giorni. La cella di ritmo a macchina libera dara i numeri veri.
- **RAM dell'host per e32k:** 25,8 GB (6,4 GB di gradiente in memoria bloccata).
- **Ordine:** la cella di ritmo dei tre bracci a macchina libera, poi e32k, e8k ed e2k. I run lunghi non sono
  benchmark: possono convivere con lavori sulla CPU, a costo di allungarsi.

## 6. Letture (non gate)

- P62, K, P con nome (64 e 512), rubrica esplicita, uso degli esperti e KL dall'uniforme, norma del gradiente, curve
  ogni 2.000 passi (BPB e sonde), tempi del metodo.
- **Ripiego per D-R** (PLAN, fase 4): si applica la regola della rubrica esplicita (margine di almeno 0,10 e 4 errori
  standard sul controllo piu forte) al run di e32k. Se non passa, dopo la fase 4 si apre una fase D-R con un nucleo
  piu grande.

## 7. Letteratura e modi noti di fallire

In [`LITERATURE.md`](LITERATURE.md), aggiunte del 2026-11-27 e del 2026-11-28.
- **Chiavi prodotto** (PKM, arXiv 1907.05242; PEER, arXiv 2407.04153; Memory Layers at Scale, arXiv 2412.09764).
  La ricerca e esatta (PKM sez. 3.1).
  - **Uso delle chiavi che crolla** con memorie grandi. In PKM senza BatchNorm l'uso e 98,97% a 65k chiavi e 25,8% a
    1M (tab. 2). Qui E = 32.768 e c'e la perdita di bilanciamento della fase 2. Lettura: l'uso e la KL per strato.
  - **Instabilita** con memorie grandi su nuclei piccoli (Memory Layers sez. 3.1.4), che usano la qk-norm "when needed".
    Lettura: la norma del gradiente; un valore non finito ferma il run.
- **Pesi quantizzati nel trasferimento:** il precedente piu vicino e il float8 all-gather di FSDP2. Qui la matematica
  non cambia (aggiunte del 2026-11-27).

## 8. Cosa toglie al design

| Elemento | Qui | Perche |
|---|---|---|
| Molti piu token per esperto della fase 2 (PLAN) | e32k ne ha 0,94 M, quanti m64 nella fase 2; e8k 3,75 M (m16); e2k 15 M | I 20 GB fissano E. Quattro volte m64 a e32k vorrebbe 960 M token, circa 4 volte questo budget. Le curve dicono se il distacco cresce ancora |
| Un gemello solo-GPU di e32k | non esiste: non entra | Equivalenza a E = 2.048 (§3.1) e nella fase 3 su 7.324 passi |
| Due seed per braccio | un seed | Costo; σ dalla fase 2. La soglia di G2 sta sopra 2,6 σ misurate |
| Teste multiple, BatchNorm sulla query, qk-norm (PKM, PEER, Memory Layers) | una testa, top-128, la perdita di bilanciamento della fase 2 | Tiene il gating della fase 2 identico; i modi di fallire si leggono (§7) |
| lr piu alto per i valori (PKM) | lr unico | La ricetta della fase 2 |
| La sonda T (legame) nel flusso | tolta | Le letture scelte sono K, esplicita e P con nome |
| Gradiente in bf16 sul bus, Adagrad per riga | no | Cambiano la matematica; solo se G1 non basta |
| Stesso micro-batch per tutti | e8k a 8 con accumulo 2 | A 16 il picco e 15,5 GB su 12; e la stessa media della perdita |
