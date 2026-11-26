# F3: il metodo, la banca nella RAM in streaming, il protocollo

2026-11-26. Fase 3 del [piano](PLAN.md), domanda **D-M** (il metodo). Codice in `benchmarks/sparse_capacity/`
(`sc3_stream.py`, `sc3_train.py`); il verdetto lo calcola `sc3_verdict.py` con le regole del §5, scritte qui prima del
run.

**Domanda.** Il trainer che tiene la banca nella RAM e la porta sulla GPU a ogni passo allena come il trainer solo-GPU
della fase 2, e quanto costa per parametro della banca?

**Riga sulla tesi.** E un test di meccanismo. La banca di m16 (101 M parametri, 1,6 GB con lo stato) entra nella VRAM:
qui serve, perche il gate chiede un riferimento solo-GPU. Un PASS dice che il metodo e giusto, sul percorso che servira
oltre la VRAM, e ne misura il costo per parametro. Non mostra una banca oltre la VRAM: quella e la fase 4. Budget da
test di meccanismo: un run.

## 1. Il metodo

Fonti, gia lette in [`LITERATURE.md`](LITERATURE.md): OOM (master fp32 e momenti di Adam sull'host, pesi portati sulla
GPU, AdamW sulla CPU), ZeRO-Offload (gradienti e passo dell'ottimizzatore sulla CPU), ES-MoE (esperti e stato
sull'host, portati sulla GPU in pipeline).

- **Dove sta cosa:**
  - **host (RAM):** i pesi degli esperti in fp32 (Wg, Wu, Wd), i loro gradienti e i due momenti di AdamW;
  - **GPU:** nucleo, router, embedding e testa, con il loro AdamW, come nella fase 2.
- **Un passo:**
  1. in ogni strato il router gira sulla GPU e sceglie i top-128 per token;
  2. l'unione U degli esperti scelti nel batch si raccoglie sull'host in un buffer bloccato (uno solo, grande come la
     banca di uno strato) e si copia sulla GPU;
  3. la matematica della banca gira solo su U. Le scale ternarie sono per riga di esperto, quindi restringere a U non
     cambia nulla. Sulla GPU restano solo gli ingressi piccoli, non i pesi;
  4. nel backward i pesi di U si riprendono dall'host, la banca si ricalcola, e i gradienti di U tornano sull'host. Con
     il checkpoint delle attivazioni della fase 2 i pesi passano tre volte a passo: forward, ricalcolo, backward;
  5. il clip della norma conta i gradienti della GPU e quelli di U, nello stesso ordine e con la stessa formula di
     `clip_grad_norm_`;
  6. AdamW sulla GPU per il resto, AdamW sull'host (fused, 6 thread) per tutta la banca.
- **Sincrono.** Nessun aggiornamento ritardato: ZeRO-Offload ne usa uno di un passo, ma cambia la matematica.
- **Modi noti di fallire:**
  - il passo di Adam sulla CPU: ES-MoE lo trova "31x slower", la 1b 2 ns per parametro. L'AdamW fused di PyTorch per
    la CPU fa 1,08 ns per parametro su 101 M (misurato con la GPU occupata da un altro lavoro);
  - l'aggiornamento ritardato: qui non c'e;
  - i pesi fuori sync fra host e GPU: li cercano i controlli del §3.

## 2. Cosa si confronta

- **Riferimento:** `f2_m16_s0`, gia allenato e committato. BPB val 1,0039, top-1 0,6217, 12,23 h (1.363 token/s).
- **Run:** lo stesso braccio con lo stesso seed. Stesso codice per dati, ordine dei batch, schedule, valutazione e
  sonde (`sc2_train`); stessa inizializzazione (il modello si costruisce sull'host come nella fase 2, poi gli esperti
  restano li). Il runner rifiuta se il codice della fase 2, torch o i dati non sono quelli del riferimento.
- **Cambia una cosa sola:** dove vivono gli esperti e come si aggiornano.
- **U.** Di solito U = E (512 token per esperto a passo), ma non sempre: nel riferimento 12 passi registrati su 148
  hanno meno del 100% degli esperti usati (al minimo il 99,4%). Il run registra il minimo e il massimo di U.

## 3. Controlli piantati e banali

- **Selftest** (14 controlli, piu i 15 della fase 2):
  - la banca in streaming contro quella della fase 2, con U piu piccolo di E: forward e gradienti identici, gradiente
    zero fuori da U, pesi presi due volte (forward e backward) e non tenuti;
  - checkpoint acceso e spento danno gli stessi gradienti;
  - **un passo dallo stesso stato:** il riferimento fa 3 passi, il suo modello e i suoi momenti si copiano nel modello
    in streaming, poi entrambi fanno un passo. Loss, norma prima del clip, banca e resto coincidono entro 1e-6
    relativo (sulla CPU coincidono esatti);
  - **negativo piantato:** lo stesso passo con una banca lasciata ferma esce di oltre 1e-3 (esce di 7e-2);
  - **ripresa:** un checkpoint scritto e letto con il codice del run, poi il passo dopo, coincide esatto;
  - i pesi finali si caricano nel modello della fase 2 e danno gli stessi logit.
- **Dry alle forme vere, sui dati veri, dallo stato allenato del riferimento** (il suo `last.pt` al passo 7.324, con i
  momenti): tre passi dallo stesso stato, il negativo piantato, la ripresa e i tempi. I numeri sono in
  `results/sparse_capacity/f3_dry_20261126_155517.json` (§8).
- **Nel run, il passo 0:** loss e norma del gradiente prima del clip devono essere quelle del riferimento (7,7810078 e
  0,71231025) entro 1e-6 relativo. Altrimenti il run si ferma subito: VOID. Una banca intera diversa sposta la loss al
  passo 0 di circa 0,002 (m4 e m64 contro m16), quindi una tolleranza di 1e-3 non basterebbe (revisore, obiezione 1).
- **Perche non un confronto su tanti passi.** I pesi ternari scattano su differenze di 1e-7, e due traiettorie giuste
  si separano dopo pochi passi: sulla CPU, prima di allineare l'ordine del clip, 1e-3 di loss in 6 passi. Per questo
  la correttezza si giudica dallo stesso stato, e il run sul rumore di seed.
- **Il gate del piano da solo non distingue un altro seed.** Provato con `f2_m16_s1` al posto del run: passa (−0,0021,
  curva al peggio 0,0073). Per questo c'e anche la condizione stretta del §5, e l'esattezza la danno i controlli qui
  sopra.
- **Controllo banale del BPB:** la temperatura scelta sulla prima meta del val, come nella fase 2. Accanto al BPB,
  sempre la top-1.

## 4. Misure

- BPB e top-1 sul val (2 M token), le curve ogni 1.000 passi, P62, K e R: lo stesso codice della fase 2.
- **Ritmo:** token del run diviso ore di training, valutazioni delle curve comprese, come per il riferimento.
- **Tempi del passo** ogni 50 passi, con sincronizzazione: forward e backward, raccolta e copia degli esperti (e
  quante per strato), ritorno dei gradienti, AdamW sulla GPU, AdamW sull'host, minimo e massimo di U.

## 5. Esiti (registrati)

σ = 0,005, il pavimento R1 e il gate del piano. σ fra seed misurata nella fase 2: 0,0019.

| Esito | Condizione | Decisione |
|---|---|---|
| VOID | run non pushato o con modifiche, passi diversi da 7.324, valore non finito, passo 0 diverso dal riferimento (§3), codice della fase 2, torch o dati diversi dal riferimento, punti delle curve diversi | si corregge e si ripete |
| **PASS** | tre condizioni insieme: (1) piano: \|BPB(stream) − BPB(rif.)\| ≤ 2σ = 0,010 e ogni punto della curva entro 0,010; (2) stretta: BPB(stream) − BPB(rif.) ≤ 2·√2·0,0019 = 0,0054 (due deviazioni della differenza fra due run, da un lato solo: un errore sottile peggiora); (3) ritmo ≥ 0,5 di quello del riferimento | si dettaglia la fase 4 (PLAN) |
| FAIL | manca una delle tre | una pagina con una raccomandazione (regola 7) |

**Letture, non gate:**
- i tempi del passo e il costo del metodo per parametro della banca (raccolta, ritorno, AdamW sull'host);
- **la proiezione per la fase 4,** lineare: costo a 20 e 40 GB di banca con lo stato, se ogni esperto viene toccato,
  e il ritmo contro il passo di m1 della fase 2 (3,62 s, il calcolo attivo). E un limite inferiore del costo: niente
  pressione sulla VRAM, niente kernel per esperto;
- P62, K, R e la differenza di top-1;
- il carico della CPU da altri lavori (dalle prenotazioni della macchina).

## 6. Predizione (registrata)

- **PASS.** |ΔBPB| ≤ 0,005: un seed uguale non e una traiettoria uguale (§3), quindi lo scarto e del livello del
  rumore di seed.
- **Ritmo 0,93-0,99 contro il riferimento.** Dal dry, 1.341 token/s senza le valutazioni contro 1.363 con le
  valutazioni: il metodo costa circa il 5% del passo (§8).
- **Il gate del ritmo non puo fallire a m16** (revisore, obiezione 4). Il riferimento passa quasi tutto il passo nel
  prodotto denso su tutti gli E·h, che il metodo rifa uguale. Il numero che conta e il costo per parametro.
- **Predizione contro lo scopo, per la fase 4 (regola 5).** Con il costo del dry (3,1 ns per parametro), a 20-40 GB
  di banca il passo cresce di 3,9-7,9 s. Contro il calcolo attivo di m1 il ritmo scende a 0,48-0,32, **sotto lo 0,5 del
  piano** (§8). Quindi la fase 4, cosi com'e, fallirebbe il suo ritmo.
  - **La decisione che cambia:** il protocollo della fase 4 deve togliere almeno la meta del costo per parametro, con
    una o piu leve gia misurate: Adagrad per riga sull'host (3-4 volte meno di Adam nella 1b, con il suo gate di
    qualita), pesi in bf16 sul bus, sovrapporre trasferimenti e calcolo, togliere il terzo passaggio dei pesi.
  - **Il ritmo "solo-GPU" oltre la VRAM** non ha un riferimento allenato. Si propone il passo di m1 (stesso nucleo,
    stesso k, stesso calcolo attivo, tutto sulla GPU): da decidere a fine fase con il piano.

## 7. Cosa toglie al design

| Elemento | Qui | Perche, e in che verso puo sbagliare |
|---|---|---|
| Banca oltre la VRAM (la tesi del metodo) | m16, 1,6 GB con lo stato | il gate chiede un riferimento solo-GPU, che oltre la VRAM non esiste. La fase 4 la porta oltre |
| Portare solo gli esperti scelti | implementato (U); a m16 U e quasi sempre E | il caso U < E lo provano il selftest e i pochi passi del run. Il trasferimento misurato e quasi pieno: un limite superiore |
| Kernel per esperto (grouped GEMM) | no: prodotto denso sugli U·h, come il riferimento | serve per l'equivalenza esatta. Oltre la VRAM, con U ≫ k, costa U/k volte il calcolo attivo e somma in un altro ordine: la fase 4 deve cambiarlo, e quel kernel qui non e certificato |
| Sovrapporre trasferimenti e calcolo | no: raccolta sincrona strato per strato | il costo misurato e un limite superiore |
| Pesi presi tre volte a passo | si (forward, ricalcolo del checkpoint, backward) | il minimo e due. Il costo misurato li conta tutti e tre |
| Buffer bloccato | uno solo, grande come la banca di uno strato (4 byte per parametro / L) | oltre la VRAM va fatto a pezzi se uno strato supera la memoria bloccabile. Windows limita le allocazioni bloccate grandi: non provato oltre 50 MB per nome |
| AdamW solo sugli esperti toccati | no: AdamW denso su tutta la banca, come il riferimento | uno sparso cambia la matematica (momenti e weight decay di chi non e scelto). Oltre la VRAM costa quanto tutta la banca: e la voce che la proiezione moltiplica |
| Pesi in bf16 sul bus (OOM) | fp32 | servono per l'equivalenza esatta. bf16 dimezza i byte del trasferimento |
| Adagrad per riga (1b: da 3 a 4 volte piu veloce sull'host) | AdamW | prima serve il suo gate di qualita (b) |
| Accumulo dei gradienti | no, un micro-batch per passo (m16 lo ha gia) | m64 usava l'accumulo 4: va aggiunto per la fase 4 |
| Un posto per esperto sulla CPU | no | la 1b (GRIGIO) dice streaming |
| Macchina dedicata | no | il ritmo ha un gate largo. Il carico degli altri lavori si legge dalle prenotazioni |

## 8. Revisore indipendente e dry

Un revisore senza la conversazione ha letto la tesi, il piano, i risultati della 1b e della fase 2, questo protocollo
e il codice, e ha sollevato 9 obiezioni.

| # | Obiezione | Risposta |
|---|---|---|
| 1 | Bloccante: il passo 0 a 1e-3 non distingue una banca sbagliata (una banca intera diversa sposta la loss di 0,002) e non ferma il run | Corretto: loss e norma prima del clip entro 1e-6 relativo dai valori del riferimento, e il run si ferma subito |
| 2 | Stato allenato e ripresa mai provati | Corretto: dry dallo stato del riferimento al passo 7.324, con i momenti; ripresa provata nel selftest e nel dry |
| 3 | Il gate a 2σ = 0,010 e 5 volte lo scarto fra seed misurato | Aggiunta la condizione stretta, da un lato: ≤ 0,0054 (§5) |
| 4 | Il ritmo non puo fallire a m16, e il denominatore e gonfiato dal prodotto denso | Dichiarato (§6); proiezione per la fase 4 contro il passo di m1 e predizione contro lo scopo con la decisione che cambia |
| 5 | Il codice certificato non e quello della fase 4 (pesi tenuti sulla GPU, buffer grande come tutta la banca) | Corretto: i pesi si riprendono nel backward e il buffer e uno, grande come uno strato. Il kernel per esperto resta fuori, dichiarato (§7) |
| 6 | Fatti sbagliati: U = E sempre; top-1 del riferimento | Corretti (§2) |
| 7 | Il runner non controlla che il riferimento sia lo stesso codice, torch e dati | Corretto: rifiuta, e il verdetto da VOID |
| 8 | Il dry non e registrato | Corretto: `f3_dry_<data>.json` committato prima del push |
| 9 | La stima per la fase 4 e ottimista | Etichettata come limite inferiore del costo (§5) |

**Dry** (`results/sparse_capacity/f3_dry_20261126_155517.json`, commit `3e568b2`): forme vere, dati veri, dallo stato
del riferimento al passo 7.324, lr del passo 7.000.
- **Un passo dallo stesso stato,** tre volte: loss, norma prima del clip e resto identici (0); banca entro 1,3e-7
  relativo, cioe AdamW sulla GPU contro AdamW sull'host.
- **Negativo piantato** (banca dello strato 0 ferma): 6,1e-4 sulla banca, 4.700 volte lo scarto del metodo giusto.
- **Ripresa:** loss e parametri identici.
- **U:** 2.048 esperti su 2.048 in ogni strato; pesi presi 3 volte a passo.
- **Tempi del passo** (mediana, s): 6,11 in tutto; dentro forward e backward, raccolta 0,167 e ritorno dei gradienti
  0,060; AdamW sull'host 0,089, sulla GPU 0,002. Sono 1.341 token/s senza le valutazioni, contro i 1.363 del
  riferimento con le valutazioni.
- **Costo del metodo:** 0,316 s a passo, 3,1 ns per parametro della banca (il 5% del passo a m16).
- **Proiezione per la fase 4** (limite inferiore del costo): a 20 GB di banca con lo stato +3,9 s a passo, ritmo
  contro m1 al massimo 0,48; a 40 GB +7,9 s, al massimo 0,32. Sotto lo 0,5 del piano in tutti e due i casi (§6).
- **Memoria:** il dry tiene insieme il riferimento e il modello in streaming, e arriva a 11,0 GB di VRAM; il run ne
  tiene uno solo.

## 9. Esecuzione

- **Dove:** sulla 3060, con la CPU libera per l'AdamW dell'host (6 thread). Il run si mette in coda nelle prenotazioni
  della macchina e parte da solo quando la GPU e la CPU sono libere.
- **Comando:** `python benchmarks/sparse_capacity/sc3_train.py --run --data-dir <pacchetto m0> --ckpt-dir <E:>`.
  - Il runner rifiuta se l'apparato ha modifiche non committate, se HEAD non e pushato, se la GPU non e la 3060, se il
    riferimento manca o e diverso (codice, torch, dati), o se uno dei selftest fallisce. Stampa e salva il CONFIG.
  - Si ferma su un valore non finito e sul passo 0 diverso. Salva ogni 30 minuti (pesi dell'host e momenti compresi);
    dopo un blocco si riparte con `--resume`.
- **Durata:** circa 12,5 h (dal dry: il passo come il riferimento, piu le valutazioni).
- **Verdetto:** `python benchmarks/sparse_capacity/sc3_verdict.py`.
- **Artefatti:** `results/sparse_capacity/f3_stream_m16_s0/` (`config.json`, `train.jsonl`, `result.json`). I pesi
  finali, nel formato del modello della fase 2, restano fuori da git, con lo SHA-256 nel log.
