# F1b: magazzino contro officina, il protocollo

2026-11-17. Fase 1 del [piano](PLAN.md), prova 1b, domanda **D-M**. Codice in `benchmarks/sparse_capacity/`. Il runner
stampa solo numeri; il verdetto lo calcola `sc1b_verdict.py` con le regole del §6, scritte qui prima del run.

**La domanda.** Sul 3600X con la 3060, sotto quanti token per esperto per passo (r\*) costa meno addestrare un esperto
sulla CPU (forward, backward e aggiornamento) invece di portarne i pesi sulla GPU e riportarne il gradiente?

**Riga sulla tesi.** La misura e alla scala della tesi di D-M: questo PC; una fetta da 201 M parametri (805 MB in fp32,
25 volte la L3) della banca di uno strato; passi da 16k token. Non e un test di meccanismo. Pero la predizione (§3) e
che, anche se r\* esce ≥ 2, alla scala voluta il guadagno sul passo sia piccolo:
- **la banca:** 3-30 G parametri in 60 GB;
- **la r media:** da 3 a 24 con l'instradamento uniforme, che lascia pochi esperti sotto r\*;
- **il guadagno sul passo:** ≤ 2% a h8 e zero a h1.

Il valore della misura sta altrove:
- **le curve,** che la fase 3 compone per l'instradamento vero, sbilanciato, dove gli esperti freddi sono di piu;
- **la possibilita di chiudere D-M,** se la CPU perde ovunque.

## 1. Cosa si cronometra

Un passo di addestramento della fetta di banca: E esperti, ognuno con esattamente r token. Il tempo diviso per i
parametri della fetta da i ns per parametro per passo.
- **Esperto:** dReLU con gate, `relu(Wg·x) * relu(Wu·x)`, poi la down; D = 256, h ∈ {128, 8, 1}.
- **Fetta e attivi fissi:** 201.326.592 parametri, quindi E = 2.048, 32.768 e 262.144. Gli attivi per token sono fissi
  a k·h = 1.024 come nel rung-1.
- **Token:** le E·r assegnazioni pescano da un passo di T = max(16.384, E·r·h/1.024) token, arrotondato alla potenza di
  2. I token di un esperto sono distinti e sparsi, e le righe dei token stanno in DRAM come nel regime.
- **Costi fissi dello strato:** trasferimento delle attivazioni, azzeramento e riduzione degli accumulatori per
  thread. Si cronometrano a parte e si addebitano alla fetta in proporzione alle sue assegnazioni: f = E·r·h / (T·1.024).
  Il tempo grezzo si registra accanto.
- **Percorso CPU** (bracci C):
  - x e dy passano dalla GPU all'host, y e dx tornano (bf16);
  - una passata di forward, poi una di backward con aggiornamento;
  - 6 thread, uno per core fisico, fissati; il pool di torch e a 1 thread, `OMP_WAIT_POLICY=PASSIVE`.
- **Percorso streaming** (bracci S):
  - pesi bf16 dall'host alla GPU a blocchi di 32 MiB su tre stream CUDA, con quattro slot;
  - forward e backward in bf16 sulla GPU;
  - gradiente bf16 indietro e aggiornamento sulla CPU, sovrapposto ai blocchi successivi. Fa eccezione S3, che
    aggiorna sulla GPU e riporta il master.
  - **I pesi passano due volte sul PCIe,** una nel forward e una nel backward. La banca non entra nella VRAM, quindi
    non resta nulla fra le due passate.
  - Il tempo di GPU occupata si registra per passata con eventi CUDA.
- **Stato dell'ottimizzatore:** si confrontano solo coppie con lo **stesso formato, lo stesso stato e la stessa
  densita**. Ogni aggiornamento e cronometrato su tutti e due i lati.

| Braccio | Percorso | Pesi | Ottimizzatore | Note |
|---|---|---|---|---|
| C1 | CPU | fp32 | SGD | come autograd: gradiente denso scritto, poi una passata di aggiornamento |
| C2 | CPU | fp32 | SGD | (a) backward e aggiornamento fusi |
| C3, C3s, C3d | CPU | fp32 | Adagrad per riga, fuso | (b) un float di stato per riga; densita primaria / 0,065 / 1 |
| C4 | CPU | fp32 | Adam denso, fuso | lo stato di OOM (master e momenti fp32) |
| C5 | CPU | bf16 + arrotondamento stocastico | Adagrad per riga, fuso | (c) |
| C6 | CPU | ternario (codici del motore) + latente fp32 | Adagrad per riga, fuso | (c) attivazioni a 63 livelli; forward con la LUT pshufb di `engine.c` a h128, LUT scalare a h8 e h1 (§8) |
| S1 / S2, S2s, S2d / S3 / S4 | streaming | come C4 / C3 / C5 / C6 | idem | coppie del §4 |

**Densita.** Un'unita nascosta senza token attivi (g > 0 e u > 0) ha gradiente zero.
- SGD e Adagrad per riga la saltano in modo esatto, e cosi fanno C e S2/S4 nell'aggiornamento sulla CPU.
- Adam denso no: i suoi momenti si muovono lo stesso.
- **Densita primaria:** 0,5 a h128 e h8; **1 a h1**, perche un micro-esperto instradato e acceso sui suoi token (un
  neurone PEER non ha mai gradiente zero). 0,065 e quella misurata sul rung denso addestrato (E3,
  `docs/ENGINE_PLAN.md`).
- **Come si imposta:** con uno scostamento sulla colonna 0 dei pesi gate e up (x₀ = 1). Per i ternari si imposta dal
  token (x₀ = |MU|·√(0,69·255)), perche il quantizzatore absmean schiaccia lo scostamento a una scala.
- **Si misura** in ogni cella e si riporta. Nel collaudo i ternari danno 0,49 e 0,998, uguali in C6 e S4.

## 2. Letteratura dei bracci, con i modi noti di fallire

- **(a) Fusione.** LOMO (arXiv 2306.09782, §3.2): "update the parameter immediately when its gradient is computed".
  - Modo di fallire (§3.3.1): "cannot directly compute the gradient norm". Il clipping per norma costa un secondo
    backward, e LOMO usa SGD semplice.
  - FBGEMM TBE (documentazione): "The backward operator is fused with optimizer".
- **(b) Stato compatto.** `EXACT_ROWWISE_ADAGRAD` di FBGEMM tiene `momentum1 (rowwise)`.
  - Adafactor (arXiv 1804.04235) fattorizza il secondo momento in O(n+m).
  - Modo di fallire (§5-6): instabilita senza warmup con β2 alto, curata dal clipping dell'update.
  - Adam sparso o "lazy" non entra: cambia la semantica dei momenti.
- **(c) bf16.** Zamirai et al. (arXiv 2010.06192, §3.1): l'arrotondamento al piu vicino "cancels small model updates";
  quello stocastico "is not a silver bullet".
- **(c) Ternario.** Pesi latenti e STE (BitNet b1.58, arXiv 2402.17764); semantica del motore in
  [`F0_QUANT_AUDIT.md`](F0_QUANT_AUDIT.md).
- **Streaming.** OOM (arXiv 2611.11364, §4.1, §7.1): master fp32 e momenti sull'host, pesi bf16 alla GPU, AdamW sulla
  CPU.
  - Quante volte i pesi passano sul PCIe in un passo non e scritto: il "due volte" del §1 e derivato, non letto.
  - Non tratta la scelta CPU/GPU per esperto.
- **La qualita di (b) e (c) non si misura qui.** Se la fase 3 li adotta, servono prima i loro gate di qualita.

## 3. Byte per parametro, e la predizione a tavolino

**I byte.** Byte per parametro per passo, solo pesi, densita 0,5, con a(r) = 1 − 0,5^r unita attive.
- **Lato CPU, r = 1 / r = 8:** C1 21,3 / 24; C2 e C3 7,3 / 12; C4 27,3 / 28; C5 3,7 / 6; C6 4,7 / 9.
- **Lato streaming:**
  - PCIe in andata 2+2 (forward, backward) e in ritorno 2; per S4 0,5+0,5 e 2.
  - RAM: S1 34; S2 12 / 18; S3 6; S4 8,3 / 13,5.
- **Il conto ingenuo di LITERATURE §3 cambia per tre ragioni:**
  - lo streaming paga anche l'aggiornamento sulla CPU: per S1 sono 32 byte di RAM contro 6 di PCIe;
  - i pesi vanno sulla GPU due volte;
  - la CPU salta le unita spente.
- **I micro-esperti pagano le attivazioni.** Ogni coppia (esperto, token) muove circa 7 vettori D sulla CPU, cioe
  9,3/h byte per parametro per token: a h1 sono piu dei pesi gia a r = 1.
- **Lo streaming ideale** (riferimento del gate, §4): bande di picco misurate, nessun calcolo sulla GPU, e nel backward
  solo le righe attive in tutti e due i versi (Adam denso escluso).

**La predizione (registrata).**
- **Ipotesi:**
  - RAM 30 GB/s, CPU 200 GFLOP/s efficaci, (4 + 14·densita)/3 FLOP per parametro per token;
  - "add" = memoria + calcolo, "ovl" = il massimo dei due;
  - PCIe 4.0 a 25 GB/s per verso, 20 in tutti e due i versi insieme.
- **Desk 2:** il runner ripete la tabella con le bande misurate prima dei bracci. Nel collaudo: RAM 31, PCIe 26 per
  verso e 13 in tutti e due i versi insieme, link gen 4 x16.

| Coppia (r\* add · contro lo streaming ideale · ovl) | h128 | h8 | h1 (densita 1) |
|---|---|---|---|
| fp32 + Adam (C4/S1) | 10 · 10 · 55 | **3,8 · 3,8** · 20 | < 1 · < 1 · 3,4 |
| fp32 + Adagrad per riga (C3/S2) | 10 · 10 · 30 | **3,6 · 3,3** · 11 | < 1 · < 1 · 1,8 |
| bf16 + Adagrad per riga (C5/S3) | 2,2 · 1,1 · 10 | 1,3 · < 1 · 3,7 | < 1 |
| ternario (C6/S4) | 7,4 · 7,3 · 22 | **2,5 · 2,1** · 7,9 | < 1 · < 1 · 1,3 |

- **Con PCIe 3.0** (12 e 10) gli r\* "add" salgono a 14 / 5,3 / < 1 (Adam), 8,1 / 3,2 / < 1 (bf16) e 8,4 / 2,9 / < 1
  (ternario).
- **Verdetto previsto: ANGOLO CPU, solo a h8.** A h1 la CPU perde gia a r = 1: muovere le attivazioni costa piu dei
  pesi.
- **Guadagno sul passo previsto** (lettura, §4), con la r media del regime e l'instradamento uniforme:
  - h8: 0,1% (fp32 + Adagrad per riga), 1,8% (bf16), 0,1% (ternario), 0 (Adam);
  - h1: 0;
  - a 128k token per passo: 0 ovunque.

**La r del regime.** Banca che sta in 60 GB, 8 strati, instradamento uniforme: r = T · 3AD / parametri per strato. Non
dipende da h.

| Formato (byte per parametro con lo stato) | Parametri | T = 16k | T = 128k |
|---|---|---|---|
| fp32 + Adam (14) | 4,3 G | 24 | 192 |
| fp32 + Adagrad per riga (6) | 10 G | 10 | 82 |
| bf16 + Adagrad per riga (2) | 30 G | 3,4 | 28 |
| ternario, latente fp32 (4,5) | 13 G | 7,7 | 62 |

## 4. La metrica

- **r\*(coppia, h):** il piu piccolo r a cui la mediana del braccio C supera quella del braccio S, interpolato in log r
  sulla griglia r ∈ {1, 2, 4, …, 128}.
  - Vale "< 1" se la CPU perde gia a r = 1, "> 128" se vince ovunque.
- **Intervallo [lo, hi]:**
  - lo: la CPU al suo peggio (il massimo dei 12 campioni) contro lo streaming al suo meglio, cioe il minore fra il
    minimo misurato e lo streaming ideale del §3 calcolato con la densita misurata;
  - hi: la CPU al suo meglio contro lo streaming misurato al suo peggio.
  - Si riporta anche lo con il solo streaming misurato.
- **Coppie:**
  - del gate: fp32-adam, fp32-rowada, bf16-rowada, tern-rowada;
  - di sensibilita: rowada a densita 0,065 e 1 (a h1 la densita 1 e gia la primaria).
- **Curve complete:** si riporta, per ogni braccio, h e r, la mediana, il minimo, il massimo e i tempi di forward,
  backward, costi fissi e GPU occupata.
- **Lettura, non gate:** il guadagno sul passo della banca, calcolato da `sc1b_verdict.py`:
  - ogni esperto riceve r di Poisson con la media del regime (§3, T = 16k e 128k);
  - va dove costa meno;
  - il guadagno e uno meno il costo cosi ottenuto diviso il costo tutto in streaming.
  - La fase 3 rifa lo stesso conto con la distribuzione del router vero.

## 5. Controlli

- **Banali (all'inizio, e RAM e FMA anche alla fine):**
  - RAM a 6 thread: lettura, scrittura non temporale, scrittura, lettura-modifica-scrittura;
  - PCIe con memoria pinned: andata, ritorno, i due versi insieme, e la pageable come riferimento;
  - generazione e larghezza del link lette **durante** una copia.
- **Pavimento:** un tempo sotto 0,9 volte i byte dei pesi divisi per la banda di picco misurata e una misura sbagliata,
  e la cella diventa VOID.
  - Scatta su un positivo noto: nel dry run la banca da 12 MB sta nella L3 e lo fa scattare.
  - Non ha termini di calcolo ne di attivazioni, e lo streaming lo usa a densita zero: prende gli errori grossolani di
    cronometro, non un braccio rotto. Per quello servono i piantati e il selftest.
- **Piantati:** sono la precondizione; il runner si ferma se escono dalla banda.
  - (P-cpu) il braccio C2 vero con nessuna unita attiva. Deve leggere le righe gate e up una volta (8/3 byte per
    parametro) e nient'altro, quindi il tempo atteso si ricava dal controllo di lettura RAM. Banda 0,8-1,5.
    - Nel collaudo a piena taglia: 1,21-1,39 su 5 chiamate. La banda 0,8-1,25 scritta prima del collaudo sarebbe
      scattata a caso.
  - (P-str) lo streaming senza calcolo ne aggiornamento: fra 0,8 e 1,3 volte quanto prevede il controllo PCIe (1,01
    nel collaudo).
  - FMA a 6 thread: frequenza implicita fra 2,8 e 4,7 GHz. Si ripete prima di ogni gruppo (giro, h): una deriva oltre
    il 10% rispetto all'inizio, anche a meta run, rende il run VOID.
- **Correttezza (selftest, all'inizio del run):** ogni braccio C e S contro un riferimento fp64 (y, dx e
  l'aggiornamento), piu il kernel LUT bit a bit contro il prodotto intero. 710 controlli. Copre:
  - h ∈ {128, 32, 8, 1} e r ∈ {1, 2, 3, 4, 9};
  - token pescati da un pool e un ultimo blocco di streaming parziale;
  - un secondo passo per i bracci fp32, per l'accumulo di Adagrad e la correzione di Adam;
  - per il ternario anche i codici riscritti.
- **Pesi non finiti:** fermano il run (PLAN §3).

## 6. Esiti e cosa decidono

Il gate legge le quattro coppie del gate a h ∈ {8, 1}, con lo calcolato contro lo streaming ideale (§4).

| Esito | Condizione | Decisione |
|---|---|---|
| VOID | un controllo fuori banda, una cella sotto il pavimento, campioni mancanti, pesi non finiti | si corregge e si ripete; nessuna lettura |
| **ANGOLO CPU** | in almeno una coppia e una h micro, r\*_lo ≥ 2 | la fase 3 sceglie il posto per esperto con queste curve. Prima di costruirlo, il suo protocollo calcola il guadagno sul passo con la distribuzione di r del router vero, e si va avanti solo se il guadagno supera la dispersione |
| **STREAMING** | in tutte le coppie a h8 e h1, la CPU al suo meglio perde contro lo streaming misurato al suo peggio gia a r = 1 | il metodo e lo streaming (OOM); D-M diventa ingegneria e si decide se il programma continua per D-R |
| GRIGIO | altrimenti | come STREAMING; le curve restano per la fase 3 |

**Letture che non sono gate:**
- h128;
- la sensibilita alla densita;
- il guadagno sul passo;
- la quota di GPU occupata nello streaming;
- quale formato vince: decide quale gate di qualita (b o c) deve precedere la fase 3;
- h8 contro h1: entra nella scelta di granularita rinviata dal piano.

## 7. Dispersione

- **Ripetizioni:** 3 giri; in ogni giro, per cella, 1 passo di riscaldamento e 4 cronometrati. Sono 12 campioni per
  cella, riportati come mediana e minimo-massimo.
- **Ordine:** dentro una coppia C e S si alternano cella per cella, e l'ordine si inverte a ogni r e a ogni giro, cosi
  le due meta di un confronto stanno a pochi secondi l'una dall'altra. L'ordine dei gruppi ruota a ogni giro.
- **Banca e token:** stessi pesi iniziali e stessi token per i due bracci di una coppia a ogni (h, r).

## 8. Cosa toglie al design

| Elemento | Qui | Perche, e in che verso puo sbagliare |
|---|---|---|
| Nucleo, router e attenzione sulla GPU | non cronometrati | uguali nei due percorsi |
| CPU e GPU in parallelo durante il passo | ogni percorso e cronometrato da solo | in un passo vero la CPU lavora mentre la GPU fa il nucleo (a favore della CPU), ma i due percorsi si dividono la RAM (contro). Il costo del sistema misto e della fase 3 |
| Instradamento sbilanciato | ogni esperto ha esattamente r token | servono le curve per r; la composizione e della fase 3 |
| Costi fissi della CPU in un sistema misto | addebitati in proporzione alle assegnazioni | se sulla CPU va solo la coda fredda, le attivazioni di tutti i token che la toccano costano di piu per esperto. Il tempo grezzo e registrato |
| Densita dReLU di un modello vero | imposta, tre livelli; 1 a h1 | 0,5 e un'ipotesi; 0,065 viene da un modello denso |
| Peso di instradamento | non applicato | e un prodotto per scalare |
| Esperto PEER (un neurone u·σ(v·x), 2D parametri, chiavi prodotto) | dReLU con gate a h1 (3D parametri), senza chiavi prodotto | la stessa famiglia dei bracci h128 e h8; la densita 1 ne copre il gradiente mai nullo |
| Ternario: forward con la LUT pshufb del motore a ogni h | LUT pshufb solo a h128; a h8 e h1 LUT scalare, una per token, 2 KB | i blocchi da 32 righe del motore non esistono sotto h32 (F0 §3). Svantaggia C6 proprio a h8 e h1: il ternario CPU e un limite inferiore |
| Ternario: down con la LUT e scala per uscita | riga della down spacchettata in fp32, scala per riga di WdT | i byte letti restano 0,5 per parametro |
| S4 con attivazioni quantizzate | x in bf16 sulla GPU | conta i byte, non la numerica |
| Streaming scritto bene | PyTorch semplice (gather, bmm, index_add, SR in piu passate) | svantaggia lo streaming, soprattutto S3, S4 e h1. Per questo il gate lo legge contro lo streaming ideale |
| Adam sparso, Adam a 8 bit | esclusi | cambiano la semantica o la qualita; si aggiungono solo con un gate di qualita |
| Clipping per norma globale con l'aggiornamento fuso | assente | LOMO: serve un secondo backward. Costo per la fase 3 |
| Esperti residenti in VRAM (OOM) | assenti | valgono uguale per i due percorsi |
| Orologio e temperatura per campione | no (Windows, senza strumenti) | sostituiti dalla sonda FMA prima di ogni gruppo |

## 9. Revisore indipendente

Un sotto-agente fresco ha letto la tesi, il piano, la letteratura, il protocollo e il codice. Dieci obiezioni,
quattro bloccanti. Tutte e quattro potevano dare un falso ANGOLO CPU.

| # | Obiezione | Risposta |
|---|---|---|
| 1 (B) | Le celle del gate avevano 256-512 token: attivazioni e accumulatori stavano in cache, non nel regime da 16k | corretta: token da un passo di ≥ 16k (§1), costi fissi a parte e addebitati in proporzione |
| 2 (B) | AVX2 a mano contro PyTorch semplice, e nessun controllo che lo streaming stia al limite del trasferimento | GPU occupata registrata per passata; il gate legge lo contro lo streaming ideale |
| 3 (B) | Solo la CPU salta le unita spente; lo streaming muove ogni riga nel backward | lo streaming ideale muove solo le righe attive; il gate lo usa |
| 4 (B) | h1 letto a densita 0,5, che non vale per esperti PEER | h1 a densita 1 per tutte le coppie del gate; per i ternari via x₀ (verificato: 0,998) |
| 5 | Gli esiti cambiano poco: alla r del regime l'angolo vale meno dell'1% | accolta come lettura: il guadagno sul passo e calcolato e predetto (§3-4). GRIGIO porta alla stessa domanda su D-R. Spostare il gate dal r\* al guadagno cambierebbe il piano: proposta, non applicata |
| 6 | C6 a h8 e h1 non usa il kernel pshufb | dichiarata (§8): il ternario CPU e un limite inferiore |
| 7 | Il P-cpu non poteva fallire (stessi kernel del controllo RAM) | sostituito dal braccio C2 vero a densita zero |
| 8 | Il selftest scopriva r = 2 e ≥ 4, il secondo passo e il blocco parziale | aggiunti (§5) |
| 9 | Deriva termica: C e S di una coppia a minuti di distanza; FMA solo a inizio e fine; due runtime OpenMP | C e S alternati cella per cella; FMA prima di ogni gruppo; torch a 1 thread e attesa passiva, stampati nel CONFIG |
| 10 | Il pavimento non scatta mai nel run vero | dichiarato (§5) |

## 10. Esecuzione

- **Dove e come:** dalla radice del worktree, a macchina scarica:
  - `python benchmarks/sparse_capacity/sc1b.py --run`
  - si rifiuta se l'apparato ha modifiche non committate, se HEAD non e pushato, se la CPU e occupata oltre il 15%,
    se i thread non si fissano, se la GPU non e la 3060, se il selftest fallisce o se un piantato esce di banda.
- **Dopo un blocco:** `--run --resume <cartella>`.
- **Verdetto:** `python benchmarks/sparse_capacity/sc1b_verdict.py <cartella>`.
- **Durata:** stima a tavolino 1,9 h piu le sonde, quindi 2-2,5 h; tetto 3,5 h.
  - La cella piu pesante (h1, r = 128, CPU) fa circa 9 s a passo, in linea con il tavolino.
- **Artefatti:** `results/sparse_capacity/f1b_<data>/`, cioe `config.json`, `controls.jsonl` e `cells.jsonl`. Sono
  piccoli e si committano con il risultato.
