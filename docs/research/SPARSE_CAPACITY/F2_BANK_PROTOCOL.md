# F2: la banca su E, il protocollo

2026-11-18. Fase 2 del [piano](PLAN.md), domande **D-0** e **D-R**, solo GPU. Codice in `benchmarks/sparse_capacity/`
(`sc2_model.py`, `sc2_probes.py`, `sc2_train.py`). Il verdetto lo calcola `sc2_verdict.py` con le regole del §6,
scritte qui prima dei run.

**Le domande.**
- **D-0:** a nucleo fisso, calcolo degli esperti fisso e token fissi, una banca di esperti piu grande abbassa il BPB?
- **D-R:** la banca fa salire solo la conoscenza, o anche il ragionamento?

**Riga sulla tesi.** E una prova piccola, non alla scala della tesi:
- **la tesi di D-M e D-R:** banche oltre la VRAM, da miliardi di parametri;
- **qui:** nucleo da 4,6 M, banca fino a 403 M (64 volte la MLP densa), 60 M token per run.

Un PASS di D-0 dice che su questa architettura la banca lavora, e giustifica le fasi 3 e 4. Una D-R VIVA sarebbe il
primo segnale, non una prova: le fasi 3 e 4 la rimisurano a banche piu grandi. Una D-R PIATTA e la predizione (§7), e
a questa scala vale "non mostrato": la domanda resta aperta, e K e R restano come letture nelle fasi 3 e 4.

## 1. Cosa si allena

| Braccio | MLP di ogni strato | Totali | Attivi | Seed | Ruolo |
|---|---|---|---|---|---|
| m1 | banca di 128 esperti h8, tutti attivi (E = k) | 11,2 M | 11,2 M | 0, 1 | base del gate |
| m4 | 512 esperti h8, top-128 | 30,8 M | 12,0 M | 0, 1 | gate |
| m16 | 2.048 esperti h8, top-128 | 109,5 M | 15,1 M | 0, 1 | gate |
| m64 | 8.192 esperti h8, top-128 | 424,1 M | 27,8 M | 0, 1 | gate |
| dense | dReLU con gate, hidden 1.024 (la MLP del rung-1 prima dell'upcycle) | 10,9 M | 10,9 M | 0 | lettura |
| dense-fp | come dense, senza quantizzazione | 10,9 M | 10,9 M | 0 | lettura sulla precisione (§3) |

- **Nucleo:** la forma del rung-1, invariata: D256, N96, H8, L8, SWA allo strato 5, dt_rank 16, x_proj a rango 26. Il
  conto del braccio dense coincide con lo `stage_D` del rung-1 (10.916.576).
- **Base del gate, m1:** ha la quantizzazione, il gating pesato e il router della banca, ma nessuna capacita in piu.
  - Cosi la differenza m1 → m64 e solo la taglia della banca (con il suo router).
  - Il dense quantizza l'hidden su 1.024 valori invece che su 8: come base non sarebbe onesto (revisore, obiezione 2).
- **Calcolo:** l'hidden attivo per token e 1.024 in tutti i bracci. Il router cresce con E e vale 1/24 dei parametri
  della banca. Gli "attivi" della tabella lo contano: 2,1 M per strato a m64.
- **Router:** softmax su tutti gli E in fp32, top-k rinormalizzato, perdita di bilanciamento del rung-1 (peso 0,01).
- **Down della banca:** inizializzata alla scala del denso (±1/32). Il fattore k dei pesi rinormalizzati sta sui pesi
  del gating, non sulla down, e la scala A2 lo assorbe (revisore, obiezione 3).
- **Dati:** il pacchetto `m0` del rung-1:
  - codice Python da The Stack, licenze permissive, decontaminato contro P62;
  - 2,3 G token, BPE V = 2.048 addestrato qui;
  - file `ts_m0.u16`, `meta_m0.json`, `bpe2048_ts.bin`, `p62_m0.u16`, su E: o nel dataset Kaggle del rung-1.
  - Finestre da 512 token pescate a caso dalla parte di training. Il 10% delle sequenze sono le sonde del §4.
- **Ricetta, uguale per tutti:**
  - AdamW (0,9, 0,95), weight decay 0,1, lr 3e-3 con 200 passi di riscaldamento e coseno fino al 10%, clip 1,0;
  - 8.192 token per passo, 7.324 passi, 60 M token;
  - bf16 nel nucleo; micro-batch 16, tranne m64 a 4 con accumulo 4 (a 8 non entra in 12 GB).
- **Calcolo della banca:**
  - gate e up in un solo prodotto denso su tutte le unita, poi si tengono i k esperti scelti;
  - le operazioni elemento per elemento solo sulla selezione;
  - la down come prodotto denso della selezione riportata in (N, E·h).
  - La matematica e quella "solo attivi": un esperto non scelto ha gradiente esattamente zero (selftest). Il costo in
    FLOP e E/k volte quello attivo, ma solo sulla GPU (§8).

## 2. La granularita: h8, non h1

La scelta e presa con i dati della 1b ([`F1B_CROSSOVER_RESULT.md`](F1B_CROSSOVER_RESULT.md)), dell'audit e della
letteratura.
- **Costo nel metodo.** Alla r del regime, in streaming, h1 costa per parametro della banca:
  - da 1,9 a 2,7 volte h8 con Adagrad per riga (S2 h1 1,26 e 1,85 contro S2d h8 0,67 e 0,68 ns, a r = 8 e 16, stessa
    densita);
  - da 1,2 a 1,8 volte con Adam.
  - Sulla CPU il rapporto sale a 3-4 volte. A pari banca e pari token la r per esperto non dipende da h, quindi il
    rapporto vale per tutto il passo della banca.
  - La causa sono le attivazioni: ogni coppia (esperto, token) muove D valori per h parametri.
- **Router.** Con lo stesso router a scansione piena, a m64 h1 vorrebbe E = 65.536: 16,8 M moltiplicazioni per token e
  per strato, 21 volte gli esperti attivi (h8: 2,1 M, 2,7 volte).
  - Per h1 servono le chiavi prodotto di PEER (arXiv 2407.04153), un meccanismo nuovo che il piano rinvia (H5).
- **Motore.** Sotto h32 il motore non ha un percorso ([`F0_QUANT_AUDIT.md`](F0_QUANT_AUDIT.md) §3).
  - A h1 la down ha una scala per peso: circa 4,5 byte per peso invece di 0,5.
  - A h8 le scale della down costano 0,5 byte per peso.
- **Cosa si perde.** PEER (Tab. 1) batte denso e MoE a pari FLOP con esperti da un neurone. Krajewski (arXiv
  2402.07871) trova ritorni della grana fine che convergono a una costante, con il costo del routing che domina gia a
  G = 16.
  - Qui h8 e G = 128 rispetto alla MLP densa.
  - La qualita di h1 a pari parametri non si misura.
- **Cosa smentirebbe la scelta:** un D-0 FAIL a h8.
  - Le sole prove positive finora sono a h128 top-8 (Probe-4, rung-1). Prima di chiudere il programma, il FAIL porta un
    braccio h128 top-8 alla taglia di m64 (E = 512, un seed, circa 25 h).
  - Il rapporto dice che h1 con chiavi prodotto non e stato provato.

## 3. La precisione, dall'audit della fase 0

- **Banca e MLP:** pesi ternari con scala per riga (absmean) e attivazioni del motore (`act_quant=engine`, F0 §4):
  - interi in [-63, 63], una scala absmax per gruppo, arrotondamento al pari;
  - gruppi: x per token su D, uguale per tutti gli esperti del token (A1); l'hidden pesato per (token, esperto) su h
    (A2); per la MLP densa l'hidden per token su 1.024 (A3);
  - prodotto intero prima, scale dopo, in fp32 (W1). I prodotti girano in fp32 pieno con TF32 spento, quindi la somma
    di interi e esatta. Sulla 3060 costa come il TF32.
- **Router e nucleo:** il router legge x non quantizzato, come `engine.c:289`. Il nucleo resta in virgola mobile (bf16).
- **Fin dal primo passo:** tutti i bracci partono da zero gia ternari, come BitNet b1.58 (arXiv 2402.17764). Niente
  rampa alpha, quindi il controllo W2 dell'export non serve: alpha vale sempre 1.
- **Selftest:** lo strato denso e la banca coincidono con una replica numpy di `quant_i8` e del prodotto intero (per la
  banca, un ciclo per token sugli esperti scelti), entro 4e-7 relativo.
- **Letture sulla precisione:**
  - dense-fp contro dense: quanto costano pesi ternari piu attivazioni a 63 livelli. L'audit lo chiedeva prima di
    scegliere la precisione, e nessuna fonte letta lo misura;
  - dense contro m1: quanto pesa la grana della scala A2 rispetto ad A3.

## 4. Le sonde: conoscenza e ragionamento

Dai compiti di Mixture of Parrots (arXiv 2410.19034 v2, §4.1): la rubrica (memorizzazione) e il cammino minimo su grafi
orientati casuali (ragionamento). Sono scritti come righe Python e mescolati al codice: il 4% delle sequenze per K e il
6% per R.

- **K, la rubrica.**
  - Tre rubriche fisse da 256, 2.048 e 16.384 voci, con la stessa quota di token ciascuna. Le esposizioni per voce sono
    circa 222, 28 e 3,5, cosi almeno una rubrica cade nel tratto che discrimina.
  - Riga: `tel["kqzvw"] = 48213907`, con nomi di 5 lettere e numeri di 8 cifre, tutti unici.
  - Punteggio: le 8 cifre giuste con il prefisso dato, su voci di training (la memorizzazione e il compito). Le voci
    valutate sono 256, 2.048 e 4.096 (un campione fisso della terza rubrica).
  - **K primaria** = media delle tre accuratezze.
- **R, il cammino minimo.**
  - Grafi nuovi a ogni uso, con 6 o 9 vertici e probabilita di arco 2,8/n.
  - La domanda ha un unico cammino minimo, lungo L ∈ {1, 2, 3, 4}. Testo: `g = {0: [3, 5], ...}`, poi
    `assert path(g, 4, 7) == [4, 2, 7]`.
  - Test: 500 istanze per cella (n, L), escluse dal training per costruzione.
  - Punteggio: il cammino esatto, cioe ogni token come argmax dato il prefisso (equivale alla decodifica greedy).
  - **R primaria** = le celle n9 L3 e n9 L4 (1.000 item): le uniche che una regola locale non risolve quasi sempre
    (revisore, obiezione 6). Le altre celle si riportano.
  - **Cella mai allenata,** n12 L5 (500 item): una procedura appresa vi si estende, frammenti memorizzati no. E una
    lettura, non un gate.
- **Contesto in valutazione:** come nel training. La finestra si riempie di righe della stessa sonda (per K, di tutte e
  tre le rubriche), senza padding: lo scorer rifiuta un contesto piu corto della finestra (selftest).
- **Modi noti di fallire** (Parrots):
  - il legame fra memorizzazione e ragionamento e detto "suggestive";
  - e provato solo con routing top-2;
  - i compiti sintetici sono allenati da soli, non mescolati.
  - Qui la miscela e una scelta nostra (§8).
  - Il nucleo e piccolo, quindi R puo restare al livello delle regole locali: per questo c'e la condizione "R non
    misurabile" (§6).

## 5. Metriche e controlli banali

- **BPB:** sui primi 2 M token della parte tenuta da parte del corpus, ogni byte contato una volta; accanto si riporta
  sempre l'accuratezza top-1. P62 e una lettura, se il file c'e.
- **Controllo banale del BPB, la temperatura:** una temperatura scalare scelta sulla prima meta del val, BPB sulla
  seconda. Se il guadagno di D-0 sopravvive solo a T = 1, il BPB non decide da solo.
- **K:**
  - fa da **positivo piantato**: con la banca deve salire (Parrots, Fig. 4a). Se non sale, la banca non memorizza e la
    lettura di D-R non vale;
  - 2.048 nomi mai visti servono solo da verifica dello scorer (indovinare 8 cifre ha probabilita 1e-8), non da
    controllo.
- **Controlli di R,** sulle celle primarie (n9 L3 e L4, dai 500 item di test per cella):
  - le stesse domande con la lista di archi di un'altra istanza (grafo scambiato), punteggio sul cammino vero;
  - tre regole senza modello:
    - primo arco in lista: 0,14 e 0,08;
    - cammino casuale senza ritorni: 0,35 e 0,27;
    - la regola locale piu forte, che va a t se e adiacente, altrimenti a un vicino adiacente a t: 0,61 e 0,52.
- **Banca usata:** su 262k token di val, per strato, il numero effettivo di esperti (esponenziale dell'entropia del
  carico) diviso per E.

## 6. Esiti e cosa decidono

- **σ_BPB:** la deviazione fra i due seed, messa in comune sui quattro bracci del gate, mai sotto 0,005 (la
  calibrazione R1).
- **σ_K e σ_R:** la deviazione fra seed, mai sotto l'errore standard della differenza appaiata sugli stessi item, ne
  sotto quello binomiale di un braccio.

| Esito | Condizione | Decisione (PLAN §5) |
|---|---|---|
| VOID | un run mancante, da commit diversi, non pushato o con modifiche; valori non finiti; in uno strato di m4, m16 o m64 gli esperti effettivi sono meno del 25% di E; la verifica dello scorer K sopra l'1% | si corregge e si ripete |
| **D-0 PASS** | il BPB medio scende da m1 a m64 (nessun passo sale di oltre 1σ), il guadagno m1 → m64 e ≥ 4σ, e resta ≥ 4σ dopo la temperatura | si va avanti |
| D-0 FAIL | altrimenti | prima di chiudere, un braccio h128 top-8 a m64 (§2); poi il rapporto |
| **D-R VIVA** | K sale (m64 − m1 > 2σ_K), R e misurabile, e R(m64) − R(m1) ≥ 2σ_R | le fasi 3 e 4 la rimisurano a banche piu grandi |
| D-R PIATTA | K sale, R misurabile, la differenza resta sotto 2σ_R | "non mostrato" a questa scala (nucleo 4,6 M, 60 M token): la domanda sul ragionamento non si chiude. Il programma va avanti con il metodo (fase 3), e K e R restano letture nelle fasi 3 e 4 |
| R NON MISURABILE | ne m1 ne m64 superano di 0,05 il controllo piu forte di R, oppure m1 e ≥ 0,95 | la fase 3 puo partire per il metodo; prima della fase 4 serve una sonda R nuova |
| D-R NON LEGGIBILE | K non sale | la banca non memorizza: si guarda prima D-0 e il routing |

**Letture che non sono gate:**
- accanto al guadagno di D-0, i parametri attivi di ogni braccio, router compreso (da 11,2 M a m1 a 27,8 M a m64): il
  verdetto li stampa nella tabella e nella riga di D-0;
- top-1 e P62;
- la curva del BPB ogni 1.000 passi;
- K per rubrica, R per cella con i controlli, la cella n12 L5;
- le due letture sulla precisione (§3);
- gli esperti effettivi;
- le ore per braccio, che entrano nel gate di ritmo della fase 3.

## 7. Predizione (registrata)

- **D-0 PASS:** da m1 a m64 la banca abbassa il BPB del codice di 0,03-0,08, oltre 6σ. Indizi: Probe-4, dove E32
  batte il denso a 22,5 M, e la letteratura MoE a pari attivi.
- **K:** sale molto con la banca, soprattutto sulla rubrica da 2.048 voci.
- **D-R PIATTA:** R non si muove oltre 2σ, come in Parrots (§4.3: "increasing the number of experts does not
  consistently improve"). Il ragionamento segue i parametri attivi, qui fissi a parte il router.
- **Rischio dichiarato:** R puo restare al livello della regola locale (0,57 sulle celle primarie). Allora l'esito e
  NON MISURABILE.
- **Precisione:** dense-fp sotto dense di 0,02-0,05 BPB; dense e m1 entro 0,02.
- **La predizione e contro lo scopo di D-R (regola 5).** La decisione che cambia e scritta nel piano: con D-R PIATTA il
  programma va avanti con il metodo, e D-R resta aperta come "non mostrato a questa scala". K e R restano letture nelle
  fasi 3 e 4, dove la banca e piu grande.

## 8. Cosa toglie al design

| Elemento | Qui | Perche, e in che verso puo sbagliare |
|---|---|---|
| E2048 × h128 del piano (256 volte la MLP) | fino a m64 (64 volte), a h8 | con Adam la banca da 256 volte non entra nei 12 GB. Il guadagno di D-0 e un limite inferiore di quello a 256. h128 entra solo dopo un FAIL (§2) |
| Il denso come base, come nel piano | base m1; il denso e una lettura con un seed | il denso ha una quantizzazione piu grossa (§1): come base gonfierebbe il guadagno |
| Parametri attivi fissi | fisso il calcolo degli esperti; il router cresce con E (16,8 M a m64) | un PASS misura banca piu router, non la sola banca. Non c'e un braccio a router congelato: il router e un costo della banca, ma la sua parte del guadagno non si separa |
| h1 e chiavi prodotto (PEER) | non provati | §2. Puo far perdere qualita alla banca |
| Calcolo solo attivi, streaming | calcolo denso su GPU con selezione | stessa matematica e stessi gradienti (selftest), costo E/k volte in FLOP. Il ritmo vero e quello del metodo, nella fase 3 |
| Router a basso costo | scansione piena su E | a m64 costa 2,7 volte gli esperti attivi in FLOP; nella fase 4 va cambiato |
| Curriculum del rung-1 (C → D → E → F, upcycle, recall, KD) | tutto da zero, gia ternario | una sola variabile, la banca. Il rung-1 e un'altra ricetta, e i numeri non si confrontano |
| lr piatto del rung-1 | coseno fino al 10% | il punto finale ha meno rumore di seed. Uguale per tutti i bracci |
| "Monotono" del piano | nessun passo sale di oltre 1σ | con due seed un passo piatto puo salire per rumore. Il guadagno totale resta a 4σ |
| Stesso micro-batch | m64 a 4, gli altri a 16 | la perdita di bilanciamento di m64 vede 2.048 token invece di 8.192: un coefficiente piu rumoroso, non una distorsione, a peso 0,01. Uguagliarla costerebbe circa 47 h |
| Compiti di Parrots da soli | mescolati al codice al 10% | la conoscenza e il ragionamento devono crescere nello stesso addestramento della banca. Toglie token al codice in modo uguale in tutti i bracci |
| Grafi di Parrots (25-55 vertici, lunghezza media 3,5) | 6 e 9 vertici, L ≤ 4, primarie n9 L3-L4 | il nucleo e piccolo e la finestra di attenzione e di 128 token. La cella n12 L5 legge l'estensione |
| Adagrad per riga, bf16 (piu veloci nel metodo, 1b) | Adam fp32 | servono i gate di qualita (b) e (c); la fase 2 non li mescola con D-0 |
| Token "Chinchilla" per il nucleo (~220 M) | 60 M | budget (§10). Ogni esperto di m64 vede comunque 0,94 M token, sopra la soglia del piano (10⁵) |
| Formato del motore sotto h32 | non esiste | la fase 5 lo scrive (F0 §3); qui conta solo la matematica dei 63 livelli |

## 9. Revisore indipendente

Un revisore senza la conversazione ha letto il piano, la tesi, questo protocollo e il codice, prima dei run, e ha
sollevato 13 obiezioni.

| # | Obiezione | Risposta |
|---|---|---|
| 1 | Il verdetto scartava i run densi (glob `f2_[!d]*`): sempre VOID | Corretto: tutte le cartelle `f2_*` tranne i dry; ogni run deve essere RUN, pushato, pulito, a 7.324 passi |
| 2 | La banca quantizza l'hidden su 8 valori, il denso su 1.024: il guadagno mescola precisione e capacita | Corretto: base del gate m1 (stessa banca, nessuna capacita in piu). Il denso diventa una lettura |
| 3 | La down della banca, inizializzata ×128, impara 128 volte piu piano con AdamW | Corretto: down alla scala del denso, fattore k sui pesi del gating; forward identico all'inizio (selftest) |
| 4 | Il router (16,8 M a m64) e calcolo e parametri in piu | Dichiarato in §8 e nello scopo di D-0: banca piu router. m1 ha lo stesso gating pesato; manca un braccio a router congelato |
| 5 | Un FAIL a h8 top-128 chiuderebbe il programma su una forma mai provata | Corretto: il FAIL porta un braccio h128 top-8 prima del rapporto (§2, §6) |
| 6 | R e quasi tutto una consultazione a un passo; il controllo usato era debole | Corretto: primarie n9 L3-L4, controlli "regola locale" e cammino casuale, cella n12 L5 mai allenata |
| 7 | Lo scorer riempiva la finestra con il token 0, mai visto in training | Corretto: contesto di righe della sonda, e lo scorer rifiuta il padding |
| 8 | "Esperto usato" con un solo token non scatta mai | Corretto: esperti effettivi (entropia del carico), VOID sotto il 25% |
| 9 | σ_R binomiale di un braccio, non appaiato; σ_K senza pavimento | Corretto: errore appaiato sugli stessi item, pavimento binomiale anche per K |
| 10 | Il controllo sui nomi mai visti non puo fallire | Accettato: e una verifica dello scorer; il controllo vero e K come positivo piantato |
| 11 | Il contesto di K in valutazione differisce dal training | Corretto con la 7 |
| 12 | La perdita di bilanciamento dipende dal micro-batch | Dichiarato in §8 (rumore, non distorsione; uguagliarla costa circa 47 h) |
| 13 | Le ore si azzeravano alla ripresa; "monotono" allentato; durata mancante | Ore cumulative nel checkpoint; 1σ dichiarato in §8; durata in §10 |

## 10. Esecuzione

- **Dove:** sulla 3060, un run alla volta. I run non chiedono la macchina scarica: il ritmo e una lettura.
- **Prima:** la fase 1a (E: collegato) secondo il piano, perche il pacchetto `m0` sta li o su Kaggle.
- **Comando:** `python benchmarks/sparse_capacity/sc2_train.py --run --arm <braccio> --seed <s> --data-dir <pacchetto m0>`.
  - Il runner rifiuta se l'apparato ha modifiche non committate, se HEAD non e pushato, se la GPU non e la 3060 o se il
    selftest fallisce (15 controlli). Stampa e salva il CONFIG.
  - Si ferma su un valore non finito.
  - Salva ogni 30 minuti; dopo un blocco si riparte con `--resume`.
- **Ordine:** m1 s0, m64 s0, m4 s0, m16 s0, poi lo stesso con s1, e per ultimi dense e dense-fp. Nessuna lettura dei
  gate prima della fine: le curve si registrano ma non si interpretano.
- **Durata:** dai dry run (12 passi, forme vere, sulla 3060).

  | Braccio | dense | dense-fp | m1 | m4 | m16 | m64 |
  |---|---|---|---|---|---|---|
  | token/s | 2.555 | 2.593 | 2.340 | 1.741 | 1.323 | 660 |
  | ore per run (60 M token) | 6,5 | 6,4 | 7,1 | 9,6 | 12,6 | 25,3 |

  In tutto circa 122 h di 3060 piu le valutazioni: circa 5 giorni e mezzo.
- **Verdetto:** `python benchmarks/sparse_capacity/sc2_verdict.py`.
- **Artefatti:** `results/sparse_capacity/f2_<braccio>_s<seed>/` (`config.json`, `train.jsonl`, `result.json`) si
  committano. I pesi finali restano fuori da git, con percorso e SHA-256 nel log.
