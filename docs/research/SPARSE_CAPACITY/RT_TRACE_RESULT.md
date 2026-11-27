# RT: la sonda R nuova, tracciamento di variabili senza addestramento, la taratura su m1

2026-11-27. Prima della fase 4 del [piano](PLAN.md), domanda **D-R**. Codice `benchmarks/sparse_capacity/sc_trace.py`;
numeri in `results/sparse_capacity/rt_tune_20261127_101157.json` e `rt_tune_planted_m1s0.txt`.

**Regola di questa prova** (data dall'utente): la sonda si tara e si registra guardando **solo m1 s0 e s1**. Gli altri
otto checkpoint della fase 2 non sono stati caricati (il modo `--tune` non li apre). Se m1 non e misurabile, si propone
una sonda da allenare.

**Esito: m1 NON MISURABILE.** m1 non lega nemmeno una variabile: in tutte le 56 combinazioni di cella e formato sta al
caso. Segue la proposta di una sonda allenata (§3).

## 1. La sonda

- **Il programma:** m catene di assegnazioni, ognuna profonda d (`q = 7`, poi `k = q`, ...), mescolate a caso tenendo
  l'ordine dentro la catena. La domanda e sull'ultima variabile di una catena. Il formato viene da Wu, Geiger, Milliere
  (arXiv 2505.20896): `var = const | var = var` con catene di distrazione. La lettura di RULER (arXiv 2404.06654) e
  la stessa, a catena.
- **Contro le scorciatoie note.** Wu et al. trovano che durante l'addestramento il modello passa per un'euristica di
  posizione: prende il letterale della prima riga, oltre il 90% di quei casi. Qui tutte le catene hanno la stessa
  profondita, quindi per simmetria il primo, l'ultimo o qualsiasi letterale e la risposta con probabilita 1/m. Il
  selftest lo verifica su 6.000 programmi: scarto sotto 0,02.
- **Celle:** m ∈ {2, 3, 4} per d ∈ {1, 2, 3, 4}, piu due celle con trappola. Nella trappola la radice viene
  riassegnata dopo l'uso, e chi risolve i nomi col valore finale sbaglia sempre.
- **Formati:** `assert k == ` e doctest (`>>> k`).
- **Contesto:** 512 token prima della domanda, presi da codice vero del val oppure da programmi risolti della stessa
  cella (esempi nel contesto).
- **Risposta:** un letterale scelto fra i 29 che il BPE scrive con un solo token (2-20, 24, 25, 30, 32, 50, 60, 64, 99,
  100, 200).
  - **Metrica:** la top-1 greedy su tutto il vocabolario.
  - **Lettura:** la scelta fra i letterali del programma (scelta multipla), dove il caso vale 1/m.
- **Controlli senza modello:** primo letterale, ultimo letterale, letterale piu vicino prima della riga chiesta, un
  salto per nome, caso, risoluzione col valore finale (solo nella trappola).
- **Quantita:** 400 programmi per cella, 14 celle, 2 formati, 2 contesti, 2 seed.

## 2. Cosa dice m1

| Formato, contesto | top-1, massimo su 14 celle (s0 / s1) | scelta multipla meno 1/m, media (s0 / s1) | massimo |
|---|---|---|---|
| assert, codice | 0,020 / 0,107 | +0,024 / +0,011 | +0,093 |
| assert, esempi | 0,035 / 0,045 | +0,006 / +0,005 | +0,047 |
| doctest, codice | 0,000 / 0,000 | +0,002 / 0,000 | +0,047 |
| doctest, esempi | 0,007 / 0,010 | +0,016 / −0,009 | +0,069 |

- **Anche la cella piu facile e al caso:** m2 d1, cioe una sola riga `x = 17` prima della domanda. La scelta
  multipla fa 0,59 / 0,56 contro il caso 0,55. Il massimo +0,093 viene da 56 confronti con 400 programmi l'uno
  (errore 0,025): e selezione, non segnale.
- **Lo strumento funziona** (`rt_tune_planted_m1s0.txt`):
  - con la riga ripetuta tre volte (`assert q == 17`), m1 risponde 17 con probabilita 0,63;
  - sul formato allenato della fase 2 legge il grafo (`[`, poi `1,` a 0,98);
  - il token della risposta nel contesto e lo stesso di quello isolato (controllato da 2 a 39).
- **Quindi a m1 manca la capacita, non e un problema di formato.** Il modello copia una riga ripetuta, ma non lega un
  nome a un valore. E coerente con la scala: 11 M parametri attivi e 60 M token. Wu et al. arrivano al 99% allenando
  solo su questi programmi.

## 3. Proposta: la stessa sonda, allenata

**Una raccomandazione:** riprendere l'addestramento dei checkpoint finali della fase 2 con i programmi di questa sonda
nel flusso, e misurare D-R su quello. Si comincia tarando su m1 s0 e s1 soltanto.

- **Flusso:**
  - 76% codice, preso in posizioni nuove dello stesso pacchetto;
  - 4% rubrica K, per leggere la conoscenza separata;
  - 20% programmi;
  - il grafo R della fase 2 esce.

  I programmi d'allenamento sono sempre nuovi e non toccano quelli di test (stesse chiavi escluse, come in R). Si
  allena e si valuta con il formato `assert`.
- **Ricetta:** 15 M token (un quarto della fase 2), lr costante al pavimento della fase 2 (3e-4) dopo 100 passi di
  warmup. Si riparte dallo stato AdamW salvato in `last.pt`. Stessi token e stessi batch per ogni braccio.
- **Taratura (prima, solo m1 s0 e s1, ~3,5 h di 3060):**
  - si fissano le celle primarie fra m ∈ {2..6} e d ∈ {1..8}: le piu difficili che passano la regola qui sotto;
  - **misurabile** = in tutti e due i seed, top-1 sopra il controllo banale piu forte di almeno 0,10 e di almeno 4
    errori standard, top-1 al massimo 0,85 (c'e spazio per salire), e scarto fra i seed al massimo la meta di quel
    margine;
  - la varianza fra seed misurata li fissa la soglia per gli altri bracci. Si registra e si pusha, poi gli altri otto
    (~26 h);
  - se m1 non passa nemmeno allenato, ci si ferma e si scrive.
- **Esiti per D-R** (da registrare con la taratura): SALE se m64 − m1 ≥ 2 σ della differenza (media dei seed, stessi
  programmi); PIATTA se sotto 1 σ; letture: K accanto, i due densi, le celle fuori distribuzione (d e m oltre il
  training: procedura contro frammenti).
- **Predizione:** PIATTA. La letteratura della fase 0 dice che il ragionamento segue i parametri attivi.
  - **La decisione che cambia:** con PIATTA, D-R a questa scala dice "solo conoscenza". La fase 4 non porta bracci per
    D-R e si occupa solo del metodo (D-M). Con SALE, la fase 4 li porta.
- **Costo totale:** circa 30 h di GPU. Si spende piu di 10 h e c'e una metrica primaria nuova, quindi prima del push
  serve il revisore indipendente.
- **Riga sulla tesi.** E una prova su un compito d'algoritmo alla scala della cella (11-28 M attivi). Un SALE sarebbe
  il primo segno che i micro-esperti aiutano a imparare una procedura, non solo a ricordare. Non dice nulla sulla scala
  dei 100B.

**Cosa la smentirebbe:** se m1 allenato resta al caso anche nelle celle con d = 1. In quel caso il compito e fuori
portata a questa scala, e serve un nucleo piu grande prima di qualsiasi sonda R.
