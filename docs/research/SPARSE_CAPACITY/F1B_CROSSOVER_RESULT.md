# F1b: magazzino contro officina, il risultato

2026-11-18. Fase 1b del [piano](PLAN.md), domanda **D-M**. Regole nel [protocollo](F1B_CROSSOVER_PROTOCOL.md), registrato
e pushato prima del run (`256a52d`). Dati in `results/sparse_capacity/f1b_20261118_091040/`, verdetto di
`sc1b_verdict.py` in `verdict.txt` nella stessa cartella.

**Verdetto: GRIGIO.** In nessuna cella del gate la CPU batte lo streaming ideale. Pero contro lo streaming misurato
la CPU vince ancora a r piccolo, quindi non e STREAMING pieno.

**Riga sulla tesi (D-M).** Per tenere la banca nella RAM su questo PC il metodo e lo streaming (OOM), senza un posto
per esperto sulla CPU. D-M diventa ingegneria, e la domanda forte resta D-R.

## 1. Validita

- **Apparato:**
  - selftest: 710 controlli passati;
  - CONFIG: HEAD `256a52d` pushato, nessun file modificato, CPU occupata all'1%;
  - celle cronometrate dalle 09:10 alle 10:03, piu il selftest. La stima era 2-2,5 h.
- **Controlli all'inizio:**
  - RAM: lettura 40,9 GB/s, lettura-modifica-scrittura 36,5 (34,3 alla fine);
  - PCIe gen 4 x16: 26,2 GB/s in andata, 26,3 in ritorno, 13,1 per verso con i due versi insieme.
- **Piantati:** P-cpu 1,23 (banda 0,8-1,5), P-str 1,17 (banda 0,8-1,3).
- **FMA:** 4,11 GHz all'inizio, 4,06-4,11 nelle 9 sonde di meta run, 4,08 alla fine. La deriva e al massimo dell'1,2%,
  sotto la soglia del 10%.
- **Celle:** 12 campioni per cella, nessun peso non finito, nessuna cella sotto il pavimento.
- **Densita misurata:** 0,46-0,47 a h128 e h8, 0,996-0,998 a h1, 0,055 per i bracci a densita bassa.

## 2. Il gate

r\* in token per esperto per passo: mediana, intervallo [lo, hi], e lo calcolato con il solo streaming misurato.

| Coppia | h8: med [lo, hi] · lo misurato | h1: med [lo, hi] · lo misurato |
|---|---|---|
| fp32 + Adam (C4/S1) | 1,6 [< 1, 4,2] · < 1 | < 1 [< 1, < 1] · < 1 |
| fp32 + Adagrad per riga (C3/S2) | 5,0 [< 1, 7,4] · **4,3** | < 1 [< 1, 1,4] · < 1 |
| bf16 + Adagrad per riga (C5/S3) | 3,0 [< 1, 5,8] · **2,4** | 2,4 [< 1, 2,6] · < 1 |
| ternario (C6/S4) | 1,0 [< 1, 2,1] · < 1 | < 1 [< 1, < 1] · < 1 |

- **ANGOLO CPU** chiede lo ≥ 2: nessuna cella lo raggiunge.
- **STREAMING** chiede hi < 1 ovunque: non vale (fp32 + Adagrad per riga a h8 ha hi 7,4).
- **L'esito e quindi GRIGIO.**
- **Perche lo esce < 1.** A r = 1 lo streaming ideale sta sotto la CPU al suo peggio in ogni cella.
  - A h8, in ns per parametro, CPU al peggio · streaming misurato al meglio · ideale:
    - Adam 2,01 · 2,01 · 0,86;
    - Adagrad per riga 0,35 · 0,47 · 0,26;
    - bf16 0,39 · 0,55 · 0,15;
    - ternario 0,52 · 0,46 · 0,17.
- **L'angolo contro lo streaming scritto qui.** Contro lo streaming in PyTorch semplice, a h8 la CPU vince sotto
  r ≈ 4 (Adagrad per riga) e r ≈ 2,4 (bf16). E l'angolo che il revisore aveva chiesto di non contare (obiezioni 2-3).
  - S3 e S4 a h8 hanno la GPU occupata per il 50-74% del passo.
  - Sono quindi limitati dai loro kernel sulla GPU, non dal PCIe.

## 3. Predizione contro misura

- **Il verdetto previsto era ANGOLO CPU a h8, ed e sbagliato.**
  - Desk 2, con le bande misurate, dava contro lo streaming ideale a h8: Adam 3,7, Adagrad per riga 3,3, bf16 1,0,
    ternario 1,9.
  - Misurato: lo < 1 in tutte e quattro le coppie.
- **L'errore sta nel modello della CPU, non nelle bande.** A h8 e r = 1 i bracci CPU sono piu lenti di byte piu FLOP:
  - C4 (Adam) 1,99 contro 0,80, cioe 2,5 volte;
  - C3 (Adagrad per riga) 0,31 contro 0,25, cioe 1,2 volte;
  - C5 (bf16) 0,32 contro 0,15, cioe 2,1 volte;
  - C6 (ternario) 0,47 contro 0,18, cioe 2,6 volte.
  - Lo streaming ideale invece usa le bande di picco.
- **Asimmetria del riferimento, per Adam.**
  - Lo streaming ideale addebita l'aggiornamento Adam sulla CPU alla banda di picco: 0,86 ns per parametro.
  - I due bracci misurati spendono circa 2 ns in quello stesso aggiornamento. S1 ha la GPU occupata per il 6-15% del
    passo e vale quanto C4.
  - Per la coppia Adam, lo < 1 viene da qui, non dal PCIe. Il verdetto non cambia: le altre tre coppie danno lo < 1
    lo stesso.
- **Guadagno sul passo.** E calcolato contro lo streaming misurato, con instradamento uniforme e T = 16k.
  - Previsto: ≤ 1,8% a h8.
  - Misurato a h8: Adagrad per riga 0,3%, **bf16 7,9%**, ternario 0, Adam 0.
  - Misurato a h1: bf16 1,9%, le altre coppie 0.
  - A T = 128k: 0 ovunque.
  - A h128, che non e del gate: Adagrad per riga 24,8%, bf16 12,2%.
  - Il 7,9% del bf16 e contro S3 cosi com'e scritto, limitato dalla GPU. Contro lo streaming ideale e 0.

## 4. Letture che non sono gate

- **Con Adam il passo e l'aggiornamento sulla CPU.**
  - C4 e S1 valgono circa 2,0 ns per parametro a h128 e h8 per ogni r ≤ 16, cioe 0,4 s a passo per la fetta da 201 M.
  - S1 a h128 ha la GPU occupata per il 7% del passo.
  - In pratica 28 byte per parametro passano a circa 14 GB/s, contro i 36,5 misurati per la lettura-modifica-scrittura.
  - Adagrad per riga (S2) costa 0,48-0,67 ns, da 3 a 4 volte meno.
  - Nel metodo di OOM con Adam, su questo PC il collo di bottiglia e il passaggio dell'ottimizzatore sulla CPU, non il
    PCIe.
- **Formato.** In streaming, a h8 e alla r del regime (da 3 a 10):
  - S2 vale 0,65-0,67, S3 0,58-0,60 e S4 0,64-0,67: per parametro sono simili.
  - Cosi come sono scritti, bf16 e ternario non accelerano il passo, ma triplicano la banca che sta nella RAM: 2 byte
    per parametro con lo stato contro 6.
  - Il gate di qualita (b), Adagrad per riga contro Adam, viene prima: vale 3 volte in velocita.
  - Il (c), il bf16, conta per la capacita.
- **h8 contro h1.** A r = 8-16, h1 costa per parametro 1,9-2,7 volte h8 in streaming (S2: 1,26 e 1,85 contro 0,67 e
  0,69) e 3-4 volte sulla CPU. Nei micro-esperti dominano le attivazioni. Il dato serve alla scelta di granularita
  rinviata dal piano.
- **Densita.**
  - A 0,065 la CPU vince molto piu in la contro lo streaming misurato: r\* 9,2 a h8 e 34 a h128.
  - A densita 1 il margine si stringe: 4,6 a h8.
  - Lo contro lo streaming ideale resta < 1 anche qui.
- **Streaming e r.** A h128 lo streaming e piatto: S2 sta fra 0,64 e 0,67 da r = 4 a r = 128. A h8 sale da r = 32 in su.

## 5. Cosa decide, e la raccomandazione

**Regola registrata (§6).** GRIGIO vale come STREAMING:
- il metodo e lo streaming (OOM), e D-M diventa ingegneria;
- si decide se il programma continua per D-R;
- le curve restano per la fase 3.

**Raccomandazione.** Il programma continua per D-R, con la banca nella RAM in streaming e senza un percorso CPU per
esperto. L'ingegneria di D-M sta in due punti, da misurare nella fase 3:
- **il passaggio dell'ottimizzatore sulla CPU:** Adam fa 2 ns per parametro. Adagrad per riga costa da 3 a 4 volte
  meno, ma solo se passa il suo gate di qualita;
- **i kernel dello streaming a h8:** S3 e S4 sono limitati dalla GPU.

**Cosa la smentirebbe.** Due cose insieme farebbero tornare il posto sulla CPU:
- uno streaming scritto bene che resta sopra ~0,35 ns per parametro a h8 e r = 1 (la C3 misurata al suo peggio);
- un router vero che lascia molti esperti a r ≤ 2.

In quel caso la fase 3 ha gia le curve per fare il conto.
