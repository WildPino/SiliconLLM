# Direttiva del 27 ottobre 2026: correggere il modo di lavorare su CHATBOT50

**Da me, a chi lavora sulla repo.** L'ha preparata, su mia richiesta, una sessione di revisione che ha letto i documenti
del branch e i record di valutazione. La giro io e vale come mia.
- **Precedenza:** e una direttiva nel senso del goal ("segui le fasi e le direttive"). Ha la precedenza su ROADMAP §11
  e sui prossimi passi scritti in §12.
- **COMMUNICATION:** chiude la voce 1.
- **Regole:** stanno anche in `CLAUDE.md`, sezione "Regole di ricerca". Qui c'e il perche.

## 0. In una riga

Il rigore sull'apparato e ottimo e resta: hash, CONFIG, VOID, controlli piantati, push prima del run. Manca il rigore
sulle domande: **una cella che non puo rispondere alla tesi, o una metrica che misura altro, passano tutti i
controlli.** E successo piu volte, e me ne sono accorto io.

## 1. Cosa e successo

### 1.1 Il costo di attivazione e sparito fra design e protocollo
- **Design T2 (25-10, sezione "Training"):** "the same 33-bucket KL to F0 as S4, plus a cost on how often the gate
  fires". La scelta aperta 2 chiedeva solo se aggiungere *anche* il bersaglio "T0 sicuro".
- **Protocollo N (26-10):** "No fire cost: the fire rate is a measured output".
  - Il motivo dato riguardava solo i byte: +4,2% anche a gate sempre acceso.
  - La predizione registrata era fire rate >= 0,5: sapevi che il ramo sarebbe venuto denso, e l'hai lanciato.
  - Il segnale "T0 sicuro", che avevo scelto per il gate ("X e si"), e finito come bersaglio dell'uscita, non del
    gate.
- **Esito:** gate acceso al 98%. I 239 M parametri sono un adattatore denso, non esperti.
- **Perche conta:** il fire rate non e un dettaglio di costo, e l'invariante della tesi. La tesi chiede byte per
  token fissi mentre la capacita cresce. A +1,56% di capacita un ramo denso costa poco; a 2x la rompe. Il protocollo
  ha ottimizzato la cella, non la tesi.
- **La letteratura e arrivata dopo:** che un gate ReLU senza penalita L1 diventi denso (ReMoE) e stato letto solo
  dopo una mia domanda.

### 1.2 R misura soprattutto una distribuzione piu piatta, non conoscenza
Numeri riproducibili con `python -I benchmarks/chatbot50/c50_audit_r_hedging.py`. Usa solo record esistenti, e prima
di tutto ricalcola R con le funzioni dei verdetti e si ferma se non coincide con quelli registrati.

| Braccio | R (BPB, quella usata) | R su accuratezza top-1 | Fiducia media nel proprio top-1 |
| --- | --- | --- | --- |
| T0 | 0 | 0 | 0,814 |
| c48 (LoRA S4) | 0,699 | 0,231 | 0,732 |
| a (S4) | 0,505 | **-0,081** | 0,723 |
| v0 = T0 + A1 | 0,673 | 0,231 | 0,750 |
| E2 | 0,685 | 0,247 | 0,746 |
| N | 0,683 | 0,233 | 0,746 |
| N-shuffle | 0,679 | 0,238 | 0,748 |
| F0 | 1 | 1 | 0,8165 |

- **Il guadagno di v0 sta dove il modello sbaglia comunque.** Il 103,5% dei bit risparmiati cade dove T0 e v0
  sbagliano entrambi il top-1. Solo il 7,7% viene da posizioni che v0 corregge, e dove T0 aveva ragione v0 perde bit.
- **Lo studente diventa piu piatto sia di T0 sia di F0.** Distribuire la probabilita abbassa il BPB su testo dove si
  sbaglia il 30% dei token, ma non aggiunge sapere.
- **Il braccio (a) peggiora l'accuratezza** e "recupera" lo stesso il 50% del divario.
- **Sull'accuratezza, N-shuffle vale quanto N:** il contenuto dei singoli esperti non porta niente.
- **Il guadagno non dipende dal dominio.** R e circa 0,6 anche sui 12 documenti di codice, ma il corpus del maestro
  non ha codice (wiki, edu, math).
- **Spiega tutto quello che e successo:** r48 = r496, la LoRA di controllo "troppo forte", N = E2, lo shuffle al 56%,
  la seconda passata a +0,01 e i pannelli dei fatti fermi.
- **Le letture di S4, T1 parte 2b e N poggiano tutte su questa metrica.**
- **I 36 documenti sono i FIT** che A4 stesso aveva giudicato probabilmente memorizzati (per questo esiste A4-bis).
  Nessun documento successivo lo nota.
- **Cautela:** una parte del guadagno puo essere conoscenza parziale (il token giusto sale senza diventare primo).
  L'aumento dell'accuratezza, circa +1,4 punti, e reale. La cella M qui sotto lo separa.

### 1.3 Celle che non potevano rispondere alla tesi
- **N copriva +1,56% di capacita.** Il protocollo stesso scrive "It does not test a 2x capacity".
- **Il blocco 1 contiene circa 3.000 voci di Wikipedia su quasi 7 milioni.** nq e tqa chiedono fatti a caso: nessun
  organo li puo migliorare se il corpus non li contiene. I pannelli dei fatti erano ciechi per costruzione.
- **La capacita della tesi non si riempie con questo maestro.** 2x capacita vuol dire circa 15 B parametri nuovi. A
  86 tok/s, F0 impiega circa 5,5 anni anche solo per dare un token per parametro (ordine di grandezza). Nessuna cella
  ha fatto questo conto.
- **Tre celle erano registrate con predizione FAIL:** S4, lo stitch addestrato e N. Nessuna diceva quale decisione il
  FAIL avrebbe cambiato, e ogni FAIL e finito con "the choice returns".

### 1.4 Il ritmo
- **Dal 24 al 27 ottobre: 8 celle sulla meta qualita e 7 menu di scelta (A/B/C, 1/2/3) girati a me.** Due correzioni
  di direzione sono arrivate da me: il 25-10 ("il punto non era fare esperti ausiliari dal 100B?") e oggi.
- **Volume dal 16 ottobre:** 107.000 parole nella cartella della campagna, 161 commit, righe di §12 fino a 1.800
  caratteri. Non riesco a seguirli, quindi non fanno da controllo: servono alla tua continuita, non alla mia
  supervisione.

## 2. Cosa resta com'e

- **La meta velocita** (A3, A8, S0, S1) e solida.
- **L'apparato:** push prima del run, CONFIG stampato e verificato, VOID tenuti, controlli piantati, nessuna firma,
  COMMUNICATION corto.
- **I verdetti registrati non si rietichettano** (regola 5): si correggono le letture, con righe CORRECTION.

## 3. Cosa fai adesso, in ordine

1. **Commit e push** di questa direttiva e di `benchmarks/chatbot50/c50_audit_r_hedging.py`, in un commit separato
   e senza firma. `CLAUDE.md` resta locale: e gitignorato e non si committa.
   - Aggiungi una riga DECISION in §12, di 300 caratteri al massimo, che linka questo file.
   - Sposta la voce 1 di COMMUNICATION nello storico.
2. **Riproduci i numeri:** `python -I benchmarks/chatbot50/c50_audit_r_hedging.py --json
   benchmarks/chatbot50/results/c50_audit_r_hedging.json`. Se non tornano, fermati e scrivimelo.
3. **Ferme due celle:** NF (il codice resta com'e, non registrato) e la cella N contro E2 sui blocchi 1+2. Il blocco 2
   di S3 finisce, perche i dati servono comunque, ma non entra in nessuna cella prima del punto 5.
4. **Cella M, la metrica, senza training.** Protocollo di circa 2 pagine, pushato prima del run.
   - **Il controllo banale:** T0 con una sola temperatura, scelta fuori dai 36 documenti. Per esempio quella che
     minimizza la KL verso F0 sugli shard di validazione 470-471, cioe una distillazione con un solo parametro. Si
     misura il suo R. Se il banale prende una parte grossa di R, R non decide piu niente da solo.
   - **Le metriche di lettura:** R su accuratezza top-1 e R su BPB calibrato, ogni braccio con la propria
     temperatura fissata dalla stessa regola. Si rileggono S4, T1 parte 2b e N: dove serve un passaggio con piu
     temperature, l'evaluator lo fa in una passata.
   - **Documenti nuovi:** R sui 72 documenti post-release di A4-bis (serve F0, dopo il blocco 2).
   - **Uscita:** un risultato breve e una riga CORRECTION in §12. Le letture "il limite sono i dati del maestro" e
     "X e N chiusi" si marcano da rileggere se la metrica corretta le cambia.
5. **Revisione del design della meta qualita contro la tesi,** prima di qualsiasi training nuovo e con il revisore
   indipendente (regola 6). La prima candidata da valutare e un'**iniezione mirata**:
   - **Fatti:** quelli che F0 sa e T0 no (gli item donor-gap ci sono gia), piu un insieme gemello non addestrato come
     controllo.
   - **Testo:** costruito su quei fatti con F0 come maestro; in valutazione, domande riformulate.
   - **Bracci:** esperti con gate sparso e costo di attivazione dal primo passo, LoRA, nessun intervento, piu lo
     shuffle.
   - **Metrica:** accuratezza sui fatti iniettati meno quella sui fatti gemelli, non BPB.
   - **Scala:** quanti fatti per GB di esperti, e quanto testo del maestro serve per arrivare alla scala della tesi.

   Se la revisione sceglie altro, lo motiva con le stesse regole.

## 4. Le regole nuove e da dove vengono

Il testo vincolante sta in `CLAUDE.md`. Qui c'e l'episodio che ha prodotto ciascuna regola.

| Regola | Nata da |
| --- | --- |
| 1. Il protocollo elenca cosa toglie al design | il costo di attivazione sparito (1.1) |
| 2. Letteratura prima del protocollo | ReMoE letto dopo una mia domanda (1.1) |
| 3. Ogni metrica primaria passa un controllo banale | R premiava la piattezza (1.2) |
| 4. Una riga sulla tesi, alla scala della tesi | N a +1,56%, pannelli dei fatti ciechi (1.3) |
| 5. Una predizione contro lo scopo e un difetto | gate predetto >= 0,5 e lanciato; tre FAIL predetti (1.1, 1.3) |
| 6. Revisore indipendente prima di spendere | due correzioni di direzione arrivate da me (1.4) |
| 7. Dopo un FAIL una raccomandazione, non un menu | sette menu in quattro giorni (1.4) |
| 8. Documenti corti | 107.000 parole dal 16 ottobre (1.4) |

## 5. Cosa non fare

- **Non rispondere con piu cerimonia.** Niente gate piu severi sulla stessa metrica: il problema e cosa misura, non
  quanto e severa.
- **Non rilanciare celle chiuse** per "ripararle" prima della cella M.
- **Non riscrivere testo gia pushato:** solo righe CORRECTION.
- **Non trasformare questa direttiva in un protocollo da 10 pagine:** la cella M sta in circa 2 pagine, la revisione
  del punto 5 in 3.
