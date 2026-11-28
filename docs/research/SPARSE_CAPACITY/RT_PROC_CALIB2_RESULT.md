# RT: le sonde R allenate, seconda taratura su m1

2026-11-28. Regola registrata prima del run: [`RT_PROC_PROTOCOL.md`](RT_PROC_PROTOCOL.md) §5 (commit `0bc4eb1`).
- Run: `results/sparse_capacity/rt2_m1_s0/` e `rt2_m1_s1/`, sul commit `a58aa5b` (codice delle sonde uguale a
  `0bc4eb1`; in mezzo solo i tempi del kernel). CONFIG verificato: pushato, nessun file modificato, tappa 0.
- La prima taratura e in [`RT_PROC_CALIB_RESULT.md`](RT_PROC_CALIB_RESULT.md).

**Esito: la rubrica esplicita NON PASSA in m1. Per la regola registrata D-R si chiude come "non misurabile a nucleo
D256".** Niente m64, niente altri bracci. La raccomandazione per la fase 4 e al §4.

## 1. Numeri (m1, fine della ripresa di 15 M token; s0 / s1)

**Rubrica esplicita** (positivo piantato, 2.048 coppie mai allenate). La regola chiedeva, in tutti e due i seed, un
margine di almeno 0,10 e di almeno 4 errori standard sul controllo piu forte.

| | s0 | s1 |
|---|---|---|
| esatto | 0,063 | 0,038 |
| controllo piu forte: regola piu frequente del test | 0,043 | 0,043 |
| nome di un'altra regola nella chiamata | 0,022 | 0,020 |
| copia dell'input | 0,014 | 0,014 |
| margine (errori standard) | +0,020 (4,3) | −0,005 (−1,2) |
| token giusti | 0,754 | 0,746 |
| curva (ogni ~2 M token, dal passo 0) | 0 → 0,004 → 0,004 → 0,010 → 0,027 → 0,031 → 0,023 → 0,031 → 0,059 | 0 → 0 → 0,004 → 0,012 → 0,014 → 0,037 → 0,027 → 0,027 → 0,041 |

**Rubriche con nome** (lettura: in m1 non si chiede niente). Esatto, s0 / s1.

| N | esposizioni per funzione | esatto | regola piu frequente | funzioni mai viste |
|---|---|---|---|---|
| 64 | 323 | 0,109 / 0,055 | 0,090 | 0,029 / 0,032 |
| 512 | 40 | 0,039 / 0,031 | 0,047 | |
| 4.096 | 5 | 0,031 / 0,029 | 0,040 | |

- **Il resto regge:** K256 0,996 / 0,988; K2048 0,094 / 0,071. BPB del val 1,033 / 1,036, top-1 0,614 / 0,612: gli
  stessi della prima taratura.
- **T (lettura):** con il 4% del flusso invece del 10%, s0 impara ancora un salto (m2 d1 0,78); s1 meno (0,48). Da d = 2
  in su, tutti e due stanno al caso.

## 2. Cosa dice

- **Con le cifre separate il nucleo comincia a eseguire, ma troppo poco.**
  - In s0 l'esplicita sale fino a 0,063 e supera il controllo di 4,3 errori standard; legge un po' il nome (0,063 contro
    0,022 con il nome sbagliato).
  - In s1 resta sotto la regola piu frequente.
  - Il margine richiesto era 0,10: s0 ne fa un quinto. In 15 M token il nucleo D256 non esegue in modo affidabile
    regole di 1-2 operazioni scritte per esteso.
- **Senza esecuzione, P con nome non puo misurare D-R.** Un buon punteggio richiede sia la memoria (quale regola) sia
  l'esecuzione. Una banca piu grande potrebbe migliorare la sola memoria, ma un esito di P con nome non separerebbe le
  due.
- **Le curve salgono ancora alla fine,** sia l'esplicita sia P64 con nome (0 → 0,09 / 0,05). Il limite e di token piu che
  di forma. Con molti piu token la stessa sonda potrebbe passare; a questa ripresa no.
- **P64 con nome sopra le funzioni mai viste** (0,109 / 0,055 contro 0,03): il modello qualcosa lega al nome. Ma sta al
  livello del controllo cieco al nome (0,09), che nella rubrica da 64 e alto perche una regola copre il 7,8% delle
  funzioni.

## 3. Conduzione

- Due run, nessuna ripresa, nessun valore non finito; 2,49 h e 2,25 h.
- Quote misurate del flusso: 1.174 / 1.165 sequenze esplicite su 29.296 (4,0%), 3.478 / 3.430 con nome (11,8%).
- La regola e stata applicata cosi come era scritta nel §5 del protocollo, sul punteggio a fine ripresa.

## 4. Raccomandazione per la fase 4

**Si procede con la fase 4 per D-M, con la leva "la banca viaggia ternaria" e un bersaglio di 20 GB. Nel suo flusso
restano, come letture, K, la rubrica esplicita e P con nome da 64 e 512, con le cifre separate.** La fase 4 vede molti
piu token, e le curve stavano ancora salendo. La lettura costa solo una quota del flusso e qualche minuto di valutazione
per curva.

- Il kernel fuso AdamW + ternario costa 1,31 ns per parametro della banca (`a58aa5b`), contro 3,34 con torch.
- Il protocollo della fase 4 mette questa lettura in "Cosa toglie al design": D-R non e una domanda della fase 4, che
  non ha gate su D-R.
- **D-R si riapre solo se,** nel run della fase 4, la rubrica esplicita supera la regola registrata qui (margine di
  almeno 0,10 e almeno 4 errori standard). Solo allora P con nome e una misura e si confrontano banche diverse.

**Cosa la smentirebbe:**
- se le curve dell'esplicita si appiattiscono sotto il margine nel run della fase 4, l'esecuzione e fuori portata a
  nucleo D256 e la lettura P va tolta;
- se l'esplicita passa ma P con nome resta al controllo, la memoria delle procedure non c'e a questa scala.
