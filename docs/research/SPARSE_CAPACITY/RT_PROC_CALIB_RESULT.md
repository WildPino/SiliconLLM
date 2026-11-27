# RT: le sonde R allenate, la taratura su m1

2026-11-27. Protocollo in bozza: [`RT_PROC_PROTOCOL.md`](RT_PROC_PROTOCOL.md), con la regola di misurabilita
registrata prima della taratura (commit `c17909e`).
- Run: `results/sparse_capacity/rt_m1_s0/` (commit `c17909e`) e `rt_m1_s1/` (`2a456d0`). Fra i due sono cambiati
  solo il kernel della fase 4 e la letteratura: il trainer e le sonde hanno lo stesso SHA-256 nei due CONFIG.
- Valutazione delle sonde rifatta sui pesi finali con `--rescore`, al commit `3eede7a` (§3); i numeri sono in
  `result_rescore.json`.
- Gli altri otto modelli non sono stati toccati.

**Esito: P NON MISURABILE su m1.** In tutte e tre le rubriche P sta al livello delle funzioni mai viste. Il modello
non ha legato nessuna regola al suo nome. Per la regola registrata ci si ferma: niente altri bracci. Una
raccomandazione e al §4.

## 1. Numeri (m1, fine della ripresa di 15 M token; s0 / s1)

| Rubrica P | esposizioni per funzione | esatto | controllo piu forte | token giusti | token giusti, funzioni mai viste |
|---|---|---|---|---|---|
| 256 | 91 | 0,010 / 0,006 | 0,055 (regola piu frequente) | 0,599 / 0,581 | 0,598 / 0,580 |
| 2.048 | 11 | 0,006 / 0,004 | 0,044 | 0,588 / 0,576 | |
| 16.384 | 1,4 | 0,006 / 0,003 | 0,038 | 0,596 / 0,584 | |

- **Esatto:** sotto il controllo piu forte in tutte le rubriche, e uguale alle funzioni mai viste (0,006 / 0,002).
- **Token giusti** (lettura aggiunta al §3): sulle funzioni allenate quanto sulle mai viste, con uno scarto di 0,001.
  Lo 0,6 sono il formato (virgolette, a capo) e il tiro a indovinare. Della regola, il nome non porta niente.
- **Curva di P256** (s0, ogni 2 M token): da 0 a 0,006, piatta. Non sta salendo lentamente: non sale.
- **Il resto regge:** K2048 sale da 0,029 a 0,086 / 0,071. Il BPB del val scende da 1,039 a 1,033 / 1,036 (top-1
  0,614 / 0,612). La ripresa allena e la rubrica K continua a imparare.
- **T (lettura),** il legame allenato:
  - impara la lettura a un salto: d = 1 vale 0,91 / 0,78 con 2 catene e 0,76 / 0,63 con 3, contro il caso 0,50 e
    0,33;
  - da d = 2 in su sta al caso: m3 d3 0,33 / 0,32 contro 1/3, m4 d4 0,23 / 0,22 contro 1/4;
  - le celle fuori distribuzione sono al caso.

## 2. Cosa dice

- **A m1, in 15 M token, P non si impara affatto.** 91 esposizioni per funzione non bastano a legare una regola a un
  nome, mentre bastano a K per ricordare un numero (K256 resta a 0,98).
- **La causa piu probabile e l'esecuzione, non la memoria.** Un fatto di K e una stringa da ricordare. Una regola di
  P va anche eseguita su un input nuovo: sommare mod 10 per cifra, permutare le posizioni. Se il nucleo non sa
  eseguire le operazioni, P non puo misurare la memoria delle procedure.
  - Questa prova non separa le due cose: manca un controllo in cui la regola sia scritta nella chiamata.
  - Lo stesso vale per T: il nucleo impara un salto, non la catena.
- **D-R resta aperta.** P cosi com'e non la misura a questa scala.

## 3. Conduzione

- **Errore nello scorer di T, trovato e corretto dopo il run.** Nel flusso il letterale e seguito da una riga vuota,
  e il BPE la fonde in un token solo (`\n\n`). Lo scorer si aspettava `\n`, quindi T valeva 0 per costruzione.
  - Corretto (`bc8faac`, poi `3eede7a` per una riga di commento rotta): si giudica solo il token del letterale.
  - Le sonde sono state rivalutate sui pesi finali, controllati con lo SHA-256 del log. L'addestramento non ne e
    toccato.
  - P era tokenizzato in modo coerente fra flusso e scorer (controllato): i suoi numeri di esatto sono quelli del run.
- **Lettura aggiunta dopo il run:** l'accuratezza per token di P. Non decide niente: la regola registrata guarda
  l'esatto.
- Due run, nessuna ripresa, nessun valore non finito.

## 4. Raccomandazione

**Una seconda taratura su m1 con P separata in esecuzione e memoria, prima di qualsiasi altro braccio.**
- **Una rubrica "esplicita"** come positivo piantato. La regola sta scritta nella chiamata
  (`assert rot1_add3("3812") == ...`), con le stesse 36 operazioni. Non c'e niente da ricordare, solo da eseguire.
  - Deve stare sopra i controlli in m1.
  - Se non ci sta, il nucleo non sa eseguire le regole e P non puo misurare D-R a questo nucleo.
- **Rubriche con nome piu piccole e piu esposte:** N = 64, 512 e 4.096, circa 440, 55 e 7 esposizioni.
- **Il flusso:** 76% codice, 4% K, 12% P con nome, 4% P esplicita, 4% T (resta una lettura).
- **Taratura:** di nuovo dallo stato della fase 2, sempre solo m1 s0 e s1 (circa 4 h). Regola di misurabilita come
  quella registrata, sulla rubrica con nome; in piu, la rubrica esplicita sopra i controlli in tutti e due i seed.

**Cosa la smentirebbe:** se anche la rubrica esplicita resta al caso in m1, l'esecuzione e fuori portata per questo
nucleo con questi token. In quel caso nessuna variante di P misura D-R a questa scala: serve un nucleo piu grande, o
D-R si chiude come "non misurabile a questa scala".
