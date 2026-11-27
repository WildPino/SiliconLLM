# RT: le sonde R allenate (P "rubrica di procedure", T legame), il protocollo

2026-11-27. Prima della fase 4 del [piano](PLAN.md), domanda **D-R**. Segue
[`RT_TRACE_RESULT.md`](RT_TRACE_RESULT.md): senza addestramento m1 non lega nemmeno una variabile. Codice:
`benchmarks/sparse_capacity/sc_proc.py` (P), `sc_trace.py` (T), `sc_rt_train.py` (la ripresa).

**Stato: bozza.** E scritta e pushata prima della taratura su m1 (§3): la regola di misurabilita e fissata prima di
vedere i numeri. Il resto (celle primarie, soglie per D-R, revisore) si completa dopo la taratura e prima di lanciare
gli altri otto modelli. Fino ad allora il runner rifiuta tutti i bracci diversi da m1 (`REGISTERED = False`).

## 1. Domanda

D-R, nella forma che conta: **una banca piu grande, a nucleo fisso, puo memorizzare molte procedure, non solo molti
fatti?**
- Il legame di variabili (T) e una procedura sola, lavoro del nucleo: e piatto quasi per costruzione. Resta come
  lettura.
- La sonda principale e P: N funzioni con un nome, ognuna una regola semplice e casuale, applicate a input mai visti
  con quel nome.
- **L'esito chiave e la pendenza di P con la banca, accanto a quella di K.** K misura i fatti.

## 2. Le sonde

- **P** (`sc_proc.py`): tre rubriche, N = 256, 2.048 e 16.384 funzioni (le taglie di K), con la stessa quota di
  token ciascuna.
  - Ogni funzione e una regola su stringhe di 4 cifre, fatta di 1 o 2 operazioni prese da 36: somma costante per
    cifra mod 10, 9 − d, prodotto per 3, 7 o 9 mod 10, inversione, rotazione, 19 permutazioni delle posizioni. Ogni
    operazione e una biiezione, quindi lo e ogni regola. L'identita e esclusa.
  - Riga d'allenamento: `assert f_kqzvw("3812") == "4923"`.
  - Test: 2 input per funzione, mai usati con quella funzione in allenamento (il selftest lo controlla su 200.000
    righe). Il contesto prima della domanda non mostra mai la funzione chiesta, quindi vale la memoria e non un
    esempio nel contesto.
  - Punteggio: tutti i token della risposta giusti in greedy, come per K.
  - Esposizioni per funzione in 15 M token: circa 91, 11 e 1,4.
- **Controlli di P:**
  - funzioni mai viste (nomi mai allenati, regole a caso): il livello a cui il modello tira a indovinare;
  - copia dell'input (2% per costruzione: certe regole lasciano fisse alcune stringhe);
  - l'uscita piu frequente della funzione in allenamento: 0 per costruzione, perche le regole sono biiezioni e gli
    input di test non sono mai allenati;
  - la regola piu frequente della rubrica applicata a tutte le funzioni, la migliore regola cieca al nome (dal 3 al
    5%).
- **T** (`sc_trace.py`, versione allenata): i programmi di legame della sonda nativa, in formato `assert`.
  - Celle d'allenamento: m ∈ {2, 3, 4} con d ∈ {1..4}, piu due celle con trappola.
  - Celle fuori distribuzione: m3 d6 e m5 d4.
  - Test disgiunti dall'allenamento. Controlli senza modello come nella sonda nativa. Solo lettura.
- **K:** la rubrica della fase 2 resta nel flusso al 4% e si valuta con lo stesso codice.

## 3. La ripresa e la taratura

- **Ripresa** (`sc_rt_train.py`):
  - parte dallo stato della fase 2 al passo 7.324, pesi e momenti di AdamW (`last.pt`);
  - 15 M token (1.831 passi), lr costante 3e-4 dopo 100 passi di warmup;
  - flusso: 76% codice in posizioni a caso, 4% K, 10% P, 10% T. Stessi batch per ogni braccio con lo stesso seed;
  - curve di P, T e BPB ogni 250 passi, piu una al passo 0.
- **Taratura: solo m1 s0 e s1** (il modo `--tune` rifiuta gli altri bracci).
- **Regola di misurabilita su P,** fissata qui prima di vedere i numeri, guardando la fine della ripresa:
  - una rubrica e **misurabile** se, in tutti e due i seed di m1:
    - P sta sopra il controllo piu forte (il massimo fra funzioni mai viste, copia e regola piu frequente) di almeno
      0,10 e di almeno 4 errori standard binomiali;
    - P sta al massimo a 0,85, cosi c'e spazio per salire;
    - lo scarto fra i due seed e al massimo la meta di quel margine;
  - **la rubrica primaria per D-R** e la piu grande fra quelle misurabili;
  - **se nessuna rubrica e misurabile,** ci si ferma e si scrive: niente altri bracci. La pagina dice se P era sotto
    (al caso) o sopra (piena), e cosa cambierebbe (piu token, rubriche diverse).
- **Dopo la taratura:**
  1. la rubrica primaria e la soglia per D-R, dalla varianza fra seed di m1, vanno in questo protocollo;
  2. si scrivono "Cosa toglie al design" e la riga sulla tesi;
  3. il revisore indipendente (regola 6);
  4. il push, con `REGISTERED = True`;
  5. solo allora gli altri otto modelli: m4, m16, m64 con due seed, i due densi con uno.

## 4. Letteratura (letta prima del protocollo)

In [`LITERATURE.md`](LITERATURE.md), aggiunte del 2026-11-27. In breve:
- **Mixture of Parrots** (arXiv 2410.19034): piu esperti a parametri attivi fissi aiutano la memoria (una rubrica) e
  non il ragionamento (un solo algoritmo, il cammino minimo). Non prova molte procedure diverse.
- **Allen-Zhu e Li, parte 3.3** (arXiv 2404.05405): circa 2 bit di conoscenza per parametro con 1.000 esposizioni, 1
  bit con 100. I MoE con 32 esperti perdono solo 1,3 volte rispetto alla stessa legge sui parametri totali. Sono
  fatti, non procedure.
- **Chen et al.** (arXiv 2208.02813) e **Kawata et al.** (arXiv 2506.01656): in teoria un MoE impara dati a cluster,
  con una funzione diversa per cluster, dove una rete sola non ci riesce. Ma con 4 cluster, e senza una scansione di
  capacita.
- **Nakamura et al.** (arXiv 2508.18672): piu esperti aiutano i compiti di memoria, mentre per il ragionamento c'e un
  ottimo (circa 20 token per parametro).
- **Nessuno dei sei** prova se una banca piu grande, a parametri attivi fissi, memorizzi **molte procedure** applicate
  a input nuovi. P sta in quel buco.
