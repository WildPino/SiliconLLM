# STRAT-02 W4 BF16-scale v2 — censimento completo count-only

**Preregistrato il 17 settembre 2026 dopo il [pilot reale limitato](../probes/STRAT_02_W4_BF16_SCALE_V2_PILOT_RESULT.md),
prima della scansione completa. Stato: PROPOSED.** Il pilot ha rappresentato
16.384 gruppi su 104.398.848, non un campione casuale. Questo esperimento
risponde a una domanda binaria: il nuovo formato W4 g128 BF16-scale v2 è
**definito per ogni gruppo F32 non nullo** dei 6.225 lineari pin-nati?
Nessuna conversione, score, inferenza, heldout, training, T4 o tempo/token.

## Input e calcolo

- Stesso donor/revisione/indice/manifest `strat02_weight_sources.json`
  del pilot, keyset modello/index di 6.259, inventario `nn.Linear` su
  `meta` di 6.225 matrici, 13.363.052.544 pesi e 104.398.848 gruppi.
  Nulla è selezionato da un pattern di stringa al posto dell'inventario.
- Visitare gli 11 shard nell'ordine nominale del manifest. Un **worker
  fresco per shard**, lanciato e monitorato sequenzialmente dal parent,
  verifica dimensione e SHA-256 dello *shard assegnato* contro il manifest
  **una volta prima di leggere i valori**. Il parent confronta manifest,
  indice e inventario. Un pass completo quindi verifica tutti gli 11
  shard una volta; un controesempio anticipato verifica solo lo shard
  necessario e non è chiamato censimento completo. Prima e dopo le
  letture del worker, controllare che file pin-nato e metadata non siano
  cambiati. Vietati download, shard non pin-nati e worker concorrenti.
- Per ogni lineare dello shard, in ordine lessicografico, leggere slice
  di al più 128 righe F32 e partizionarle in gruppi contigui da 128.
  Per ogni gruppo finito: `m=max(abs(w))` in F32; `m==0` è zero vero e
  valido. Per `m>0`, calcolare con l'esatta sequenza F32 del
  [codec v2](../../../../benchmarks/donor_adaptation/density/strat02_w4_bf16_codec.py)
  le scale ai due estremi `c=0,50` e `c=1,00`, cast BF16 RNE, poi
  ri-espansione F32. La monotonicità delle operazioni positive rende
  questi estremi sufficienti: se il minimo è positivo e il massimo è
  finito, gli altri nove candidati sono validi. Validare questo shortcut
  contro il loop a 11 candidati su controlli sintetici piantati: zero,
  min-subnormal/underflow, halfway RNE, finite/overflow e gruppi
  casuali log-uniformi. Confrontare alcuni gruppi reali validi con il
  riferimento scalare completo (senza usarlo come scorciatoia globale).
- Scrivere soltanto **conteggi per tensore**: gruppi, zero veri,
  source-nonfinite, underflow scala, overflow/nonfinite scala. Niente
  coordinate massive né valori dei pesi. Output JSONL append-only con
  flush, SHA e manifest write-once. Cap 16 MiB per il file conteggi:
  superarlo è `INCOMPLETE`, non troncare e dichiarare `COMPLETE`.

## Stop, risorse e interpretazione

Il primo gruppo non nullo con qualunque scala BF16 candidata invalida
è un **controesempio decisivo** `FORMAT_INVALID_COUNTEREXAMPLE`:
registrare tensore, shard, riga, gruppo, classificazione, `absmax`,
scale estreme e conferma del codec scalare v2 su quei 128 F32; fermarsi
senza scandire gli shard restanti. Un valore sorgente non finito è
`VOID_SOURCE`, non una prova sulla bontà del BF16. Un mismatch tra
endpoint e 11 candidati, hash, inventario o contabilità è
`VOID_APPARATUS`. Stop da cap/supervisione è `VOID_RESOURCE`; conservare
sempre parziali come `INCOMPLETE`, senza resume implicito.

Preflight RAM fisica disponibile ≥8 GiB e spazio output ≥1 GiB.
Monitorare **solo il PID worker lanciato** ogni 5 s; fermarlo se RAM
disponibile <4 GiB o private commit >8 GiB. Cap 15 minuti per shard,
60 minuti complessivi (inclusi hash). Ogni shard chiuso libera la sua
memoria prima del successivo; niente secondo modello F32 in RAM.
Directory risultato nuova/write-once, log e stdout/stderr per worker.

Un `COMPLETE/FORMAT_VALID_FULL` richiede 11 SHA corretti,
6.225/6.225 lineari e 104.398.848/104.398.848 gruppi contabilizzati,
zero invalidi/source-nonfinite, e i **256 gruppi** delle righe shared
router dei 16 layer non nulli e BF16-validi. Controllare anche
`model.layers.0.mlp.experts.0.down_proj.weight[0,3]` non nullo e valido:
sono controlli positivi emersi dal v1, non gruppi da saltare.

Se completo e valido, autorizza **solo** un brief per export completo
v2 e successivo gate BPB/task paired; non un quality/rate PASS né un
port in `engine.c`. Un controesempio vieta l'export uniforme v2 e
richiede una precision map nuova. A prescindere dall'esito, i
661.782.528 byte W4-all per token restano il ledger aritmetico e
richiedono una riduzione strutturale per il target 50 tok/s.
