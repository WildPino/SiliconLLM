# STRAT-02 Stage 0 — primo forward del teacher

## Tentativo 1, 16 settembre 2026 ore 23:15 locali

Output write-once:
[`strat02_smoke_20260916_231512`](../../../../benchmarks/donor_adaptation/density/results/strat02_smoke_20260916_231512).
Il supervisore ha creato il worker PID 8516 e lo ha terminato subito, a
`elapsed_seconds=0.0`, prima di qualunque campione, hash degli shard o
forward. `worker_result.json` non esiste. La causa osservata è
`AttributeError: 'Popen' object has no attribute 'memory_info'`: il
campionatore riceveva il controllo `subprocess.Popen` anziché
`psutil.Process(pid)`.

Il JSON originale riporta `VOID_RESOURCE` come categoria del supervisore,
ma il verdetto scientifico corretto è **VOID_APPARATUS**. Non è evidenza di
insufficienza RAM, OOM, qualità o velocità del donor. Non si sovrascrive né
si cancella l'output del tentativo. Il codice corretto deve superare un
self-test con un vero processo figlio minuscolo e campionamento `psutil` prima di
una nuova esecuzione write-once.
