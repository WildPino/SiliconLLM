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

## Tentativo 2, 16 settembre 2026 ore 23:36 locali

Output write-once:
[`strat02_smoke_20260916_233639`](../../../../benchmarks/donor_adaptation/density/results/strat02_smoke_20260916_233639).
Il worker è uscito dopo 55,468 s con `NotImplementedError: Cannot copy out
of meta tensor; no data!`; non esiste un punteggio BPB. Il modello costruito
su `meta` conserva `model.rotary_emb.inv_freq` come buffer **non persistente**:
non compare nello `state_dict`, quindi `assign=True` dei 6259 tensori F32
non può rimpiazzarlo. Il piccolo test riproducibile
[`strat02_meta_rope_selftest.py`](../../../../benchmarks/donor_adaptation/density/strat02_meta_rope_selftest.py)
costruisce la vera classe Emo pin-nata con 74.176 parametri casuali, ottiene
lo stesso errore senza il ripristino e poi `max_abs_hidden=0` fra modello
CPU e modello `meta`+`assign=True` con RoPE ricostruito. Non apre shard del
pretrained: dimostra la correzione dell'apparato, non qualità donor.

Un secondo difetto del supervisore ha invalidato i campioni di memoria del
tentativo 2: su Windows `.venv\Scripts\python.exe` è un launcher che crea
un altro interprete. Il monitor ha campionato il launcher (~5 MB), non il
worker reale. Perciò anche questo tentativo è **VOID_APPARATUS**, non una
misura di RAM o velocità. Il lancio successivo bypassa il launcher per il
worker e il self-test controlla che `Popen.pid` identifichi l'eseguibile
effettivamente campionato. Gli output precedenti restano intatti.
