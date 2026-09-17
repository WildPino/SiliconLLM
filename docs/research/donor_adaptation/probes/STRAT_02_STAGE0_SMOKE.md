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

## Tentativo 3, 17 settembre 2026 ore 09:46–10:08 locali

Output write-once:
[`strat02_smoke_20260917_094609`](../../../../benchmarks/donor_adaptation/density/results/strat02_smoke_20260917_094609).
Snapshot esplicita E:, runtime pin-nato, nessuna T4. Il worker reale PID
19184 ha riverificato gli 11 shard, caricato il teacher mmap e completato
il forward della riga 0 di calibrazione. Esito del supervisore:
`APPARATUS_OBSERVATION`, exit 0, 267 campioni, 1.336,391 s complessivi.
La riga fissata di codice (hash
`17020c6b2147d0d33555bb96dbde6302b9e4859523e8d0bd1a9d73bb0177d11e`)
ha 1.788 token su 8.192 byte e 2.933,250114861314 bit, cioè
**0,3580627582 BPB**. Non è il baseline heldout, un delta, un gate di
qualità o token/s di decode.

Nel log: working set massimo campionato **44.838.666.240 B**, private
commit massimo campionato **56.918.380.544 B**, peak pagefile riportato
dall'OS **59.835.797.504 B**, RAM fisica disponibile minima
**29.657.972.736 B**. Nessun cap operativo è stato superato. Questi sono
numeri del worker reale, non del redirector del tentativo 2. La crescita
del private commit oltre 1 GB appare a 1.278,72 s, ma il worker non emette
timestamp di fase: attribuire il tempo precedente al solo hashing sarebbe
un'inferenza non verificata.

SHA-256 degli artefatti: `supervisor_manifest.json`
`89ec29d7e2938c3befa96ad6dc6941a13414b355e8a3bfe33bb8a610f6f4634a`,
`supervisor_log.jsonl`
`91025b7f40703b0f2d0242eb020c39bbbd980b85cfa2a7781ff98809682b9dd57`,
`worker_result.json`
`828bbeaa909df6bb0cd420b9456dfcdcc4868cc068ba1c90302f8e03f0a41876`,
`supervisor_result.json`
`3276e242fb238a3476bf99f863aca2ae1a288e1b93dfd76544cff68f877a5ccc`.
Il prossimo gate informativo è lo scoring **di tutti i 96 documenti
heldout**, con il teacher caricato una volta sola; nessuna decisione su W4/W2
va tratta da questo smoke di calibrazione.
