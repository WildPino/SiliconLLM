# STRAT-02D tentativo 1 — `VOID_RESOURCE` prima di ogni score

**17 settembre 2026.** Il run write-once
`benchmarks/donor_adaptation/density/results/strat02_w4_organ_attribution_20260917_180628`
è stato fermato dal supervisore dopo `1287,843 s`:
`child_private_commit_above_70_gib`. Ultimo campione: private commit
`110.446.510.080` B (102,86 GiB), working set `685.555.712` B,
RAM fisica disponibile `67.438.862.336` B. Non esiste `worker_result.json`
né alcun JSONL per i quattro bracci; stdout e stderr sono vuoti.
**Nessun forward, BPB, task, rollout o rate è stato misurato.** Il
tentativo non è un `FAIL_BPB` e non cambia il risultato W4 precedente.

Il runner aveva aperto contemporaneamente 11 handle `safe_open` sui file
safetensors F32 e *poi* allocato la singola arena candidata F32. Il log
mostra un salto del private commit a 52,22 GiB a `1262,7 s` e a 102,86
GiB a `1287,8 s`, prima che il working set superasse 0,7 GiB. Una prova
di apparato isolata sullo shard 11 ha osservato l'apertura di `safe_open`
aumentare il private commit di `5.384.077.312` B (file
`4.346.432.736` B); la chiusura dell'handle ha rilasciato gran parte
del commit. Questi dati sostengono la causa delle mappature concorrenti,
non una carenza di RAM fisica durante il forward.

SHA-256: supervisor result
`6911790db7ce6c9c58a20dd0eca94f7ee9d0f10d8a402fa5abd2445b8f7632de`,
manifest `c2a5376e8df47db7e6ca05a7bbc0f2c9c4c2a722e0543678391917c45a6be55b`,
resource log `175799d998d2fba7818f285ea48b356e33b60e87f9dfe7275d8bc889b198e961`.
Il codice runner fissato era SHA-256
`fa42d9856bd5b1062a096ab4652e9458abb3345f856b885e7e514ed6f4a63a1c`.

## Addendum di risorsa prima del retry

È consentita **solo** una modifica di lifetime degli handle F32: validare
il keyset degli shard sequenzialmente, chiudendo ciascuno; durante il
ripristino di un braccio aprire al massimo **uno shard sorgente alla
volta**, copiare i suoi tensori nelle stesse viste dell'unica arena,
chiuderlo prima di passare al successivo. Non accumulare viste/handle
che mantengano mappati tutti gli shard. Lasciare invariati selezione dei
quattro bracci, 48 documenti, ordine, score, rollback bitwise, sentinella,
formato W4, teacher, soglie e cap di risorsa. Nuovo output write-once,
nessun resume o overwrite del tentativo 1. Se il controllo di equivalenza
o il cap fallisce, classificare nuovamente `VOID`, senza promuovere
score parziali. Nessun heldout e nessuna T4.
