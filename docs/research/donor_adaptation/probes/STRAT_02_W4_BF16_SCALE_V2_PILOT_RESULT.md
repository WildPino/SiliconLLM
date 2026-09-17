# STRAT-02 W4 g128 BF16-scale v2 — pilot reale limitato

**17 settembre 2026. Esito: `PILOT_COMPLETE` per l'apparato e i cinque
slice preregistrati; formato completo ancora non verificato.** Il
[brief congelato](../briefs/BRIEF_STRAT_02_W4_BF16_SCALE_V2_PILOT.md)
precede il run. Il codice è nel commit `4f98c00` (nessuna modifica al
W4-v1). La directory [raw write-once](../../../../benchmarks/donor_adaptation/density/results/strat02_w4_bf16_pilot_20260917_131442/)
conserva manifest, log di supervisione, risultato worker e cinque blob.

Il preflight ha confermato revisione
`d2a4949c9d4ad6cf47fbac131f7e020077332b21` di
`allenai/StdMoE_1b14b_1T_Preanneal`, runtime Transformers 4.57.1,
Torch 2.6.0+cu124, tokenizers 0.22.2, safetensors 0.8.0, keyset
modello/index di 6.259 tensori e 6.225 `nn.Linear` con 13.363.052.544
pesi. Il worker ha verificato gli 11 shard pin-nati una sola volta,
poi ha letto soltanto le prime `min(256,out)` righe dei cinque lineari.
Nessun heldout, forward del donor, download, T4 o conversione completa.

| Organo | Slice F32 `[out,in]` | Gruppi | Blob B | SHA-256 blob |
|---|---:|---:|---:|---|
| `model.layers.0.self_attn.k_proj.weight` | `[256,2048]` | 4.096 | 270.336 | `cf8322ab79fa8f929d868a17dd0acd4c8ca38456240d8592eb478c8a15ca9784` |
| `model.layers.0.mlp.gate.weight` | `[128,2048]` | 2.048 | 135.168 | `3b3ed4db0a85b5d9776c86e84310ac432e14230908ee3aa1192e1fdb7ebe8704` |
| `model.layers.0.mlp.experts.0.down_proj.weight` | `[256,1024]` | 2.048 | 135.168 | `872d471b5c1c0c86497ef490b5d7c080baf847c7c44507168a86cc06d4956315` |
| `model.layers.0.mlp.experts.0.gate_proj.weight` | `[256,2048]` | 4.096 | 270.336 | `672cf44470e4b6ce0f95505d5f0b88735c6dbcb956089300b2e9b0104e1a7d40` |
| `lm_head.weight` | `[256,2048]` | 4.096 | 270.336 | `d0d0304c5a8ec6eca0d2ded08008f134cf1e7598204ac8cc06fe609bfeb81f17` |

Per tutti i **16.384 gruppi campionati**: zero gruppi sorgente
realmente zero, zero scale BF16 invalide, byte tile identici al
riferimento scalare row-wise, decodifica scalar/vector F32 bit-identica
dal blob e `nn.Linear(X)` entro `atol=1e-5,rtol=1e-6` rispetto a
`X @ W_hat.T`. Ho ricalcolato indipendentemente SHA e dimensione
`66 byte/gruppo` di ogni blob: 5/5 corrispondono ai record raw.
Diagnostico direttamente dai blob: il router shared riga 127/gruppo 0
ha scala BF16 F32 `1,1793786077416215e-25` e 109 codici non nulli;
il primo gruppo expert-down che aveva fallito W4-v1 (riga 0/gruppo 3)
ha scala `1,5352302762394743e-16` e un codice non nullo. Questo prova
la rappresentabilità **di quei gruppi**, non la fedeltà delle loro uscite.

Supervisor `COMPLETE`, worker exit 0, elapsed 170,968 s, 34 campioni.
Private commit massimo worker 2.477.146.112 B, RAM fisica disponibile
minima 68.936.925.184 B: nessun cap risorsa superato. I test sintetici
del codice (10/10) e il preflight metadata/meta-model erano passati
prima del run. SHA-256 dei file di controllo:

| File | SHA-256 |
|---|---|
| `pilot_manifest.json` | `91cd6ee6282b9608d7b6a8c3be9710f9d7300fac14bfe3a0de8fc4731316119a` |
| `worker_result.json` | `b370ed9ff3560d54568ad3e7651541248000c62620d0634cf3562ff20a8fdd8e` |
| `supervisor_result.json` | `36b41b4929d7b4296996592e196084ff93342b07dac160ffa7d53f3d8ac4cfde` |
| `supervisor_log.jsonl` | `9a0c9631f4d3e57ff3ebd3e9a48823be9e9a80d878039a364c76a44b7e4a1bc2` |

**Decisione:** autorizza soltanto il prossimo censimento *count-only* di
tutti i 104.398.848 gruppi con cap e stop propri, come previsto dal
brief; non un export immediato. Il campione copre circa lo 0,016% dei
gruppi totali ed è selezionato per organo, non casuale. La precisione
BF16 della scala è diversa da F16: il pilot non misura ΔBPB, task,
generazione, traffico/token, kernel C o tok/s. Anche un futuro pass
qualitativo W4-all richiederà una riduzione strutturale dei byte attivi
prima di poter soddisfare 50 tok/s nell'SSM nativo.
