# STRAT-03 — risultato diagnostico CPU

**Data:** 2026-09-16. Nessuna T4 o download di pesi.

**Verdetto:** `NONLINEAR_BENEFIT=false`, `ROUTER_TRANSFER=false`,
`EXECUTABLE_LOCAL_SIGNAL=false`. Apparato `VALID`; il run è un diagnostico
locale su cinque layer FFN del donor `Qwen/Qwen2.5-1.5B`, revisione
`8faed761d45a263340a0528343f099c05c9a4323`, in CPU fp32/eager.

Il confronto usa 294.912 pesi shared sia per `O-L96` sia per `O-N64`; il
residuo fit-target è `R=y-y_sparse_oracle`. Gli aggregati sono somme SSE, non
medie di percentuali.

| layer | `O-N64/O-L96` | `X-N64/O-N64` | `X-N64/MASS` | recall top-3 router |
|---:|---:|---:|---:|---:|
| 1 | 0.5128793020 | 1.6088320503 | 1.4601628039 | 0.4153645833 |
| 7 | 1.0059415752 | 1.1283767817 | 0.9885715962 | 0.3404947917 |
| 14 | 1.0633533913 | 1.0618661010 | 0.8504136557 | 0.2968750000 |
| 21 | 1.0277456361 | 1.1830551697 | 1.0258264309 | 0.3743489583 |
| 27 | 1.1058742387 | 1.1003353110 | 0.5344148248 | 0.7682291667 |
| **aggregato / media recall** | **1.0863560374** | **1.1110939899** | **0.5782647468** | **0.4390625000** |

SSE aggregate: `MASS=7,988,513.917761`, `O-L96=3,827,099.313126`,
`O-N64=4,157,592.444421`, `X-L96=4,100,669.157156`,
`X-N64=4,619,475.977642`.

## Gate indipendenti

- **`NONLINEAR_BENEFIT`: FAIL.** Clausole: `O-N64/O-L96 ≤ 0.80` aggregato:
  **false** (`1.0863560374`); almeno 4/5 layer con SSE `≤ O-L96`:
  **false**, solo **1/5**; nessun layer `>1.05×O-L96`: **false** (layer 14 e
  27 violano). Il nonlineare batte il lineare solo al layer 1, ma fallisce
  l'aggregato e le clausole per-layer.
- **`ROUTER_TRANSFER`: FAIL.** Clausole: `X-N64/O-N64 ≤ 1.20` aggregato:
  **true** (`1.1110939899`); nessun layer `>1.50×O-N64`: **false** (layer 1);
  recall top-3 media `≥0.50`: **false** (`0.4390625`). Quindi il router
  fallisce nonostante il costo SSE aggregato rientri in `1.20`.
- **`EXECUTABLE_LOCAL_SIGNAL`: FAIL.** Clausole: `X-N64/MASS ≤0.50`
  aggregato: **false** (`0.5782647468`); almeno 4/5 layer `≤0.80`:
  **false**, **1/5**; nessun layer `>1.05`: **false** (layer 1);
  p95/token non superiore a MASS in almeno 4/5 layer: **false**, **2/5**.

## Provenienza e controlli

Il fit usa `calib` 8×512 (seed 424242), lo score le posizioni heldout E67/E68
fisse (sequenze 0–3, posizioni 384–511); `heldout_used_for_fit=false`.
I MASS anchor E68 sono verificati dal runner entro tolleranza relativa `1e-5`.
Il JSON completo registra `planted_selftests=PASS`, ricostruzione
selective-vs-dense entro `1e-5` e protocollo write-once.

Hash SHA-256 verificati (atteso=osservato nel provenance): brief
`386c3bf121c8d0a5243561a8f14405d07d858fa21baa8aa8d610125d85cd2fff`, runner
STRAT-03 `9613890c9ccb9bfe853ced29bc7dfd14240591b525b9ceebc673d14a31257765`,
runner E68 `0fd10804742ce856f626314a2a251a8982f4a9ff6d63da2d48f79ca815df867e`,
risultato E68 `02e5ae986ad3f59241d78602378bb7e17575c3bbf38f78db13244442da91be47`,
labels `c39d0740b7f754daf168d811c77120c5394ae677e3882fef208d331d8c333c9c`,
`density/common.py` `7ac00b31e91d0018c775e2e9b26120c0ec025075d7913218ea3cdcfe3196e461`.
Hash file del risultato completo: `581399a72c94ec6d3b8be33f8dbccd3ec1bcaca3c61186b88e324ba2d07de51e`;
smoke: `5104e7b7a6b254a500a50a45e1e3b430a711448adc7de1a3186cd2f4741c5bd6`.

Il smoke è `NOT_APPLICABLE_SMOKE` (un layer, una sequenza), non un secondo
verdetto.
Esecuzione locale con `.venv/Scripts/python.exe -u` del runner: `--smoke`
prima, poi senza flag per il full run, con `HF_HUB_OFFLINE=1` e
`TRANSFORMERS_OFFLINE=1`; entrambi terminati con exit code 0.
Tempi di processo registrati nel JSON: `62.85 s` smoke e `134.88 s` full,
**solo osservazioni operative**, non tempi di inferenza/token.

Le loss di calibrazione di router e SwiGLU diminuiscono in tutti i cinque
layer, ma questo non equivale a generalizzazione: al layer 1 il rapporto
`X-N64/MASS` passa da `0.6668` su calibrazione a `1.4602` su heldout.
È un segnale di mismatch/overfit compatibile con l'avvertenza E68, non
un'identificazione causale del problema. **Decisione:** non usare questa
specifica coppia shared/router per un pilot T4 o per un export. Un nuovo
tentativo richiede un operatore, target di routing o partizione distinti,
un budget byte esplicito e un nuovo brief; non si ritoccano gli step o le
soglie di STRAT-03 dopo l'esito.

## Limiti

Questo risultato non prova BPB, generazione, qualità end-to-end, costo byte
esatto, RAM/cache, parità o throughput dell'engine, tok/s, addestramento
congiunto, scala 10B o un risultato T4. Non dimostra inoltre che ogni router
economico fallisca, né refuta un shared nonlineare jointly trained, altri
selector/partizioni/rank o altre geometrie.

Fonti: [brief congelato](../briefs/BRIEF_STRAT_03_EXECUTABLE_SHARED_RESIDUAL.md),
[runner](../../../../benchmarks/donor_adaptation/engine/strat03_executable_shared_residual.py),
[risultato completo](../../../../benchmarks/donor_adaptation/engine/results/strat03_executable_shared_residual.json),
[risultato smoke](../../../../benchmarks/donor_adaptation/engine/results/strat03_executable_shared_residual_smoke.json),
[risultato E68](../../../../benchmarks/donor_adaptation/engine/results/e68_shared_residual.json),
[nota E68](E68_SHARED_RESIDUAL.md).
