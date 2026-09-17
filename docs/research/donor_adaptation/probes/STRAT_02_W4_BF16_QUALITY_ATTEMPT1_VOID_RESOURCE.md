# STRAT-02 W4-BF16-v2 — primo tentativo di qualità, `VOID_RESOURCE`

**17 settembre 2026. Nessun risultato qualitativo.** Il run CPU supervisionato è stato avviato sull'unico export preregistrato, senza T4, nella directory write-once `benchmarks/donor_adaptation/density/results/strat02_w4_bf16_quality_20260917_155914/`. Il runner era il commit `d617979` (senza firma). Prima del lancio erano passati 16 test sintetici di loader+runner, il `--preflight` reale di corpus/tokenizer/hash e la preparazione completa metadata+formato del candidato.

Il supervisore ha fermato il **solo worker diretto** dopo 696,625 s e 132 campioni: `VOID_RESOURCE`, motivo `child_private_commit_above_70_gib`. All'ultimo campione il commit privato era **75.683.090.432 B**, contro il limite preregistrato di 70 GiB = 75.161.927.680 B; working set **50.375.950.336 B** e RAM fisica disponibile **18.525.188.096 B**. I file `calibration_checks.jsonl` e `candidate_scores.jsonl` non esistono: **0/48** documenti di calibrazione e **0/96** heldout valutati. Nessun forward, BPB, task, rollout o rate valido deriva da questo tentativo.

| File di controllo | SHA-256 |
|---|---|
| `supervisor_manifest.json` | `0b106f0e4a42e916ad8d144f0c03c40ff0a5311dfac35de3bdd9685996152195` |
| `supervisor_log.jsonl` | `378d98fc06dd4b4a18fc71fe1b0b1d492638144ef30ba551e70d36265df2023e` |
| `supervisor_result.json` | `c071675fec5771fc5bfeaac9bcdbb4cdb2385ce26e46428388264b34b5a6c2d9` |

L'ipotesi operativa è che l'allocazione separata di 6.259 tensori F32 durante la decodifica accumuli più commit del payload F32 teorico (~54,27 GB). Il log non localizza l'allocazione responsabile, quindi **non è una causa dimostrata**. Un'eventuale correzione deve mantenere l'esatto artefatto W4, il denominatore teacher, i cap di 70/8 GiB e il gate CI invariati, con test e addendum prima di ritentare. Non alzare il limite a posteriori e non classificare il formato come `FAIL_BPB`.
