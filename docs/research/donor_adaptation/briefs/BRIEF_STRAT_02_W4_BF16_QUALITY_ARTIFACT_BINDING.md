# STRAT-02 W4-BF16-v2 — binding dell'artefatto per il gate qualità

**Addendum preregistrato il 17 settembre 2026 dopo l'export e prima di
qualsiasi forward o BPB del candidato.** Non cambia soglie, dataset,
quantizer o protocollo del
[gate paired](BRIEF_STRAT_02_W4_BF16_QUALITY_SCORE.md); specifica solo
quale artefatto già prodotto quel gate deve leggere. Nessuna selezione
fra vari export: lo smoke parziale non è un candidato.

L'unico candidato autorizzato è la directory write-once
`benchmarks/donor_adaptation/density/results/strat02_w4_bf16_full_export_20260917_143753/`
con `COMPLETE/EXPORT_VERIFIED` e i seguenti SHA-256 dei file di controllo:

| File | SHA-256 |
|---|---|
| `artifact_manifest.json` | `06924c13b5ad2106929ee51c84442bf63b4b47c6528e90b1ea91de891d4a0a5c` |
| `plan_manifest.json` | `1c1512b3735f0f9a80b638908801dfba7894b781047c767078a8b0cba54c3e42` |
| `supervisor_result.json` | `d52e09748d594b26af89b8223dfcd372343e7b344bc9f10199d379e5e4c7faca` |

Il manifest dell'artefatto lega gli SHA dei 11 payload e dei loro record;
il loader deve ricalcolarli, verificare **11/11** shard e il ledger
6.259 record / 6.225 lineari / 104.398.848 gruppi /
7.712.677.888 byte. Un secondo audit ha già confrontato con `Get-FileHash`
gli 11 SHA e le lunghezze dei payload: tutti uguali al manifest. Il
run non ha eseguito forward del candidato, calibrazione o heldout.

Il denominatore rimane il file teacher write-once
`results/strat02_teacher_baseline_20260917_105941/teacher_scores.jsonl`
SHA-256 `96d284da8f223d613039cb9e2a87b24e7b6130701c0873eea0854c3e8290b224`.
Il prerequisito di stabilità teacher è
`results/strat02_teacher_repeatability_20260917_142728/supervisor_result.json`
SHA-256 `89390013e926fddfad5c3bd651b29f1ae861de2bb98996d297300343cf65d41f`,
con `PASS_REPEATABILITY`. Qualsiasi mismatch di questi hash prima del
forward del candidato è `VOID_APPARATUS`, non un nuovo braccio W4.

Il binding non dimostra che il modello ricostruito passi ΔBPB, task o
rollout, né che il Transformer MoE sia eseguibile sullo SSM
`benchmarks/phase60/engine.c` o a >=50 tok/s.
