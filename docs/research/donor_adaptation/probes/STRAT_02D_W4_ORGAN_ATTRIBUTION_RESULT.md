# STRAT-02D — attribuzione controfattuale del danno W4 per organo

**Eseguito il 17 settembre 2026. Stato: `COMPLETE`.** Diagnostico CPU
eseguito esclusivamente sui 48 documenti di calibration, 48/48 per ciascuno
dei quattro bracci. Non sono stati letti heldout né eseguiti task, rollout o
misure di token/s. Il precedente [gate W4-v2 heldout](STRAT_02_W4_BF16_QUALITY_RESULT.md)
resta `FAIL_BPB`; il [primo tentativo di questa cella](STRAT_02D_ATTEMPT1_VOID_RESOURCE.md)
resta `VOID_RESOURCE` separato, prima di qualunque score.

## Risultato

Il controllo W4-only ha prodotto **0,6526310642410273 BPB**. Il gain di ogni
braccio è `(bits_W4 - bits_arm) / bytes_cal` contro le stesse 48 righe W4:

| braccio | lineari F32 ripristinate | gain calibration BPB | BPB braccio | byte attivi extra/token |
|---|---:|---:|---:|---:|
| `ROUTER_F32` | 16 | 0,00691488289323727 | 0,6457161813477901 | 14.614.528 |
| `HEAD_F32` | 1 | 0,003552017620359213 | 0,6490790466206681 | 716.111.872 |
| `ATTENTION_F32` | 64 | 0,00876496219380739 | 0,6438661020472200 | 935.329.792 |
| `NONEXPERT_F32` | 81 | 0,019693910396024004 | 0,6329371538450033 | 1.666.056.192 |

Il ledger W4-all di controllo è 661.782.528 B/token. I bracci sono
attribuzioni controfattuali locali: i gain non sono una previsione di
qualità heldout, non sono additivi e non costituiscono un formato candidato
o un’approvazione di deployment.

| braccio | gain code BPB | gain prose BPB | gain technical/general BPB |
|---|---:|---:|---:|
| `ROUTER_F32` | 0,005368056746611005 | 0,007311001086216296 | 0,008144505764022036 |
| `HEAD_F32` | 0,001941074976173685 | 0,005053059747119892 | 0,003537716875127384 |
| `ATTENTION_F32` | 0,006242299901111637 | 0,011692897884906683 | 0,008055721899934976 |
| `NONEXPERT_F32` | 0,014371375581789580 | 0,023668680940776676 | 0,020818011765337263 |

## Identità e metodo congelato

Il candidato è l’export W4-v2 completo in
`benchmarks/donor_adaptation/density/results/strat02_w4_bf16_full_export_20260917_143753`.
Revisione sorgente: `allenai/StdMoE_1b14b_1T_Preanneal` @
`d2a4949c9d4ad6cf47fbac131f7e020077332b21`; tutti gli 11 shard sorgente sono
stati verificati. È stato usato un solo modello in RAM; i record W4 sono
stati ripristinati in-place dopo ogni braccio. Tutte le quattro verifiche
sentinel e rollback hanno valore `true`. `heldout_access=false`.

Il run ha riusato le 48 righe di calibration nell’ordine originale, con
tokenizer, EOS-prefix, assenza di troncamento, chunk head 128 e operatori
F32/eager del gate precedente. Schema: `strat02_w4_organ_attribution_v1`.

Hash di controllo dell’artefatto:

| oggetto | SHA-256 |
|---|---|
| `artifact_manifest.json` | `06924c13b5ad2106929ee51c84442bf63b4b47c6528e90b1ea91de891d4a0a5c` |
| `plan_manifest.json` | `1c1512b3735f0f9a80b638908801dfba7894b781047c767078a8b0cba54c3e42` |
| export `supervisor_result.json` | `d52e09748d594b26af89b8223dfcd372343e7b344bc9f10199d379e5e4c7faca` |
| W4 calibration control | `ef1d7f42ad30ff5727432bcc4f0c1a5299ff681f52a5a7690f55602f0171a142` |

## Audit del run

Il supervisor ha riportato `ok=true`, `COMPLETE`, exit code `0`, durata
`5917,953 s` e 1000 campioni. Gli hash output registrati e verificati
indipendentemente sono:

| output | SHA-256 |
|---|---|
| `router_f32.jsonl` | `f7913767de5c65396d90473cfd7302961c615ec09bcc0eeb65b3ff428a78550b` |
| `head_f32.jsonl` | `ab56638f422927444f83619ed64acb6694f3c3afc904c944205573f0136d7e34` |
| `attention_f32.jsonl` | `ca14f53a415ad06147b30ef14baa7c7b1006578de8d80962e8bb2a916d091b74` |
| `nonexpert_f32.jsonl` | `092753c83c6cbd0d82f5fc904ee9579f215f1d1106e8ea001aaa8ac2bec39d6e` |
| `supervisor_log.jsonl` | `8d15f285a6affaf44419c7931d0c5c0ccfc1fbdf808f90246e4ec4af83caafdb` |
| `supervisor_manifest.json` | `211abdd601a1ee40452699ff3815217a6c653a717b2930019f21a3f0b973bc23` |
| `worker_result.json` | `841546c6dd21f8cff06e70f6e9c01ca196a800101123fb254b055b8269799f68` |
| `worker_stdout.log` | `c3d23f2f5ea0da6af238c3789ef6c52494f287259bfae991df4e4686b4813f88` |
| `worker_stderr.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

Hash di provenienza registrati dal supervisor: brief
`29cfb7023c79b03f56859ec442811f1af730dce1ecefde249e93d8c02e4e1957`,
candidate loader `64fdefa71abea2eb18c8dae380bbe49dbe67320d382476099cfd355264b913ce`,
export verifier `63974d5665691eaaa5a0fb68627b583dc7dcefe2c3891fcd9df7e2b34db93b30`,
runner `628ba9660bb5ee9cdd2677476b2d9b24d45b994fba7e888999f68cfd18ac7a77`,
score core `b4259f1a5c1d99121dd9e705af45675209acdb9368c3f35111522ba3bace1056`,
teacher `47e047e94c94359a8f036b41b0a0ac40072fb59e010819f35ddf6a6f89b7a435`,
token audit `08b2e493bd0bb5149c6d7af6b021fdf2987d13605419343f38de0c387cb0e243`,
bounded supervisor `a761caffd46053c3df7d450b105e62f57530b84c5222217b88e19aaa35410886`.

Audit indipendente supplementare: SHA-256 di `supervisor_result.json`
`11866b2ec2c6ecc857e02c9b6c91dd157e7a3a1ea7b50fcdf37d3c440881b690`;
`worker_result.json` coincide con `841546c6dd21f8cff06e70f6e9c01ca196a800101123fb254b055b8269799f68`.
Il log contiene 1003 righe con `monitor.samples=1000`; peak private commit
`57,713 GiB`, peak working set `51,888 GiB`, minima RAM disponibile
`10,856 GiB`, stderr vuoto. I nove hash nella mappa output del supervisor
sono stati tutti verificati indipendentemente; lo SHA del supervisor è
riportato separatamente e non è incluso nella propria mappa. Gli ID/byte
dei documenti coincidono col controllo W4, 48 documenti unici per braccio;
sentinel e rollback sono `true` per tutti e quattro.

Questo documento registra il risultato diagnostico e non prescrive il passo
successivo né autorizza deployment.
