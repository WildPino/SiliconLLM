# STRAT-02E — scout W2-BF16 degli expert con router preservato

**Eseguito il 18 settembre 2026. Stato: `COMPLETE_DIAGNOSTIC`.** Diagnostico
CPU limitato alla calibration congelata: entrambi i bracci hanno 48/48 righe,
con gli stessi 48 ID unici, categorie, byte e token del controllo W4 e del
controllo router-F32. Non sono stati letti heldout né eseguiti task, rollout,
T4, port C o misure di token/s.

## Risultato

Il controllo W4 ha prodotto **0.6526310642410273 BPB**. I due bracci sono:

| braccio | BPB calibration | Δ vs W4 | Δ vs router-F32 control | byte attivi/token |
|---|---:|---:|---:|---:|
| `W2_EXPERTS_ROUTER_W4` | 0.8983155207143131 | +0.24568445647328604 | +0.2525993393665234 | 473038848 |
| `W2_EXPERTS_ROUTER_F32` | 0.856283509419243 | +0.20365244517821593 | +0.21056732807145342 | 487653376 |

Il ripristino del router F32 recupera **0.0420320112950701 BPB** dentro
W2. Questo gain è paired e non additivo con altri gain.

| braccio | code BPB | prose BPB | technical/general BPB | token |
|---|---:|---:|---:|---:|
| `W2_EXPERTS_ROUTER_W4` | 0.6087529350802249 | 1.1590494961367133 | 0.906539831150562 | 27189 / 32014 / 29553 |
| `W2_EXPERTS_ROUTER_F32` | 0.572274052713786 | 1.1182039657233556 | 0.8569903068714994 | 27189 / 32014 / 29553 |

Il corpus complessivo è 362405 byte e 88756 token; le categorie sono 16
documenti ciascuna, con 121152 / 131072 / 110181 byte per code / prose /
technical_general.

## Identità, apparato e controlli

Il brief congelato è
`briefs/BRIEF_STRAT_02E_W2_BF16_EXPERT_SCOUT.md`. L'output autorevole è
`benchmarks/donor_adaptation/density/results/strat02e_w2_bf16_expert_scout_20260918_094818`.
Il candidato è il preciso W2-BF16 g128 PTQ degli expert (6144 tensori), con
16 tensori router; un solo modello in RAM e 11 shard sorgente F32 verificati.
Il supervisor ha riportato `ok=true`, `COMPLETE_DIAGNOSTIC`, child exit 0,
`heldout_access=false`, 8966.297 s, 1584 campioni e `model_copies=1`.
`sentinel_w4_before`, `sentinel_w2_router_rollback` e `sentinel_w4_final`
sono tutti `true`.

La prova è una calibrazione del formato, non un giudizio universale su W2,
QAT o altri codebook. I byte/token sono ledger attivi teorici del brief e
non una misura di rate o di inferenza.

## Hash dei risultati

| output | SHA-256 |
|---|---|
| `w2_experts_router_w4.jsonl` | `b095a7169fe045db05f89471f7b8e1618bf735e350e41fe6faafee301aef9236` |
| `w2_experts_router_f32.jsonl` | `5b1e241f9dac94970beb9098916d3e00f59b1a3d92624215c493d5843dd46634` |
| `w2_tile_audit.jsonl` | `e39e613667e721c7c5dafa82ed32acb20efd263ddf372cd38467a96a31fbb766` |

Il risultato è quindi **non competitivo in questa specifica PTQ W2-BF16**
rispetto al W4 calibration control; non riapre né promuove il vecchio Stage 2
condizionale. Questo documento registra un diagnostico, non un successo di
progetto.
