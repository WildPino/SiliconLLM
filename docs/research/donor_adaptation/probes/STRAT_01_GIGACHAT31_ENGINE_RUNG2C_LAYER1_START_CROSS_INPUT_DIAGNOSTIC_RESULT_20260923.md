# STRAT-01 Rung-2C layer-1-start cross-input result

**Verdict:** `BLOCK0_TERMINAL_RESIDUAL_SUFFICIENT`.

Exact captured reference `l_out-0` passes the unchanged C layer-1 builder at
all eleven checkpoints through `kqv_out-1`. The Rung-2C failure is therefore
caused by propagation of the accepted block-0 terminal residual, not by a
standalone layer-1 normalization, projection, Q/KV construction, attention,
cache, or V-B defect.

## Reference-input arm

All checkpoints use the unchanged Rung-2C limits, NRMSE `<= 0.002` and
normalized maximum `<= 0.01`.

| checkpoint | NRMSE | normalized maximum | result |
|---|---:|---:|---|
| `attn_norm-1` | `0` | `0` | exact PASS |
| `q-1` | `7.8461e-8` | `1.8949e-7` | PASS |
| `kv_cmpr_pe-1` | `7.3952e-8` | `5.4799e-8` | PASS |
| `k_pe-1` | `6.7839e-8` | `9.5555e-8` | PASS |
| `kv_cmpr-1` | `2.0649e-7` | `1.8258e-7` | PASS |
| `q_pe-1` | `9.9296e-8` | `9.7431e-8` | PASS |
| `q_nope_absorbed_perm-1` | `6.0574e-6` | `5.2977e-5` | PASS |
| `Qcur-1` | `6.0435e-6` | `5.2977e-5` | PASS |
| `Kcur-1` | `6.7884e-8` | `9.5555e-8` | PASS |
| `Vcur-1` | `2.0649e-7` | `1.8258e-7` | PASS |
| `kqv_out-1` | `0.0003855999` | `0.0011237641` | **PASS** |

The largest per-token `kqv_out-1` NRMSE is token 5 at `0.0009195369`, still
inside the frozen aggregate and per-token descriptive scale. By contrast, the
native C-input replay remains the exact Rung-2C failure at NRMSE `0.0061362`.

## Controls and canonical evidence

- all eleven C-input checkpoints replay byte-exactly;
- all frozen C/reference metrics reproduce within `1e-12`;
- both one-byte input mutations are refused and arm-label swapping rejects;
- token-7 negation rejects at NRMSE `0.8199017`;
- row-0/row-7 input swapping rejects at NRMSE `1.0468494`;
- donor graph executions: `0`.

The scientific invocation ran from HEAD
`dd9b13f6a8654ba0d3b622bdbe6227c822dc120c`. Canonical directory:

`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2c_layer1_start_cross_input_20260923/`

| record | SHA-256 |
|---|---|
| adjudication | `7c36f4df4d5b3b7e8947d77a3d9e83ce1f6b224d398870367a4888672d08f2da` |
| run manifest | `e71643986e2124341217ae8f7a27923e3a7454c71b72a084ac1c00b927d9a681` |
| reference-input `attn_norm-1` | `fc9d710b1124f79d7040b7d519930d6f0962f4d071fbb3e41fb7fa827de90606` |
| reference-input `Qcur-1` | `5d7b6e2d6ea176b5f09f9101726e27f4e0cc172ad59a199c2c3301c94d23285a` |
| reference-input `Kcur-1` | `d5965e028c08b8bb49588c5dbdc2d6028789c7842ed70fe701700e522b262b41` |
| reference-input `kqv_out-1` | `412af905430447831b956c05852236dd01dd15326408177884d931f8e27a3589` |

## Consequence and next coordinate

Do not modify layer-1 operators or their gates. The next admissible diagnostic
is an offline decomposition of block-0 terminal addition:

`l_out-0 = ffn_inp-0 + ffn_out-0`.

Using already captured reference and C terms, construct four exact additions:
reference/reference, C/C, C-`ffn_inp`/reference-`ffn_out`, and
reference-`ffn_inp`/C-`ffn_out`. Feed each result through the already validated
layer-1 builder and judge `kqv_out-1`. This determines whether the inherited
attention-stream residual, the dense-FFN residual, or only their interaction
is sufficient. It executes zero donor graphs and does not repeat any closed
checkpoint measurement.

This result does not repair or promote Rung 2C and makes no claim about MoE,
later layers, tokenizer, logits, generation, quality, RAM, or rate.
