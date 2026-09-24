# STRAT-01 layer-1 routed-SwiGLU component cross-input result

**Canonical verdict:** `LAYER1_ROUTED_UP_INPUT_RESIDUAL_SUFFICIENT`

The repaired scientific invocation authorized by the frozen
[protocol](STRAT_01_GIGACHAT31_ENGINE_LAYER1_ROUTED_SWIGLU_COMPONENT_CROSS_INPUT_PROTOCOL_20260924.md)
completed from commit `498c9d6dbed89392ddde99bd97ee10901c0469be`.
It reports no error, exactly nine completed Q6 arms, and zero donor/reference
graph executions.

Canonical raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer1_routed_swiglu_component_cross_input_repair1_20260924/`

| record | SHA-256 |
|---|---|
| `adjudication.json` | `04b46ac959b096bf8d2df8f1735882e37afbafacdc43c1c4eadff7560093a705` |
| C binary | `2189ebafc88c0b3eabc6e23ded5db0f8198d24e17b4e7a66df66cf0372e8c528` |

The orchestration took `39.958` seconds. This is diagnostic cost, not
emitted-token throughput.

## Scientific outcome

Under the byte-exact no-FMA SSE2 reference expression, replacing only the up
operand with its captured production C value exactly reproduces the complete
downstream failure. Replacing only the gate operand passes by more than four
orders of magnitude at the NRMSE threshold.

| arm | downstream NRMSE | normalized maximum | result |
|---|---:|---:|---|
| captured reference SwiGLU | `0` | `0` | PASS |
| captured production C SwiGLU | `0.002642086799405445` | `0.004687597394884091` | **FAIL** |
| scalar reference gate + reference up | `0` | `0` | PASS |
| scalar C gate + C up | `0.002642086799405445` | `0.004687597394884091` | **FAIL**, byte-exact C replay |
| SSE2 reference gate + reference up | `0` | `0` | PASS, byte-exact reference replay |
| SSE2 C gate + reference up | `1.467939211440747e-7` | `4.900371526567658e-7` | PASS |
| SSE2 reference gate + C up | `0.002642086799405445` | `0.004687597394884091` | **FAIL**, exact predecessor output |
| SSE2 C gate + C up | `0.002642086799405445` | `0.004687597394884091` | **FAIL** |
| token-7-negated reference control | `0.20962964201860176` | `0.3668577386863162` | expected rejection |

The decisive up-only arm's downstream SHA-256 is
`81635464fe7ec02d659bbc70d4d961e07743a670720bb3c4c1842dfcbc3ac254`,
identical to the captured production-C failure. The gate-only arm ends at
`b6d6d8b2e43ffa14637db5c767b08f9db250977bb6c1e6d4456e88a0f3483e96`.

Direct descriptive errors show the amplification chain:

| boundary | gate-only NRMSE | up-only NRMSE |
|---|---:|---:|
| routed SwiGLU | `1.2128371e-7` | `1.2415878e-7` |
| selected expert down | `8.9373569e-8` | `7.9511690e-5` |
| routed output | `5.9278452e-8` | `1.3546824e-5` |
| complete downstream | `1.4679392e-7` | `0.0026420868` |

The scalar reference expression differs from the captured reference SwiGLU at
`4.5150771e-8` NRMSE, but current Q6 quantizes that difference away and its
downstream output is byte-identical. Expression semantics are therefore not
sufficient here. `SSE2_FULL_C_REPAIR_PASSES = false`: changing expression
semantics alone cannot repair the up-input residual.

## Controls and stop rule

Reference and production anchors, scalar C replay, schedule twins, all frozen
hashes, every one-byte mutation refusal, label-swap rejection, planted control,
self-tests, and execution counters pass.

Close routed gate input, expression semantics, Q6/down, router/reduction,
shared, terminal, and layer 2. Do not repeat this matrix or infer that every up
matrix is defective. The sole non-duplicate successor is to recompute the same
32 selected current Q4_K up projections on immutable reference versus
production `ffn_norm-1`, then propagate each up payload through exact reference
gate, SSE2, current Q6, and the same downstream amplifier. This distinguishes
upstream normalized-input residual from up-projection residual without running
either graph.

No production repair, later-layer, tokenizer, logits, generation, task
quality, RAM, or rate claim follows. No `SPEED_LEDGER.md` update is due.
