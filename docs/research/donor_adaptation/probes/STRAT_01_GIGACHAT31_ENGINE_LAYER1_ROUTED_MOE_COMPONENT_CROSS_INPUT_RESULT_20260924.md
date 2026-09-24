# STRAT-01 GigaChat 3.1 layer-1 routed-MoE component cross-input result

**Canonical verdict:** `LAYER1_ROUTED_EXPERT_DOWN_OUTPUT_RESIDUAL_SUFFICIENT`

The sole scientific diagnostic authorized by the frozen
[protocol](STRAT_01_GIGACHAT31_ENGINE_LAYER1_ROUTED_MOE_COMPONENT_CROSS_INPUT_PROTOCOL_20260924.md)
completed from qualified producer commit
`37c639f38222fc864b84cbec48c61b69e2bd4cba`. It reports no error, exactly
one diagnostic invocation, and zero donor/reference graph executions.

Canonical raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer1_routed_moe_component_cross_input_20260924/`

| record | SHA-256 |
|---|---|
| `adjudication.json` | `ac78794d24467b2ca4c3fce092b969bfd9fa9a2698d2227d5ea5e5c67640ab59` |
| C binary | `cffd6985a9a76cad861b44714f7f77e0c576c43f788a8f2a9b2ab390a0592ed4` |

The complete orchestration took `40.342` seconds. This is diagnostic cost,
not emitted-token throughput.

## Scientific outcome

Replacing only the four selected expert-down outputs with their captured C
values exactly reproduces the predecessor failing downstream hash. Replacing
only the normalized router weights passes by more than four orders of
magnitude at the NRMSE gate.

| arm | downstream NRMSE | normalized maximum | result |
|---|---:|---:|---|
| captured reference routed output | `0` | `0` | PASS |
| captured C routed output | `0.002642086799405445` | `0.004687597394884091` | **FAIL** |
| reference down × reference weights | `0` | `0` | PASS, byte-exact replay |
| C down × C weights | `0.002642086799405445` | `0.004687597394884091` | **FAIL**, byte-exact replay |
| C down × reference weights | `0.002642086799405445` | `0.004687597394884091` | **FAIL** |
| reference down × C weights | `1.467939211440747e-7` | `4.900371526567658e-7` | PASS |

The decisive expert-down arm has weighted SHA-256
`c26abad90be611b33824373d48b3d016248099884400d499ffe6bdcfdd582de5`,
routed-output SHA-256
`a63d05e6532b6a6c71aa6b09520c4fb8cf076546e6bd293bc97c3aa06c205d81`,
and downstream SHA-256
`81635464fe7ec02d659bbc70d4d961e07743a670720bb3c4c1842dfcbc3ac254`.
The inverse arm's downstream SHA-256 is
`b6d6d8b2e43ffa14637db5c767b08f9db250977bb6c1e6d4456e88a0f3483e96`.

Direct C-versus-reference errors are descriptive:

| component | NRMSE | normalized maximum |
|---|---:|---:|
| selected expert-down outputs | `7.95119111602408e-5` | `9.274169798632739e-5` |
| normalized router weights | `4.17108215410039e-7` | `4.913789453949477e-7` |
| routed ordered sum | `1.3547864451924633e-5` | `1.1138183142366114e-5` |

## Replay and controls

- top-4 IDs are byte-identical in both schedules and origins;
- captured anchors and computed reference/reference and C/C arms replay every
  downstream stage byte-exactly;
- all frozen weighted and routed-output hashes match;
- prefill and cached-composition twins are byte-exact;
- one-byte mutations of every admitted payload and top-4 IDs are refused;
- arm-label swapping is rejected;
- negating all four reference down slots at token 7 rejects downstream at
  NRMSE `0.20962964201860176`;
- all apparatus self-tests and Python tests passed before the diagnostic.

## Interpretation and stop rule

Close router selection, normalized router weights, the routed weighting/sum,
shared expert, terminal addition, and layer 2. Do not repeat this split.

The old [layer-1 Q6 cross-input result](STRAT_01_GIGACHAT31_ENGINE_LAYER1_Q6_CROSS_INPUT_RESULT_20260923.md)
already measured the same selected Q6 matrices on exact reference routed
SwiGLU and passed the direct down-output gate at NRMSE `7.10952e-8`. That local
gate no longer settles causality because this result proves downstream
amplification of a locally passing down residual. The sole non-duplicate
successor is therefore the separately frozen
[routed-Q6 residual propagation](STRAT_01_GIGACHAT31_ENGINE_LAYER1_ROUTED_Q6_RESIDUAL_PROPAGATION_PROTOCOL_20260924.md):
propagate the already captured Q6(reference-SwiGLU) output through the newly
closed weighting and downstream path. Do not execute Q6 or either graph again.

No repair, later-layer, tokenizer, logits, generation, task quality, RAM, or
rate claim follows. No `SPEED_LEDGER.md` update is due.
