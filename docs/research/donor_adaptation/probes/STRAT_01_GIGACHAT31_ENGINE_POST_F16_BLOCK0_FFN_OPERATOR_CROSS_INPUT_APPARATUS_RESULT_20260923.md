# STRAT-01 post-F16 block-0 FFN operator-chain apparatus result

**Status:** `APPARATUS_READY_NO_DONOR_EXECUTION`

The repaired apparatus for the frozen
[post-F16 FFN operator-chain protocol](STRAT_01_GIGACHAT31_ENGINE_POST_F16_BLOCK0_FFN_OPERATOR_CROSS_INPUT_PROTOCOL_20260923.md)
is qualified. [VOID 1](STRAT_01_GIGACHAT31_ENGINE_POST_F16_BLOCK0_FFN_OPERATOR_CROSS_INPUT_APPARATUS_VOID1_20260923.md)
remains apparatus history and contributes no scientific values.

Raw qualification directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_post_f16_block0_ffn_operator_cross_input_apparatus_repair1_20260923/`

- status: `APPARATUS_READY_NO_DONOR_EXECUTION`;
- errors: none;
- diagnostic/model invocations: 0;
- donor/reference graph executions: 0 / 0;
- all 158 STRAT-01 Python tests pass in 112.732 seconds;
- all 20 registered C self-tests pass, including the repaired six-check
  operator-chain self-test and the 73,024-check legacy kernel suite;
- all four frozen inputs, three predecessor adjudications, the exact
  all-reference identity bridge, and all 64 reference/current full-layer
  checkpoints validate;
- all eight source controls pass: CLI registration, production SwiGLU, Q6,
  and complete layer-1 delegation, no local quantized dot, exact helper
  accounting, zero-graph contract, and protocol precedence.

| record | SHA-256 |
|---|---|
| repaired apparatus `adjudication.json` | `3838129ffabb5f9c4745efe771fd93af77cc4e215a2d0476e69436f808f9afab` |
| compiled apparatus binary | `d4fa5f1fce6be7bd6adc87ad747c7bc7f1351034d7eff220546a9a07f0786179` |

The apparatus observed documentation commit
`1369b81725dcba8d0e639176a2b4afe256b545a6` and completed in `120.584`
seconds. These are qualification durations, not model-rate measurements.

## Authorization and stop rule

Apparatus construction is closed. After committing its exact sources,
exactly one scientific diagnostic invocation is authorized. It may read only
block-0 `ffn_down.weight` and the qualified layer-1 tensors, build the
Q6-only and expression+Q6 arms plus two causal controls, and execute four
complete layer-1 propagations. Expected exact totals are `4608` QK calls,
`524288` value calls, and four block-0 Q6 down arms.

Do not rerun apparatus, block 0, gate/up, either terminal-component cell, or
the old FFN-down/SwiGLU cells. No quality, RAM, or rate claim follows from
apparatus readiness.
