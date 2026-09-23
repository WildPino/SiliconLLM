# STRAT-01 post-F16 block-0 terminal-component apparatus result

**Status:** `APPARATUS_READY_NO_DONOR_EXECUTION`

The apparatus for the frozen
[post-F16 block-0 terminal-component protocol](STRAT_01_GIGACHAT31_ENGINE_POST_F16_BLOCK0_TERMINAL_COMPONENT_CROSS_INPUT_PROTOCOL_20260923.md)
is qualified. It reuses the production block-0 and complete layer-1
functions, introduces no local quantized-dot implementation, and stops before
reading or executing the accepted artifact.

Raw apparatus directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_post_f16_block0_terminal_component_cross_input_apparatus_20260923/`

- status: `APPARATUS_READY_NO_DONOR_EXECUTION`;
- errors: none;
- diagnostic/model invocations: 0;
- donor/reference graph executions: 0 / 0;
- all 32 immutable reference checkpoints and all 32 post-F16 current
  checkpoints validate;
- the four frozen component/start-state inputs validate by containment, byte
  count, SHA-256, dtype, shape, and finiteness;
- the reference block-0 terminal sum reconstructs byte-for-byte before any
  model execution;
- all seven source controls pass;
- 152 STRAT-01 Python tests pass in 124.116 seconds;
- all 19 C self-tests pass, including the new five-check apparatus self-test
  and the 73,024-check legacy kernel suite.

| record | SHA-256 |
|---|---|
| apparatus `adjudication.json` | `f1f118968e04423d72b63ec9bbbe2d29a2863ea1b2b668104832e2afee93b010` |
| compiled apparatus binary | `a915ca81a1b238700fad0be600a70351651e9be21707291c14b600f0545b3a90` |

The apparatus observed documentation commit
`f5bf842823f898140ea0e56c421eef5e60646097` and completed in 131.876
seconds. These are qualification durations, not model-rate measurements.

## Authorization and stop rule

Apparatus construction is closed. After committing its exact sources,
exactly one scientific diagnostic invocation is authorized. It may produce
the current post-F16 block-0 components once, require exact homogeneous
replay, and propagate only the two frozen hybrids plus two causal controls
through current layer 1. Expected exact helper totals are `5760` QK and
`655360` value calls.

Do not rerun apparatus, the old component split, Stage A, either Rung-2C
schedule, or the closed post-F16 layer-1-start cell. No quality, RAM, or rate
claim follows from apparatus readiness.
