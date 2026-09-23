# STRAT-01 post-F16 layer-1-start cross-input apparatus result

**Status:** `APPARATUS_READY_NO_DONOR_EXECUTION`

The apparatus for the frozen
[post-F16 protocol](STRAT_01_GIGACHAT31_ENGINE_POST_F16_LAYER1_START_CROSS_INPUT_PROTOCOL_20260923.md)
is qualified. It reuses the production layer-1 attention and MoE functions,
accepts only the exact reference and post-F16 current `l_out-0` payloads, and
can emit all 32 canonical checkpoints for both arms without executing a donor
or reference graph.

## Preserved apparatus VOID

The first apparatus invocation is retained as
`VOID_POST_F16_LAYER1_START_CROSS_INPUT`. It stopped before compilation
because four runner calls passed positional arguments to the keyword-only
`run_command` helper:

`unexpected TypeError: run_command() takes 1 positional argument but 4 were given`

No diagnostic invocation, model read, donor graph, or reference graph ran.
Its `adjudication.json` SHA-256 is
`03be72ab9001d3724eb7283b0bd498a34ea4bc42afb377aacec7226a5bbe9786`.
Repair 1 changed only those call sites to keyword arguments and moved the
fresh qualification to a separate directory.

## Repair-1 qualification

Raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_post_f16_layer1_start_cross_input_apparatus_repair1_20260923/`

- status: `APPARATUS_READY_NO_DONOR_EXECUTION`;
- errors: none;
- diagnostic invocations: 0;
- donor/reference graph executions: 0 / 0;
- all 32 immutable reference payloads and all 32 post-F16 current payloads
  validate by manifest containment, byte count, SHA-256, dtype, shape, and
  finiteness;
- post-F16 prefill and cached manifests have identical hashes for every
  checkpoint, validating the frozen single-schedule reduction;
- all seven source controls pass, including production attention/MoE reuse,
  no local Q4/Q6 reimplementation, exact-helper accounting, and zero-graph
  reporting;
- 146 STRAT-01 Python tests pass in 115.492 seconds;
- 18 C self-tests pass, including the new five-check apparatus self-test and
  the 73,024-check legacy kernel suite.

| record | SHA-256 |
|---|---|
| repair-1 `adjudication.json` | `7c75e6f423f5d81350a959245751b9b0d29b8ad7b6e6c5073a231c933212a859` |
| compiled apparatus binary | `7fed157ebea54d7a1f4cfcee955c6780629f7072d1b63b96dec4f854a5b47406` |

The apparatus observed protocol commit
`2a5f54adeeb361cceff498f65cd56bd2d052fe21` and took 127.411 seconds in
total. These durations are apparatus diagnostics and are not model-rate
measurements.

## Authorization and stop rule

Apparatus construction is closed. After committing the qualified sources,
exactly one offline diagnostic invocation is authorized. It must read the
accepted artifact once, run the two frozen start-state arms plus the two
causal controls, emit exact helper counts, and keep donor/reference graph
counts at zero.

Do not repeat either apparatus run, overwrite the preserved VOID, run either
Rung-2C graph, or modify F16, Q4, SwiGLU, Q6, routing, or tolerances before
that result. No quality, RAM, or rate claim follows from apparatus readiness.
