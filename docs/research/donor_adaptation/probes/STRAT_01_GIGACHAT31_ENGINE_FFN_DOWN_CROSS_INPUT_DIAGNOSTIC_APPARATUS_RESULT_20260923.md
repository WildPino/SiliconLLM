# STRAT-01 FFN down-projection cross-input apparatus

**Status:** `APPARATUS_READY_NO_DONOR_EXECUTION`

The frozen
[FFN down protocol](STRAT_01_GIGACHAT31_ENGINE_FFN_DOWN_CROSS_INPUT_DIAGNOSTIC_PROTOCOL_20260923.md)
is implemented without opening the accepted GGUF or executing a donor or
reference graph. The canonical apparatus-only directory is:

`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_ffn_down_cross_input_apparatus_20260923/`

Its adjudication and run-manifest SHA-256 values are
`f30438275cdb40f2c342b5019fc6ff8092659bbe2ba0178dda7c6bbbba595395`
and `d58c0e91d362cb42ddfce0c9900db2d83a569c1c067c31aa88a841e893cb6dea`.

## Qualified surface

The diagnostic-only engine command:

- validates the five frozen SwiGLU/down-output/reference-residual payloads;
- admits only the block-0 Q6_K down tensor and the seven frozen layer-1
  attention tensors;
- delegates both generated outputs to the unchanged production
  `strat01_r2b_q6_matmul_batch` helper;
- requires the C-input generated output to byte-replay frozen C `ffn_out-0`;
- composes captured-reference, exact-input/current-Q6, and
  C-input/current-Q6 terminal arms with exact reference `ffn_inp-0`;
- delegates all three sums to the unchanged qualified layer-1 builder and
  emits the same eleven checkpoints;
- emits direct/downstream negation and row-swap controls;
- records tensor descriptors, input/source hashes, and zero donor/reference
  graph executions, without self-certifying a result.

The external adjudicator independently binds the previous
`FFN_RESIDUAL_SUFFICIENT` record, validates report containment and provenance,
applies direct and propagated gates, and requires byte/hash/metric replay of
the prior captured-reference and C-FFN arms.

## Qualification evidence

- New FFN-down C self-test: PASS, 7 checks.
- Terminal-component C self-test: PASS, 9 checks.
- Rung-2A/Rung-2B/Rung-2C model-free regression: PASS, 28 + 6 + 14 checks.
- New Python tests: PASS, 5 tests.
- Terminal-component, layer-1-start, and Rung-2C Python regressions: PASS,
  16 tests.
- Legacy kernel self-test: PASS, 73,024 checks; worst error zero.
- Clang C11 `-O3 -mavx2 -mfma`, no fast-math: PASS with empty compiler
  stderr.
- Donor graph executions: 0.
- Reference graph executions: 0.

## Authorization and non-claims

This qualification authorizes exactly one non-VOID invocation of the frozen
cell after the implementation is committed. It is not an upstream-versus-Q6
result and makes no claim about a repair, gate/up/SwiGLU attribution, MoE,
later layers, generation, quality, RAM, or rate.
