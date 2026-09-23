# STRAT-01 FFN SwiGLU cross-input apparatus repair 1

**Status:** `APPARATUS_READY_NO_DONOR_EXECUTION`

This is the fresh apparatus qualification required by
[VOID 1](STRAT_01_GIGACHAT31_ENGINE_FFN_SWIGLU_CROSS_INPUT_DIAGNOSTIC_VOID1_20260923.md).
The estimand, five arms, controls, thresholds, ordering, external adjudicator,
and frozen decision rule are unchanged.

The only C change removes the foreign assertion that required the
captured-reference/current-Q6 terminal sum to be byte-identical to a terminal
sum made from captured reference `ffn_out-0`. The correct cross-run replay is
already enforced externally against the preceding
`reference_swiglu_current_q6` checkpoints. A source regression now refuses
reintroduction of the wrong assertion while preserving all C/C identity
checks.

The canonical apparatus-only directory is:

`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_ffn_swiglu_cross_input_apparatus_repair1_20260923/`

Its adjudication and run-manifest SHA-256 values are
`b9afce85a39b5aa4c7b5dfdc2657b34384f239b34f3faab05c5896eb12a59c08`
and `931a4be95c4deb493cbf316e36d4bc3cba550cfb90ded1faaf0dc89fe62627cc`.

## Qualification evidence

- New SwiGLU C self-test: PASS, 7 checks.
- FFN-down and terminal-component C self-tests: PASS, 7 + 9 checks.
- Layer-1-start and Rung-2A/Rung-2B/Rung-2C model-free regressions: PASS,
  8 + 28 + 6 + 14 checks.
- New Python tests: PASS, 6 tests, including the repair regression.
- Terminal-component, layer-1-start, and Rung-2C Python regressions: PASS,
  14 tests.
- Legacy kernel self-test: PASS, 73,024 checks; worst error zero.
- Clang C11 `-O3 -mavx2 -mfma`, no fast-math: PASS with empty compiler
  stderr.
- Donor graph executions: 0.
- Reference graph executions: 0.

Exactly one repaired non-VOID invocation is authorized after this repair is
committed. This apparatus result supplies no gate/up/operator verdict and no
claim about quality, generation, RAM, or rate.
