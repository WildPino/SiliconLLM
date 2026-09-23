# STRAT-01 block-0 terminal-component cross-input apparatus

**Status:** `APPARATUS_READY_NO_DONOR_EXECUTION`

The frozen
[terminal-component protocol](STRAT_01_GIGACHAT31_ENGINE_BLOCK0_TERMINAL_COMPONENT_CROSS_INPUT_DIAGNOSTIC_PROTOCOL_20260923.md)
is implemented without opening the accepted GGUF or executing a donor or
reference graph. The apparatus-only record is:

`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_block0_terminal_component_cross_input_apparatus_20260923/`

Its adjudication and run-manifest SHA-256 values are
`e2ea82bb09f3267443664d4fe7f0a2d7edb83d3434b0d71247bb5870dee48d22`
and `d96d4731f24f733b34a227f8d6e6b80298f4cb2fdf8d77f07f29aadc9236ed47`.
The record was built at protocol commit `ac55eb9` with a dirty tree containing
only the new apparatus implementation plus the pre-existing user-owned files
listed in its provenance.

## Qualified surface

The diagnostic-only engine command:

- hash-checks all six frozen C/reference component and replay payloads;
- constructs reference/reference, C/C, C-attention/reference-FFN, and
  reference-attention/C-FFN sums with scalar binary32 additions;
- refuses unless both homogeneous sums reconstruct their frozen `l_out-0`
  byte-for-byte;
- delegates every arm to the unchanged qualified layer-1 builder;
- emits the same eleven checkpoints through `kqv_out-1` for each arm;
- emits token-7-negation and row-swap causal controls;
- records exact addend origins, parsed layer-1 tensor descriptors, source
  hashes, and zero donor/reference graph executions;
- cannot self-certify a scientific verdict.

The external runner independently reconstructs the four sums, validates the
full report schema and containment, enforces byte-exact C/C replay, replays all
frozen C/reference metrics within `1e-12`, applies the unchanged
NRMSE/normalized-maximum gates, and implements the frozen decision order.

## Qualification evidence

- New C self-test: PASS, 9 checks.
- Shared layer-1-start C self-test: PASS, 8 checks.
- Rung-2A/Rung-2B/Rung-2C C self-tests: PASS, 28 + 6 + 14 checks.
- New Python tests: PASS, 5 tests.
- Layer-1-start, Q×KV cross-input, and Rung-2C Python regressions: PASS,
  16 tests.
- Legacy kernel self-test: PASS, 73,024 checks; worst error zero.
- Clang C11 `-O3 -mavx2 -mfma`, no fast-math: PASS with empty compiler
  stderr.
- Donor graph executions: 0.
- Reference graph executions: 0.

## Authorization and non-claims

This qualification authorizes exactly one non-VOID scientific invocation of
the frozen four-arm cell after the implementation is committed. It is not a
component-attribution result and makes no claim about a natural mixed FFN
forward pass, Rung-2C repair, later layers, generation, quality, RAM, or rate.
