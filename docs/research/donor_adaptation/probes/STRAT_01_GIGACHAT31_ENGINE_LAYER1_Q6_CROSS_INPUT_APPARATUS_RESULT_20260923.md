# STRAT-01 layer-1 Q6 cross-input apparatus result

**Verdict:** `APPARATUS_READY_NO_DONOR_EXECUTION`

The apparatus required by the frozen
[layer-1 Q6 cross-input protocol](STRAT_01_GIGACHAT31_ENGINE_LAYER1_Q6_CROSS_INPUT_PROTOCOL_20260923.md)
is qualified. It validates and reuses the immutable `prefill8` top-k,
reference/current SwiGLU, and reference/current down payloads. It executes no
donor or reference graph. This is an apparatus result, not an attribution of
the remaining layer-1 residual.

## What the apparatus proves

- The canonical propagation predecessor matches SHA-256
  `79a19806fcf80df3e4d8a84335eaaecfc7beb9aef543f021dd1da01f6558b9c3`,
  status `FAIL_ENGINE_REFERENCE_GENERIC_PROPAGATION`, and empty errors.
- All nine frozen payload identities, sizes, and finite-value requirements
  pass, including the exact top-4 expert IDs.
- The diagnostic implements separate routed and shared Q6_K paths, current
  replay arms, negated-input controls, and a mutated-expert-ID control while
  preserving zero graph accounting.
- The ordinary Clang C11 `-O3 -mavx2 -mfma` engine compiles without error.
- All **135** registered STRAT-01 Python tests pass.
- All **17** registered C self-tests pass, including the new diagnostic's
  **9 checks** and the legacy suite's **73,024 checks** with worst
  scalar-reference error `0`.
- `donor_graph_executions = 0` and `reference_graph_executions = 0`.

## Reproducibility

Observed frozen-protocol commit:
`9b4a8070bfe8d934844d4c85a12348e8b4962329`

Runner:
`benchmarks/donor_adaptation/engine/run_strat01_layer1_q6_cross_input.py`

Raw apparatus directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer1_q6_cross_input_apparatus_20260923/`

`adjudication.json` SHA-256:
`1a4cd1c6f27363f121e8d81ef0474c1e1ad73e30ae86e68c330d99a379e45186`

Compiled apparatus binary SHA-256:
`a654234b450fcdd0a5fb37838ea87e9239e03d02a63aa3e9c64078b71ce6227b`

## Scope and next action

The apparatus establishes that the direct cross-input coordinate is live and
fail-closed. It establishes no routed/shared Q6 verdict, graph parity,
production repair, later-layer coverage, quality, RAM, or rate. After the
implementation and this qualification are committed, the next and only
authorized action is one scientific diagnostic invocation on the accepted
GGUF. That invocation hashes and reads the model and exact Q6 tensors, but
runs zero donor and zero reference graphs. The raw apparatus directory remains
immutable and outside source control.
