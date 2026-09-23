# STRAT-01 FFN SwiGLU cross-input apparatus

**Status:** `APPARATUS_READY_NO_DONOR_EXECUTION`

The frozen
[FFN SwiGLU protocol](STRAT_01_GIGACHAT31_ENGINE_FFN_SWIGLU_CROSS_INPUT_DIAGNOSTIC_PROTOCOL_20260923.md)
is implemented without opening the accepted GGUF or executing a donor or
reference graph. The canonical apparatus-only directory is:

`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_ffn_swiglu_cross_input_apparatus_20260923/`

Its adjudication and run-manifest SHA-256 values are
`af840f35e1da683e1ef8d68e5fea8df49419aca3f941d9d88cd396c59af5435c`
and `f60daca2a84490e8a6fb263f563e5a8219b12b8f6b6c687871bf82def8192147`.

## Qualified surface

The diagnostic-only engine command:

- validates the seven frozen C/reference gate, up, SwiGLU, and residual
  payloads before use;
- admits only the block-0 Q6_K down tensor and the seven frozen layer-1
  attention tensors;
- emits a captured-reference control, the current SwiGLU expression on exact
  reference gate/up inputs, both one-input hybrids, and a C/C replay;
- requires the C/C current expression to byte-replay frozen C
  `ffn_swiglu-0`, then requires its Q6 output and terminal sum to replay the
  already measured C arm;
- delegates every generated SwiGLU payload to the unchanged qualified
  Q6_K×Q8_K down helper, exact reference residual addition, and unchanged
  layer-1 builder;
- emits negated-reference-gate and swapped-reference-up-row causal controls;
- records exact tensor descriptors, input/source hashes, and zero donor or
  reference graph executions without self-certifying a scientific verdict.

The external adjudicator independently binds the previous
`SWIGLU_INPUT_RESIDUAL_SUFFICIENT` record and the reference FFN output,
validates report containment and provenance, applies direct and propagated
gates, and requires byte/hash/metric replay of both inherited terminal arms.

## Qualification evidence

- New SwiGLU C self-test: PASS, 7 checks.
- FFN-down and terminal-component C self-tests: PASS, 7 + 9 checks.
- Layer-1-start and Rung-2A/Rung-2B/Rung-2C model-free regressions: PASS,
  8 + 28 + 6 + 14 checks.
- New Python tests: PASS, 5 tests.
- Terminal-component, layer-1-start, and Rung-2C Python regressions: PASS,
  14 tests.
- Legacy kernel self-test: PASS, 73,024 checks; worst error zero.
- Clang C11 `-O3 -mavx2 -mfma`, no fast-math: PASS with empty compiler
  stderr.
- Donor graph executions: 0.
- Reference graph executions: 0.

## Authorization and non-claims

This qualification authorizes exactly one non-VOID invocation of the frozen
cell after the implementation is committed. It does not establish whether the
residual is caused by current SwiGLU semantics, gate input, up input, or their
joint perturbation. It makes no claim about a repair, MoE, later layers,
generation, quality, RAM, or rate.
