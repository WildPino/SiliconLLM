# STRAT-01 layer-1 Q6 cross-input result

**Verdict:** `LAYER1_SWIGLU_RESIDUAL_SUFFICIENT`

The frozen
[layer-1 Q6 cross-input protocol](STRAT_01_GIGACHAT31_ENGINE_LAYER1_Q6_CROSS_INPUT_PROTOCOL_20260923.md)
closes validly. Exact reference routed and shared SwiGLU inputs pass through
the current layer-1 Q6_K down paths by roughly four orders of magnitude below
the frozen NRMSE gate. The current-input arms reproduce the two captured
failures byte-for-byte. The Q6 operators are therefore sufficient on the
actual layer-1 selected/shared matrices; the admitted upstream SwiGLU
residual is sufficient to cause the downstream failure.

## Direct result

| arm | NRMSE | normalized maximum | NRMSE gate | result |
|---|---:|---:|---:|---|
| reference routed SwiGLU → selected Q6 | `7.10952e-8` | `9.03365e-8` | `0.002` | PASS |
| current routed SwiGLU → selected Q6 | `0.00385395` | `0.00261083` | `0.002` | FAIL, byte-exact replay |
| reference shared SwiGLU → shared Q6 | `6.12734e-8` | `4.98279e-8` | `0.002` | PASS |
| current shared SwiGLU → shared Q6 | `0.00288634` | `0.00105641` | `0.002` | FAIL, byte-exact replay |

Every one of the 32 reference routed token/slot comparisons passes; every one
of the eight reference shared-token comparisons passes. The current-input
discrepancy is sharply localized: all four routed slots fail for tokens 4 and
5 while the other 24 routed slots pass, and only tokens 4 and 5 fail on the
shared branch. This token localization is descriptive evidence from the
frozen direct outputs, not a new gate.

## Validity and controls

- `errors = []`.
- Both current-input outputs replay the canonical propagation outputs
  byte-for-byte.
- Negated reference routed/shared inputs and the mutated expert-ID arm all
  reject their corresponding gates.
- One-byte mutations of all nine admitted payloads are refused.
- The frozen expert IDs, Q6 tensor descriptors, source hashes, compiler,
  output containment, finiteness, and deterministic Q8 controls pass.
- The scientific runner re-passed all 135 STRAT-01 Python tests and all 17 C
  self-tests.
- `donor_graph_executions = 0` and `reference_graph_executions = 0`.

## Reproducibility

Scientific source commit:
`13c4c7ed7d9b561df104759dca5e5b0bba0fe03e`

Raw result directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer1_q6_cross_input_20260923/`

`adjudication.json` SHA-256:
`db9b478e208c35460d4bd1ae54eb56cfb2a8106361b19f0b7f1f024699509148`

Compiled scientific binary SHA-256:
`d04f7a50883d6772f37dd2202d834ca16f0c470e19897e4d4c7cebbf886f624b`

## Consequence and stop rule

Do not rewrite or retest Q6_K, repeat this cross-input, or rerun either graph.
The layer-1 FFN chain is numerically sufficient when given the exact reference
SwiGLU input, and prior evidence already closes the Q4 and SwiGLU operator
semantics. The remaining error is inherited from the small non-exact layer-1
FFN input lineage and amplified at tokens 4 and 5 through both FFN branches.

Any successor must change a strictly earlier coordinate and explain why it is
not a duplicate of the closed layer-1 attention, Q4, SwiGLU, or Q6 cells. A
downstream repair is not authorized by this result. No tokenizer, later-layer,
full-logit, generation, quality, RAM, or rate claim follows.
