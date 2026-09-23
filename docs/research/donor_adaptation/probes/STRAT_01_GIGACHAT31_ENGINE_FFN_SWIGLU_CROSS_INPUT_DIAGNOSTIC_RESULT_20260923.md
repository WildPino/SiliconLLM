# STRAT-01 FFN SwiGLU cross-input diagnostic result

**Verdict:** `GATE_AND_UP_RESIDUALS_INDEPENDENTLY_SUFFICIENT`

**Repair implementation commit:** `9ca42f1316a1583526e9c1822fef8c46d414e1a5`

The repaired invocation of the frozen
[protocol](STRAT_01_GIGACHAT31_ENGINE_FFN_SWIGLU_CROSS_INPUT_DIAGNOSTIC_PROTOCOL_20260923.md)
is valid. [VOID 1](STRAT_01_GIGACHAT31_ENGINE_FFN_SWIGLU_CROSS_INPUT_DIAGNOSTIC_VOID1_20260923.md)
remains apparatus history and contributes no scientific values.

The canonical result directory is:

`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_ffn_swiglu_cross_input_repair1_20260923/`

Its adjudication and run-manifest SHA-256 values are
`0f897c815c1f5d5dc3ede8f0aacca687df325f10a2d4b504be679f65b42922f5`
and `e55645b0247b01b14ea76f5341a85842979c8b213cdffd908e33e83275a1c28d`.
The run binds the accepted 6,474,702,976-byte GGUF, all seven immutable
payloads, and the committed engine/diagnostic sources. Donor and reference
graph executions are both zero.

## Measurements

All values below are against the same immutable reference targets. The gates
are NRMSE `<= 0.002` and normalized maximum `<= 0.01`.

| arm | SwiGLU NRMSE | `ffn_out-0` NRMSE | `kqv_out-1` NRMSE | first failure |
|---|---:|---:|---:|---|
| captured reference/current Q6 control | `0` | `8.34832e-8` | `0.000385600` | none |
| reference gate + reference up, current expression | `4.99848e-8` | `9.81351e-8` | `0.000385600` | none |
| **C gate + reference up** | `0.000207224` | `0.000722420` | **`0.005655289`** | `kqv_out-1` |
| **reference gate + C up** | `0.000226113` | `0.000851198` | **`0.006351015`** | `kqv_out-1` |
| C gate + C up replay | `0.000307537` | `0.000991352` | **`0.006136636`** | `kqv_out-1` |

Every arm passes the normalized-maximum gate at the three displayed
boundaries. Both one-C-input hybrids also pass the aggregate SwiGLU and direct
down-output NRMSE gates, then independently exceed the propagated layer-1
NRMSE gate. The verdict is therefore about downstream sufficiency, not a claim
that either local tensor is already outside its direct acceptance bound.

The current production expression on exact reference gate/up inputs is
numerically equivalent at SwiGLU and remains on the same all-pass downstream
trajectory. `SiLU(gate) * up` ordering and `expf` semantics are not the
residual source in this cell.

## Controls and replay

- Captured-reference/current-Q6 checkpoints exactly replay the preceding
  `reference_swiglu_current_q6` arm.
- C/C SwiGLU, Q6 output, terminal sum, all eleven checkpoints, and prior
  metrics replay exactly or within the frozen `1e-12` requirement.
- One-byte mutations of all seven payloads are refused.
- Operand-origin relabeling is rejected.
- Negated reference gate and swapped reference-up rows both fail direct and
  downstream gates by wide margins.
- Source, descriptor, containment, compiler, and zero-producer accounting
  checks pass.

## Interpretation and next boundary

Captured C gate and captured C up residuals are each independently sufficient
to recreate the layer-1 failure when paired with the other exact reference
operand. The nonlinear SwiGLU operator, Q6 down operator, block-0 terminal
addition, layer-1 builder, and layer-1 attention are already closed and must
not be changed or rerun.

This result does **not** decide whether each operand residual comes from the
shared `ffn_norm-0` input or from its Q4_K projection. The earlier Rung-2B
cross-input result tested a predecessor input lineage; before opening another
cell, the production-integration `ffn_norm-0` identities must be compared with
that evidence. Any successor must cross the current production C/reference
normalized inputs through both unchanged gate and up Q4_K operators in one
paired zero-producer diagnostic, unless identity audit proves that coordinate
is already measured.

No repair, MoE, later-layer, tokenizer, generation, quality, RAM, or rate claim
is made.
