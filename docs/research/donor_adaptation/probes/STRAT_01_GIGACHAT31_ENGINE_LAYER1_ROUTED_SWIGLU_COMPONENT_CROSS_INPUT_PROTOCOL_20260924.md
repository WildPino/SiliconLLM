# STRAT-01 GigaChat 3.1 layer-1 routed-SwiGLU component cross-input protocol

**Status:** frozen before apparatus implementation or scientific execution.

## Question

The closed
[routed-Q6 propagation](STRAT_01_GIGACHAT31_ENGINE_LAYER1_ROUTED_Q6_RESIDUAL_PROPAGATION_RESULT_20260924.md)
proves that current Q6 on exact reference routed SwiGLU passes the newly
sensitive downstream gate exactly, while the captured production C down path
fails. Which component already present at `ffn_moe_swiglu-1` is sufficient:

1. current scalar SwiGLU expression semantics;
2. current selected-expert gate projections;
3. current selected-expert up projections;
4. only their composition?

The previously validated no-FMA four-lane SSE2 primitive is included both as
an expression-semantics control and as a candidate repair. This is not a
repeat of the block-0 cell: the operands are the 32 layer-1 routed expert
rows, and the estimand is the sensitive layer-2 downstream gate discovered
after that repair.

## Immutable evidence

Accepted GGUF: `6474702976` bytes, SHA-256
`68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.

All four routed operands contain `8 × 4 × 1280 = 40960` little-endian F32
values (`163840` bytes):

| operand | SHA-256 |
|---|---|
| reference gate | `341d79877218695fe5d26ab66fb6fb8a25737107c5199e8177eca466deed1a23` |
| reference up | `e14f712c83cfd96404dd832ac851d06b6ab81d882567af8cb6590cc858182c9b` |
| production C gate | `2261171cad3033adc659e1f70348d1c5ea9ac8d1adb57f40aba679f853db1293` |
| production C up | `7c2f80efe17e22ac2ef5bbf3059d67dcb45ff593cb36e9f4b54c639817a51859` |
| captured reference SwiGLU | `5097dc8599d0635477ad83f63f7e2de21b659775abc345dc933ce1e173be29e8` |
| captured production C SwiGLU | `68fbf31260eacf458be0032e63fffa4f5ea33280ecdb81a53c88a815b2326a8c` |

The ordered top-4 expert IDs are `128` bytes, SHA-256
`557f502a4cc9f24ec866d11e63b857b9c60e40cc0cef43bfeb95c6ee59648e95`.
Reference normalized weights, shared output, FFN residual, layer-2 query/key,
and all downstream tensor identities remain exactly those frozen by the
routed-MoE and routed-Q6 protocols. The predecessor adjudication is bound by
SHA-256 `2eee409b8f007b9e53136e7a3d14d250af93a4d951b1ce859d4080fcfe6a9bc2`.

## Fixed arms

For every non-control arm, run the same selected current Q6 expert-down
matrices, exact reference normalized weights, ordered four-slot reduction,
exact reference shared output and `ffn_inp-1`, then the already closed layer-2
downstream path.

| arm | routed SwiGLU payload |
|---|---|
| `captured_reference_swiglu` | immutable reference capture |
| `captured_production_c_swiglu` | immutable production C capture |
| `scalar_reference_gate_reference_up` | `silu_scalar(ref_gate) * ref_up` |
| `scalar_c_gate_c_up` | `silu_scalar(c_gate) * c_up` |
| `sse2_reference_gate_reference_up` | no-FMA SSE2 primitive on both reference operands |
| `sse2_c_gate_reference_up` | only gate operand from C |
| `sse2_reference_gate_c_up` | only up operand from C |
| `sse2_c_gate_c_up` | no-FMA SSE2 primitive on both C operands |
| `control_reference_token7_negated` | captured reference SwiGLU with token 7 negated |

The two scalar arms must replay their corresponding captures byte-exactly.
Failure of either replay is `VOID_APPARATUS_OR_PROVENANCE`, not evidence about
the model.

## Gates and adjudication

Use the already frozen complete downstream limits: NRMSE `<= 0.002` and
normalized maximum `<= 0.01`, globally and per token. Report direct SwiGLU,
expert-down, routed-output, and final downstream metrics for every arm.

Preconditions:

1. `captured_reference_swiglu` passes;
2. `captured_production_c_swiglu` fails and byte-replays the closed
   predecessor;
3. both scalar replay identities hold;
4. the planted control fails;
5. schedule twins, frozen hashes, mutation refusal, arm labels, source hashes,
   self-tests, and execution counters all pass.

Canonical causal decision, in order:

- if `scalar_reference_gate_reference_up` fails while
  `sse2_reference_gate_reference_up` passes:
  `LAYER1_ROUTED_SWIGLU_EXPRESSION_RESIDUAL_SUFFICIENT`;
- otherwise, under a passing `sse2_reference_gate_reference_up`, a failing
  `sse2_c_gate_reference_up` alone establishes
  `LAYER1_ROUTED_GATE_INPUT_RESIDUAL_SUFFICIENT`;
- a failing `sse2_reference_gate_c_up` alone establishes
  `LAYER1_ROUTED_UP_INPUT_RESIDUAL_SUFFICIENT`;
- if both one-C-operand arms fail, report
  `LAYER1_ROUTED_GATE_AND_UP_INPUT_RESIDUALS_SUFFICIENT`;
- if both one-C-operand arms pass but `sse2_c_gate_c_up` fails, report
  `LAYER1_ROUTED_GATE_UP_COMPOSITION_RESIDUAL_SUFFICIENT`;
- if no arm accounts for the captured failure, or a precondition fails, report
  `VOID_INCONSISTENT` with no repair claim.

Independently report `SSE2_FULL_C_REPAIR_PASSES` only if
`sse2_c_gate_c_up` passes every complete downstream gate. This secondary flag
does not override the causal verdict.

## Execution and stop rule

Qualify a model-free apparatus first. Commit its exact sources before one
scientific invocation. The scientific run may open the accepted artifact and
execute the fixed Q6 arms, but must execute zero donor and zero reference
graphs. No timing or rate claim is permitted.

After a valid result, close every passing or sufficient component named by the
matrix. Do not rerun prior Q6, routed reduction, shared, terminal, layer-2,
block-0 semantics, or graph cells. A repair may be integrated only if the
candidate flag passes and a separately frozen production-integration cell
preserves all earlier checkpoints.
