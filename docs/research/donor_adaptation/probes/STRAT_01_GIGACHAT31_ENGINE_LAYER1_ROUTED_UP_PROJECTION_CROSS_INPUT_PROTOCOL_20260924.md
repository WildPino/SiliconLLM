# STRAT-01 layer-1 routed-up projection cross-input protocol

**Status:** frozen before apparatus implementation or scientific execution.

## Question

The closed
[routed-SwiGLU component result](STRAT_01_GIGACHAT31_ENGINE_LAYER1_ROUTED_SWIGLU_COMPONENT_CROSS_INPUT_RESULT_20260924.md)
proves that the captured production routed-up operand alone is sufficient to
reproduce the complete downstream failure. Is that residual caused by the
production `ffn_norm-1` input, or does it remain when the selected current
Q4_K up matrices consume the exact reference normalized input?

This estimand was not measured by the prior direct checkpoint gate. That gate
compared only complete captured up tensors locally; this cell changes their
input origin and judges them under the newly established downstream amplifier.

## Immutable evidence

Accepted GGUF: `6474702976` bytes, SHA-256
`68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.

| payload | bytes | SHA-256 |
|---|---:|---|
| reference `ffn_norm-1` | `49152` | `3f4826dd184c9442fa34807c9649aeae88343c5894c7aac531e5b5a209806dde` |
| production C `ffn_norm-1` | `49152` | `6d2080dcb6eb22c6e9b8f38f8d2beef04cf56aec02161775b65f911dc595f8d7` |
| captured reference routed up | `163840` | `e14f712c83cfd96404dd832ac851d06b6ab81d882567af8cb6590cc858182c9b` |
| captured production C routed up | `163840` | `7c2f80efe17e22ac2ef5bbf3059d67dcb45ff593cb36e9f4b54c639817a51859` |
| captured reference routed gate | `163840` | `341d79877218695fe5d26ab66fb6fb8a25737107c5199e8177eca466deed1a23` |
| ordered top-4 IDs | `128` | `557f502a4cc9f24ec866d11e63b857b9c60e40cc0cef43bfeb95c6ee59648e95` |

Reference normalized weights, shared output, FFN residual, layer-2 query/key,
and downstream identities remain those already frozen. Bind the predecessor
adjudication SHA-256
`04b46ac959b096bf8d2df8f1735882e37afbafacdc43c1c4eadff7560093a705`.
The current matrix is exactly `blk.1.ffn_up_exps.weight`, Q4_K shape
`[1536,1280,64]`.

## Fixed arms

1. `captured_reference_up`: immutable reference up capture;
2. `captured_production_c_up`: immutable production C up capture;
3. `current_up_on_reference_norm`: selected current Q4_K up matrices applied
   to immutable reference `ffn_norm-1`;
4. `current_up_on_c_norm`: the same matrices applied to immutable production C
   `ffn_norm-1`;
5. `control_reference_norm_token7_negated`: current up projection after
   negating only token 7 of the reference normalized input.

Arm 4 must reproduce the captured production C up tensor byte-exactly or the
run is `VOID_APPARATUS_OR_PROVENANCE`. For every arm, combine the up payload
with exact captured reference gate through the no-FMA SSE2 primitive, then run
the same selected current Q6 down matrices, exact reference normalized route
weights, ordered sum, reference shared output and residual, and closed layer-2
downstream path.

## Gates and decisions

Use complete downstream limits NRMSE `<= 0.002` and normalized maximum
`<= 0.01`, globally and per token. Report direct up, SwiGLU, down, routed
output, and complete downstream metrics.

Preconditions: captured reference passes; captured C fails and byte-replays
the predecessor; computed C-input up is byte-identical to captured C up; the
planted control fails; schedule twins, hashes, mutation refusal, labels,
sources, tests, and counters pass.

- if `current_up_on_reference_norm` passes:
  `LAYER1_ROUTED_UPSTREAM_NORM_RESIDUAL_SUFFICIENT`;
- if it fails:
  `LAYER1_ROUTED_UP_PROJECTION_RESIDUAL_SUFFICIENT`;
- otherwise or on any precondition failure: `VOID_INCONSISTENT`.

## Execution and stop rule

Qualify model-free first and commit exact sources before one scientific
invocation. The run may execute the five fixed selected-up/Q6 arms but must
execute zero donor and zero reference graphs. No timing or rate claim is
permitted.

After a valid result, close the winning side. If the normalized-input arm wins,
split production of `ffn_norm-1` without reopening the up matrices. If the
projection arm wins, split Q4_K decode/accumulation from immutable matrix
identity without reopening RMSNorm. No repair is integrated without a
separately frozen production cell.
