# STRAT-01 layer-1 numerical-primitives compile-parity protocol

**FROZEN BEFORE IMPLEMENTATION OR EXECUTION:** 2026-09-25

**Cell:** `STRAT-01-ENGINE-LAYER1-NUMERICAL-PRIMITIVES-COMPILE-PARITY`

## Question and non-duplication

The valid [Q6 production-integration result](STRAT_01_GIGACHAT31_ENGINE_Q6K_Q8K_REFERENCE_GENERIC_PRODUCTION_INTEGRATION_RESULT_20260925.md)
is exact through `ffn_norm-1`. It exposes two independent local residuals on
byte-exact operands:

1. the 1536-wide F32 router dot products first differ at
   `ffn_moe_logits-1`, while top-4 IDs and all selected expert up/gate
   projections remain exact;
2. routed and shared scalar-libm SwiGLU outputs differ despite byte-exact
   gate/up operands.

Question: do the exact pinned CPU numerical primitives reproduce the immutable
reference payloads for each coordinate independently?

This is not another router-selection, routed/shared attribution, Q4, Q6,
block-0 SwiGLU, graph, or production-integration experiment. Do not propagate
either arm through Q6 or combine their outputs in this cell.

## Immutable bindings

- accepted GGUF: 6,474,702,976 bytes, SHA-256
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`;
- production-integration recovery adjudication SHA-256
  `cb2a2d9cbfc80d8eb08dfc32d0f850fae717cf9a76525a6f81a9f600269a3b93`;
- production execution commit
  `54b4c381a214eb7a14de8e0cb5e1e40953ccd8f5`;
- pinned llama.cpp revision
  `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`;
- schedules `prefill8` and `cached7p1`; corresponding candidate and immutable
  reference operands must be schedule twins before any arm runs.

## Arm A — F32 router reduction

### Pre-implementation compiler-path correction

Source/build inspection after freezing the question but before implementation
showed that the pinned reference `vec.cpp` command uses `-DGGML_CPU_GENERIC`
and no `-mavx`, `-mfma`, or `-msse3`. The compiler's default predefines include
SSE/SSE2 but not SSE3, AVX, or FMA. Consequently `GGML_SIMD` is not defined for
this translation unit and `ggml_vec_dot_f32` takes its scalar fallback:
`x[i]*y[i]` is rounded as F32, accumulated into `ggml_float` (`double`), then
converted once to F32. This correction supersedes the initially named
AVX2/FMA reduction; no implementation or diagnostic had run under that
incorrect assumption.

Read the exact captured `ffn_norm-1` inputs and only the accepted
`blk.1.ffn_gate_inp.weight` F32 matrix. For all 8 tokens × 64 rows:

- a production-scalar replay must reproduce captured C
  `ffn_moe_logits-1` byte-for-byte;
- a candidate must reproduce the pinned `GGML_CPU_GENERIC`
  `ggml_vec_dot_f32` scalar fallback exactly: F32 product, F64 accumulation,
  one final F32 conversion, in source order;
- the candidate must equal immutable reference `ffn_moe_logits-1`
  byte-for-byte;
- a float-accumulator control, a wrongly promoted F64-product control, and
  one-value input/weight mutations must reject.

The copied candidate implementation and an independently compiled pinned
operation must agree exactly. The helper must be non-inlined and targeted
`no-avx,no-avx2,no-fma`; product precision, accumulator precision, source
order, row ordering, tensor offset/type/shape, compiler predefines, reference
build command, and all source hashes must be explicit.

## Arm B — routed and shared SwiGLU semantics

Use only captured byte-exact gate/up operands:

- routed: `ffn_moe_gate-1` and `ffn_moe_up-1`, 40,960 values each;
- shared: `ffn_gate-1` and `ffn_up-1`, 10,240 values each.

For each population:

- scalar `expf` replay must reproduce the captured C SwiGLU byte-for-byte;
- the existing pinned four-lane no-FMA SSE2 primitive must reproduce the
  immutable reference SwiGLU byte-for-byte;
- scalar and pinned outputs must remain distinguishable on the real
  population;
- swapped gate/up, one-value mutation, wrong label/origin, and truncated-file
  controls must reject.

This arm reuses a proven primitive but tests new layer-1 routed/shared operand
populations. It does not reopen the closed block-0 semantics result.

## Apparatus, accounting, and gates

Before reading model or captured payloads, qualify synthetic exactness,
candidate/oracle independence, source ownership, target attributes, shape and
offset validation, mutation/truncation controls, and deterministic output.
Model, payload, diagnostic, producer, donor-graph, and reference-graph counts
must all be zero during apparatus qualification.

The scientific diagnostic may read the one accepted F32 router matrix and the
bound captured/reference payloads. It may execute exactly one local diagnostic
binary and no model graph. It must report separate verdicts for router, routed
SwiGLU, and shared SwiGLU; no aggregate success may hide a failed component.

## Decision and stop rule

- `LAYER1_ROUTER_AND_SWIGLU_EXACT_PRIMITIVES` only if all three component
  candidates are byte-exact and every replay/control/accounting gate passes;
- `LAYER1_ROUTER_EXACT_SWIGLU_INSUFFICIENT` if only router is exact;
- `LAYER1_SWIGLU_EXACT_ROUTER_INSUFFICIENT` if both SwiGLU populations are
  exact but router is not;
- `LAYER1_NUMERICAL_PRIMITIVES_INSUFFICIENT` if neither coordinate closes;
- `VOID_LAYER1_NUMERICAL_PRIMITIVES_COMPILE_PARITY` for any identity, source,
  shape, replay, mutation, truncation, oracle-independence, finiteness, or
  accounting failure.

Do not modify production in this cell. A positive result may authorize a new,
separately frozen production-integration cell installing the independently
qualified primitives. A negative result must localize the first arithmetic
boundary before any new graph. No tokenizer, generation, task quality, RAM,
throughput, or later-layer claim follows; no `SPEED_LEDGER.md` update is due.
