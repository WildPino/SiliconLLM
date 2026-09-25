# STRAT-01 layer-1 router-weight normalization compile-parity protocol

**FROZEN BEFORE IMPLEMENTATION OR SCIENTIFIC EXECUTION:** 2026-09-25

**Cell:** `STRAT-01-ENGINE-LAYER1-ROUTER-WEIGHT-NORMALIZATION-COMPILE-PARITY`

## Question and non-duplication

The valid [production-integration result](STRAT_01_GIGACHAT31_ENGINE_LAYER1_NUMERICAL_PRIMITIVES_PRODUCTION_INTEGRATION_RESULT_20260925.md)
is exact through selected unbiased router weights. Its first failure is the
four-weight normalization, after which routed weighting and terminal output
diverge. All upstream router, expert, shared, Q6 and SwiGLU coordinates are
closed.

Question: does the pinned CPU `SUM_ROWS -> CLAMP -> DIV` arithmetic reproduce
immutable `ffn_moe_weights_norm-1` exactly from the captured selected weights?

This is not another router projection, selection, expert, Q6, graph or
production experiment. Do not recompute logits, sigmoid or top-4 and do not
propagate through expert outputs in this cell.

## Immutable bindings

- production-integration adjudication SHA-256
  `3cec456bc7b1d2798976cb9544cd91d36d45451b94911efa1f0ec1ac97437816`;
- production execution commit
  `25547e97fd2b70c3125675c898ff593c16c559d3`;
- pinned llama.cpp revision
  `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`;
- schedules `prefill8` and `cached7p1`, each containing 8 tokens × 4 ordered
  selected F32 weights and immutable normalized reference outputs;
- minimum normalization denominator `6.103515625e-5f`.

Candidate and reference operands must be byte-exact schedule twins before any
scientific arm runs.

## Frozen arithmetic arms

For each token:

1. replay the production F32 source-order sum, F32 clamp and four F32
   divisions; it must exactly reproduce captured production normalization;
2. candidate: convert each selected F32 weight individually to `ggml_float`
   (`double`), sum in source order, convert the sum once to F32, clamp in F32,
   then perform four F32 divisions;
3. independent oracle: compile the pinned `ggml_vec_sum_f32` semantics in a
   separate generic C++ translation unit and apply the same clamp/division;
4. compare candidate and oracle byte-for-byte with immutable reference.

The candidate helper must be non-inlined and isolated from AVX/FMA lowering.
Float sequential, float pairwise, unrounded-F64-denominator, one-value
mutation, slot permutation, wrong origin and truncated-file controls must
reject. Report exact changed counts plus NRMSE and normalized maximum for every
non-exact arm.

## Exploratory observation firewall

After the production FAIL but before this protocol, a read-only exploratory
calculation found the proposed double-sum candidate exact on the 32 preserved
values. That observation is explicitly non-evidentiary. The formal cell must
independently compile its candidate/oracle, validate source ownership and run
all frozen controls; it may not copy an exploratory output as its result.

## Apparatus, accounting and decision

Before reading captured payloads, qualify source independence, compiler flags,
synthetic exactness, output inventory, clamp boundary, mutation/permutation,
wrong-origin and truncation controls. Apparatus payload, diagnostic, producer,
donor-graph and reference-graph counts must all be zero.

The scientific cell may execute one local diagnostic against only preserved
payloads. It must not open the 6.47 GB model or execute any model graph.

- `LAYER1_ROUTER_WEIGHT_NORMALIZATION_EXACT` only if replay, candidate/oracle,
  candidate/reference and every control/accounting gate pass exactly;
- `LAYER1_ROUTER_WEIGHT_NORMALIZATION_INSUFFICIENT` if the cell is valid but
  candidate/reference is not exact;
- `VOID_LAYER1_ROUTER_WEIGHT_NORMALIZATION_COMPILE_PARITY` for any identity,
  source, replay, oracle, shape, finiteness, control or accounting failure.

A positive result may authorize a separately frozen production integration.
No later-layer, quality, generation, memory or rate claim follows.
