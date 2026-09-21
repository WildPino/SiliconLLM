# STRAT-01 GigaChat 3.1 F16 vector-dot diagnostic

**State:** FROZEN PROTOCOL; NOT YET EXECUTED
**Date frozen:** 21 September 2026
**Scope:** zero-donor offline test of the shared QK and value-reduction dot semantics; no production-engine change, quality, RAM, or speed claim

## Question and source basis

The closed attention-stage diagnostic finds two independent failures around a
passing softmax: scalar QK NRMSE `2.12556e-4` and scalar softmax×V NRMSE
`1.52773e-4`.  Pinned llama.cpp commit
`5b335f413e4f73b0809c4fe39af894efbcc6a0d2` defines, for `GGML_TYPE_F16`,
`from_float=ggml_cpu_fp32_to_fp16`, `vec_dot=ggml_vec_dot_f16`, and
`vec_dot_type=GGML_TYPE_F16` in `ggml-cpu.c`.  Therefore both graph matmuls
convert their F32 right operand to F16 before an F16×F16 vector dot.  The
project reconstruction currently retains that operand as F32.

The experiment asks whether this one source-derived semantic closes both
captured failures.

## Immutable inputs

Reuse only hash-bound payloads from the accepted attention-stage trace:

- `Qcur-0` SHA-256
  `4aca21f044acf71404ef0a7a000ed7b1bfe894efa82c5c1e49f7a5764cc6314b`;
- `Kcur-0` SHA-256
  `2860d9791b620d19788b8112e5366424be21669153eb1ec3255fd5a6c1b167f3`;
- mapped occupied `kq-0` SHA-256
  `121c689214a7bcf2e709b8de896df14aabcb8fb00580bab297231b8887a3b009`;
- full padded `kq_soft_max-0` source SHA-256
  `8def8f8d6969dab39987e915084a74cb0c766c9d842db8b272886c248d60092e`;
- mapped true `kqv-0` SHA-256
  `541183b4eb5992cfcfc711fe3c9f179ab41685106ba9cfe2aa7099f45f821312`.

No model, reference graph, or C donor engine may execute.

## Arms and gates

For QK, round both the cached key and F32 query to exact GGML F16, then test:

- scalar F16×F16 accumulation, to separate conversion from reduction order;
- independent pinned `ggml_vec_dot_f16` over length 576.

For value reduction, preserve all 256 padded slots, convert captured softmax
probabilities and the latent F16 cache column to F16, and test:

- scalar F16×F16 accumulation over length 256;
- independent pinned `ggml_vec_dot_f16` over length 256.

Compare QK outputs with mapped captured `kq-0` and value outputs with mapped
true `kqv-0`, each at NRMSE `2e-6`, normalized maximum `1e-5`.  Preserve the
old F16×F32 scalar controls.  Require byte-exact F16 conversion between the
project converter and pinned GGML on all tested queries/probabilities.

Negative controls: flip one converted F16 query mantissa bit and one converted
softmax-probability mantissa bit; each must break its target gate.  Refuse any
identity, shape, padding, finiteness, conversion, output-completeness, or
negative-control failure.

## Adjudication

| Label | Frozen rule |
|---|---|
| `ATTRIBUTED_BOTH_ATTENTION_DOTS_TO_F16_VEC_DOT` | Pinned F16 vec-dot passes both QK and value targets; both old F16×F32 controls fail. |
| `ATTRIBUTED_QK_TO_F16_VEC_DOT_ONLY` | Pinned F16 vec-dot passes QK only; value remains out of gate. |
| `ATTRIBUTED_VALUE_REDUCTION_TO_F16_VEC_DOT_ONLY` | Pinned F16 vec-dot passes value only; QK remains out of gate. |
| `F16_CONVERSION_ONLY_SUFFICIENT` | Scalar F16×F16 passes both targets, so vector reduction order is not required at the frozen gate. |
| `PARTIAL_F16_VEC_DOT_ATTRIBUTION` | At least one F16 arm improves materially but no unique pass rule above applies. |
| `F16_VEC_DOT_HYPOTHESIS_REJECTED` | Controls valid and neither F16 arm materially improves the applicable stages. |
| `VOID_F16_VEC_DOT_DIAGNOSTIC` | Any required control or completeness condition fails. |

Run one non-VOID offline cell and stop.  Do not tune SIMD width, accumulation
order, padding extent, conversions or thresholds after observation.

## Consequence boundary

Only a two-stage passing result authorizes a shared F16-dot implementation and
a separately frozen full block-0 confirmation.  A one-stage or partial result
requires a narrower diagnostic.  No Rung 2B, quality, RAM or speed claim moves.

## Precondition VOID A — project F16 conversion mismatch

The first offline launch at commit `77b24f3` is preserved as
`VOID_F16_VEC_DOT_DIAGNOSTIC`, adjudication SHA-256
`8a94fa1230b92184ca4ef1fcee35c3178f554cac15317af4e86d77a66d64c2c9`.
It executed no donor and stopped before emitting dot products because the
project binary16 converter disagreed with pinned GGML on at least one finite
immutable activation.  This is a failed frozen control, not evidence for or
against the vec-dot hypothesis.  Do not weaken the control or use partial
products.  Complete the separately frozen converter audit/repair before a
new F16 vec-dot launch.

## Apparatus VOID B — non-causal vector negative controls

The first post-converter launch at commit `4f2fa6d` is preserved as
`VOID_F16_VEC_DOT_DIAGNOSTIC`, despite the runner-emitted label
`F16_CONVERSION_ONLY_SUFFICIENT`.  Adjudication SHA-256
`72e6109eba8cedcc10d5b9219213687702a73c81612418dd196c7cb10afa227f`.
Both direct vector-dot streams and their mutations were identically all zero.
The runner incorrectly called each mutation control active merely because the
mutated stream failed the target gate; it did not require mutation to change
the corresponding baseline.  Therefore the frozen negative-control and
output-validity requirements were not met.

Repair only the apparatus as follows: call the pinned F16 dot through the
exported `GGML_TYPE_F16` CPU type trait after CPU initialization; add a fixed
model-free nonzero dot check; compute the already-frozen `0x0100` mantissa-bit
mutations through the scalar F16 arm as an independent causal control; and
require every mutation stream to differ bytewise from its baseline as well as
fail its target gate.  No input, conversion, mutation bit, threshold, arm,
label, or scientific hypothesis changes.  Use a new raw directory, zero donor
executions, and stop after one non-VOID repair launch.
