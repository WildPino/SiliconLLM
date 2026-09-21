# STRAT-01 GigaChat 3.1 attention-stage diagnostic

**State:** FROZEN PROTOCOL; NOT YET EXECUTED
**Date frozen:** 21 September 2026
**Scope:** one pinned-reference block-0 prefill trace plus offline QK/softmax/value-stage attribution; no production-engine change, quality, RAM, or speed claim

## Question and source basis

The direct pre-V-B capture closed as
`ATTRIBUTED_RESIDUAL_TO_ATTENTION_RECONSTRUCTION`: the prior reconstructed
latent differs from pinned `kqv-0` at NRMSE `2.159975473e-4`, while true-latent
V-B and final layout pass.  Pinned llama.cpp commit
`5b335f413e4f73b0809c4fe39af894efbcc6a0d2` exposes two existing block-0
callbacks inside the non-flash attention branch:

1. `kq` immediately after `ggml_mul_mat(k, q)`, before scale/mask/softmax;
2. `kq_soft_max` immediately after `ggml_soft_max_ext` with the frozen
   DeepSeek2 KQ scale and causal mask.

Capturing those nodes separates QK dot accumulation from softmax and the
subsequent softmax×V reduction.

## Frozen identity and execution

- Exact GGUF: 6,474,702,976 bytes, SHA-256
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.
- Clean pinned llama.cpp revision
  `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`; graph-source SHA-256
  `000c88afa5ebd4f1d20821054ef9dc4d10a0c9cea4f440d179eaea8723ad8dee`.
- Same fixed eight tokens/positions and CPU context as Rung 2A: one thread,
  F16 K/V cache, flash attention off, no KQV or op offload.
- Execute one new pinned `prefill8` trace only.  The C donor engine does not
  execute.  Preserve it in a distinct immutable raw directory.
- Reuse the immutable `Qcur-0`, `Kcur-0`, `Vcur-0` and captured true `kqv-0`
  identities; refuse any mismatch.

## Frozen mappings and diagnostic arms

Source shape propagation freezes `kq-0` and `kq_soft_max-0` as `[8,8,32]`
(`slot,query-token,head`).  Canonical bytes reshape as
`[head,query-token,slot]` and transpose once to
`[query-token,head,slot]`.  No post-result axis search is allowed.

Using exact F16 cache roundtrips and the frozen scale
`(1 + 0.1*log(64))^2 / sqrt(192)`:

- compute scalar project raw QK dots from immutable Q/K and compare with
  captured `kq-0` at NRMSE `2e-6`, normalized maximum `1e-5`;
- apply stable causal scaled softmax to captured `kq-0` and compare with
  captured `kq_soft_max-0` at the same gate;
- reduce captured softmax against the latent 512 components of the exact F16
  cache and compare with captured/mapped true `kqv-0` at the same gate;
- also preserve the fully project-composed path to establish whether errors
  compound, without using it to override the three independent stage gates.

Required negative controls: one Q value mutation must break the raw-QK gate;
one legal-past softmax probability swap must break the latent gate; and using
unrounded F32 K/V instead of the F16 cache must break at least one applicable
captured gate.  Any negative that does not fire voids the cell.

## Adjudication

| Label | Frozen rule |
|---|---|
| `ATTRIBUTED_ATTENTION_RESIDUAL_TO_QK_DOT` | Controls valid; project raw QK fails; captured-QK softmax and captured-softmax value reduction pass. |
| `ATTRIBUTED_ATTENTION_RESIDUAL_TO_SOFTMAX` | Controls valid; raw QK passes; captured-QK project softmax fails; captured-softmax value reduction passes. |
| `ATTRIBUTED_ATTENTION_RESIDUAL_TO_VALUE_REDUCTION` | Controls valid; raw QK and captured-QK softmax pass; captured-softmax value reduction fails. |
| `ATTENTION_STAGE_DIAGNOSTIC_PASS` | All three independent stages pass; prior reconstruction mismatch came from composition/apparatus rather than an isolated arithmetic stage. |
| `MIXED_ATTENTION_STAGE_RESIDUAL` | Controls valid but multiple independent stages fail or the pass/fail pattern does not match a unique rule. |
| `VOID_ATTENTION_STAGE_DIAGNOSTIC` | Any identity, build, occurrence, shape, finiteness, mapping, oracle, negative-control, or completeness requirement fails. |

Run one non-VOID cell and stop.  Do not tune accumulation order, exponential,
mask constants, thresholds, or mappings after observing results.

## Eight-item control card

1. **Nearest prior cell:** [pre-V-B latent result](STRAT_01_GIGACHAT31_ENGINE_PRE_VB_LATENT_CAPTURE_RESULT_20260921.md).
2. **Changed coordinate:** expose existing `kq` and `kq_soft_max` callbacks and gate three attention stages independently.
3. **Why prior is insufficient:** it localizes before V-B but combines QK dot, softmax and value reduction.
4. **Identity/controls:** exact donor/revision/config/tokens, immutable Q/K/V/true latent, frozen mappings and three planted negatives.
5. **Separate gates:** raw QK, softmax and value reduction only; no production repair, downstream operator or speed gate.
6. **Claim labels:** callback locations/shapes are source-derived; numerical attribution remains proposed.
7. **Void/stop:** one non-VOID prefill cell; no post-result arithmetic or mapping search.
8. **Raw/canonical record:** planned raw directory `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_attention_stage_diagnostic_20260921/`; future result beside this protocol.

## Consequence boundary

Only a uniquely attributed stage authorizes a separately tested arithmetic
repair.  Do not repeat prior Rung-2A/Q8_K/V-B cells, integrate multiple new
coordinates at once, advance to Rung 2B, or move any speed claim.
