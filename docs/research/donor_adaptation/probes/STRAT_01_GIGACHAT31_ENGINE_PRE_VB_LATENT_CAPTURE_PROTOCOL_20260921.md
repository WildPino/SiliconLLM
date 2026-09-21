# STRAT-01 GigaChat 3.1 pre-V-B latent capture protocol

**State:** FROZEN PROTOCOL; NOT YET EXECUTED  
**Date frozen:** 21 September 2026  
**Scope:** one pinned-reference trace extension at block 0, followed by offline attribution; no production-engine change, quality, RAM, or speed claim

## Question and source basis

The closed `PARTIAL_VB_Q8K_ATTRIBUTION` cell improves `kqv_out-0` NRMSE from
`0.01199013510` to `0.00125348044`, but misses the tight target gate.  Its
attention latent was reconstructed rather than captured, so the residual is
ambiguous.

Pinned llama.cpp commit
`5b335f413e4f73b0809c4fe39af894efbcc6a0d2` already exposes the two required
nodes in `src/llama-graph.cpp` (file SHA-256
`000c88afa5ebd4f1d20821054ef9dc4d10a0c9cea4f440d179eaea8723ad8dee`):

- `cb(kqv, "kqv", il)` immediately after `ggml_mul_mat(v, kq)`;
- `cb(kqv, "kqv_mla", il)` immediately after `ggml_mul_mat(v_mla, kqv)`;
- the existing `kqv_out` callback follows permutation and 2-D contiguity.

The experiment asks whether the residual precedes V-B, occurs in V-B, or is
only a final-layout interpretation error.

## Frozen identity and execution

- Exact donor GGUF: 6,474,702,976 bytes, SHA-256
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.
- Pinned llama.cpp revision: `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`, clean worktree required.
- Reuse the Rung-2A fixed eight token IDs, positions and CPU configuration:
  one thread, F16 K/V cache, flash attention disabled, `offload_kqv=false`,
  `op_offload=false`, and the same resolved context/batch dimensions.
- Execute the reference producer once for the `prefill8` arm only.  This is one
  reference donor execution; the C engine does not execute.
- Extend a distinct trace apparatus/output directory.  Do not mutate or
  overwrite prior Rung-2A reference evidence.

## Required captures and checks

Capture full canonical F32LE payloads plus callback name, occurrence, op,
type and logical shape for block-0 `kqv`, `kqv_mla` and `kqv_out`.  The
apparatus must refuse missing or ambiguous occurrences and non-finite values.

Offline, on the same captured tensors:

1. compare the prior reconstructed latent (SHA-256
   `88990b6ba96f21a001fa591a0fe63365531f31672fbb43360467df3a2407465d`)
   with captured `kqv-0`, after a source-derived and manifest-recorded axis
   mapping;
2. apply both project and pinned Q4_K×Q8_K V-B operators to captured `kqv-0`;
3. compare both outputs with captured `kqv_mla-0` under the tight NRMSE
   `2e-6` / normalized maximum `1e-5` gate and require project-versus-pinned
   exact Q8_K bytes plus tight numerical agreement;
4. map captured `kqv_mla-0` through the source-derived final permute/contiguous
   layout and compare it with captured `kqv_out-0` at the same tight gate;
5. retain the prior scale-mutation, transpose and packed-scale negatives.

Any axis/layout mapping must be derived before examining numerical residuals
from callback metadata and the pinned source operations.  No post-result
permutation search is allowed.

## Adjudication

| Label | Frozen rule |
|---|---|
| `ATTRIBUTED_RESIDUAL_TO_ATTENTION_RECONSTRUCTION` | Controls valid; reconstructed latent fails the tight captured-`kqv` gate; project and pinned V-B on true `kqv` pass captured `kqv_mla`; final layout passes. |
| `ATTRIBUTED_RESIDUAL_TO_VB_GRAPH_SEMANTICS` | Controls valid; reconstructed latent passes captured `kqv`; project/pinned agree with each other but fail captured `kqv_mla`; final layout passes. |
| `ATTRIBUTED_RESIDUAL_TO_FINAL_LAYOUT` | Controls valid; latent and V-B gates pass, but the preregistered `kqv_mla`→`kqv_out` layout gate fails. |
| `PRE_VB_LATENT_CAPTURE_PASS` | All three tight comparisons pass; the previous residual came from using the reconstructed latent and/or target layout rather than an engine operator defect. |
| `MIXED_PRE_VB_RESIDUAL` | Controls valid but more than one stage fails, or project/pinned V-B fail to agree tightly. |
| `VOID_PRE_VB_LATENT_CAPTURE` | Identity, build, occurrence, shape, finiteness, completeness, oracle, negative-control, or frozen-mapping requirement fails. |

Run one non-VOID cell and stop.  Preserve any void in a distinct directory.
No result authorizes a production change unless its isolated operator stage
passes the corresponding true-intermediate oracle gate.

## Eight-item control card

1. **Nearest prior cell:** [partial `kqv_out-0` attribution](STRAT_01_GIGACHAT31_ENGINE_KQV_OUT_DIAGNOSTIC_RESULT_20260921.md).
2. **Changed coordinate:** capture existing pre/post-V-B graph nodes instead of reconstructing their shared boundary.
3. **Why prior is insufficient:** it has no direct reference latent and leaves a `0.00125348` residual.
4. **Identity/controls:** exact donor/revision/config/tokens, clean pinned source, callback occurrence metadata, two independent V-B implementations and planted negatives.
5. **Separate gates:** attention latent, V-B projection and final layout are adjudicated independently; no downstream FFN or speed gate.
6. **Claim labels:** source callback locations are source-derived; numerical attribution remains proposed until measured.
7. **Void/stop:** one non-VOID prefill cell; no permutation search or equation tuning after observation.
8. **Raw/canonical record:** planned raw directory `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_pre_vb_latent_capture_20260921/`; future result beside this protocol.

## Consequence boundary

Do not rerun Rung 2A, the Q/KV projection diagnostic, Q8_K repair,
changed-coordinate confirmation, or the reconstructed-latent V-B diagnostic.
This cell cannot advance Rung 2B, quality, RAM or accepted-token rate.  It
exists only to choose the next justified engine coordinate.
