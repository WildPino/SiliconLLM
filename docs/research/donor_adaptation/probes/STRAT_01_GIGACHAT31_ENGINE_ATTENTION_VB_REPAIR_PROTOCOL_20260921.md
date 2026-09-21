# STRAT-01 GigaChat 3.1 attention and V-B production repair protocol

**State:** FROZEN PROTOCOL; NOT YET EXECUTED
**Date frozen:** 21 September 2026
**Scope:** integrate two independently closed operator semantics and confirm both rung-2A schedules through block-0 attention output; no Rung 2B, quality, RAM, or speed claim

## Source-derived changed coordinates

1. In `strat01_r2a_attend_one`, convert every composed query component to the
   repaired project binary16 format before QK multiplication.  After the
   existing F32 softmax normalization, convert every normalized probability to
   binary16 before multiplication by the first 512 components of the F16 key
   cache.  Accumulate products in F32, preserving loop order, scale, causal
   extent, padding semantics, and cache layout.
2. In `strat01_r2a_vb_batch`, replace dequantize-to-F32 Q4_K multiplication
   with the already accepted scalar Q4_K×Q8_K operator.  Quantize each frozen
   512-component latent activation by head and token to two Q8_K blocks, then
   stream the corresponding 192 Q4_K rows for that head.  Preserve tensor
   orientation and output layout.

The F16-dot result closes coordinate 1.  The true-latent V-B diagnostic and
accepted Q4_K×Q8_K repair close coordinate 2.  Combining them in one C
execution avoids re-reading the 10B artifact while the final full-block
adjudication still exposes either repair independently by tensor boundary.

## Required controls

- compile the committed C engine and pass rung-2A, legacy, Q4_K×Q8_K, F16
  edge, prefill/cached equivalence, and Python apparatus tests;
- validate the accepted GGUF identity and immutable pinned-reference outputs;
- execute the accepted artifact in the C engine exactly once; do not rerun the
  pinned donor reference;
- require both `prefill8` and `cached7p1` to pass every available rung-2A
  tensor/cache gate through `kqv_out-0` and `ffn_inp-0`;
- require token-7 prefill/cached continuity at the tight gate;
- report every tensor metric and the first remaining failure per arm.

Labels: `PASS_ENGINE_ATTENTION_VB_REPAIR`, `FAIL_ENGINE_ATTENTION_VB_REPAIR`,
or `VOID_ENGINE_ATTENTION_VB_REPAIR` for any identity, build, test,
completeness, or control failure.  One non-VOID execution closes the cell.

## Consequence boundary

PASS establishes the current block-0 attention path through its residual
output for both schedules and permits the separately frozen next operator.
FAIL localizes the next boundary and forbids broader integration.  Neither
label establishes full-model generation, task quality, RAM, or token rate.
