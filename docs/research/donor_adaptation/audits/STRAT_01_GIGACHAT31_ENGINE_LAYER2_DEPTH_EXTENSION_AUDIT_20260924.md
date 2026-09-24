# STRAT-01 layer-2 depth-extension audit

**Disposition:** `LAYER2_IS_THE_NEXT_NON_DUPLICATE_DEPTH_BOUNDARY`

This is a static design audit. It executes no donor or reference graph and
makes no numerical, quality, RAM, or rate claim.

## Why this is the next boundary

The closed
[SwiGLU production-integration result](../probes/STRAT_01_GIGACHAT31_ENGINE_POST_F16_SWIGLU_PRODUCTION_INTEGRATION_RESULT_20260924.md)
establishes accepted-artifact fidelity through dense block 0 and complete
MoE layer 1. It does not execute or observe layer 2. Repeating any upstream
operator, schedule, cache, or cross-input cell would duplicate closed
evidence; moving directly to logits or generation would skip 24 unmeasured
transformer blocks.

The accepted GGUF metadata reports:

| field | value |
|---|---:|
| `deepseek2.block_count` | 26 |
| `deepseek2.leading_dense_block_count` | 1 |
| `deepseek2.embedding_length` | 1536 |
| `deepseek2.expert_count` | 64 |
| `deepseek2.expert_used_count` | 4 |
| `deepseek2.expert_feed_forward_length` | 1280 |

`blk.2` has the same sixteen attention/router/routed-expert/shared-expert
tensor types and shapes as `blk.1`: F32 norms/router/bias, Q4_K Q/KV/output
and up/gate matrices, Q5_0 K-B, and Q6_K down matrices. The artifact remains
bound to 6,474,702,976 bytes and SHA-256
`68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.

## Reuse versus hard-coding

The compute primitives in `strat01_gguf_rung2c.h` already accept tensor
pointers and are structurally reusable for another MoE layer:

- `strat01_r2c_build_attention_range`;
- `strat01_r2a_run_schedule`;
- `strat01_r2c_run_moe`;
- the accepted Q4, Q5, Q6, F16, RMSNorm, RoPE, routing and residual helpers.

The two-layer limit is orchestration and evidence plumbing, not a missing
operator. It is currently encoded in:

- literal `blk.1.*` tensor inventories;
- explicit layer-0/layer-1 arm allocation and call sequencing;
- logical checkpoint names ending in `-1`;
- two-layer cache-manifest loops and four cache-path slots;
- the pinned reference callback whitelist, rank table, logical selection,
  and cache contract listing only layers `[0,1]`;
- runner schemas expecting 32 checkpoints per arm and two cache layers.

The current helper accounting also predicts the exact additional work. One
more layer across both schedules adds `2304` QK and `262144` value-dot calls,
moving the totals from `4608` / `524288` to `6912` / `786432`.

## Consequence

The next cell should add a separate layer-2 evidence surface while leaving
the accepted Rung-2C path and layer-1 arithmetic unchanged. It must capture
an independent pinned reference for `l_out-1` plus the complete layer-2
boundary, validate three cache layers, and execute one C producer only after
model-free apparatus qualification and an exact-source commit.

This audit does not authorize generalized 26-layer execution. A layer-2 pass
would establish the first repeated-MoE depth step and inform whether a later
loop generalization is justified; a failure must be localized at its first
checkpoint before any operator edit.
