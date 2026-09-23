# STRAT-01 post-F16 layer-1-start cross-input protocol

**Frozen:** 2026-09-23, before implementation or execution

**Cell:** `STRAT-01-ENGINE-POST-F16-LAYER1-START-CROSS-INPUT`

**Purpose:** determine whether the remaining accepted block-0 terminal
residual is sufficient to produce the post-F16 layer-1 down failures, or
whether a local layer-1 residual remains when the exact reference start state
is supplied.

## Why this is new and not a repetition

The earlier layer-1-start cross-input used the pre-reference-generic,
pre-F16-reduction C start state and stopped at `kqv_out-1`. Since then two
causal coordinates changed:

1. reference-generic Q4 lowering made block-0 terminal output nearly exact;
2. exact pinned `GGML_CPU_GENERIC` F16 reductions reduced `kqv_out-1` NRMSE
   from `3.53432e-4` to `5.03513e-5`.

The valid
[F16 propagation result](STRAT_01_GIGACHAT31_ENGINE_F16_VECTOR_REDUCTION_PROPAGATION_RESULT_20260923.md)
still fails at routed/shared Q6 down, now concentrated at token 6. The old
cross-input cannot adjudicate this changed operator composition or the new
current start-state hash. This cell runs the complete current layer 1 from
two frozen inputs and executes no donor or reference graph.

## Frozen identities

- Accepted GGUF: 6,474,702,976 bytes, SHA-256
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.
- Pinned llama.cpp revision:
  `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`.
- F16 propagation adjudication SHA-256:
  `512e3ea7754c5e8fa067dab5692a8a192879be26895547cbedbb6023a896edcf`.
- Reference `l_out-0`, F32LE `[8,1536]`, 49,152 bytes:
  `385073c91f472dd9ffc1c86bcb63c6ed50256a6d5645e240d61ccdb613d814aa`.
- Post-F16 current `l_out-0`, F32LE `[8,1536]`, 49,152 bytes:
  `7fb5fe52df684e4af82413004f6845133528241e3b030530d41595f74298bdc4`.
- Pinned-reference prefill manifest SHA-256:
  `d7506adfd7cb20a54da2d406446c5c261452ecf0eecaa3663ef6a4a7acca1451`.

Every admitted manifest and payload must be validated by containment, byte
count, SHA-256, dtype, shape, and finiteness. A one-byte mutation of either
input or either manifest must be refused before model execution.

## Arms and changed estimand

Use only the prefill schedule because the post-F16 producer proves identical
checkpoint hashes between `prefill8` and `cached7p1`, with exact continuity.
Read the accepted GGUF once and execute these offline arms:

1. `reference_start`: exact reference `l_out-0` through the current layer-1
   RMSNorm, projections, exact F16 attention reductions, output projection,
   router, selected and shared FFNs, Q6 down paths, and terminal addition;
2. `current_start`: post-F16 current `l_out-0` through the same operations.

This command is a diagnostic helper invocation, not a graph producer. It must
reuse production functions and weights without modifying production
semantics. It may not run embeddings, block 0, a llama.cpp reference graph,
or either Rung-2C schedule.

## Required outputs and gates

Emit every canonical layer-1 checkpoint from `attn_norm-1` through
`l_out-1`, including exact I32 top-4 IDs and the routed/shared intermediate
tensors. Compare both arms with the immutable pinned-reference prefill
payloads under the existing Rung-2C gates:

- general F32: NRMSE `<= 0.002`, normalized maximum `<= 0.01`;
- `ffn_inp-1` and `l_out-1`: NRMSE `<= 0.001`, normalized maximum `<= 0.005`;
- top-4 IDs: exact equality.

The `current_start` arm must byte-replay every post-F16 prefill checkpoint,
including:

- `kqv_out-1`:
  `482e240b5723dfdccda2c8624674caff2848104c8a6bb6a66444bcf936198827`;
- `ffn_moe_down-1`:
  `447f1e4afaa8e2f08376696f542e66867e6c80d43d2bbe56037a16477e3553ee`;
- `ffn_shexp-1`:
  `26087ec234a97c4382d6fe00a4fc387abe94e6b9180ec71ef0016acee65ddab9`;
- `l_out-1`:
  `06c08094299c0fa9901f589f8a7504e18d71a76f0d15b64d7444087031323157`.

Report aggregate and per-token metrics for both arms, especially token 6,
exact route IDs and weights, exact-helper invocation counts, source hashes,
artifact identity, and donor/reference graph counts.

## Causal controls

- Negating token 6 of the reference start state must reject at least one
  downstream numerical gate.
- Swapping reference rows 0 and 7 must reject.
- The current arm must replay all registered hashes exactly; a near numerical
  match is insufficient.
- The reference and current labels must not be interchangeable.
- Exact F16 helper counts must be nonzero and mode must be
  `pinned-generic-f64`.
- Donor and reference graph executions must both remain zero.

## Verdicts and stop rule

- `POST_F16_BLOCK0_TERMINAL_RESIDUAL_SUFFICIENT` if the reference-start arm
  passes every checkpoint, the current-start arm byte-replays the registered
  post-F16 failures, and all controls pass.
- `POST_F16_LAYER1_LOCAL_RESIDUAL_AT_<CHECKPOINT>` if the reference-start arm
  first fails a frozen gate at the named checkpoint while the replay and all
  controls remain valid.
- `VOID_POST_F16_LAYER1_START_CROSS_INPUT` for any identity, replay, build,
  source, mutation, accounting, helper-reach, or causal-control failure.

A terminal-residual verdict closes the current layer-1 operators and requires
the next cell to move strictly upstream into a post-F16 block-0 decomposition.
A local-layer-1 verdict permits only a separately frozen diagnostic at the
named earliest boundary. No outcome authorizes another donor/reference graph,
tolerance relaxation, or changes to Q4, F16 conversion/reduction, SwiGLU,
Q6, routing, tokenizer, quality, RAM, or rate.

Planned raw directories:

- apparatus:
  `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_post_f16_layer1_start_cross_input_apparatus_20260923/`;
- scientific:
  `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_post_f16_layer1_start_cross_input_20260923/`.
