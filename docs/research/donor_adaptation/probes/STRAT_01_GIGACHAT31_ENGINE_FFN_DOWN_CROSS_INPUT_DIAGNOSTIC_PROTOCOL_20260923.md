# STRAT-01 GigaChat 3.1 FFN down-projection cross-input protocol

**Frozen:** 2026-09-23, before implementation or adjudication

**Cell:** `STRAT-01-ENGINE-FFN-DOWN-CROSS-INPUT-DIAGNOSTIC`

**Purpose:** determine whether the sufficient captured `ffn_out-0` residual
is already present in its `ffn_swiglu-0` input or is introduced by the current
Q6_K×Q8_K down-projection operator when started from exact reference input.

## Prior result and changed estimand

The zero-producer
[terminal-component result](STRAT_01_GIGACHAT31_ENGINE_BLOCK0_TERMINAL_COMPONENT_CROSS_INPUT_DIAGNOSTIC_RESULT_20260923.md)
is `FFN_RESIDUAL_SUFFICIENT`. C `ffn_inp-0` paired with exact reference
`ffn_out-0` passes the unchanged layer-1 path, while exact reference
`ffn_inp-0` paired with C `ffn_out-0` fails at `kqv_out-1` and nearly
reproduces C/C.

This cell moves one operator upstream. It feeds either captured C or reference
`ffn_swiglu-0` into the unchanged production Q6_K×Q8_K down projection for
`blk.0.ffn_down.weight`, adds exact reference `ffn_inp-0`, and propagates the
result through the unchanged qualified layer-1 builder. It does not execute
block-0 attention, normalization, up/gate, SwiGLU, a donor/reference graph, or
any MoE surface.

## Immutable inputs and provenance

Only `prefill8` payloads are admitted. A repeated 7+1 schedule would duplicate
the already established continuity evidence.

| payload | shape | bytes | SHA-256 |
|---|---:|---:|---|
| C `ffn_swiglu-0` | `[8960,8]` | 286,720 | `e4073a39ca0d6f904dab90b22f8fac9c9516c219b265611a102da8dd3597bc8d` |
| reference `ffn_swiglu-0` | `[8960,8]` | 286,720 | `de9245a2f6f60e8d7511d03fe981e6b55ef21a8634d5071dd36727a1a5950cef` |
| C `ffn_out-0` replay target | `[1536,8]` | 49,152 | `8433164833fe4daecd24a840bb4cd70c41d390a15655abe479ef058248a6fb4f` |
| reference `ffn_out-0` target | `[1536,8]` | 49,152 | `f5514fe59d64b517a685ec70004a75152a5f4d511417148f34e4d13b3134bef4` |
| reference `ffn_inp-0` | `[1536,8]` | 49,152 | `baa389195bff42ff92593d92650ed9defdd1b1382469e5dde3516dfa82b76ec1` |

The C source manifest, block-0 production adjudication, and run manifest are
pinned by
`67b1c187ec129e4be32a37fa7792d7f6499843c105e9c73e535f9d95824118a3`,
`58f661d9461f3dc17b2a52f830dcc9f36423621528941d475e23ee2aed8e0bf2`,
and `7b8de95a3766e4f76bde37f4038ece2986b9067ad9ac7a0e4cc11855a3062497`.
The pinned reference manifest is
`7873d9e59b3a089232dd6b15b0ba2fcfc4d37df2fb1f32d0488d6914d43882a9`.
The preceding terminal-component adjudication and run manifest are
`edbb82a814a92be9f822d37f6683375dfa437b5b0774cbae13b86632d2a9a6f5`
and `4f46f0781e823b9d2fe2f77b01e882ce6f420bf1c9b3f6ce0434f79d85ce8514`.

The accepted GGUF is 6,474,702,976 bytes with SHA-256
`68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.
Only `blk.0.ffn_down.weight` with frozen type Q6_K and shape `[8960,1536]`,
plus the seven frozen layer-1 attention tensors, may be read. All descriptors,
offsets, file offsets, and spans must be recorded and fail closed.

## Frozen arms

1. **`REFERENCE_CAPTURED_CONTROL`:** captured reference `ffn_out-0` plus
   reference `ffn_inp-0`, propagated through layer 1. It must reproduce the
   prior all-pass reference/reference arm.
2. **`REFERENCE_SWIGLU_CURRENT_Q6`:** captured reference `ffn_swiglu-0`
   through the current production Q6_K×Q8_K operator, plus reference
   `ffn_inp-0`, then layer 1. This is the operator-sufficiency arm.
3. **`C_SWIGLU_CURRENT_Q6`:** captured C `ffn_swiglu-0` through the same
   operator, plus reference `ffn_inp-0`, then layer 1. Its generated
   `ffn_out-0` must byte-reconstruct the frozen C output and its composed
   `l_out-0` must equal terminal-component SHA-256
   `b822e425fb6c88faa62cc59dec8011a7869ebb9d64c3731e54beb585f66cb541`.

The third arm is a replay control and must reproduce the preceding
reference-attention/C-FFN failure and its metrics within `1e-12`.

## Implementation contract

Add a diagnostic-only command that:

1. hashes and validates the accepted GGUF and all five immutable payloads;
2. invokes the unchanged `strat01_r2b_q6_matmul_batch` on each captured
   `ffn_swiglu-0` input;
3. emits both generated `ffn_out-0` payloads and compares the C-input output
   byte-for-byte with its frozen replay target;
4. forms the three terminal sums with scalar binary32 addition and validates
   both captured/replayed sum identities;
5. copies each sum into a fresh layer-1 arm and delegates to the unchanged
   `strat01_r2c_build_attention_range` and `strat01_r2a_run_schedule`;
6. emits the same eleven layer-1 checkpoints for every arm;
7. records zero donor and zero reference graph executions and never
   self-adjudicates.

No helper may fork Q8_K activation quantization, Q6_K dot/layout semantics,
normalization, RoPE, K-B, cache, attention, or V-B. Compile with Clang C11,
`-O3 -mavx2 -mfma`, no fast-math, and FP contraction disabled.

## Gates and controls

Compare both generated down outputs directly with frozen reference
`ffn_out-0`, and compare all eleven propagated checkpoints with their frozen
reference targets. Use the unchanged NRMSE `<= 0.002` and normalized maximum
`<= 0.01` limits everywhere; both must pass separately. Per-token metrics are
descriptive.

Required controls:

- the captured reference-output control passes all eleven layer-1 gates;
- the C-input Q6 output, terminal sum, all eleven propagated outputs, and
  frozen metrics replay exactly as specified above;
- one-byte mutations of all five payloads are refused before execution;
- any wrong weight type, shape, layer, offset, or span is refused;
- negating all 8×8,960 components of the reference SwiGLU input changes the
  generated output and fails at least one direct or downstream gate;
- swapping reference SwiGLU rows 0 and 7 fails at least one direct or
  downstream gate;
- arm/input-origin relabeling is detected by provenance validation;
- source inventory binds the shared Q6 and layer-1 implementations.

A model-free self-test must cover input decoding, Q6 helper delegation,
fresh-arm reset, three-arm routing, mutations, descriptor refusal, checkpoint
enumeration, and report paths.

## Decision rule

Apply in order:

1. **`VOID_FFN_DOWN_CROSS_INPUT`** for any identity, replay, control, source,
   or apparatus failure.
2. **`Q6_DOWN_OPERATOR_RESIDUAL_SUFFICIENT`** if
   `REFERENCE_SWIGLU_CURRENT_Q6` fails either the direct `ffn_out-0` gate or
   any propagated layer-1 gate. The current down operator is then inadequate
   even on exact captured input; diagnose its Q8_K activation bytes and Q6_K
   dot semantics without changing upstream FFN stages.
3. **`SWIGLU_INPUT_RESIDUAL_SUFFICIENT`** if the exact-input current-Q6 arm
   passes every direct/downstream gate while the valid C-input replay fails.
   Keep the Q6 operator closed and split the captured SwiGLU construction
   upstream.

An all-pass C replay is an apparatus contradiction and therefore VOID, not a
new verdict. Exactly one non-VOID invocation is allowed. Gates and ordering
must not change after values are observed.

## Non-claims and stop rule

This cell cannot repair or promote Rung 2C. It does not distinguish gate from
up projection or SiLU multiplication if the upstream-input branch wins, and
it does not adjudicate another Q6 implementation if the operator branch wins.
It makes no claim about MoE, later layers, tokenizer, logits, generation,
quality, RAM, or rate and cannot update `SPEED_LEDGER.md`. Document and index
the result before implementing a repair or a finer split.
