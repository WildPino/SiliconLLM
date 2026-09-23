# STRAT-01 GigaChat 3.1 FFN SwiGLU cross-input protocol

**Frozen:** 2026-09-23, before implementation or adjudication

**Cell:** `STRAT-01-ENGINE-FFN-SWIGLU-CROSS-INPUT-DIAGNOSTIC`

**Purpose:** determine whether the sufficient captured `ffn_swiglu-0`
residual is attributable to the gate input, the up input, current
`SiLU(gate) * up` semantics, independent gate/up effects, or only their joint
interaction.

## Prior result and changed estimand

The zero-producer
[FFN-down result](STRAT_01_GIGACHAT31_ENGINE_FFN_DOWN_CROSS_INPUT_DIAGNOSTIC_RESULT_20260923.md)
is `SWIGLU_INPUT_RESIDUAL_SUFFICIENT`. Exact captured reference SwiGLU through
the unchanged production Q6_K×Q8_K down path passes direct output at NRMSE
`8.35e-8` and all eleven downstream layer-1 gates. Captured C SwiGLU
byte-replays the preceding failure.

This cell moves one construction upstream. It crosses captured C/reference
`ffn_gate-0` and `ffn_up-0`, computes SwiGLU with the unchanged production
scalar expression, and propagates every generated payload through the now
closed Q6 down helper, exact reference `ffn_inp-0`, and the closed layer-1
builder. It executes no block-0 attention, RMSNorm, gate/up projection, donor
or reference graph, or MoE surface.

## Immutable inputs and provenance

Only `prefill8` payloads are admitted.

| payload | shape | bytes | SHA-256 |
|---|---:|---:|---|
| C `ffn_gate-0` | `[8960,8]` | 286,720 | `bb52399fa69bc0f9295c0b2b2fc9588ec7689e440b7306f6df2e43f53c3710fa` |
| reference `ffn_gate-0` | `[8960,8]` | 286,720 | `5c30c0ada2e96ce43a92b22057b4d593b0b9ca342feade7092bfec084e8c6c2a` |
| C `ffn_up-0` | `[8960,8]` | 286,720 | `34a98ab5d44a7c1282901db0be2e152ea1a37f0cdf88366e65acd0597dbc390f` |
| reference `ffn_up-0` | `[8960,8]` | 286,720 | `2b608af95db90fcde83c29946ba2ac680fb6465cbaaf518fc12874ab510b29e4` |
| C `ffn_swiglu-0` replay target | `[8960,8]` | 286,720 | `e4073a39ca0d6f904dab90b22f8fac9c9516c219b265611a102da8dd3597bc8d` |
| reference `ffn_swiglu-0` target | `[8960,8]` | 286,720 | `de9245a2f6f60e8d7511d03fe981e6b55ef21a8634d5071dd36727a1a5950cef` |
| reference `ffn_inp-0` | `[1536,8]` | 49,152 | `baa389195bff42ff92593d92650ed9defdd1b1382469e5dde3516dfa82b76ec1` |

The block-0 production C manifest, adjudication, and run manifest are pinned
by `67b1c187ec129e4be32a37fa7792d7f6499843c105e9c73e535f9d95824118a3`,
`58f661d9461f3dc17b2a52f830dcc9f36423621528941d475e23ee2aed8e0bf2`,
and `7b8de95a3766e4f76bde37f4038ece2986b9067ad9ac7a0e4cc11855a3062497`.
The pinned reference manifest is
`7873d9e59b3a089232dd6b15b0ba2fcfc4d37df2fb1f32d0488d6914d43882a9`.
The preceding FFN-down adjudication and run manifest are
`6d3ab2b6a1097502060380350238940d434d38b2b41bcdff8d7c586f5f8d05d0`
and `fbe5857f338acbd10e8ffa703746e02a2c084533ba9148fc2a5cfd4e0d02645f`.

The accepted GGUF is 6,474,702,976 bytes with SHA-256
`68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.
Only block-0 `ffn_down.weight` with type Q6_K and shape `[8960,1536]`, plus
the seven frozen layer-1 attention tensors, may be read. All descriptors,
offsets, file offsets, and spans must be recorded and fail closed.

## Frozen arms

1. **`REFERENCE_CAPTURED_CONTROL`:** captured reference SwiGLU, then current
   Q6, exact reference residual, and layer 1. It must replay the preceding
   exact-input all-pass arm.
2. **`REFERENCE_GATE_REFERENCE_UP`:** current SwiGLU expression on exact
   captured reference gate/up. This tests the local operator semantics.
3. **`C_GATE_REFERENCE_UP`:** only the captured gate input is C.
4. **`REFERENCE_GATE_C_UP`:** only the captured up input is C.
5. **`C_GATE_C_UP`:** both inputs are C. Its generated SwiGLU must
   byte-reconstruct the frozen C target and replay the complete preceding
   C-SwiGLU failure within `1e-12`.

Every generated arm continues through the same current Q6 operator, exact
reference `ffn_inp-0`, and unchanged layer-1 path. Hybrid labels describe
captured operand origins, not natural recomputed projection pairs.

## Implementation contract

Add a diagnostic-only command that:

1. hashes and validates the accepted GGUF and all seven immutable payloads;
2. computes four generated SwiGLU arrays using exactly the production scalar
   expression `(gate / (1.0f + expf(-gate))) * up`;
3. emits every generated SwiGLU and requires C/C to byte-replay its frozen
   target;
4. runs captured-reference plus all four generated arrays through unchanged
   `strat01_r2b_q6_matmul_batch`;
5. adds exact reference `ffn_inp-0`, delegates each sum to the unchanged
   layer-1 builder, and emits the same eleven checkpoints;
6. records zero donor and zero reference graph executions and never
   self-adjudicates.

No helper may fork `expf`, multiplication ordering, Q8_K quantization, Q6_K
dot/layout, normalization, RoPE, cache, attention, or V-B. Compile with Clang
C11, `-O3 -mavx2 -mfma`, no fast-math, and FP contraction disabled.

## Gates and controls

Compare generated SwiGLU, generated `ffn_out-0`, and every propagated
checkpoint with their immutable reference targets using unchanged NRMSE
`<= 0.002` and normalized maximum `<= 0.01` limits. Direct metrics are
diagnostic; sufficiency verdicts below require a propagated layer-1 failure.

Required controls:

- captured-reference replays the prior all-pass Q6/downstream arm;
- C/C byte-replays frozen C SwiGLU, C down output, terminal sum, all eleven
  propagated outputs, and prior metrics within `1e-12`;
- one-byte mutations of all seven inputs/targets are refused;
- wrong tensor type, shape, layer, offset, or span is refused;
- negating reference gate and swapping reference-up rows 0 and 7 each changes
  generated SwiGLU and fails at least one direct or downstream gate;
- arm/operand-origin relabeling is rejected;
- source inventory binds the production SwiGLU expression, Q6 helper, and
  layer-1 builder.

A model-free self-test must cover expression ordering, finite-value refusal,
fresh-arm reset, five-arm routing, mutations, descriptor refusal, checkpoint
enumeration, and report paths.

## Decision rule

Apply in order:

1. **`VOID_FFN_SWIGLU_CROSS_INPUT`** for any identity, replay, control,
   source, or apparatus failure.
2. **`SWIGLU_OPERATOR_RESIDUAL_SUFFICIENT`** if
   `REFERENCE_GATE_REFERENCE_UP` fails any propagated layer-1 gate.
3. If the reference/reference generated arm passes, classify the two hybrid
   arms by propagated layer-1 gates:
   - both fail: **`GATE_AND_UP_RESIDUALS_INDEPENDENTLY_SUFFICIENT`**;
   - only C-gate/reference-up fails: **`GATE_RESIDUAL_SUFFICIENT`**;
   - only reference-gate/C-up fails: **`UP_RESIDUAL_SUFFICIENT`**;
   - both pass while valid C/C fails: **`GATE_UP_RESIDUAL_JOINT_ONLY`**.

An all-pass C/C replay is an apparatus contradiction and therefore VOID.
Exactly one non-VOID invocation is allowed. Gates and ordering must not change
after values are observed.

## Non-claims and stop rule

This cell cannot repair or promote Rung 2C. It cannot distinguish projection
input versus Q4_K operator semantics if a gate or up branch wins; that would
require a separately frozen successor. It makes no claim about MoE, later
layers, tokenizer, logits, generation, quality, RAM, or rate and cannot update
`SPEED_LEDGER.md`. Document and index the result before any repair or finer
split.
