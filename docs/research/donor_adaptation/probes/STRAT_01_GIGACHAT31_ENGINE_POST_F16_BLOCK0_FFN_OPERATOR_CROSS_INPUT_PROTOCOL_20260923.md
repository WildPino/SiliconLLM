# STRAT-01 post-F16 block-0 FFN operator-chain cross-input protocol

**Status:** `FROZEN BEFORE IMPLEMENTATION OR EXECUTION`

## Changed coordinate and no-duplication proof

The valid
[post-F16 terminal-component result](STRAT_01_GIGACHAT31_ENGINE_POST_F16_BLOCK0_TERMINAL_COMPONENT_CROSS_INPUT_RESULT_20260923.md)
is `POST_F16_BLOCK0_FFN_OUT_RESIDUAL_SUFFICIENT`. Current `ffn_inp-0` is
byte-identical to reference, while current `ffn_out-0` alone reproduces the
complete post-F16 layer-1 failure.

The current output has SHA-256
`7d0037ab06440493b146e9839d5c89ffa8fdea2b952389f12a38ccea2a94e684`.
That is exactly the output already generated in the earlier valid SwiGLU
cross-input cell from immutable **reference gate + reference up**, the
production C SwiGLU expression, and the current Q6 down helper. The resulting
terminal sum is also byte-identical to the new current `l_out-0`.

Consequently, recomputing or crossing current gate/up operands cannot be the
minimal next experiment: all-reference operands already reproduce the
sufficient residual. The earlier SwiGLU/Q6 cells stopped at the old partial
layer-1 attention surface and therefore did not adjudicate whether either
tiny operator residual is sufficient under the complete post-F16 layer 1.
This cell changes only that downstream amplifier.

## Frozen identities

All operands are F32LE `prefill8` payloads. Gate/up/SwiGLU tensors are
`[8,8960]`, 286,720 bytes; FFN and terminal tensors are `[8,1536]`, 49,152
bytes.

| payload | SHA-256 |
|---|---|
| reference `ffn_gate-0` | `5c30c0ada2e96ce43a92b22057b4d593b0b9ca342feade7092bfec084e8c6c2a` |
| reference `ffn_up-0` | `2b608af95db90fcde83c29946ba2ac680fb6465cbaaf518fc12874ab510b29e4` |
| captured reference `ffn_swiglu-0` | `de9245a2f6f60e8d7511d03fe981e6b55ef21a8634d5071dd36727a1a5950cef` |
| production expression on reference gate/up | `ac2b46108a7e18ddebdb1b3a26462bca0faebea536361a5ca4069c4fa7ddcf40` |
| exact reference `ffn_out-0` | `f5514fe59d64b517a685ec70004a75152a5f4d511417148f34e4d13b3134bef4` |
| current Q6 on captured reference SwiGLU | `f52e0e5bdb22a17ae4a9e0ee9e0c6ffc6bb485b3e61e3f821039eae38697d6ce` |
| current Q6 on generated reference/reference SwiGLU | `7d0037ab06440493b146e9839d5c89ffa8fdea2b952389f12a38ccea2a94e684` |
| reference/current `ffn_inp-0` | `baa389195bff42ff92593d92650ed9defdd1b1382469e5dde3516dfa82b76ec1` |
| exact reference `l_out-0` | `385073c91f472dd9ffc1c86bcb63c6ed50256a6d5645e240d61ccdb613d814aa` |
| captured-reference-SwiGLU/current-Q6 start | `a71c814dc43360f73b49cb165e7cdf7128f2501015685ff16e7e524dd1f05f11` |
| generated-reference/reference/current-Q6 start | `7fb5fe52df684e4af82413004f6845133528241e3b030530d41595f74298bdc4` |

The predecessor adjudication SHA-256 is
`4d30ffb5d79d0682b1522157caf27bded8edd655bfe4e7bdf8c5353a59d515fb`.
The prior valid SwiGLU and FFN-down adjudications are respectively
`0f897c815c1f5d5dc3ede8f0aacca687df325f10a2d4b504be679f65b42922f5`
and `6d3ab2b6a1097502060380350238940d434d38b2b41bcdff8d7c586f5f8d05d0`.

The accepted GGUF remains 6,474,702,976 bytes with SHA-256
`68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.
Every admitted file must pass containment, byte count, SHA-256, dtype,
shape, and finiteness checks before model access.

## One authorized diagnostic

After apparatus qualification and a source commit, exactly one C diagnostic
invocation may:

1. parse/hash the accepted GGUF once and read only block-0 `ffn_down.weight`
   plus the already qualified layer-1 tensors;
2. regenerate `SiLU(reference_gate) * reference_up` with the unchanged
   production scalar expression and require the exact frozen generated hash;
3. form the **Q6-only arm** by sending captured reference SwiGLU through the
   unchanged current Q6 helper, requiring its frozen output/start hashes;
4. form the **expression+Q6 replay arm** by sending the generated
   reference/reference SwiGLU through the same Q6 helper, requiring exact
   current `ffn_out-0` and `l_out-0` hashes;
5. pass both starts through the unchanged complete current layer 1 and emit
   all 32 canonical checkpoints;
6. require the expression+Q6 arm to replay all 32 predecessor current
   checkpoints byte-for-byte;
7. execute token-6 negation and row-0/7 swap controls on captured reference
   SwiGLU, each through the same Q6 and layer-1 path;
8. report exact helper counts, source/tensor identities, and zero
   donor/reference graph executions.

No block-0 attention, RMSNorm, Q4 projection, router, tokenizer, donor graph,
reference graph, generation, or timing/rate measurement is allowed.

There are exactly four layer-1 invocations: two scientific arms and two
causal controls. Therefore exact expected totals are `4608` QK calls and
`524288` value calls. Exactly four block-0 Q6 down arms must be recorded.

## Gates, controls, and verdicts

Use the unchanged complete-layer checkpoint limits and exact I32 routing
equality. The exact reference-start all-pass baseline is inherited from the
hash-bound predecessor and must not be rerun. Generated-expression replay,
both Q6 output/start hashes, all 32 expression+Q6 checkpoint hashes, helper
counts, source delegation, mutations, arm labels, and both causal controls
are mandatory.

Apply verdicts in this order:

1. `VOID_POST_F16_BLOCK0_FFN_OPERATOR_CROSS_INPUT` for any identity, replay,
   source, helper, mutation, accounting, or causal-control failure;
2. the expression+Q6 replay arm must reproduce the predecessor failure; an
   all-pass or non-replaying arm is VOID;
3. `POST_F16_BLOCK0_Q6_RESIDUAL_SUFFICIENT` if the Q6-only arm fails any
   complete layer-1 gate;
4. `POST_F16_BLOCK0_SWIGLU_EXPRESSION_RESIDUAL_SUFFICIENT` if the Q6-only
   arm passes all 32 gates while the mandatory expression+Q6 replay fails.

If Q6-only fails, this cell establishes Q6 sufficiency but does not claim the
expression residual is independently harmless. If Q6-only passes, the only
changed input between the two scientific arms is the frozen C-expression
SwiGLU residual, so its sufficiency is identified.

## Stop rule and non-claims

Do not rerun block 0, either terminal-component cell, the old FFN-down or
SwiGLU cells, the gate/up hybrids, Stage A, or either Rung-2C graph. No
scientific invocation is allowed before apparatus qualification is recorded.

This cell makes no claim about repair, later layers, tokenizer, logits,
generation, quality, RAM, or rate and cannot update `SPEED_LEDGER.md`.
