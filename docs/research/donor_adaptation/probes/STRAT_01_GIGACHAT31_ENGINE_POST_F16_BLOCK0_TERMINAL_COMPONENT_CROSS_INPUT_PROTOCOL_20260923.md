# STRAT-01 post-F16 block-0 terminal-component cross-input protocol

**Status:** `FROZEN BEFORE IMPLEMENTATION OR EXECUTION`

## Changed coordinate

The valid
[post-F16 layer-1-start result](STRAT_01_GIGACHAT31_ENGINE_POST_F16_LAYER1_START_CROSS_INPUT_RESULT_20260923.md)
is `POST_F16_BLOCK0_TERMINAL_RESIDUAL_SUFFICIENT`: exact reference `l_out-0`
makes all 32 current layer-1 checkpoints pass, while the post-F16 current arm
replays every failure exactly. Layer 1 is closed.

The earlier
[block-0 terminal-component split](STRAT_01_GIGACHAT31_ENGINE_BLOCK0_TERMINAL_COMPONENT_CROSS_INPUT_DIAGNOSTIC_RESULT_20260923.md)
used the old C block-0 components before exact reference-generic Q4 and exact
F16 reduction propagation. Repeating that cell would be redundant, but its
verdict cannot be transferred to the new `l_out-0` state. This cell changes
only that composition: it produces current block-0 components with all now
accepted production semantics, crosses them with immutable reference
components, and uses the already qualified current layer 1 as a downstream
amplifier.

## Frozen identities

Predecessor recovery adjudication SHA-256:
`766061a9f54b601b8f54f08b6aaf6d533928aa19638f04ac2dd6cc23f865c09b`.

All component payloads are `[8,1536]` F32LE, 49,152 bytes.

| payload | SHA-256 |
|---|---|
| reference `ffn_inp-0` | `baa389195bff42ff92593d92650ed9defdd1b1382469e5dde3516dfa82b76ec1` |
| reference `ffn_out-0` | `f5514fe59d64b517a685ec70004a75152a5f4d511417148f34e4d13b3134bef4` |
| reference `l_out-0` | `385073c91f472dd9ffc1c86bcb63c6ed50256a6d5645e240d61ccdb613d814aa` |
| post-F16 current `l_out-0` | `7fb5fe52df684e4af82413004f6845133528241e3b030530d41595f74298bdc4` |

The accepted GGUF remains 6,474,702,976 bytes with SHA-256
`68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.
The pinned reference and post-F16 current manifests retain their hashes from
the parent cell. Every admitted file must pass containment, size, dtype,
finiteness, and SHA-256 checks before model execution.

## One authorized diagnostic

After apparatus qualification and a source commit, exactly one C diagnostic
invocation may:

1. parse/hash the accepted GGUF once;
2. execute only the production prefill block 0 with accepted exact Q4 and
   pinned-generic-F64 F16 reductions;
3. emit current `ffn_inp-0`, `ffn_out-0`, and `l_out-0`, requiring scalar F32
   `ffn_inp-0 + ffn_out-0` to reproduce the frozen current `l_out-0`
   byte-for-byte;
4. require the immutable reference components to reconstruct reference
   `l_out-0` byte-for-byte;
5. construct exactly two hybrid starts:
   `current ffn_inp + reference ffn_out` and
   `reference ffn_inp + current ffn_out`;
6. pass those two hybrids through the unchanged complete current layer 1 and
   emit all 32 canonical checkpoints;
7. execute token-6-negation and row-0/7-swap controls on the first hybrid;
8. report exact helper counts, tensor/source identities, and zero
   donor/reference graph executions.

The expected exact helper totals are `5760` QK calls and `655360` value calls:
one block-0 attention arm plus two scientific hybrid layer-1 arms and two
layer-1 causal-control arms. No reference graph, Rung-2C graph, Stage A,
tokenizer, logits, generation, or timing/rate measurement is allowed.

## Gates and verdicts

Use the unchanged parent limits and exact I32 routing equality at every
checkpoint. Current block-0 homogeneous replay, reference homogeneous replay,
component mutations, origin-label swaps, helper counts, and both causal
controls are mandatory.

Allowed verdicts:

- `POST_F16_BLOCK0_FFN_INP_RESIDUAL_SUFFICIENT` if only the current
  `ffn_inp-0` hybrid fails;
- `POST_F16_BLOCK0_FFN_OUT_RESIDUAL_SUFFICIENT` if only the current
  `ffn_out-0` hybrid fails;
- `POST_F16_BLOCK0_COMPONENT_RESIDUALS_INDEPENDENTLY_SUFFICIENT` if both fail;
- `POST_F16_BLOCK0_TERMINAL_RESIDUAL_JOINT_ONLY` if neither hybrid fails;
- `VOID_POST_F16_BLOCK0_TERMINAL_COMPONENT_CROSS_INPUT` for any identity,
  replay, source, helper, mutation, accounting, or causal-control failure.

The old pre-F16 terminal-component diagnostic, post-F16 layer-1-start
diagnostic, Stage A, and both Rung-2C schedules must not be repeated. No
scientific invocation is allowed before apparatus qualification is recorded.
