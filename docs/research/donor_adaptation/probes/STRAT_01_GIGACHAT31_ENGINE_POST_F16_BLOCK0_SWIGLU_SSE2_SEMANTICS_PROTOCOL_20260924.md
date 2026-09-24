# STRAT-01 post-F16 block-0 SwiGLU SSE2-semantics protocol

**Status:** `FROZEN BEFORE IMPLEMENTATION OR EXECUTION`

## Changed coordinate and no-duplication proof

The valid
[operator-chain result](STRAT_01_GIGACHAT31_ENGINE_POST_F16_BLOCK0_FFN_OPERATOR_CROSS_INPUT_RESULT_20260923.md)
is `POST_F16_BLOCK0_SWIGLU_EXPRESSION_RESIDUAL_SUFFICIENT`. Captured reference
SwiGLU through unchanged current Q6 passes all 32 complete layer-1 gates,
whereas the production expression on the same exact reference gate/up inputs
first fails at routed `ffn_moe_down-1` and exactly replays the predecessor.
Q6, gate/up, block 0, and layer 1 are therefore closed.

The earlier SwiGLU cell used the same scalar production expression but an old
partial downstream surface. Its statement that expression semantics were not
sufficient is scoped to that surface and is superseded for the complete
post-F16 layer-1 amplifier; repeating that cell is not authorized.

Static inspection of the immutable reference producer identifies one precise
changed coordinate:

- GigaChat dense block 0 calls `build_ffn(..., LLM_FFN_SILU, LLM_FFN_PAR)`;
- `build_ffn` dispatches `ggml_swiglu_split`;
- the captured reference build compiles `ggml-cpu/vec.cpp` with
  `-DGGML_CPU_GENERIC`, `-O3`, and no AVX/FMA flags;
- on x86-64, `ggml_vec_swiglu_f32` therefore selects its four-lane SSE2 path;
- that path uses the pinned polynomial `ggml_v_expf`, not scalar `expf`;
- the row width is 8960, exactly divisible by four, so no scalar tail runs.

The existing production helper instead evaluates
`(gate / (1.0f + expf(-gate))) * up` element by element. This protocol tests
only whether reproducing the pinned SSE2 numerical semantics repairs the
sufficient residual.

## Frozen reference-source evidence

The reference source revision is
`llama.cpp@5b335f413e4f73b0809c4fe39af894efbcc6a0d2` with a clean checkout.

| evidence | bytes | SHA-256 |
|---|---:|---|
| `ggml/src/ggml-cpu/vec.cpp` | 25,922 | `a946fee202dfe4528453865a6402d13a004e31ea73e88c3586793bbcc05994f7` |
| `ggml/src/ggml-cpu/vec.h` | 67,630 | `8817801355b20079de39fd4c67c7453ca2033cdb69fc5bd1f71bb66f12f57318` |
| captured `compile_commands.json` | 112,047 | `6864a9b60e55b83d53fb26cf1f3590ae3a03aea7ff1369728ddc5efa6deffb60` |
| captured `build.ninja` | 237,630 | `a25a67b5ad155d9511f84db00ef82830b1ec4ee94df24992b0b6bd892962bc27` |

The implementation must transcribe only the relevant no-FMA SSE2
`ggml_v_expf`, `ggml_v_silu`, and four-lane multiply semantics into a
diagnostic helper. It must record a source-origin comment and may not link or
execute the reference graph.

## Frozen data identities

All payloads are F32LE `prefill8`. Gate/up/SwiGLU have shape `[8,8960]` and
286,720 bytes; FFN/terminal payloads have shape `[8,1536]` and 49,152 bytes.

| payload | SHA-256 |
|---|---|
| reference `ffn_gate-0` | `5c30c0ada2e96ce43a92b22057b4d593b0b9ca342feade7092bfec084e8c6c2a` |
| reference `ffn_up-0` | `2b608af95db90fcde83c29946ba2ac680fb6465cbaaf518fc12874ab510b29e4` |
| captured reference `ffn_swiglu-0` | `de9245a2f6f60e8d7511d03fe981e6b55ef21a8634d5071dd36727a1a5950cef` |
| scalar-libm replay SwiGLU | `ac2b46108a7e18ddebdb1b3a26462bca0faebea536361a5ca4069c4fa7ddcf40` |
| Q6 on captured reference SwiGLU | `f52e0e5bdb22a17ae4a9e0ee9e0c6ffc6bb485b3e61e3f821039eae38697d6ce` |
| Q6 on scalar-libm replay | `7d0037ab06440493b146e9839d5c89ffa8fdea2b952389f12a38ccea2a94e684` |
| reference/current `ffn_inp-0` | `baa389195bff42ff92593d92650ed9defdd1b1382469e5dde3516dfa82b76ec1` |
| captured-reference-Q6 `l_out-0` | `a71c814dc43360f73b49cb165e7cdf7128f2501015685ff16e7e524dd1f05f11` |
| scalar-libm replay `l_out-0` | `7fb5fe52df684e4af82413004f6845133528241e3b030530d41595f74298bdc4` |

The predecessor adjudication SHA-256 is
`ad03099c59a2bb1ee30bb0063c2b4b9f393b1551ec0e19931d0bf518dd1095eb`.
The accepted GGUF remains 6,474,702,976 bytes with SHA-256
`68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.

## Apparatus gate

Before any scientific diagnostic, model-free tests must establish:

1. the exact source constants, operation ordering, four-lane width, and
   no-FMA SSE2 implementation are present;
2. scalar-libm output still byte-replays its frozen hash;
3. synthetic edge and lane tests distinguish the SSE2 candidate from scalar
   `expf` and reject lane/order/source mutations;
4. all input, predecessor, source, and engine identities fail closed;
5. the existing Q6 and complete post-F16 layer-1 helpers are delegated rather
   than copied;
6. the report schema fixes exactly two scientific arms and two causal
   controls, with zero graph and timing/rate fields.

The apparatus run may compile and execute only model-free self-tests. It may
not open the GGUF or any donor/reference payload. Qualification must be
recorded and committed before the scientific run.

## One authorized diagnostic

After apparatus qualification and a source commit, exactly one C diagnostic
invocation may:

1. validate the accepted GGUF, all four immutable inputs, the predecessor,
   and the pinned reference-source evidence;
2. form a mandatory **scalar-libm replay arm** with the unchanged production
   expression and require its frozen SwiGLU, Q6, terminal, and all 32
   predecessor checkpoint identities;
3. form one **pinned SSE2-semantics arm** from the same reference gate/up,
   using the transcribed no-FMA four-lane reference operations;
4. send that candidate through the unchanged current Q6 down helper, exact
   reference `ffn_inp-0`, and complete post-F16 layer 1;
5. execute token-6 negation and row-0/7 swap controls through the same pinned
   SSE2, Q6, and layer-1 path;
6. emit candidate hashes, all 32 checkpoints, exact helper counts, source
   identities, and zero donor/reference graph executions.

There are exactly four layer-1 invocations. Expected totals are `4608` QK
calls, `524288` value calls, and exactly four block-0 Q6 down arms.

## Gates and verdicts

Use the unchanged complete-layer limits and exact I32 routing equality. Apply
verdicts in this order:

1. `VOID_POST_F16_BLOCK0_SWIGLU_SSE2_SEMANTICS` for any identity, scalar
   replay, source, helper, mutation, accounting, or causal-control failure;
2. `POST_F16_BLOCK0_SWIGLU_SSE2_EXACT_REPAIR` if candidate SwiGLU is
   byte-identical to captured reference, candidate Q6/terminal identities
   match the passing predecessor arm, and all 32 gates pass;
3. `POST_F16_BLOCK0_SWIGLU_SSE2_NUMERIC_REPAIR` if the candidate is not
   byte-identical but all 32 gates pass;
4. `POST_F16_BLOCK0_SWIGLU_SSE2_INSUFFICIENT` if the valid candidate fails
   any complete layer-1 gate.

Exact repair authorizes a separately frozen production integration cell; it
does not itself modify the production path or establish full-model quality.

## Stop rule and non-claims

Do not rerun the reference graph, donor graph, old SwiGLU/gate/up cells, Q6,
block 0, F16/Q4 propagation, or layer 1. Do not broaden this cell to AVX2,
FMA, compiler sweeps, or approximate-activation tuning; those are different
coordinates and are allowed only if the pinned SSE2 result is insufficient.

This cell makes no later-layer, tokenizer, logits, generation, quality, RAM,
or rate claim and cannot update `SPEED_LEDGER.md`.
