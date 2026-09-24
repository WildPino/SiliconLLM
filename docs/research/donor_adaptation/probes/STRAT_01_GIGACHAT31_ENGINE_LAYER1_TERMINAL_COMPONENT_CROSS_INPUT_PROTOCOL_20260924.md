# STRAT-01 GigaChat 3.1 layer-1 terminal-component cross-input protocol

**State:** FROZEN BEFORE IMPLEMENTATION OR EXECUTION

## Question and non-duplication boundary

The canonical [attention RMSNorm result](STRAT_01_GIGACHAT31_ENGINE_LAYER2_ATTN_RMSNORM_CROSS_INPUT_RESULT_20260924.md)
is `LAYER1_TERMINAL_RESIDUAL_SUFFICIENT`: exact reference `l_out-1` passes the
complete closed layer-2 path byte-exactly, while C `l_out-1` exactly replays
the failure.

Which immutable input to the terminal layer-1 addition is sufficient:
`ffn_inp-1`, `ffn_out-1`, both independently, or only their combination? This
does not rerun either layer producer or retest any layer-2 operator. It changes
only the two captured operands of `l_out-1 = ffn_inp-1 + ffn_out-1` and reuses
the already closed downstream path.

## Immutable evidence

Bind the predecessor adjudication at SHA-256
`a42540afb96a70dfb411092d3f416cbbf840c83479302ac2eb1fd8bc9ad5d430`
and the accepted layer-1 production integration adjudication at SHA-256
`d543c56d7e463cfec23e4fa3a7ec7fd5bfecbdccfdf84351ddf12085fcc3cc94`.

| source | logical tensor | bytes | SHA-256 |
|---|---|---:|---|
| reference | `ffn_inp-1` | 49,152 | `99b8fb7f27962c2d31583e148a978b3ed4a86d54e9ee26a02e10967af61005e8` |
| C | `ffn_inp-1` | 49,152 | `c506fcc0e51d6b80c90958374527f48c1dd25150bf91b9ff83851c059ec890b2` |
| reference | `ffn_out-1` | 49,152 | `e8cdbbf3154447c90fa2dbddff5a454bd37200091fa2c5aaf7021104e76c104e` |
| C | `ffn_out-1` | 49,152 | `2932f1d3b23fc439ebdbee791a031a93e0681724f62935967e324b19a09196f5` |
| reference | `l_out-1` | 49,152 | `40d5a0f07fbb77c1ef73ca24f81cb35ea0df32457faa8d04d6c5cd33cd9f1d5f` |
| C | `l_out-1` | 49,152 | `9af8cec3f42781e9f4cac6af1ea13f6a63b751a5323201a8a18f91b3f7b7bbcb` |

All prefill payloads must equal their cached-composition twins. Reuse the
predecessor's immutable reference Q, K, target, accepted artifact, layer-2
attention RMSNorm weight, and all closed downstream operators.

Descriptively, C versus reference NRMSE is `5.400853707562925e-08` for
`ffn_inp-1`, `1.247051153528305e-05` for `ffn_out-1`, and
`9.31051008726708e-06` for `l_out-1`. This predicts FFN-output sufficiency but
does not count as the downstream verdict.

## Frozen arms and arithmetic

Perform each addition in production F32 element order, then run production
layer-2 attention RMSNorm and the closed KV-A, compressed-KV RMSNorm,
cache/attention, and V-B path.

| arm | terminal sum | purpose |
|---|---|---|
| `captured_ref_l_out` | captured reference `l_out-1` | exact downstream anchor |
| `captured_c_l_out` | captured C `l_out-1` | predecessor replay |
| `computed_ref_inp_ref_out` | reference input + reference FFN output | reference addition replay |
| `computed_c_inp_c_out` | C input + C FFN output | C addition replay |
| `cross_ref_inp_c_out` | reference input + C FFN output | FFN-output sufficiency |
| `cross_c_inp_ref_out` | C input + reference FFN output | FFN-input sufficiency |
| `control_ref_out_token7_negated` | reference input + mutated reference output | component control |

Hash every terminal sum, normalized output, KV-A projection, compressed prefix,
and downstream output. Report only
`OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION`, with no timing/rate claim and
zero donor/reference graphs.

## Apparatus and gates

Qualify model-free first with Clang C11 `-O3 -mavx2 -mfma`, a new addition and
label self-test, new Python tests, inherited attention-RMSNorm/KV-A/KV-RMSNorm/
partition tests, and the legacy engine self-test. The apparatus may open
neither model nor payload. Commit exact qualified sources before one permitted
scientific invocation.

Require byte-exact schedule twins, source hashes, identities, mutations, arm
labels, captured anchors, reference/reference replay, C/C replay, and all C
downstream stages. Judge the two cross arms only at the predecessor's
downstream gates: NRMSE `<=0.002` and normalized maximum `<=0.01`.

- `LAYER1_FFN_OUTPUT_RESIDUAL_SUFFICIENT` if `cross_ref_inp_c_out` fails and
  `cross_c_inp_ref_out` passes.
- `LAYER1_FFN_INPUT_RESIDUAL_SUFFICIENT` for the inverse outcome.
- `LAYER1_TERMINAL_COMPONENT_RESIDUALS_INDEPENDENTLY_SUFFICIENT` if both fail.
- `LAYER1_TERMINAL_COMPONENT_RESIDUALS_JOINTLY_SUFFICIENT` if both pass while
  the exact C/C replay fails.
- `VOID_LAYER1_TERMINAL_COMPONENT_CROSS_INPUT` for any other outcome.

After a non-VOID verdict, never repeat this cell. Split only the sufficient
component at its already captured immediate inputs; do not rerun producers.

## Non-claims

This cell does not repair layer 1 or 2, test an uncaptured operator, execute a
graph, or establish tokenizer, later-layer, logits, generation, task-quality,
RAM, or rate behavior.
