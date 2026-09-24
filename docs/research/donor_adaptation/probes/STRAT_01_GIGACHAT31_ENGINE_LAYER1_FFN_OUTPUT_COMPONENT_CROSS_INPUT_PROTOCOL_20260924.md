# STRAT-01 GigaChat 3.1 layer-1 FFN-output component cross-input protocol

**State:** FROZEN BEFORE IMPLEMENTATION OR EXECUTION

## Question and non-duplication boundary

The canonical [terminal-component result](STRAT_01_GIGACHAT31_ENGINE_LAYER1_TERMINAL_COMPONENT_CROSS_INPUT_RESULT_20260924.md)
is `LAYER1_FFN_OUTPUT_RESIDUAL_SUFFICIENT`: C `ffn_out-1` with reference
`ffn_inp-1` exactly reproduces the failing downstream hash, while the inverse
cross passes.

Which immutable addend of `ffn_out-1 = ffn_moe_out-1 + ffn_shexp-1` is
sufficient: routed MoE output, shared-expert output, both independently, or
only their combination? This does not rerun a producer, terminal addition,
RMSNorm, projection, or attention. It changes only the two captured FFN-output
components and reuses the exact reference attention residual and closed
layer-2 path.

## Immutable evidence

Bind the predecessor adjudication at SHA-256
`9bdda63cabb7bfe00d7e9dbf76cbead7216aa03064e816cdfa62748fe314ff98`
and its decisive failing downstream payload at SHA-256
`81635464fe7ec02d659bbc70d4d961e07743a670720bb3c4c1842dfcbc3ac254`.

| source | logical tensor | bytes | SHA-256 |
|---|---|---:|---|
| reference | `ffn_inp-1` | 49,152 | `99b8fb7f27962c2d31583e148a978b3ed4a86d54e9ee26a02e10967af61005e8` |
| reference | `ffn_moe_out-1` | 49,152 | `9d09e5582483a471ec57712ad42b5b868f2ee1715f186e1324f9904ebebb08c8` |
| C | `ffn_moe_out-1` | 49,152 | `9d0c6f49f7f6c8ffc68f442d2138e0b30b43a9791e8f861088e02cce882887dc` |
| reference | `ffn_shexp-1` | 49,152 | `7bc8b2104cacb7742e57df0c34a1af64fadbff8cc61006ed1aedc034a030ddae` |
| C | `ffn_shexp-1` | 49,152 | `2fc5d9d6248452269bb6850571559a246bd9703ab984210cffdf0f15e92bbe10` |
| reference | `ffn_out-1` | 49,152 | `e8cdbbf3154447c90fa2dbddff5a454bd37200091fa2c5aaf7021104e76c104e` |
| C | `ffn_out-1` | 49,152 | `2932f1d3b23fc439ebdbee791a031a93e0681724f62935967e324b19a09196f5` |

All prefill payloads must equal their cached-composition twins. Reuse the
predecessor's immutable Q, K, reference target, accepted artifact, and all
closed layer-2 operators.

Descriptively, C versus reference NRMSE is `1.3547864451924633e-05` for the
routed output and `1.358712399393316e-07` for the shared output. This predicts
routed-output sufficiency but is not the downstream verdict.

## Frozen arms and arithmetic

First form each candidate `ffn_out-1` in production F32 order
`ffn_moe_out + ffn_shexp`; then form `l_out-1 = candidate +` exact reference
`ffn_inp-1`. Run production layer-2 attention RMSNorm and the closed KV-A,
compressed-KV RMSNorm, cache/attention, and V-B path.

| arm | candidate `ffn_out-1` | purpose |
|---|---|---|
| `captured_ref_ffn_out` | captured reference output | exact downstream anchor |
| `captured_c_ffn_out` | captured C output | predecessor replay |
| `computed_ref_moe_ref_shared` | reference routed + reference shared | reference addition replay |
| `computed_c_moe_c_shared` | C routed + C shared | C addition replay |
| `cross_c_moe_ref_shared` | C routed + reference shared | routed-output sufficiency |
| `cross_ref_moe_c_shared` | reference routed + C shared | shared-output sufficiency |
| `control_ref_moe_token7_negated` | mutated reference routed + reference shared | component control |

The two frozen cross-sum hashes are respectively
`2d0b1e880f91423cfe59444ba89972e8943a3f1d61a8f50bdb89b2fe1a57ee1c`
and `c79eb03d686e492968bec34321f918709131e8f37b4ea4b314930228b5779a7f`.
Hash every candidate FFN output, terminal sum, normalized output, KV-A
projection, compressed prefix, and downstream output. Report only
`OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION`, with zero graph executions and
no timing/rate claim.

## Apparatus, gates, and decisions

Qualify model-free first with Clang C11 `-O3 -mavx2 -mfma`, a new component
addition/label self-test, new Python tests, inherited terminal-component and
closed layer-2 self-tests, and the legacy engine self-test. The apparatus may
open neither model nor payload. Commit exact qualified sources before one
permitted scientific invocation.

Require byte-exact twins, identities, mutations, labels, captured anchors,
reference/reference replay, C/C replay, and the predecessor failing downstream
hash. Judge only the two cross arms at downstream NRMSE `<=0.002` and
normalized maximum `<=0.01`.

- `LAYER1_ROUTED_MOE_OUTPUT_RESIDUAL_SUFFICIENT` if
  `cross_c_moe_ref_shared` fails and `cross_ref_moe_c_shared` passes.
- `LAYER1_SHARED_EXPERT_OUTPUT_RESIDUAL_SUFFICIENT` for the inverse outcome.
- `LAYER1_FFN_OUTPUT_COMPONENT_RESIDUALS_INDEPENDENTLY_SUFFICIENT` if both fail.
- `LAYER1_FFN_OUTPUT_COMPONENT_RESIDUALS_JOINTLY_SUFFICIENT` if both pass while
  the exact C/C replay fails.
- `VOID_LAYER1_FFN_OUTPUT_COMPONENT_CROSS_INPUT` for any other outcome.

After a non-VOID verdict, never repeat this cell. Split only the sufficient
component at already captured immediate inputs; do not rerun producers.

## Non-claims

This cell does not repair layer 1 or 2, execute a graph, test an uncaptured
operator, or establish tokenizer, later-layer, logits, generation,
task-quality, RAM, or rate behavior.
