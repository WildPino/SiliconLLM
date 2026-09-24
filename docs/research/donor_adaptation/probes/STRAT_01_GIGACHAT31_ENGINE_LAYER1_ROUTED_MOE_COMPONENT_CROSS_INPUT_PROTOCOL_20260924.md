# STRAT-01 GigaChat 3.1 layer-1 routed-MoE component cross-input protocol

**State:** FROZEN BEFORE IMPLEMENTATION OR EXECUTION

## Question and non-duplication boundary

The canonical [FFN-output component result](STRAT_01_GIGACHAT31_ENGINE_LAYER1_FFN_OUTPUT_COMPONENT_CROSS_INPUT_RESULT_20260924.md)
is `LAYER1_ROUTED_MOE_OUTPUT_RESIDUAL_SUFFICIENT`: the captured C routed MoE
output alone exactly reproduces the downstream failure, while the C shared
expert output alone passes.

Which immutable input of the routed reduction
`ffn_moe_out-1 = sum_slot(ffn_moe_down-1 * ffn_moe_weights_norm-1)` is
sufficient: expert-down output, normalized router weight, both independently,
or only their combination? This does not rerun routing, experts, a producer,
or either graph. The top-4 IDs are already byte-identical and fix slot order.

## Immutable evidence

Bind predecessor adjudication SHA-256
`4d03a081ab06cf695c4569c4d8ab615d2e2b88a2bfc520874da89e32685e984a`
and its decisive failing downstream payload SHA-256
`81635464fe7ec02d659bbc70d4d961e07743a670720bb3c4c1842dfcbc3ac254`.

| source | logical tensor | bytes | SHA-256 |
|---|---|---:|---|
| reference and C | `ffn_moe_topk-1` | 128 | `557f502a4cc9f24ec866d11e63b857b9c60e40cc0cef43bfeb95c6ee59648e95` |
| reference | `ffn_moe_down-1` | 196,608 | `d19ce37968e1c2c6b04dffbc56ebba7ed25d704db2447e45031786b4710bb0af` |
| C | `ffn_moe_down-1` | 196,608 | `c30893f3dd0a751f9312c16fc09638e4f0b55265a5dccaba7205edf3caccc2ce` |
| reference | `ffn_moe_weights_norm-1` | 128 | `075f2321d029c8ca30385a5cc0d8555b8a70cf0e1fec1e4c0108c11447572f03` |
| C | `ffn_moe_weights_norm-1` | 128 | `30cabc5a31c66444fa63734e94c4072a99868920320954b0b466f182f185707a` |
| reference | `ffn_moe_weighted-1` | 196,608 | `875d7a321cfb4c28343edc04bdd1f9940ce014fcfa2aa1306922291c014009b2` |
| C | `ffn_moe_weighted-1` | 196,608 | `9288f02aad131b78eff128ddbd64a4933c659ccffb80f2978d211c34c7c26a94` |
| reference | `ffn_moe_out-1` | 49,152 | `9d09e5582483a471ec57712ad42b5b868f2ee1715f186e1324f9904ebebb08c8` |
| C | `ffn_moe_out-1` | 49,152 | `9d0c6f49f7f6c8ffc68f442d2138e0b30b43a9791e8f861088e02cce882887dc` |

All prefill payloads must equal their cached-composition twins. Reuse the exact
reference `ffn_shexp-1`, `ffn_inp-1`, Q, K, target, accepted artifact, and
closed layer-2 operators from the predecessor.

Descriptively, C versus reference NRMSE is `7.95119111602408e-5` for expert
down, `4.17108215410039e-7` for normalized weights, and
`1.3547864451924633e-5` for the routed sum. This predicts expert-down
sufficiency but is not the downstream verdict.

## Frozen arms and arithmetic

For every token and routed slot, multiply in production F32 order
`weighted = down * weight_norm`, then sum slots 0 through 3 in production F32
order. Add exact reference shared output and exact reference `ffn_inp-1`, then
run the closed layer-2 path.

| arm | routed output construction | purpose |
|---|---|---|
| `captured_ref_moe_out` | captured reference routed output | exact anchor |
| `captured_c_moe_out` | captured C routed output | predecessor replay |
| `computed_ref_down_ref_weights` | reference down × reference weights | reference arithmetic replay |
| `computed_c_down_c_weights` | C down × C weights | C arithmetic replay |
| `cross_c_down_ref_weights` | C down × reference weights | expert-down sufficiency |
| `cross_ref_down_c_weights` | reference down × C weights | router-weight sufficiency |
| `control_ref_down_token7_negated` | negate all four reference down slots at token 7 | component control |

The frozen per-slot weighted hashes for the four computed arms are,
respectively, `875d7a321cfb4c28343edc04bdd1f9940ce014fcfa2aa1306922291c014009b2`,
`9288f02aad131b78eff128ddbd64a4933c659ccffb80f2978d211c34c7c26a94`,
`c26abad90be611b33824373d48b3d016248099884400d499ffe6bdcfdd582de5`,
and `1fb97c90a161cbc09968177efcb29efe61a30598618893347ca7405fa61760e9`.
Their routed-output hashes are `9d09e5582483a471ec57712ad42b5b868f2ee1715f186e1324f9904ebebb08c8`,
`9d0c6f49f7f6c8ffc68f442d2138e0b30b43a9791e8f861088e02cce882887dc`,
`a63d05e6532b6a6c71aa6b09520c4fb8cf076546e6bd293bc97c3aa06c205d81`,
and `f48dfa00abf5fdeaf0b5cb06459102bc81e00e0543ccdb7ace96c69271e392e6`.

Hash weighted slots, routed output, FFN output, terminal sum, layer-2 norm,
projection, prefix, and downstream output. Report only
`OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION`, zero graph executions, and no
timing/rate claim.

## Apparatus, gates, and decisions

Qualify model-free first with Clang C11 `-O3 -mavx2 -mfma`, a new
multiplication/ordered-sum/label self-test, Python tests, all inherited
component and closed-layer-2 self-tests, and the legacy engine self-test. The
apparatus may open neither model nor payload. Commit exact qualified sources
before one permitted scientific invocation.

Require byte-exact twins, identities, mutations, labels, captured anchors,
reference/reference replay, C/C replay, all frozen weighted/routed hashes, and
the predecessor failing downstream hash. Judge only the two cross arms at
downstream NRMSE `<=0.002` and normalized maximum `<=0.01`.

- `LAYER1_ROUTED_EXPERT_DOWN_OUTPUT_RESIDUAL_SUFFICIENT` if
  `cross_c_down_ref_weights` fails and `cross_ref_down_c_weights` passes.
- `LAYER1_NORMALIZED_ROUTER_WEIGHT_RESIDUAL_SUFFICIENT` for the inverse.
- `LAYER1_ROUTED_MOE_INPUT_RESIDUALS_INDEPENDENTLY_SUFFICIENT` if both fail.
- `LAYER1_ROUTED_MOE_INPUT_RESIDUALS_JOINTLY_SUFFICIENT` if both pass while
  the exact C/C replay fails.
- `VOID_LAYER1_ROUTED_MOE_COMPONENT_CROSS_INPUT` for any other outcome.

After a non-VOID verdict, never repeat this cell. Split only the sufficient
captured component at its immediate inputs; do not rerun a producer or graph.

## Non-claims

This cell does not repair routing, expert arithmetic, layer 1 or layer 2; test
an uncaptured operator; or establish tokenizer, later-layer, logits,
generation, task-quality, RAM, or rate behavior.
