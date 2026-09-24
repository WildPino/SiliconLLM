# STRAT-01 GigaChat 3.1 layer-1 routed-Q6 residual propagation protocol

**State:** FROZEN BEFORE IMPLEMENTATION OR EXECUTION

## Question and non-duplication boundary

The canonical [routed-MoE component result](STRAT_01_GIGACHAT31_ENGINE_LAYER1_ROUTED_MOE_COMPONENT_CROSS_INPUT_RESULT_20260924.md)
is `LAYER1_ROUTED_EXPERT_DOWN_OUTPUT_RESIDUAL_SUFFICIENT`: captured C
expert-down outputs alone reproduce the downstream failure; C normalized
router weights alone pass.

The earlier [layer-1 Q6 cross-input](STRAT_01_GIGACHAT31_ENGINE_LAYER1_Q6_CROSS_INPUT_RESULT_20260923.md)
already evaluated the same 32 selected expert matrices on exact reference
routed SwiGLU. Its output passes the *direct* down gate at NRMSE
`7.1095157598044e-8`, but was never propagated through the now-known sensitive
layer-2 amplifier. Reexecuting Q6 would be a duplicate. This cell instead
propagates that immutable old output beside exact reference and current down
anchors through the newly closed weighting and downstream path.

## Immutable evidence

Bind the new predecessor adjudication SHA-256
`ac78794d24467b2ca4c3fce092b969bfd9fa9a2698d2227d5ea5e5c67640ab59`,
the old Q6 adjudication SHA-256
`db9b478e208c35460d4bd1ae54eb56cfb2a8106361b19f0b7f1f024699509148`,
and the predecessor failing downstream SHA-256
`81635464fe7ec02d659bbc70d4d961e07743a670720bb3c4c1842dfcbc3ac254`.

| source | logical tensor | bytes | SHA-256 |
|---|---|---:|---|
| reference capture | exact `ffn_moe_down-1` | 196,608 | `d19ce37968e1c2c6b04dffbc56ebba7ed25d704db2447e45031786b4710bb0af` |
| old Q6 cell | current Q6(reference routed SwiGLU) | 196,608 | `609aa170ac5acf0cec49e403e28e1dde831d14cb76f1de35d1ce51e39e9df91f` |
| current production | current `ffn_moe_down-1` | 196,608 | `c30893f3dd0a751f9312c16fc09638e4f0b55265a5dccaba7205edf3caccc2ce` |
| reference | normalized router weights | 128 | `075f2321d029c8ca30385a5cc0d8555b8a70cf0e1fec1e4c0108c11447572f03` |
| reference | shared expert output | 49,152 | `7bc8b2104cacb7742e57df0c34a1af64fadbff8cc61006ed1aedc034a030ddae` |
| reference | `ffn_inp-1` | 49,152 | `99b8fb7f27962c2d31583e148a978b3ed4a86d54e9ee26a02e10967af61005e8` |

Bind also the predecessor's exact reference Q, K, downstream target, accepted
artifact, and closed layer-2 operators. The exact-reference and current down
anchors must retain their prefill/cached twins. The old Q6 output is bound to
its source adjudication and is not claimed to have a second schedule twin.

## Frozen arms and arithmetic

For each down payload, multiply by exact reference normalized weights, sum
slots 0 through 3 in production F32 order, add exact reference shared output,
then exact reference `ffn_inp-1`, and run the closed layer-2 path.

| arm | selected expert-down payload | purpose |
|---|---|---|
| `captured_ref_down` | exact reference capture | exact downstream anchor |
| `captured_q6_ref_swiglu_down` | old Q6(reference SwiGLU) output | Q6 residual sufficiency |
| `captured_c_down` | current production capture | predecessor replay |
| `control_q6_ref_down_token7_negated` | negate all four token-7 slots of old Q6 output | component control |

Hash weighted slots, routed output, FFN output, terminal sum, layer-2 norm,
projection, prefix, and downstream output. The diagnostic executes neither Q6
nor any donor/reference graph and reports no timing/rate claim.

## Apparatus, gates, and decisions

Qualify model-free first with Clang C11 `-O3 -mavx2 -mfma`, a new immutable
payload/order/label self-test, Python tests, inherited reduction/downstream
self-tests, and the legacy engine self-test. The apparatus may open neither
model nor payload. Commit exact qualified sources before one permitted
scientific invocation.

Require identities, available twins, mutations, labels, exact reference and C
downstream anchors, and exact current replay. Judge the old Q6(reference
SwiGLU) arm at downstream NRMSE `<=0.002` and normalized maximum `<=0.01`.
The planted control must reject.

- `LAYER1_ROUTED_SWIGLU_RESIDUAL_SUFFICIENT` if the old
  Q6(reference-SwiGLU) arm passes while captured C down replays the failure.
- `LAYER1_ROUTED_Q6_RESIDUAL_SUFFICIENT` if that arm fails validly.
- `VOID_LAYER1_ROUTED_Q6_RESIDUAL_PROPAGATION` for any other outcome.

After a non-VOID verdict, never repeat this cell. If SwiGLU is sufficient,
split only its already captured immediate operands/expression semantics. If
Q6 is sufficient, reopen only exact arithmetic attribution under a new frozen
protocol; do not rerun the old direct cell.

## Non-claims

This cell does not execute or repair Q6, routing, experts, layer 1 or layer 2;
rerun a graph; or establish tokenizer, later-layer, logits, generation,
task-quality, RAM, or rate behavior.
