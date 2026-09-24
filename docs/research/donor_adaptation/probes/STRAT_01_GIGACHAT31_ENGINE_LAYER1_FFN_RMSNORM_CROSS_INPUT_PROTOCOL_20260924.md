# STRAT-01 layer-1 FFN RMSNorm cross-input protocol

**Status:** frozen before apparatus implementation or scientific execution.

## Question and non-duplication boundary

The canonical
[routed-up projection result](STRAT_01_GIGACHAT31_ENGINE_LAYER1_ROUTED_UP_PROJECTION_CROSS_INPUT_RESULT_20260924.md)
is `LAYER1_ROUTED_UPSTREAM_NORM_RESIDUAL_SUFFICIENT`: the selected production
Q4_K up projections are byte-exact on reference `ffn_norm-1`, while production
C `ffn_norm-1` reproduces the complete downstream failure.

Does production `blk.1.ffn_norm.weight` and the accepted RMSNorm operator pass
on exact reference `ffn_inp-1`, or is the admitted C `ffn_inp-1` residual
already sufficient? This is not the layer-0 float-versus-double accumulator
cell and does not reopen RMSNorm semantics generically. It changes only the
immutable input to layer-1 FFN RMSNorm and judges the result under the newly
established routed-up downstream amplifier.

## Immutable evidence

Bind the routed-up adjudication at SHA-256
`2de61cdc02deeb9639044cb75cc614705a41eb3e133b00f3ad2746b80bc7c71d`.
Accepted GGUF: `6474702976` bytes, SHA-256
`68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.

| payload | bytes | SHA-256 |
|---|---:|---|
| reference `ffn_inp-1` | `49152` | `99b8fb7f27962c2d31583e148a978b3ed4a86d54e9ee26a02e10967af61005e8` |
| production C `ffn_inp-1` | `49152` | `c506fcc0e51d6b80c90958374527f48c1dd25150bf91b9ff83851c059ec890b2` |
| reference `ffn_norm-1` | `49152` | `3f4826dd184c9442fa34807c9649aeae88343c5894c7aac531e5b5a209806dde` |
| production C `ffn_norm-1` | `49152` | `6d2080dcb6eb22c6e9b8f38f8d2beef04cf56aec02161775b65f911dc595f8d7` |
| reference routed gate | `163840` | `341d79877218695fe5d26ab66fb6fb8a25737107c5199e8177eca466deed1a23` |
| ordered top-4 IDs | `128` | `557f502a4cc9f24ec866d11e63b857b9c60e40cc0cef43bfeb95c6ee59648e95` |

Every prefill payload must equal its cached-composition twin. Descriptively,
C versus reference `ffn_inp-1` has NRMSE
`5.400853707562925e-08`, normalized maximum
`5.066021404143073e-08`, and absolute maximum `2.9802322387695312e-08`.
C versus reference `ffn_norm-1` has NRMSE
`8.355378529300766e-08`, normalized maximum
`1.2984894931424172e-07`, and absolute maximum
`1.1920928955078125e-07`. These measurements are descriptive and do not
adjudicate downstream sufficiency.

The newly parsed tensor is `blk.1.ffn_norm.weight`, F32 `[1536]`, tensor
offset `504978688`, file offset `511081600`, span `6144` bytes, payload
SHA-256 `8cbc0bc528143e0ee25f3b23d4e95c30e63f6a86b5aee4bbb42f15f04fba4681`.

## Frozen arms and operators

Apply production RMSNorm with epsilon `1e-6` and the already accepted pinned
accumulation semantics. Then apply the same selected current
`blk.1.ffn_up_exps.weight` Q4_K matrices, exact captured reference routed gate,
the no-FMA SSE2 SwiGLU primitive, selected current Q6 down matrices, exact
reference normalized route weights, ordered reduction, reference shared
output and FFN residual, and the closed layer-2 downstream path.

| arm | normalized output | purpose |
|---|---|---|
| `captured_reference_norm` | captured reference `ffn_norm-1` | exact downstream anchor |
| `captured_production_c_norm` | captured production C `ffn_norm-1` | predecessor replay |
| `computed_norm_on_reference_input` | RMSNorm(reference `ffn_inp-1`) | operator on exact input |
| `computed_norm_on_c_input` | RMSNorm(C `ffn_inp-1`) | input sufficiency and production replay |
| `control_reference_input_token7_negated` | RMSNorm(mutated reference input) | input sensitivity |
| `control_norm_weight_negated` | RMSNorm with negated norm weight | operator sensitivity |

Hash each `[8,1536]` normalized output, `[8,4,1280]` selected up output,
SwiGLU output, `[8,4,1536]` Q6 down output, `[8,1536]` routed sum, and
`[8,6144]` complete downstream output. Report direct metrics at every stage,
but adjudicate only the complete downstream output.

## Gates and decisions

Use complete downstream limits NRMSE `<=0.002` and normalized maximum
`<=0.01`, globally and per token.

Require byte-exact captured reference/C downstream replay and byte-exact
computed-C replay of captured production C `ffn_norm-1` and every downstream
stage. The computed-reference arm is the estimand and must not be forced to
replay its target before adjudication; report its direct and downstream
metrics instead. Also require schedule twins; artifact, payload, tensor,
label, source and mutation controls; both planted controls failing; and exact
execution accounting.

- if `computed_norm_on_reference_input` passes and
  `computed_norm_on_c_input` fails:
  `LAYER1_FFN_INPUT_RESIDUAL_SUFFICIENT`;
- if `computed_norm_on_reference_input` fails after all exact replay and
  control gates pass:
  `LAYER1_FFN_RMSNORM_FAILS_EXACT_REFERENCE_INPUT`;
- otherwise, including an unexpected computed-C pass:
  `VOID_LAYER1_FFN_RMSNORM_CROSS_INPUT`.

## Apparatus and execution discipline

Qualify model-free first with Clang C11 `-O3 -mavx2 -mfma`, new Python/C
tests, inherited routed-up/SwiGLU/Q6 tests, and the legacy self-test. The
apparatus must open neither model nor payloads. Commit exact sources, then
permit one zero-graph diagnostic invocation. No donor or reference graph and
no timing or rate claim is permitted.

After a valid result, never repeat this cell. If the C FFN input residual is
sufficient, close FFN RMSNorm and move upstream into the already captured
layer-1 attention output construction without rerunning any producer. If
exact reference input fails, freeze a production RMSNorm repair confirmation
without reopening the up projection or downstream operators.

## Non-claims

This cell does not repair layer 1, rerun either producer, test the already
closed routed-up or downstream operators independently, or establish later
layers, tokenizer, generation, quality, RAM, or rate.
