# STRAT-01 layer-1 attention-output residual cross-input protocol

**Status:** frozen before apparatus implementation or scientific execution.

## Question and non-duplication boundary

The canonical [FFN RMSNorm result](STRAT_01_GIGACHAT31_ENGINE_LAYER1_FFN_RMSNORM_CROSS_INPUT_RESULT_20260924.md)
is `LAYER1_FFN_INPUT_RESIDUAL_SUFFICIENT`: production RMSNorm is exact on
reference `ffn_inp-1`, while the tiny captured C input residual reproduces the
complete downstream failure.

The immediate construction is

`ffn_inp-1 = Q4_K(blk.1.attn_output.weight, kqv_out-1) + l_out-0`.

Does the current output projection plus residual addition pass on exact
reference `l_out-0`, while the current post-SwiGLU C `l_out-0` alone is
sufficient to replay the failure? This is not a repeat of the post-F16
layer-1-start cell. That earlier cell propagated a predecessor block-0 state
with SHA-256
`7fb5fe52df684e4af82413004f6845133528241e3b030530d41595f74298bdc4`
through the whole layer-1 attention graph. The present cell binds the later
production-integrated block-0 state
`a71c814dc43360f73b49cb165e7cdf7128f2501015685ff16e7e524dd1f05f11`,
uses the now byte-identical captured `kqv_out-1`, and changes only the residual
operand of the output projection/add boundary.

## Immutable evidence

Bind the predecessor RMSNorm adjudication at SHA-256
`831fe805bdffeee26693543bed60e96f9f0fe42945e315b16108677736589943`.
Accepted GGUF: `6474702976` bytes, SHA-256
`68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.

| payload | bytes | SHA-256 |
|---|---:|---|
| reference `l_out-0` | `49152` | `385073c91f472dd9ffc1c86bcb63c6ed50256a6d5645e240d61ccdb613d814aa` |
| production C `l_out-0` | `49152` | `a71c814dc43360f73b49cb165e7cdf7128f2501015685ff16e7e524dd1f05f11` |
| reference `kqv_out-1` | `196608` | `fa1006c4e2c365d3a5540baebbe4c88180f524d5f69c610fcc3b4205a856d489` |
| production C `kqv_out-1` | `196608` | `fa1006c4e2c365d3a5540baebbe4c88180f524d5f69c610fcc3b4205a856d489` |
| reference `ffn_inp-1` | `49152` | `99b8fb7f27962c2d31583e148a978b3ed4a86d54e9ee26a02e10967af61005e8` |
| production C `ffn_inp-1` | `49152` | `c506fcc0e51d6b80c90958374527f48c1dd25150bf91b9ff83851c059ec890b2` |

Every prefill payload must equal its cached-composition twin. Descriptively,
C versus reference `l_out-0` has NRMSE `5.4808396483682273e-8`, normalized
maximum `6.81948775008596e-8`, and absolute maximum
`2.9802322387695312e-8`. The `kqv_out-1` pair is byte-identical. C versus
reference `ffn_inp-1` has NRMSE `5.400853707562925e-8`, normalized maximum
`5.066021404143073e-8`, and absolute maximum
`2.9802322387695312e-8`. These are descriptive checks, not the causal verdict.

The admitted matrix is `blk.1.attn_output.weight`, Q4_K `[6144,1536]`, tensor
offset `315482112`, GGUF file offset `321585024`, span `5308416` bytes, payload
SHA-256 `021962db599c0f951aea6e26d69c46774fba1cfb1a845d7264cac907355812ea`.

## Frozen arms and operators

Run the accepted production Q4_K×Q8_K output projection on the immutable,
byte-identical reference `kqv_out-1`. Add the indicated residual operand with
the production float-add order. Then apply the already accepted production
FFN RMSNorm and the exact same selected-up, captured routed-gate, no-FMA SSE2
SwiGLU, selected Q6-down, normalized-route-weight, ordered-reduction,
reference-shared-output, and closed layer-2 downstream amplifier used by the
predecessor.

| arm | `ffn_inp-1` construction | purpose |
|---|---|---|
| `captured_reference_ffn_input` | immutable reference capture | downstream anchor |
| `captured_production_c_ffn_input` | immutable production C capture | predecessor replay |
| `computed_projection_plus_reference_residual` | current output projection + reference `l_out-0` | output operator/add on exact residual |
| `computed_projection_plus_c_residual` | current output projection + production C `l_out-0` | residual sufficiency and C replay |
| `control_reference_residual_token6_negated` | current projection + mutated reference residual | residual sensitivity |
| `control_kqv_token7_negated` | projection of mutated exact `kqv_out-1` + reference residual | projection sensitivity |

Hash the projected `[8,1536]` operand, resulting `[8,1536]` `ffn_inp-1`,
normalized input, selected up, SwiGLU, Q6 down, routed sum, and complete
`[8,6144]` downstream output. Report direct metrics at every stage, but
adjudicate only the complete downstream output.

## Gates and decisions

Use complete downstream limits NRMSE `<=0.002` and normalized maximum
`<=0.01`, globally and per token.

Require byte-exact captured reference/C downstream replay. The
computed-reference arm is the estimand and must not be forced to replay the
reference before adjudication; report exact-replay flags and direct metrics at
every stage. The computed-C arm must replay production C for a residual-
sufficiency verdict. Also require the two captured `kqv_out-1` hashes to be
identical, schedule twins, artifact/payload/tensor/label/source/mutation
controls, both planted controls failing, and exact execution accounting.

- if the computed-reference arm passes, the computed-C arm fails, and both
  exact replays hold: `LAYER1_BLOCK0_TERMINAL_RESIDUAL_SUFFICIENT`;
- if the computed-reference arm fails after every identity, replay, and
  control gate passes: `LAYER1_ATTN_OUTPUT_PROJECTION_FAILS_EXACT_KQV`;
- otherwise: `VOID_LAYER1_ATTENTION_OUTPUT_RESIDUAL_CROSS_INPUT`.

## Apparatus and execution discipline

Qualify model-free first with Clang C11 `-O3 -mavx2 -mfma`, new Python/C
tests, inherited FFN-RMSNorm/routed-up/SwiGLU/Q6/propagation tests, and the
legacy self-test. Apparatus-only must open neither the GGUF nor scientific
payloads. Commit exact sources before permitting one zero-graph scientific
invocation. No donor/reference graph, tokenizer, generation, timing, or rate
claim is permitted.

After a valid result, never repeat this cell. If the current `l_out-0`
residual is sufficient, move upstream to a current-state component split of
`l_out-0 = ffn_inp-0 + ffn_out-0`, binding the post-SwiGLU-production hashes
and reusing this downstream amplifier. If the output projection fails exact
input, freeze a projection-arithmetic diagnostic without reopening any
attention producer or downstream operator.

## Non-claims

This cell does not rerun or adjudicate the layer-1 attention graph, alter the
already exact `kqv_out-1`, repair block 0, establish full-model fidelity, or
make quality, RAM, generation, or throughput claims.
