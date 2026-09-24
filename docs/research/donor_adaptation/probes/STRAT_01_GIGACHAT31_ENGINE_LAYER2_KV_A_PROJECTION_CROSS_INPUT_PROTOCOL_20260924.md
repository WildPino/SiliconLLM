# STRAT-01 GigaChat 3.1 layer-2 KV-A projection cross-input protocol

**State:** FROZEN BEFORE IMPLEMENTATION OR EXECUTION

## Question and nearest evidence

The canonical
[KV RMSNorm result](STRAT_01_GIGACHAT31_ENGINE_LAYER2_KV_RMSNORM_CROSS_INPUT_RESULT_20260924.md)
is `LAYER2_PROJECTED_KV_PREFIX_RESIDUAL_SUFFICIENT_RECOVERED`: production KV
RMSNorm is byte-exact, and the sufficient residual already exists in the first
512 values of `kv_cmpr_pe-2`.

`kv_cmpr_pe-2` is produced by applying `blk.2.attn_kv_a_mqa.weight` to
`attn_norm-2`. Does the production Q4_K projection pass on exact reference
input, or is the admitted C `attn_norm-2` residual sufficient?

This is not a repeat of Q4 kernel parity, RMSNorm, partition, or attention.
It changes only the immutable input to this layer-2 matrix and propagates the
result through the closed production KV RMSNorm and downstream path.

## Immutable evidence and payloads

Bind the canonical RMSNorm recovery at SHA-256
`e4f4fb67e73e7089fee3f7900c768894b9c9bf0399b5c978169c06f7b07ca976`
and the original layer-2 producer. Use only these F32LE payloads:

| source | logical tensor | bytes | SHA-256 |
|---|---|---:|---|
| reference | `Qcur-2` | 589,824 | `46ebbd7e0281249cb01abab38e3b30b3d3efd18ccd13b1ae54324edb73c65dea` |
| reference | `Kcur-2` | 18,432 | `b00da572671dcaf694cf8e2206b3f72ef797a944c680851ca9feb8bedd392a4b` |
| reference | `attn_norm-2` | 49,152 | `7bc7ba62b9e5b779cf079ab212f9b328e247c979423992f42aff786560115f02` |
| reference | `kv_cmpr_pe-2` | 18,432 | `81c439e40a9dfd06221a86b7cdc9b6019ea8742e6cf11371d1d2c50546346c61` |
| reference | `kqv_out-2` target | 196,608 | `d249db76e99ddb1f93062b1b438fb65d049e6e3ef2bd32281fc1a7e5972f802e` |
| C | `attn_norm-2` | 49,152 | `b55cdc958cb9e48ff014c740dc4bfe6b1ec81ef839ee95618018e426673e112d` |
| C | `kv_cmpr_pe-2` | 18,432 | `aae2f54a6391b84c156c3b61f3206be1170afb7e09995cc891f1fd9e7ebfc444` |

Every prefill payload must equal its cached-composition twin. The immutable
C-projection downstream output is SHA-256
`81635464fe7ec02d659bbc70d4d961e07743a670720bb3c4c1842dfcbc3ac254`,
NRMSE `0.002642086799405445`, normalized maximum
`0.004687597394884091`.

## Frozen operators and arms

The helper may parse only:

- `blk.2.attn_kv_a_mqa.weight`, Q4_K `[1536,576]`, offset `578311424`, file
  offset `584414336`, span `497664` bytes;
- `blk.2.attn_kv_a_norm.weight`, F32 `[512]`;
- `blk.2.attn_v_b.weight`, Q4_K `[512,192,32]`.

Reuse production Q4_K matmul, KV RMSNorm, cache/attention, and V-B functions.
For every arm, take projection output `[8,576]`, normalize its first 512
values, append immutable reference positional tail and use immutable reference
Q. Execute:

| arm | projection output | purpose |
|---|---|---|
| `captured_ref_projection` | captured reference | exact downstream control |
| `captured_c_projection` | captured C | predecessor replay |
| `computed_ref_input` | production projection(reference `attn_norm-2`) | operator on exact input |
| `computed_c_input` | production projection(C `attn_norm-2`) | production replay/input sufficiency |
| `control_ref_input_token7_negated` | projection(mutated reference input) | input control |
| `control_ref_projection_prefix_negated` | negated first 512 computed reference outputs | postprojection control |

Hash each projection output, normalized prefix, and `[8,6144]` downstream
output. Report only `OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION`, zero donor
graphs, and no timing/rate claim.

## Apparatus and execution discipline

Qualify model-free first with Clang C11 `-O3 -mavx2 -mfma`, new Python/C
tests, inherited RMSNorm/partition/attention tests, and legacy self-test.
Record zero artifact access and zero graphs. Commit exact sources, then run
one diagnostic invocation. It may hash/parse the GGUF only for the three named
weights and may execute no donor/reference graph.

## Gates and decision rule

Require exact captured reference and C downstream replays; computed C
projection bytes must equal captured C projection bytes, its normalized prefix
and downstream output must also replay byte-exactly; all identities,
descriptors, schedule twins, mutations, labels, source hashes, controls, and
execution accounting must pass. Judge downstream outputs at NRMSE `<=0.002`
and normalized maximum `<=0.01`; intermediate metrics are descriptive.

- `LAYER2_KV_A_PROJECTION_FAILS_EXACT_REFERENCE_INPUT` if computed reference
  input fails after all exact replays pass.
- `LAYER2_ATTN_NORM_INPUT_RESIDUAL_SUFFICIENT` if computed reference input
  passes and computed C input fails.
- `VOID_LAYER2_KV_A_PROJECTION_CROSS_INPUT` for any other outcome or failed
  gate, including an unexpected computed-C pass.

After a non-VOID result, never repeat this cell. If input residual is
sufficient, split production of `attn_norm-2` from immutable `l_out-1`; if the
projection fails exact input, repair only that operator before extending depth.

## Non-claims

This cell does not repair layer 2 or test layer-2 attention RMSNorm input,
output projection, FFN/MoE, later layers, tokenizer, generation, quality, RAM,
or rate.

## Addendum A — predecessor-path repair, frozen before repair execution

The first post-qualification runner attempt is preserved at
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer2_kv_a_projection_cross_input_20260924/`;
its `adjudication.json` SHA-256 is
`4e00a58e324650c22497999986ebf3385c80c76f163454a18740d5e0bff43c2f`.
It is `VOID_LAYER2_KV_A_PROJECTION_CROSS_INPUT` with
`diagnostic_invocations=0`, zero donor/reference graphs, and no arm output.

The pre-execution predecessor guard addressed
`captured_c_norm.f32le` below `rms.RAW`, although the frozen RMSNorm output is
below `rms.DEFAULT_OUTPUT`. The expected SHA-256 was already correct and the
file at the latter path matches it exactly. Repair only that directory base,
add a static regression test distinguishing the two roots, and use `repair1`
output directories. No payload, arm, operator, threshold, decision rule, or
scientific execution allowance changes. Requalify model-free and commit the
exact repaired sources before the still-unused single diagnostic invocation.
