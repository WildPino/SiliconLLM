# STRAT-01 GigaChat 3.1 layer-2 attention RMSNorm cross-input protocol

**State:** FROZEN BEFORE IMPLEMENTATION OR EXECUTION

## Question and non-duplication boundary

The canonical
[KV-A result](STRAT_01_GIGACHAT31_ENGINE_LAYER2_KV_A_PROJECTION_CROSS_INPUT_RESULT_20260924.md)
is `LAYER2_ATTN_NORM_INPUT_RESIDUAL_SUFFICIENT`: production KV-A is exact on
reference input, and C `attn_norm-2` reproduces the complete downstream failure.

Does production `blk.2.attn_norm.weight` pass on exact reference `l_out-1`, or
is the admitted C `l_out-1` residual already sufficient? This is not another
generic RMSNorm, Q4, KV, or attention test. It changes only the immutable input
to layer-2 attention RMSNorm and propagates through the closed production path.

## Immutable evidence

Bind the KV-A adjudication at SHA-256
`d16af478a412e22457f11e4d0946e8c7d9fec1c842dfee9657fb213ec8cc06d1`.
Use only the existing layer-2 producer payloads:

| source | logical tensor | bytes | SHA-256 |
|---|---|---:|---|
| reference | `l_out-1` | 49,152 | `40d5a0f07fbb77c1ef73ca24f81cb35ea0df32457faa8d04d6c5cd33cd9f1d5f` |
| C | `l_out-1` | 49,152 | `9af8cec3f42781e9f4cac6af1ea13f6a63b751a5323201a8a18f91b3f7b7bbcb` |
| reference | `attn_norm-2` | 49,152 | `7bc7ba62b9e5b779cf079ab212f9b328e247c979423992f42aff786560115f02` |
| C | `attn_norm-2` | 49,152 | `b55cdc958cb9e48ff014c740dc4bfe6b1ec81ef839ee95618018e426673e112d` |
| reference | `Qcur-2` | 589,824 | `46ebbd7e0281249cb01abab38e3b30b3d3efd18ccd13b1ae54324edb73c65dea` |
| reference | `Kcur-2` | 18,432 | `b00da572671dcaf694cf8e2206b3f72ef797a944c680851ca9feb8bedd392a4b` |
| reference | `kqv_out-2` target | 196,608 | `d249db76e99ddb1f93062b1b438fb65d049e6e3ef2bd32281fc1a7e5972f802e` |

Every prefill payload must equal its cached-composition twin. Descriptively,
C versus reference `l_out-1` has NRMSE `9.31051008726708e-06` and normalized
maximum `6.116194296827801e-06`; C versus reference `attn_norm-2` has NRMSE
`8.798445530899196e-06` and normalized maximum
`6.012043159006656e-06`. Both differences are concentrated at token 5.

The only newly parsed tensor is `blk.2.attn_norm.weight`, F32 `[1536]`, tensor
offset `578811136`, file offset `584914048`, span `6144` bytes, payload
SHA-256 `64a8670a1953e5094ec27d3836c80bb4a944255355768829e782577b26532356`.

## Frozen arms and operators

Apply production RMSNorm with epsilon `1e-6` and its accepted accumulation
semantics, then reuse production KV-A, compressed-KV RMSNorm,
cache/attention, and V-B. Immutable reference Q, positional K tail, and all
other inputs remain fixed.

| arm | normalized output | purpose |
|---|---|---|
| `captured_ref_norm` | captured reference `attn_norm-2` | exact downstream control |
| `captured_c_norm` | captured C `attn_norm-2` | predecessor replay |
| `computed_ref_input` | RMSNorm(reference `l_out-1`) | operator on exact input |
| `computed_c_input` | RMSNorm(C `l_out-1`) | input sufficiency / production replay |
| `control_ref_input_token7_negated` | RMSNorm(mutated reference input) | input control |
| `control_norm_weight_negated` | RMSNorm with negated norm weight | operator control |

Hash each `[8,1536]` normalized output, `[8,576]` KV-A projection,
`[8,512]` compressed-KV prefix, and `[8,6144]` downstream output. Report only
`OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION`, zero donor graphs, and no timing
or rate claim.

## Apparatus and execution discipline

Qualify model-free first with Clang C11 `-O3 -mavx2 -mfma`, new Python/C tests,
inherited KV-A/RMSNorm/partition tests, and the legacy self-test. The apparatus
must open neither model nor payloads. Commit exact sources, then permit one
zero-graph diagnostic invocation.

## Gates and decision rule

Require byte-exact captured reference/C downstream replay and byte-exact
computed-C replay at normalized output, projection, prefix, and downstream.
Require exact identities, tensor descriptor, schedule twins, mutations,
labels, source hashes, controls, and execution accounting. Judge only the
downstream outputs at NRMSE `<=0.002` and normalized maximum `<=0.01`;
intermediate metrics are descriptive.

- `LAYER2_ATTN_RMSNORM_FAILS_EXACT_REFERENCE_INPUT` if computed reference
  input fails after all exact replays pass.
- `LAYER1_TERMINAL_RESIDUAL_SUFFICIENT` if computed reference input passes and
  computed C input fails.
- `VOID_LAYER2_ATTN_RMSNORM_CROSS_INPUT` for every other outcome, including an
  unexpected computed-C pass.

After a non-VOID result, never repeat this cell. If the input residual is
sufficient, close layer-2 attention RMSNorm and localize the already admitted
`l_out-1` residual without rerunning any producer. If exact reference input
fails, repair only layer-2 attention RMSNorm before extending depth.

## Non-claims

This cell does not repair layer 1 or 2, rerun either producer, or test earlier
layer-1 internals, later layers, tokenizer, generation, quality, RAM, or rate.

## Addendum A — initial apparatus attempt is VOID

The initial model-free apparatus attempt is preserved at
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer2_attn_rmsnorm_cross_input_apparatus_20260924`.
Its adjudication SHA-256 is
`aabeabefcdb1ef2621e18bc30b6c85ab322fc394e8cb710603990555c74c6256`.
It is `VOID_LAYER2_ATTN_RMSNORM_CROSS_INPUT`: compilation stopped at an
undeclared cleanup symbol before any diagnostic invocation; the artifact was
not opened, and donor/reference graph executions are both zero.

The only permitted repair replaces the misspelled cleanup call
`strat01_inventory_free` with the existing `strat01_free_inventory` and moves
the apparatus destination to the immutable `apparatus_repair1_20260924`
directory. Payloads, arms, production operators, gates, and decision rules are
unchanged. Scientific execution remains forbidden until repair 1 qualifies.
