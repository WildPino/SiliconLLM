# STRAT-01 GigaChat 3.1 layer-2 KV RMSNorm cross-input protocol

**State:** FROZEN BEFORE IMPLEMENTATION OR EXECUTION

## Question and nearest evidence

The canonical
[KV-partition result](STRAT_01_GIGACHAT31_ENGINE_LAYER2_KV_PARTITION_CROSS_INPUT_RESULT_20260924.md)
is `LAYER2_LATENT_VALUE_RESIDUAL_SUFFICIENT`: with reference query and
positional tail fixed, the C 512-value `kv_cmpr-2` prefix fails the downstream
`kqv_out-2` gate at NRMSE `0.002642086799405445`.

That prefix is produced by applying `blk.2.attn_kv_a_norm.weight` RMSNorm to
the first 512 values of `kv_cmpr_pe-2`. Is the admitted projected-prefix input
residual sufficient under the accepted RMSNorm, or does that RMSNorm fail even
on exact reference input?

This is not a repeat of the partition or attention cells. Query, positional
tail, cache, attention, and V-B stay fixed at their accepted reference paths;
only the input to the already-used layer-2 KV RMSNorm is crossed.

## Immutable evidence and payloads

Bind the predecessor adjudication at SHA-256
`c6f59af8cf83f413cee14ae38ed451270531badc1a71095875a5f39680eebe7f`
and its sole output directory:

`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer2_kv_partition_cross_input_20260924/`

Bind the canonical recovered layer-2 adjudication at SHA-256
`0b5d34ea652676c8ec04b8a87f89ff1e08757c370ebcfdab939871ca6558b46e`
and use only these prefill F32LE payloads from its immutable producer:

| source | logical tensor | bytes | SHA-256 |
|---|---|---:|---|
| reference | `Qcur-2` | 589,824 | `46ebbd7e0281249cb01abab38e3b30b3d3efd18ccd13b1ae54324edb73c65dea` |
| reference | `Kcur-2` | 18,432 | `b00da572671dcaf694cf8e2206b3f72ef797a944c680851ca9feb8bedd392a4b` |
| reference | `kv_cmpr_pe-2` | 18,432 | `81c439e40a9dfd06221a86b7cdc9b6019ea8742e6cf11371d1d2c50546346c61` |
| reference | `kv_cmpr-2` | 16,384 | `0486a48dcd7289236760cc5238d31f78c6035fc5001942eb38302dd1ac4abb91` |
| reference | `kqv_out-2` target | 196,608 | `d249db76e99ddb1f93062b1b438fb65d049e6e3ef2bd32281fc1a7e5972f802e` |
| C | `kv_cmpr_pe-2` | 18,432 | `aae2f54a6391b84c156c3b61f3206be1170afb7e09995cc891f1fd9e7ebfc444` |
| C | `kv_cmpr-2` | 16,384 | `e35f17903938047d1f699a92a7e0a85c6abb691a834ffc1cfe363429620372d9` |

Every listed prefill payload must be byte-identical to its cached-composition
twin. The first 512 values of reference `Kcur-2` must equal reference
`kv_cmpr-2`. The predecessor C-prefix/reference-tail output is immutable at
SHA-256 `81635464fe7ec02d659bbc70d4d961e07743a670720bb3c4c1842dfcbc3ac254`.

## Frozen operators and arms

Create a separate boundary helper. It may parse only these accepted-GGUF
tensors:

- `blk.2.attn_kv_a_norm.weight`, F32 `[512]`, offset `578809088`, file offset
  `584912000`, span `2048` bytes;
- `blk.2.attn_v_b.weight`, Q4_K `[512,192,32]`, offset `589434112`, file offset
  `595537024`, span `1769472` bytes.

Reuse the production `strat01_r2a_rmsnorm_pinned` implementation and the
accepted cache/attention/V-B path. For computed arms, RMS-normalize only
`kv_cmpr_pe-2[:,0:512]`, append the immutable reference `Kcur-2[:,512:576]`,
and use immutable reference Q. Execute these six arms:

| arm | prefix supplied downstream | purpose |
|---|---|---|
| `captured_ref_norm` | captured reference `kv_cmpr-2` | exact downstream control |
| `captured_c_norm` | captured C `kv_cmpr-2` | predecessor replay |
| `computed_ref_input` | current RMSNorm(reference projected prefix) | operator on exact input |
| `computed_c_input` | current RMSNorm(C projected prefix) | production RMSNorm replay/input sufficiency |
| `control_ref_input_token7_negated` | RMSNorm(mutated reference input) | planted input control |
| `control_norm_weight_negated` | RMSNorm(reference input, negated weight) | planted operator control |

Each downstream output is `[8,6144]`, 196,608 F32LE bytes. The helper must
also hash the four 512-value normalized prefixes it computes or consumes. It
may report only `OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION`, zero donor
graphs, and no timing/rate claim.

## Apparatus and execution discipline

Before artifact access, an apparatus-only run must compile with Clang C11
`-O3 -mavx2 -mfma`, pass the new helper self-test, Python tests, inherited
partition and whole-KV tests, and the legacy engine self-test, while recording
zero artifact access and zero graphs. Commit the exact passing sources. Then
run exactly one diagnostic invocation. It may hash/parse the GGUF solely for
the two named weights; it may not execute a donor or reference graph.

## External gates and controls

Require:

- `captured_ref_norm` byte-exact to the immutable reference target;
- `captured_c_norm` byte-exact to the predecessor C-prefix/reference-tail
  output and reproduction of its metrics within `1e-12`;
- `computed_c_input` normalized prefix byte-exact to captured C `kv_cmpr-2`
  and its downstream output byte-exact to `captured_c_norm`;
- exact input sizes, hashes, finiteness, paths, schedule twins, prefix/tail
  identity, source hashes, arm labels, and both tensor descriptors;
- one-byte input mutations, block-1 substitutions, wrong descriptors,
  changing the 512 split, and scientific-arm label swaps rejected;
- both planted controls outside at least one reference-target gate.

Judge all scientific downstream arms against the immutable reference target
using unchanged limits: NRMSE `<= 0.002`, normalized maximum `<= 0.01`.
Per-token and direct normalized-prefix metrics are descriptive.

## Decision rule

- `LAYER2_KV_RMSNORM_FAILS_EXACT_REFERENCE_INPUT` if
  `computed_ref_input` fails after all exact replays pass.
- `LAYER2_PROJECTED_KV_PREFIX_RESIDUAL_SUFFICIENT` if
  `computed_ref_input` passes and `computed_c_input` fails.
- `VOID_LAYER2_KV_RMSNORM_CROSS_INPUT` for any apparatus, identity,
  descriptor, exact replay, metric replay, production-RMSNorm replay,
  control, provenance, or execution-accounting failure, including an
  unexpected pass of `computed_c_input`.

After a non-VOID result, never repeat this cell. If the projected prefix is
sufficient, the next split may address its immutable `attn_norm-2` input
versus `blk.2.attn_kv_a_mqa.weight`; if exact reference input fails, repair the
RMSNorm semantics before any further depth extension.

## Non-claims

This cell does not repair layer 2, execute or adjudicate KV-A projection,
reopen attention, or test output projection, FFN/MoE, later layers, tokenizer,
logits, generation, quality, RAM, or rate. It cannot update `SPEED_LEDGER.md`.
