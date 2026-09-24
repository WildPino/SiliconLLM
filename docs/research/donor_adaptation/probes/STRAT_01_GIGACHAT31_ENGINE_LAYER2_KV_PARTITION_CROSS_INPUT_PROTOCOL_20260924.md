# STRAT-01 GigaChat 3.1 layer-2 KV-partition cross-input protocol

**State:** FROZEN BEFORE IMPLEMENTATION OR EXECUTION

## Question and nearest evidence

The canonical
[layer-2 attention cross-input result](STRAT_01_GIGACHAT31_ENGINE_LAYER2_ATTENTION_CROSS_INPUT_RESULT_20260924.md)
is `LAYER2_KV_RESIDUAL_SUFFICIENT`: with reference query fixed, the complete C
compact-KV row fails the unchanged `kqv_out-2` gate at NRMSE
`0.0027922348709553107`, while the exact-reference operator is byte-exact.
Which disjoint part of that 576-value row is sufficient: the first 512
compressed latent values, which are also the value view, the final 64
positional/RoPE key values, or only their interaction?

This is not a repeat of the whole-KV cross-input cell. That cell closed query
and operator causality but changed all 576 KV values together. This cell holds
the reference query fixed and changes the two semantically distinct KV
partitions independently.

## Immutable inputs and predecessor

Bind the predecessor adjudication at SHA-256
`266eba9376ea338db89b6caf9cb86b956d88d25c48e517b9c1102874b61575c9`
and its scientific output directory:

`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer2_attention_cross_input_20260924/`

Also bind the canonical recovered layer-2 adjudication at SHA-256
`0b5d34ea652676c8ec04b8a87f89ff1e08757c370ebcfdab939871ca6558b46e`
and its sole producer directory:

`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer2_depth_extension_20260924/`

Use only these F32LE payloads:

| source | logical tensor | bytes | SHA-256 |
|---|---|---:|---|
| reference | `Qcur-2` | 589,824 | `46ebbd7e0281249cb01abab38e3b30b3d3efd18ccd13b1ae54324edb73c65dea` |
| reference | `Kcur-2` | 18,432 | `b00da572671dcaf694cf8e2206b3f72ef797a944c680851ca9feb8bedd392a4b` |
| reference | `Vcur-2` | 16,384 | `0486a48dcd7289236760cc5238d31f78c6035fc5001942eb38302dd1ac4abb91` |
| reference | `kqv_out-2` target | 196,608 | `d249db76e99ddb1f93062b1b438fb65d049e6e3ef2bd32281fc1a7e5972f802e` |
| C | `Kcur-2` | 18,432 | `163c0eb7f03892c0ff86cfd1ad455382aa9f59a245c2f849918a92d5537a41f9` |
| C | `Vcur-2` | 16,384 | `e35f17903938047d1f699a92a7e0a85c6abb691a834ffc1cfe363429620372d9` |

For each source, `Vcur-2` must be byte-exact to the first 512 values of every
`Kcur-2` row. All listed prefill payloads and their cached-composition twins
must be byte-identical. Execute one prefill replay only.

The full-C-K/reference-query predecessor output is immutable at SHA-256
`a127b01e4c8f06a7e9a37dcc4dabe307a7a8733c6a6a84c8810d81125a884a92`.
Its frozen metrics against the reference target are NRMSE
`0.0027922348709553107` and normalized maximum
`0.004362653758957389`.

## Changed coordinate and arms

Create a separate boundary helper. It must parse and validate only
`blk.2.attn_v_b.weight` as Q4_K `[512,192,32]`, reuse the accepted F16 cache,
causal-attention, and V-B kernels, and execute no model graph. The reference
query is fixed in every arm. Compose K rows before cache write at the exact
partition boundary `[0,512)` / `[512,576)`.

| arm | latent/value prefix 512 | positional tail 64 | purpose |
|---|---|---|---|
| `ref_latent__ref_positional` | reference | reference | exact-input operator control |
| `c_latent__c_positional` | C | C | byte-exact predecessor replay |
| `c_latent__ref_positional` | C | reference | latent/value-residual sufficiency |
| `ref_latent__c_positional` | reference | C | positional-tail-residual sufficiency |
| `control_ref_latent_token7_negated` | mutated reference | reference | planted prefix/value control |
| `control_ref_positional_rows0_7_swapped` | reference | mutated reference | planted positional control |

Each output is `[8,6144]`, 196,608 F32LE bytes in token-major order. The
helper may report only `OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION`, zero
donor graphs, and no timing or rate claim.

## Apparatus and execution discipline

Before artifact access, an apparatus-only run must compile with Clang C11
`-O3 -mavx2 -mfma`, pass the new helper self-test, its Python tests, the
inherited whole-KV self-test, and the legacy engine self-test, while recording
zero artifact access and zero graphs. Commit the exact passing sources. Then
run exactly one diagnostic invocation. It may hash and parse the accepted
GGUF solely to read and apply `blk.2.attn_v_b.weight`; it may not execute a
donor or reference graph.

## External gates and controls

Require all of the following:

- `ref_latent__ref_positional` byte-exact to the immutable reference target;
- `c_latent__c_positional` byte-exact to the immutable predecessor output;
- reproduction of the frozen predecessor metrics within `1e-12`;
- exact input sizes, hashes, finiteness, paths, labels, source hashes, and the
  layer-2 V-B descriptor;
- both V/K-prefix equalities and all prefill/cached twin equalities;
- one-byte identity mutations of every Q/K/V input rejected;
- block-1 substitution, wrong layer-2 descriptor, partition-boundary change,
  and mixed-arm label swapping rejected;
- token-7 latent negation and positional row-0/row-7 swap each outside at
  least one reference-target gate.

Judge the four scientific arms against the immutable reference target using
the unchanged aggregate limits: NRMSE `<= 0.002` and normalized maximum
`<= 0.01`. Per-token metrics are descriptive only.

## Decision rule

- `LAYER2_EXACT_REFERENCE_FAILS_KV_PARTITION_OPERATOR` if the exact-reference
  arm fails; this contradicts the closed predecessor and implicates the new
  composition path.
- `LAYER2_LATENT_VALUE_RESIDUAL_SUFFICIENT` if only the C-prefix/reference-tail
  arm fails.
- `LAYER2_POSITIONAL_TAIL_RESIDUAL_SUFFICIENT` if only the
  reference-prefix/C-tail arm fails.
- `LAYER2_LATENT_AND_POSITIONAL_RESIDUALS_INDEPENDENTLY_SUFFICIENT` if both
  mixed arms fail.
- `LAYER2_JOINT_KV_PARTITION_INTERACTION_SUFFICIENT` if both mixed arms pass
  but the complete C-K arm fails.
- `VOID_LAYER2_KV_PARTITION_CROSS_INPUT` for any apparatus, identity, exact
  replay, metric replay, descriptor, control, provenance, or execution-
  accounting failure, including an unexpected pass of the complete C-K arm.

After a non-VOID result, never repeat this cell. Follow only the named causal
partition; authorize a new graph only if the required pre-partition
intermediate is absent from immutable evidence.

## Non-claims

This diagnostic does not repair layer 2 and does not test the Q/KV projection
that created the changed partition, output projection, FFN/MoE, later layers,
tokenizer, logits, generation, quality, RAM, or rate. It cannot update
`SPEED_LEDGER.md`.
