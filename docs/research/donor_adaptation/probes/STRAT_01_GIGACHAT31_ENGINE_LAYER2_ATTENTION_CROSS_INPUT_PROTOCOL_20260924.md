# STRAT-01 GigaChat 3.1 layer-2 attention cross-input protocol

**State:** FROZEN BEFORE IMPLEMENTATION OR EXECUTION

## Question and nearest evidence

The canonical
[layer-2 result](STRAT_01_GIGACHAT31_ENGINE_LAYER2_DEPTH_EXTENSION_RESULT_20260924.md)
passes every attention input through `Vcur-2` and first fails at `kqv_out-2`
with NRMSE `0.003014631769821054`. Does the accepted causal-attention plus
`blk.2.attn_v_b.weight` operator pass on exact reference Q/KV inputs, and is
the admitted C query residual, the admitted C KV residual, or only their
interaction sufficient to cross the unchanged gate?

The prior layer-1 cross-input cell answered the same algebraic question for
different inputs and `blk.1.attn_v_b.weight`. Its helper structure may be
reused, but its measurement cannot transfer to layer 2.

## Immutable inputs

Bind the canonical recovered adjudication at SHA-256
`0b5d34ea652676c8ec04b8a87f89ff1e08757c370ebcfdab939871ca6558b46e`
and the sole producer directory:

`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer2_depth_extension_20260924/`

Use these F32LE payloads only:

| source | logical tensor | bytes | SHA-256 |
|---|---|---:|---|
| reference | `Qcur-2` | 589,824 | `46ebbd7e0281249cb01abab38e3b30b3d3efd18ccd13b1ae54324edb73c65dea` |
| reference | `Kcur-2` | 18,432 | `b00da572671dcaf694cf8e2206b3f72ef797a944c680851ca9feb8bedd392a4b` |
| reference | `Vcur-2` | 16,384 | `0486a48dcd7289236760cc5238d31f78c6035fc5001942eb38302dd1ac4abb91` |
| reference | `kqv_out-2` target | 196,608 | `d249db76e99ddb1f93062b1b438fb65d049e6e3ef2bd32281fc1a7e5972f802e` |
| C | `Qcur-2` | 589,824 | `a1bcd7529bae496e48db28964123f8f28bfc26fd2559c54ada51fb9ae49464d5` |
| C | `Kcur-2` | 18,432 | `163c0eb7f03892c0ff86cfd1ad455382aa9f59a245c2f849918a92d5537a41f9` |
| C | `Vcur-2` | 16,384 | `e35f17903938047d1f699a92a7e0a85c6abb691a834ffc1cfe363429620372d9` |
| C | `kqv_out-2` target | 196,608 | `47c8b7db2e7a7a01886033870c0806987f804a0ac5ec01e5e8b1db8a285a2ae2` |

Reference and C `Vcur-2` are byte-exact prefixes of their corresponding
`Kcur-2` rows. For all eight listed tensors, prefill and cached-composition
hashes are byte-identical. The diagnostic therefore executes one prefill
replay only; repeating the cached arm would measure the same bytes.

Also bind the prefill manifests:

- reference SHA-256
  `088c75d7be93865abfcea1f22d34080477c360bfd2cfd06738cf86b02cc6b035`;
- C SHA-256
  `b0f7bddb442ba89da0edb129ff20387b3c4016a661a3e7858ebe458c09558989`.

## Changed coordinate and arms

Create a separate layer-2 boundary helper. It must parse and validate only
`blk.2.attn_v_b.weight` as Q4_K `[512,192,32]`, then reuse the accepted
cache-write, causal-attention, and V-B kernels without executing a model
graph. Six frozen arms are required:

| arm | query | KV | purpose |
|---|---|---|---|
| `c_q__c_kv` | C | C | exact native replay |
| `ref_q__ref_kv` | reference | reference | exact-input operator test |
| `c_q__ref_kv` | C | reference | query-residual sufficiency |
| `ref_q__c_kv` | reference | C | KV-residual sufficiency |
| `control_ref_q_token7_negated` | mutated reference | reference | query control |
| `control_ref_kv_rows0_7_swapped` | reference | mutated reference | KV control |

Each output is `[8,6144]`, 196,608 F32LE bytes in token-major order. The
helper may report only `OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION`, zero
donor graphs, and no timing/rate claim.

## Apparatus and execution discipline

Before model access, an apparatus-only run must compile with Clang C11
`-O3 -mavx2 -mfma`, pass the new helper self-test, its Python tests, and the
legacy engine self-test, while recording zero model access and zero graphs.
Commit the exact passing sources. Then run exactly one diagnostic invocation.
It may hash/parse the accepted GGUF solely to read and apply the layer-2 V-B
matrix; it may not execute a donor or reference graph.

## External gates and controls

Require:

- `c_q__c_kv` byte-exact to the immutable C target;
- all output sizes, digests, finiteness, paths, arm labels, and source hashes;
- both V/K-prefix equalities and the exact layer-2 V-B descriptor;
- reproduction of the frozen C/reference metrics within `1e-12`;
- one-byte identity mutations of all six Q/K/V input files rejected;
- block-1 substitution and wrong layer-2 descriptor rejected;
- token-7 query negation and KV row-0/row-7 swap each outside at least one
  reference-target gate;
- mixed-arm label swapping rejected.

Judge the four scientific arms against the immutable reference target using
the unchanged general limits: NRMSE `<= 0.002` and normalized maximum
`<= 0.01`. Report per-token metrics descriptively; aggregate gates decide.

## Decision rule

- `LAYER2_EXACT_REFERENCE_FAILS_ATTENTION_VB_OPERATOR` if exact reference
  Q/KV fails: the operator/weight path is implicated.
- `LAYER2_QUERY_RESIDUAL_SUFFICIENT`, `LAYER2_KV_RESIDUAL_SUFFICIENT`, or
  `LAYER2_QUERY_AND_KV_RESIDUALS_INDEPENDENTLY_SUFFICIENT` according to which
  mixed arms fail after exact reference passes.
- `LAYER2_JOINT_QUERY_KV_INTERACTION_SUFFICIENT` if both mixed arms pass but
  native C/C fails.
- `VOID_LAYER2_ATTENTION_CROSS_INPUT` for any apparatus, identity, replay,
  descriptor, control, provenance, or execution-accounting failure.

After a non-VOID result, never repeat this cell. Any successor must follow the
named causal side; it may use immutable replay before authorizing new graphs.

## Non-claims

This diagnostic does not repair layer 2 and does not test output projection,
FFN/MoE, later layers, tokenizer, logits, generation, quality, RAM, or rate.
It cannot update `SPEED_LEDGER.md`.
