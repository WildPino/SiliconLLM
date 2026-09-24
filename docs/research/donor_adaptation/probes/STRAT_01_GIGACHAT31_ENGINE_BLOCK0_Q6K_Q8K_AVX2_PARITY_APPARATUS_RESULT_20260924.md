# STRAT-01 block-0 Q6_K×Q8_K AVX2 parity — apparatus qualification

**Status: `APPARATUS_READY_NO_SCIENTIFIC_EXECUTION`.** This is a model-free
readiness record. It makes no full-matrix, downstream, quality, throughput, or
causal claim.

## Canonical apparatus record

The sole qualifying raw record is
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_q6k_q8k_avx2_parity_apparatus_20260924`
(unversioned). Its adjudication SHA-256 is
`c6682d5e7b05997cf544d30b1d3a4f13c2cc1c781c4ac8f86ec6a42a52238d34`.
It observed source HEAD `d21462a62789ef999e5f75ccc4b03d9a6ae23c43`
and pinned x86 `quants.c` SHA-256
`99a98747c1ac84ec40e2d1a31227b947aeb0a05bf1d8e79ec630a6d783b89f86`.

The apparatus compiled the standalone C candidate with Clang C11
`-O3 -mavx2 -mfma` and built a clean llama.cpp oracle whose compile database
selects exactly one x86 `quants.c` with AVX2/FMA and without
`GGML_CPU_GENERIC`. Stored-block fixtures cover vector lengths 256, 512, 1536,
and 8960 (1, 2, 6, and 35 Q6_K blocks). The candidate is byte-exact to the
pinned oracle at every length. The existing scalar/generic reduction differs
on the multi-block control, a one-byte Q6 mutation changes the AVX2 result,
and short reads plus non-block-aligned lengths are rejected. Both dedicated
tests returned rc 0 in 35.042 seconds; this duration is not a performance
measurement.

All scientific counters are zero: model reads, scientific payload reads, Q8
population emissions, Q6 matrix reads, diagnostics, donor graphs, and
reference graphs. Apparatus-only opened neither the accepted GGUF nor any
frozen `.f32le` payload.

## Source bindings

| source | SHA-256 |
|---|---|
| standalone Q6 primitive | `2b43497b749e9055cb00c61db8b12ca4157db82a5d96b4d9774ffeb83817c6cb` |
| C stored-block probe | `ed7004e318d46d5da77bce74ed565d53890eaa926723188aac397d2ee043c4a9` |
| pinned-oracle wrapper | `817a3628b5a98e446ee777304bc3d8a58d643294715f48f4154dc09354f5fce9` |
| oracle builder | `5ede5ca6d1d925f39b22d08b8d89b4e0d17b55957cf09bf625f1cb97e1dcfc15` |
| parity tests | `b5fd8b82bdf03d538c71dcae484102011c2fe44b18774005ea43f50769c98909` |
| apparatus runner | `20c7270bf88bd1420d1ca5721ebcbc5832cdbecbc3f72d2122231a5c4e71e19b` |
| frozen protocol | `9906a92a66159430514942748986cd8ae44ec64ebd8bf6fe81c185f320709f94` |

## Authorized next step

The operation-order hypothesis is now mechanically viable and distinct from
the existing generic path. Build the protocol's C-side full-matrix diagnostic
and pinned full-matrix oracle around this qualified primitive, retain the
model-free branch and zero counters, then commit their exact sources before
the sole zero-graph scientific invocation. Readiness does not establish Q8_K
population parity or repair the immutable block-0 output.
