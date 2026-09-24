# STRAT-01 layer-1 attention-output residual cross-input — apparatus qualification

**Status: `APPARATUS_READY_NO_DONOR_EXECUTION`.** This is a mechanical
readiness record only. It makes no scientific, model-quality, throughput, or
causal claim.

## Canonical apparatus record

The sole qualifying raw record is
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer1_attention_output_residual_cross_input_apparatus_20260924`
(unversioned directory). It reports no error and binds:

- adjudication SHA-256:
  `bebfddc03f68017112bc147bede5efd984cb31b0cc76aeba6b8c5751433dd921`;
- binary SHA-256:
  `9ad7f3555aa7ecc2c139cb48838a2bffd79b90db665641c81d84a6b893fa9863`;
- observed source HEAD:
  `482bc0ee596bb06a53b992138277192d6cf18af1`;
- expected artifact: 6,474,702,976 bytes, SHA-256
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`;
  `opened=false`, observed bytes and SHA-256 are null.

All scientific counters are zero: diagnostic, output-projection,
constructed-FFN-input, RMSNorm, selected-up, downstream propagation, donor
graph, and reference graph. Clang compilation, the new C self-test, inherited
FFN-RMSNorm/routed-up/SwiGLU/Q6/propagation self-tests, the 73,024-check legacy
suite, and six Python tests all returned rc 0. The new C self-test passed six
checks. Apparatus orchestration took `7.524976300002891` seconds; this is not a
timing or rate result.

Source bindings:

| source | SHA-256 |
|---|---|
| runner `run_strat01_layer1_attention_output_residual_cross_input.py` | `58072d5e28968639b118ceb6d44030302855e56af8d3c13a20f2012b407785bd` |
| tests `test_strat01_layer1_attention_output_residual_cross_input.py` | `f12220a22f773123765e07cef781bbb4c6921a4692265bbf3e0388bbeb2e428d` |
| frozen protocol | `def90ba239967b353afa630edac82f1efe9114b7937bce64567e3d77e60f7580` |
| `engine.c` | `8d694f7327677461aca954677a5af5cc6260d16b0bc6d44443938072334e7bd1` |
| apparatus header | `e8f290a8c69cf6a6eb0fcb45bf80ee9e3a2c7a1818f907d624104509f6da3180` |
| FFN RMSNorm runner | `df87ae2d51b812d3b845b82b2edd141765021cbaee4450b23580251eb48be0e8` |
| base metric runner | `3a170204ad5b08716bce29b9b8e5f7046bdc8b371d7e1f969d675fa75c5edafd` |

The apparatus binds predecessor adjudication
`831fe805bdffeee26693543bed60e96f9f0fe42945e315b16108677736589943`
and its production-C downstream tensor
`captured_production_c_norm.downstream.f32le`, SHA-256
`81635464fe7ec02d659bbc70d4d961e07743a670720bb3c4c1842dfcbc3ac254`.

## Scientific safeguards

The six-arm contract contains captured reference/C anchors, current output
projection plus reference/C residuals, and two causal controls. The C path
evaluates two distinct Q4_K projections, constructs four FFN inputs, and then
uses the already closed six-arm downstream chain. Aggregate and per-token
gates are both enforced externally. Exact arm labels/order, output sizes and
hashes, source hashes, tensor descriptors, schedule twins, one-byte mutation
refusal, and label-swap rejection are mandatory.

The frozen protocol was clarified before qualification so the computed-
reference arm is not forced to replay the target before adjudication. This
keeps `LAYER1_ATTN_OUTPUT_PROJECTION_FAILS_EXACT_KQV` reachable if the current
projection genuinely fails. The stronger residual-sufficiency verdict still
requires both computed reference and computed C to replay every stage exactly.
Unit tests verify that apparatus-only branches before `evidence()`, so the
qualification cannot open frozen payloads while merely reporting zero
counters.

## Authorized next step

After an exact-source commit, perform one scientific zero-graph invocation.
It may open the accepted artifact and immutable payloads but may not run an
attention producer, donor graph, or reference graph. Readiness alone does not
decide whether current `l_out-0` is sufficient or whether the output
projection fails exact `kqv_out-1`.
