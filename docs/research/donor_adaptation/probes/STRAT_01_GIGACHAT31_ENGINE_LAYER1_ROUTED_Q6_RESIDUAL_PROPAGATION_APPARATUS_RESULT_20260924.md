# STRAT-01 GigaChat 3.1 layer-1 routed-Q6 propagation apparatus result

**Canonical verdict:** `APPARATUS_READY_NO_DONOR_EXECUTION`

The model-free apparatus for downstream propagation of the immutable old
Q6(reference-SwiGLU) payload qualified on its first attempt. It opened neither
the accepted artifact nor any scientific payload and executed no Q6, graph,
or diagnostic arm.

## Qualification

- Clang C11 `-O3 -mavx2 -mfma` build passed;
- new routed-Q6 propagation self-test: 5/5 checks;
- inherited routed-MoE component self-test: 7/7;
- inherited FFN-output component and terminal self-tests: 6/6 each;
- inherited closed layer-2 self-tests: 5/5, 4/4, 7/7, and 10/10;
- legacy engine: 73,024 kernel checks plus exponential check;
- Python tests: 8/8;
- diagnostic invocations: 0; Q6 executions: 0; donor/reference graphs: 0;
- accepted artifact opened: false; errors: none.

Canonical raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer1_routed_q6_residual_propagation_apparatus_20260924/`

| record | SHA-256 |
|---|---|
| `adjudication.json` | `26a6e0952099c72981204474dd298968a0bea9f6dfb061a7966039fd34989eef` |
| C binary | `128326f23700ad020227412eecb2a85e40b5c30f0fe6e40e1f60ab0500fa6c33` |

The apparatus observed HEAD
`4da272e4b4ce75bb34ca66bf741bdfd302fb999e` and records the exact
uncommitted source hashes. Commit those exact sources before the sole permitted
zero-Q6, zero-graph scientific invocation.

No Q6 execution or repair, causal verdict, tokenizer, later-layer, quality,
RAM, or rate claim is made.
