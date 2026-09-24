# STRAT-01 GigaChat 3.1 layer-2 attention RMSNorm cross-input apparatus result

**Canonical verdict:** `APPARATUS_READY_NO_DONOR_EXECUTION` (`repair1`)

The narrowly repaired model-free apparatus is qualified without opening the
accepted GGUF or any frozen evidence payload. It tests only the boundary
`l_out-1 -> blk.2.attn_norm.weight` and the already closed KV-A/downstream
chain. It contains no replacement RMSNorm, projection, attention, or V-B
operator.

## Qualification

- Clang C11 `-O3 -mavx2 -mfma` build passed;
- new attention RMSNorm self-test: 5/5 checks;
- inherited KV-A projection self-test: 4/4;
- inherited KV RMSNorm self-test: 7/7;
- inherited KV-partition self-test: 10/10;
- legacy engine: 73,024 kernel checks plus exponential check;
- Python tests: 8/8, including the cleanup-symbol regression guard;
- diagnostic invocations: 0; donor/reference graphs: 0;
- accepted artifact opened: false; errors: none.

Canonical raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer2_attn_rmsnorm_cross_input_apparatus_repair1_20260924/`

| record | SHA-256 |
|---|---|
| `adjudication.json` | `42a3b0f4578c6aa24d2858534916346ebfb0203210f6399572d6198d6c40340e` |
| C binary | `fd452a782d754c2b691cf87c3723d2016e571f1aeb6e004e82a0beb55766f2ca` |

The apparatus observed HEAD
`7ab15daef7a3c6536d8243a6ec3d43841ec794eb` and records the exact uncommitted
source hashes. Commit those exact sources before the one permitted zero-graph
scientific invocation.

The superseded initial directory without the `repair1` suffix is preserved.
Its adjudication SHA-256 is
`aabeabefcdb1ef2621e18bc30b6c85ab322fc394e8cb710603990555c74c6256`.
It is VOID because the misspelled cleanup symbol prevented compilation; it
opened no artifact and executed no diagnostic or graph. Protocol Addendum A
limits the repair to that symbol and the immutable output directory.

No causal, repair, quality, RAM, or rate claim is made. Exactly one scientific
execution is now authorized after the source commit.
