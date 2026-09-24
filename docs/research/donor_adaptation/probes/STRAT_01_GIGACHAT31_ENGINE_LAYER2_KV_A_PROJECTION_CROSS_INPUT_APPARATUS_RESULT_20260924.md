# STRAT-01 GigaChat 3.1 layer-2 KV-A projection apparatus result

**Canonical verdict:** `APPARATUS_READY_NO_DONOR_EXECUTION` (`repair1`)

The Addendum-A path-only repair is qualified without opening the accepted GGUF
or any evidence payload. The initial apparatus and subsequent pre-execution
VOID remain provenance, but are superseded for execution by `repair1`.

## Qualification

- Clang C11 `-O3 -mavx2 -mfma` build passed;
- new KV-A projection self-test: 4/4 checks;
- inherited KV RMSNorm self-test: 7/7;
- inherited KV-partition self-test: 10/10;
- legacy engine: 73,024 kernel checks plus exponential check;
- Python tests: 7/7, including projection `(8, 576)` metric shape and the
  predecessor-root regression guard;
- diagnostic invocations: 0; donor/reference graphs: 0;
- accepted artifact opened: false; errors: none.

The helper uses the production Q4_K KV-A projection and the already closed
RMSNorm, cache/attention, and V-B operators. It contains no replacement
attention kernel and makes no claim from the accepted model.

Canonical raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer2_kv_a_projection_cross_input_apparatus_repair1_20260924/`

| record | SHA-256 |
|---|---|
| `adjudication.json` | `6173c1876d202e2cbebf34a77abfe122c98d4166f6cdf89f56f37cde0b9a7acd` |
| C binary | `8c3d8006c6d0a40d27030f860088d49fe86be3a8129e6949d635b0d3c750eaba` |

The canonical apparatus observed HEAD
`1ca153d7133e4de05bb1bc8cc0e57c0b68f7671a` and the uncommitted Addendum-A
repair source hashes recorded in its adjudication. Commit those exact sources
before the one permitted zero-graph scientific invocation.

The superseded initial apparatus is preserved at the same path without the
`repair1` suffix; its adjudication SHA-256 is
`2863723bab80e928aaecff889c71fe6e4be00c6d94a72b555ea811bd48c5ecff`.
The later raw VOID is bound in protocol Addendum A. Neither executed a
diagnostic arm.

No causal, repair, quality, RAM, or rate claim is made.
