# STRAT-01 post-F16 SwiGLU production-integration result

**Verdict:** `PASS_ENGINE_POST_F16_SWIGLU_PRODUCTION_INTEGRATION`

The sole production invocation authorized by the frozen
[protocol](STRAT_01_GIGACHAT31_ENGINE_POST_F16_SWIGLU_PRODUCTION_INTEGRATION_PROTOCOL_20260924.md)
completed at producer commit `cb0f5260044bc209175edd324e1580db272370a8`.
It reports no error, exactly one standard Rung-2C invocation, exactly two
donor graph schedules, and zero reference graph executions.

Raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_post_f16_swiglu_production_integration_20260924/`

- `adjudication.json` SHA-256:
  `d543c56d7e463cfec23e4fa3a7ec7fd5bfecbdccfdf84351ddf12085fcc3cc94`;
- compiled binary SHA-256:
  `a5e2da7bfca4814b2ed4412b3e08088d3bd12805e5acd3c5959bbb5209e55b87`;
- accepted GGUF SHA-256:
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`;
- predecessor SSE2-semantics adjudication SHA-256:
  `46f127215e4bfc00b80ba4eb151d20c294482d2cb5f3c976616e953d21852c0e`;
- orchestration / production durations: `161.127` / `30.230` seconds.

Those durations are fidelity-instrument costs, not emitted-token throughput.

## Scientific outcome

The ordinary accepted-artifact path now uses the shared no-FMA four-lane
SSE2 SwiGLU primitive in block 0. Both frozen schedules, `prefill8` and
`cached7p1`, pass every one of their 32 checkpoints against the immutable
reference. All 64 token-7 continuity comparisons also pass and all six cache
comparisons are exact.

The largest checkpoint NRMSE is `7.9511911e-5`, at routed
`ffn_moe_down-1`; the two terminal `l_out-1` comparisons are
`9.3105101e-6`. Ordered top-4 routing is exact. The repaired block-0 start
hash is `a71c814dc43360f73b49cb165e7cdf7128f2501015685ff16e7e524dd1f05f11`
in both schedules and displaces the scalar predecessor hash
`7fb5fe52df684e4af82413004f6845133528241e3b030530d41595f74298bdc4`.

| surface | passed | failed |
|---|---:|---:|
| reference checkpoints | 64 | 0 |
| prefill/cached continuity | 64 | 0 |
| cache comparisons | 6 | 0 |
| causal negative controls | 10 | 0 |
| source controls | 11 | 0 |

Helper accounting is exact: `4608` QK invocations and `524288` value
invocations under `pinned-generic-f64`. The production `CONFIG` declares
`block0=dense-swiglu-sse2-nofma4`; the diagnostic and production path both
delegate to the single shared implementation, while layer-1 SwiGLU remains
unchanged. The complete regression comprises 172 Python tests and 21 C
self-tests, all passing.

## Interpretation and stop rule

The proven SwiGLU numerical repair is installed in the normal
accepted-artifact path and closes production fidelity through block 0 and the
complete layer-1 boundary on the frozen eight-token trace. This is no longer
a diagnostic-only result.

Do not repeat Rung-2C, the SSE2 diagnostic, Q4/F16 parity or propagation,
gate/up, Q6, block-0 composition, layer-1 propagation, either schedule, or
the immutable reference producer. The next engine-fidelity work must change
the depth coordinate and be separately frozen before execution. It should
extend the accepted-artifact path beyond layer 1 while reusing these closed
operators and controls, not reopen them.

No later-layer, tokenizer, logits, generation, task-quality, RAM, or rate
claim follows. No `SPEED_LEDGER.md` update is due.
