# STRAT-01 post-F16 block-0 SwiGLU SSE2-semantics result

**Verdict:** `POST_F16_BLOCK0_SWIGLU_SSE2_EXACT_REPAIR`

The sole scientific diagnostic authorized by the frozen
[protocol](STRAT_01_GIGACHAT31_ENGINE_POST_F16_BLOCK0_SWIGLU_SSE2_SEMANTICS_PROTOCOL_20260924.md)
completed at producer commit
`005e604ea368585affb8ec804bdd7fc58d0e290e`. It reports no error, exactly
one diagnostic invocation, and zero donor/reference graph executions.

Raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_post_f16_block0_swiglu_sse2_semantics_20260924/`

- `adjudication.json` SHA-256:
  `46f127215e4bfc00b80ba4eb151d20c294482d2cb5f3c976616e953d21852c0e`;
- compiled binary SHA-256:
  `0a63f04aa515b2de30fd397bdb6e4f44e7e419f2da1d251744460e154740a17d`;
- predecessor adjudication SHA-256:
  `ad03099c59a2bb1ee30bb0063c2b4b9f393b1551ec0e19931d0bf518dd1095eb`;
- orchestration / diagnostic durations: `446.351` / `46.709` seconds.

Those durations are diagnostic costs, not emitted-token throughput.

## Scientific outcome

On the immutable reference gate/up tensors, the pinned four-lane no-FMA SSE2
polynomial produces a byte-identical copy of the captured reference SwiGLU
tensor. Its output then passes unchanged current Q6 down and every one of the
32 complete post-F16 layer-1 checkpoints. The scalar-libm replay remains the
exact predecessor trajectory and first fails at routed `ffn_moe_down-1`.

Selected metrics are:

| arm | checkpoint | NRMSE | result |
|---|---|---:|---|
| scalar `expf` replay | `ffn_swiglu-0` | `4.99848e-8` | PASS locally |
| scalar `expf` replay | routed `ffn_moe_down-1` | `0.00376095` | **FAIL** |
| scalar `expf` replay | `ffn_moe_out-1` | `0.00247392` | **FAIL** |
| scalar `expf` replay | `l_out-1` | `0.00176698` | **FAIL** |
| pinned no-FMA SSE2 | `ffn_swiglu-0` | `0` | byte-exact |
| pinned no-FMA SSE2 | `ffn_out-0` | `8.34832e-8` | PASS |
| pinned no-FMA SSE2 | routed `ffn_moe_down-1` | `7.95119e-5` | PASS |
| pinned no-FMA SSE2 | `ffn_moe_out-1` | `1.35479e-5` | PASS |
| pinned no-FMA SSE2 | `l_out-1` | `9.31051e-6` | PASS |

The arm identities are:

| payload | scalar-libm replay | pinned SSE2 candidate |
|---|---|---|
| `ffn_swiglu-0` | `ac2b46108a7e18ddebdb1b3a26462bca0faebea536361a5ca4069c4fa7ddcf40` | `de9245a2f6f60e8d7511d03fe981e6b55ef21a8634d5071dd36727a1a5950cef` |
| `ffn_out-0` | `7d0037ab06440493b146e9839d5c89ffa8fdea2b952389f12a38ccea2a94e684` | `f52e0e5bdb22a17ae4a9e0ee9e0c6ffc6bb485b3e61e3f821039eae38697d6ce` |
| `l_out-0` | `7fb5fe52df684e4af82413004f6845133528241e3b030530d41595f74298bdc4` | `a71c814dc43360f73b49cb165e7cdf7128f2501015685ff16e7e524dd1f05f11` |

The SSE2 `ffn_swiglu-0` hash is exactly the immutable reference hash. The
candidate `ffn_out-0` and `l_out-0` hashes also exactly reproduce the prior
captured-reference-SwiGLU arm, so the repaired trajectory is independently
linked to the preceding operator-chain experiment.

## Replay and controls

- the scalar-libm arm byte-replays all 32 predecessor checkpoints;
- one-byte mutations of gate, up, captured SwiGLU, and `ffn_inp-0` are
  refused;
- arm-origin swapping is rejected;
- exactly four Q6 down arms ran and exact helper accounting passes;
- token-6 gate negation and up-row 0/7 swapping both reject downstream;
- all eleven source controls and all pinned reference-source/build checks
  pass;
- 165 Python tests and 21 C self-tests pass before the diagnostic.

## Interpretation and stop rule

The scalar `expf` versus pinned `ggml_v_expf` evaluation semantics are the
sufficient cause of the measured post-F16 block-0 SwiGLU residual. The exact
reference-build semantics repair the diagnostic coordinate without changing
gate/up inputs, Q6, block 0 composition, or layer 1.

Do not repeat this diagnostic, any predecessor split, either graph, or an
AVX2/FMA sweep. This result qualifies the numerical primitive; it does not by
itself prove that the ordinary accepted-artifact production path uses it.
The only next engine-fidelity coordinate is a separately frozen production
integration cell: install these exact no-FMA four-lane semantics in the
normal SwiGLU path, qualify the model-free apparatus, then allow one accepted
GGUF execution through block 0 and the complete layer-1 checkpoint chain.

No later-layer, tokenizer, logits, generation, quality, RAM, or rate claim
follows. No `SPEED_LEDGER.md` update is due.
