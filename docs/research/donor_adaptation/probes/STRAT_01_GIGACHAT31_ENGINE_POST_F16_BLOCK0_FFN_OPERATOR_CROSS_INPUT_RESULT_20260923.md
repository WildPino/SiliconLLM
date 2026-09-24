# STRAT-01 post-F16 block-0 FFN operator-chain cross-input result

**Verdict:** `POST_F16_BLOCK0_SWIGLU_EXPRESSION_RESIDUAL_SUFFICIENT`

The sole scientific diagnostic authorized by the frozen
[protocol](STRAT_01_GIGACHAT31_ENGINE_POST_F16_BLOCK0_FFN_OPERATOR_CROSS_INPUT_PROTOCOL_20260923.md)
completed at producer commit
`c772f29f6f4297c9694a38a80e0317b2cdb2eb38`. It reports no error, exactly
one diagnostic invocation, and zero donor/reference graph executions.

Raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_post_f16_block0_ffn_operator_cross_input_20260923/`

- `adjudication.json` SHA-256:
  `ad03099c59a2bb1ee30bb0063c2b4b9f393b1551ec0e19931d0bf518dd1095eb`;
- compiled binary SHA-256:
  `441ea9a4a6cec7d81d4d7a5d8cbf12d454566376ecd4e594cfbc757ed011a1fd`;
- predecessor adjudication SHA-256:
  `4d30ffb5d79d0682b1522157caf27bded8edd655bfe4e7bdf8c5353a59d515fb`;
- orchestration / diagnostic durations: `164.317` / `30.255` seconds.

Those durations are diagnostic costs, not emitted-token throughput.

## Scientific outcome

The Q6-only arm sends the immutable captured reference SwiGLU payload through
the unchanged current Q6 down projection and the complete post-F16 layer 1.
It passes all 32 checkpoints. The expression+Q6 arm differs only by replacing
that captured payload with the output of the production C expression on the
same exact reference gate/up operands. It exactly replays the predecessor
current trajectory and first fails at routed `ffn_moe_down-1`.

Selected metrics are:

| arm | checkpoint | NRMSE | result |
|---|---|---:|---|
| captured reference SwiGLU + current Q6 | `l_out-0` | `5.48084e-8` | PASS |
| captured reference SwiGLU + current Q6 | `attn_norm-1` | `5.84100e-8` | PASS |
| captured reference SwiGLU + current Q6 | routed `ffn_moe_down-1` | `7.95119e-5` | PASS |
| captured reference SwiGLU + current Q6 | `ffn_moe_out-1` | `1.35479e-5` | PASS |
| captured reference SwiGLU + current Q6 | `l_out-1` | `9.31051e-6` | PASS |
| production expression + current Q6 | `l_out-0` | `6.35179e-8` | PASS |
| production expression + current Q6 | routed `ffn_moe_down-1` | `0.00376095` | **FAIL** |
| production expression + current Q6 | `ffn_moe_out-1` | `0.00247392` | **FAIL** |
| production expression + current Q6 | `l_out-1` | `0.00176698` | **FAIL** |

The arm identities are:

| payload | captured-reference arm | production-expression arm |
|---|---|---|
| `ffn_swiglu-0` | `de9245a2f6f60e8d7511d03fe981e6b55ef21a8634d5071dd36727a1a5950cef` | `ac2b46108a7e18ddebdb1b3a26462bca0faebea536361a5ca4069c4fa7ddcf40` |
| `ffn_out-0` | `f52e0e5bdb22a17ae4a9e0ee9e0c6ffc6bb485b3e61e3f821039eae38697d6ce` | `7d0037ab06440493b146e9839d5c89ffa8fdea2b952389f12a38ccea2a94e684` |
| `l_out-0` | `a71c814dc43360f73b49cb165e7cdf7128f2501015685ff16e7e524dd1f05f11` | `7fb5fe52df684e4af82413004f6845133528241e3b030530d41595f74298bdc4` |

The exact reference gate/up inputs are unchanged at
`5c30c0ada2e96ce43a92b22057b4d593b0b9ca342feade7092bfec084e8c6c2a`
and
`2b608af95db90fcde83c29946ba2ac680fb6465cbaaf518fc12874ab510b29e4`.
Therefore the only changed scientific coordinate between the passing and
failing arms is the frozen C-expression SwiGLU residual.

## Replay and controls

- the expression+Q6 arm byte-replays all 32 predecessor checkpoints;
- both Q6 output/start identities and all four admitted input identities pass;
- exact helper accounting passes at `4608` QK and `524288` value calls;
- exactly four block-0 Q6 down arms ran;
- one-byte mutations of all four inputs are refused, and arm-origin swapping
  is rejected;
- token-6 negation rejects at terminal NRMSE `0.356681`;
- row-0/7 swapping rejects at terminal NRMSE `0.672545`;
- all eight source controls pass;
- 158 Python tests and 20 C self-tests pass before the diagnostic.

## Interpretation and stop rule

The unchanged Q6 down projection is not independently sufficient to cause the
post-F16 layer-1 failure: with captured reference SwiGLU, every checkpoint
passes. The production C SwiGLU expression residual is sufficient, despite
its very small local error, because it is amplified by the already-qualified
downstream path.

Do not repeat Q6, the gate/up split, either terminal-component cell, block 0,
Q4 propagation, F16 propagation, or layer 1. The next admissible coordinate
must split the production SwiGLU expression's numerical semantics on the same
immutable reference gate/up tensors, then use the same Q6 and complete
post-F16 layer-1 amplifier. It must be frozen before implementation and must
not execute a donor/reference graph.

No repair, later-layer, tokenizer, logits, generation, quality, RAM, or rate
claim follows. No `SPEED_LEDGER.md` update is due.
