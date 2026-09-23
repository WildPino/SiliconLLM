# STRAT-01 post-F16 block-0 terminal-component cross-input result

**Verdict:** `POST_F16_BLOCK0_FFN_OUT_RESIDUAL_SUFFICIENT`

The single scientific diagnostic authorized by the frozen
[protocol](STRAT_01_GIGACHAT31_ENGINE_POST_F16_BLOCK0_TERMINAL_COMPONENT_CROSS_INPUT_PROTOCOL_20260923.md)
completed at producer commit
`03df503bcaf3e5d7d9817a03007382e9ce03a9a8`. It reports no error, one
diagnostic invocation, and zero donor/reference graph executions.

Raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_post_f16_block0_terminal_component_cross_input_20260923/`

- `adjudication.json` SHA-256:
  `4d30ffb5d79d0682b1522157caf27bded8edd655bfe4e7bdf8c5353a59d515fb`;
- compiled binary SHA-256:
  `890f6502e5dfd88a38ed1ff967184a02772ea34af305040b83bfa0b0ab55b0db`;
- predecessor recovery SHA-256:
  `766061a9f54b601b8f54f08b6aaf6d533928aa19638f04ac2dd6cc23f865c09b`;
- orchestration / diagnostic durations: `157.449` / `28.386` seconds.

Those durations are diagnostic costs, not emitted-token throughput.

## Scientific outcome

The newly produced current `ffn_inp-0` is byte-identical to the immutable
reference payload:
`baa389195bff42ff92593d92650ed9defdd1b1382469e5dde3516dfa82b76ec1`.
Replacing only current `ffn_out-0` with the reference output therefore
reconstructs reference `l_out-0` byte-for-byte and makes all 32 complete
current layer-1 checkpoints pass.

Keeping the reference-identical `ffn_inp-0` and substituting the new current
`ffn_out-0` reproduces the post-F16 current terminal state and the downstream
failure. Selected metrics are:

| arm | checkpoint | NRMSE | result |
|---|---|---:|---|
| current `ffn_inp` + reference `ffn_out` | `l_out-0` | `0` | PASS |
| current `ffn_inp` + reference `ffn_out` | routed `ffn_moe_down-1` | `8.86595e-8` | PASS |
| current `ffn_inp` + reference `ffn_out` | shared `ffn_shexp-1` | `7.36996e-8` | PASS |
| current `ffn_inp` + reference `ffn_out` | `l_out-1` | `1.57455e-7` | PASS |
| reference `ffn_inp` + current `ffn_out` | `l_out-0` | `6.35179e-8` | PASS |
| reference `ffn_inp` + current `ffn_out` | `kqv_out-1` | `5.03513e-5` | PASS |
| reference `ffn_inp` + current `ffn_out` | routed `ffn_moe_down-1` | `0.00376095` | **FAIL** |
| reference `ffn_inp` + current `ffn_out` | shared `ffn_shexp-1` | `0.00237204` | **FAIL** |
| reference `ffn_inp` + current `ffn_out` | `l_out-1` | `0.00176698` | **FAIL** |

The first failure is routed `ffn_moe_down-1`. The current component hashes
are:

| component | SHA-256 |
|---|---|
| `ffn_inp-0` | `baa389195bff42ff92593d92650ed9defdd1b1382469e5dde3516dfa82b76ec1` |
| `ffn_out-0` | `7d0037ab06440493b146e9839d5c89ffa8fdea2b952389f12a38ccea2a94e684` |
| `l_out-0` | `7fb5fe52df684e4af82413004f6845133528241e3b030530d41595f74298bdc4` |

## Identity bridge to the next boundary

The new current `ffn_out-0` is byte-identical to the output already produced
in the valid earlier SwiGLU cell by **reference gate + reference up**, the
unchanged production C expression, and the unchanged current Q6 down
projection. Its terminal sum with reference `ffn_inp-0` is likewise the exact
new current `l_out-0`:

| earlier all-reference operand chain | frozen SHA-256 | new post-F16 identity |
|---|---|---|
| generated C-expression SwiGLU | `ac2b46108a7e18ddebdb1b3a26462bca0faebea536361a5ca4069c4fa7ddcf40` | upstream witness |
| current-Q6 `ffn_out-0` | `7d0037ab06440493b146e9839d5c89ffa8fdea2b952389f12a38ccea2a94e684` | exact current `ffn_out-0` |
| terminal sum | `7fb5fe52df684e4af82413004f6845133528241e3b030530d41595f74298bdc4` | exact current `l_out-0` |

This is stronger than a metric match: exact reference gate/up operands are
already sufficient to reproduce the remaining current failure input. A new
gate/up split would therefore repeat a coordinate that is not necessary for
the residual. The earlier Q6 and SwiGLU cells only propagated through the old
partial layer-1 attention surface; they did not test the now-sensitive full
post-F16 layer 1.

## Replay and controls

- both homogeneous block-0 sums reconstruct byte-for-byte;
- all 32 checkpoints of the current-output hybrid replay the frozen current
  trajectory;
- all 32 checkpoints of the reference-output hybrid pass;
- exact helper accounting passes at `5760` QK and `655360` value calls in
  `pinned-generic-f64` mode;
- one-byte mutations of all four admitted inputs are refused and origin-label
  swapping is rejected;
- token-6 negation rejects at NRMSE `1.1749282`;
- row-0/7 swapping rejects at NRMSE `1.0468528`;
- all seven source controls pass.

## Interpretation and stop rule

The post-F16 block-0 attention/residual stream is closed: its output is exact
reference. The only sufficient block-0 residual is in `ffn_out-0` production.
Do not repeat either terminal-component split, block 0, Q4 propagation, or the
gate/up hybrid experiment.

The next admissible cell is the separately frozen
[post-F16 block-0 FFN operator-chain split](STRAT_01_GIGACHAT31_ENGINE_POST_F16_BLOCK0_FFN_OPERATOR_CROSS_INPUT_PROTOCOL_20260923.md).
It reuses exact reference gate/up and SwiGLU witnesses to distinguish the
current SwiGLU-expression residual from the unchanged Q6 down residual under
the complete post-F16 layer-1 amplifier. It executes no block-0 graph.

No tokenizer, generation, quality, RAM, or rate claim follows. No
`SPEED_LEDGER.md` update is due.
