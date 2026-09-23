# STRAT-01 post-F16 layer-1-start cross-input result

**Verdict:** `POST_F16_BLOCK0_TERMINAL_RESIDUAL_SUFFICIENT`

The single scientific C diagnostic authorized by the frozen
[parent protocol](STRAT_01_GIGACHAT31_ENGINE_POST_F16_LAYER1_START_CROSS_INPUT_PROTOCOL_20260923.md)
completed at producer commit
`8f03fa3963723fc2083cc2d608437d1a487c9666`. Its external runner initially
preserved a VOID because it expected twice the correct helper-call totals.
The separately frozen
[offline recovery](STRAT_01_GIGACHAT31_ENGINE_POST_F16_LAYER1_START_CROSS_INPUT_RECOVERY_PROTOCOL_20260923.md)
then adjudicated the unchanged hash-bound output at commit
`ea9a68453e14fb0a7dbfe0398403acd156d236a9`, without compiling or executing
the model or a graph.

Recovery record:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_post_f16_layer1_start_cross_input_recovery1_20260923/adjudication.json`

Its SHA-256 is
`766061a9f54b601b8f54f08b6aaf6d533928aa19638f04ac2dd6cc23f865c09b`.
The recovery reports no error, zero new diagnostic invocations, zero donor
graphs, and zero reference graphs. It inherits exactly one successful C
diagnostic invocation from the preserved raw VOID.

## Scientific outcome

Feeding exact reference `l_out-0` into the complete current layer 1 passes all
32 canonical checkpoints. There is no first failure. Selected reference-arm
NRMSE values are:

| checkpoint | NRMSE | result |
|---|---:|---|
| `l_out-0` | `0` | PASS |
| `kqv_out-1` | `0` | PASS |
| `ffn_inp-1` | `0` | PASS |
| routed `ffn_moe_down-1` | `8.8659536e-8` | PASS |
| shared `ffn_shexp-1` | `7.3699554e-8` | PASS |
| `ffn_out-1` | `2.0701810e-7` | PASS |
| `l_out-1` | `1.5745451e-7` | PASS |

The worst per-token `l_out-1` NRMSE is `3.1206569e-7` at token 1, far below
the frozen `1e-3` terminal gate. Thus the remaining post-F16 failures are not
caused by the current layer-1 attention, routing, expert, shared-expert, Q6,
or residual operators when given the exact block-0 terminal state.

## Replay and controls

- all 32 post-F16 current checkpoints replay byte-for-byte;
- exact helper accounting passes at `4608` QK and `524288` value calls in
  `pinned-generic-f64` mode;
- one-byte mutations of both start states are refused and arm-label swapping
  is rejected;
- negating reference token 6 rejects at NRMSE `1.1749282`;
- swapping reference rows 0 and 7 rejects at NRMSE `1.0468528`;
- both current schedules remained identical before this split;
- the raw VOID, C report, count report, producer binary, and producer commit
  all match the recovery protocol's frozen hashes.

## Interpretation and stop rule

The new post-F16 `l_out-0` residual is sufficient to explain the downstream
layer-1 failures. Layer 1 is closed again under the changed exact-Q4 plus
exact-F16 composition. Do not reopen F16, Q4, SwiGLU, Q6, routing, layer-1
operators, or any earlier layer-1-start cell.

The next admissible boundary is strictly upstream: decompose the post-F16
block-0 terminal sum `l_out-0 = ffn_inp-0 + ffn_out-0`. The old terminal
component split used the pre-exact-Q4/pre-exact-F16 C components and cannot
adjudicate this changed state. A new cell must produce the current post-F16
block-0 components once, prove their homogeneous sum replays the frozen new
`l_out-0`, cross them with the immutable reference components, and propagate
only the two hybrids through the already qualified current layer 1.

No tokenizer, generation, quality, RAM, or rate claim follows. No
`SPEED_LEDGER.md` update is due.
