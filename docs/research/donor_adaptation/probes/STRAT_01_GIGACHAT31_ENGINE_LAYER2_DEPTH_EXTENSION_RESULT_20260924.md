# STRAT-01 layer-2 depth-extension result

**Canonical verdict:** `FAIL_ENGINE_LAYER2_DEPTH_EXTENSION_RECOVERED`

The accepted engine remains faithful through the layer-2 attention inputs,
but the first frozen failure is `kqv_out-2`. The result is canonicalized by
the model-free [offline-recovery protocol](STRAT_01_GIGACHAT31_ENGINE_LAYER2_DEPTH_EXTENSION_OFFLINE_RECOVERY_PROTOCOL_20260924.md)
without rerunning either producer.

Canonical raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer2_depth_extension_offline_recovery1_20260924/`

- canonical `adjudication.json` SHA-256:
  `0b5d34ea652676c8ec04b8a87f89ff1e08757c370ebcfdab939871ca6558b46e`;
- recovery commit:
  `9c3c15180f1962e93ce0aad71c45ebf7f99f6409`;
- immutable source-run adjudication SHA-256:
  `65cf9f30bdf4c2028f5c13f91cf4dfc44d28b3c04d07bc254c395e0c74e0693c`;
- source-run commit:
  `9360378657a74f764b0374424f3cb325c61fd764`.

## Execution and validity

The source run used exactly one pinned-reference invocation and one C-engine
invocation. Each completed `prefill8` and `cached7p1`, for two reference and
two C graphs. Recovery added zero producer invocations and zero graphs.

All identity, source, schema, manifest, path-containment, type, shape,
operation, count, predecessor, and execution-accounting controls pass:

- 52/64 C-versus-reference checkpoints pass;
- 64/64 prefill/cached token-7 continuity comparisons pass;
- 9/9 three-layer cache comparisons pass;
- 8/8 immutable predecessor-reference comparisons are exact;
- helper counts are exactly `6912` QK and `786432` value invocations;
- all twelve source controls pass;
- all ten causal mutations reject after the preregistered branch-local repair.

The original aggregate `omit_shared_expert` mutation is retained as a
diagnostic: NRMSE `0.0009646364838712824`, normalized maximum
`0.0004489167347438313`, so it was masked by the aggregate output scale. The
replacement compares zero with `ffn_shexp-2` under the same unchanged limits
and rejects at NRMSE/normalized maximum `1.0/1.0` in both schedules.

## Numerical boundary

Both schedules produce identical metrics and fail the same six checkpoints:

| checkpoint | NRMSE | normalized max | frozen NRMSE limit | result |
|---|---:|---:|---:|---|
| `kqv_out-2` | `0.003014631769821054` | `0.004362653758957389` | `0.002` | fail, first |
| `ffn_inp-2` | `0.0014701406975772728` | `0.0006663977196261136` | `0.001` | fail |
| `ffn_moe_gate-2` | `0.0022977119747694093` | `0.00030269323793067276` | `0.002` | fail |
| `ffn_up-2` | `0.004580183921816701` | `0.0031372537559942568` | `0.002` | fail |
| `ffn_swiglu-2` | `0.0045149498569080765` | `0.0016160748039858776` | `0.002` | fail |
| `ffn_shexp-2` | `0.008183208718458058` | `0.004774060308168831` | `0.002` | fail |

All preceding attention surfaces pass in frozen order: `l_out-1`,
`attn_norm-2`, `q-2`, `kv_cmpr_pe-2`, `k_pe-2`, `kv_cmpr-2`, `q_pe-2`,
`q_nope_absorbed_perm-2`, `Qcur-2`, `Kcur-2`, and `Vcur-2`. Layer-2 routing,
routed up/SwiGLU/down/weighting/output, final FFN sum, and `l_out-2` also pass.
The later isolated failures therefore do not move the first boundary away
from attention output composition/projection.

## Interpretation and next action

Depth extension itself is wired correctly: schedule equivalence, all caches,
the inherited start state, routing, and terminal layer output remain inside
their gates. The first unsupported transfer is the layer-2 attention output,
whose error is larger than at the accepted layer-1 boundary despite passing
Q/K/V inputs.

Do not rerun Rung-2C, the layer-2 reference, or the layer-2 C producer. Freeze
a separate diagnostic at `kqv_out-2` and first reuse the immutable Q/K/V,
cache, weight, and output payloads. Only a genuinely missing intermediate may
authorize a new producer under a new protocol.

No claim follows for layers 3–25, tokenizer, logits, generation, model
quality, RAM, or throughput. `SPEED_LEDGER.md` is unchanged.
