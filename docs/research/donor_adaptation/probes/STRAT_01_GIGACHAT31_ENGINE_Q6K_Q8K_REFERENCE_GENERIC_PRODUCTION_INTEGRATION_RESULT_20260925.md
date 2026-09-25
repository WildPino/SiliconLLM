# STRAT-01 Q6 reference-generic production-integration result

**Date:** 2026-09-25

**Verdict:** `FAIL_ENGINE_Q6_REFERENCE_GENERIC_PRODUCTION_INTEGRATION`

## Valid recovered execution

The sole producer invocation from commit
`54b4c381a214eb7a14de8e0cb5e1e40953ccd8f5` completed both frozen schedules
before the [mechanical VOID](STRAT_01_GIGACHAT31_ENGINE_Q6K_Q8K_REFERENCE_GENERIC_PRODUCTION_INTEGRATION_VOID1_20260925.md).
The preregistered offline recovery at commit
`a2dce6a8979fb9d290e2001a10ee1992840a7276` validated and adjudicated those
immutable outputs without executing a producer or graph.

- Raw recovery directory:
  `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_q6k_q8k_reference_generic_production_integration_offline_recovery1_20260925/`.
- Adjudication SHA-256:
  `cb2a2d9cbfc80d8eb08dfc32d0f850fae717cf9a76525a6f81a9f600269a3b93`.
- Run-manifest SHA-256:
  `50d54e9013db59a1654773c96d676085be177562c6ef0440006d1e6a2ddd0856`.
- Source production invocations: 1; source donor graph completions: 2.
- New production invocations: 0; new donor graphs: 0; reference graphs: 0.
- Errors: none; all source, artifact, log, manifest, payload, count, and
  configuration bindings passed.

## Scientific result

The production Q6 repair succeeds exactly at its intended block-0 boundary.
Both schedules emit the required `l_out-0` SHA-256
`385073c91f472dd9ffc1c86bcb63c6ed50256a6d5645e240d61ccdb613d814aa`;
the old hash is displaced. Every checkpoint from `l_out-0` through
`ffn_norm-1` is byte-exact, including complete layer-1 attention and its first
residual. This closes the shared Q6 dispatch integration and all upstream
block-0/attention coordinates.

The exact full-layer gate nevertheless fails at the first remaining local
operator, the F32 router projection:

| checkpoint | changed / total | NRMSE | normalized maximum |
|---|---:|---:|---:|
| `ffn_moe_logits-1` | 473 / 512 | `5.60101e-7` | `1.15498e-6` |
| `ffn_moe_probs-1` | 469 / 512 | `7.23255e-7` | `4.19871e-7` |
| `ffn_moe_probs_biased-1` | 11 / 512 | `1.48686e-8` | `9.73050e-8` |
| `ffn_moe_weights_norm-1` | 28 / 32 | `4.28524e-7` | `4.91379e-7` |

The selected `ffn_moe_topk-1` IDs remain byte-exact. Consequently all 32
selected routed `ffn_moe_up-1` and `ffn_moe_gate-1` projections are byte-exact.
The shared `ffn_up-1` and `ffn_gate-1` projections are also byte-exact.

Two independent scalar-SwiGLU residuals then appear on exact operands:

| checkpoint | changed / total | NRMSE | normalized maximum |
|---|---:|---:|---:|
| routed `ffn_moe_swiglu-1` | 10,241 / 40,960 | `4.51508e-8` | `2.17306e-8` |
| shared `ffn_swiglu-1` | 2,561 / 10,240 | `6.08697e-8` | `9.70064e-8` |
| final `l_out-1` | 9,830 / 12,288 | `1.50674e-7` | `6.15853e-8` |

Both schedules have identical counts and outcomes. Overall, 38/64 checkpoint
comparisons are byte-exact and 26/64 fail the deliberately exact gate. All
64/64 candidate/reference continuity comparisons pass, all six cache
comparisons pass, helper counts are exact, and all ten causal negative controls
reject.

## Interpretation and stop rule

This is a valid scientific FAIL of complete exact production integration, not
a regression of the Q6 repair. It proves a stronger new frontier: the accepted
artifact is byte-exact through complete layer-1 attention and `ffn_norm-1`.
The residual local coordinates are the F32 router reduction and layer-1
SwiGLU numerical semantics. Their errors are small and top-k stable, but the
frozen gate was exact and must not be relaxed after seeing them.

Do not repeat the producer, offline recovery, Q6 compile parity, block 0,
layer-1 attention, routing selection, Q4 projections, or Q6 down attribution.
The next distinct cell is the separately frozen
[layer-1 numerical-primitives compile-parity protocol](STRAT_01_GIGACHAT31_ENGINE_LAYER1_NUMERICAL_PRIMITIVES_COMPILE_PARITY_PROTOCOL_20260925.md).
It tests router reduction and routed/shared SwiGLU independently on preserved
exact operands with zero graphs. No rate, quality, generation, later-layer, or
RAM claim follows, and `SPEED_LEDGER.md` does not change.
