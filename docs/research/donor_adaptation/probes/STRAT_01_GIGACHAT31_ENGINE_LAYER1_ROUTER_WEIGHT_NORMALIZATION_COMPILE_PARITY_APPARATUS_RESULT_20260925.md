# STRAT-01 layer-1 router-weight normalization compile-parity apparatus result

**Date:** 2026-09-25

**Status:** `APPARATUS_READY_NO_PAYLOAD_EXECUTION`

**Disposition at pause:** saved but not executed scientifically; see the
[pause record](../PAUSE_20260925.md). The final next-step instruction below is
suspended. Provenance clarification: the repaired self-test fixture was
selected after exploratory inspection of preserved real operands. The zero
payload counters describe the apparatus invocation only, not that external
inspection. This is an engineering regression fixture, not independent
held-out evidence for normalization parity.

The repaired apparatus qualifies the local cell frozen in the
[protocol](STRAT_01_GIGACHAT31_ENGINE_LAYER1_ROUTER_WEIGHT_NORMALIZATION_COMPILE_PARITY_PROTOCOL_20260925.md).
The candidate and generic C++ oracle compile separately; synthetic
F64-sum→F32 parity, clamp-boundary behavior and all registered Python tests
pass. The explicit truncated-file command rejects with nonzero status.

Qualifying record:

- raw directory:
  `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer1_router_weight_normalization_compile_parity_apparatus_repair1_20260925/`;
- adjudication SHA-256:
  `ce83b76f0624448502a1e88962e6b4894b2d4efb5c7dd0a8eb68e76c6b45a929`;
- observed pre-commit HEAD:
  `449e22142657631ac9c5ed0708ba45a3b30cb2af`;
- four Python tests, both compile steps, link and self-test pass;
- truncated-file control rejects;
- diagnostic invocations, payload reads, model reads, producer invocations,
  donor graphs and reference graphs: all zero;
- duration: 1.500 s;
- binary SHA-256:
  `60c3dba6a226a24f06a684dff08ac3130b6f78cda1b027513d4080c91dc00c45`.

Source identities include runner
`eae25cf56365ad4047762e2f47ec459501f3addedf8250433ed6d33ab59c475d`,
probe `74a4ad7b94b6798ca3bb54b0f7a637288c29952170f0c78ab4d9cfd14d84a516`,
oracle `0d47a8363009f6dd19f6bc51b5641cef8bd8b5f9ae2a97cf63e5a543f1296673`,
and candidate header
`622cf4ecc43813e042d82c2949fa66003a6af7c8c9355984f7e9965e9761d447`.

The initial apparatus directory is preserved as VOID, adjudication SHA-256
`7b99e993b4f999785fbc872bc10f66912dc06b356d7f38a9920e9ea5d879ea65`.
Its arbitrary synthetic fixture did not distinguish every wrong arithmetic
arm; no real payload or model was read. Repair 1 changed only that fixture and
the output directory.

Commit these exact sources, then run one local diagnostic on the 32 preserved
weights. No model or graph is authorized. This apparatus itself establishes no
normalization parity, production, quality, memory or rate result.
