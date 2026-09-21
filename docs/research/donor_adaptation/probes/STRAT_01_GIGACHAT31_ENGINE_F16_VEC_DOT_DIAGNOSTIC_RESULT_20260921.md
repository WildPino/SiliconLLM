# STRAT-01 GigaChat 3.1 F16 dot diagnostic result

**Verdict:** `F16_CONVERSION_ONLY_SUFFICIENT`
**Date:** 21 September 2026
**Scope:** zero-donor offline attribution of the two attention dots; no production integration, quality, RAM, or speed claim

## Result

Exact F32-to-F16 conversion of the right operand closes both attention-stage
failures.  Scalar F16×F16 accumulation is already well inside the frozen gate;
the pinned F16 vector dot reproduces each captured target bit-for-bit.

| Stage and arm | NRMSE | normalized maximum | Gate |
|---|---:|---:|---:|
| QK, old F16×F32 | `2.125561e-4` | `2.419999e-4` | FAIL |
| QK, scalar F16×F16 | `1.762007e-7` | `3.905095e-7` | PASS |
| QK, pinned F16 vector dot | `0` | `0` | PASS |
| softmax×V, old F16×F32 | `1.527725e-4` | `2.678969e-4` | FAIL |
| softmax×V, scalar F16×F16 | `4.083053e-8` | `8.447008e-8` | PASS |
| softmax×V, pinned F16 vector dot | `0` | `0` | PASS |

All frozen controls pass: source identity, zero donor executions, exact
conversion bytes, nonzero vector outputs, and causal query/probability
mutations.  The mutated QK and value arms fail at NRMSE `8.408701e-6` and
`4.507730e-3`, respectively.

## Provenance

- Accepted apparatus commit: `82c3e0a16f2c08c46706aad86d2af0066969975c`.
- Canonical adjudication:
  `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_f16_vec_dot_repair2_20260921/adjudication.json`,
  SHA-256 `d9691cbcd569681f970e5e4af553cec75a7f86ec47f66acbec2a111288239e0a`.
- QK vector output SHA-256 equals captured target:
  `121c689214a7bcf2e709b8de896df14aabcb8fb00580bab297231b8887a3b009`.
- Value vector output SHA-256 equals captured target:
  `541183b4eb5992cfcfc711fe3c9f179ab41685106ba9cfe2aa7099f45f821312`.
- Project and pinned conversion stream SHA-256:
  `7cdec0744fb5c9dcc6225dd70a0c952283128d4f8668d9861131b868379defbc`.

## Preserved VOID history

The precondition launch failed exact conversion.  The first post-converter
launch produced all-zero vector arms and used a non-causal negative-control
predicate.  Both are apparatus VOIDs described in the amended protocol and
must not be used as scientific results.

## Consequence

The attention repair is precisely to convert query and normalized softmax
probabilities to binary16 before their scalar products with the F16 cache.
Pinned SIMD reduction order is not required to meet the frozen gate.  Freeze
production integration and a full block-0 confirmation before changing the C
engine.  Do not repeat this diagnostic.
