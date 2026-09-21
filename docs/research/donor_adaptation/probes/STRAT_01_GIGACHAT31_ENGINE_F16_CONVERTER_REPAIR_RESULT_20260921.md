# STRAT-01 GigaChat 3.1 F16 converter repair result

**Verdict:** `PASS_F16_CONVERTER_REPAIR`
**Date:** 21 September 2026
**Scope:** exact binary16 converter repair and offline confirmation; zero donor executions and no attention-dot, integration, quality, RAM, or speed claim

## Result

Changing the early-underflow guard from `exp < -24` to `exp < -25` makes the
project converter byte-identical to pinned GGML on the complete frozen
217,620-value population.

| Control | Result |
|---|---:|
| finite mismatches | 0 |
| NaN policy mismatches | 0 of 3 |
| K/Q/softmax/boundary mismatches | 0 / 0 / 0 / 0 |
| exhaustive unbiased-exponent `-25` checks | 16,777,216 |
| donor executions | 0 |

The project and pinned F16 streams have the same SHA-256,
`7cdec0744fb5c9dcc6225dd70a0c952283128d4f8668d9861131b868379defbc`.
Targeted signed checks cover exact `2^-25`, the next representable F32 value
above it, and `2^-24`.  The production rung-2A model-free self-test also passes
with 28 checks.

## Provenance

- Repair commit: `ee4d70f84d125247b3bfa99229a6f3c0e8dd0208`.
- Canonical adjudication:
  `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_f16_converter_repair_20260921/adjudication.json`,
  SHA-256 `969bf38c628a66db86cc4471ead2a6f862d7e96acec32bbf06bde356315c256b`.
- Pinned llama.cpp: `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`.
- Source-order witness SHA-256:
  `05c0867e42de6ce45f912f202ca38316d5e01214b3c1fffa37a0b93f1aa22ed0`.

## Consequence

The exact-conversion precondition is closed and must not be measured again.
Resume the unchanged F16 vector-dot diagnostic using the already captured
attention inputs.  This PASS does not establish either attention dot or
authorize production attention integration by itself.
