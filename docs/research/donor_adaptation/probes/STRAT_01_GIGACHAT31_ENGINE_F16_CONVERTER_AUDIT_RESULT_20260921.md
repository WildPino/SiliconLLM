# STRAT-01 GigaChat 3.1 F16 converter audit result

**Verdict:** `FAIL_F16_CONVERTER`
**Date:** 21 September 2026
**Scope:** byte-level project-versus-pinned binary16 conversion; zero donor executions and no dot, integration, quality, RAM, or speed claim

## Result

The repaired-order audit finds 843 finite conversion mismatches among 217,620
frozen values.  Every mismatch is below the F16 normal range and has the same
form: project conversion returns signed zero while pinned GGML returns signed
minimum subnormal (`0x0001`/`0x8001`).

| Segment | Values | Mismatches |
|---|---:|---:|
| K cache input | 4,608 | 0 |
| Q input | 147,456 | 842 |
| padded softmax | 65,536 | 1 |
| fixed IEEE boundary suite | 20 | 0 |

NaN policy matches on all three NaN cases.  The parallel F32 source-bit stream
is byte-identical to the adjudicator's reconstructed `K, Q, softmax, boundary`
population, closing the prior stream-order defect.

## Root cause

`strat01_r2a_f32_to_f16` returns zero when unbiased F32 exponent is below
`-24`.  That is one exponent too aggressive.  Values strictly above the
halfway point `2^-25` must round to the smallest F16 subnormal `2^-24`; exact
`2^-25` remains zero by ties-to-even.  The source-derived repair is to change
the early-zero guard from `exp < -24` to `exp < -25` and let the existing
round-to-nearest-even subnormal path handle exponent `-25`.

## Provenance

- Accepted audit commit: `493eccd` (full hash in raw adjudication).
- Canonical adjudication:
  `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_f16_converter_audit_repair1_20260921/adjudication.json`,
  SHA-256 `21293ae8c5ddd041ac736405197674606e23f46dfae639c37835711896b3136e`.
- Project F16 stream SHA-256:
  `a7a1d99d66e13ce77b87a569b534bb34af8eaddff0f210f5355a84f55d8ba8ec`.
- Pinned F16 stream SHA-256:
  `7cdec0744fb5c9dcc6225dd70a0c952283128d4f8668d9861131b868379defbc`.
- Source-order witness SHA-256:
  `05c0867e42de6ce45f912f202ca38316d5e01214b3c1fffa37a0b93f1aa22ed0`.
- `donor_executions=0`.

## Preserved apparatus VOID

The first audit raw set (adjudication SHA-256
`b6caf4b26c19f6805f9eabb00a0ae6f878d1cf92df18b40f0998c9f99afc46a0`)
interleaved Q and softmax conversions but interpreted them as contiguous
segments.  It is apparatus history only.  Repair A added the exact source-bit
witness and produced the canonical result above.

## Consequence

Do not repeat the audit or generalize beyond project/pinned binary16 semantics.
Apply the one-line threshold repair to the production converter and diagnostic
copy, require zero mismatches on the same population plus targeted subnormal
edge regressions, then resume the still-unmeasured F16 vec-dot protocol.  No
attention integration or speed claim is authorized by this audit alone.
