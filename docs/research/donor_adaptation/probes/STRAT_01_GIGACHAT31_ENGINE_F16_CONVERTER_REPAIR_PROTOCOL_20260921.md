# STRAT-01 GigaChat 3.1 F16 converter repair protocol

**State:** FROZEN PROTOCOL; NOT YET EXECUTED
**Date frozen:** 21 September 2026
**Scope:** exact binary16 converter repair and offline confirmation; no donor, attention-dot, quality, RAM, or speed claim

## Changed coordinate

In both production `strat01_r2a_f32_to_f16` and the independent diagnostic
copy, change only the early-underflow condition from `exp < -24` to
`exp < -25`.  Do not change NaN canonicalization, normal rounding, subnormal
rounding, overflow handling, or F16 decoding.

## Required confirmation

1. Re-run the hash-bound 217,620-value converter audit with exact source-order
   witness and require zero finite mismatches plus matching NaN policy.
2. Add model-free assertions for exact `2^-25` (zero, ties-to-even), the next
   representable positive/negative F32 values above it (signed minimum F16
   subnormal), `2^-24`, and their signs.
3. Exhaust every F32 mantissa for unbiased exponent `-25`, both signs, against
   pinned GGML conversion, or an equivalently complete deterministic proof
   harness.  Report checked count and zero mismatch requirement.
4. Keep all prior rung-2A and converter-audit model-free tests passing.

Labels: `PASS_F16_CONVERTER_REPAIR`, `FAIL_F16_CONVERTER_REPAIR`, or
`VOID_F16_CONVERTER_REPAIR` for any identity/build/completeness/control
failure.  Run one non-VOID offline confirmation and stop.

## Consequence boundary

Only PASS clears the exact-conversion precondition and permits resuming the
unchanged F16 vec-dot diagnostic.  It does not itself prove either attention
dot, full block integration, quality, RAM, or speed.
