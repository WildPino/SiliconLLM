# C50-S1 result: speed is invariant with capacity at 2x and 4x, in both router formats (PASS)

20 October 2026. Protocol: [C50_S1_CAPACITY_INVARIANCE_PROTOCOL_20261018](C50_S1_CAPACITY_INVARIANCE_PROTOCOL_20261018.md)
with addenda 1 and 2. Run record: `results/c50_s1_run_3.json` (09:06-13:42, sha256 `8dd56dd8…`). Report:
`results/c50_s1_report_3.json`. Runs 1 and 2 of the same day are VOID at the pre-flight and kept (§12, 20 October).

## Verdict

**PASS in both formats.** The CI95 of the residual ρ_f lies inside [0.97, 1.03] at m2 and at m4, with the F32 router
and with the BF16 router. By the protocol's outcomes, the Phase-1 invariance claim holds at m ≈ 2 and m ≈ 4: when T0's
stored experts grow 2x and 4x at fixed active bytes, the time per token grows only by the router's priced bytes.

| Format | m | E | v_E tok/s | v_pred tok/s | ρ | ρ CI95 | raw v(m)/v(1) | raw CI95 | inside τ |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |
| F32 | m1 | 256 | 37.40 | - | - | - | 1 | - | - |
| F32 | **m2** | 536 | 35.91 | 35.53 | **1.0106** | **0.9976..1.0238** | 0.9601 | 0.9475..0.9729 | yes |
| F32 | **m4** | 1,088 | 32.63 | 32.34 | **1.0089** | **0.9974..1.0205** | 0.8725 | 0.8617..0.8835 | yes |
| F32 | m6 (reported) | 1,600 | 29.96 | 29.86 | 1.0033 | 0.9913..1.0157 | 0.8010 | 0.7905..0.8120 | yes |
| BF16 | m1 | 256 | 38.34 | - | - | - | 1 | - | - |
| BF16 | **m2** | 536 | 37.25 | 37.14 | **1.0031** | **0.9907..1.0159** | 0.9716 | 0.9593..0.9842 | yes |
| BF16 | **m4** | 1,088 | 35.08 | 34.97 | **1.0032** | **0.9915..1.0156** | 0.9150 | 0.9037..0.9268 | yes |
| BF16 | m6 (reported) | 1,600 | 32.57 | 33.18 | 0.9819 | 0.9349..1.0141 | 0.8496 | 0.8085..0.8778 | no |

- Rates are the mean of the per-process median tok/s: 20 processes per gated arm, 10 per m6 arm, context 2048, 128
  timed steps after a 2,048-token fill. v_pred prices the router's extra bytes with c_f, measured at m1 only.
- **The run is COMPLETE.** Binaries, engine source (`c50_engine.c` at `1eb08f9`) and all four model hashes matched
  their pins. Pre-flight foreign occupancy 4.27% (bar 4.39%). **K1** (BF16 router at m1) and **K2** (arm H at m2, m4,
  m6, both formats) were all **byte-identical** to the A3 binary on `par1.tsv`. No process failed and every CONFIG line
  confirmed. 11 processes at or above the bar were replaced: 4 in m2/E-BF16, 4 in m2/H-F32 and 3 in m6/E-F32; the
  limit is more than 5 per arm.
- **Conduct of the run.** It ran in the agent session with the memory reaper off. From ~09:30 short bursts of
  foreign load (a daemon of another project and the browser) caused the replacements. That daemon was suspended at
  09:55, a decision taken during the run, and resumed after the run. No replacement occurred after 09:55.
- **m6 BF16** is outside τ, reported only. Its arm has CV 6.9%: one process (round 8) ran at 26.4 tok/s with foreign
  load 3.94%, under the bar. Without a rule fixed before the run it stays in the mean.

## The router's price

| Format | c_f s/byte (m1 organs) | CI95 | as GB/s | router lap at m1 / m2 / m4 / m6, ms | lap per byte as GB/s, m1 / m2 / m4 / m6 |
| --- | --- | --- | ---: | --- | --- |
| F32 | 3.228e-11 | 3.196..3.261e-11 | 31.0 | 1.286 / 2.676 / 5.289 / 7.748 | 31.0 / 31.2 / 32.0 / 32.1 |
| BF16 | 3.884e-11 | 3.826..3.948e-11 | 25.7 | 0.774 / 1.602 / 3.236 / 4.627 | 25.7 / 26.0 / 26.2 / 26.9 |

- **The organ laps are constant per byte across m** (within 5%). The router streams at about the weights' rate in
  F32 (S0: 31-37 GB/s per organ); BF16 streams its bytes ~17% slower but has half of them, so its lap is 0.60 of
  F32's.
- **The independent check, arm E minus arm H,** (1/v_E − 1/v_H) / ΔR, in s/byte:

  | Format | m2 | m4 | m6 |
  | --- | --- | --- | --- |
  | F32 | 2.32e-11 (1.24..3.35) | 3.09e-11 (2.75..3.38) | 2.35e-11 (1.95..2.73) |
  | BF16 | 4.12e-11 (2.42..5.76) | 2.94e-11 (2.40..3.53) | 3.82e-11 (2.81..5.31) |

  It contains c_f at F32 m2 and m4 and at BF16 m2 and m6. It lies below c_f at F32 m6 and BF16 m4. The difference
  between two whole-process rates carries both arms' dispersion. The organ laps, the direct measurement, do not vary.
- **The BF16 router at m1** runs 2.5% faster than the F32 router in the same session (38.34 against 37.40 tok/s). This
  is the "v0 with the BF16 router beside v0" that ROADMAP §5 asks for, as a same-session ratio. The m1 F32 arm here
  (37.40) is not compared with S0's v0 (35.94): that was another session, and `c50_engine.c` at another commit
  (`7835687`, binary `786e1c6d…`).
- **BF16 expansions against the F32 T0 (includes the router upgrade):** 0.996 at m2, 0.938 at m4, 0.871 at m6.

## Pool effect and selection spread (arm H, reported)

| Format | m2 | m4 | m6 |
| --- | --- | --- | --- |
| F32 v_H / v(1) | 0.996 (0.978..1.013) | 1.003 (0.987..1.019) | **0.939 (0.913..0.966)** |
| BF16 v_H / v(1) | 1.005 (0.990..1.020) | 0.980 (0.967..0.995) | 0.977 (0.958..0.995) |

- Up to m4 the pool effect is within 2% in both formats. At m6 (58.6 GB mapped, ~10 GB of RAM left) arm H loses 6.1%
  in F32 and 2.3% in BF16. The F32 m6 H arm also has CV 4.5% and the longest fills (74 s). Arm E at m6 passes ρ in F32
  all the same. That is the E − H estimate below c_f in the table above, so at m6 the two arms do not decompose
  cleanly. m6 is reported, not gated, and is the edge of what fits in this RAM.
- **Spread:** arm E's copy-0 share is 0.479 at m2 (the protocol's "unspread" label needs > 0.75: not applied), 0.241 at
  m4, 0.143 at m6. The distinct-expert fraction is 0.259 / 0.189 / 0.165 (T0: 0.407). The perturbed router spreads
  its picks over the copies.

## Arm M: memory-bank lookups (reported)

Median token time over 3 fresh processes, 19 layers × k rows of 4,096 bytes, pageable memory:

| Table | k = 2 (38 rows) | k = 8 (152 rows) | k = 32 (608 rows) |
| --- | --- | --- | --- |
| 1 GB | 15.4 µs, 10.1 GB/s | 29.4 µs, 21.2 GB/s | 107.2 µs, 23.2 GB/s |
| 4 GB | 14.8 µs, 10.5 GB/s | 30.5 µs, 20.4 GB/s | 90.1 µs, 27.6 GB/s |
| 16 GB | 14.3 µs, 10.9 GB/s | 28.6 µs, 21.8 GB/s | 82.7 µs, 30.1 GB/s |
| 48 GB | 15.0 µs, 10.4 GB/s | 30.5 µs, 20.4 GB/s | 84.2 µs, 29.6 GB/s |
| 48 GB / 1 GB | 0.974 | 1.037 | 0.785 |

- **The lookup cost does not grow with table size from 1 to 48 GB** at any k (the O(1) claim of ROADMAP §3). At k = 32 the 1 GB
  table is the slowest; with 3 processes per cell this is not explained here.
- **Scale:** k = 8 per layer costs ~30 µs per token, about 0.1% of a ~27 ms token at m1. Small k is latency-bound
  (~10 GB/s at k = 2); from k = 8 the rows stream at 20-30 GB/s.

## Registered predictions, scored

- **At m ≈ 4 the raw slowdown is 10-16% (F32) and 5-9% (BF16): RIGHT.** F32 12.8% (CI 11.7-13.8%), BF16 8.5% (CI
  7.3-9.6%; the CI's upper end crosses 9%).
- **The residual ρ lies within 3% in both formats: RIGHT** (m2 and m4; ρ within 1.1%).
- **The BF16 router is byte-identical to T0 at m = 1 (K1): RIGHT.**
- **Arm H's pool effect at m ≈ 4 is below 3%: RIGHT on the point.** F32 +0.3%, BF16 −2.0%; BF16's CI reaches −3.3%.

## What this says, and what it does not

- **Capacity costs only the router bytes, up to 4x on this box.** The raw slowdown at 4x (12.8% F32, 8.5% BF16) is the
  router's price, predicted to within 1% from bytes and a lap measured at m1. A router that does not grow with E
  (O4: int8 or two-level) is where the remaining slowdown is.
- **After S1, by ROADMAP §5:** K1 passed, so S4 and S5 use the BF16 router. T0 stays P0.
- **Scope:** one box (this 3600X), one artifact family (T0 and its byte-copy expansions), context 2048, decode only,
  `--mem map`. The copies are not knowledge, and arm E's spread is a perturbed router's, not a trained one's. m6 is
  the largest pool that fits in this RAM; beyond it the pool is out of scope.

## Next

S3 (option A: local top-k from F0). First the F0 GPU study (fidelity against F0's CPU scores; ubatch throughput),
then the S3 protocol, pushed before any teacher data is kept.
