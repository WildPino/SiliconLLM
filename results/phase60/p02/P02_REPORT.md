# P0.2 — Freshness check of the E2/E4 top-1 coincidence (RESOLVED: true coincidence, no harness bug)

**Date:** 2026-07-03 · **Trigger:** Architect's pending verification (E4 adjudication, `docs/ENGINE_PLAN.md`):
E2-G3 and E4-rung-3b both reported top-1 = 99.7363% (27/10240) on *different models* — "coincidences get checked".

## Method (pre-declared)
1. Added a diagnostic `--dumpflips <file>` to `e2_engine.c` (G3) and `e4_engine.c` (G5b): one line per
   disagreeing position `<global_val_index> <ref_argmax> <opt_argmax>`. Gate semantics untouched.
2. Re-ran both gates fresh (weights re-exported 2026-07-03), offset 0 — dump the flip positions.
3. Counter-measurement: re-ran both with the slice shifted by one window (`--offset 512`).

## Results

| run | slice | flips | agreement |
|---|---|---|---|
| E2 G3 (LUT vs fp32, `sp58_base`) | offset 0 | **27**/10240 | 99.7363% |
| E4 G5b (LUT+skip vs ref, `moe_gran`) | offset 0 | **27**/10240 | 99.7363% |
| E2 G3 | offset 512 | **26**/10240 | 99.7461% |
| E4 G5b | offset 512 | **27**/10240 | 99.7363% |

**Position overlap at offset 0: ZERO.** E2 flips at val-indices {307, 624, 830, 1093, 1429, 1458, 1528, 2031,
2173, 2305, 2473, 3461, 3579, 3666, 3726, 3839, 3894, 5649, 6295, 6927, 6985, 7010, 8430, 8610, 9243, 9832,
10040}; E4 flips at {127, 455, 686, 863, 1770, 2036, 2411, 2598, 2986, 4229, 4615, 5262, 5542, 6432, 6646,
7389, 7455, 7608, 7721, 8124, 8258, 8764, 9080, 9147, 9432, 9563, 9853}. No index in common; the argmax
pairs also differ. Raw dumps: `flips_e2_off0.txt`, `flips_e4_off0.txt`, `flips_e2_off512.txt`,
`flips_e4_off512.txt` (this directory).

## Verdict
**True count coincidence — both measurements are fresh and independent.**
- Structurally, neither gate reads a shared dump: both compute the two paths in-process on `ids.u16`.
- The flip *positions* are disjoint (rules out a reused slice/dump); the *counts* move independently when
  the slice shifts (26 vs 27 at offset 512).
- Interpretation: ~0.26% top-1 flip rate is the stable signature of int8 per-token-absmax activation
  quantization on this model family at this scale; on two same-length slices, two counts drawn from a
  distribution centered near ~27 collided once. Nothing to correct.

Both measurements also reproduced today from scratch (fresh exports): the numbers in `docs/ENGINE_PLAN.md`
stand as-is.
