# METH-31: CPU LUT selected-expert pool scaling, Qwen rank-8 geometry

**Uncertainty.** METH-26 shows that exhaustive routing becomes
impossible as `E` grows, but does not measure the fixed-top-4
selected-expert LUT path. NES-02 measured a smaller L6/D256/h128
native ternary model up to synthetic E1280; it cannot price the
Qwen L24/D896/rank-8 geometry. This test asks whether the existing
`engine.c` ternary lookup kernel itself stays within a useful CPU
latency bracket as the RAM-resident expert pool grows by 10×.

**Changed coordinate and fixed controls.** Add a weight-free
`--rank8-lut-pool E` microbench in `benchmarks/phase60/engine.c` that
uses its existing byte-coded ternary `matvec_lut_full` and LUT builder.
Each of 24 layers addresses top-4 distinct experts; each expert
has an A projection D896→rank8 (padded to 32 rows by this kernel)
and B projection rank8→D896. Each code is a byte-valued two-trit
index; the pool is fully allocated and initialized before timing.
Selected IDs are precomputed by deterministic xorshift64 from seed
`0x9e3779b97f4a7c15`, excluding duplicate IDs within a layer/token.
No router scoring or route selection is included in the timed loop.
Inputs/LUTs, nonlinearity and accumulation remain the same across E.
The same synthetic code formula is used in every pool; these codes
are not learned expert factors and have no quality interpretation.

Measure E128, E1280 and E12800 with 128 warmup tokens and 512 timed
tokens in each of four repetitions, for 1 and 6 OpenMP threads.
Record all repetitions and use the median of repetitions 2–4.
Check the existing `--kselftest` before measurements and verify
checksum equality between thread counts at each E. Report pool
bytes, selected code bytes/token, latency/token and effective
selected-byte rate. Compile with local clang using `-O3 -mavx2
-mfma -march=znver2 -fopenmp`, without fast-math. The pool is
about 55 MB, 550 MB and 5.505 GB respectively; a machine with
~80 GB physical RAM has capacity for the largest. A direct
E273,547 (~100B added parameters at the METH-26 geometry) pool
would require ~117.6 GB in this *padded byte-code* layout and is
outside local RAM. No 100B speed or quality is inferred from it.

**Decision gate.** For the E1280→E12800 10× pool expansion, the
6-thread selected path must stay ≤1.25× its E1280 median and
≤5 ms/token. Passing would make the selected LUT path a viable
component for further work, not a joint 50 tok/s result. Failing
would require a layout/kernel change before assuming RAM-only E
scaling. In either case, the unresolved exhaustive-router cost,
packed-factor quality, and end-to-end accepted-token rate remain
separate gates. The 1-thread arm diagnoses thread overhead.

**Budget and stop.** No GPU or T4. Limit E to 12800, the
allocation to <6 GiB, and elapsed experiment wall time to 15
minutes. If the fully initialized allocation fails or the limit
is reached, report the stop and retain completed rungs; do not
extrapolate a pass. Do not modify model weights or existing
inference paths.
