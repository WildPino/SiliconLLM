# METH-52: CPU selected-path cost of exact-effective BF16 factors

**Uncertainty.** METH-51 provides a quality-exact trained E128 factor
bank, but its 2,752,512 selected factor bytes/token have no CPU timing.
METH-31 measured a synthetic ternary LUT path, whose real trained factor
formats failed quality. Determine whether a direct BF16 selected path is
fast enough to remain a fallback when expert count and RAM grow 10×,
before spending effort on another lossy LUT. This isolates expert compute
and DRAM access; no whole-model or large-E quality claim follows.

Bind the METH-51 factor artifact SHA-256
`53f2849da55f0da7f91f7d097ab3357308abfd85174cdf8a82b7e226cd738473`
and L24/D896/rank8/top4 layout. An exporter verifies all tensor shapes,
BF16 dtype and artifact metadata and writes an interleaved layer/expert
binary seed (`a[8,896]`, then `b[896,8]`, little-endian BF16). The C
benchmark loads that seed and expands E128 rows deterministically to
E2,735 and E27,355. Expanded rows are **replicas**, not distinct learned
experts. E273,547 is a storage projection only: its ~188 GB factor bank
exceeds this host's RAM budget.

For each synthetic token/layer, generate a deterministic D896 input and
four distinct selected IDs, execute all eight BF16 A dot products, SiLU,
all 896 BF16 B row dot products, fixed four-way gate weighting and full
output accumulation. Benchmark E128, E2,735 and E27,355 with one and six
CPU threads. A scalar-versus-AVX2 dot self-test must pass before timing;
one- and six-thread checksums must match at each E. Use 64 warmup tokens,
then four 256-token repetitions per arm. Report median ms/token,
projection, output/combination and dispatch time, exact selected bytes,
pool footprint and process/RSS cost. Use the same deterministic route/input
stream across E and thread arms, with selected IDs in range.

**Decision bounds:** if the fastest E27,355 selected path alone exceeds
3 ms/token on this host, it is too costly as the currently measured
factor fallback within a 20 ms whole-model goal. If it is ≤3 ms and
its E2,735→E27,355 median ratio is ≤1.5 for the same fastest thread
configuration, retain it as a CPU component candidate. Passing leaves
the core, router, exact routing quality, tokenizer, semantics and
accepted-token throughput unproven. A different CPU or RAM amount needs
its own measurement; do not extrapolate a 100B rate from the 10B point.

Compile with local clang `-O3 -mavx2 -mfma -march=znver2 -fopenmp -lm`.
Local CPU only, ≤32 GiB RSS and ≤15 minutes for the complete sweep;
stop on seed/hash/self-test/allocator failure. No T4.
