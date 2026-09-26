# METH-31: synthetic Qwen-shaped selected LUT path survives a 10× pool

The [prospective protocol](METH_31_RANK8_LUT_POOL_PROTOCOL_20260926.md)
isolates selected-expert lookup cost, prompted by METH-26's large-E
router byte wall and the user's RAM-scaled expert-count objective.
The benchmark reuses `engine.c`'s existing AVX2 ternary LUT kernel
at L24/D896/rank8/top4, with a fully initialized byte-code pool.
The codes and route IDs are synthetic. This is a **CPU component
measurement**, not timing or quality of the R8+E128 adapter.

| Expert count | Pool bytes | Threads | Median selected path, µs/token | Effective addressed-code GB/s |
|---:|---:|---:|---:|---:|
| 128 | 55,050,240 | 1 | 261.699 | 6.574 |
| 128 | 55,050,240 | 6 | 446.350 | 3.854 |
| 1,280 | 550,502,400 | 1 | 297.825 | 5.776 |
| 1,280 | 550,502,400 | 6 | 444.002 | 3.875 |
| 12,800 | 5,505,024,000 | 1 | 348.213 | 4.940 |
| 12,800 | 5,505,024,000 | 6 | **504.572** | 3.409 |

Each result is the median of repetitions 2–4 of four 512-token
measurements after 128 warmup tokens, with fixed 96 distinct
expert touches/token and **1,720,320 addressed code bytes/token**.
The selected-byte rate divides addressed bytes by wall time; it is
not a hardware DRAM-counter measurement. For E1,280→E12,800 the
six-thread latency ratio is **1.1364×**, below the fixed 1.25×
threshold, and 0.5046 ms/token is below the 5 ms component budget.
The predeclared selected-LUT gate therefore passes at this 10×
pool step. A single thread is faster at every E; launching workers
for each rank8 B projection costs more here than it saves.

The local host is an AMD Ryzen 5 3600X, 6 physical/12 logical cores,
~80 GiB RAM, Windows. Compile command:
`clang -O3 -mavx2 -mfma -march=znver2 -fopenmp benchmarks/phase60/engine.c -o results/native_expert_scaling/engine_meth31_rank8_lut.exe -lm`.
The binary's `--kselftest` passed 73,024 LUT checks with zero
scalar-int error before the sweep. For each E in 128, 1280, 12800,
run the binary with `--rank8-lut-pool E --threads 1` and then
`--threads 6`. All 12 paired repetition checksums match between
thread counts. Total valid sweep wall time was 8.088 s; the largest
pool allocation/initialization took 0.934 s in the one-thread
process. No T4 or GPU was used.

The [raw log](meth31_rank8_lut_pool_raw.log) SHA-256 is
`25dfe3251799a0561ca1db87b01f33e806ddfb4138e61cb711a534a3b9df7e77`;
the [validated machine summary](meth31_rank8_lut_pool_result.json)
SHA-256 is
`c0bab86534b998ec7bbb9a3c3bc530a8f9e37e465ff49be7400458c9f688befd`.
`engine.c` SHA-256 is
`017020eec61e5b65c63e51ae2ff0deea5f5449f1b32d45dc03f2d26f3c8b4231`;
the local binary SHA-256 is
`c3e87c933d71ef81159c28dd202e6b164db54ad421db0720fd2c8c9bc8c14c18`.
The summary parser recomputes pool/selected bytes, validates all
six arms and four repetitions, and checks thread-count checksum
equality. The local binary is retained under ignored results.

The first sweep omitted the planned SiLU and full top-4 output
accumulation. It is retained as [invalid apparatus log](meth31_rank8_lut_pool_invalid_apparatus.log)
and [invalid summary](meth31_rank8_lut_pool_invalid_apparatus.json),
and is excluded from the table and gate. The repair added those
two operations before repeating the unchanged E/threads/repetitions
and decision thresholds. Both sweeps are synthetic, and only the
repaired sweep supports the component verdict.

**Decision and limits.** This existing LUT kernel has ample
selected-path latency margin in this synthetic rank8 geometry,
even when the expert pool grows 10× to 5.505 GB. This does not
show that the real fp32 factors tolerate ternary quantization,
that the CPU router can find the correct experts cheaply, or that
the Qwen donor/core and selected path together meet ≥50 accepted
tok/s. At E273,547 the same padded byte-code layout would need
117.65 GB of pool storage, beyond this host's physical RAM;
100B behavior is unmeasured. The next decisive work is a faithful,
bounded router and a quality-valid packed-factor export, followed
by full `engine.c` integration on the **same** artifact.
