# METH-177: selected LUT cost rises beyond frozen relative limit at E128,000

**Decision: the preregistered 10× relative scaling gate fails.** The local `engine.c` ternary selected path takes 562.925 µs/token with a fully initialized 55.050 GB synthetic E128,000 pool, versus 429.945 µs/token with a 5.505 GB E12,800 pool in the same sweep, at six threads. The ratio is **1.3093×**, above the frozen 1.25× limit. The absolute 5 ms/token component limit passes by a wide margin. This result identifies measurable pool-size cost, while the selected LUT arithmetic remains a small component at this rung. It does not measure the router or a useful learned artifact.

The [protocol](METH_177_LARGE_RAM_LUT_POOL_PROTOCOL_20260930.md) was committed at `7ada832` before implementation or timing. The [benchmark extension](../../../benchmarks/phase60/engine.c) and [guarded runner](../../../benchmarks/native_expert_scaling/meth177_large_lut_pool.py) were committed at `26ac08a` before the sweep. The [machine result](meth177_large_lut_pool_result.json) has SHA-256 `0222734c3eb48e24016f06b928832dcde4493a0709d1920def4edd091b06e9b3` and binds the source, executable and all four raw logs. `--kselftest` passed 73,024 exact LUT checks and its exp approximation check.

| Experts | Pool bytes | Threads | Median of reps 2–4, µs/token | Init, s | Available RAM before, GiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 12,800 | 5,505,024,000 | 1 | 282.839 | 0.712 | 66.16 |
| 12,800 | 5,505,024,000 | 6 | 429.945 | 1.019 | 66.07 |
| 128,000 | 55,050,240,000 | 1 | 375.846 | 10.164 | 65.81 |
| 128,000 | 55,050,240,000 | 6 | **562.925** | 9.509 | 66.34 |

The one-thread ratio is 1.329×; one thread is faster than six at both sizes. Each arm ran 128 warmup tokens and four 512-token timed repetitions with identical L24/D896/rank8/top4 work and 1,720,320 addressed code bytes per token. All four repetition checksums match across thread counts at each E. Independent parsing recomputed every median and verified each log SHA. The four E128,000 six-thread repetitions were 570.004, 585.018, 562.925 and 560.164 µs/token. Both large arms passed the 64 GiB available-RAM preflight and completed in about 14 seconds per process; each allocated and initialized the complete 51.27 GiB pool before timing. No GPU or T4 was used. Timed page faults were not recorded, so this is not a cold-DRAM latency or physical-residency audit.

This uses synthetic repeated ternary codes and random precomputed expert IDs. It does not establish packed-factor quality, semantic selection, throughput of `engine.c` generation, or scaling to E273,547 (the ~100B added-parameter geometry, whose same padded LUT layout exceeds local RAM). METH-176 already failed to establish useful E12,800 specialists. A future target design should evaluate a layout or access strategy against this relative cost, then pair it with a quality-valid router and factors. The high-level >=50 accepted-token/s requirement remains untested for the same full artifact.
