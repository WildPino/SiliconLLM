# METH-197: prefetch regresses the selected-expert LUT path

The [protocol](METH_197_LARGE_POOL_PREFETCH_PROTOCOL_20260930.md),
benchmark-only [engine flag](../../../benchmarks/phase60/engine.c)
and [runner](../../../benchmarks/native_expert_scaling/meth197_large_pool_prefetch.py)
were committed at `d37ce1a` before execution. The same newly
compiled C binary passed the 73,024-check LUT self-test, then ran
baseline and explicit selected-A/B cache-line prefetch arms with
identical synthetic routes, 1,720,320 selected code bytes/token,
and matching checksums at E=12,800/128,000 and one/six threads.

| Six-thread resident pool | Baseline median | Prefetch median | Prefetch / baseline |
|---|---:|---:|---:|
| E=12,800 | 430.521 µs/token | 552.500 µs/token | 1.283× |
| E=128,000 | 394.591 µs/token | 515.606 µs/token | 1.307× |

The prefetch E128,000/E12,800 ratio is 0.933 and absolute cost
0.516 ms/token, but the frozen <=1.10 small-pool regression gate
fails by a large margin. **Do not use this prefetch variant.** It
adds roughly 0.12 ms/token at both pool sizes in this run. The
[raw result](meth197_large_pool_prefetch_result.json), SHA-256
`e9435e4ad58c4dc92420b395b1876edfbabf9e48328c34db6581b39abfec644a`,
holds all four timed repetitions, 1-thread arms, memory readings,
native logs and checksums. No T4 was used.

The no-prefetch E128,000/E12,800 ratio in this run is 0.917,
whereas the prior [METH-177 result](METH_177_LARGE_RAM_LUT_POOL_RESULT_20260930.md)
was 1.309, with E128,000 at 562.92 µs/token. These independent
system runs disagree enough that one-shot ratio claims about
large-n scaling are not reliable. A paired repeated audit is needed.
These synthetic selected-factor measurements cannot establish full
native accepted-token speed or quality of many useful experts.
