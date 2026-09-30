# METH-198: six-thread large-pool LUT scaling is timing-inconclusive

The [frozen protocol](METH_198_PAIRED_LUT_SCALING_PROTOCOL_20260930.md)
and [runner](../../../benchmarks/native_expert_scaling/meth198_paired_lut_scaling.py)
were committed at `f329cae` before execution. They reused the exact
METH-197 C binary/source, passed the 73,024-check self-test again,
and ran six no-prefetch measurements in small/large,
large/small, small/large order. Every E=12,800 checksum sequence
matched the other small-pool runs; likewise for E=128,000. Each run
selected the same 1,720,320 code bytes/token.

| Pair | E=12,800 median | E=128,000 median | Large/small |
|---:|---:|---:|---:|
| 1 | 425.883 µs/token | 494.309 µs/token | 1.161× |
| 2 | 599.906 µs/token | 417.727 µs/token | 0.696× |
| 3 | 606.049 µs/token | 431.151 µs/token | 0.711× |

Within-size max/min is **1.423×** for E=12,800 and **1.183×** for
E=128,000, both above the frozen 1.10 repeatability ceiling.
The prescribed decision is **inconclusive**, regardless of the
paired ratios. All large-pool absolute medians remain below
0.5 ms/token, but that isolated synthetic component does not prove
full accepted-token rate. The [raw result](meth198_paired_lut_scaling_result.json),
SHA-256 `199f76710429e27038f45f78cb28c7dcdc5e73313318301ba10c94d3eaa7ad62`,
contains every repetition, process log, checksum, binary hash and
memory reading. No T4 was used.

The ratio disagreement with METH-177 and METH-197 now has direct
repeatability evidence. Do not claim that the selected LUT path
passes or fails the <=1.25 large-n scaling gate from these runs.
A controlled six-core OpenMP binding is the next timing apparatus
check; even a stable pass would still be synthetic and would say
nothing about the quality of newly added experts.
