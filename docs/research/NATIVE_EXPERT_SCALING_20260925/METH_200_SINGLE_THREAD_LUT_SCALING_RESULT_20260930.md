# METH-200: single-thread control also fails repeatability

The [protocol](METH_200_SINGLE_THREAD_LUT_SCALING_PROTOCOL_20260930.md)
and [runner](../../../benchmarks/native_expert_scaling/meth200_single_thread_lut_scaling.py)
were committed at `6e4ce82` before execution. They reused the
same bound METH-197 C binary/source with no prefetch, passed its
73,024-check self-test and ran three one-thread pairs in alternating
small/large order. All checksums matched across repetitions at the
same E; selected factor bytes/token and code path were unchanged.

| Pair | E=12,800 median | E=128,000 median | Large/small |
|---:|---:|---:|---:|
| 1 | 259.749 µs/token | 265.976 µs/token | 1.024× |
| 2 | 280.236 µs/token | 266.996 µs/token | 0.953× |
| 3 | 295.393 µs/token | 332.689 µs/token | 1.126× |

The within-size max/min ratios are **1.137×** and **1.251×**, both
above the frozen <=1.10 repeatability requirement. METH-200 is
therefore **inconclusive for even the one-thread scaling ratio**.
All selected-LUT component medians are below 0.34 ms/token, but
neither that nor a noisy large/small ratio establishes the full
model's >=50 accepted tok/s target. The [raw result](meth200_single_thread_lut_scaling_result.json),
SHA-256 `6625df0c0d821d03437216f3d248f0d655ade4f9254ea64be30f8c542ede9eb6`,
records all repetitions, logs, hashes, memory readings and gates.
No T4 was used.

METH-177's one-shot 1.309 ratio, METH-197's opposite one-shot
0.917 ratio, and METH-198/199/200's repeatability failures do not
support a reliable pass or fail for tenfold LUT-pool scaling on this
host. Do not keep retiming this synthetic harness without a concrete
apparatus change that explains the drift. The separate quality
question remains: E12800 training has not shown tenfold additional
useful specialists on untouched sources.
