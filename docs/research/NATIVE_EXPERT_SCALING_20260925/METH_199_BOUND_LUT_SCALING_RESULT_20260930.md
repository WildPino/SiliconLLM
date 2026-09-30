# METH-199: OpenMP core binding does not stabilize the six-thread ratio

The [protocol](METH_199_BOUND_LUT_SCALING_PROTOCOL_20260930.md)
and [runner](../../../benchmarks/native_expert_scaling/meth199_bound_lut_scaling.py)
were committed at `06f24b6` before execution. They reused the
identical METH-197/198 C binary, passed its self-test, and captured
runtime confirmation of `OMP_PLACES=cores` and
`OMP_PROC_BIND=spread` in every run. The synthetic routes,
1,720,320 selected code bytes/token and within-size checksum
sequences remained identical.

| Pair | E=12,800 median | E=128,000 median | Large/small |
|---:|---:|---:|---:|
| 1 | 417.612 µs/token | 382.493 µs/token | 0.916× |
| 2 | 567.855 µs/token | 386.574 µs/token | 0.681× |
| 3 | 663.620 µs/token | 387.936 µs/token | 0.585× |

The large pool is repeatable at 1.014× max/min, but the small pool
is not: 1.589× max/min against the frozen <=1.10 criterion. The
decision remains **inconclusive for the six-thread scaling ratio**.
Binding is not a sufficient remedy for the observed timing drift.
The [raw result](meth199_bound_lut_scaling_result.json), SHA-256
`25fecb0a5e1453dd2418968a77764462c8418d7bee28ca8ce9470c4c0b8c45a6`,
stores all repetitions, runtime environment output, checksums and
memory readings. No T4 was used.

A one-thread paired audit can isolate whether the instability
persists without OpenMP. Its result can diagnose this component
measurement but cannot promote a six-thread native decoding claim.
