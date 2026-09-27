# METH-63: 10× expert geometry is allocatable and routes exactly

**Decision:** the CPU storage and exact-routing preflight passes for
E128=8×16 and E1280=32×40 over 24 layers. This supports a subsequent
distinct-expert training experiment; the 1,280 slots are currently
initialized, **not learned**. No donor-relative quality, CPU LUT or
full-model rate claim follows from this result.

The [protocol](METH_63_E1280_TRAINING_GEOMETRY_PROTOCOL_20260927.md)
was committed before execution. The [runner](../../../benchmarks/donor_adaptation/s1/meth63_e1280_geometry_preflight.py)
SHA-256 is `edf3b851b9dd15f5b415c5567c79f2e4366062d4c921d5e9580a8f8dca54809d`.
Its [raw result](meth63_e1280_geometry_preflight_result.json) SHA-256 is
`e76c7c62f7aae9ca92a40dc017a922c37b55d608881cf45edee4efa5d61300d0`.
The runner allocated independently addressed FP32 A/B factors and
rank-64 product keys for every layer, with zero B factors and the
METH-55 initialization rule. The base function was identity.

| Measure | E128 | E1280 |
|---|---:|---:|
| FP32 training factor parameter bytes | 176,160,768 | 1,761,607,680 |
| FP32 router parameter bytes | 5,652,480 | 5,947,392 |
| Expected BF16 inference factor payload | 88,080,384 | 880,803,840 |
| Selected top-four BF16 factor bytes/token | 2,752,512 | 2,752,512 |
| Exact pair-oracle cases | 768/768 | 768/768 |
| Zero-B identity failures | 0 | 0 |
| Process peak RSS | 605,491,200 | 2,344,222,720 bytes |
| Cumulative elapsed | 0.625 | 2.984 s |

The 32 fixed vectors per layer make only 128 selections/layer, so
E1280's 109–124 selected slots and 30× worst maximum/mean route load
are merely short-input initialization diagnostics. They do not
measure training coverage or load after adaptation. Parameter bytes
exclude gradients, optimizer state, donor core and runtime workspace.
Dense FP32 Adam training would need at least four copies of the factor
bank (parameters, gradients and two moments), or 7,046,430,720 bytes,
before those other allocations. At this turn's observation the local
12 GiB RTX 3060 had only 5,470 MiB free because another process was
using it; a naive all-GPU E1280 training launch would therefore exceed
available memory. A later training experiment needs free GPU capacity
or a measured sparse/offloaded update scheme.
The selected factor figure is addressed payload arithmetic, not
measured DRAM traffic. The 10× storage increase with constant top-four
factor work matches the intended RAM-dependent expert-count dial at
this small geometry, but learned specialization and quality with more
choices remain the next decisive gate.
