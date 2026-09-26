# METH-35: fp32 rank-64 shortlist passes the reused route gate

The [prospective protocol](METH_35_LOW_RANK_ROUTER_PROTOCOL_20260926.md)
tested an uncentered SVD sketch of each frozen trained E128 router.
The R8 core, E128 adapter, 24-layer top-4 route and exact model
trajectory are the same as METH-29/30. Its 24 exact-route stream
hashes reproduce METH-29. The 24 METH-27 prompts had already been
observed; this is a routing diagnostic, not independent model quality.

| Sketch rank | Exact candidates | Top-4 ID inclusion | Full-set match | Lowest category inclusion | Float multiply ratio vs E128 exhaustive | Gate |
|---:|---:|---:|---:|---:|---:|---|
| 16 | 32 | 77.803% | 37.992% | 75.907% | 0.393× | fail |
| 16 | 64 | 92.623% | 73.912% | 91.750% | 0.643× | fail |
| 32 | 32 | 90.138% | 66.490% | 89.048% | 0.536× | fail |
| 32 | 64 | 98.090% | 92.633% | 97.788% | 0.786× | fail |
| 64 | 32 | 98.540% | 94.339% | 98.289% | 0.821× | fail |
| **64** | **64** | **99.929%** | **99.717%** | **99.900%** | **1.071×** | **pass** |

The passing arm exceeds the frozen ≥99.9% pooled selected-ID,
≥99% pooled full-set and ≥99% per-category inclusion gates on
147,456 input-layer cases. Mean missed exact gate mass is 0.000487.
Rank-64 retains 65.952% of router-weight squared energy on average;
low weight reconstruction energy alone would not predict its good
top-4 retrieval with a 64-row candidate set. At E128 this arm
performs *more* float multiply terms than exhaustive routing,
122,880 versus 114,688 per layer, before selection overhead.

If the sketch codes were one byte, a hypothetical E273,547 pool
would address **420,168,192 sketch code bytes/token** across 24
layers, plus row scales, the rank-64 projection and 64 exact
candidate rows/layer. At a favorable 40 GB/s that code traffic
alone costs 10.50 ms/token; this is arithmetic, not a measured CPU
rate. The present experiment used fp32 GPU sketches. It does not
show that int8 quantization retains recall, that the C implementation
meets a 20 ms end-to-end budget, or that E273,547 distinctly learned
experts route or produce useful quality like E128.

The [machine result](meth35_low_rank_router_result.json), SHA-256
`f701f1968987db0d107fe4383166837e07be985771754ee1bce36de6615710d7`,
contains per-layer/category counts, spectrum energy and all route
hashes. Command:
`.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth35_low_rank_router.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth35_low_rank_router_result.json`.
The local RTX 3060 run took 60.25 s, peaked at 2.275 GB allocated
GPU memory and ended at 3.282 GB RSS. No T4 was used.

**Decision:** retain rank-64/64 as a *candidate* for a one-byte
stored-sketch and independent-input audit. The five other arms fail
the fixed route gate. Only if a quantized, independent route test
passes should this index be implemented and timed on CPU. The
parent R8+E128 model still fails generation, and larger distinct
expert capacity is unproven.
