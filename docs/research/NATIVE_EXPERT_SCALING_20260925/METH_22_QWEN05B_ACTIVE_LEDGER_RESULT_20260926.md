# METH-22 result: tied-head precision is a required cost decision

**Decision:** the existing Qwen native donor path's fp32 tied head
prevents an *ideal* one-byte-body + E128 adapter from fitting the
20 ms/token payload budget at the 40 GB/s streaming yardstick. A
quality-preserving head representation is part of the next
conversion experiment. This is shape arithmetic and a conditional
bandwidth estimate, not measured C latency or a quantized model.
The [protocol](METH_22_QWEN05B_ACTIVE_LEDGER_PROTOCOL_20260926.md),
[machine ledger](meth22_qwen05b_active_ledger.json) and
[tool](../../../benchmarks/donor_adaptation/s1/meth22_qwen05b_active_ledger.py)
bind the calculation.

```text
.venv/Scripts/python.exe benchmarks/donor_adaptation/s1/meth22_qwen05b_active_ledger.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth22_qwen05b_active_ledger.json
```

The pinned Qwen2.5-0.5B safetensors contains **494,032,768**
parameters: **136,134,656** in the tied embedding/output head,
**313,786,368** FFN matrix parameters, **44,040,192** attention
matrix parameters, and **71,552** norm/bias vectors. Every vocabulary
row of the tied matrix is addressed for the output head on each
decoded token. The E128/top-4/rank-8 adapter adds an exhaustive
fp32 router of **11,010,048 addressed bytes/token** and selected
fp32 expert factors of **5,505,024 bytes/token**. These are model
shape counts, not a measured memory trace.

| Ideal addressed payload | MB/token | ms at 40 GB/s | 560 MB design allotment |
|---|---:|---:|---|
| BF16 core + fp32 adapter | 1,004.724 | 25.118 | Over |
| FP32 core + fp32 adapter | 1,992.646 | 49.816 | Over |
| One-byte body, fp32 tied head + fp32 adapter | **919.166** | **22.979** | Over |
| Half-byte body, fp32 tied head + fp32 adapter | 740.253 | 18.506 | Over |
| One-byte head/body + fp32 adapter | **510.762** | **12.769** | Within, before overhead |
| Half-byte head, one-byte body + fp32 adapter | 442.695 | 11.067 | Within, before overhead |

All figures are decimal MB and assume control vectors fp32. The
one-byte/half-byte rows deliberately omit scale tables, format
metadata, alignment, memory-transaction granularity, arithmetic,
attention and dispatch. At 40 GB/s, a 50 tok/s token has 20 ms
**total**, and the project's 560 MB design allotment leaves about
6 ms for the other work. The fp32 tied head alone addresses
**544.539 MB/token**; adding the E128 router and selected factors
already reaches **561.054 MB/token**, before any FFN, attention or
control weights. Even half-byte body weights with that head leave
only 1.494 ms beneath the 20 ms payload-only ceiling. None of the
rows measures actual DRAM bytes or accepted-token throughput.

Code inspection places the current Qwen donor implementation in
`benchmarks/donor_adaptation/engine/donor_engine.c`, separate from
`benchmarks/phase60/engine.c`. Its tied-head branch executes a
fp32 matrix-vector product over the embedding matrix. The native
phase60 engine has its own SSM/SWA and expert format. The Qwen donor
runtime can supply fidelity and kernel references, but its rates
cannot be combined with METH-19/21's adapter quality as one model.
An extension that retains the complete dense donor and fp32 head
would also miss the architectural active-cost thesis.

The next precision experiment should isolate a compressed tied
head and body, measure donor-relative document/task/generation
quality on a bound set, and then verify the composition with this
exact adapter. Only a quality-valid compact core merits native C
export and end-to-end timing. Independently, growing E with RAM
requires sublinear CPU route selection: METH-22 still charges all
128 router rows and says nothing about trained E≈1,700/17,000.
