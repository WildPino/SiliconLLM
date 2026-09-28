# METH-134: sparse CPU-master path passes a one-layer E12800 apparatus gate

**Decision.** Keep the third-tier, CPU-master sparse-gradient mechanism
as an executable starting point for a separately frozen E12800
training and quality run. This experiment establishes correct sparse
gather/backward/update and exact cloned-factor initialization on one
actual Qwen layer. It does **not** establish 12,800 useful distinct
learned experts, full-model quality, CPU route/LUT latency or accepted
token throughput.

The [protocol](METH_134_SPARSE_E12800_TRAINING_PROTOCOL_20260928.md)
was committed at `35478cf` and the
[apparatus](../../../benchmarks/native_expert_scaling/meth134_sparse_e12800_training.py)
at `7cfe79e` before execution. The
[machine-readable result](meth134_sparse_e12800_training_result.json)
binds the exact centered BF16 METH-126 bank SHA-256
`1d9071456a644344d46203ad7ded8fe2c4d01ca0580b5e41718366783408aff1`
and METH-125's 256 actual E1280 pre-MLP state vectors SHA-256
`f7af00b4b4ce417664848950770f63b78520ca3c983748a692640704c203b699`.
It reads layer 0. A seeded, **untrained** third projection/key set
chooses one of ten grandchildren below each selected E1280 child;
the four original selected children, gates and shared A remain fixed.
Every grandchild initially copies its source child B exactly.

| Frozen apparatus check | Measured result |
|---|---:|
| E1280 source B, FP32 layer 0 | 36,700,160 B |
| E12800 CPU-master B, FP32 layer 0 | 367,001,600 B |
| Projected 24-layer FP32 CPU B master | 8,808,038,400 B |
| Projected 24-layer BF16 inference B | 4,404,019,200 B |
| Selected occurrences / unique source children / unique grandchildren, 256 states | 1,024 / 275 / 467 |
| Grandchild ID / 10 equals selected child ID | all |
| Cloned-factor FP32 output max absolute error before update | 0 |
| Small duplicate-ID gradient max absolute error vs dense PyTorch | 0 |
| Small one-SGD-step bank max absolute error | 1.19e-7 (limit 1e-6) |
| Small unselected-row change | 0 |
| Actual one-step gradient L2 / unique rows | 3.47e-6 / 467 |
| Actual selected rows changed / nonselected rows unchanged | 464 / 12,333 |
| Forward B transfer / raw gradient transfer, one layer | 29,360,128 / 29,360,128 B |
| Summed unique gradient storage, one layer | 13,389,824 B |
| Elapsed / process RSS / peak allocated GPU | 2.657 s / 2.189 GB / 141 MB |

The small oracle deliberately repeats 11 of 44 selected IDs. Its
selected-row gradients are exactly equal to dense autograd, and a
single SGD update agrees within 1e-6; every unselected row is
unchanged. On actual states, three selected rows round back to their
original FP32 values after the small one-step update, which is why
464 of 467 selected rows changed. No dense E12800 bank or gradient
was allocated on the GPU. The one-layer process remained below the
frozen ten-minute, 4 GiB RSS/GPU and 500 MB new-disk stops. No T4 was
used.

The full 24-layer memory figures are arithmetic projections, not
measured peak training memory. The apparatus omits optimizer moments,
model-wide backpropagation, checkpoint/recompute, throughput under
real training batches, router/key learning, corpus diversity and
held-out semantics. The 256 vectors and source checkpoint were viewed
in earlier experiments, so cloned output identity is a binding check,
not a fresh quality test. One-step changes at a tiny gradient scale
cannot be called meaningful specialization. The next training
protocol must account for CPU optimizer state, selected-row transfer
at all 24 layers and a new-source donor-relative quality audit; the
CPU third-tier router and selected-factor cost need native measurement
before any RAM-scaled latency claim.
