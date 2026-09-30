# METH-180: real 10B pretrained expert rank screen result

**Decision: reject direct, weight-only rank-192 int8 factor export.** The [prospective protocol](METH_180_GIGACHAT_EXPERT_RANK_SCREEN_PROTOCOL_20260930.md) required median rank-192 optimal Frobenius energy at least 95% and every sampled matrix at least 90%. The measured median is 47.243%; the maximum of all 27 is only 54.113%. This is a screen on matrix reconstruction, not a full-model quality result.

## Source, apparatus and audit

- Source: locally bound `ai-sage/GigaChat3.1-10B-A1.8B-bf16` at revision `189fff27a1dee68473960c3d5bca53e0e07a3191`; source-binding report SHA-256 `0cd0699fb100e66eb89b37ed088a942228446c5b20939dcc682eec186772182f`. The [JSON result](meth180_gigachat_expert_rank_result.json), SHA-256 `ecc9a3432a25a4a7fc0f2881c421947aeb097b7a4cf32a746ebcdefa10081b8d`, records source index/config, three shard and all 27 raw BF16 tensor hashes.
- Sample fixed before measurement: base layers 1, 13, 25 × experts 0, 32, 63 × gate/up/down projection. All 27 identities are unique and present; shapes are 1280×1536 or 1536×1280. Layer 26 MTP is excluded. This is 27 of the base model's routed-expert projections, not a census.
- [Script](../../../benchmarks/native_expert_scaling/meth180_gigachat_expert_rank_screen.py): BF16→FP32, six-thread local CPU LAPACK full singular values, sum-of-squares check, optimal unweighted rank reconstruction. The maximum relative discrepancy between singular-value squared sum and input Frobenius squared norm is `8.43e-7`. The run finished in 66.985 seconds; maximum recorded row RSS was 471,498,752 bytes, within the frozen 15-minute/6-GiB limits. No GPU or T4 was used.
- The first invocation stopped before its first SVD because the SciPy `svdvals` API did not accept `lapack_driver`. The [preserved apparatus failure](meth180_gigachat_expert_rank_result.pre_svd_api_failure.json) has zero spectral rows. The call was corrected to `scipy.linalg.svd(..., compute_uv=False, lapack_driver='gesdd')` and the entire sample rerun; no rank result was used to choose the correction.

## Measured best-case weight reconstruction

| Rank | Min energy | Median energy | Max energy | Matrices ≥95% |
|---:|---:|---:|---:|---:|
| 64 | 17.824% | 21.773% | 30.477% | 0/27 |
| 128 | 31.729% | 36.063% | 43.719% | 0/27 |
| **192** | **43.320%** | **47.243%** | **54.113%** | **0/27** |
| 384 | 68.611% | 71.539% | 75.511% | 0/27 |
| 512 | 79.841% | 81.936% | 84.653% | 0/27 |
| 768 | 93.219% | 94.154% | 95.106% | 3/27 |

At rank 192 the median is 51.076% for gate, 47.042% for up and 46.776% for down; by layer it is 48.646%, 46.731% and 47.042% for layers 1, 13 and 25. Thus the failure is present across the sampled projections and depths, not one anomalous tensor. The JSON gives every residual RMS and row identity.

The original projection has 1,966,080 elements. Rank 192 has 540,672 int8 factor elements, half the **idealized** 0.55-byte/weight Q4 projection payload, before factor scales and implementation costs. Rank 768 has 2,162,688 int8 elements, twice that idealized Q4 payload, yet the median still misses 95%. Actual mixed-precision GGUF bytes, kernel traffic, and activation behavior can differ from this arithmetic; [METH-01](METH_01_GIGACHAT_Q4_ACTIVE_LEDGER_20260925.md) remains the source for the actual donor payload.

## Interpretation and next method boundary

The frozen rank-192 weight-only screen fails decisively, so do not build a full export around direct unweighted SVD factors at that rank. Low unweighted Frobenius energy does **not** by itself establish a BPB or task-quality loss: activation frequencies and output sensitivity can make some directions more important than their weight norm. It also does not rule out activation-aware or trained factors, structured neuron preservation, outlier handling, or a different representation. Any such candidate needs a separately frozen source-bound quality comparison, measured active bytes and native selected-kernel cost before a same-artifact full-model rate claim. This screen says nothing about useful distinct experts at E12,800/E128,000 or 100B transfer.
