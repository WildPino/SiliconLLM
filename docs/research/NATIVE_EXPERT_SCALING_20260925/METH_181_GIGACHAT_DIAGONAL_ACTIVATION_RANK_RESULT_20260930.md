# METH-181: disjoint activation-weighted expert rank screen result

**Decision: reject diagonal activation-weighted rank-192 direct export.** The [frozen protocol](METH_181_GIGACHAT_DIAGONAL_ACTIVATION_RANK_PROTOCOL_20260930.md) required at least 95% median and 90% minimum retained energy on a disjoint-domain input-second-moment proxy, plus no matrix more than 0.5 percentage point worse than ordinary SVD. Measured retained energy is 49.152% median and 42.433% minimum; 0/27 matrices reach 90%. The worst weighted-minus-ordinary difference is **-4.227 percentage points**. No full-model conversion is licensed by this screen.

## Bound inputs and verification

- Reused the exact 27 METH-180 BF16 expert tensors: layers 1, 13, 25 × experts 0, 32, 63 × gate/up/down, with every raw tensor hash rechecked. The source index/config and all three shard hashes were also rechecked against the [METH-180 result](METH_180_GIGACHAT_EXPERT_RANK_RESULT_20260930.md).
- Fit moments came from the preexisting 106-chunk English/code/technical source-ID BF16 imatrix, SHA-256 `ef5de87fc4fa0d1382ed2988e59e9472fb1d7fb2c9bed0e8edcced6ff9bd4f9c`. Evaluation moments came from the separate 125-chunk Cyrillic source-ID BF16 imatrix, SHA-256 `5c529f9009f0f241dfc50d0a1f598b0e183c137693a951f04963dc729a344d08`. The merged 231-chunk imatrix was not used to fit. Each of the 27 expert/projection instances had at least **711 fit** and **467 evaluation** observations, exceeding the frozen 64-observation floor.
- The [implementation](../../../benchmarks/native_expert_scaling/meth181_gigachat_diagonal_rank.py) reads GGUF v3 F32 second-moment/count tensors directly, uses six local CPU BLAS threads, and performs a rank-192 SVD of each `W diag(sqrt(v_fit))`. The ordinary rank-192 SVD of `W` has identical ideal factor count. It evaluates both against `v_test`; a test-domain weighted SVD is also computed as an oracle for this diagonal proxy. Direct reconstruction agrees with the analytical fit-domain SVD energy to within `5.78e-8` absolute. No GPU or T4 was used. Runtime was 38.875 seconds; maximum recorded row RSS was 532,922,368 bytes, within the frozen 15-minute/6-GiB stops.
- [Machine result](meth181_gigachat_diagonal_rank_result.json) SHA-256 `5b226744cde42404a4811ff32740a6b744328db0d697d3acafaf8880c0f3d7b0` contains all 27 identities, counts, energy and residual values. An independent read confirmed 27 unique rows, exact stored SHA, every numerical gate and the aggregate summary.

| Rank-192 diagonal proxy | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Fit-domain weighted optimum | 44.753% | 51.915% | 68.088% |
| **Disjoint-domain weighted factors** | **42.433%** | **49.152%** | **62.388%** |
| Disjoint-domain ordinary factors | 42.775% | 48.229% | 58.848% |
| Disjoint-domain oracle optimum | 44.684% | 52.297% | 71.433% |

The weighted fit gains 0.974 percentage point over ordinary SVD in the median **paired row difference**, but loses on 3/27 rows. The largest loss, -4.227 points, is layer 25/expert 32/down. Even the test-domain oracle has only 52.297% median at rank 192, so the failure cannot be explained solely by English-to-Cyrillic calibration shift under this diagonal metric. The oracle is diagnostic and cannot be used to choose factors for held-out quality.

## Scope and method consequence

The imatrix records input second moments, not cross-channel covariance. The retained-energy fraction is an output-error proxy for linear projections under a diagonal covariance assumption; it is neither actual donor BPB nor a model-level perturbation estimate after SwiGLU, routing and later layers. The two calibration domains are distinct, but are not a broad task-quality set. This result rules out **this direct diagonal-weighted rank-192 export as the next candidate** under its frozen proxy gate. It does not rule out full-covariance activation methods, trained corrections, structured neuron preservation, or a different target geometry. Any new representation should first price its complete active path, including GigaChat MLA and head, then undergo donor-relative quality and native cost gates on one artifact. The original goal still requires useful additional experts and CPU routing as the bank grows with RAM; this 64-expert donor screen alone does not demonstrate either.
