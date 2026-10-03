# METH-310: ten input regions fail full nonlinear-parent output preservation

Freeze `79b2a23`, exit0, 52.000s, final RSS 1058033664B (budget checks between regions, not sampled peak). No GPU/new collection. All actual source capture/hash/coordinate and tensor checks pass, including gate/up/down alignment, full split-column output closure, basis orthonormality, fit/SVD energy and held-out oracle inequalities. Reuse all22,549 fit and15,977 test down states from nine source parents; associated1536D actual source input defines routing.

The fixed uncentered PCA16 query/spherical10 input regions/rank128 output-space rule **FAILS all four prospective gates**. It fits optimistic projection coefficients using true full outputs; an implemented child has not been trained. This changes298's global projection to input regions, and299's omission to full nonlinear output-space projection.

| Layer/expert | Fit/test rows | Test median/p95 error (%) | Test retained energy (%) | Global fit-rank128 energy (%) | Test-region oracle energy (%) |
| --- | --- | --- | --- | --- | --- |
| 1/0 | 2534/1753 | 83.380/89.356 | 47.293 | 45.272 | 87.394 |
| 1/32 | 3378/1393 | 57.645/88.368 | 72.609 | 67.574 | 96.850 |
| 1/63 | 3504/5423 | 79.204/88.200 | 35.817 | 37.037 | 85.524 |
| 13/0 | 711/855 | 93.265/96.117 | 16.347 | 28.550 | 91.653 |
| 13/32 | 1973/2420 | 83.887/93.024 | 38.900 | 39.594 | 76.446 |
| 13/63 | 2536/2264 | 83.012/87.280 | 33.788 | 38.087 | 78.374 |
| 25/0 | 2284/771 | 65.903/88.931 | 43.142 | 41.324 | 99.558 |
| 25/32 | 2511/631 | 80.895/90.597 | 52.696 | 50.777 | 97.791 |
| 25/63 | 3118/467 | 55.485/87.246 | 67.316 | 63.054 | 99.253 |

Pooled fit error median23.854%/p9573.544%; test81.607%/92.093%. All nine parents fail median1%/p955%; all nine fail99% energy. Four of90 children have<32 fit rows,35 have<16 test rows (some zero). Layers1/13/25 PCA16 query captures21.321%/20.061%/38.154% of uncentered fit-input energy. Record this exposure and fit-domain limitation; no favorable child/source omission or router changes after observation.

Seven of nine parents also have test-fitted regional oracle energy<99%, even though each region may choose its own optimal test output basis under this fixed input router. This is an optimistic finite-sample diagnostic, not a deployable held-out model; small regions can be fitted perfectly. It shows both fit/generalization failures and output-width limits in this specific partition. It does not refute every nonlinear regional model or supervised router.

## Decision and next uncertainty

Close exact PCA16/spherical10/fit-rank128 stand-alone parent conversion. No unchanged student fitting/export/native promotion. Neither quantity of synthetic labels nor exact CPU arithmetic supplies transferred knowledge.

The proposed complete target also has a shared compact core. This experiment's parent approximation used only one regional output subspace, so it does not test whether a shared output subspace plus regional residual can preserve the parent. [Shared-core residual proposal](GIGACHAT_SHARED_REGIONAL_BOUND_PROPOSAL_20261003.md) names that changed variable and the required next prospective test. It cannot qualify existing failed timing profiles, whole-MoE composition or useful new n.

## Reproducibility

[Protocol](METH_310_GIGACHAT_REGIONAL_SUBSPACE_PROTOCOL_20261003.md), [raw result](meth310_gigachat_regional_subspace_result.json) SHA `1ce6972c99d2db3057b989ec327c7d3e015f2be2d125107cadc3d1ce4058cac1`, controller `8fed130998c32a3cd1ba342958bce2ab179e9834b6bc64b666c30ab8fb352719`, protocol `676df994c43f654191cdc585db27c3b52486f406ada6ce52cc19f687aa4d5f4d`. Every state's error, all child counts/actual widths, fitted keys and query/basis hashes are retained. Captures/tensors remain locally bound to the frozen donor assets; no generic donor port resumed.

Command: `.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth310_gigachat_regional_subspace.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth310_gigachat_regional_subspace_result.json`. Refuses existing result/failure; preserve observations. Mathematical FP64 projection uses full source post-SwiGLU outputs and no realized query/child atI4. Both diagnostic domains were previously consumed; future full quality needs untouched data. Source pipeline, accepted>=50 on same artifact, large-RAM actual routing/DRAM and cross-family/100B evidence remain unqualified.
