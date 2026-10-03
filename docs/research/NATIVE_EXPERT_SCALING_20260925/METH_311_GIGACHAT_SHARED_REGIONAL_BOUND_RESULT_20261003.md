# METH-311: shared output space helps, fixed conversion still fails

Freeze `3c16483`, exit0, 44.484s; maximum checked RSS 1483382784B, not OS sampled peak. No GPU/new capture. Reuse all22,549/15,977 actual fit/test nonlinear parent states, exact310 query/key hashes and every route count. Source/hash/coordinate/full-output closure, orthonormality/energy and fixed-common test-oracle checks pass.

**All four prospective gates FAIL.** Adding one shared fit-output256D basis per layer improves every parent's held-out output energy by9.477–29.095 percentage points versus310, but median error remains42.344–80.517%, above1%/5% gates. The common/regional coefficients remain optimistic projections requiring true outputs, not executable learned functions.

| Layer/expert | Test median/p95 error (%) | Full output energy (%) | Matched global residual energy (%) | Fixed-common test-region oracle (%) |
| --- | --- | --- | --- | --- |
| 1/0 | 69.541/74.837 | 63.513 | 62.391 | 91.418 |
| 1/32 | 47.189/73.667 | 82.085 | 79.732 | 97.945 |
| 1/63 | 62.904/71.492 | 58.976 | 59.891 | 90.904 |
| 13/0 | 80.517/83.719 | 37.520 | 45.666 | 93.876 |
| 13/32 | 68.929/76.821 | 58.637 | 59.546 | 84.210 |
| 13/63 | 62.146/66.712 | 62.884 | 65.241 | 87.994 |
| 25/0 | 58.032/77.911 | 57.468 | 54.690 | 99.664 |
| 25/32 | 65.489/76.143 | 67.818 | 67.629 | 98.572 |
| 25/63 | 42.344/71.194 | 78.922 | 77.511 | 99.548 |

Pooled fit median18.879%/p9558.131%; pooled test63.968%/77.578%. All nine parents fail error and99%energy. Four regional fits are>.5pp worse than matched global residual. Known exposure gate remains unchanged:4/90 children fit<32,35/90 test<16. Seven/nine fixed-common test-region oracles remain<99%; small test regions can be fitted perfectly, so this is no population bound.

Decision: close exact common256/PCA16/spherical10/fit-residual128 independent-parent conversion. No student fitting/export or native promotion. This does not refute every joint nonlinear mixture approximation: a complete source MoE output sums four weighted parents plus shared output, while these gates demand preservation of each parent separately. The next necessary target is that complete mixture on actual source inputs, with independent arithmetic/route controls. Reuse captured inputs rather than repeating49min source collection without need.

[Frozen protocol](METH_311_GIGACHAT_SHARED_REGIONAL_BOUND_PROTOCOL_20261003.md), [raw result](meth311_gigachat_shared_regional_bound_result.json) SHA `f9862093fd96f4025dd61a7da49f43e34e7506d8be61451468db5f57591489bd`. Controller `b7cba01652ca92a097cdf9eced8044c0e805cd3c005aaf1832d183a45127d08c`, protocol `fdf43abc36c84297810e373bb9f0ea10bee5afca56f2aa377e967eb02cf8a95b`. All state errors, shared/residual basis hashes, fixed route counts and child exposures retained.

Command: `.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth311_gigachat_shared_regional_bound.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth311_gigachat_shared_regional_bound_result.json`; refuses existing result/failure. No complete attention/causal/native model, quantized/shared coefficient functions, useful added n or same-artifact50token/s proven. Domains previously consumed; final whole quality needs untouched data.
