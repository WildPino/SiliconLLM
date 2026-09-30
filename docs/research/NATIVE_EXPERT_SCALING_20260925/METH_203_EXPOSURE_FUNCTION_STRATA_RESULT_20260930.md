# METH-203: high exposure raises child spread but remains far below the gate

The [frozen protocol](METH_203_EXPOSURE_FUNCTION_STRATA_PROTOCOL_20260930.md)
and [runner](../../../benchmarks/native_expert_scaling/meth203_exposure_function_strata.py)
were committed at `49dae2a` before execution. The first run stopped on
a tensor-shape error while constructing the MLP-energy denominator;
it produced no result. The [failure record](meth203_exposure_function_strata_result.failure.json)
is retained. Commit `e16d344` fixed only that shape before the
successful rerun. All eight METH-201 source windows, 24 layers,
selected-child reconstruction and full MLP outputs reproduced the
prior result, and all three exposure bins reconcile in counts and
energies. The exact METH-175 bank and its training route counts were
bound by SHA-256. No new source or target was used.

Across the 24 METH-175 layers, 273,364 of 276,480 content-child
slots were selected during training. The median selected child
received 117 selections; 51.83% of selected children received fewer
than 128, 81.30% fewer than 512, and 91.31% fewer than 1,024.
Each B matrix has 7,168 stored values. Selection count alone is not
an independent-sample count or a rank/learnability bound.

On the 95,616 viewed real-state content selections, the sibling
output RMS spread around the content-parent mean was:

| Parent training content selections | Viewed selections | Sibling/mean RMS | Sibling/full-MLP RMS per selected path |
| --- | ---: | ---: | ---: |
| <2,048 | 13,036 | 0.556% | 0.00113% |
| 2,048–8,191 | 30,118 | 1.039% | 0.00250% |
| ≥8,192 | 52,462 | 1.885% | 0.00624% |

Higher exposure is associated with more child-specific function on
this sample, but even the high bin ranges only 1.674–2.592% sibling/
mean across layers (median 2.027%). It has at least 1,450 viewed
selections in every layer. **Zero of 24 high-exposure layers** reaches
both frozen 10% sibling/mean and 1% sibling/full-MLP thresholds.
The denominator for the last column repeats each token's full MLP
energy once for each selected content path, as specified; it differs
from METH-201's once-per-token pooled denominator. Neither version
approaches 1%.

The [raw result](meth203_exposure_function_strata_result.json),
SHA-256 `edb309d640e8bec3442e09ece9297b2399854b3270fd70b303a3c81c2db3c26f`,
contains per-window/layer/bin energies, exact baseline checks, exposure
quantiles and runtime. The successful local RTX 3060 run took 46.328 s
including initial file binding, with 4.673 GB peak GPU allocation and
11.332 GB ending RSS, within the frozen limits. No T4 was used.

**Decision:** exposure is a plausible contributing constraint, but a
repeat of the same hash-route/shared-base long training is not
justified by low exposure alone. Change the route/training coupling,
then run a bounded specialist pilot and demand measured route-specific
function plus source-held-out quality before larger-n promotion. This
diagnostic neither proves a causal effect of exposure nor validates
10B/100B transfer, CPU LUT scaling or native end-to-end rate.
