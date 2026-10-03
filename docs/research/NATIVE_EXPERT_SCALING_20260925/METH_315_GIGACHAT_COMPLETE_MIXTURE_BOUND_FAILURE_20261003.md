# METH-315: partial complete-mixture diagnostic, wall stop

Freeze `58af1c1`; exit1 after1217.187s at `layer13_test_joint_oracle`,
`AssertionError('M315 wall/RSS stop')`. The prospective20min wall gate
stopped at a512-state budget check. No six-case result or promotion.

| Completed case | Regional median / p95 | Joint-span oracle median / p95 | Regional / oracle retained energy |
| --- | --- | --- | --- |
| layer1 FIT | 19.209% / 30.781% | 14.253% / 25.205% | 95.950% / 97.650% |
| layer1 TEST | 53.790% / 79.047% | 46.364% / 71.984% | 64.809% / 72.198% |
| layer13 FIT | 40.598% / 49.225% | 33.697% / 42.137% | 85.185% / 90.179% |

All three completed cases fail prospective1%median/5%p95/99%energy
even with true-output joint coefficients. The fixed fields cannot meet
these local gates through coefficient-function training alone. This is
not a general rejection of conditional architectures or joint-trained fields.

[Immutable partial record](meth315_gigachat_complete_mixture_bound_result.failure.json)
SHA `41cf6365c6268d2e47d76fadd3f309c2adeb99db2f85e3bc711c5456d8a265ae`
retains fitted layer1/13 fields and three complete per-state case arrays;
the interrupted layer13 TEST case is absent. Source archives unchanged.

An execution repair may use conditioned Gram Cholesky with SVD fallback,
one BLAS thread for projections, all-state optimality/containment controls
and fixed-index SVD comparisons. Preserve exact fitting, all states, gates
and20min/12GiB budget. Freeze a NEW protocol before that run. No native
speed, whole-model quality, useful extra experts or final goal claim follows.
