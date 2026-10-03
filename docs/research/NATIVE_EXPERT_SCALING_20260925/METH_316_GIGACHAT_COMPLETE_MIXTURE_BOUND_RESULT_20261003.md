# METH-316: complete source mixture rule fails all gates

Freeze `b2f2a63`, exit0. All37,381 actual states/six cases completed in
284.828s; maximum CHECKED RSS3,252,183,040bytes, final589,647,872bytes.
This is algebra cost, not native model throughput.

| Case | Regional median / p95 | Joint-span oracle median / p95 | Regional / oracle retained energy |
| --- | --- | --- | --- |
| layer1 FIT | 19.209% / 30.781% | 14.253% / 25.205% | 95.950% / 97.650% |
| layer1 TEST | 53.790% / 79.047% | 46.364% / 71.984% | 64.809% / 72.198% |
| layer13 FIT | 40.598% / 49.225% | 33.697% / 42.137% | 85.185% / 90.179% |
| layer13 TEST | 66.767% / 75.098% | 59.084% / 68.890% | 55.502% / 64.094% |
| layer25 FIT | 11.175% / 16.577% | 7.662% / 13.197% | 98.803% / 99.416% |
| layer25 TEST | 21.560% / 60.015% | 18.114% / 54.669% | 89.586% / 91.721% |

Primary four gates ALL FAIL: every-case1%median/5%p95, every-case99%energy,
no>.5pp loss versus matched parent-global field, every-child32FIT/16TEST.
1250/1920 children have<32FIT;1381/1920 have<16TEST. Separate joint-span
diagnostic gates BOTH FAIL. No copied/poorly covered640 useful functions claim.

All source reconstruction/field/operator controls pass. Entire315 layer1/13
fit fields/key hashes match exactly; every prior completed-case row error,
rank and child ID matches within the prospective numerical tolerance.
37,381 conditioned Cholesky projections, zero data-driven SVD fallbacks;
minimum accepted rcond0.000819253. Maximum all-state normal-equation error
8.781e-16;192 fixed-index SVD reference comparisons, maximum relative
projection difference2.063e-15. Model-free duplicate-column fallback,
independent projection and zero fault controls pass. No precision/row/rank
or gate change rescued315; its interrupted record remains preserved.

[Full per-state raw result](meth316_gigachat_complete_mixture_bound_result.json)
SHA `021670ecf043eb81cfd74fb86d2354740c7dfe6a31a8cbc3cfb6c6ebc39e07b4`.
Source314 archives and actual original GGML/BF16 targets unchanged.

**Decision:** close this fixed complete-mixture shared256/regional128 rule
before student training/export. Even unrestricted true coefficients within
its fields cannot meet local gates; coefficient training alone cannot repair
them. Changed output representation/router/coverage requires a prospective
new candidate. This does not refute every nonlinear or jointly trained
conditional architecture. Local consumed/anchor-conditioned states are not
untouched whole-model quality. Useful large-n, actual CPU cost and same-artifact
>=50 accepted tokens/s remain unverified.
