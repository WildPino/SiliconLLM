# METH-316: equivalent conditioned joint projection

Freeze before observations,2026-10-03. Execution repair of315's wall stop,
preserved at `c115b5a`. No change to source data, fitted fields, coefficients
allowed, target, routes, ranks, coverage requirements, scientific gates or
20min/12GiB budget. All37,381 states and six cases retained.

Use the complete315 protocol for source reconstruction, fit-only common256,
query16, seeded315000+layer*100+parent spherical ten-way routing, regional128
and parent-global residual fields, gated sum, controls and decisions.
Fitting remains six BLAS threads. Every fitted field/key/count/hash in
completed315 layers must match EXACTLY the immutable315 partial record
SHA41cf6365c6268d2e47d76fadd3f309c2adeb99db2f85e3bc711c5456d8a265ae.
All previous completed case child IDs/ranks must match; every previous row
error in all four arms must differ by<=1e-8. These are equivalence gates,
not licenses to select rows or relax scientific quality/exposure gates.

## Projection repair and numerical controls

Use one BLAS thread during each per-state projection phase. For concatenated
regional columns B, form FP64 Gram B.T B and attempt lower Cholesky.
Accept only LAPACK dpocon status0, finite reciprocal1-norm condition>=1e-6;
then project B*solve(Gram,B.T*trueResidual). Otherwise use original thin
gesdd SVD and original eps64*max(shape)*largest-singular numerical rank.
Empty space projects zero. LRU64 caches exact sorted-child fields and factors.
No regularization, truncation, ridge, fitted/test coefficients or rerouting.

EVERY state requires maxabs(B.T*oracleError)/fullOutputNorm<=1e-8 and
oracleErrorNorm<=primaryErrorNorm+1e-10*fullOutputNorm. For FIXED first32
states of EACH case compare independent original-SVD projection: relative
prediction difference<=1e-8 and exact numerical rank. Retain maxima/counts,
Cholesky/fallback usage and minimum accepted rcond. Original SVD fallback
orthonormality1e-10/common orthogonality1e-8 retained. Post-freeze model-free
controls require independent-column exact projection, duplicate-column SVD
fallback rank and projection, and detection of a zero-projection fault.

## Gates, command, stop and scope

Primary unchanged: every-case median<=1%,p95<=5%,energy>=99%, regional
energy>=matched parent-global energy-.005; every1920 child>=32FIT/>=16TEST.
Separate joint-oracle1%/5%/99% diagnostic gates cannot override these.
Wall20min/RSS12GiB checked between fits/each512 states; preserve failure,
never overwrite results. No other model/rate job, no GPU or source collection.

`.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth316_gigachat_complete_mixture_bound.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth316_gigachat_complete_mixture_bound_result.json`

This closes the incomplete diagnostic without changing its scientific
question. Algebra runtime is not a deployed CPU routing/LUT benchmark.
True-output coefficients remain unavailable to a student. No whole-model
quality, useful extra n, native accepted rate or general-family claim.
