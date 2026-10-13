# Necessary rank within the token-only prototype/shared linear correction family

Freeze before this bound calculation;only after complete parent auditPASS.
Reuse actual FIT H,source G,group weights and stored covariance spectra.
Recalculate FULL weighted Gram spectrum independently,not only admitted1024
eigenvector identities. No source model/GPU/RESERVED,new learned candidate or
threshold changes. CPU6cores/1200s/OS8GiB,small JSON output.

For any fixed residual operator A,conditional mean mu_t=E[H|token=t] minimizes
E||W(G-A)(H-mu_t)||^2 over every token-only prototype. Other prototypes add
a positive semidefinite between-mean term. A shared rank-k operator's optimal
weighted linear residual energy equals the discarded spectrum of the weighted
response covariance. This is a statement about this FIT linear model family.

Stored prototypes round mu to BF16. Their covariance adds the PSD offset term
C_offset. Its weighted trace is computed directly from exact FIT means and
BF prototypes. Therefore tail_energy(true means)>=tail_energy(stored BF means)
-trace(weighted C_offset),clipped at0. Recompute full eigenvalues to verify
the stored tails. This yields a conservative floor for ANY token-only prototype
and ANY single shared linear rank-k correction within the reference model.

For the frozen two F32/BF16 rounding stages around MuP,use u32=2^-24,
uBF=2^-8,eta1=(1+u32)(1+uBF)-1,eta=(1+eta1)^2-1.
Include subnormal absolute allowance alpha from half-minimum-subnormal values,
actual MuP and group weights. Weighted FIT reference norm=sqrt(5) because
each group's weight is inverse RMS energy. All5 fields<=1% in every FIT case
implies aggregate weighted error eps<=sqrt(5)*.01. Any feasible rounded
candidate's real output norm<=(sqrt(5)+eps+alpha)/(1-eta). Measured source
real-GH versus stored CUDA/BF reference discrepancy is e_source. Necessity:

sqrt(max(tail_energy-BF_mean_offset_energy,0)) <=
e_source+eps+eta*(sqrt(5)+eps+alpha)/(1-eta)+alpha.

The smallest rank meeting this NECESSARY condition is not a sufficient or
constructive quality result. It cannot rule out token-dependent correction
operators,multiple routed subspaces,nonlinear functions,trainable model changes
or relaxed/new whole-model hypotheses. No universal impossibility claim.
F64 eigen/algebra calculations and NumPy BLAS rounding are numerical witnesses;
the separate full parent audit/source identities remain required.
