# Saved-only geometry after fixed local recovery

9 October 2026. Freeze before SVD/correction observations. Recovery completed
256 effective updates/site; site0 DEV error ratio .96285, site23 .41972.
Absolute fidelity fails both sites. No more training is selected by this protocol.

Uncertainty: does the learned correction miss the required direction, and how
much of the observed DEV input/output lies outside the FIT linear span?
Reuse all four captured original BF16 x/y and selected before/final after packets;
verify exact hashes and F64 aggregate errors against their existing reports.
No model weights/forwards, generation, updates, export, native or RESERVED calls.

For desired correction r=y-y_before and learned correction c=y_after-y_before,
report norms, cosine, oracle scalar dot(r,c)/norm(c)^2 and the resulting residual.
Check norm(y_after-y)^2=norm(r)^2+norm(c)^2-2dot(r,c) to 1e-12 relative precision.
Oracle values use source targets, are diagnostic and cannot be used at deployment.

For uncentered FIT X and source Y (274 rows x2048 each/site), compute F64 thin SVD.
Numerical rank uses singular value >1e-10*s_max. Separately keep the fewest modes
covering >=99% squared singular energy. Report stable rank and each DEV matrix's
normalized residual outside both FIT spans. Preserve every singular value.
BF16 rounding can populate small modes; numerical and energetic ranks differ.
Linear subspace coverage is descriptive, not a nonlinear generalization bound,
fraction of knowledge, causal diagnosis or an architecture-quality certificate.

Decision use: scalar-oracle failure rules out amplitude-only repair of the actual
learned correction on these operands. Limited support plus FIT/DEV separation
motivates broader calibration before longer fixed-prefix fitting, without proving
coverage is the only cause. Neither possible outcome licenses full-site deployment.

CPU six cores, no concurrent model/compiler/benchmark, 120s family/20s reserve,
2GiB OS/zero GPU allocation/8MiB output/log4MiB. Stop on bounds/nonfinite/hash or
identity failure; retain first fault. No success threshold is retrofitted to SVD.
