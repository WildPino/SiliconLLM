# Coupled recurrence/output sensitivity before further basis fitting

12 October2026. **SELECTED / PROPOSED / UNIMPLEMENTED / UNBOUND.**
Recurrent-only covariance plus adaptive allocation repairs the site23 local
operator budget, but site12 remains severe. Do not infer a general384-channel
capacity bound or proceed to long unchanged basis fitting. The current GM is
an average sensitivity without the full postgate RMS Jacobian, and its pairing
with a separately averaged GY discards same-time gate/recurrent dependence.

## Exact operator algebra

Let M=3072, g=SiLU(source gate), u=g*y, r=sqrt(||u||²/M+eps), F=O diag(gamma).
The UNROUNDED F64 source operator is o=F u/r. Its full derivative with respect
to ungated output is

```
J_y = F/r * (I-u*u^T/(M*r²)) * diag(g)
J_y e = F*(g*e)/r - o * [u^T(g*e)/(M*r²)]
```

For head h, G=F^T F, v=G*u and k=||F*u||²:

```
J_h^T J_h = diag(g_h) * [
 G_hh/r² - (v_h*u_h^T+u_h*v_h^T)/(M*r^4)
 + k*u_h*u_h^T/(M²*r^6)
] * diag(g_h)
```

The previously averaged GM retains the first term (modulo its frozen F32 gate/
denominator arithmetic);radial terms and cross-head residual contributions are
not in that proxy. This is a diagnosis of a surrogate, not a proof those omitted
terms alone explain site12 failure. BF16 rounding makes the actual stored
operator discontinuous;J above is for the declared unrounded F64 operator only.

## Finite diagnostic; no new fit/model forward

Reuse all48 histories/sites0/12/23, actual source y/x/gates/out weights and NEW
FIT recurrent-only bases. Start with both existing layouts;source denominators
and fullskip reconstruction fixed. Exact F64 projector residual
e=(V W^T-I)(y-D*x),plus separately recorded actual-scan residual, distinguishes
representation error and finite scan arithmetic. All trajectories were qualified
in the preceding full temporal audit;do not simulate discarded states away.

For each history, retain:

- Exact finite unrounded source output difference for y+e versus y.
- Full J_y e,first-order remainder and radial correction.
- Per-head J_h e_h energies,the total energy and cross-head inner product sum.
- Actual paired mean ||J_h(t)e_h(t)||² versus independent-time trace proxy
  tr[(I-P_h) GY_rec_h (I-P_h)^T GM_h].
- FIT-only corrected mean Gram E[J_h^T J_h],PSD and difference from old GM;
  DEV diagnostics are evaluation,never fitted rank/basis inputs.

Qualify derivatives against independent finite differences on fixed FIT cases/
directions with frozen step sizes/tolerances, plus direct explicit head64 Gram
and full radial/tangent action controls. Complete stored-output/covariance/
energy/hash audit even if approximation errors are large. No sampled subset
may substitute for full case/statistic computation.

## Decision this changes

If omitted radial sensitivity materially explains the proxy mismatch, select a
FIT-only corrected-GM fit. If same-time dependence or cross-head cancellation
dominates,select a coupled finite output objective before more averaged fits.
If finite perturbations invalidate first-order geometry, optimize/check the
actual joint normalization rather than claiming a tangent metric certifies it.
If a larger/nonlinear/conditional memory geometry is needed, state that choice
and measure native active cost before long adaptation. All numerical decision
tolerances/resource limits must be frozen in an implemented protocol before run;
this plan does not execute or admit any alternative.

Site23 accepted local operator is now a candidate for a source-derived native
SSM layout/cost pilot. Its injected full3072 generators/skip/gates/conv buffers
must be included or explicitly replaced;384x256 recurrent bytes alone are not
cost. Separate state96 composition requires a combined source-history test.
No generic full-cost donor runtime may be presented as goal achievement.
Donor-relative conversation,multiple families/scales,n/DRAM/IDs+masses and same
useful artifact>=50 remain required. No T4/delegation;hours-long work follows
the user's announce/bind-launch/pause instruction.
