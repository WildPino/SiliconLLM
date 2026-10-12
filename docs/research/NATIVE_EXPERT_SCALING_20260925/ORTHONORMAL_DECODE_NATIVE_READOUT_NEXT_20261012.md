# Exact orthonormal decoding and native scalar-norm readout: next record

12 October2026. **Historical prospective record; now COMPLETE**, see
[actual result and full independent audit](GAUGED_NATIVE_READOUT_RESULT_20261012.md).
The following text records the original proposal,not current execution status.
Originally **PROPOSED/UNIMPLEMENTED/UNBOUND**. Follow broad exact residual
coverage/gradient audit. No new epoch or native artifact yet. Selected to connect
the actual trained-source factorization back to original engine.c normalization
and head kernels before spending more optimization time on the poor fit.

## Exact map gauge; no information or quality improvement claimed

Current column map: z=W^T(h-mu),y=mu+V z. Thin QR V=Q R, Q^T Q=I. Define
E=W R^T,c=Q^T mu,mu_perp=mu-Q c,b=c-E^T mu. Then

`t=E^T h+b = R z+c`

`y=Q t+mu_perp`, with Q^T mu_perp=0 in real arithmetic.

Same reconstructed source vector and same readout probabilities: this is an
exact representation/gauge transformation,not a better model or a new projector.
Generic W^T V!=I is NOT repaired by QR. If E^T Q!=I,retain that fact and do not
assume an identity source-reencoding/skip. True causal code acquisition and
source nonlinear approximation remain separate missing steps.

## Scalar norm and affine reader compilation

For source dimension Ds=2048,target d=256,source epsilon eps_s=1e-5:

`epsilon_t=(||mu_perp||^2+Ds*eps_s)/d`

`r_t=sqrt(||t||^2/d+epsilon_t)`

`r_source(y)=sqrt(d/Ds)*r_t`.

For actual source reader A including source input gamma/multipliers,

`A y/r_source = C (t/r_t) + beta/r_t`

where C=sqrt(Ds/d) A Q,beta=sqrt(Ds/d) A mu_perp.
Thus the dense norm metric/cross term can be eliminated by coordinate choice,
while retaining an artifact-specific scalar epsilon and normalized reader bias.
With zero mu_perp,epsilon_t=8e-5,beta=0. Do not silently use engine's fixed1e-5
or omit beta when the source center has an orthogonal component.

This applies to the actual final head and input-normalized trained readers;
it does NOT convert source nonlinear SSM/gating/FFN operators by itself. Source
gamma folds into readers,not an averaged projected norm gamma.

## Concrete next implementation and gates to freeze

1. After full broad audit closure,bind the chosen source parameter point and
   original engine/actual source/config/runtime/compiler extents. Derive/retain
   Q/R/E/b/mu_perp,scalar epsilon,head C/beta and coordinate transformation.
   Verify full decoded map/norm/every head coefficient/FIT+DEV probability
   preservation against the original factor predictor,not source quality admission.
2. Native component uses original hsum256/dotf/matvec bodies unchanged. A named
   wrapper must supply scalar epsilon and inverse RMS for beta;state the precise
   minimal change to the original RMS body. Original engine.c remains immutable
   during this component qualification. No generic donor runtime introduced.
3. Export actual F32 head/bias/epsilon and encoded states. Validate on all8808
   existing source labels,full vocabulary,normalization and probability/IDs,
   preserve every failure and independently audit source-map/precision outputs.
   Fix tolerance/budget/output format BEFORE observation. Do not assume F32
   equivalence or substitute sampled outputs for a claimed full-state check.
4. Price selected head67MB,bias~.26MB and normO(256)/biasO(65537) operations,
   conditional chart K costs and actual CPU/DRAM work. These remain component
   measurements,not end-to-end accepted token throughput or useful expert scaling.

Function-preserving QR may also expose poor optimizer parameter conditioning:
V->V T,W->W T^-T leaves V W^T unchanged for invertible T,while the current
normalized whole-group gradient is not gauge invariant. Any comparison of
gradient response before/after gauge must be a separate FIT-only frozen study,
with identical initial output and independently verified gradient. Do not infer
that conditioning is the sole cause of the remaining errors without that test.

Core dynamics/own-history behavior,ternary functions,useful n/CPU IDs+mass/physical
DRAM and same useful artifact>=50 plus multiple families/scales remain required.
No T4/hours-long optimization chosen. Announce budget/stops before work;follow
user's pause rule only for an actual hours-long frozen job. No Luna/delegation.
