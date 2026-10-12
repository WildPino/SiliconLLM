# FIT recurrent-output bases, full source skip retained

12 October2026. **SELECTED / PROPOSED / UNIMPLEMENTED / UNBOUND.**
Selected after the complete recurrent/full-skip comparison and its full
independent numerical audit PASS. This changes the basis target, not merely the
evaluation of an unchanged basis. Local model gates remain failed.

## Isolated problem

Old dual balanced bases fit `GY=E[y y^T]`, while `y=y_rec+D*x`. With a full skip
branch, recurrent memory should represent `y_rec=y-D*x`. At site12 recurrent and
skip projection errors have negative inner products on nearly every DEV history;
the all-output basis can exploit cancellation inappropriate to preserving
recurrent content separately. Removing skip error alone barely changes the
remaining ungated recurrent error. This does not prove a384-channel capacity
floor, much less failure of conditional/nonlinear representations.

## Finite next implementation

FIT-only equal-case dual balanced bases for `GY_rec=E[y_rec y_rec^T]`, per48
heads/64 source channels, both fixed8 and adaptive384 budgets. Retain the same
source-gate/output sensitivity `GM` first to isolate the covariance change.
Compute square roots and balanced SVD `sqrt(GM)*sqrt(GY_rec)`; choose adaptive
ranks with the same min1/head and336 remaining squared-singular-value policy.
Freeze all input weights/histories, numerical checks and unchanged local quality
budgets before observing new bases. No DEV basis fitting/rank choice.

Generate actual compact scans with the NEW W (old latents cannot be reused for a
changed encoder). Compare with direct source projected recurrent output on all
histories; retain independent scalar sequence and full output reconstruction
audit. Decode recurrent output and add FULL source `D*x`, then jointly gate and
normalize. Repeat both old full-skip and projected-skip controls to retain the
causal attribution. F64/BF16 diagnostic precision bridge failures from the prior
run remain historical; no implicit F32 equivalence admission.

`GM` averaged independently of `GY_rec` is still a separable surrogate; it is
not the coupled sequence/output optimum. If it fails, inspect actual gated RMS
Jacobian sensitivities and within-head recurrent singular structure before
selecting a stronger objective or changing the compact-core dimension. Do not
repeat unmodified all-output fitting.

## Composition and deployment boundaries

The instantaneous operator is stateless ONLY given post-convolution source x.
Source depthwise conv4 buffers, input/gate generators and full residual-width
features remain history dependent and injected here. Report these retained
states/work and the3072 skip features;384x256 state bytes alone are not total
causal memory or native active cost. Per-head common scalar decay is the reason
within-head projection commutes; different-head decays cannot be mixed freely.

State256->96 projection is separately tested, not automatically composable with
these channel bases. A tensor-product state/read/write projection needs its own
combined source-history test and native layout. Conditional chart changes would
need explicit state transport between charts; routing a new static decoder does
not recover discarded past information.

If local gates pass, implement a pretrained causal operator/native cost pilot
before long joint distillation with the verified moving-decoder source moments.
If they fail, choose the next algebraic repair from the residual structure.
No more unchanged emission oracle iteration. Actual conversation/family/n/DRAM/
structured IDs+masses/same useful artifact>=50 remain required.

Provisional producer1800s/audit1800s, OS6GiB, GPU allocated4GiB/reserved5GiB;
specific binding and output limit must be frozen before execution. This document
does not launch a run. Any required hours-long run follows announce/launch/pause;
no Luna/delegation/T4 allocation.
