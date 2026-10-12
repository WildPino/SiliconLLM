# Exact recurrent coordinate whitening before another finite update

12 October2026. **SELECTED / PROPOSED / UNIMPLEMENTED / UNBOUND.**
The full finite-gradient/update/actual-scan audit passed. One selected step
improves FIT loss only.745%,DEV operator RMS79.27%->78.93%;all local gates fail.
Larger normalized parameter steps rapidly increase finite loss. This suggests
conditioning should be tested before long repeated Euclidean-gradient epochs;
it proves neither poor conditioning is the sole cause nor a384-channel floor.

## Exact coordinate change, same representation and objective

Use FIT-only recurrent covariance Sigma_h=GY_rec_h. Symmetric PSD check and
invertible regularization must be frozen before execution. Initial proposal:
floor eigenvalues at1e-6*maximum eigenvalue (relative PSD tolerance1e-10),yielding
condition<=1e6. S_h=sqrt(Sigma_regularized_h),symmetric and invertible.

For row-vector source recurrence R_h=y_h-D_h*x_h:

```
Rbar_h = R_h S_h^{-1}
Wbar_h = S_h W_h
Vbar_h = S_h^{-1} V_h
R_h W_h V_h^T = (Rbar_h Wbar_h Vbar_h^T) S_h
Wbar_h^T Vbar_h = W_h^T V_h = I
```

Decode to PHYSICAL channels before adding full source skip,gating and the joint
source-epsilon RMS. Do not normalize whitened features as if they were source
channels or move S through SiLU/convolution. Transformation adds no information,
rank,expert or memory coordinate. At an unchanged point,physical projector and
every finite source output must match the previous point within frozen precision
gates. Per-head common scalar decay commutes with fixed linear transformations;
arbitrary different-head mixing is still not permitted.

The physical finite output loss and fixed teacher/case weights remain unchanged.
Gradients transform by chain rule:
G_Vbar=S^T G_Vphys,G_Wbar=S^{-T} G_Wphys. Changing optimization coordinates
changes the metric on parameter updates,not the admitted model class. Qualify
full transformed backward/autograd,biorthogonal tangent/retraction and physical
parameter replay independently before a new candidate.

## Finite next implementation and decisions

Site12 adaptive384,current selected point/ranks frozen;FIT-only S,ridges and
direction construction. No DEV fitting,rank choice or coordinate selection.
First quantify finite-loss curvature along directions with COMPARABLE first-
order predicted output change;parameter Frobenius units in different gauges are
not a fair step-size comparison. A FIT-only RMS-Jacobian direction normalization
can provide this comparison;exact policy/tolerances/caps require an implemented
protocol before running. Retain original coarse FD nonacceptance and use the
separately qualified fine controls explicitly,not silently changed thresholds.

At most one bounded FIT-selected physical update after the coordinate/control
qualification;then NEW actual compressed scans,full CPU sequential reference,
BF16 outputs and unchanged local held-out gates. If conditioning repair does
not materially improve finite loss,inspect larger/nonlinear/conditional state
geometry rather than claiming another long plain-gradient epoch is justified.

## Connection to the full goal

This is a local source-injected transfer component. It still retains source
generators/gates/conv/state256/fullskip. Site23 accepted local representation
remains a candidate for explicit source-derived native layout/cost qualification
and a separate state96 composition test. Count all retained work;no generic
full-cost donor runtime may stand in for the efficient target. Whole causal
own-history/chatbot preservation,n/DRAM/IDs+masses,multiple families/scales and
same useful artifact>=50 remain open. No T4/delegation;required hours-long work
follows user announce/bind-launch/pause instruction. This plan launches no run.
