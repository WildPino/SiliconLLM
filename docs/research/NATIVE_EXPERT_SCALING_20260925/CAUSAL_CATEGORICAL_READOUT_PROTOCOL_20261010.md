# Shared categorical adapter to original C: frozen FIT protocol

10 October2026. One new convex FIT-only head branch and actual packed artifact.
[Decision and algebra](CAUSAL_CATEGORICAL_READOUT_NEXT_20261010.md),
[approximate native feature admission](CAUSAL_READOUT_COORDINATES_RESULT_20261010.md).
Goal incomplete. No implicit old optimizer, recurrent dose or head scale restart.

## Custody and exact method

24 FIT/4422 existing output-label positions and cached BF16 source logits.
Use qualified original C actual27 normalized feature files, paired phi/gain,
frozen newly paired A=aK65537x255, current old packed original core/final norm.
Check exact positions/IDs/domain/label counts through both source/native metadata.
No DEV file consumed for geometry, initialization, fit or head selection.
No source generation/core history/expert/router/native/T4 call in this stage.

Equal-case w_j=1/(24 m_case). One thin SVD of Z=sqrt(w) f, support
s^2>=1e-12 s_max^2. This computes covariance eigensystem directly without
squaring its condition in the numerical factorization. Save C=Z^T Z,U,s,VT,
features/weights/warm targets; W=VT_support/s_support, t=f W^T. Require discarded
positive trace fraction<=1e-8 and relative E_w[t t^T]-I<=1e-8. Theta0=sum w
(phi/a) t^T, norm<=8 +roundoff1e-10. No new head/multiple initializations.
The source warm-code RMS<=8 proves initializer Frobenius<=8 in exact arithmetic.

Optimize Theta255xrank in a Frobenius ball16; J=Theta W. New shared-parameter
domain, different from per-label code ball16. Teacher moments m=A_centered^T q
and c=sum q log q cached once using ALL4422 labels, no teacher model head call.
A centered over vocabulary for objective/gradient; uncentered A for real export.
FG returns sum w[c+logsumexp(A Theta t)-m^T Theta t] and exact analytic gradient
sum w[A^T p-m] t^T. F64,TF32 off, deterministic Torch/CUBLAS :4096:8, NumPy2.4.6,
Torch2.6.0+cu124,psutil7.2.2,seed0/no random starts,6 worker threads.

## Fixed projected accelerated descent

At update0 theta=Y=Theta0,t_accel=1,L=1. Evaluate full FG(Y); all candidates
project to Frobenius16, line search accepts F(candidate)<=F(Y)+<g,delta>
+.5 L||delta||^2+1e-10. Double rejected L,max30 backtracks; accepted L never
shrinks. FISTA extrapolation t_next=(1+sqrt(1+4t^2))/2,
Y_next=candidate+(t-1)/t_next*(candidate-old_theta). No alternate scale/restart.
Full FG includes every rejected trial. Best feasible observed point is retained,
including feasible rejected trials; count actual evaluations, not just32 updates.

For every tangent point (possibly extrapolated outside ball), lower=
max(0,F-<g,Y>-16||g||_F). Store actual best lower point/gradient/F and best feasible
upper point. Independent recomputation at both final points counts as two more
FG calls. Stop only32 accepted updates, global upper-lower<=1e-5, or mechanical/
resource cap. No quality-triggered early exit/audit omission. Checkpoint every8
accepted updates and final; retain theta,Y,FY,gY,best/lower points/gradients,L,
acceleration and exact counters. First fault records actual partial calls/stage/
artifacts/last durable checkpoint; no completed update replay.

## Metrics, export and uncertainty

Fixed FIT quality gates from prior codec: equal-case meanKL<=.01/disagreement<=1%,
every-case KL<=.05/disagreement<=5%,every-domain meanKL<=.05/disagreement<=5%.
All cases/masses retained regardless of outcome. Proxy pass only pending actual
native/DEV; this stage never admits a chatbot, accepted speed or useful n.

Offline fuse J=Theta W, H_real=A_uncentered J, H_native=F32(H_real). New packed
artifact copies old520MB weights and changes ONLY exact old head field bytes.
Verify identical header/table/all109 other fields via complete prefix/suffix byte
hashes plus parsed ABI; Vx256 F32 shape unchanged. No extra inference matrix,
carrier,inversion,donor read or runtime kernel. SSM/SWA/ternary/LUT stay original.
Save real/native heads/J and actual packed artifact. Save scalar loss/argmax for
ALL4422 labels; full-V scores are recomputed in complete audit rather than
retained twice (~4.6GB). These audit products are arithmetic checks, not history.

For forced prefixes only, feature bound E_j propagated conservatively:
epsilon_j=E_j max_v||H_real_v||_2+(||fhat_j||+E_j)
*[max_v||H_native_v-H_real_v||_2 +gamma39 max_v||H_native_v||_2].
A source-conditional KL can differ by at most2 epsilon_j under this F32 arithmetic
model (logsumexp and probability-weighted linear term each Lipschitz in infinity
norm). Save every bound; no interval or free-running history guarantee. Native
new-artifact observations remain authoritative. Native observer if amplification
reveals missing precision; do not silently treat recovered states as exact bits.

## Full audit and resources

Before calls freeze code/protocol/binding, exact input extents and same criteria.
GPU label chunks64, all computations F64. Capture1200s including pre/post hashes,
worker reserve120s,OS4GiB/GPUallocated4GiB/reserved5GiB/output1GiB/log8MiB.
Forecast4-10min based prior72-label cost, not a promised measurement. No T4.
Audit600s/OS4GiB/output2MiB/log8MiB/noGPU. No owned benchmark overlap; preserve
foreign publisher. Never restart a live handle after an observation timeout.

Independent CPU audit hashes every actual input/output, exact FIT feature/phi/
position/teacher provenance, full weighted SVD identity/orthogonality/support,
whitening and initializer, all teacher moments/entropy/argmax using independently
normalized full-V q, initial/best/lower full objectives/gradients/bounds,4 math.fsum
scalar gradient witnesses, every checkpoint Y objective/gradient/feasibility/
monotone envelope, every case/domain metric/branch, all head fusion/cast/packed
bytes and uncertainty bounds/counters/resources. No SVD or descent replay.
Numeric tolerances: factor relative1e-10,orthogonality1e-10,whitening1e-8,
initializer1e-8,moment/metric1e-9,gradient1e-8,bound/fusion1e-8,radius slack1e-10.

Next after full audit: qualify this exact new artifact with original packed C
consumer on FIT/DEV and own-history chatbot tasks, then same-artifact useful50.
A readout-only optimizer cannot by itself validate causal knowledge/functions,
structured CPU ID/mass,n scaling,physical DRAM or other donor family.
