# After480: decision margins, conserved mass and the whole transfer budget

5 October2026. Derived mathematics and proposed next work, not additional
numerical evidence.480 result is independently admitted; original goal ACTIVE.
[Complete result](METH_480_LEARNED_ROUTER_RESULT_20261005.md).

## 1. What actually advanced

We can reproducibly extract native support/mass supervision from a pretrained
router, train complete conditional geometry, export physical coefficients,
execute it in C and independently admit every saved update and new dot/p field.
This is a real implemented transfer procedure. The specific480 geometry fails
its source-fidelity gates, so it is not a usable transferred whole model.

Measurements:201993/238872IDs changed; ALL12banks fail; first divergence
103014at depth0,56482at1,25621at2,11191at3,5685at4,0at5/6. Similar development
and validation failures. Every root loss decreases, but mass is still wrong
even with a correct selectedID. Apparatus/rounding/order are admitted; this
does not identify a global optimum or uniquely assign failure to the optimizer,
loss or representation. Do not rerun32steps, tune K or time this failed router.

## 2. Winner choice is a margin problem

For a group's child sets L/R, let M_L=max(s_e:e in L), M_R=max(s_e:e in R).
Write D=M_R-M_L and support errors ε_L=Mhat_L-M_L, ε_R=Mhat_R-M_R. Then

```
Dhat = D + ε_R - ε_L.
```

For D nonzero, |ε_R-ε_L|<|D| is sufficient to preserve its sign. Equal-score
cases need the original ID tie rule explicitly. A shared error in both supports
cancels; a tiny differential error can reverse a tiny margin. Independent
squared support errors do not directly optimize this decision condition.
480 first-divergence counts motivate looking at the margin before another fit;
they do not prove a direct margin learner will succeed.

In real arithmetic a convex prototype v=sum(a_e*w_e), a>=0/sum(a)=1, gives
v·x<=M_group(x). Two such forms approximate a support function with two exposed
forms. A donor group may need many exposed vertices on its actual input domain.
Number of stored IDs is not the number of vertices needed to preserve every
decision. Native F32 arithmetic is separately qualified; this inequality is a
statement about the ideal affine functions, not a byte-level F32 certificate.

Next feasibility question: can a much cheaper DIRECT child-decision function
retain the relevant signs and original ties? First examine source margins and
conditional confusion rather than repeating the failed independent support-MSE
fit. An affine feasibility problem, if used, must distinguish positive-margin
separability from exact classification allowing zero/tie cases. A Farkas or
convex-hull witness can reject its specified linear constraints; it cannot
reject all piecewise/nonlinear routes or a differently partitioned tree.

## 3. Mass fidelity is a log-error problem

Ideal reference A=log(sum_e exp(s_e)); selected p=exp(s_chosen-A).
If the same original ID/score is selected and Ahat=A+δ, then

```
phat/p = exp(-δ).
relative error <= 0.01  iff  -log(1.01) <= δ <= -log(0.99).
```

This is about0.01log-units, while480mean |δ_native| is0.897728/max17.469203.
The native target is s_chosenF32+log(nativeZ_F64), with separately rounded
exp terms/pF32. Therefore the displayed identity explains the ideal scale of
the required error; actual acceptance still uses original native pF32, not an
unrounded replacement. The same-ID p failures isolate mass as a second problem.

480 enforces sum_j b_j=n and c_j in the n-simplex for
Ahat=log(sum_j b_j*exp(sum_e c_je*s_e)). Define α_e=sum_j b_j*c_je. Global
normalization implies only sum_e α_e=n, not α_e=1 for each original expert.
By Jensen,

```
sum_j b_j*exp(sum_e c_je*s_e) <= sum_e α_e*exp(s_e).
```

Even if every α_e were1, this construction would be a LOWER bound on the real
partition, typically strict. Normalizing total prototype mass alone does not
make the partition source-faithful or ensure phat<=1. Clamping phat would hide
the error and would not recover the source weight; it is not a solution.

For affine source scores, Hessian(A) is their softmax-weighted covariance.
With n forms its rank is at most n-1 (and at most input dimension); actual rank
on the source domain is unmeasured here. A16affine-form log-partition has rank
at most15. This is an expressiveness restriction, not a proof that donor
curvature requires rank127 or that the observed failure is irreducible.

A different mass representation can be derived by centering:

```
μ = mean_e(s_e), r_e = s_e-μ
A = μ + log(n) + log(mean_e exp(r_e))
variance = x^T C_w x, C_w=mean_e[(w_e-wbar)(w_e-wbar)^T].
```

The common affine shift needs one stored mean row. If residual scores are
small enough, a quadratic representation targets partition curvature directly.
For |r_e|<=B and exact mean(r)=0, Taylor's theorem and Jensen give

```
|mean(exp r) - (1+mean(r^2)/2)| <= exp(B)*mean(|r|^3)/6
|log(mean(exp r)) - log(1+mean(r^2)/2)| <= same bound.
```

Both log arguments are at least1. A rank-ρ PSD covariance approximation adds
at most ||C_w-Cρ||_op*||x||^2/2 to this log approximation error. It costs one
mean dot plusρprojection dots, squares and a log, with all coefficients and
rounding paid. The remainder may be far too large; no Gaussian assumption or
automatic small-rank claim is permitted. Source native rounding needs its own
error term. Earlier394score-rank/472FFN-input-rank failures do not evaluate this
different quadratic PARTITION target, and their original decisions remain.

## 4. The whole model still determines success

For a selected donor function e, y=p*F_e(x). With candidate identity ehat,

```
Δy = (phat-p)*F_e(x) + phat*(F_ehat(x)-F_e(x)).
```

Route labels, mass and functional differences jointly matter. Downstream logit
response is locally J*Δy plus higher-order terms; autoregressive states then
change.477-R1's finite head changes despite small FFN RMS are a direct warning
against replacing fresh prediction/generation/tasks with local vector error.
An ID change is not automatically an equal-sized quality loss, and equivalence
of different functions must be established, not asserted to excuse routing errors.

Source128-to-cheap-router alone does not create useful new experts.123/183
already provide positive small-Qwen learned128->1280 evidence;369causal
Switch identity and373boundedFULL64->256cost remain meaningful. Preserve them,
but do not combine their quality/rate with another artifact. Source-matching
and newly conditional specialized functions are separate transfer operations.

RAM and throughput constraints, with actual buffer/context terms, are

```
RAM = B_core + n*B_expert + B_route(n) + B_state + B_workspace
T_token = T_core + T_route(n) + T_active_experts + T_real_memory + T_other
50 accepted batch1 IDs/s requires T_token <= 20ms on the declared workload.
```

A balanced binary route stores O(n)gate parameters but reads O(log n)gates per
query; physical memory traffic/access latency still needs measurement. Larger
n can fit RAM while route fidelity, function exposure or DRAM stalls fail.
An ID carries at most log2(n)bits; this alone gives no cheap-computation theorem.
If per-depth conditional decision errors are q_l, union bound gives overall
error<=sum_l q_l. Holding a small total error while increasing depth requires
correspondingly small per-depth errors, with ties/rare functions retained.

458matched assays put router around1.45%/2.28% of those whole workloads. Amdahl:
if a component's matched fraction is f, removing it gives speedup<=1/(1-f).
Router removal in THOSE profiles gives only about1.015/1.023; it matters for
large-n growth but cannot supply the missing compact-core transfer by itself.
Experts around24.86%/27.09%, dense work around40..44%, head around6.33% remain
relevant. These fractions are not a forecast obtained by combining separately
configured absolute rates. C LUT/quantization requires SAME fresh whole-artifact
quality AND accepted rate, plus actual DRAM, not logical byte savings alone.

## 5. Next bounded decision, before another trained candidate

ONE source-only algebraic inquiry should resolve whether the next change is
in the decision target or the partition representation. Reuse admitted479
original128scores/targets/roles/weights and480physical route fields; no native
capture/main/audit replay, no new model, no GPU/T4. Freeze complete code/runtime/
input hashes/coverage/decision/cost before numerics. Prospective ceiling:
600s/4GiB/new64MiB/BLAS1, all12banks/allUIDs/all roles, no sweep or sampling.
If this cannot be priced within the ceiling, reprice BEFORE invocation.

Required output:

- Group decision margin and first-divergence analysis linked to original
  function IDs, with ties and ALL rare/exposure denominators explicit; establish
  which differential support errors reverse decisions. No scalar MSE proxy.
- Centered partition residual/variance/Taylor remainder and native-rounding
  separation, including full per-bank/role maxima and thresholds. Decide if a
  certified moment representation is viable before any new fitted mass head;
  raw variance/rank alone is not success.
- A conservative physical form/metadata/work budget for any proposed new
  geometry, including branches actually visited and shared-core/DRAM limits.

This is a proposed inquiry, NOT yet frozen/executed and no result claimed.
If moment remainder or decision margins cannot justify a cheap replacement,
do not launch another support/prototype fit. Move to direct branch/conditional
function distillation with an explicit capacity and whole-model loss argument,
or another geometry justified by the resulting constraints. No automatic LP,
SVD/training/seed/K sweep is authorized by this outline.

Any locally passing replacement then needs physical C admission/cost, a full
transformed artifact, fresh own-state donor-relative prediction/generation/task
quality AND SAME>=50, useful distinct-n scaling with RAM/CPU LUT/mass/physical
DRAM, and another actual family/scale before broad generality. Goal unchanged.
