# METH482: all-bank root affine margin feasibility and physical error bounds

5 October2026. Prospective complete science/input/runtime/decision/cost contract,
BEFORE any482numerical import/control. Previous goal turn PROGRESS:481complete
source geometry independently admitted; specific second-moment/PSD path closed.
Original full goal ACTIVE/INCOMPLETE; this is a necessary root-geometry inquiry,
not a substitute for pretrained transfer/useful n/whole quality and SAME>=50.

## Question and immutable inputs

Does the current original balanced tree admit cheap affine root decisions with
useful verified margins? A group union of winner cells need not be linearly
separable. Do not assume two affine heads per visited node can preserve it.
Retain every12bank/238872UID/387036occurrence/192book/sourceID/tie/rare/accepted
field. Only159414development UIDs construct heads;79458consumedval evaluated
after construction,overlap0. Roots are a declared prerequisite for this fixed
tree,not a choice made after selecting successful nodes.

Original complete480binding2c33137557c5a958680633187b536dde37df75e12ce3882e0cff04330259579e,
all6224source files/full7.542GBpayload/manifests/479R2target wires/native roles/
Python/NumPy/psutil/threadpoolctl/DLLs/compilerassets/OS files freshly admitted
before NumPy import. Retain480and481 main/audit fault chains and outputs; no
completed source/main/native/audit replay.481RAW73f2da6d...,
481RET a8573719fa6fc34fa91be6844dfa94068be47867d62a8c44d3a87ccbd2821111,
actual481R2Windowsfd7066ade489b641010b438d91c282d92c10a3cb198af4b4e3036de87f2090e7.
Require all5/6old gates and actual PID/create_time/timezone-aware terminal times.

New SciPy1.18.1 runtime supplementd506a86753ea0f1e1a16253461e07811602272dfdffafa4b45e13567381f17ad:
1536installed non-pyc wheel source/native/data/dist-info files,108056238B,including
scipy.libs/DLL and installed optimize/highspy wrappers. Hash before import.
Generated pyc excluded; no install/update needed. Actual imported modules/pools
bound and1thread. Interface reference:[official SciPy highs-ds documentation]
(https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs-ds.html),
accessed5October2026/manual1.18.0; actual1.18.1 source bytes are authority.

New controller/math/Windows/this protocol/runtime binding physically HEAD
before first execution. All3unrelated physical hashes exact,staging empty/
unstaged-M subset only. CPU0/BLAS1/600s watchdog/4GiB/new64MiB/admission180s.
Early AND late process gates,exact2argv pythonw publisherdaemon exemption only;
separate PowerShell preflight inspected before launch. No unrelated kill.
Live SAME handle/small PowerShell/commentary only,no edits/hashes/science/Git.
Firstfault retained before numbered repair. No T4/GPU/native compile or program.

## Fixed optimization; no parameter sweep

At each root,rederive L/Rmaximum and earliest original winner from128scores,
BYTE compare all955488maximum/winner fields to qualified479R2targets. Source
right label=(MR>ML) or equal-score RWID<LWID; independently check membership
of native sourcewinner in right child. Include every tie; y=+1right,-1left.

With augmented h_i=y_i*[x_i,1],d=769,solve ONE development LP per bank:

```
maximize t
h_i^T(u-v) >= t  for EVERY development UID
sum_j(u_j+v_j) <= 1
u>=0,v>=0,t>=0.
```

1539variables,n_dev+1inequalities,at most~182.8MB dense matrix before solver
copies. Sequential banks;4GiB measured peak cap. scipy.optimize.linprog
methodhighs-ds,presolveTrue,devex,primal/dual tolerance1e-9,threads1 forwarded
through the installed wrapper,solver time_limit25s per root. Save that documented
OptimizeWarning explicitly. No backend retry,seed/step/K/rank/threshold search.
12source LP calls plus3small analytic control LPs. Each call conversion/verification
time also charged;25s is solver limit,not a promise that whole call is25s.
Status1(time/iteration limit) is INCONCLUSIVE,not evidence of impossibility.
Status2/3contradict this trivially feasible bounded formulation;status4numerical
solver issue stops as apparatus fault. Solver success alone is not a proof.

Save ALLraw primal/inequality-dual fields/iterations/status/message/warnings,
complete developmentUID/label list,normalized thetaF64 and thetaF32. Positive
rescaling by max(1,2*outward_norm1_upper) is fixed algebra,not another fit;
verify actual saved theta has outward norm1upper<=1. Export copies include bias.
Unavailable time-limit candidates have explicit availability0,empty raw fields;
zero placeholder vectors never count as successful heads.

## Outward verification and dual margin upper bound

For ANY nonnegative dyadic weights lambda,not necessarily optimal multipliers,

```
max_{norm1(theta)<=1} min_i h_i^T theta
 <= norm_inf(sum_i lambda_i*h_i) / sum_i lambda_i.
```

Proof:min<=weighted average;Holder bounds theta dot weighted h by its infinity
norm. Choose max(0,-saved marginal),divide by its positive maximum if any;
these savedF64 dyadics define a NEW exact nonnegative witness. Do not assume
numerically normalized weights sum exactly1. Keep every development coordinate.
Use nextafter outward products and sequential sums for all769dual residual
coordinates and weight sum. Upper=outward division of largest interval absolute
residual by positive weight-sum lower;otherwise unavailable/null. No small
residual treated as exact zero;no unrestricted Farkas/infeasibility claim.

For EVERY unique input,compute outward exact-real dot intervals of saved thetaF64
and exported thetaF32. Each F64 product bounded by nextafter,then each addition
outward. Subnormal nextafter/add/multiply controls before data and after every
LP detect FTZ/DAZ changes; finite overflow/NaN is apparatus fault. IEEE elementary
arithmetic assumptions explicit; these are floating interval evaluations of
fixed dyadic inputs,not a statement of native execution or model quality.

Physical contract PROPOSED,not integrated/executed C:finiteF32 weights/inputs,
augmented bias1,products exactF64,any sum tree with768F64 additions,one correctly
roundedF32 cast,FTZ/DAZ off. With gamma768=768u64/(1-768u64),u64=2^-53,
E64<=gamma768*outward_sum_abs_products. F32 cast adds
u32*(outward_max_abs_dot+E64)+2^-150,u32=2^-24. Inflate every operation outward.
Physical interval=exported exact dot interval plus/minus total arithmetic E.
Strict signed LOWER>0 is sufficient;ties/straddling intervals are NOT proved.

Separately compute a UNIFORM sufficient export/arithmetic envelope for ANY
norm1(theta)<=1 on development coordinate boundB=max(1,max_abs_x_dev):

```
a=u32+d*2^-150
E_uniform=B*(a+(gamma768+u32*(1+gamma768))*(1+a))+2^-150.
```

Decimal100 and outwardF64 conversion;d769. If dual_margin_upper<=E_uniform,
no unit-L1head can attain the STRICT margin required by THIS uniform sufficient
envelope. This does NOT rule out another head whose actual error is smaller,
all affine classifiers,other partitions/nonlinear geometry or final quality.
Saved candidate's narrower physical intervals are evaluated independently.
Verify its development ideal lower margin never exceeds the dual upper bound.

## Controls, complete outputs and decision boundaries

Before source inquiry:analytic LP optima1(separable1D),0(nonseparable−1/0/+1
alternating labels),1(constantpositive labels/bias). Independently check dual
upper/positive physical sign where applicable;Decimal100cancellation witness
and subnormal bit controls. Controls/schema/resources fail as apparatus.

88B/UID wire,24Bheader M482CER1/88/12/238872,canonical UID/EOF:
6U32(uid,bank,sourceID,sourceRight,sourceTie,candidateAvailable),8F64(ideal
lo/hi,exported exactlo/hi,physical lo/hi,sum_abs_products_upper,arithmeticEupper).
21,020,760B total.12complete witness NPZfiles,banks/controls/reports/RAW.
ALL39unique/72occurrence/4608book/9216source-ID views,nulls/empty slots/accepted/
rejected counts and full denominators. No rare/example/failed-bank removal.

5apparatus gates:immutable source/runtime;controls/wires/roles;ALL12LPsaved
witnesses/source labels/UIDintervals;complete reports;resources/preservation/no
replay/sweep. Science decisions separate:all roots consumed-domain physical
sign sufficient;any uniform sufficient envelope unattainable;all solvers0.
Actual terminal then ONE frozen main Windows query and a separately frozen
COMPLETE independent audit before interpretation. Proposed audit600s/4GiB/
32MiB/admission180s;all saved witnesses/UID fields/reports rederived,sourceID/
labels/coefficients BYTE,interval fields BYTE or justified outward enclosure
checked separately,integers/null exact,reportF64 abs1e-10+rel1e-9. No imported
math helper/sampling/LP or native replay. Freeze full audit before invocation.

If candidate root signs pass,complete tree/mass/physical C/whole artifact remain
unproved. If uniform margin fails,close ONLY that sufficient affine-root budget;
choose geometry/feature or joint function distillation with complete cost/loss
argument. Full final requirements unchanged:pretrained useful capacity,compact
core/functions,own-state fresh donor prediction/generation/tasks AND SAME>=50,
useful distinctn/RAM/CPU LUT/physicalDRAM/multiple actualfamilies/scales.
