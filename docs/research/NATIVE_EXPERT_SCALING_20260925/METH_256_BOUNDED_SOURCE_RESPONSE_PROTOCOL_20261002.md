# METH-256: frozen bounded nonlinear source-response amplitudes

METH-254/255 close forced full source-feature promotion through the adapted
readout. Keep the actual METH-247 E16 parent function and learn signed
amplitudes of original-source minus shared-Q8 nonlinear responses. This is
a continuous, nondeployable pilot, not an already qualified native blend.

## Bindings, samples and actual response

Bind METH-255 raw SHA
`243f748287a50b88c4aeaeecaa362f7f8e7f742271c692a495d5497ac17691a2`,
METH-253 native raw, METH-247 raw/actual snapshot, METH-240 raw/snapshot,
METH-238 fixture/control ledger, original capture/router and locally cached
pinned BF16 donor file plus layer12 gate/up/down hashes. No acquisition.
Replay original512 fit windows x128 exposures, labels/counts and16 original
FP32 parent SSEs/coefficient hashes exactly, including pooled Python sum.
Retain all actual parent fields, excluding unused old E160 factors.

For each parent compute actual shared FP32 LUT features phi, original BF16
gate/up weights decoded to FP32 through the same LUT, and their FP64
difference delta. With full decoded FP64 effective parent W, atom j is
`delta[:,j]*W[:,j]`. Reference is the unchanged actual FP32 parent output;
residual is reference converted FP64 minus target converted FP64.
Gram G=(delta' delta) elementwise times (W' W); cross c=sum(delta*(r W)).
No output/readout/routing refit, source reset, intercept or fitted derivative.

## Frozen selection, scaling and bounded solver

Eligible units have positive G diagonal and nonzero original BF16 versus
decoded-Q8 gate/up row discrepancy. Per-parent response energy is
`v_j=G_jj/n`, floored at max(1e-4 mean eligible energy,1e-12). This is
uncentered energy, not a claim of variance estimated from constant cells.
Use original tau=1024. Parent objective is
`alpha' G alpha+2 c' alpha+tau sum(v*alpha^2)`, alpha in[-1,1].

Choose32 atoms greedily. Starting with no atoms, at each iteration compute
the best bounded scalar addition with other coefficients fixed:
`a_j=clip(-current_cross_j/(G_jj+tau*v_j),-1,1)` and regularized decrease
`-(2*a_j*current_cross_j+a_j^2*(G_jj+tau*v_j))`. Exclude already selected
units, choose greatest positive finite decrease, lowest source ID on ties.
Sort selected IDs ascending, jointly solve all selected amplitudes with
previous amplitudes as warm start and newly selected amplitude zero.
Update cross to c+G[:,selected]*alpha, retaining all cross-unit terms.
Stop before32 is a valid incomplete-pair stop; no padding or altered rule.

Normalize beta=sqrt(v)*alpha. Symmetrized normal matrix is
`G/(sqrt(v) outer sqrt(v))+tau I`, linear term c/sqrt(v). Require finite,
positive v, Gram relative asymmetry<=1e-10 and condition<=1e8. Use cyclic
ascending-coordinate exact bounded minimization, at most256 sweeps.
Projected KKT infinity norm divided by max(|linear term|,1e-12)<=1e-7;
at lower bound retain negative gradient, at upper retain positive gradient.
Require objective not increasing (1e-10 max(abs(before),1) allowance),
finite solution and bounds with1e-12 allowance. Two analytic controls:
G=[[4,1],[1,3]],c=[-30,2],v=[2,1],tau3 yields[1,-.5]; inactive-coordinate
G=diag(2,0),c=[-1,0],v=[.4,.8],bounds[-1.8,-.8]..[.2,1.2],tau3 yields[.2,0].
Controls are executed only after this protocol/code freeze.

Recompute the actual stored32 BF16-row response (matrix shape differs from
full-source selection), require response difference relative to original
parent output<=1e-5. Recompute32 Gram/cross/energy with the same inherited
full-dictionary floor, then one final bounded parent solve warm-started by
the provisional solution. No renewed selection after actual-row recompute.

## Matched child hierarchy and guards

Keep the parent's same32 atoms, source rows, W and energy v in all10
children. Child objective fits adjustment around the complete fitted
parent, using child-support Gram/cross, tau1024 and inherited parent v.
Bounds are[-1-parent_alpha,1-parent_alpha]. No changed ridge or derivative
fallback for constant-input cells: zero-response coordinates retain the
parent. Store clipped total alpha in[-1,1]. Require anchored versus total
prediction conservation relative L2<=1e-10 and all explicit SSE quadratic
closures<=1e-8 relative original support SSE.

Require all16 dictionaries/all160 fits complete,176 distinct signatures
binding W/source-row hashes and actual amplitude-weighted W columns
(canonical zero, not ID differences alone). On the same first64 parent
fit states, compare parent plus10 children: every relative output Frobenius
distance against original parent norm>1e-7. Duplicate/inactive child
functions fail the pair; no perturbations, padding or weakened gate.

After all fits save the actual snapshot even if distinctness later fails:
retained parent tensors, BF16 gate/up[16,32,896], uint16 IDs[16,32], FP64
selected W[16,896,32], response energies[16,32], parent alpha[16,32] and
child alpha[160,32]. Size<120,000,000bytes, all tensors finite and readback
exact, old fields byte/value unchanged. The selected W is diagnostic
duplication, not a claimed compact deployable model. Original FP32 scores
remain exact; new comparisons consistently subtract in FP64. Fit E160
NMSE<=.01, parent no worse than original FP64 reference, child no worse
than fitted parent (1e-12 relative allowance). All prerequisites before
reading validation targets.

## Consumed development screen, decision and resources

Only if all fit gates pass, read the original128 consumed validation
windows once. Compare actual original parent, fitted E16, fitted E160,
within-parent rotated E160 and unchanged original source parent/child
priors. Replay every old per-window energy/FP32 reference/prior SSE
exactly. New comparisons use FP64 subtraction and accumulation throughout.
E160 NMSE<=.01 and SSE<=.9 each E16, rotated E160, original reference and
each source prior. Paired10,000-window bootstrap E16-E160 gain P05>0,
seed256257. This screen is already consumed development, not independent
document/full-model quality. Failure closes this fixed amplitude recipe;
no tau, bounds, rank or screening threshold retry on these observations.
Pass only licenses a separately frozen native encoded blend/learned bank.

Local RTX3060/six threads, deterministic/TF32 off,20min after imports,
20GiB RSS/10.5GiB GPU,>=2GiB free disk. Record partial/failure stages.
No T4. No native scalar blend/learned bank, large-n CPU LUT/DRAM,
independent prediction/generation/tasks, full accepted rate, new family
or10B/100B promotion. Those remain required on actual saved artifacts.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth256_bounded_source_response.py --checkpoint results/native_expert_scaling/meth256_layer12_bounded_source_response_functions.safetensors --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth256_bounded_source_response_result.json
```
