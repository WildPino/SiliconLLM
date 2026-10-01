# METH-247: frozen activity-weighted separate residual E16/E160 pair

## Question,bindings and matched arms

METH-244 exposes49.53% raw E160 fit gain erased by full readout requantization.
METH-246 qualifies the same source/base plus rank32 BF16 correction at9.637ms.
Test whether **learned encoded residual factors** conserve useful count gain.
Retain all original METH-240 raw solutions,source priors,tau1024,full input
keys,labels,feature variances,row-Q8 inputs and513-entry lookup. No new
training iterations/strength/rank/input maps/source derivatives/data.

Bind METH-240 result/snapshot from METH-244,encoding-audit result SHA256
`5164e4deb8ee6866b9d3b1f62f0d3468f6e7d814c517000876ccbf8305949849`,
and METH-246 native result
`bb84bdb1544025866e7411dd830ffea3bc5daf4b73447bf194c45c55ba9da202`.
All original capture/source/native-input/route controls remain required.

Both E16/E160 arms use one unchanged **stored METH-240 fitted E16 parent**
as base plus the same rank32 correction arithmetic. E16 factor targets
raw E16 minus its stored base;E160 targets raw child minus that same stored
parent. This matches active shape/precision across counts. Base is not
reencoded or replaced by an unrounded ancestor. Original priors and
uncorrected parent are diagnostic controls,not runtime fallbacks.

## Fixed factorization and prerequisites

Replay all176 original centered FP64 ridge systems using frozen priors/
variances. Require every regenerated mixed code/scale/index/escape/bias
byte-exact to METH-240;raw pooled fit scores match METH-244 within1e-12
relative FP64 summation tolerance. These replays extract frozen continuous
solutions,not changed label fits. Rank32 is fixed before observing results.

For each cell,delta=rawW-decoded_stored_parent. On only its fit phi,
center phi and form T=centered_phi*delta.transpose(),C=T.transpose()*T
in FP64. Select32 largest eigenvectors of symmetric C. Require finite
eigenvalues,minimum>=-1e-10*maximum,spectral energy agreement<=1e-8,
eigen residual and basis orthogonality<=1e-8. Canonical sign:largest-
absolute output entry,lowest row at ties,positive there. If T has exactly
zero energy,use top32 FP64 CUDA gesvdj left singular vectors of delta,
preserving source-derived off-support coefficients rather than injecting
noise;require spectral energy agreement<=1e-8. Report each fallback.

Let U be this basis,B=U.transpose()*delta. Balance each column/row using
a=sqrt(norm(Brow)),a=1 for zero row;encode left=BF16(U*a),right=BF16(B/a).
Report weighted retained energy,truncation and rounded coefficient/function
errors. Source values/slopes are inherited from raw solutions;there is
no new source-point reset or exact BF16-Jacobian claim.

Actual GPU/C contract is `((mixed_base_without_bias(phi)+linear(linear(phi,
right.float()),left.float()))+base_bias)+residual_bias`. Preserve this
addition order for trained nonzero biases. Set residual_bias FP32 from
the **raw solution's mean output minus the actual pre-bias combined fit
mean**,then require actual mean conservation relative L2<=1e-5. This
conserves frozen shrunk intercepts;it does not refit directly to target mean.

Save physical layer12 snapshot with shared controls/variance,16 unchanged
base readouts and16+160 right/left/bias arrays;read every tensor/key exact,
<200MB. Prior controls remain only in separately bound METH-240 snapshot.
Hash176 decoded FP64 composite coefficient matrices(base+left*right),
require all distinct,without bias/metadata-only differences. No forced
noise or duplicate-count repair. Require actual E16/E160 fit NMSE<=.01,
new E16 fit no worse than original stored E16,and E160 fit SSE<=90% E16.
Any prerequisite/fit failure stops **before validation**.

## Consumed validation and decision

Only after all factorization/readback/fit gates,score the same128 already
consumed windows once for factored E16/E160/rotated-child E160,original
parent prior/child prior and uncorrected stored-base E16. All per-window
energies and unchanged three controls must exactly replay METH-240.
Retain E160 SSE/energy<=.01,and SSE<=90% of each factored E16,rotated,
child prior,parent prior and stored-base E16. Paired10,000 window-bootstrap
gain P05>0,seed247248. Keep all rows/factors/effective hashes/cost.

Pass licenses separately frozen **actual stored bank C arithmetic/routing/
DRAM** before full composition/fresh quality. Failure closes this fixed
activity-weighted rank32 encoding;no rank/tau/additional-fit retry. Either
outcome is one-layer development evidence,not independent full prediction/
generation/tasks,large-RAM n,>=50 acceptedtok/s or approximately10B transfer.

## Budget and command

One local3060 run,six host threads,20min after imports,20GiB RSS,10.5GiB GPU;
2GiB free disk,one <200MB snapshot plus JSON. No T4/download/new data.
Save partial completed cells/failure stage. Freeze before execution.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth247_weighted_residual_pair.py --checkpoint results/native_expert_scaling/meth247_layer12_weighted_residual_functions.safetensors --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth247_weighted_residual_pair_result.json
```
