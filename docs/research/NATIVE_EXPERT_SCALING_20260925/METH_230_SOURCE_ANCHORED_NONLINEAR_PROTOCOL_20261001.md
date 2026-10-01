# METH-230: fixed source nonlinear features and parent-anchored readouts

Freeze before feature selection/readout fitting/validation. METH-229
actual combined H2048 native component passes. This new representation
adds exact selected source nonlinearity to full-input affine response;
METH-225 common and METH-227 tangent-only recipes remain stopped.

## Bound inputs, routes and features

Original hash-bound Qwen0.5B-Instruct layer12 weights, saved METH-222
512 fit/128 consumed-validation raw windows,128tokens each. No new
source inference, validation-derived fit, independent-quality claim or
row replacement. Reload METH-227 full-input FP32 parent/child keys;
recompute all fit labels/counts and require exact hashes/counts.

For each parent use only fit x and original source weights. For each4864
unit form its nonlinear Taylor remainder about that parent's center:
`h(x)-h(c)-J_h(c)@(x-c)`, h=silu(Gx)*Ux. Score=FP64 centered remainder
variance times original down-column squared norm. Select2048 highest
scores,stable descending order (lowest original ID breaks ties). No
validation choice. Scores ignore cross-unit cancellation and are not a
retained-output-energy bound. Selected G/U remain original BF16 source
rows; share that set across the parent and all ten children. Every selected
index/source row is bound/read back. No G/U training.

Compile source prior `W_affine=J_full(c)-D_selected@J_h_selected(c)`,
B=D_selected, BF16 coefficient weights; FP32 bias recomputed to preserve
source value at c. Require all16 center errors<=1e-5 and parent0 complete
gradient autograd relative Frobenius<=1e-5/max error relative peak<=1e-4
before readout fitting. Save all16 source-prior controls in the snapshot.

## Exact fixed source/parent anchoring

Parent features z=[x896,h_selected2048], FP32 nonlinear arithmetic, no
activation quantization, TF32 off. Full fit feature covariance is used
in regression; diagonal parent variance defines only the prior penalty.
Compute each parent feature variance FP64 and floor each coordinate at
max(1e-4 mean variance,1e-12). Use this same parent variance for its
readout and all ten children. Fixed prior strength tau=1024 sample
equivalents. No hyperparameter/feature-width retry or best selection.

For source-prior(parent) or stored effective trained-parent(child) W0,b0,
R=y-(z@W0.T+b0),all FP64. Center X=z-mean(z),Rc=R-mean(R). Solve

`(X.T@X + tau*diag(parent_variance)) @ deltaW.T = X.T@Rc`.

If states<2944 use the algebraically equivalent dual solve:
`deltaW.T=(X/variance).T @ solve((X/variance)@X.T+tau*I,Rc)`.
Otherwise primal. Check full normal-equation residual/cross norm<=1e-7,
denominator floor1e-12,all finite. Round W0+deltaW BF16. Define effective
delta from rounded new weights minus effective prior weights. FP32 bias
is `b0 + states/(states+tau)*mean(R) - effective_delta@mean(z)`.
Mean-shift rounding error/mean target norm<=1e-5. This shrinks both
coefficient and intercept deviations; children are anchored to already
rounded parent values. Report every solve/count, not independent sample
counts. Fold child deviations into one affine/B coefficient pair+bias;
do not read parent and child output matrices simultaneously at inference.

No regression-rank minimum is asserted for small cells: the positive prior
defines their solution. All160 occupied and all176 trained coefficient
pairs must differ in stored BF16 bits. Reused data/low support still limits
generalization. Save input banks, source-prior/E16/E160 affine/down/bias,
keys and source indexes; require every tensor readback exact.

## Fixed evaluation and stop

Record fit SSE from effective coefficients and parent-batched FP32 source
features; no intermediate validation. Only after all fitting/storage,
recompute consumed-validation routes and features grouped by selected
parent, score128 windows from reloaded tensors. Split arithmetic is
`(affine output + nonlinear output)+bias`, matching native kernel.
E16/E160 both execute one same-width function. Controls: initialized
source prior, trained E16, trained E160, and selected child rotated+1
modulo10 with the same parent/feature input. Every teacher sequence
energy reconciles exactly to METH-227. Require E160:

- normalized complete-function SSE<=.01;
- SSE<=90% trained E16,rotated child and initial source-prior SSE;
- paired10,000-window-bootstrap normalized gain P05>0,seed230231;
- bound source/routing/derivative/solve/storage/distinct-function checks.

Fail stops this fixed feature/readout/regularizer geometry. No width,
regularizer, source-unit choice, route/bias or precision retry. Pass
licenses actual stored-bank C parity/routed cost, then causal composition;
it is not independent LLM BPB/top1/generation/task quality or same-artifact
>=50tok/s. CPU large-RAM LUT/DRAM, real second-family/10B/100B remain open.

Budget30min after imports,20GiB RSS,10.5GiB GPU,<1.3GB new output (snapshot
approximately1.132GB including controls),at least2GiB disk free. Local3060
and six host threads; deterministic CUDA,CUBLAS4096:8,TF32 off. No T4,
external data, new source inference or native retiming during fit.

```powershell
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth230_source_anchored_nonlinear_pair.py --checkpoint results/native_expert_scaling/meth230_layer12_source_nonlinear_functions.safetensors --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth230_source_anchored_nonlinear_result.json
```
