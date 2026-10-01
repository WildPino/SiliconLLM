# METH-231: original-source slopes in every nonlinear child prior

Freeze before source compilation/child fitting/validation. METH-230
fits E160 SSE.007234 but fails its distinct-weight gate before validation.
Independent fit-data diagnosis finds one repeated-input child with no
feature variance. The new variable is **source derivative information**
in every child's weight prior, not a width/strength/precision retry.

## Immutable reusable parents and diagnosis

Hash-bind METH-230 result/snapshot, source and capture. Reproduce all
METH-227 fit counts/label hashes. Require the sole old duplicate group
E16 parent15/E160 child156;18 states,one unique input,zero variance.
Otherwise stop apparatus rather than silently treating another condition
as this evidence. Keep every source selection/input bank,16 trained
parents,16 source-prior controls and route key exactly unchanged.
Recompute parent fit features/variance floors exactly; require unchanged
E16 total fit metric. Do not rerun source selection or parent fitting.

## Changed transfer mechanism and fixed learning

At each160 frozen child centers c, compute original source full f(c),J(c).
Keep that parent's stored BF16 fitted nonlinear B and selected original
G/U. For h(x)=silu(Gx)*Ux, compile

`W_child_prior = J_source(c) - B_parent@J_h(c)`.

BF16-round affine,inherit B_parent BF16 and set FP32 bias to
`f_source(c) - (W_stored@c + B_parent@h(c))`.
This gives actual source slopes even when all local observations repeat;
it changes all child priors consistently and does not insert an arbitrary
nonzero weight perturbation. Before rounding the complete prior matches
the original derivative; nonlinear knowledge learned by the parent is
preserved. All160 finite stored-center relative L2<=1e-5; complete
autograd checks on children0/156 within1e-5 Frobenius/1e-4 peak-element.

Fit each child with METH-230's identical centered FP64 primal/dual solve,
parent feature variance/floor,1024 sample-equivalent coefficient and
intercept shrink, BF16 final weights/FP32 bias. All normal-equation and
mean-shift checks remain. Fold into one selected affine/B pair, sameH2048
active function as E16. Snapshot reuses unchanged METH-230 controls and
adds child-prior affine/bias controls (nonlinear B comes from parent16
control); verify every tensor and all176 fitted weight-pair hashes distinct.

## Evaluation and gate

No intermediate validation. After all fitting/storage, use the same
consumed128 windows,exact target energies,reloaded parameters,parent-
grouped source features and `(affine+nonlinear)+bias` FP32 forward.
Controls:unchanged E16,source-derivative child prior,fitted E160,rotated
child+1 inside same parent. E160 must have normalized SSE<=.01,10% SSE
gain over E16/rotated/its new child prior,10,000 paired-window bootstrap
gain P05>0,seed231232,and all source/storage/distinct-function controls.

Failure stops this fixed source-derivative child anchor. No strength,
feature, route, width or precision retry after outcome. Passing licenses
actual stored-bank C parity/routed/DRAM cost and causal composition, not
independent LLM prediction/generation/tasks or full >=50tok/s. Real10B/
multi-family and RAM-bounded useful n/CPU LUT remain unproven.

Budget30min after imports,20GiB RSS,10.5GiB GPU,<1.6GB additional outputs,
>=4GiB free disk. Local3060/six threads,deterministic CUDA/CUBLAS4096:8,
TF32 off;noT4/external data/new source model inference/native retiming.
Original FFN function/derivative queries use already bound source matrices.

```powershell
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth231_source_derivative_child_prior.py --checkpoint results/native_expert_scaling/meth231_layer12_source_derivative_children.safetensors --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth231_source_derivative_child_result.json
```
