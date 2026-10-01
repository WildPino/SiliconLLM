# METH-227: common-free full-input E16/E160 donor function pair

Freeze before clustering, compilation or validation. METH-226 source
derivatives and serialized affine CPU operator pass. The changed
mechanism uses complete source-local functions, not the failed METH-225
common or METH-222 PCA64 coefficient regressions.

## Frozen pair and inputs

Use actual layer12 of the original hash-bound Qwen0.5B-Instruct source,
saved METH-222 BF16 inputs/outputs:512 fit windows,128 consumed validation
windows,128tokens each. No new source inference, independent-quality
claim, selection on validation, target replacement or projected input.

Full896-input Euclidean hierarchical kmeans:16 parents,10 children per
parent,20 Lloyd steps each,seed227227 for parents and227228+parent for
children. Sample distinct fit-state indexes uniformly without replacement.
Direct squared differences reduced FP32 in blocks2048; argmin lowest-ID
tie. Selected means FP32; empty Lloyd cells retain their prior key, no
adaptive restart. Final keys are the final updated keys; assignments are
recomputed after20 steps. Stop if any parent has<10 states or any of160
children is unoccupied. No regression-rank/minimum65-state gate: functions
are source-derived, not coefficients learned from the cell's y samples.
Report every count; occupancy alone is not functional usefulness.

At each of the16+160 saved FP32 centers, compile the source f(c),J(c)
with the qualified METH-226 formula, using all original gate/up/down
weights decoded FP32. BF16-round each full896x896 Jacobian; output bias
FP32=`f(c)-J_stored@c`. All center reconstructions must have relative
L2<=1e-5 and all values finite. Check parent0 and leaf0 complete
Jacobian against autograd, same1e-5 Frobenius/1e-4 peak-element limits.
Require all176 weight hashes distinct, every serialized tensor readback
exact. Save FP32 route keys, BF16 function weights and FP32 biases.
Enable deterministic CUDA algorithms,CUBLAS4096:8,TF32 off.

## Conditional screen and stop

After fitting/compiling, use only reloaded parameters on all128 existing
validation windows. E16 selects one parent; E160 selects one child inside
that parent. Both execute one896x896 affine function, no common/additional
expert. Rotate chosen child+1 modulo10 with the same parent as a control.
Reconcile every target sequence energy to METH-222. Candidate gates:

- E160 pooled output SSE/teacher energy<=.01;
- E160 SSE<=.9 E16 SSE and<=.9 rotated-child SSE;
-10,000 paired-window bootstrap gain P05>0,seed227228;
- source derivative, finite/storage/center/occupancy/unique-function checks.

Also report fit/validation selected-center distances and input-energy
fractions, without a post-hoc threshold. These describe locality, not
unseen-source generality. A failure stops this fixed first-order tangent
geometry. No retry width, extra functions, coefficient refit, clustering
seed, whitening or gate relaxation after outcome. A local pass licenses
actual routed-bank C parity/cost and causal-composition development, not
accepted full LLM quality. All independent BPB/top1/generation/task and
same-artifact >=50tok/s, large-RAM CPU route/LUT and multi-family/10B
requirements remain open.

Budget25min after imports,20GiB RSS,10.5GiB GPU,<350MB new outputs,
local RTX3060/six threads,no T4 or new external data. Snapshot expected
approximately284MB for one layer. Across24 layers at E160, conditional
weights/bias alone would be6.179GB; no such complete bank is created.
Native METH-226 excludes routing/large-bank access; do not derive a full
rate or n-scaled traffic claim from that component.

```powershell
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth227_conditional_donor_tangents.py --checkpoint results/native_expert_scaling/meth227_layer12_tangent_functions.safetensors --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth227_conditional_donor_tangent_result.json
```
