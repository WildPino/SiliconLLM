# METH-333: specified arithmetic implementation correctness

NEW NUMERICAL ESTIMAND. Original329/330/332 original officialCPU1 probability
1e-6 qualification remains FAIL; this protocol does not repair or promote
those observations. It implements the prospective next-policy proposal after
331/332. Distinguish correctness of target arithmetic from preservation of
pretrained quality. ORIGINAL unmodified official donor remains PRIMARY for a
future NEW untouched quality cohort; no such cohort scored here.

## Prescribed recipe and independent reference

Original F32 source weights, activations/cache, residual additions, ReLU and
selected-probability multiplication unchanged. Every projection/router/head
matrix and attention QK product uses F64 operands/products/sums then F32 output.
Norm: F32 squared operands, F64 mean then rounded F32; F32 epsilon addition,
F32 sqrt then reciprocal, F32 hidden-times-scale then norm weight. Softmax:
F32 shifted logits/exp outputs, F64 denominator/division, F32 probabilities.
Attention weighted values: original F32 probabilities/value operands, F64
products/sum then F32 output. Head scaling original D**-0.5 rounded F32
(C computes F64 reciprocal square root then casts; avoids a different F32 sqrt
rounding). No FMA/fastmath, batch1/CPU1, source architecture/capacity unchanged.

New333 C copies332 loader/architecture/dot/norm, changes attention dots/softmax/
weighted values, router softmax denominator and head scale only. New opt-in
engine macro; default body exact. Independent TorchDispatchMode reference on
unchanged official4.57.6 architecture intercepts mm/bmm, softmax, plus instance
LayerNorm overrides. It uses vectorized Torch F64 operations, does not copy C
reductions. Record actual intercepted calls; require mm, bmm and softmax all
observed per case. No package source edits. NumPy primitive oracle fixed seed333
cancellation/D768 matrix, attention DK64, softmax N256 and norm: exact rounded
F32 matrix results, softmax/norm maxabs<=2e-7. Fail primitives/Tiny before full
source forward. C/Torch exp implementations need not be bit-identical.

## Frozen controls and decisions

Same328 exact Tiny weights/IDs/caps1 and64; all pooled/every-state<=1e-4,
probability maxabs<=1e-6, exact choices/capacity/greedy, same nine semantic faults
logit error>1e-4. Same two consumed source engineering prompts/forced token IDs
from329. Fresh whole source hashes, all6392 C mapped tensor hashes EXACT327,
unchanged source manifest. Complete source reference states/full logits/routes
saved NPZ with SHA; same full source numeric1e-4, probability1e-6, exact route/
capacity/greedy and zero-head fault>1e-4 versus MATCHED specified reference.
No source case reselection, no old gate relaxation. Restore norm instance
methods after each reference call; installed original official source unchanged.

Compile120s; MAIN20min AFTER imports; checked combined32GiB; existing isolated324
reference only, no GPU/download/training/concurrent timing. Preserve raw output,
binary/build/runtime/source/code hashes, full commands, failures/stage/partial
files. Immutable previous raws pin329/330/331/332. Freeze all code/protocol
before first observations. Native times are apparatus resource records only.

PASS: target arithmetic correctness eligible for separate compact integer/
scaling reference and actual cost preflight. FAIL: preserve and close unchanged
candidate. Neither decision establishes original donor quality, useful extra n,
LUT efficiency or final same-artifact>=50 accepted batch1 tokens/s. Actual heldout
reconstruction/prediction/generation/tasks and family/100B proof remain missing.
F64 accuracy apparatus is not the final fast compact representation.

Command:
```
results\native_expert_scaling\meth324_switch_reference\venv\Scripts\python.exe benchmarks/native_expert_scaling/meth333_switch_full_source.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth333_switch_full_source_result.json
```
