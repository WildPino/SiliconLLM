# METH-258: frozen read-only existing-bank diversity diagnosis

Bind failed METH-257 record SHA
`fc0fb83cb370a165b89fc3616224a47648077aefa42c13aa766e531329507fcb`,
require first-layer effective-bank stop/no completed rows and361 source
segments. Pin original parent training/checkpoint, child checkpoint and
METH-125 original BF16 source states through existing module constants.
No donor model/core load, fitting, new target/data acquisition or perturbation.

Reconstruct the original24 centered banks with the same METH-128 GPU
arithmetic, using placeholder dense Identity modules which are never
executed. Require10 exact sibling A copies before extracting parent A,
and original FP32 mean B versus learned parent B max error<=1e-7 in each
layer. Read raw child B values from the original checkpoint, compare per
parent (10 children) at three stages: raw child FP32 B, centered actual
FP32 A/B, and actual BF16-execution A/B. Within each group, count raw B
hashes and composite A/B hashes. Exact zero B canonicalizes to one zero
correction independently of A. Require nBF16<=nFP32<=nraw, no added noise.
Record all signatures, group zero counts and exact parent-mean error.

Report cross-parent composite function-parameter counts per layer with the
same exact-zero rule, nominal1280 route labels separately, numbers of
groups already duplicated before centering, merging at FP32 centering
and merging at BF16 casting. Do not infer inactive training/visitation
solely from zero B or claim parameter signatures prove semantic capacity.

For every sibling group use the same first64 original stored BF16 states
of that layer, no routes/labels/targets. Compute BF16 `hidden=SiLU(x A')`
and every BF16 `out=hidden B'` with fixed ordinary linear arithmetic. Count
bitwise-distinct outputs and minimum FP64 Frobenius distance between
different effective parameter codes divided by the input-state norm
(floor1e-12). No probe threshold or selection: report collisions/zeros
as measured, and never call this a new source-quality/count gain screen.
This is an explicit per-expert diagnostic, not full gated-kernel parity.

Local RTX3060/six threads, deterministic/highest/TF32 off,20min after
imports,20GiB RSS/10.5GiB GPU. Preserve per-layer partial/failure rows.
Freeze code/protocol before executing once. Pass means read-only audit
completed and permits a separately frozen unique-storage/alias-map
export. It does not override METH-257's all10-distinct stop, convert
aliases into private capacity, or qualify large-n CPU LUT/DRAM, quality,
native same-artifact rate,10B/100B or cross-family transfer.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth258_effective_bank_diversity.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth258_effective_bank_diversity_result.json
```
