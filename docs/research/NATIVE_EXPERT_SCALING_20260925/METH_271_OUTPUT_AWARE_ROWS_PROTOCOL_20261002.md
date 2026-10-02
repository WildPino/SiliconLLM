# METH-271: fixed128 output-aware private-row selection

## Question, changed variable and decision

270's coefficient-ranked128 rows miss the predeclared15% mean-state gate.
269's source-feature oracle indicates recoverable input-function error, but
coefficient discrepancy does not price activation/readout sensitivity.
Test a different selector,not more rows or a relaxed gate. Keep private128,
all original32 IDs/source rows,FP32 internal arithmetic,LUT513,down escapes,
rank32/biases and all learned banks/routes/alias maps exactly unchanged.

Use125's256 previously consumed BF16 states/layer: even token indices128
for selection,odd indices128 for reserve control. This is a newly fixed
partition of consumed evidence,not fresh independent model-quality data.
No weight fitting,selector-count sweep,source filtering or reserve retuning.
Bind270 raw `b14fdd1b7c8867ff01ee36600c8d8903ab74f1abeb60cd4051dfe9b0e93308bd`,
and its bound269/252/253/source/archive/helpers/vectors. Freeze apparatus
before any271 selection/observation.

## Exact selector

On even states form the actual source BF16 MLP target y and current32-row
FP32 pre-cast candidate z. Let E=y.float()-z and energy=||y.float()||².
For every feature j,delta_j is exact-source BF16 gate/up weights evaluated
with the unchanged candidate FP32/LUT arithmetic minus current candidate
feature j. Thus delta is the actual proposed row replacement,not an oracle
BF16 feature that the native operator cannot compute.

Decode the unchanged effective linear readout W: rowQ8+BF16 escapes plus
FP32 product of the stored BF16 rank32 left/right. Biases cancel. Rank j by
the FP64 mean individual normalized-error reduction

`mean_even((2*delta_j*(E @ W[:,j])-delta_j²*||W[:,j]||²)/energy)`.

This ignores cross terms among jointly selected rows and final BF16 output
rounding; direct full candidate checks must validate it. Keep old32 slots,
exclude their IDs and select exactly96 additional highest positive scores,
stable ties by lowest unit ID. Fewer96 positive scores stops the selector
before complete export. Sort physical IDs. No greedy second pass/refit,
lookahead,alternative ranks or choices after reserve observation.

## Frozen gates and budget

Require exact all289 unchanged segments/all361 readback/source rows/old32
subset and all6144 baseline269 controls; all outputs finite. All-state mean
relative squared error <=.85*.0001972897928984215; all-state maximum layer
energy error <=.00030542892636731267. In addition, odd-reserve mean error
<=.85 of its paired original32 baseline and odd-reserve maximum layer
energy error <= its paired original32 maximum. Fit reductions are descriptive.
Do not substitute pooled energy for mean-state error or rescue a failure.

Only if all source gates pass compile the unchanged253 split kernel with
PRIVATE_UNITS128 (experiment labels/output header only further changes).
All384 native vectors must meet original median<=1e-4/max<=5e-4 relativeL2;
original three256-token24-layer/six-thread sweeps median<=10ms. Synchronize
GPU first; no overlapping model job. Same337,596,452-byte fixture,extra
8,262,144bytes versus252. Native failure closes this fixed selector/recipe.

Local3060/sixthreads,10min selection/scoring/export,20GiB RSS/10.5GiB GPU,
3GiB output; native compile/check/timing separately5min,no T4/download.
Record all per-layer scores/IDs/fit/reserve/fidelity/hash/resource evidence;
preserve failures. Pass only licenses separately frozen full archive and
consumed regression followed by newly excluded-source generation/task/
anonymous-semantic controls.259 remains closed by267; no component result
demonstrates useful new n,RAM route/LUT/DRAM quality,>=50 or family transfer.

## Apparatus freeze and command

`meth271_output_aware_rows.py` verifies the decoded readout against the fixed
FP32 operator (maximum per-state relativeL2<=1e-5 apparatus guard) before
using its scores. It binds270 code/raw record and all inherited source
checks, stores all4864 scores/layer and even-fit/odd-reserve metrics, and
stops before native work on any fidelity/reserve failure. The C runner
verifies that its source is exactly253 with count/labels/header changed.
Code and this protocol are committed before execution; no271 observations
exist at this freeze. The decoded matrix is selector-only and is not used
as a replacement candidate operator.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth271_output_aware_rows.py --binary results/native_expert_scaling/meth271_output_aware_rows_fixture.bin --exe benchmarks/native_expert_scaling/meth271_output_aware_rows_cpu.exe --check results/native_expert_scaling/meth271_output_aware_rows.check.bin --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth271_output_aware_rows_result.json
```
