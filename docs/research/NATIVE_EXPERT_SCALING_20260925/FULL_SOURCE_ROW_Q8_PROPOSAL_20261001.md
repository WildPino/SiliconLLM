# Proposal: full source nonlinear feature coverage with row-Q8 operators

**Status: implemented; native cost rejected; conditional fitting stopped.**
[METH-232 result](METH_232_FULL_SOURCE_ROW_Q8_RESULT_20261001.md) passes
codec/numerical/source-function fidelity but11.023ms exceeds10ms. The
proposal below preserves its pre-execution reasoning and thresholds.
Next: METH-233 phase-cost diagnosis; no cost-gate reopening.
METH-233 later finds3.004ms in scalar SiLU/product. The changed
[METH-234 lookup operator](METH_234_ROW_Q8_SILU_LUT_RESULT_20261001.md)
passes8.522ms and scoped source/numerical fidelity with every projection
byte unchanged. METH-232 remains rejected; full-feature fitting is unexecuted.
METH-229 prices the combined H2048 BF16/FP32 operator at8.875ms/24 layers.
METH-230/231 make source-based nonlinear readout transfer executable;
source derivatives resolve one repeated-input copy, but consumed function
error remains9.38%,not<=1%. This does not prove missing features alone
cause failure. The next representation would retain **all4864 original
nonlinear directions**, with lower weight traffic and a simpler operator.

## Changed representation

Share complete original gate/up feature maps once per source layer;
conditional functions are output readouts over that full nonlinear basis.
No full896x896 affine term. Encode gate/up/down with symmetric **per-row
int8** codes and FP32 row scales. Scale=max_abs/127,zero rows scale1;
round-to-nearest-even codes in[-127,127]. Retain every source unit and
row identity. Output bias FP32. No activation quantization.

The actual row operator is `scale_row * dot(int8_codes_as_float,x)`,
FP32 SiLU/product,then the same row-scaled output operator plus bias.
Scale once **after** the row reduction, not inside every group64.
GPU/C oracle must use precisely this arithmetic; a pre-multiplied
decoded-weight matmul is not automatically the same FP32 operator.
This changes scale granularity/precision and row reduction from the
cost-rejected METH-220 grouped-Q8 float-input path. Its12.540ms result
does not price this operator. Row quantization might hurt source fidelity;
it must be measured, not presumed compensated by larger hidden width.

## Whole-path and resident shape arithmetic

All24 FFNs:313,786,368 int8 weight bytes plus1,019,904 FP32 scales and
86,016 output biases = **314,892,288 selected bytes**. Full-input16/10
route keys add2,236,416. Keeping the other METH-211 ideal addressed organs
would give

`559,794,176 - 323,592,192 - 11,280,384 + 314,892,288 + 2,236,416 = 542,050,304`.

This is shape arithmetic,not actual traffic,complete saved composition,
native attention/head fidelity or accepted rate. Source full-width
compute is larger than H2048; fewer bytes and one row scale do not prove
<=10ms. Measure before fitting. Original full-width BF16 would not fit
this ledger,so this is a format/functional-width change,not an unpriced
width retry of METH-231.

Sharing all gate/up maps gives210,124,800 resident bytes across24 layers.
One independent output function costs104,767,488bytes across24 layers.
Thus hypothetical function residency is1.886GB(E16),16.973GB(E160),
167.838GB(E1600),excluding other core/route storage. No such24-layer bank
exists. Active output readout count stays one; useful distinct functions,
n-scaled routing/LUT/DRAM and full quality remain required measurements.

## Next exact action: METH-232 prerequisite

Freeze/implement the codec and complete AVX2/FMA row-Q8 FFN operator,
using all original24 source layers and existing METH-125 states. Read
every code/scale back,compare C to the declared FP32 row-scaled oracle,
retain median<=1e-4/worst<=5e-4 relative L2 and six-thread24-layer
component median<=10ms,three fixed passes,no retiming. Also measure
pooled output SSE/energy against the original-source BF16-weight/FP32
FFN on those component states,require<=.01 before conditional fitting.
That fixture source-fidelity screen is not original BF16 whole-model
quality or independent prediction/task/generation evidence.

A failure stops this fixed format/operator before readout learning.
A pass licenses a separately frozen full-feature conditional method:
source-anchored output fitting,actual quantized coefficient encoding,
point/derivative knowledge when data repeat,and E16/E160/rotated controls.
How original-source derivatives project into the full nonlinear output
basis without an extra affine branch must be implemented and qualified;
do not assume the old affine-prior mechanism transfers unchanged.
The complete-function<=.01,relative10%/bootstrap and final LLM gates remain.
Declare local budgets before execution;no T4 or new external resources
are needed for the native prerequisite.

This is a dense-source variant proposal. GigaChat's sparse FFN/top4/shared
operator and larger mixer/input width require a separately priced source
function variant. No approximately10B or100B applicability is established.
