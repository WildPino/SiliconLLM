# METH-239: frozen fixed32 mixed-precision output-only source priors

## Uncertainty and changed variable

METH-238 qualifies actual32 BF16 output escapes plus row-Q8 remainder
at8.727ms/24 layers with.01390% scoped source error. Does its actual
encoding conserve projected original source values/derivatives? METH-235
continuous QR works but uniform row-Q8 slopes fail; METH-236 scale-only
bounds and METH-237 four encoded-feedback cycles also fail. Change output
precision/initial physical source readout,not threshold or cycle count.

Bind METH-238 result SHA256
`61b496f66455e764039877ddaf17cedaa3aab49c287f09178e085afd56dbe684`,
fixture/source controls; original pinned Qwen0.5B layer12 BF16 matrices,
METH-227 source/route result and actual16 parent/16x10 child keys; METH-222
capture. Reuse first512x128 fit inputs only. Captured output labels,
validation inputs/labels and fresh quality sources are not evaluated.
All parent fit labels/hash/counts and original source tensor hashes replay.

## Same continuous projection, changed actual encoder

Use actual full4864 row-Q8 gate/up and shared FP32 LUT. Analytic feature
Jacobian A and original smooth-SiLU source f(c),J(c) as METH-235. B0 now
decodes the actual METH-238 source output remainder/scales/IDs/BF16 escapes.
At each frozen parent center,thin FP64 QR A=Q*T and residual R=J-B0*A;
deltaB=solve(T^T,R^T)^T*Q^T. **One** continuous correction,no damping,
alternating cycles,rank truncation or tuned precision count.

Cast B0+deltaB to FP32;apply exactly METH-238's32 largest-absolute-per-row
stable selection/lower-ID tie rule,BF16 exceptions,uint16 IDs,zero selected
int8 codes,row-max/127 remainder scales. Actual output is scaled dense
dot plus separate indexed BF16 dot plus FP32 bias. Bias conserves original
source f(c) under this declared operator. Decode both physical branches for
stored analytic Jacobian;do not claim continuous reconstruction extends
exactly through mixed rounding.

Save sixteen actual mixed priors,shared gate/up/table,physical source-down
baseline,router keys;read back every tensor. Coefficient hashes concatenate
codes/scales/IDs/BF16 exception bits;bias/noise alone cannot establish
distinct functions. No extra affine term/hidden dimension is added.

## Frozen gates (METH-235 tolerances unchanged)

- Every source/native/route/control binding and snapshot tensor exact.
- All16 QR condition numbers<=1e8,finite coefficients/scales/bias;
  correction Frobenius/base norm<=.25 for every parent.
- Parents0,15 feature analytic derivatives versus the same fixed16 rows
  and five JVP directions in METH-235:every relative error<=1e-5.
- Every FP64 raw coefficient*A source-J relative Frobenius<=1e-5;
  parents0,15 complete unrounded FP32 output autograd relative
  Frobenius<=1e-5,peak-relative max<=1e-4.
- Every raw/stored source-center output relative L2<=1e-5.
- **Every stored mixed-codec Jacobian relative Frobenius error<=.01**.
- Pooled65536 fit-input original BF16-weight/FP32 source FFN SSE/energy<=.01.
  Report all16 cell errors/energies and unchanged mixed source-down baseline;
  no baseline-relative gain or separate cell-error threshold.
- All16 mixed coefficient representations distinct.

All pass licenses separately frozen actual full-feature E16/E160 fitting,
using full source derivatives when data repeat. A failure stops this fixed32
mixed prior recipe before fitting/validation. No exception-count retry.
The source C fixture's cost does not replace actual trained-bank C/routing/
DRAM measurement. Full held-out prediction/generation/tasks,>=50 accepted
tokens/s on same artifact,useful large-n RAM scaling and real second family/
approximately10B remain required;no sparse-Giga applicability assumed.

## Local budget and command

One local3060 run,six host threads,max20min after imports,20GiB RSS,
10.5GiB GPU,2GiB free disk,<100MB snapshot+small JSON,no T4/external source.
Condition failure records stage/progress and stops;other failed gates record
all completed audits. Freeze source/protocol first;result CRLF pinned,
weights ignored. Partial record after every parent.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth239_mixed_output_priors.py --checkpoint results/native_expert_scaling/meth239_layer12_mixed_output_priors.safetensors --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth239_mixed_output_prior_result.json
```
