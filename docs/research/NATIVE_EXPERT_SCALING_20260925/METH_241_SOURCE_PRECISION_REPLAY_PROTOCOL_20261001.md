# METH-241: frozen source precision replay and sensitivity diagnosis

## Question and evidence

METH-240's full-feature mixed functions conserve donor outputs but E160
loses22.39% versus fitted E16,including a worse fit metric. Its child priors
recompile original **FP32** source values/Jacobians after parental data
fitting,while actual targets were captured in a **BF16** donor forward.
Determine whether that source-precision mismatch is measurable and whether
a precision-matched source operator reproduces the captured target.
This is diagnosis,not a new fit/precision-matched prior or cause proof.

Bind METH-240 result SHA256
`3c02724b1194f305036aae6896a489e1d19b477d7d8fe1d3ddc45fa9579fc505`,
its actual1.591GB snapshot,original pinned layer12 BF16 matrices,METH-222
capture and METH-238 input/table controls. Only first512 fit windows.
No validation/new source,fitting,changed iterations/native timings.

## Fixed replay and decision

METH-222 hooks original MLP inputs/outputs of shape**(4,128,896)**.
Replay each512-token fit block in that exact leading shape using original
BF16 weights,input,gate/up matmuls,SiLU,product and down matmul;convert
only final output to FP32 for comparison. Also compute original
BF16-weight/**FP32** smooth FFN on the same shaped inputs. Pool FP64 SSE/
energy against actual captured BF16 output;report every block and actual
BF16 output bit/element mismatch count. No whole-model rerun is needed.

Precision-matched source contract qualifies for a separately frozen prior
variant only if BF16 replay SSE/energy<=1e-8 and BF16 SSE<=10% of FP32 SSE.
Otherwise stop that proposal pending source-replay diagnosis. A replay
pass cannot establish count gain,causality or final quality.

At known child156,replay exactly18 fit states/one unique input and its
BF16-rounded center. Report unique captured output count and point error
to captured target mean for FP32 source,canonical BF16 source,fitted
parent15,source child prior156,fitted child156. Keep the original learned
weights/codec/bias;no intercept oracle or correction is fitted.

At parent0,parent15,child156,compare five fixed input-direction probes
(axes0,127,511,895,center/||center||). FP32 reference uses the analytic
smooth source Jacobian. BF16 probe uses autograd JVP through the canonical
BF16 execution with straight-through dtype conversions and rounded
backward operations. It is **not a mathematical derivative of discrete
quantization**,nor guaranteed linear sensitivity. Report finite responses/
relative differences,without a sensitivity equality/quality gate.
Canonical point value repeats center to(4,128,896),returns first output;
this keeps the diagnosed capture geometry explicit.

## Budget and command

One local3060 run,six host threads,10min after imports,20GiB RSS,10.5GiB GPU,
small JSON only;reuse existing source/capture/snapshot,no T4/download.
Freeze source/protocol before execution;failure stage/completed replay
blocks/probes retained. No METH-240 n/function gate reopening.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth241_source_precision_replay.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth241_source_precision_replay_result.json
```
