# METH-248: frozen continuous readout consumed-validation diagnosis

METH-247's actual rank32 fit gain11.72% passes,but consumed E160 loses2.26%
to E16. METH-244 raw fit gain49.53% alone cannot establish generalization.
Determine whether original continuous readouts conserve **consumed** count
gain before considering a changed encoding versus continuous hierarchy.
This is a nondeployable diagnostic,no rank/strength/fit retry.

Bind METH-240 raw result/snapshot,METH-244 audit and METH-247 result
`a433b502960ee2c50c5ab1fa1b214d61327e4584935d27fa10682b08a4ab7ee9`,
actual151,680,804byte factored snapshot,original capture/route/native
identities. Replay all176 original fixed systems,require all regenerated
mixed coefficients/biases exact and raw fit scores within1e-12 relative
METH-244. Same stored priors,keys,variances,tau1024 and FP32 nonlinear
row-Q8/LUT features;these are FP64 **readouts**,not a fully FP64 ancestor
hierarchy or original donor. No source compilation/new label fit.

Predict raw E16/E160/rotated-child E160 in FP64 on same128 already consumed
input windows,cell by cell to bound memory. Read validation targets only
after all fit replays. Replay actual METH-247 factorized outputs on same
keys/features and require all per-window energies/encoded scores exact.
For each arm record raw SSE and actual encoded SSE. Ledger with
e=raw-target,d=encoded_F32_as_F64-raw obeys SSEencoded64=SSEraw+||d||²+
2<e,d>,relative closure<=1e-9. Keep FP32-subtraction encoded scores for
exact old-score control,distinguishing them from FP64 ledger subtraction.

Fixed diagnostic decisions: raw E160 SSE/energy<=.01,raw E160<=90% raw
E16 and rotated raw E160,and paired10,000-window-bootstrap gain P05>0,
seed248249. If all pass,raw consumed gain exists and a separately priced
representation mechanism may preserve it. Otherwise change continuous
function/hierarchy before codec-only count attempts. Rank32 factorization
remains closed either way. Neither outcome is fresh independent quality,
actual C bank/n/DRAM or full accepted rate/10B/second family evidence.

One local3060 run,six host threads,20min after imports,20GiB RSS,10.5GiB GPU,
JSON only;176 exact replays,weights reused,no new checkpoint/T4/download.
Keep partial completed cells/failure stage;freeze before execution.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth248_raw_validation_audit.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth248_raw_validation_result.json
```
