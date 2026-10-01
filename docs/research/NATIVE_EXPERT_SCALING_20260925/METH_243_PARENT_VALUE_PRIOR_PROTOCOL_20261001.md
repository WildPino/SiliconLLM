# METH-243: frozen fitted-parent value and source-slope hierarchy

## Changed information coupling

METH-240 full-feature E160 loses22.39% to E16;METH-241 establishes a
BF16/FP32 source-value mismatch and a repeated-input child where source
reset erases better fitted parent output. METH-242 corrects source values
to BF16 yet loses24.03%. Source-value precision alone is closed as the
count solution. Test a different hierarchy: preserve **fitted parent value**
at each child center, retaining original-source child slope information.
This changes prior semantics to preserve learned parent adaptation; it
does not claim exact original donor values or BF16 source derivatives.

Use the original METH-240 checkpoint,not the rejected METH-242 artifact.
Bind METH-240 result/snapshot SHA256
`3c02724b1194f305036aae6896a489e1d19b477d7d8fe1d3ddc45fa9579fc505` /
`23f4a92b8092ccfbf79575ecd2dd59bd300b50fd6124e818cfbf862327def611`,
METH-241 diagnosis and METH-242 result SHA256
`a3bb1e94ff8decc40363b904c0b39a3d583d13d57a721e69fd1e6d913ea5ddfd`.
All original source/capture/native/route bindings remain required.

## Fixed algorithm and gates

Keep every actual parent/child coefficient array,source-FP32 sensitivity
projection,input map,LUT,key,label,feature variance,codec and tau1024.
For each160 center c,compute actual mixed fitted parent output p(c).
Set `bp_new=FP32(p(c)-actual_mixed_child_readout_without_bias(phi(c)))`.
No return to original-source point value. Source-derived encoded slopes
remain those qualified in METH-240,including the distinct repeated-input
child; no new slope/data fit or hash noise.

Transport the fitted child bias by METH-242's centered-ridge identity:
`delta=FP64(bp_new)-FP64(bp_old)`, `gamma=n/(n+1024)`,
`bf_new=FP32(FP64(bf_old)+(1-gamma)*delta)`.
The constant prior-value change cancels from centered slope residuals;
all slopes and their codec selection remain unchanged. Independently
check actual center conservation relative to p(c)<=1e-5 and defining
FP64 mean/intercept equation relative to mean-target norm<=1e-5 for all160.

Require same physical tensor keys/readback,all tensors except
child_prior.bias/e160.bias byte-unchanged,all176 effective weight hashes
distinct/equal METH-240,exact fit counts/label hashes and parent fit scores.
Then score same128 consumed windows once with actual mixed operator for
parent_prior/E16/hierarchical child prior/E160/rotated-child E160. Energies
and every parent score must replay exactly. No new data or fresh-quality claim.

Retain absolute E160 SSE/energy<=.01,E160 SSE<=90% of each E16,rotated E160,
child prior and parent prior. Paired10,000-window-bootstrap gain P05>0,
seed243244. Report all audits,fit metrics,validation rows,physical/effective
hashes and cost. Failure closes this fixed parent-value/source-slope
hierarchy; no strength/count or interpolation-coefficient retry. A pass
licenses a separately frozen native stored-bank route/LUT/DRAM test only.
Full model,independent prediction/generation/tasks,>=50 acceptedtok/s and
actual larger-donor capacity remain unestablished.

## Budget and command

One local3060 run,six host threads,10min after imports,20GiB RSS,10.5GiB GPU;
4GiB free disk,one complete <1.7GB snapshot plus JSON. No T4/download/new
label solve. Freeze source/protocol before execution,preserve failure stage.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth243_parent_value_prior_pair.py --checkpoint results/native_expert_scaling/meth243_layer12_parent_value_functions.safetensors --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth243_parent_value_prior_result.json
```
