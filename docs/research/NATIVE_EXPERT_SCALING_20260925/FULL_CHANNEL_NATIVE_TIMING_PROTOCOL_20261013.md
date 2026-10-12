# Fixed full-channel site0/12/23 component timing after audit

ONLY after full independent numerical PASS. All3 fixed sites,two48-history
passes/site,288 case rows/135,282 supplied positions. Three shortest FIT history
warmups per site by length/ID. Native one thread/affinity0,ownnorm-only,reset
state per history. Inputs allocated/loaded outside each native call;C state
allocation/free/output writes/finalstate copy charged. Every normalized input,
rawF32 score,BF16 output/finalstate bit must match qualified artifact after call.

Per-call perf_counter,QPC stages B/C projection,scan,gate/RMS,normalize/output,
clock frequency/resolution,before-after process CPU snapshots. Complete stage
sum/ID/history arithmetic adjudication after finish. Actual DRAM bytes/clean
system isolation/whole-chatbot speed NOT admitted. Comparison with earlier384
pilot is different runs/workloads,not a simultaneous matched latency test.

Separate source input-generator6704x2048 GEMV at each site,on128 fixed arbitrary
vectors `ldexp((arange(128*2048)%31)-15,-4)`,one warmup/two passes. Source matrix
F32 54,919,168B/site. Final deterministic output858,112 coordinates/site compared
with full NumPyF64 contraction1e-5 outside timing;both repeats'checksum retained.
Separate generator/operator workloads,not measured joined source-layer time.
No source hidden/genconv/attention/FFN/head/causal target execution.

Budget900s/OS6GiB,no large new arrays written,all parameters/source/result hashes
before-after. No GPU/T4/refit/DEV choice/RESERVED. Freeze source/protocol/binding/
adjudication/compiler/runtime launch before timing. No n/actual expert DRAM/
multiple family/same useful model>=50 claim from component times.
