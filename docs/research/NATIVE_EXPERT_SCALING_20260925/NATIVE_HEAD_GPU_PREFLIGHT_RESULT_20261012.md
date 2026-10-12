# Current-head GPUF64 objective: complete numerical qualification

12 October2026. COMPLETE/full independent CPU audit PASS. Goal ACTIVE/INCOMPLETE;
no owned job live. No optimization or new native code/model produced by this run.

[Producer](../../../benchmarks/native_expert_scaling/native_head_gpu_preflight.py),
[frozen protocol](NATIVE_HEAD_GPU_PREFLIGHT_PROTOCOL_20261012.md),
[result](native_head_gpu_preflight_result_20261012.json).
All4422 FIT/all4386 DEV independently qualified initial v,m,H and CPU objective/
257-gradient/ID,current exported F32 head/bias/epsilon promoted F64,full65537
vocabulary. Torch2.6.0+cu124/NumPy2.4.6/psutil7.2.2;deterministic CUDA,float64,
TF32 off,workspace4096:8. No autograd/source model/fitting/T4/RESERVED/longalign.

| GPU versus original qualified CPU | FIT | DEV |
|---|---:|---:|
| Max absolute loss difference | 2.48690e-14 | 2.19824e-14 |
| Max absolute gradient difference | 2.53131e-14 | 2.48690e-14 |
| Gradient Frobenius relative difference | 3.62836e-15 | 2.60292e-15 |
| Argmax changes | 0/4422 | 0/4386 |

ALL frozen1e-8 absolute/1e-9 relative/zeroID flags PASS. All148 calls float64:
139 complete batches plus2 warmups/7 measured repeats on first FIT64. Repeated
outputs bit exact. Old real/native1e-5 flags remain FALSE;these are different
CPU/GPU F64 arithmetic checks,not a retroactive precision-gate promotion.

## Full independent audit

[CPU auditor](../../../benchmarks/native_expert_scaling/native_head_gpu_preflight_audit.py),
[protocol](NATIVE_HEAD_GPU_PREFLIGHT_AUDIT_PROTOCOL_20261012.md),
[adjudication](native_head_gpu_preflight_adjudication_20261012.json).
No Torch import/GPU replay. Independently recompute all577249896 probability
words:head sums eight32-feature blocks plus bias,logaddexp normalization,
4096-vocabulary expectation blocks,eight-coordinate moment dot sums. Verify all
GPU AND original CPU losses/gradients/IDs,shapes/dtypes/summaries/caps/flags.
Max independent relative2.77690e-11,reported scalar discrepancies0. Full relevant
input/output before/after hashes PASS. INDEPENDENT_STORED_PASS,GPU_numeric_pass=true.

## Cost scope and clock precision

GPU0 RTX3060 12GiB,driver610.47,
UUID GPU-1cce7fba-8103-2d4b-ad9b-df8e162f8221. NVML/driver and all Torch runtime
DLLs/relevant Python/compiled binary files bound. Desktop processes retained;
initial observed GPU usage1503MiB,not attributed to this nonexistent-at-that-time
worker. Timing admission=false;no clean comparative/optimizer/chatbot speed claim.

Synchronized resident-batch64 call durations:.031,.016,.031,.032,.016,.031,.031s;
median.031s,nominal139-batch pass **about4.3s**. Original CPU median.203s/pass28.2s.
These calls use Python3.12 time.monotonic/GetTickCount64;queried clock resolution
**.015625s**. Do not infer sub-tick GPU timing or a precise speedup ratio from these
samples. Aggregate total/resource measurements and broad budget estimate remain
useful;new long solver should use perf_counter for durations and monotonic for caps.

Two output packets18,251,672B. Whole run16.906s,OS1,208,004,608B;
GPU allocated236,201,984B/reserved243,269,632B. Preparation10.156s,handle15890
CLOSED0,no numeric inputs. Producer24009 CLOSED0,PID30420/create1791821766.9565108.
Source/protocol d84a2de,[launch](native_head_gpu_preflight_launch_20261012.json)c7123a8.
BindingSHAce9c84996482b22272f885c9a51949adf12d61a76f21c1a140167ec5ccef1fc2.
ResultSHAa5d87cedad1bc90636a1e18e611541617251d3e15c944164db8704df7b7d4aa2;
terminal6653f839032cdda1eb1cd64bc93336e38cb0243b295e6ccf534c981f1475cd6c.

Audit42536 CLOSED0,PID31496/create1791822119.755816,result57.359s/terminal57.375s,
OS478,359,552B. Source/protocol29699ce,
[launch](native_head_gpu_preflight_audit_launch_20261012.json)28e47d4.
AdjudicationSHA6497f552285b9bbc9811f817d18370dbaedc3a04be475cc934b6c6e65b2c54ce.
All600s/4GiB/2GiB allocated/reserved/64MiB producer and600s/4GiB audit caps PASS.
[Closure](native_head_gpu_preflight_session_closure_20261012.json);original engine/
foreign files unchanged,no owned process live.

## Selected next toward transfer

GPU backend qualifies for a **bounded all8808-label probability-image study**.
New [selected design](ALL_LABEL_NATIVE_PROBABILITY_GAP_DESIGN_20261012.md)
is PROPOSED/UNIMPLEMENTED/UNBOUND:convex half-ball projected descent/supporting
lowers,independently evaluated positive-sphere native feasible codes,unchanged
head and actual source distribution references. No winner-only ray variant or
unchanged old paired255 repetition. Existing sufficient moments avoid source
replay while searching;head changes would require their recompilation.

The study must distinguish fixed reader/domain limits from current code loss,
then return to causal compact SSM/SWA/ternary transfer. A good per-label oracle
code is not a learned encoder or preserved chatbot. Useful n/CPU IDs+mass/physical
DRAM/multiple families/same useful artifact>=50 remain unproved.
