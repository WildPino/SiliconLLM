# METH-306: packed-I4 controls pass, passive-policy cost fails

**Stable native cost FAIL before collection/training.** Freeze de3d67d;
terminalexit0, controller6.516s/native4.046s including3.270s initialization.
Packed-I4/four-row tiles/serial tiny-query candidate retains all9.437B routed
coefficients and640 synthetic functions per layer; coefficient precision
changed, so it has no source-I8 hash/quality equivalence claim.

Complete descriptor310,571,680B/token PASS560MB; active273,571,840
coefficients use136,785,920 code bytes and1,902,656 row-scale bytes. Total
9,651,773,440 coefficients use4,825,886,720 packed code bytes; allocation
5,106,289,792B/peakRSS5,127,254,016B. Original Q6 full head/source router/
norms/components remain, physically populated full bank consulted.

Own format fidelity PASS:7,176 exact scalar/scaled rows,97,194 bank-edge
checks,4,080 nibble/input combinations/16,320 tile-lane checks,30,976 mixed
adjacent pairs; signed extrema/quantizer/sum-bias/packed-nibble/head/bad-bank
faults detected. Integer decoded relativeL2 3.525595089393004e-8<=1e-6,
real Q6 head64-row1.9982288773633764e-7<=1e-5,25 source macro/child controls
pass. All30 own output/route hashes repeat exactly; selected-child union36–40.

With explicitly PASSIVE OpenMP wait policy, three medians
**21.5582/20.1688/20.4452ms FAIL14ms**, repeatability1.0688886PASS. No
source-trained weights/function quality/causal LM/useful n/physical DRAM or
accepted-token rate measured. 305 and306 combined precision/scheduling runs
do not isolate each change's effect; reducing bytes did not meet the budget.

Next uncertainty: hundreds of small OpenMP regions may pay worker sleep/wake
cost because the runners explicitly set passive waiting. Freeze a bounded
PASSIVE/ACTIVE/PASSIVE comparison using the EXACT306 executable/spec/math,
all prior output/control hashes and unchanged14ms gate. Runtime waiting is
an execution-profile variable, not a geometric/source-quality rescue. Do not
train this passive profile or assume active waiting will pass.

## Bindings and independent recomputation

- [raw result](meth306_packed_i4_preflight_result.json), SHA256
  `bb723d4cdb5027869e26668bf652fdd122ffe6a66385ca2608208fcd24328182`.
- C `c3a1610dfeef1b98edd9d5d8e41509493084841c8cb3f70259ecafcb4f2e729c`;
  controller `21ed4f3563b30dbd9021a3f4f01d9da1f3d9eb14ba784d241ff81f54a9fa1f26`;
  phase60 `0071a2101a3b36871ed705faa5fef1ccc87116ab1dc79ee57820e8d9c86dac11`.
- Executable `f7a22f25a7531bce4d9f8d67a05839039e764f9889907af69bc2727ccfa16de1`;
  spec/source component bindings exact303. Stdout
  `results/native_expert_scaling/meth306_packed_i4/meth306_native_run1.jsonl`, SHA
  `b80ebad4c29e01ced8ff3e644b028e6fb60cc9843acb4fdc8162898b6b4b1c27`;
  stderr/compile diagnostics empty.

Independent three-median/hash/file-binding recomputation passes. Frozen
source/controller/engine bytes matched filtered HEAD before execution. Old
sources/results retained. Final complete transfer/quality/useful RAM-scale n/
SAMEartifact accepted50/multiple-family and scale requirements remain open.
