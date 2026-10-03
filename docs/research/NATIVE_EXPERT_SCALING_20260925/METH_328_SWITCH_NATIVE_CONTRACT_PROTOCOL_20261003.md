# METH-328: complete native source-compatible Switch/T5 operator bridge

Prospective, after324 official tiny reference PASS. While real326 transfer is
live, implement necessary engine operators before full source mapping. Reuse
source smaller-active geometry321/325; high-active Giga319 remains closed.
This is a prerequisite, no synthetic-weight knowledge or rate promotion.

New opt-in SILICON_SWITCH_F32_REFERENCE prefix in engine.c includes complete
FP32 encoder/decoder/head bridge. Legacy default body remains byte-exact.
Manifest SWF32A01:13uint32 config values(D,FF,heads,DK,ENCL,DECL,N,capacity,
vocab,buckets,distance,encoder/decoder sparse steps), F32epsilon, uint32 file/
tensor counts; length-prefixed UTF8 absolute file names; each tensor(name,
file index,ndim,2dimensions,uint64 offset/elements). Read-only Windows mappings,
all record bounds and required source matrices checked; no default weights.
Preserve all actual source organs including full tied head, original relative
buckets, unscaled QK scores, RMS norm, ungated ReLU, probability multiplier,
per-call/per-sequence top1 capacity, encoder prefill/cross-KV, causal self-cache.
Manifest supports source256/128 geometry, but applicability to actual weights
unvalidated by this experiment. CTX<=256/batch1,32buckets/maxdistance128 only.
Current F32 arithmetic bridge is not final LUT/precision conversion.

Freeze source/controller/engine/protocol BEFORE compilation/observations.
Clang21.1.8 and libomp hashes from prior source cost apparatus; explicit
O3/x86-64-v3/no-fast-math/FPcontractoff/fopenmp. One CPU thread, unbound,
OMP_WAIT_POLICY=PASSIVE/KMP_AFFINITY=none/OMP_PROC_BIND absent as319 startup
repair. No performance measurement or overlapping benchmark; no GPU.
10min/3GiB combined sampled controller+childRSS guard,120s compiler timeout.

Official untouched4.57.6 source/hash324, Torch2.6, seed328. Two whole Tiny models
identical weights via repeated seed, same323 dimensions(D8/FF16/heads2/DKV4/
2layers each/onesparse each/E2/vocab32/dropout0/jitter0), capacities1 and64.
SourceIDs[2,3,4,5,6,7], decoderIDs[0,8,9,10]. Export all state_dict F32 tensors
and manifest with fresh hashes, no prior fixture reuse. Reference encoder
called once; four official cached decoder calls, including saturated encoder
capacity1 behavior. No saturated full-prefix equivalence assumption.

Require native full encoder initial/block/final states, every cached decoder
initial/block/final state and full32logits relativeL2<=1e-4 independently per
state AND pooled, nonzero reference norm, exact greedy top1 at all4steps.
Entire ordered raw-router choice/capacity acceptance exact, selected probability
maximum absolute error<=1e-6. Capacity1 must actually drop at least one source
token. Capacity64 unsaturated should preserve all call-level admissions.
Relative bucket boundary control: both causal/bidirectional source32/128 at
[-4096,-2048,-128,-64,-16,-8,-1,0,1,8,16,64,128,2048,4096], EXACT official IDs.

Explicit separate executable faults: zero relative bias, zero cached crossV,
omit probability multiplier, unlimited capacity(cap1 only), zero full head.
Each whole logit error must exceed1e-4. No source/gate/seed repair after outputs.
All cases retained even on numerical gate failures; exceptions preserve partial
observations and logs. Both frozen gates PASS permit separate actual source
mapping against completed327 weights, NOT final native quality/performance.

Command using isolated reference Python:
```
results\native_expert_scaling\meth324_switch_reference\venv\Scripts\python.exe benchmarks/native_expert_scaling/meth328_switch_native_contract.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth328_switch_native_contract_result.json
```

Real source engine loading/export, useful n/capacity behavior, precision/LUT/
DRAM/whole accepted>=50 and untouched donor-relative quality remain required.
