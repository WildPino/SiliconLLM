# METH-356: ALL serialized I8 inputs use A16, full numerical contract

Freeze BEFORE compile/reference observations.351 whole multispan fidelity
FAIL retained30ca546;355 consumed-state attribution PASS retained3dc671b:
original-state/SAME A16 head leaves8 versus40 masked changes; native original
F32 head leaves40, unquantized head activation leaves40. At32 natural first
divergences original state matches23, native F32 head4. This motivates upstream
activation precision, not a claim that weight error or routing can be ignored.
Single new execution variable: all quantized core/expert inputs use A16 rather
than A8; head already A16. SAME338 actual14.818GB I8 codes/scales/F32 controls/
router/shared lookup/all12x256 banks. No new payload, checkpoint, training,
functions or source data.351 failures remain; no354 rate run.

## Mathematical and native change

A16 per-row input scale=F32(absmax/32767), zero vector scale1; nearest-even
F32 division/rint clipped[-32767,32767]. Exact I8xI16 AVX2 VPMADDWD I32 lanes,
cols<=4096: each lane<=512*127*32767=2,130,641,408<INT32_MAX. Horizontal/tail
I64 global sum<=17,045,131,264. F32((F64dot*F64weightScale)*F64inputScale).
Apply in scalar mv and existing exact batched encoder QKV/cross-KV mv_batch;
all I8 projections, not just head. F32 router mv/norm/attention/softmax/
lookup/cache/capacity/greedy lowest-ID/EOS/closing semantics unchanged.
Logical matrix code/scale byte descriptor remains129,017,344B/decoder position;
activation scratch doubles only, stored learned-capacity payload unchanged.
Dynamic n bounded by existing loader/RAM, real256/Tiny2 only qualified here.

New356 native source/cost entry/natural-generation entry, opt-in
SILICON_SWITCH_W8A16_ALL in engine.c; old default body BYTE EXACT0ff9705.
Existing --int-dot now exercises general A16 mv and emits SW16R001, as does
--head-int-dot for head_mv; this profile's explicit changed control format.
Independent reference scopes R336 integer-projection replacement to qualified
R345 NumPy I64 A16 projection for EVERY serialized quantized Linear. Existing
F32 controls/rounded norm/attention semantics reused; restore projection and
all forwards on exit including exceptions. Official META architecture loaded
only with target controls; no unquantized full donor fallback.

## ALL numerical gates

Physical HEAD source/controller/protocol/reference/dependencies/engine/raw
bindings, pinned official4.57.6/Torch2.6.0+cu124/source324/compiler/DLL hashes.
Both general/head native primitive paths:13 cases tails8/15/16/17/31/32/33/
768/3072/4096, ties, zeros, signed4096 extremes; exact codes/scales/I64 dots/
independent Python scalar/F32 results, extrema17,045,131,264. Existing347
primitive input/parser and345 projection reused, not changed frozen sources.

SAME328 Tiny source seed/weights/capacities1/64, actual serialized target.
Full independent teacher arrays and natural own-cache greedy cap16/closing31
exact C for threads1/6, profile0/1 and warm/repeat; matched official.generate
vs manual-cache choices/logits exact. All nine prior attention/cross-V/gating/
capacity/head faults >1e-4 relative detected. Save full arrays/hashes.

Fresh whole338 actual payload/spec SHA/bytes with size/mtime fixed throughout.
Whole engineering two sources: independent full encoder/decoder/logits/routes
BYTE EXACT, forced source/decoder fixtures347 and natural cap16/closing32098,
threads1/6 xprofiles0/1, generation warm1/rep1 all exact. No unchanged347
upstream byte equality expected: upstream precision deliberately changes.
Both long original fixtures347 (source9/64, forced32 positions): independent
complete reference ALL states/routes/logits exact Cthreads1/6. Both fixed
consumed351 multispan development cases(book0/case0,book23/case3), source29/
forced14 plus natural cap64/closing32095: independent whole outputs exact,
threads1/6 warm1/rep1. Their quality/elapsed observations cannot qualify NEW
quality or accepted rate. Choices/cache follow changed candidate naturally.
Scoped reference restoration required. No optional case pruning/reruns.

## Budget and next gate

CPU only/sequential original reference1/native1or6; passiveOMP/KMPnone/
OMP_PROC_BIND absent. No other model/native timing worker. MAIN<=30min after
imports/combinedRSS<=16GiB. Original F32 full model is not loaded. No GPU,
network/download or fit. Preserve failure and all partials FIRST before a
new named repair. Numeric PASS licenses separately frozen357 actual cost
on SAME compiled356 executable/payload (forced-fixture scope, not accepted
rate), then NEW source-only cohort/original-primary FULL prediction/natural
generation/known-answer/fidelity/health with unchanged351 criteria. Only full
quality PASS licenses accepted FULL rate on that SAME artifact. Larger-n/
LUT/physical DRAM/families/~100B final goal remains open.

```powershell
results\native_expert_scaling\meth324_switch_reference\venv\Scripts\python.exe benchmarks\native_expert_scaling\meth356_switch_all_a16_contract.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth356_switch_all_a16_contract_result.json
```
