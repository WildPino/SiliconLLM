# METH-305: exact signed-I8 biased storage/two-part activation cost

Freeze protocol/new C/controller/phase60 selector BEFORE observations.
303/304 established correct compact fixtures but stable native cost failure,
with304 projection work90.478%. This new variable changes integer instructions
and reversible weight storage, preserving signed-I8 precision, geometry and
full303 composition/routing/head math. No teacher collection/training yet.

## Exact identity, implementation and independent bounds

For original w[-127,127], store u=w+128 as unsigned byte[1,255]. For SAME
quantized x[-127,127], form h=trunc(x/2)[-63,63], l=x-2h[-1,1], sum_x=sum(x).
Exact dot:2*sum(u*h)+sum(u*l)-128*sum_x. Store identical seeded303 decoded
weights/scales in the same row/bank order; no weight/channel approximation.
One byte per weight retained. Input splitting adds two buffers and a scalar
sum; charge all preparation/correction inside measured operator execution.

Compiler avx2intrin.h confirms unsigned first/signed second operands and
saturating signedI16 adjacent-pair sum for `_mm256_maddubs_epi16`; bind its
file SHA. High pair maximum2*255*63=32,130 and low2*255=510, below32,767.
Multiply/add pair sums by I16 ones intoI32, accumulate high/low separately,
double high then add low and subtract128*sum_x. Full unsigned intermediate
absolute<=1536*255*127=49,743,360, correction<=128*1536*127=24,969,216;
all individual lanes and horizontal intermediates fitI32. Final signed
absolute<=1536*127*127=24,774,144. Direct unsplit byte-pair product can
saturate and is explicitly forbidden/tested as a negative control.

308 matrices,273,571,840 active coefficients/1,902,656B row scales and
complete descriptor447,357,600B/token remain303. Physically populate ALL
9,437,184,000 routed coefficients/640 functions per layer; total I8 bank
9,651,773,440 coefficients. Only64 source macro parents are real. The
synthetic children are neither learned nor useful extra donor capacity.
Original source Q6 full head/router/norms/components/spec SHA remain exact.

## Controls and frozen gates

- Verify complete execute, preparation/router, Q6 decoding/reference and
  scalar source route blocks byte-exact303; F32 quantizer exact after removing
  only the additional integer split. Preserve old sources/controllers.
- All65,025 representable weight/input combinations,32 repeated coefficients
  each, must match scalarI64 and32*w*x. Verify5,929 mixed adjacent pairs from
  seven signed weights{-127,-126,-1,0,1,126,127} and eleven inputs
  {-127,-126,-3,-2,-1,0,1,2,3,126,127}, including exact high/low pair lanes,
  signed extreme dots, source ties/zero/NaN controls. Direct255*127 pair
  saturates to32767 rather than64770 and new split dot stays exact. Inject
  sum_x+1 to prove row-bias corruption changes the independently decoded dot.
- Keep303's97,194 bank-edge checks,7,176 byte-exact scaled/scalar rows,
  relativeL2<=1e-6, real Q6 pooled64-row<=1e-5,25-layer source macro/child
  controls and bad-bank/packed-head faults. Entire old selftest fields exact.
- Every one of the30 complete output/route hashes must equal original303,
  not just repetitions within this candidate. All outputs finite; same10
  inputs×3 repetitions/two warmup/eight measured, full bank unchanged.
- Source/header/component/spec binding, active/stored/allocation counts exact.
  Complete descriptor<=560MB, max/min three medians<=1.10 and EACH<=14ms.
  Variation stops as inconclusive; stable failure closes this unchanged
  implementation before capture/training. No favorable subsets/regrading.
- A pass only licenses separately frozen complete teacher-function fit. No
  fixture timing can prove learned quality, useful n, actual causal KV/head/
  routing, physical DRAM or accepted decode rate. Original303 failures stay.

Timing includes all original input norms/conversion, integer projections,
source macro routing/child search, selected FFNs/SwiGLU/down/mixture and full
source Q6 head/Q8_K input conversion, plus split/correction work. Input-ready
latent contexts, absent causal attention/cache/RoPE/embedding/layer-to-layer
residual composition remain303; retain6ms balance for omitted model costs.

## Resources and reproduction

Local Ryzen5 3600X/six CPU threads, no model job overlap/GPU/T4/download/
delegation. Expected CPU seconds to minutes;600s child/24GiB peakRSS/120s
compile,>=16GiB available. Fresh~172MB source component hashing and~10GB
physically populated bank, two added static input buffers per query. Source
allocation byte ledger stays exact303; actual peakRSS records buffer cost.
Save new isolated logs/spec/executable/result/failure, preserve originals.

```powershell
& .\.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth305_biased_i8_preflight.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth305_biased_i8_preflight_result.json
```

The final pretrained-transfer/useful RAM-scale n/quality/SAMEartifact accepted
>=50 batch1token/s/multiple-family and scale requirements remain unchanged.
