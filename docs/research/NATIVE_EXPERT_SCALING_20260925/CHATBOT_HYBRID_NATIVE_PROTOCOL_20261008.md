# Packed-only whole recurrent/ternary C prerequisite: frozen protocol

8 October2026. Freeze code/protocol before first export/new C observation.
Reuse final pilot learner SHA75b2317efe99fe66fc16f2b0e6df1f5001ef8b9c243c150b87b24e1f433793d9,
ALL6 calibration cases/32 saved final full-vocabulary learner logits, original
source contract and original phase60 engine.c matrix/LUT functions. No Python
model/source forward, training, donor selection or previous C packet replay.

Uncertainty: can the actual learned compact SSM/SWA/ternary target be exported
without expert reference copies and executed through original C kernel algebra
with accumulated recurrent state, preserving its whole output? This makes a
missing pipeline stage real before sustained transfer fitting. C numerical
failure stops long learning pending specific operator/format diagnosis. Even
a pass is not quality retention, accepted50 or useful large-n admission.

## Export/format, one actual conversion

Actual target254,932,736 F32 master/control parameters,211 named tensors. Replace
ONLY36 expert gate/up/down master arrays with byte-pair codes. CUDA F32
round(master/learned_scale), clip[-1,1], matching learner quantization primitive/
device; reconstruct all codes from pairs and require equality. Scales positive
>=1e-8. All other F32 fields unchanged. One64-F32 RoPE frequency field from the
learner's exact CPU formula (its source buffer was nonpersistent),256 extra bytes.
No P matrix in inference. No original expert master or unpacked int8 references
inside model or native loader.

Header80B little-endian:8-byte SLH1PK01,16 u32 version/dimensions/SWA mask/field
count/endian marker, u64 complete file bytes.212 records104B each:64-byte NUL
name, u32 dtype(1F32/2pair)/rank, four u32 dimensions, u64 offset/bytes. Offsets
contiguous/aligned4. F32 row-major; each expert code matrix tile-major
[expert,input_pair,output_row], codes0..8 representing(w0+1)*3+(w1+1).

Payload425,188,608B =340,253,952 F32-control/scale/frequency bytes+84,934,656 codes.
Whole file425,210,736B. Save each field and immutable metadata before final
stitching; partial converter repair reuses completed field bytes, not rerounding.
Native loader reads one blob, validates every dimension/type/extent/finite
coefficient/positive scale/valid code and exact one-time field use. No baseline
master/unpacked copies. Actual malloc/model/state extents and OS peaks recorded.

## C execution and reused original kernel boundaries

Extract hsum256/dotf/matvec/silu/softplus/acc_add_i8x32/matvec_lut_full/
build_lut_t3/bc_tm/ref_t3/quant_i8 body text directly from bound phase60 engine.c;
balanced braces, per-function body hashes, generated header bytes retained.
Original float dot uses explicit AVX2 FMA, original LUT int32 reductions/pshufb.
Compile clang -O3 -mavx2 -mfma -ffp-contract=off, no fastmath/OpenMP, one thread.
FMA intrinsic remains explicit; other scalar contraction disabled.

D512/L12,10 SSM768/N256/48head*16/conv4/gate-before-RMS, SWA5/11 with4head*128/
window128/theta1e11, F32 sensitive organs, E72/k8/h128 signed SwiGLU. Baked
source multipliers are not applied again. AQ63 reuses original quant kernel
for absmax>=1e-12; below it uses the learner's explicit1e-12 clamp rather than
old exact-zero scale policy. FE_TONEAREST, FTZ/DAZ off. Matrix products integer
then learned row-scale then activation scale, matching no-grad learner order.
Flat72 router stable lower-ID ties, top8 selected normalized softmax mass;
expert output accumulation ascending bank ID, matching learner index_add order.

Conv oldest->newest, learned A_log/delta-bias/D, softplus delta clamp0..infinity,
one exp(delta*A) per head, recurrence across all positions, gated group1 RMS
then out-projection/residual. SWA caches both K/V and applies actual RoPE/window.
State zeros only once per case; evolves continuously thereafter. Head computed
only at supervised positions (prefill last+continuation), full65,537 logits.
uint32 token IDs, no truncation. Original interaction already serialized in
retained fixture IDs; native tokenizer/generative chat loop not qualified here.

## Fixed complete numerical gates and retained diagnostic data

ALL6 retained prefixes,235 prompt IDs+26 continuation input IDs=261 total.
ALL32 positions/full65,537-vocabulary logits. Relative row logit RMS<=1e-4 each
and ALL greedy learner IDs equal, frozen in previous selected NEXT and unchanged.
No favorable case stopping; complete metric failure gets a FAIL result, not a
loosened tolerance or incomplete evidence promoted to pass. Source-model quality
is NOT the C parity oracle: compare the actual learned target's saved outputs.

Original integer LUT versus scalar fixtures include all9 valid pair codes,
signed ±63 activations, nearest-even half ties, zero/clamp and negative SiLU gate.
Native trace ALL261*12 records: residual/core input/output, FF input/output,
postblock state, ALL72 router scores/top8 IDs/mass. Independent NumPy stable
sort verifies ALL IDs and F64 mass error<=1e-6. Retain final recurrent/conv/KV
state for every case; all finite. Core/head/LUT timing is diagnostic, includes
trace/state writes, NOT end-to-end accepted token rate or physical bandwidth.
No source/Python target/cached pilot observation rerun to generate reference.

## Cost and resource guard, two sequential families

Export: local CPU checkpoint load/CUDA quantization only, hard600s FAMILY,
8GiB OS, GPU allocated512MiB/reserved1GiB, output1GiB (field parts+final file),
4MiB worker log. No subprocess/model forward or fitting. CPU6 threads.
Native: hard600s FAMILY/2GiB OS/256MiB output/4MiB log; compiler then native
single-thread child, never simultaneous experiments. Expected~425MB model,
9,117,696 actual logical state bytes+1,920 derived A bytes. Planned trace
39,588,480B/final states54,706,176B/logits8,388,752B, all retained.

Launcher holds worker handle through exit and verifies all bound inputs before/
after. Worker holds compiler/native handles through actual exit/peak working set,
records commands/PIDs/status/extents; compiler subordinate process names/RSS
sampled and guarded but their independent through-exit peaks are not captured.
Conservatively sum worker peak+direct-child peaks for cap; track compiler RSS
descendants. Name whitelist clang/ld.lld/lld/ld/hybrid_native only. No unexpected
children or overlaps; exact unrelated publisher exception remains preserved.
Selected code/Python/compiler/checkpoint/export/reference/foreign bytes and
isolated versions bound; not full DLL/compiler-library tree certification.

Retain every first fault and partial. Numbered repairs reuse completed export
fields and completed C outputs. A metadata/comparison fault after completed C
does not justify re-executing C. A partial native runtime fault requires a
separate missing-scope repair before any new execution; do not blindly restart
completed prefixes. Stop on shape/finite/resource/fixture/extent failure. On
whole parity FAIL, diagnose saved traces before changed-code observations.

## Outcome and next full goal work

Pass makes packed-only native whole-target execution a qualified short-prefix
pipeline stage; then broad balanced transfer calibration/long contexts/own-prefix
supervision and separately excluded fresh chatbot dialogue/tasks. Failure
localizes deployment-algebra work before long adaptation. No T4 launched here.
Whole useful chatbot capacity+>=50 accepted batch1 IDs/s on SAME artifact,
window eviction/long drift, canonical live chat, structured CPU LUT winner/mass,
useful much larger n/RAM, physical DRAM and actual other families/~10B/~100B
remain required, with no generality promotion from this family-specific target.
