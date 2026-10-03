# METH-303: compact I8 functions, source router and actual full Q6 head

Freeze protocol/controller/new C source/opt-in phase60 selector BEFORE native
observations. New scientific variable: the joint compact geometry proposed
after301/302; no full-source vector4/table-only rescue or donor-quality claim.
This is an enabling cost screen BEFORE teacher collection or training.

## Geometry and source binding

26 layers, residual1536, KV512, vocabulary128256; MLA8 heads with query192
and value192 per head, KV-A576; first dense1024, shared256, routed128.
25 layers retain64 source macro parents/top4. Each parent has TEN width128
synthetic nonlinear children:640 functions/layer, one selected per parent.
The fully populated routed I8 bank has9,437,184,000 coefficients, exactly the
original64x1280 routed coefficient capacity. These are deterministic fixtures,
not transferred or useful new functions. No synthetic100B/extra parents claim.

Use fresh descriptor/header checks of the existing6,474,702,976-byte Q4 GGUF.
Read original Q6_K full output head, F32 source attention/FFN/KV/final norms,
macro routers and correction biases; hash every consulted source component
in Python and again in C. Source config/hash fixes noaux_tc/sigmoid/top4,
one group, normalized unbiased gates, scaling1. The binary2000-byte catalogue
binds component offsets and SHA256. Previous whole-file SHA is provenance;
the6.47GB full payload is not rehashed. Preserve all immutable old helpers.

308 generated I8 matrices price273,571,840 active coefficients/token and
1,902,656 FP32 row-scale bytes. With source router9,830,400B, Q6 head
161,602,560B, norms/bias/one embedding row386,144B and selected child keys
64,000B, complete addressed-weight scenario is447,357,600B, required<=560MB.
This is a descriptor address count, not hardware DRAM traffic.

## Executed operator and explicit boundaries

Actual `engine.c` selector `SILICON_COMPACT_I8_PREFLIGHT` enters new source.
Six CPU threads, no fast-math, fixed source F32-product/FP64-sum router helper,
row-major signed I8 weights, FP32 scales. Each generated stored bank has
distinct seeded code bytes/scales and is fully allocated/written in RAM.
Activations convert by max(abs)/127, nearbyintf ties-to-even, signed[-127,127],
zero scale1; reject nonfinite/invalid inputs. AVX2 sign-extends toI16, uses
non-saturating multiply/add and exactI32 sum. Maximum1536x127x127 fitsI32.

Timed region includes source input norms; query/KV/K-B/V-B/output projections;
KV norm; all I8 activation conversions; full source F32 macro scan/bias/stable
top4/unbiased gate normalization; I8 shared16D child query; ten F32 keys for
each selected parent/stable top1; actual selected bank offsets; complete
routed/shared/dense SwiGLU/down and mixture; final norm/Q8_K activation
quantization and ALL128256 original Q6 head rows using the pinned AVX2 helper.
Distinct K-B/V-B heads and four routed downs have distinct input vectors.

Inputs for each layer and eight latent attention contexts are prepared
outside timing. Causal attention, softmax, RoPE, KV-cache read/write, embeddings
and residual layer-to-layer composition are absent. KV norm/projection and
K-B are computed and retained in output hashes but no causal attention is
formed. Input conversion/nonlinear cost is real; this is still an input-ready
operator fixture, not a complete model or accepted decode token. Initialization,
selftests, hashes/logging are outside the timed region and separately priced.

## Frozen controls, gates and stop

- 308 catalogue matrix shapes/active counts/scales and complete byte arithmetic
 must reconcile independently. Verify first AND last eight generated code
 bytes of EVERY stored bank; sample eight spanning rows of every active bank.
 Exact AVX2 I32 equals scalarI64; scaledFP32 result byte-exact with scalar
 same grouping; pooled decoded-FP64 relativeL2<=1e-6. Test signed extrema,
 zero/ties/NaN quantizer and reject bank ID==bank count before pointer reads.
- Independently recompute source macro probabilities/ranking/gates and child
 ranking using scalar roundedF32 products/FP64 sums and separately sorted
 score/ID records. All25 routed layers must have exact probabilities, IDs,
 gates and actual gate/up/down selected-address binding.
- Independently unpack64 uniformly spanning original Q6 rows; compare native
 logits to FP64 Q6/Q8 decoded products, pooled relativeL2<=1e-5. Deterministic
 search across those rows/first128 packed bytes must find a one-bit change
 detected by both reference/native outputs; restore before timing. This
 sentinel search is a fault sensitivity test, not favorable quality selection.
- All core/KVnorm/mixture/head outputs must be finite. Same10 fixtures across
 three repetitions, two warmup/eight measured each; retain all30 timings and
 complete output/route hashes. All three hash sequences must match exactly.
- Repeatability max/min of three measured medians<=1.10. If it fails, stop
 inconclusive. EACH median<=14ms is required, leaving6ms of the20ms target
 for omitted model costs. Stable failure closes this UNCHANGED implementation
 before teacher collection/training. Do not adjust gates or choose subsets.
- A pass only licenses separately frozen whole-function fit/teacher target
 planning. It proves no approximation fidelity, beneficial child diversity,
 source region exposure, composed quality, large-parent scaling or final rate.

## Resources and reproduction

Local Ryzen5 3600X/six CPU threads, approximately80GiB installed RAM; no GPU,
T4, downloads, new agents or simultaneous model/performance job. Expected
seconds to CPU minutes; hard600s child/24GiB peakRSS, at least16GiB available
before launch, compile120s. Controller polls RSS every5s; C guards after
initialization/repetitions. Read~172MB source components, populate~10GB RAM;
save executable/spec/rawstdout/stderr/JSON, no trained model output.
Preserve failed logs and failure stage before any narrowly frozen repair.

```powershell
& .\.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth303_compact_i8_preflight.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth303_compact_i8_preflight_result.json
```

Source quality/generation/tasks/useful n and SAMEartifact>=50 accepted batch1
token/s remain mandatory, with verified multiple-family/scale applicability.
