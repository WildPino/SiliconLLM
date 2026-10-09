# Target F32 SSD storage: qualified at the actual long training shape

9 October2026. [Frozen protocol](CHATBOT_TARGET_SSD_STORAGE_PROTOCOL_20261009.md).
Freeze ac941b5dd62d50f658f76d59f928279efc163ee4;37-input
[binding](chatbot_target_ssd_storage_binding_20261009.json) SHA256
63dc3b8fcece32f9f2f46e47e041b16c5e427fcdbdda07664241a9a1ce1fc3e5.
[Result](chatbot_target_ssd_storage_result_20261009.json) SHA256
f94b513f3a474ed65498e8abcba17844f99f2cf4274ed8e38595416093747516;
[terminal receipt](chatbot_target_ssd_storage_result_20261009.terminal.json).

## What changed and why it remains the same model

Four worker-local SSD contractions tile an existing independent output/chunk
axis and preserve each COMPLETE reduction. No source-file/model/precision/
chunk-size/ternary/routing/inference-geometry change. The largest product
is 75,497,472B per destination tile instead of7,247,757,312B at R96.
This is an algebraic identity in real arithmetic; observed F32 comparisons
below qualify its finite implementation scope.

Both original-cost cases reuse audited checkpoint286/Adam294 and restored
CPU/CUDA RNG, full-vocabulary case KL and training-only AQ dither .025.
No optimizer step. Original costs/resource failure remain retained.

| TF IDs | Original KL = tiled KL | Max relative gradient-group norm difference | Tiled forward s | Tiled backward s |
|---|---:|---:|---:|---:|
|58|10.07216739654541|6.495e-9|4.047|6.000|
|1507|9.960439682006836|9.421e-8|5.657|18.281|

Both211 gradients finite/present;all12 core/bank/norm groups positive/finite.
Every model/Adam step/moment tensor compared unchanged. Norm comparison is
not whole-model gradient direction equivalence: old full arrays were not saved.
Long tiled timing includes actual operand/output CPU capture; it is this
probe's observation,not a clean paired throughput benchmark.

## Actual operands and local adjoints

Saved FOUR block0 long-case operand sets,CUDA tiled outputs and actual loss
output adjoints. Each original/tiled isolated formula replayed F32 on CPU
with that SAME actual adjoint. All forward output coordinates are bit-equal;
7/8 input VJP arrays bit-equal. Interchunk state VJP relative RMS1.0091e-7,
max-scaled absolute1.4493e-7. CPU/CUDA tiled output RMS maximum9.6348e-8,
max-scaled absolute maximum1.6119e-7. All fixed1e-5/1e-4 gates pass.
These are actual longest-FIT block0 directions,not all blocks/unseen inputs.

## Resource outcome and decision

GPU allocated5,053,017,600B/reserved5,683,281,920B versus original long
19,312,518,656B/22,003,318,784B. Both fixed11/12GiB CUDA gates pass.
CUDA allocator counters do not certify physical VRAM residency/DRAM traffic.
Held worker OS peak17,366,630,400B includes isolated CPU dense reference
within its separately declared32GiB reference budget. Family114.266s,
worker106.344s,launcher8864/worker16768,exit0/session70614 closed.
11 outputs/774,614,745B;all input/resource/numerical/equality gatesPASS.

Decision TARGET_F32_SSD_STORAGE_PASS. Use the SAME bound helper in the
[one broader-data pilot](CHATBOT_BROAD_PILOT_PROTOCOL_20261009.md).
No source calls,updates,native/T4,quality/accepted50/useful-n/DRAM admission.
Original engine.c/LUT/ternary/compact SSM destination is unchanged.
