# Actual packed compact target and first original-kernel C execution

9 October 2026. Goal ACTIVE/INCOMPLETE. **Export COMPLETE; native prefix parity
FAIL; independent saved-output audit PASS; common-input bank diagnostic COMPLETE.**
This is a real deployment path for the learned target, with a numerical gap and
poor donor-relative quality. Neither chatbot preservation nor accepted50 is admitted.

## What became executable

Reuse the [actual whole learner](CHATBOT_HYBRID_PILOT_RESULT_20261008.md), its final
checkpoint and ALL32 saved full-vocabulary outputs. No source generation,
Python whole-model forward, fit or completed C prefix was repeated in this work.

The target is D512/L12: ten SSM768/N256/48heads*16/conv4 blocks and two
4head*128 SWA blocks/window128, E72/k8/h128 ternary signed SwiGLU/AQ63,
F32 control organs, V65537 untied head and uint32 IDs. Flat routing is temporary.

The [exporter](../../../benchmarks/native_expert_scaling/chatbot_hybrid_native.py)
replaces all36 master bank arrays with CUDA-quantized tile-major byte-pair
codes. Decoded codes equal the learner's quantized weights. The211 state fields
and64 derived RoPE frequencies give212 export fields. The P basis and
optimizer are conversion data, absent from inference.

| Actual exported/allocated extent | Bytes |
|---|---:|
| Pair-code coefficients |84,934,656|
| F32 control/scale/frequency coefficients |340,253,952|
| Coefficient payload |425,188,608|
| Header and descriptors |22,128|
| Whole model file AND native model heap |425,210,736|
| Native expert master/unpacked reference copies |0|
| Recurrent/conv/KV state heap |9,117,696|
| Derived per-head A array |1,920|

The [C target](../../../benchmarks/native_expert_scaling/chatbot_hybrid_native.c)
extracts unchanged function bodies from original
[phase60 engine.c](../../../benchmarks/phase60/engine.c): hsum256, dotf, matvec,
silu, softplus, acc_add_i8x32, matvec_lut_full, build_lut_t3, bc_tm, ref_t3,
quant_i8. Per-body hashes and generated header are retained. AVX2/FMA integer-LUT
machinery remains original. Target projection/conv/per-head recurrent scan,
gate-before-RMS, signed SwiGLU, two SWA sites, packed-only loader and32-bit IDs
are explicit adaptations. AQ uses the learner's clamp only below absmax1e-12.

This is a standalone compact target using original kernels, not yet an
integrated tokenizer/chat client in phase60 engine.c. Every model field is
validated and consumed once; there is no live donor or reference expert fallback.

## First whole evolving-state observation and fixed failure

[Frozen native protocol](CHATBOT_HYBRID_NATIVE_PROTOCOL_20261008.md),
[actual native result](chatbot_hybrid_native_result_repair3_20261008.json) and
[terminal receipt](chatbot_hybrid_native_result_repair3_20261008.terminal.json).

ONE native execution processes all6 prefixes,235 prompt IDs+26 continuation
input IDs=261. State zeros once per case and evolves thereafter. All32 supervised
positions emit all65,537 logits. All261*12=3132 block/router records and all6
final states are retained. Prefixes are short (max57); window eviction is untested.

| Frozen necessary gate | Observed |
|---|---|
| Each logit row relative RMS<=1e-4 |13/32 PASS, **19/32 FAIL**|
| ALL32 greedy learner IDs equal |32/32 PASS|
| Maximum relative logit RMS |0.00783855972784284 = **0.783856%**|
| Original integer/scalar-LUT/rounding fixtures |PASS|
| Packed-only bank residency |PASS|
| ALL3132 flat router ID and normalized-mass checks |PASS; max mass abs error2.5844930607e-7|
| Trace and final-state finiteness |PASS|

Gate failure stays **NATIVE_PREFIX_PARITY_FAIL**. Matching32 IDs neither removes
the logit failure nor recovers source quality. The learner's calibration DEV
decisions still disagree with the donor12/12; FIT10/20 disagree after8 updates.

The [independent saved-only audit](chatbot_hybrid_native_audit_result_20261009.json)
checks all212 payload hashes/extents against the export manifest, code ranges,
F32 finiteness and actual reference-copy absence. All2,097,184 C/learner logit
coordinates are represented as exact integers times2^-149. For threshold1e-4,
the exact squared-energy test is `error_integer*100000000<=reference_integer`.
It certifies the same13 passes/19 failures and all32 greedy IDs. **Audit PASS
certifies a native FAIL**, not a successful native candidate. No numerical replay.

## New common-input bank evidence

[Frozen protocol](CHATBOT_HYBRID_COMMON_BANK_PROTOCOL_20261009.md),
[result](chatbot_hybrid_common_bank_result_20261009.json),
[terminal](chatbot_hybrid_common_bank_result_20261009.terminal.json).

All3132 C-trace FF input vectors were supplied to the stateless CUDA bank
function, using ACTUAL packed codes/scales/router, TF32 off. These are common C
operands, not a replay of the GPU whole-model trajectory. No SSM/SWA/head/source
forward or learning occurs. All new FF outputs and per-frame metrics are retained.

- Router IDs agree in3132/3132 rows. Maximum score difference4.768371582e-6;
  same-ID normalized-mass error<=6.258487702e-7.
- 3131/3132 FF relative RMS values are<=1e-4; their maximum is1.462580588e-6.
- ONE material row: layer7/global input160, `fit_code` position33,
  RMS0.008310663371334967 (**0.831066%**), same IDs
  `[52,29,10,39,1,49,40,2]`. Decision **COMMON_INPUT_BANK_DIFFERENCE**.

Both later supervised `fit_code` positions41/42 pass the original whole logit
gate. Therefore this anomaly is a real common-input bank backend difference,
but cannot be claimed to explain the19 whole failures in other cases. Small
upstream differences crossing activation-rounding boundaries remain a hypothesis.
With `q=round(63*x/absmax(x))`, crossing a half-integer is discontinuous; low
floating RMS alone cannot guarantee equal integer codes. No causal localization
of a particular scalar/quantizer operation has been performed yet.

## Actual cost and limits

Local Ryzen53600X/80GiB/RTX306012GB, isolated Python3.12.10,
Torch2.6.0+cu124/NumPy2.4.6. No T4. Families run sequentially and all exit0.
Code/dependencies/inputs frozen before observations; bindings checked before/after.

| Completed family | Family wall seconds | Worker OS peak through exit | GPU allocated/reserved peak |
|---|---:|---:|---:|
| Export |167.157|3,625,799,680B|37,896,192 /39,845,888B|
| C build/execution/comparison |16.953|157,024,256B|none|
| Exact saved-native audit |5.750|202,600,448B|none|
| Common-input bank probe |8.937|1,175,388,160B|110,172,160 /140,509,184B|

Native direct child actual wall4.750s/OS peak439,676,928B through exit;
compiler direct child2.422s/4,775,936B. Compiler descendants are sampled/guarded,
not independently measured through exit. Conservative worker+direct-child
peak sum passes2GiB. All declared successful-family resource gates pass.

C component clocks: core1.934341699s, MLP0.733192300s, head0.267135800s,
forward2.936124400s, process4.191344700s. Trace writes are included; head runs
only32 times versus261 core steps. **Do not divide261 by these clocks and call
it accepted token rate.** No own-history generation, full-head batch1 rate,
physical DRAM-bandwidth certification or large-n routing measurement exists here.

Export426 files total850,545,792B includes immutable field parts plus stitched
model (disk duplication, no native reference copies). Native trace39,588,480B,
states54,706,176B, logits8,388,752B. Common probe output7,716,515B.

## Retained first apparatus failures

Before any whole C model execution:

1. Initial compiler wrapper spawned unlisted `clang-21.exe`; launcher stopped
   the apparatus. [Failure](chatbot_hybrid_native_result_20261008.launcher_failure.json).
   [Repair1](CHATBOT_HYBRID_NATIVE_REPAIR1_20261008.md) binds the actual driver.
2. Repair1 spawned system `conhost.exe`, rejected by the child guard.
   [Failure](chatbot_hybrid_native_result_repair1_20261008.launcher_failure.json).
   [Repair2](CHATBOT_HYBRID_NATIVE_REPAIR2_20261008.md) uses hidden child processes
   and binds the exact system helper path.
3. Repair2 compiler exited1: target macro K replaced original kernel parameter
   names. [Failure](chatbot_hybrid_native_result_repair2_20261008.launcher_failure.json).
   [Repair3](CHATBOT_HYBRID_NATIVE_REPAIR3_20261008.md) puts target K after the
   original header, leaving every original function body unchanged.

All raw failures/diagnostics and bindings remain. No successful export or
completed C/Python/model observation was rerun to repair these build faults.

## Identity, command records and next decision

Final learner SHA75b2317efe99fe66fc16f2b0e6df1f5001ef8b9c243c150b87b24e1f433793d9.
Actual model SHA191d1d20058946702330038336553cd6588efbcebdbac85f70410a03126c3571.
Actual C executable SHA4d0dd12d91dc17479668ce0adfd18a5bea2676ab5eb126b8334ce2869f2ae9c5.
Native trace SHA7adc3b0bac9a98d3740221d3e75ad491df32565359a0650bcecd41ab0b969c0f.

| Stage | Code/protocol freeze | Binding |
|---|---|---|
| Export |dfb4c0e3d3a59f497905ee00f2f7284b7f35f373|[export](chatbot_hybrid_export_binding_20261008.json)|
| First successful C |e912e6e14a47ef91c0432aac64aeda824df6e096|[native repair3](chatbot_hybrid_native_binding_repair3_20261008.json)|
| Saved-only audit |d6210ee89999b8b480e724c4db33c81dfcbf154e|[audit](chatbot_hybrid_native_audit_binding_20261009.json)|
| Common banks |5c8acec2e2fe0c9ab0020cc7fa0ed2d2fbcdd2eb|[common banks](chatbot_hybrid_common_bank_binding_20261009.json)|

Each terminal receipt contains the exact command, PID, elapsed time, result
hash and saved extents. Older bindings refer to launcher bytes at their freeze;
later schema additions must not be substituted when reconstructing those runs.
Large checkpoints/model/trace/outputs remain off-repo at the bound absolute paths.

**Decision:** packed native conversion and evolving-state execution are real;
strict numerical fidelity still fails. Follow
[bounded backend diagnosis then broad transfer](CHATBOT_HYBRID_BACKEND_ADAPTATION_NEXT_20261009.md).
Preserve original kernels and qualify any scalar/core changes against their own
learner forward. Longer training addresses source quality, not apparatus error.
Compact useful chatbot quality AND accepted50, useful much larger n, structured
CPU LUT IDs/mass, physical DRAM and other actual families/scales remain open.
