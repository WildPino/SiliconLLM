# Pretrained-to-native conditional-capacity method

9 October2026. Branch `research/native-expert-scaling`. Goal ACTIVE/INCOMPLETE.
This is the procedure in construction, not a completed universal converter.
[INDEX](INDEX.md) gives the current resumption. Older procedure/history is
preserved BYTE in [method through pilot](METHOD_THROUGH_HYBRID_PILOT_20261008.md)
and [method through498](METHOD_THROUGH_498_20261006.md); old NEXTs are historical.

## Required result and actual scope

Pretrained CHATBOT -> compact reusable SSM/SWA core + useful selectively consulted
ternary functions -> original engine.c machinery. Preserve tokenizer/roles/history/
stopping and fresh donor-relative dialogue/generation/tasks, with >=50 accepted
batch1 IDs/s on the SAME artifact (100 stretch). Require useful larger n/RAM,
structured CPU LUT winner AND normalized mass, physical DRAM and actual variants
for additional families/~10B/~100B as resources permit.

**Available:** source triage/interaction/capture, source-informed compact whole
learner, actual8-update resource/recovery pilot, packed-only exporter and complete
evolving-state C target using original matrix/LUT/AQ63 kernels. **Failed necessary
gate:**19/32 native logit RMS rows fail fixed1e-4 despite32/32 matching learner IDs,
independently exact-certified. **Missing:** useful compact chatbot quality,
own-history/native chat, accepted50 and large-n capacity. Selected donor is
Falcon-H1-1.5B-Instruct; generality is not demonstrated.

## 1. Bind source, interaction and applicability

Pin revision, complete weight/header/config/tokenizer bytes, family operators,
vocabulary, roles/history, BOS/EOS, source multipliers and precision. Screen
usefulness before major adaptation; exclude consumed cases from final quality.
Tools: [preflight](../../../benchmarks/native_expert_scaling/chatbot_donor_preflight.py),
[census](../../../benchmarks/native_expert_scaling/chatbot_operator_census.py),
[interaction evidence](CHATBOT_INTERACTION_RESULT_20261007.md).
Qwen plain-text canonical input is qualified; native/tools remain open. Giga
template/ID adoption and MLA-cache choices need their own checks.

[Five donors/40 budgets](CHATBOT_ENGINE_TARGET_TRIAGE_RESULT_20261008.md) is
actual metadata/count analysis, not inference. [FalconTiny scan](CHATBOT_FALCON_SOURCE_SCAN_RESULT_20261008.md)
has12 real source-state packets and1% C scan passes, but teacher5/16 usefulness
FAIL; it is not selected for whole adaptation.

[Falcon1.5B](CHATBOT_HYBRID_TEACHER_RESULT_20261008.md): pinned revision
80ebc50d7799a440b96c93bb6686a3924a09b0cb,3,109,773,032B safetensors/
1,554,863,488 BF16 coefficients. Screen14/16 PASS;26.922s family/GPU4.45GB/
OS3.74GB through exit. D2048/L24 parallel attention+SSM, SwiGLU, V65537 untied
head, original BOS/template/bothEOS11/228. Source1.420B products/token is donor
cost, not compact deployment. Qwen/Giga evidence does not transfer automatically.

## 2. Choose geometry; separate identities, approximations and learning

[New controlled group geometry](CHATBOT_HYBRID_GROUP_SUM_RESULT_20261009.md):
all3132 old C operands with INITIAL projected F64 FFN masters. Even per-operand
oracle scaling of the initialized top8 mixture leaves DEVmedian88–94% residual;
all1092 fail1%. This rules out scalar-only reconciliation in this scope,not
better routing/overlap/functions/whole transfer. It does not measure checkpoint286
FFN error or source two-block composition. [Common/private proposal](CHATBOT_HYBRID_SHARED_PRIVATE_NEXT_20261009.md)
is the next construction:2 commonH128 functions plus72 private/top8;extra4.719M
products/2.433MB codes+scales. Unimplemented/unmeasured;fresh capacity/rate open.

[Engine pipeline priority](ENGINE_PIPELINE_PRIORITY_20261009.md) identifies
the next decision-bearing gaps: source group sum versus selected normalized
mixture, redundant atom coverage/coefficients, accessible-state information
loss and full chatbot readout. Close/adopt the current recovery before new
observations. More experts cannot distinguish histories mapped to the same
complete accessible state; no actual collision lower bound is yet measured.
Use staged adaptation and high precision intermediates where necessary, then
validate the final combined original-LUT/ternary/SSM artifact.

[Target contract](chatbot_engine_target_contract_v1.json): compact SSM/SWA +
selected ternary LUT functions. Costed extensions serve bounded active cost.
Flat E scans and simultaneous F32/int8 expert reference copies remain problems.

Actual target D512/L12,10 SSM768/N256/48head*16/conv4,2 SWA4head*128/window128,
E72/k8/h128 signed SwiGLU/AQ63, F32 organs/scales, V65537. Active matrix products
69,632,512;254,932,736 master/control trainable parameters.

| Category | Current operations and limits |
|---|---|
| Algebraic identities |Bake source scalar multipliers into matrices; pair-code decode/int32 LUT products; one exp(delta*A) per shared head. No source width/depth preservation claim |
| Approximations |2048->512 embedding/head projection (~47.1991% equal-trace Gram energy);12/24 core sites;16/64 head channels; SSM/SWA replacing parallel branches; projected norms; paired FFN groups/top8/72; ternary/AQ rounding |
| Learning |All control/bank/norm/scales/router/head parameters trainable. STE is a surrogate gradient, not the derivative of discontinuous rounding |

Source-row slot counts do not prove retained information. Dense group SUM and
normalized selected mixture have different magnitudes; measure initialization
energies/covariance before long fitting. Shared base/overlap/redundancy/expanded
expert variants remain hypotheses with active-byte/product and recovery gates.

## 3. Build and measure the whole learner

[Target implementation](../../../benchmarks/native_expert_scaling/chatbot_hybrid_target.py)/
[actual pilot](CHATBOT_HYBRID_PILOT_RESULT_20261008.md): source-compatible SSD
chunk16/activation checkpointing, F32 organs and ternary/AQ63 signed SwiGLU.
No live teacher fallback. Six NEW calibration cases/32 full-vocab labels,8 AdamW
updates,ALL12 positive finite core/bank/norm gradients,ALL211 parameter tensors
changed;2,039,461,888B actual Adam moments. Independent F64/state audit PASS.
FIT KL27.80248->5.94091/IDs20->10 disagreements;DEV27.61444->26.51797/12->12.
This qualifies numerical feasibility/short FIT recovery, not chatbot capacity.
Actual200.125s family/GPU4.4497GB/OS8.2430GB through exit on local12GB GPU.
Initial/final checkpoints retained; final learner SHA
75b2317efe99fe66fc16f2b0e6df1f5001ef8b9c243c150b87b24e1f433793d9.

Broader [Falcon acquisition/adoption](CHATBOT_HYBRID_TRANSFER_CAPTURE_RESULT_20261009.md)
is COMPLETE:128 FIT/32 DEV/eight authored template domains,4643/1103 full-vocab
labels/753.15MB BF16 packets.137 EOS/23 length96 partials/context512;64 RESERVED
unqueried. ALL saved source-JSON-tokenizer/bit/argmax/EOS/position transport PASS,
376,575,602 coordinates; shared Rust tokenizer, not independent BPE. Source
861.782s/GPU5.715GB/OS3.742GB, adopter8.016s/109.9MB through exit/all caps PASS.
Partial replies are prefix supervision, not full-answer quality. ONE NEW
[balanced whole recovery](CHATBOT_HYBRID_RECOVERY_PROTOCOL_20261009.md) is implemented/
frozen1f2a30a and [TERMINAL/INCOMPLETE](CHATBOT_HYBRID_RECOVERY_RESULT_20261009.md).
Actual prior model+Adam restored;512 fixed case/domain-balanced updates,training-only
AQ normalized dither.025,no changed inference formula,all160 NEW initial/final
observations retained. ALL160 NEW initial complete,FIT/DEV caseKL24.0623/24.7119,
99.806%/99.819% differing IDs. Actual286/512 updates;first and first-longest
183-input backward has all12 finite positive core/bank/norm gradient groups.
First updates10–12s;original deadline reserve stopped at3540.563s after update286.
Actual3588.125s family/OS8.335GB through exit/recovery.pt3.059GB;0 after outputs,
no GPU peak available. Independent saved-only audit PASS/exit0 under85aaaa8/
9c97ad57,63.094s family/OS6.795GB through exit/all resource-input gates PASS.
Actual boundary286/Adamstep294/211 changed finite tensors;all160 before F64
checks/IDs pass,max label reduction delta2.116e-5. Quality admission remainsfalse.
Local3600s/16GiB OS/11GiB allocated GPU/12GiB output;
Full512 budget/final recovery not established; original incomplete gate retained.
Actual completed epochs1/2 online caseKL6.5398/2.3661 and training1336.703/
1328.663s. These are not fixed-checkpoint DEV values; original cap was not changed.
No completed update or initial observation replay. Historical routing support
was not checkpointed and stays missing. Separate final-state evaluation needed.
Broad preservation/own-prefix recovery remain missing. Qwen200-case tensors
belong to another donor. Pilot/screen/calibration stay excluded from fresh tests.

[Separate fixed-state evaluation](CHATBOT_HYBRID_STATE_EVALUATION_RESULT_20261009.md)
is now COMPLETE at actual286:ALL160/5746 NEW rows,original AQ63/no updates.
FITcaseKL1.5135/labeldis33.5344%,DEV3.0810/41.6138%. Relative recovery/support
passes,4 absolute/domain gatesFAIL;not eligible quality/native. F32/F64delta
max5.347e-6,IDs exact.471.969s/OS5.387GB through exit/GPU1.407GB allocated,
all resource/input gatesPASS. Original512 run remains incomplete,no historical
support reconstruction. Defer automatic continuation;new functions/data needed.

## 4. Export and run the actual packed candidate in C

[Driver](../../../benchmarks/native_expert_scaling/chatbot_hybrid_native.py)/
[C target](../../../benchmarks/native_expert_scaling/chatbot_hybrid_native.c)/
[protocol](CHATBOT_HYBRID_NATIVE_PROTOCOL_20261008.md)/
[actual result](CHATBOT_HYBRID_NATIVE_RESULT_20261009.md).

Actual212-field export replaces36 master bank arrays with learner-device-quantized
byte-pair tile-major codes; other F32 fields and64 RoPE frequencies retained.
File425,210,736B, payload425,188,608B, codes84,934,656B,0 inference master/
unpacked expert copies, no P/optimizer. Export167.157s family/GPU37.9MB/OS3.626GB.

Original phase60 matrix/LUT/AQ63 function bodies extracted unchanged and hashed.
Clang O3/AVX2/FMA/scalar contraction off/one thread. Target projection/conv4/
per-head scan/gate-before-RMS/SWA/32-bit IDs are explicit adaptations. Signed
SiLU avoids old ReLU zero-skip assumptions. Stable flat72 top8+selected softmax
mass is charged; structured large-n routing remains missing.

ONE whole6-case/261-input run starts zero per case and evolves continuously;
ALL32 full-vocab learner rows compared. Packed loader/original fixtures/3132
router ID+mass gates PASS; actual model heap425,210,736B/state9,117,696B,
native peak439,676,928B through exit. Family16.953s/native4.750s. Header/build/
queries/logits/trace/final states/commands/PIDs/exits/hashes retained. Three
pre-execution apparatus faults retained/repaired without scientific replay.

**Native numerical FAIL:**13/32 rows<=1e-4 RMS,19 fail,max0.783856%;32/32 IDs
match. [Exact audit](chatbot_hybrid_native_audit_result_20261009.json) verifies212
payload extents/hashes and all2,097,184 coordinate-derived integer energy tests;
5.750s/202.6MB through exit. No tolerance relaxation/admission. Short max57-ID
prefixes do not qualify long drift/window eviction. Head runs32/core261 times:
these component clocks do not establish accepted batch1 rate or DRAM bandwidth.
Standalone C target uses original kernels; native tokenizer/chat integration is open.
The Falcon target is not yet wired into the phase60/engine.c entrypoint. That
integration and persistent incremental chat interface are required deployment
work,not evidence supplied by copying/validating the original matrix bodies.

Prepared [complete cohort interface](CHATBOT_HYBRID_NATIVE_COHORT_PROTOCOL_20261009.md)
and [streamed auditor](CHATBOT_HYBRID_NATIVE_COHORT_AUDIT_PROTOCOL_20261009.md)
extend query/count/streaming mechanics, not kernels. UNEXECUTED;Python syntax PASS,C uncompiled;
160 cases/5746 rows/15999 input IDs imply5,408,639,320B priced total output
including allowance. Only full512-update eligible recovery with independent
audit can bind this path. Partial checkpoint evaluation needs its own frozen
protocol. Algebraic certificate checker is also UNEXECUTED.

## 5. Bound numerical diagnosis, then recover source quality

Current operative next step is the separate common/private variant and broader
supervision under [NEXT](CHATBOT_HYBRID_SHARED_PRIVATE_NEXT_20261009.md).
Complete160-case boundary286 quality and initial-group directional residual
are now measured. Common-bank/core diagnostics below remain closed/reusable;
no new diagnostic is selected solely to perfect old numerical artifacts.

[Common-input banks](chatbot_hybrid_common_bank_result_20261009.json): actual
packed codes/ALL3132 stored C operands, no source/whole replay. All router IDs
equal;3131 FF outputs<=1e-4 RMS,one0.831066% anomaly at layer7/fit_code position33.
Later original code logits pass; it does not explain all19 whole errors.
Family8.937s/OS1.175GB/GPU110.2MB through exit. Quantization-boundary amplification
is a hypothesis, not established causality.

[Common-core result](CHATBOT_HYBRID_COMMON_CORE_RESULT_20261009.md) is COMPLETE:
ALL3132 core/6264 norms CLOSE at1e-4; max2.8729e-6/7.8623e-7.54.735s/OS1.238GB/
GPU157.5MB through exit, all caps PASS/no fault. Stop this diagnostic. There is
no localized material common-core defect; quantization-boundary amplification
under upstream perturbations remains a hypothesis, not a causal whole proof.

Current [NEXT](CHATBOT_HYBRID_BALANCED_RECOVERY_NEXT_20261009.md): reuse adopted
source corpus, implement/freeze a finite balanced whole robust-QAT pilot.
Training-only dithering/margin terms are unimplemented hypotheses; inference
geometry/ternary/AQ63/C kernels stay original. Old nativeFAIL remains, NEW trained
artifact requires a frozen whole native check before long adaptation. No source
replay, broad preservation or native50 admission follows from capture/transport.

## 6. Required fresh quality, physical rate, useful n and variants

1. Freeze broad per-domain held-out recovery/plateau/compute stops before fitting;
   measure final outputs, expert exposure and complete throughput/memory.
2. Verify original-template/BOTH-EOS own-history dialogue, complete generation
   and tasks against donor on fresh excluded cases.
3. Require >=50 accepted batch1 IDs/s on the SAME quality-qualified packed model,
   including full head, routing/mass, cold/first request and contexts.
4. Measure selected bytes/actual DRAM/core-dispatch-LUT overhead; no assumed
   warm cache/locality/ideal bandwidth.
5. Increase distinct useful functions at bounded active geometry; verify utility
   interventions/removal/permutation and fresh quality, not duplicate/synthetic capacity.
6. Demonstrate structured CPU LUT winners AND normalized mass at larger n,
   actual second-family/~10B/~100B variants and resource prerequisites.

These stages are missing. [511](METH_511_WHOLE_HYBRID_HEAD_RESULT_20261007.md)
retains bounded-infilling quality/warm52.56 only; first27.14/all-book failure.
Qwen1280-update fit retains absolute/category failures; order control/source-shaped
Transformer runtime are deferred diagnostics. Local/J/H/affine/quadratic/selector
recipe failures retain their scopes and do not establish a general impossibility.

## Resources and reproducibility

Before substantial work fix uncertainty/reused evidence/decision/budget/stops.
Bind source/checkpoint/code/data/runtime/compiler extents and retain first faults,
commands and result/resource receipts. Hash before/after; large data stay off-repo.
Older bindings need their frozen launcher bytes, not newer schema versions.
No completed source/capture/fit/export/C-prefix/audit/common-bank replay or timing
overlap. Preserve foreign SHA/publisher. T4 permitted in scope, but communicate
actual reason/budget/stops and measure FP16-compatible feasibility before allocation.
No training/T4/current owned worker. INDEX supersedes historical LIVE/NEXTs;
source capture/adopter/core/bank namespaces are complete and reused as bytes.
