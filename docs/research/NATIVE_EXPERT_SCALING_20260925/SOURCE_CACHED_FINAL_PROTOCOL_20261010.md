# Paired cached source final-state control

10 October2026. PROSPECTIVE. One missing source-state observation, no learner
or whole-cohort inference replay. Original ALL48 source alignment FAIL retained.

## Evidence and changed variable

Complete source final readout and independent audit: meanKL around.0005/dis<1%
but3/53 changes in one DEV case fail5% per-case gate. Stored margin diagnosis:
three positive teacher gaps, same contenders in BF16/F64, no +/-1 correction.
The new variable is the pre-final-norm state actually used by cached generation,
paired with that same invocation's normalized feature/logits.

Pinned Transformers5.13.1 source inspection finds different numerical paths:
FalconH1Mixer.torch_forward prefill SSD contractions convert x/B/C toF32;
cached recurrence computes a candidate and Cache.update_recurrent_state copies
it into a BF16 state, inherited from the BF16 convolution cache. It then reads
the state as C's dtype. This is an implementation deduction, not yet an observed
cause of the three changed IDs. Chunk tiling install used by BOTH retained paths
is reused unchanged; it preserves products/reductions in its qualified scope.

For fixed identical inputs the recurrent error satisfies
e_t=A_t e_(t-1)+r_t, with r_t the state-rounding residual. Additional changes
in projections/B/C/dt and attention add terms. Stable A_t=exp(negative rate*dt)
does not guarantee tiny cumulative error when A_t is close to1. Witnessing a
nonzero r_t does not prove it is the sole cause of the whole-model discrepancy.

## Inputs and exact custody

One already identified DEV conversation broad_dev_everyday_conversations_036:
148 prompt IDs,53 cached labels/200 teacher-forced history positions147..199.
Pin source revision80ebc50d7799a440b96c93bb6686a3924a09b0cb, original config/
generation config and inspected source runtime/temporary tiled SSD method.
Reuse qualified source package and prior weights/runtime identities; do not
replace old labels. Bind new script/protocol, source fields/record, retained
prefill raw h24/postnorm/BF16 logits and cached original logits.

Run the SAME deterministic generate settings: BF16/eager, no sampling/cache
enabled/max_new_tokens256/EOS11 or228, six CPU threads, TF32 off/manual seed0.
Instrument read-only hooks at base entry, final norm before/after, LM head count;
retain only the last prompt position then each decode position. Verify consumed
IDs/cache lengths/position IDs for every invocation. Capture actual raw h24,
postnorm and original raw logits. One new generation, normally53 base forwards/
53 source heads; if trajectory differs, preserve actual up-to256 counts/failure.

Read-only wrapper around original Cache.update_recurrent_state calls the
unchanged original and returns its unchanged output. At sites0/11/23 and steps
0/21/23/33/52 save six fixed coordinate witnesses each: input F32 bit pattern,
stored BF16 bit pattern, dtype/shape. No cache mutation or alternative recurrence.
Count every24-per-step update. Observe before/after actual411 source parameter
values/identities/versions; all parameter BF16 bytes must match original safetensor
fields and remain equal. Nonparameter buffers are not covered by that claim.

## Prospective gates and output reconstruction

Save all completed observation before decision. Gates: generated token IDs EXACT
against retained record; cached logits EXACT BF16 bytes against original labels;
actual same-call normalized feature EXACT against actual source norm of captured
state; same-call logits EXACT against source BF16 head/multiplier on captured
normalized feature, one row at a time. Count one new norm call and m new source
head reconstructions explicitly. These gates qualify paired operator custody,
not source full-prefill equivalence, compact quality or a chatbot conversion.

Verdict PAIRED_CACHED_STATE_PASS iff all four exact gates pass; otherwise FAIL.
Keep scientific FAIL even after complete independent adjudication. Never relax
old ALL48 gates or fit DEV. No optimizer/native/RESERVED access/T4.

Independent stored audit checks all input/output hashes, full before/after
parameter hash manifests/source-field identities, all exact gates, norm features
against analytic F64 norm/gamma relativeRMS<=1%, and <=24 scalar head witnesses
math.fsum absolute error<=1.0 (source BF16 rounding allowed; not F64 exact head).
Verify every captured cache coordinate by independent integer BF16 nearest-even
conversion. No full source head contraction or history replay in stored audit.
Compare cached h24/postnorm against retained prefill on contexts with identical
consumed prefixes; report first-position exact equality, changed coordinates,
relativeRMS and the three pairwise logit-margin perturbations. Do not claim
unique causal attribution from these observations.

## Costs, immutable execution and stops

Prior source generation for this case9.187s. Forecast30–90s plus load/hash/IO,
including actual parameter checking and53 head reconstructions; finite held600s,
OS8GiB/GPU allocated10GiB/reserved11GiB/output128MiB/log4MiB. Expected numeric
payload14,328,020B for53 rows; extra coordinate/metadata/parameter records small.
If trajectory changes, up-to256 rows remain below128MiB. Binder preparation
separate; hashes whole source plus411 fields, no new inference in binder.
Separate held stored audit120s/OS512MiB/out256KiB/log4MiB, no GPU/model instance.

One frozen script supplies binder/worker/auditor/held launcher. Freeze binding/
code/protocol before observation, hold exact Win32 processes through exit;
verify resource unions/inputs before and after, retain all first faults. Exact
publisher daemon exempt, foreign work untouched, no owned timing overlap.
Hard stops: resource caps/deadline, nonfinite values, malformed row/position/
cache witness, parameter change, numerical verification or hash mismatch.
No completed source-state capture replay after a fault; new missing-only plan
must identify what remains. Same-invocation paired arrays are durable first.

If PASS, infer observed cached/full-prefill final-state drift only in this case
and define how to capture consistent paired supervision before a compact fit.
If FAIL, distinguish changed token trajectory/custody/norm/head/operator issue
using retained outcomes, then isolate it. Original full objective remains useful
engine.c chatbot ANDsame-artifact50, useful RAM-driven n/CPU ID-mass/DRAM/families.
