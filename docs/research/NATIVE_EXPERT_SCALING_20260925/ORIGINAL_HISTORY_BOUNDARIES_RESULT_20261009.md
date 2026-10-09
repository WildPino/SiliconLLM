# Actual teacher residual histories: complete, projection qualified

9 October2026. Goal ACTIVE/INCOMPLETE. All owned jobs terminal; no T4.
This adds real offline supervision for pretrained-to-original-engine conversion.
No candidate optimizer update or native computational operator changed.

## Decision

All48 existing canonical histories/22547 time positions now have raw BF16
source h0,h4,h8,h12,h16,h20,h24 and fixed-P256 F32 projections.
All288 selected source SSM output streams matched inherited full-prefill
operands bytewise. Independent CPU F64 audit checks every157829 projected row:
worst relative RMS8.077043877e-7, below1e-5. All raw/projected extents/finite
and energy gates PASS. These quantities are ready for auxiliary supervision.

Provenance is COMPLETE_WITH_PARENT_RESOURCE_AND_IDENTITY_GAPS: first family
failed its reserved-memory cap and did not reach the final411 parameter
aggregate check. Completion qualifies its own31 jobs/resources/parameter
identity, adopts the old17 complete and2 partial boundary fields, and retains
those parent limits. Do not describe the original family as resource/identity PASS.

The [training-only adapter](../../../benchmarks/native_expert_scaling/original_joint_history_learner.py)
is prepared, not executed/gradient-qualified. Whole native donor preservation
and useful chatbot speed remain missing. Next is a bounded matched whole-model
recovery comparison, including actual native own-history generation.

## Actual source histories and algebra

Pinned useful Falcon-H1-1.5B revision80ebc50d7799a440b96c93bb6686a3924a09b0cb,
411 BF16 parameters1554863488 coefficients, source width2048/24decoder layers.
Same retained48 canonical histories,24FIT/24DEV/12domains/split. Each raw
boundary includes real SSM, parallel attention, nonlinear FFN and residual
composition at all positions. h0 is multiplied embedding, later h4..h24 are
decoder residual outputs. No replacement by a sum of independently evaluated
FFNs. Zero-state full prefill, not cached-reply bit parity. No LM head, new answer
labels/generations/optimizer/native/RESERVED calls.

P is the exact inherited original-transfer2048x256 F32 basis; no refit on FIT
or DEV. P^T P maximum defect7.697597102e-7. z=h.float()@P.float() CUDA F32,
TF32 disabled; CPU F64 summaries after saved precision. Raw payload646467584B,
projected161616896B. Raw complete states stay reusable for other representations.

Equal-case mean retained squared energy, percentages (not knowledge percentages):

| Boundary | FIT total | DEV total | FIT centered | DEV centered |
|---|---:|---:|---:|---:|
| h0 | 27.005 | 26.949 | 26.362 | 26.341 |
| h4 | 11.788 | 11.752 | 11.461 | 11.456 |
| h8 | 12.163 | 12.096 | 11.708 | 11.674 |
| h12 | 12.590 | 12.492 | 11.975 | 11.921 |
| h16 | 13.842 | 13.740 | 12.877 | 12.801 |
| h20 | 14.505 | 14.404 | 12.987 | 12.894 |
| h24 | 36.070 | 35.387 | 29.466 | 28.690 |

For full h4..h20, the fixed input-derived basis retains about12..14% of source
energy; it does not provide exact residual transport. Energy is not importance
for token prediction: no chatbot-quality bound or universal D256 impossibility
follows. Final centered retention below total retention shows why mean and
input-varying components must be reported separately. Projection and RMS do
not commute. Auxiliary targets are residual coordinates, not exact source
logits or compact recurrent-state transport. Whole KL remains necessary.

Independent F64 audit also computes actual discarded residual energy using
||h-zP^T||^2 = ||h||^2 -2<hP,z> + <z(P^TP),z>. Difference from the worker's
orthogonal energy estimate relative to source energy is at most6.128880576e-8.
All336 case-boundary energy summaries match exactly (delta0); domain summaries,
minima and every per-case projection witness retained in
[stored adjudication](original_history_boundaries_stored_adjudication_20261009.json).

## First fault and missing-only depth continuation

First unexecuted freeze92a4023/bind2e53c486 had an optimistic1200s budget.
Before any inference, retained capture logs motivated a1800s cap and corrected
rough15–25minute estimate, freezefe60e9f/bind2d565b4e. First binding remains
historical/unexecuted. Actual inference was faster with much less serialization.

The actual first worker completed17 histories and saved h0/h4 of FITmagpie022
before reserved GPU9919528960B exceeded9GiB. Allocated5983299072B stayed within
8GiB. Guard stopped at h4, after source site3 output witness103 passed.
[Fault](original_history_boundaries_first_fault_20261009.json),
[launcher receipt](original_history_boundaries_result_20261009.launcher_failure.json),
[log](original_history_boundaries_result_20261009.worker.log) and original
worker/bindings/262file namespace preserved. Source final411 identity check missing.

New [completion protocol](ORIGINAL_HISTORY_BOUNDARIES_FINISH_PROTOCOL_20261009.md),
freeze50841c1/bindingab6bd670, adopts all17 case records/raw arrays and partial
h0/h4. For that1507-position case it resumes from the saved BF16 h4/full sequence,
constructs source masks/rotary coordinates, executes source layers4..23 only.
Each remaining layer consumes the previous residual sequence with zero own
cache/history; no embedding/layers0..3 replay. Remaining five source SSM streams
match retained outputs bytewise. Then30 new full history calls. Completed
coverage is47 completed full calls plus the partial prefix followed by a20-layer
tail, not48 newly repeated base-model calls.185 new+103 old witnesses total288.

Release unused CUDA cache between cases; source tensors/operators/precision
unchanged. Completion peak allocated stays5983299072B, reserved drops to
6884950016B and passes9GiB. This observed change is consistent with cached
allocation accumulation; it is not a reduced-source-compute or causal timing claim.
Completion411 parameter identities/versions unchanged; original aggregate gap
remains explicit. Parent failure cannot be retroactively removed by this result.

## Artifacts, execution and resources

[Initial protocol](ORIGINAL_HISTORY_BOUNDARIES_PROTOCOL_20261009.md),
[initial r2 binding](original_history_boundaries_binding_r2_20261009.json),
[finish binding](original_history_boundaries_finish_binding_20261009.json),
[raw finish result](original_history_boundaries_finish_result_20261009.json),
[terminal](original_history_boundaries_finish_result_20261009.terminal.json),
[finish log](original_history_boundaries_finish_result_20261009.worker.log),
[audit code](../../../benchmarks/native_expert_scaling/original_history_boundaries_audit.py).
Finish binding SHAab6bd6704233d9066bc4d17aaae61cd1dd3b9dc18809036effc20949d2b3800f;
raw result SHAe39d38387f3c787475b4ceec806e86ae7b8af010c1b8b02f35a2083871883989.
Raw h/z paths/shapes/extents/SHA and inherited/new scopes are in the result.

First launcher12296/worker25640/create_time1791571763.125661/exit1 held128.703s,
worker123.547s/OS3735998464B/launcher29016064B; reserved gate FAIL as above.
Completion launcher23396/worker19796/create_time1791572193.6524937/exit0 held
143.437s, raw worker131.984s; OSworker3736162304B/launcher29995008B,
conservative sum3766157312B. No descendants/compiler. New time/OS12GiB/GPU8/9GiB/
namespace/log gates PASS. GPU measures allocator, not full device residency.

Old262files249225554B + new462files561428816B =724files810654370B,
combined held272.140s; combined namespace<1GiB/deadline<1800s. All590 inputs/
462 new outputs hashed by stored audit; old namespace included in bound inputs.
CPU-only F64 audit13.468s, no source/CUDA/optimizer/native call; not separately
held family resources or a training benchmark. No owned process remains.

Exact next: qualify prepared joint learner's real full-history gradient path,
implement/freeze finite A:KL versus B:KL+six boundary losses and unchanged native
packed consumer with persistent own-history generation. Same actual27 states/
Adam/RNG,24FIT histories/order/onepass perarm; full native DEV/domain and own-history
criteria before observations. Original LUT/ternary/AQ63/dReLU/SSM/SWA unchanged.
No T4/quality/same-artifact50/useful-large-n/CPU structured mass/DRAM/family
admission. [Operational resumption](ORIGINAL_ENGINE_TRANSFER_NEXT_20261009.md).
