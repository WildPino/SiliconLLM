# Native expert scaling: research control index

9 October 2026. Branch `research/native-expert-scaling`. Goal ACTIVE/INCOMPLETE.
All research jobs are terminal. No model, training, native or T4 job is running.

## Goal and constraints

Reproducible pretrained **CHATBOT -> compact SSM/SWA core + useful selectively
consulted ternary functions -> original engine.c**. Preserve fresh donor-relative
interaction/generation/tasks AND >=50 accepted batch1 IDs/s on the SAME artifact
(100 stretch). Useful larger n/RAM, CPU LUT winners AND normalized mass, actual
DRAM and variants for more families/~10B/~100B remain required. Stored capacity,
active work and retained useful capacity are different quantities.

Ryzen 5 3600X/80GiB/RTX3060 12GB. Freeze code/criteria/inputs/caps before observations;
retain first faults and durable state. No timing overlap. Preserve foreign work
and publisher. Routine Graphify disabled. Donor-adaptation remains operationally
frozen; its evidence is reusable. Communicate T4 reason/budget/stops before use.
[Engine priority](ENGINE_PIPELINE_PRIORITY_20261009.md),
[target contract](chatbot_engine_target_contract_v1.json),
[method](METHOD.md) and [prior evidence](PRIOR_EVIDENCE.md) govern.

## Position of the two research questions

| Question | Available evidence | Missing |
|---|---|---|
| Useful capacity at bounded active cost | Original trained E32 native quality/parity; packed compact target and actual engine chat | Useful larger n; structured CPU selector IDs/mass; DRAM; fresh quality +50 |
| Pretrained chatbot conversion | Pinned Falcon1.5B; source-informed learner; supervision/recovery; source FFN quantization and local learning | Faithful ternary/AQ functions; useful selection/core compression; new qualified C artifact; family/scale variants |

Target D512/L12/10 SSM/2 SWA/E72/k8/H128/V65537 uses original LUT/matrix/AQ63
bodies. Full head costs33.55M of69.63M matrix products/token (48.2%). Original
E32 ~701.7/s is a small cache-resident result. Fixed k does not eliminate a flat
router's O(n) scores or guarantee preserved capacity/DRAM speed at larger n.

## Latest decisive evidence

**[Local source FFN recovery](CHATBOT_SOURCE_FFN_LOCAL_RESULT_20261009.md): COMPLETE/absolute fidelity FAIL.**
Actual source x/y at sites0/23, FIT2/DEV2/542 labels/8.88MB; all35.52M source
logit coordinates match retained BF16 bits. Trit-only function error .517-.562;
AQ-only .194-.244. Diagonal and corrected input grids fail overall; site23 hidden
balance improves23.54%. Same active rows/operators, fixed256 effective updates/site:
DEV site0 .539/.539, site23 .191/.224. Only site23 passes relative improvement;
neither passes absolute error<=.10/cosine>=.99. Combined92.031s/0.894GB GPU.
First logging fault after32/durable2 retained:510 repair executions,512 effective,
542 actual including30 discarded/recomputed updates. No broad quality/native claim.

**[Saved geometry and mean control](CHATBOT_SOURCE_FFN_RECOVERY_GEOMETRY_RESULT_20261009.md): COMPLETE.**
Site0 learned correction DEV cosine~.30; scalar oracle still error>.53.
Uncentered FIT output energy99 rank203/site0 versus29/site23, not knowledge fractions.
New FIT-only constant nearly matches site23 learned error: .199/.236 versus .191/.224.
91-94% of site23 squared-error gain comes from the mean; centered DEV error remains
.467/.492. This changes the decision: broaden function coverage and measure common
AND input-varying response; do not prolong fitting on the same two prefixes.
CPU families22.203/4.985s, no model calls/GPU/native/RESERVED.

Full arithmetic control retains source width/other organs: all48/8808 rows,
DEV KL2.09967/42.9549% differing IDs, known screen1/16 versus source14/16,
own-history0/4;7/10 gatesFAIL. [Result](CHATBOT_SOURCE_FFN_CONTROL_RESULT_20261009.md).
F32-only control retains all411 source objects and exactly all16 source generated
sequences/text;14/16 screen. [Result](CHATBOT_SOURCE_FFN_CAST_RESULT_20261009.md).
Ternary/AQ effects precede compact-core/selection losses. Source-width runtime
is an offline intermediate with full active donor cost, not the final engine.

## Reusable pipeline and limitations

| Stage | Record and status |
|---|---|
| Applicability/source | [Five donors/40 budgets](CHATBOT_ENGINE_TARGET_TRIAGE_RESULT_20261008.md); [Falcon1.5B](CHATBOT_HYBRID_TEACHER_RESULT_20261008.md), revision80ebc50d/14 of16/1.555B; generality open |
| Recurrence bridge | [FalconTiny](CHATBOT_FALCON_SOURCE_SCAN_RESULT_20261008.md),12 source packets pass1% C scan; teacher5/16 fails usefulness |
| Broad supervision | [Capture/adoption](CHATBOT_BROAD_CAPTURE_RESULT_20261009.md),48/8808 full-vocab rows/577.25M verified coordinates;24 EOS/24 partials;1091.453s +8.657s |
| Offline learner storage | [Qualification](CHATBOT_TARGET_SSD_STORAGE_RESULT_20261009.md),same reductions/geometry, loss/local adjoints pass; GPU19.31->5.053GB |
| Compact recovery/coverage | [286 evaluation](CHATBOT_HYBRID_STATE_EVALUATION_RESULT_20261009.md),oldDEV KL3.081/41.61%; [broad24](CHATBOT_BROAD_PILOT_RESULT_20261009.md),newDEV KL6.991/87.48%,oldDEV50.41%;5/8 gatesFAIL |
| Packed native | [C result](CHATBOT_HYBRID_NATIVE_RESULT_20261009.md),425.21MB packed/state9.12MB;32/32 learner IDs agree but strict1e-4 logit RMS FAIL19/32, independently certified |
| Actual engine chat/cost | [Result](CHATBOT_HYBRID_ENGINE_PROBE_RESULT_20261009.md),one C process/five persistent requests/full head; exact split-prefill and own-history reuse; raw46.01-58.84 IDs/s,18.12-40.97 including prefill; degenerate old8-update replies |
| Source FFN transport | [Preflight](CHATBOT_SOURCE_TERNARY_FFN_PREFLIGHT_RESULT_20261009.md),679.477M coefficients/7.390s/340.82MB; six actual integer-dot/F64 witnesses exact |

Source broad cohort:24FIT/24DEV across12 domains. Longalign remains unqueried,
context8186-20874 deferred.104 new+64 old RESERVED remain untouched. Other576
adopted public prefixes have no new source replies. Current broad24/Adam318
candidate16a85448 and recovery286/Adam294/7b95e699 are retained; not full512
native eligibility. No new318 C artifact. Reuse existing observations.

Initialized group-sum versus normalized top8: all1092 DEV scalar-oracle residuals
fail1%; scalar reconciliation is insufficient in that scope, not a general
impossibility. [Common/private](CHATBOT_HYBRID_SHARED_PRIVATE_NEXT_20261009.md)
code b936b60 remains frozen/UNEXECUTED/deferred; no automatic admission from the
new mean finding. Redundancy requires coverage and coefficient control; duplicates
alone do not establish useful n. Qwen/Giga/511/actual256 evidence stays reusable
with its existing limits; none admits this compact chatbot goal.

## Exact point of resumption

**First action:** implement/freeze broader missing FFN operand acquisition per
[recovery next](CHATBOT_SOURCE_FFN_RECOVERY_NEXT_20261009.md). Original48/8808
trajectories/logits exist; reuse current4/542 hidden packets.44 cases/8266 NEW
hidden packets remain. Total two-site BF16 x/y144,310,272B;NEW135,430,144B.
No original reply regeneration. Verify each new source logit against retained bits.
Prospective1800s acquisition cap is UNIMPLEMENTED/UNPRICED/UNBOUND, not a measured
budget. Source parameters/SSD helper and engine kernels remain unchanged.

Then prepare a separately frozen fixed256-step/site coverage comparison across
all12 FIT domains under the SAME ternary/AQ63 forward/active rows. Measure mean,
varying response, per-domain absolute fidelity and retention; DEV never selects.
Loss weighting/common-private changes are later separate variables. No completed
four-case grids/learning/SVD/centering should be replayed as the default next step.

Latest recovery namespace `chatbot_source_ffn_local_recovery_repair1_20261009`,
freeze24c86354/bindingeb53c727, result1bf8b912/exit0/session31303 closed.
Geometry freeze51b1c97/b6290fe6/result7f8f9b06/exit0/session75652 closed;
centering9fce840/c27e6a31/result1e43c2af/exit0. Exact binding/result/terminal JSONs,
logs, commands and output hashes are beside this index. No owned live job.
All-site/whole-source recovery, compact selection/core transfer, fresh useful
same-artifact engine quality+50, useful n/LUT IDs+mass/DRAM/family variants remain
required. Full-source optimizer/month-plus T4 feasibility is not yet priced.
