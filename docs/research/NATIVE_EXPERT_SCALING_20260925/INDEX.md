# Native expert scaling: research control index

9 October 2026. Branch `research/native-expert-scaling`. Goal ACTIVE/INCOMPLETE.
All research jobs terminal. No source, training, native or T4 job is running.

## Goal and constraints

Reproducible pretrained **CHATBOT -> compact SSM/SWA core + useful selectively
consulted ternary functions -> original engine.c**. Preserve fresh donor-relative
interaction/generation/tasks AND >=50 accepted batch1 IDs/s on the SAME artifact
(100 stretch). Useful larger n/RAM, CPU LUT winners AND normalized mass, actual
DRAM and variants for more families/~10B/~100B remain required. Stored capacity,
active work and retained useful capacity are different quantities.

Ryzen 5 3600X/80GiB/RTX3060 12GB. Freeze code/criteria/inputs/caps before observations;
retain first faults and durable state. No timing overlap. Preserve foreign work
and publisher. Routine Graphify disabled. Donor-adaptation operationally frozen;
its evidence is reusable. Communicate T4 reason/budget/stops before use.
[Engine priority](ENGINE_PIPELINE_PRIORITY_20261009.md),
[target contract](chatbot_engine_target_contract_v1.json),
[method](METHOD.md) and [prior evidence](PRIOR_EVIDENCE.md) govern.

## Position of the two research questions

| Question | Available evidence | Missing |
|---|---|---|
| Useful capacity at bounded active cost | Original trained E32 native quality/parity; packed compact target and actual engine chat | Useful larger n; structured CPU IDs/mass; DRAM; fresh useful quality +50 |
| Pretrained chatbot conversion | Pinned Falcon1.5B; source-informed learner; broad supervision; actual 2-common/6-private state conversion and two whole-model updates | Useful joint function/state transfer; qualified useful C artifact; family/scale variants |

Target D512/L12/10 SSM/2 SWA/E72/k8/H128/V65537 uses original LUT/matrix/AQ63
bodies. Full head costs33.55M of69.63M matrix products/token (48.2%). Original
E32 ~701.7/s is small/cache-resident. Fixed k does not remove flat O(n) router
scores or guarantee capacity/DRAM speed. Source width is offline scaffolding.

## Latest decisive evidence

**[Fixed active-work feasibility](CHATBOT_FIXED_WORK_FEASIBILITY_RESULT_20261009.md):
COMPLETE/PASS for conversion and actual learning.** Parent broad24/Adam318 becomes
two common plus six selected ternary functions per site, without increasing the
current eight-function width. First backward fails in an observation hook; retain
its exact state0 and 89.422s cost. Repair restores weights/moments/RNG, keeps
checkpoint consistency checks, and completes two NEW longest FIT updates.
Private slots318->320/common0->2; all gradients/state finite; actual six-ID/mass
and two-common exposure verified. Long FIT KL5.686->3.949; short2.526->2.583.
This is feasibility, not a construction advantage or fresh chatbot quality.
Combined failed+successful family219.156s<=300s; GPU allocated5.069GB/reserved
5.836GB; held OS4.963GB. Actual final state d6bac2d0 retained. Next is the
[matched 0+8 versus 2+6 continuation](CHATBOT_FIXED_WORK_PAIRED_NEXT_20261009.md).

**[Original-engine reconciliation](CHATBOT_ENGINE_ANCHOR_AUDIT_RESULT_20261009.md):
COMPLETE/static deductions; operational priority corrected.** Current target
uses4x selected-expert products,5.98x core matrix products and128x head products
versus original; sharing LUT bodies does not inherit its cost or recurrence.
Current bank stores one quarter of source FFN coefficient count; this is not a
knowledge fraction. Current C fixes E72/512MiB; n288+ cannot load. Flat router
is O(n), and dense Adam bank storage is160GB/10B or1.6TB/100B. Variable-n packed
storage, learned addressing and bounded optimizer residency remain required.
Source local absolute fidelity is a diagnostic criterion, not a necessary
condition for joint chatbot adaptation. [Construction](CHATBOT_ENGINE_FIXED_WORK_NEXT_20261009.md)
uses2 common+6 selected H128 functions within the current8-function budget.
Actual conversion/learning feasibility is now verified; its C export/dispatch
remains missing. It does not restore the original smaller active geometry.
The source-site0 dose is DEFERRED; no T4 allocation follows from this audit.

**[Broad fixed FFN recovery](CHATBOT_SOURCE_FFN_BROAD_RECOVERY_RESULT_20261009.md): COMPLETE/absolute fidelity FAIL.**
Actual source sites0/23, all48/8808,24FIT/24DEV/12 domains. Same original Student
AST/STE/whole loss/Adam/native arithmetic, fresh selected original sectors,
256 updates/site;512 complete, no fault.44 new initial +4 reused and48 after/site.
13965 training rows/site/all4422 uniqueFIT positions; fixed updates do not mean
matched rows or compute against the earlier two-prefix run.
Site0 DEVwhole .560->.519/centered .558->.521;site23 .488->.196/.556->.373.
Only site23 meets10% improvement; both retain everycase, neither meets absolute
whole<=.10/cosine>=.99/centered<=.10. Site23 worst centeredDEV .970: means hide tails.
Tritchanges2.24%/site0,1.62%/site23; site0 FIT still .475. Extra local dose remains
an optional diagnostic, superseded as the next priority.122.625s/OS1.705GB/
GPUallocated1.142/reserved1.342GB/all input/resourcePASS/exit0/session24779 closed.

**[Broad source operands](CHATBOT_SOURCE_FFN_BROAD_CAPTURE_RESULT_20261009.md): COMPLETE/PASS.**
44 NEW cases/8266 labels +4 old/542 reused. Total BF16 x/y144,310,272B;NEW135,430,144B.
All541.73M NEW +35.52M reused source logit coordinates match bits;577.25M total.
Original411 parameter identities/versions and source file bytes unchanged.
1246.485s/OS3.726GB/GPU5.102/9.181GB/all input/resourcePASS/exit0/session62694 closed.
No source reply generation. These are actual internal operands, not a quality rerun.

[Earlier four-case recovery](CHATBOT_SOURCE_FFN_LOCAL_RESULT_20261009.md):
trit-only error .517-.562/AQ-only .194-.244. Original and corrected diagonal
finite grids fail overall; site23 hidden balance improves23.54%.256/site on
only2 FIT prefixes fails absolute fidelity. [Saved mean control](CHATBOT_SOURCE_FFN_RECOVERY_GEOMETRY_RESULT_20261009.md)
shows91-94% of that model's site23 squared-error gain is mean correction;
its centeredDEV error remains .467/.492. Do not transfer this percentage to
new broader model. Site0 scalar-oracle repair fails. All old artifacts reusable.

Full package all48/8808:DEV KL2.09967/42.9549% differing IDs,screen1/16 versus
source14/16,own-history0/4;7/10 gatesFAIL. [Result](CHATBOT_SOURCE_FFN_CONTROL_RESULT_20261009.md).
F32-only retains all411 objects and exactly all16 source generation sequences/
text;14/16 screen. [Result](CHATBOT_SOURCE_FFN_CAST_RESULT_20261009.md).
Arithmetic recovery diagnoses a recipe; final combined quality governs admission.

## Reusable pipeline and limitations

| Stage | Record and status |
|---|---|
| Applicability/source | [Five donors/40 budgets](CHATBOT_ENGINE_TARGET_TRIAGE_RESULT_20261008.md); [Falcon1.5B](CHATBOT_HYBRID_TEACHER_RESULT_20261008.md), revision80ebc50d/14 of16/1.555B; generality open |
| Recurrence bridge | [FalconTiny](CHATBOT_FALCON_SOURCE_SCAN_RESULT_20261008.md),12 source packets pass1% C scan; teacher5/16 fails usefulness |
| Broad supervision | [Capture/adoption](CHATBOT_BROAD_CAPTURE_RESULT_20261009.md),48/8808 full-vocab rows/577.25M coordinates;24 EOS/24 partials;1091.453s +8.657s |
| Offline storage | [Qualification](CHATBOT_TARGET_SSD_STORAGE_RESULT_20261009.md),same reductions/geometry, loss/local adjoints pass; GPU19.31->5.053GB |
| Compact recovery/coverage | [286 evaluation](CHATBOT_HYBRID_STATE_EVALUATION_RESULT_20261009.md),oldDEV KL3.081/41.61%; [broad24](CHATBOT_BROAD_PILOT_RESULT_20261009.md),newDEV KL6.991/87.48%,oldDEV50.41%;5/8 gatesFAIL |
| Fixed active-work conversion | [Actual result](CHATBOT_FIXED_WORK_FEASIBILITY_RESULT_20261009.md),283 tensors/259.670M masters; exact parent/fault adoption; two NEW updates/private320/common2; FIT-only, no relative construction or quality admission |
| Packed native | [C result](CHATBOT_HYBRID_NATIVE_RESULT_20261009.md),425.21MB/state9.12MB;32/32 learner IDs agree but strict1e-4 RMS FAIL19/32, independently certified |
| Actual engine chat/cost | [Result](CHATBOT_HYBRID_ENGINE_PROBE_RESULT_20261009.md),one C process/five persistent requests/full head; exact split-prefill/own-history reuse;raw46.01-58.84 IDs/s,18.12-40.97 including prefill;old8-update replies degenerate |
| Source FFN transport | [Preflight](CHATBOT_SOURCE_TERNARY_FFN_PREFLIGHT_RESULT_20261009.md),679.477M coefficients/7.390s/340.82MB; six int-dot/F64 witnesses exact |

Longalign remains unqueried/context8186-20874 deferred.104 new+64 old RESERVED
untouched. Other576 adopted prefixes have no new source replies. Broad24/Adam318
candidate16a85448 and recovery286/Adam294/7b95e699 retained; not full512 native
eligibility. No new318 C artifact. No old source/capture/fit/export replay.

Initialized group-sum versus top8:all1092 DEV scalar-oracle residuals fail1%;
not a general impossibility. [Common/private](CHATBOT_HYBRID_SHARED_PRIVATE_NEXT_20261009.md)
code b936b60 frozen/UNEXECUTED/deferred; no automatic admission from mean findings.
Redundancy needs coverage/coefficient control; duplicates alone do not prove
useful n. Qwen/Giga/511/actual256 evidence retains its scope and existing limits.

## Exact point of resumption

**First action:** implement/freeze/bind the
[paired continuation worker](CHATBOT_FIXED_WORK_PAIRED_NEXT_20261009.md), still
UNIMPLEMENTED/UNEXECUTED. Arm A adopts actual broad24/Adam318 (16a85448);
arm B adopts actual fixed-work2/private320/common2 (d6bac2d0). Complete two NEW
A longest FIT updates using B's recorded starting RNG policy; reuse completed
B updates/initial observations. Then one fixed24-case pass per arm, both private
slots344/B common26, with broad/domain/old32 retention criteria frozen beforehand.
This prospective schedule/budget is a proposal until code/protocol/binding exist.
Retain each actual durable boundary; no local FFN prerequisite or automatic T4.
Dose draft is syntax-checked/unbound/DEFERRED; original states256 remain intact.

Completed feasibility namespaces `chatbot_fixed_work_feasibility_20261009`
(first fault/session9819) and `chatbot_fixed_work_feasibility_repair1_20261009`
(freeze1dc25904/bindingcc402a03/resulta15e2ee7/exit0/session11388), both CLOSED.
Exact JSONs/logs/commands/AST/output hashes beside this index; large tensors off-repo.
No owned live job. Useful selection/compact state and joint chatbot recovery,
fresh useful same-artifact original-engine quality+50,useful n/CPU IDs+mass/DRAM,
family/scale variants and full-model/month-plus T4 price remain required.
