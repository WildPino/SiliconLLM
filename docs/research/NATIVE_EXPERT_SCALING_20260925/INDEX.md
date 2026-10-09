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
| Pretrained chatbot conversion | Pinned Falcon1.5B; source-informed learner; broad source supervision/internal operands; source FFN quantization/local learning | Faithful ternary/AQ functions; useful selection/core compression; qualified useful C artifact; family/scale variants |

Target D512/L12/10 SSM/2 SWA/E72/k8/H128/V65537 uses original LUT/matrix/AQ63
bodies. Full head costs33.55M of69.63M matrix products/token (48.2%). Original
E32 ~701.7/s is small/cache-resident. Fixed k does not remove flat O(n) router
scores or guarantee capacity/DRAM speed. Source width is offline scaffolding.

## Latest decisive evidence

**[Broad fixed FFN recovery](CHATBOT_SOURCE_FFN_BROAD_RECOVERY_RESULT_20261009.md): COMPLETE/absolute fidelity FAIL.**
Actual source sites0/23, all48/8808,24FIT/24DEV/12 domains. Same original Student
AST/STE/whole loss/Adam/native arithmetic, fresh selected original sectors,
256 updates/site;512 complete, no fault.44 new initial +4 reused and48 after/site.
13965 training rows/site/all4422 uniqueFIT positions; fixed updates do not mean
matched rows or compute against the earlier two-prefix run.
Site0 DEVwhole .560->.519/centered .558->.521;site23 .488->.196/.556->.373.
Only site23 meets10% improvement; both retain everycase, neither meets absolute
whole<=.10/cosine>=.99/centered<=.10. Site23 worst centeredDEV .970: means hide tails.
Tritchanges2.24%/site0,1.62%/site23; site0 FIT still .475. This motivates isolating
optimization dose before changing loss/representation.122.625s/OS1.705GB/
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
Arithmetic recovery precedes composing selection/compact-core losses.

## Reusable pipeline and limitations

| Stage | Record and status |
|---|---|
| Applicability/source | [Five donors/40 budgets](CHATBOT_ENGINE_TARGET_TRIAGE_RESULT_20261008.md); [Falcon1.5B](CHATBOT_HYBRID_TEACHER_RESULT_20261008.md), revision80ebc50d/14 of16/1.555B; generality open |
| Recurrence bridge | [FalconTiny](CHATBOT_FALCON_SOURCE_SCAN_RESULT_20261008.md),12 source packets pass1% C scan; teacher5/16 fails usefulness |
| Broad supervision | [Capture/adoption](CHATBOT_BROAD_CAPTURE_RESULT_20261009.md),48/8808 full-vocab rows/577.25M coordinates;24 EOS/24 partials;1091.453s +8.657s |
| Offline storage | [Qualification](CHATBOT_TARGET_SSD_STORAGE_RESULT_20261009.md),same reductions/geometry, loss/local adjoints pass; GPU19.31->5.053GB |
| Compact recovery/coverage | [286 evaluation](CHATBOT_HYBRID_STATE_EVALUATION_RESULT_20261009.md),oldDEV KL3.081/41.61%; [broad24](CHATBOT_BROAD_PILOT_RESULT_20261009.md),newDEV KL6.991/87.48%,oldDEV50.41%;5/8 gatesFAIL |
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

**First action:** implement/freeze/bind [additional site0 dose](CHATBOT_SOURCE_FFN_DOSE_NEXT_20261009.md),
currently UNIMPLEMENTED/UNEXECUTED. Restore actual source site0 model/Adam/RNG/
history256 from broad recovery:site00.state256.pt339,929,096B,
SHAa260302562f5ac40fef1d5043989f7e867b4a1026c468cc13b9e90279f0b2590.
Validate six slots/step256/native pairs/scales/original Student AST. Reuse all48
step256 outputs as before. Global steps257..1024 =768 NEW updates, same schedule/
data/loss/precision/active rows; no completed update/initial/source observation replay.
Site23 stays256/SHA b45e8d13. Freeze prospective whole/centered relative+absolute
and individual retention gates. Proposed300s/45reserve envelope is not a binding;
price firsttwo NEW updates, preserve258 and final1024. No T4 allocation.

Completed capture namespace `chatbot_source_ffn_broad_capture_20261009`,
freeze4eb02126/bindingf18c52d1/resultfc992f45/exit0/session62694 closed.
Completed broad recovery `chatbot_source_ffn_broad_recovery_20261009`,
freeze8bb6af56/bindingb9955431/result981679c9/exit0/session24779 closed.
Exact JSONs/logs/commands/AST/output hashes beside this index; large tensors off-repo.
No owned live job. All-site/whole-source recovery, useful selection/compact state,
fresh useful same-artifact original-engine quality+50,useful n/CPU IDs+mass/DRAM,
family/scale variants and full-model/month-plus T4 price remain required.
