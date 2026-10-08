# Finite balanced whole recovery with training-only AQ robustness

9 October2026. PRE-OBSERVATION contract. This is a NEW quality-recovery pilot,
not an original source/capture/learner/native replay. Goal ACTIVE/INCOMPLETE.

## Uncertainty, reused evidence and decision

Can the existing original-LUT/ternary/compact-recurrent target recover broader
source outputs with connected balanced training, while remaining numerically
finite and feasible locally? Does training over neighboring activation codes
provide a candidate for a NEW subsequent native qualification? The second
question is not answered by the GPU training result alone.

Reuse the actual8-update Falcon-informed learner+Adam checkpoint SHA
75b2317efe99fe66fc16f2b0e6df1f5001ef8b9c243c150b87b24e1f433793d9,
the adopted160-case corpus/5746 full-vocabulary labels and all previous failures.
Common C-input core/norm closeness rules out a material discrepancy on those
specific operands. Whole native FAIL19/32 and poor DEV12/12 remain. Quantizer
discontinuity is a motivated hypothesis, not a localized cause of all failures.

A complete eligible pilot proceeds to saved-only metric audit and NEW packed
export/C-prefix comparison, original-template own-history/fresh quality and
accepted>=50 on SAME artifact. A failed or incomplete pilot diagnoses its
specific recipe/resource envelope; it does not disprove conversion. Long T4
work follows measured feasibility and a communicated budget/stopping rule.

## Frozen geometry and arithmetic

D512/L12,10 SSM768/state256/48heads*16,2 SWA(window128),72 ternary
SwiGLU banks/site,k8,h128,V65537.254,932,736 F32 trainable parameters,
211 tensors,69,632,512 active matrix products at decode including flat router.
Original target file remains unchanged. Same embedding/head/norms/core/scales,
stable descending router ties, selected softmax mass, ascending bank accumulation.
No generic original-donor runtime, changed tokenizer or synthetic/duplicate pool.

`chatbot_hybrid_recovery.py` installs an explicit worker-local AQ function.
For training with gradients ONLY, with original detached
`a=max(absmax(x),1e-12)/63`, calculate
`q=clip(round(x*(1/a)+u),-63,63)`, independent uniform normalized-coordinate
`u in [-0.025,0.025]` per call/element. Scale is not perturbed. Existing ternary
weight codes, learned scales, integer-dot forward and master/activation STE
remain. Gate/up/down calls sample independently, including repeated inputs.
This noise is<=1/40 of a code interval and trains nearby decisions; it is not
a deterministic guarantee or an ablation isolating the noise's causal benefit.
No margin penalty is included. At inference/no-grad, use the unchanged original
AQ63 function exactly; no random numbers are drawn. GPU/C equality is still a
separate gate. Training changes weights jointly, not the exported format.

Whole-block nonreentrant activation checkpointing explicitly preserves CPU/CUDA
RNG state. F32,TF32 off,source-compatible eager SSD chunk16. Seed20261009.
No full GPU bitwise determinism claim. Final/recovery checkpoint retains actual
model,moments,step,config,data order,training contract and CPU/CUDA RNG states.
The new RNG starts from a fixed seed; the prior pilot did not save its RNG.

## Data, update rule and finite selection

Adopted corpus:128 FIT/32 DEV,eight domains with16/4 cases each. Teacher prefixes
only.23 length96 partials retained as prefix labels;137 EOS stops do not establish
answer truth. Related authored templates limit generalization.64 RESERVED texts
stay unqueried; all old6 pilot/16 screen cases stay excluded from fresh testing.
No teacher load, generation, download or new source call.

ALL160 initial whole outputs on this NEW corpus saved once before training;
ALL160 final outputs once after512 updates. Full logits little-endian F32 and
per-label KL/argmax observations retained before later admission assertions.
Original BF16 labels lift by u16->u32<<16->F32; saved transport already qualified.
Temperature1,forward KL P_teacher||Q_target,mean across generated positions of
one case per update,including EOS. All parameters optimized with actual prior
AdamW moments/step8. lr5e-5,weight_decay0,default original betas/epsilon,
foreach=False,global grad clipping1.0,positive scale floor1e-8 after every update.
No teacher gradients. No DEV gradient or model selection.

Four complete balanced epochs=512 updates. Binding stores exact order before
values: seeded shuffle within each domain per epoch,16 rounds,one case per
domain in a seeded permuted domain order per round. Every FIT case appears once
per epoch. Equal nominal case contribution/domain exposure; sequential Adam is
not mathematically identical to a single mean gradient across cases. Report
case-weighted AND label-weighted KL and disagreements per split/domain.
Choose fixed final update512, not best DEV epoch. No metric-driven early stop.
Late online plateau (epochs3/4 both>=.99 preceding epoch online mean) is reported
only; these changing-model online metrics are not fixed-checkpoint evaluations.

All211 gradient shapes/presence checked each update; finite total norm required
before clipping. First update,first longest FIT sequence and final update512
also record F64 positive finite gradients for every core/bank/norm layer group.
Actual longest case selected by input length only, before outputs. Complete
post-update CPU snapshots of model,moments,RNG checked finite and retained
each step. This costs about3.06GB transfer per update, approximately1.57TB for
512; its real time/peak memory is charged. Previous boundary retained until a
complete new snapshot succeeds. Only complete admitted update records count.
On failure retain first fault and prior complete recovery boundary. If a family
kill prevents saving the last in-memory boundary, missing state is unrecoverable
and that attempt remains incomplete: do not replay completed observations.

Read-only pre-hooks recompute stable router IDs on the same observed inputs to
count support by phase/domain/site/bank. Collection disabled during backward
recomputation, no duplicate support counts. Diagnostic overhead is included.
Counts certify visits, not useful capacity or structured large-n routing.

## Predetermined exploratory recovery gates

ALL must pass for BALANCED_RECOVERY_ELIGIBLE:

1. Complete512 finite connected updates and ALL160 initial/final records.
2. FIT case-weighted mean KL<=.50 of its NEW initial value.
3. DEV case-weighted mean KL<=.75 of its NEW initial value.
4. DEV case-weighted mean KL<=1.0 and label disagreement<=20%.
5. EVERY domain DEV case-weighted mean KL<=2.0 and label disagreement<=35%.
6. At least8 distinct routed banks per site in each before/training/after phase;
   all counts/unused banks reported. This does not prove a useful72-bank pool.
7. Fixed process/resource gates below, exact preserved inputs through exit.

These are pilot eligibility thresholds, not final chatbot-preservation
criteria. Any failure stays explicit. Saved-only independent F64 metric/state
audit is required before promotion. No retroactive tolerance adjustment.

## Resource budget, launch and retained faults

ONE local family<=3600s,16GiB OS peak,11GiB GPU allocated,12GiB GPU reserved,
12GiB output,4MiB worker log. Worker stops at a guard after3540s to reserve up
to60s for recovery. Launcher retains actual Windows handle through exit and
checks code/checkpoint/packet/foreign hashes before and after successful exit.
No overlapping Python/clang/engine worker except the exact allowed publisher.
No subprocesses permitted. No T4 or environment modification.

Expected persistent extents: initial+final full logits
2*5746*65537*4=3,012,604,816B,one learner checkpoint approximately3.06GB,
and possibly one failed recovery checkpoint approximately3.06GB. Reports/rows
well below remaining12GiB cap. Model+Adam GPU arrays3.059GB before gradients/
activations. CPU old/next snapshots can overlap at~6.12GB. Long sequence eager
SSD/backward peak is unmeasured; actual first short/long update establishes it.
512-step fit within one hour is a hypothesis, not a throughput prediction.

Freeze implementation/protocol/launcher first,then generate binding with exact
hashes/order without target observations. Run using isolated Python3.12.10,
`-I -S -B -X utf8`; shared launcher preloads only safe runtime `site` in the
command and launches this worker. Binding/result/terminal/log supply exact
command,revision/PID/limits/output hashes. Earlier bindings require their frozen
launcher bytes; schema extension does not replace historical run identities.

No automatic scientific repair/restart. A numbered apparatus repair must retain
all faults and reuse completed bytes/state. Useful-n/RAM,structured CPU LUT
winners AND mass,physical DRAM,own-history/fresh quality+rate,other families and
10B/100B remain unqualified. Goal stays ACTIVE/INCOMPLETE.
