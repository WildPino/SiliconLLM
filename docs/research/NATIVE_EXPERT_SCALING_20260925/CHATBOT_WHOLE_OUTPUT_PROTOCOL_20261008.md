# ONE finite whole-output learner: prospective frozen protocol

8 October 2026. Freeze all code, protocol, exact inputs/runtime before the first
new whole student logits, gradients or updates. Goal ACTIVE/INCOMPLETE.

## Changed uncertainty, reused evidence and decision

Actual ALL24 installation and dense gradient/Adam allocation/FIRST digest/count
audit pass. This stage asks whether a finite joint output objective transfers
the pretrained distribution through the whole changed hidden trajectory.
It reuses qualified balanced keys/source G/U/D priors/core and cached original
teacher distributions. New actual observations: full student forwards/backward/
updates, baseline/final full-vocabulary outputs and real training resources.
No original teacher model forward, old y/J/H/source feature/fit namespace replay
or endpoint answer/query. Original core loads only as conversion prerequisite.

Source Qwen2.5-0.5B-Instruct pinned7ae557604adf67be50417f59c2c2f167def9a775;
896/4864/24/151936/GQA14/2, tied BF16 head and218 frozen core parameters.
24 compact shared512+16private128 full896 SwiGLU fields; query32/selected4/
normalized mass, frozen F32 NEW routing witnesses. Warm0/12 G/U/D priors only;
their old routing buffers must not override new geometry.165150720 F32 trainable
elements/1224 objects; no dense donor FFN fallback/live teacher model.

## Exact finite teacher-forced objective and schedule

Qualified200 cases in immutable manifest order:160 FIT/2437 labels,40 DEV/613.
Each case's source prompt length p, k generated labels1..16. Input sequence
prompt_ids+source_generated_ids[:k-1], decision indices p-1+s, s=0..k-1.
Full student recomputes ALL24 current hidden states and routes; no source x
substitution. Read cached QWGL0001 BF16 full151936 logits/header24B/vector303872B
with exact metadata/frame hashes. No tokenizer or reserved endpoint encoding.

One batched full sequence forward with GPU index tensor `logits_to_keep`
selects k positions before the full tied head. Source capture was incremental
BF16/eager/cache; student uses BF16 source stream/F32 compact fields/BF16 casts,
eager attention/cacheFalse. Different execution order is explicitly allowed;
no bitwise source arithmetic equivalence assumed. Loss KL(p_teacher||q_student)
at T1, computed F32 full-vocabulary log-softmax/exp and mean over k labels, mask
ones; one case per update means equal case weighting in the training schedule.
Evaluation summary is label weighted, reported separately per category.

**Exactly8 epochs/1280 updates**, manifest FIT order repeated; seed261008,
batch1, lr1e-4/betas(.9,.999)/eps1e-8/weight_decay0/foreachFalse/fusedFalse AdamW,
dense resident gradients/two moments/CPU steps from the qualified component.
Never clear gradients to None; zero all resident buffers, backpropagate, clip
global gradient norm to1 with error_if_nonfinite, then one Adam update.
Nonreentrant checkpoints/train mode for fitting, eval/no_grad for baseline/final;
no scheduling/DEV checkpoint selection/restarts/early favorable stopping.

Exactly200 initial eval forwards,1280 training forwards/backwards/updates,200
final eval forwards =1680 full student forwards. Checkpoint recomputation is
part of backward, not another top-level model call. Dense Adam charges all1224
parameters each update, including zero current gradients for unselected leaves;
all step scalars end1280. Source core/router/tied head bytes unchanged at end.
No whole-network exact BF16 quantizer derivative: Torch uses the BF16-cast
autograd surrogate, as previously qualified in its one-block scope.

## Retained observations and FIRST numerical checks

Initial/final full BF16 student logits: typed QWSL0001 header24B and all3050
vectors per phase,926809600B payload plus24B header per wire;
retain case IDs/input sequences/decision indices/metrics/offsets/
SHA/journals. Record current student parent/leaf selection occurrence counts
over ALL input rows per case, not distinct hidden-state or utility certificates.
No extra route arithmetic for observers; observe the route already used.

First update retains before-Adam witness and metadata BEFORE the actual optimizer
call, then full update witness separately: exact teacher/student BF16 logits,
BF16 output gradient, mask,1224 F32 parameter gradient norms, ALL24 nonzero layer
gradient norms, three deterministic coordinates[0,mid,last] of every parameter
before/clipped gradient/after and both moments. Actual loss/norm/counts/resources
and per-update journal retained/fsynced. Zero-first-gradient or nonfinite values
stop. First sample is3672 coordinates, not an all-coordinate Adam certificate.

Save one fixed final24 coefficient files including frozen routing buffers;
no intermediate checkpoint chosen. Total165150720 trained F32 elements; each
file <=28MiB. Saved arrays, source/core and exact final parameter shapes checked.
Loss or resource failure retains RAW/prefix/fault before any numbered repair;
no replay of completed logits/updates to fix administration.

CPU-only FIRST auditor independently reads source metadata/typed wires and
scores ALL400 saved cases in F64. Stable max-subtracted softmax; validate logits
argmax, per-label KL/source-ID log probabilities, summaries, masks/offsets, route
count conservation, update ordering/counts and all final checkpoint finite shapes/
NEW routing bytes. Source teacher winners are already qualified metadata; no
original argmax acquisition audit replay. FIRST logits loss and gradient checked:
gradient=(q-p)/k, absolute envelope2e-6/k+2^-8*abs(reference) for F32 softmax/BF16
rounding. F32 metric gap <=2e-5*max(1,abs(F64 reference)); gradient norm relative
1e-5*max(1,norm). FIRST clip/sample moments/update F64 envelope checks use
32/64F32 eps*abs(expected)+2^-126 (FP32 subnormal allowance), final param4ULP
plus64eps*abs(delta)+2^-126. Verify before samples against qualified input G/U/D.
No independent full network differentiation/routing arithmetic/all-coordinate
Adam computation. Borderline gate disagreement with F64 auditor fails admission,
not silently promoted.

## Prewritten candidate screen, not fresh quality admission

All eight gates must pass on the fixed final checkpoint:

1. FIT label-weighted mean KL <=0.10 nats.
2. FIT source greedy-ID disagreement fraction <=0.15.
3. FIT final mean KL <=0.20*initial mean KL.
4. DEV label-weighted mean KL <=0.10 nats.
5. DEV source greedy-ID disagreement fraction <=0.15.
6. DEV final mean KL <=0.20*initial mean KL.
7. Each DEV category mean KL <=0.20 nats.
8. Each DEV category greedy-ID disagreement fraction <=0.25.

Passing plus FIRST audit admits only a whole-output investigation candidate.
Failing closes this fixed objective/schedule/representation recipe, not all
conditional networks or an information-theoretic impossibility. Old local1%/3%
recipes remain closed with unchanged verdicts; no regrading them by new metrics.
Do not run an unmotivated epoch/rank/alpha/precision ladder after observing DEV.

[Fresh behavioral contract](CHATBOT_FRESH_BEHAVIOR_CONTRACT_20261008.md) and16
new two-turn dialogs are frozen BEFORE learning;64 existing endpoint tasks
remain excluded/unqueried. Numeric-prefix DEV is consumed development, not
source-relative own-history usefulness. Native export/new operator/catalog/
canonical chat/BOTH EOS and fresh quality AND accepted50 SAME artifact follow
only with their own actual evidence. Larger useful n/LUT mass/DRAM/actual family/
scale variants stay mandatory; this E16 dense Adam does not scale training by RAM.

## Prospective bounds/runtime/closure

Local306012GiB, Ryzen53600X/80GiB, Python3.12.10 safe noArrow/datasets view;
Torch2.6.0+cu124/Transformers5.13.1/Tokenizers0.22.2/NumPy2.4.6. Frozen72 safe
runtime roots; CPU auditor29 pureTorch roots. Worker CPUs0..10/launcher11,
Torch6/interop1/OpenBLAS1/OMP6, deterministic/TF32off/CUBLAS4096:8.
Estimated fit tens of minutes, actual unknown until first complete updates.
Worker7200s/family7800s/OS16GiB/GPU allocated8GiB/reserved9GiB/output6GiB/log2MiB;
ordinary case/update30s, FIRST update90s includes coordinate witnesses.
Per first tensor witness32MiB, main2MiB; no new external/T4 resources.
FIRST audit600s/family900s/OS4GiB/output1MiB/log2MiB. Family cap covers input/
runtime/output hashing; actual process exit0/all original gates required.
Held Windows handle samples true through-exit OS peak; runtime/input/foreign
SHAs beforeafter, no conflicting workers/unexpected descendants; exact publisher
exception preserved. Launcher final receipt/stdout tail outside last snapshot.
Typed UTC/FILETIME administrative four-instance closure and positive fault controls.
No active process re-launch on an observation timeout; poll the existing handle.
