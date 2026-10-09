# Next: learn conditional chatbot functions within the engine active budget

9 October 2026. SELECTED construction. Target code implemented; syntax/source
review only. Binding, model execution, training/export unvalidated.
[Reconciliation](CHATBOT_ENGINE_ANCHOR_AUDIT_RESULT_20261009.md) supersedes
source-width local dose as the operational next step. All completed evidence
and actual checkpoints remain reusable. No owned job or T4 allocation is live.

## Decision-bearing question

Can shared and selected functions recover pretrained chatbot behavior while
preserving the existing active expert width1024 per site? The existing sum/top8
initialization loses direction, source local recovery remains poor, and the
earlier common/private proposal adds two active functions to eight. Test the
new allocation inside the eight-function allowance rather than increasing
active work to recover quality. This is a representation/transfer experiment,
not a claim that this allocation suffices for a1.5B,10B or100B donor.

At a target site, compare the current construction

\[
F_A(x)=\sum_{e\in S_8(x)}\alpha_e(x)f_e(x)
\]

with the new candidate

\[
F_B(x)=c_1(x)+c_2(x)+\sum_{e\in S_6(x)}\beta_e(x)r_e(x),
\quad \sum_{e\in S_6}\beta_e=1.
\]

Both use H128 functions, AQ63/ternary forward and original LUT primitives at
deployment. The two common functions are summed with coefficient1; six private
functions have selected-softmax mass. Retain D512/L12/SSM10/SWA2/V65537 for this
controlled comparison. The compact core is a costed extension; its unchanged
geometry in this comparison does not prove that it retained source history.

Per-token expert products remain3*12*512*128*(2+6)=18,874,368. With the existing
private n72 flat router and full head, counted total remains69,632,512.
Two extra stored common functions add4,718,592 masters and18,432 scales, making
259,669,760 total F32 trainable entries if all old arrays are retained. Extra
parameter/gradient/Adam storage is75,792,384B before workspace. Common/private
dispatch, nonlinear cost and memory traffic still require measurement; equal
product counts do not inherit46-59 raw IDs/s or guarantee50 accepted IDs/s.

## First implementation and reuse boundary

1. [Separate versioned2-common/6-private target](../../../benchmarks/native_expert_scaling/chatbot_hybrid_fixed_work_target.py)
   is implemented without changing either old target. Reuse the frozen
   original target arithmetic and the existing2-common/8-private construction
   where applicable. Do not mutate the old target or claim its unexecuted
   zero-output equivalence experiment validates the new selection count.
2. Start from the actual broad24/Adam318 compact checkpoint identified in
   [its result](CHATBOT_BROAD_PILOT_RESULT_20261009.md), preserving original
   tensors/moments/RNG/history and every previous failure. Seed common gate/up
   from declared existing functions with zero down if that initialization is
   selected. Reducing private k8->6 changes the response even with zero common
   output; measure that new initial loss. Existing baseline48 observations can
   be reused. New common gate/up gradients may initially be zero; zero down
   does not justify a false all-parameter-positive-gradient requirement.
3. Bind existing24 FIT/24 DEV source-output packets and old32 DEV retention
   records, exact data order, update count, loss, precision and optimizer policy.
   The48-case broad cohort is consumed development data, not fresh admission
   evidence. Other576 adopted prefixes currently lack teacher reply labels.
4. Before a paired recovery pilot, price actual initialization and two NEW
   update steps on the longest retained FIT trajectory. Preserve actual state
   and count them toward the registered dose. Inherit neither the prior5.053GB
   storage qualification nor its training peak blindly. Preliminary envelope:
   <=300s feasibility family, <=8GiB held OS, <=10/11GiB CUDA allocated/reserved,
   <=4GiB output; these are proposals until code/protocol/binding are frozen.
5. Select a finite matched extra dose from that price, with identical new data
   schedule/loss for baseline and candidate, separate optimizer state and fixed
   final evaluation. Freeze domain/retention and whole-output criteria before
   fitting. No local source-FFN absolute gate blocks this comparison. A loss
   improvement alone does not admit useful generation or capacity preservation.

Do not replay completed updates or source observations. Do not restart the
already failed full-source rounding controls, the finite diagonal grids or
old native numerical diagnostics merely to reconfirm their result.

## From this pilot to the required pipeline

Conditional representation, history/state information and quantization remain
different problems. Joint chatbot supervision can adapt them together, while
staged controls identify a limiting change when progress stalls. Enlarging n
can improve available functions; it cannot distinguish histories collapsed to
the same complete accessible state. Copying atoms or matching parameter counts
does not establish preserved donor ability. Source-informed initialization and
whole donor-relative recovery remain required.

If this construction recovers materially, implement its versioned packed export
and actual engine dispatch before spending weeks on it. Measure fresh own-history
dialogue/tasks and complete rate on the SAME candidate. If it fails, use the
observed whole-output/retention behavior to select core-state transfer or expert
initialization as the next variable; do not automatically increase k/D/L or add
unbounded local fidelity fitting.

Useful-n expansion is part of the converter: variable-n packed fields/native
loader, structured CPU addressing with declared IDs/mass reference, dormant
bank/optimizer storage and independently useful functions. A private E288 bank
matches source FFN coefficient count; E576 doubles it under current geometry.
The two additional common functions must be accounted separately. These are useful
storage envelopes for the human's redundancy idea, not quality equivalences or
ready checkpoints. An E72-only fixed512MiB format cannot implement either.

Long adaptation may be necessary. After a functioning target-shaped learner,
price T4-compatible precision, teacher coverage, resident optimizer blocks,
transfers and quality progress per compute. Communicate reason/GPU-hour budget/
checkpoint/plateau stops before allocation. Months of T4 do not make dense Adam
for a10B/100B bank fit16GiB; long-time and storage feasibility are separate.
Additional donor families and10B/100B cases follow a demonstrated first recipe.

## Exact resumption

Read INDEX, this plan and the actual broad24 result/checkpoint identity:
`results/native_expert_scaling/chatbot_broad_pilot_repair1_20261009/candidate.pt`,
3,059,728,406B/SHA16a85448a706f96719c8d952bbfd5309b8bdf60237f763dfdf101bad1bb9983f.
No new
model result exists in this plan. First implement its initial-loss/update-cost
binding/worker using the separate2+6 target; then freeze and run that feasibility step.
The source-site0 dose draft is DEFERRED and has no binding/result/checkpoint.
Full source chat quality, useful selective capacity, native numerical/behavioral
qualification, accepted>=50, structured large-n routing/DRAM and scale variants
remain open. Goal ACTIVE/INCOMPLETE.
