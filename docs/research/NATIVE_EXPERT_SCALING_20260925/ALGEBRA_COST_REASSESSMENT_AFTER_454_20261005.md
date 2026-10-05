# Reassessment after454: capacity algebra holds, kernel cost does not improve

5 October 2026. Derived analysis of authoritative retained results; no new
activation/fit/benchmark. Goal ACTIVE/INCOMPLETE. [454 result](METH_454_SWITCH_SPARSE_WO_COST_RESULT_20261005.md)
qualifies actual C bytes/primal but closes direct and full-pair-LUT cost recipes.

## Capacity and RAM: exact actual-source accounting

The two real same-geometry source artifacts have D768, M3072, twelve sparse banks.
Their unique trained counts from389 match391's bound source record. Exact formulas
fit BOTH original source points, including one D-wide router row per expert:

    P(E) = 166,280,448 + 12*E*(2*768*3072 + 768)
    B_original(E) = 265,878,016 + 12*E*(4,733,952 + 3072)

| Actual source | Unique trained parameters | Serialized whole payload bytes |
| --- | ---: | ---: |
| E128 |7,415,217,408|7,541,946,880|
| E256 |14,664,154,368|14,818,015,744|

Fixed trained parameters and fixed physical bytes differ: source embedding is
F32 with one physical shared copy, a separate I8 tied-head view is also stored,
other precisions/scales differ. Router occupies4*D bytes per row, not the trained
parameter count D. Weight counts alone cannot substitute for physical RAM.
[Exact derivation](ALGEBRA_454_STORAGE_DERIVATION_20261005.json) retains the input
record/hash, formulas and scalar calculations; no model forward performed.

At the same D/M/L and fixed portion, nominal100B/10B expert counts are in a ratio
about10.15, consistent with the user's approximately10x intuition. These labels
do not identify an actual trained checkpoint or integer E. Original source256
uses independently learned core/function values: equal geometry does not license
appending its experts to source128's core. Existing cross-core failures remain.

For453's hypothetical all-bank representation, coefficient/scale/sign bytes would
follow265,878,016+12*E*(3,689,472+3072)+12*768 before new headers/context/workspaces.
This formula describes storage, not a new export;454 kernels failed actual cost.
Payload-only capacity against ALL80GiB is an upper bound. OS, other workloads,
context buffers and runtime requirements reduce it; extra stored slots do not
create pretrained functions or establish useful capacity.

## Selection is the E-dependent active term

Original flat router score work grows as E*D per bank and mass sum as E.
Router coefficient bytes grow as4*E*D, even with top1 expert consultation held fixed.
As expert-side accessed bytes shrink, routing's relative cost can become larger.
Do not infer a constant active cost from the fixed selected-expert count.
393's softmax-tail evidence still requires winner AND mass to be controlled;
certified refinement/fallback and hardware traffic must be charged at larger E.

No actual useful larger-E source has been produced here. n scaling needs trained
distinct functions, two real increments, natural held-out/causal benefits, complete
quality, CPU selection/LUT scaling and physical DRAM. Duplicated banks, parameter
labels and RAM formulas cannot satisfy those requirements.

## Arithmetic cost: information conservation is not instruction elimination

453 proves exact omission of WO zero integer products.454 carries that identity
into native C at1344 candidate activations, plus336 original records. Yet direct
mean cost234.43us versus237.68us and p95 worsens; LUT718.82us. Both fixed recipes
fail all cost gates. These are selected-FFN costs at one worker/one bank/consumed
prefixes; they do not determine full-model accepted rate or actual DRAM traffic.

WI has the same2,359,296 scalar products. Block64 precision adds per-block decoding,
integer horizontal reduction and weighted F64 scaling. LUT replaces products with
builds/indirection/reductions, which are themselves work. Aggregate results cannot
locate the sole bottleneck; source-level operation counts are not measured cycles.

NEW455 component profiles should separate original matrices/quantizers, basis,
direct reduction/scales, LUT build/lookups, ReLU/WO quantization, zero scan and
column accumulation/output. Exact source recovery/primal must precede any profile.
Select a single different physical algebra only when its measured component is
the relevant constraint; [next diagnosis](SWITCH_NATIVE_COMPONENT_COST_NEXT_20261005.md)
lists exact output tiling, active-column pairing and signed-pair symmetry as
conditional proposals, not a kernel sweep or already validated solutions.

## Whole cost and composition remain separate requirements

For a measured additive full-cost fraction f and speedup S of that component,
S_whole=1/((1-f)+f/S).389's retained small profiles have source9/decoder4;
391 accepted runs use source29 and naturally variable generation.454 one worker
also differs from391 three-worker context. No matched full-cost fraction has
been established by454. A source-specific full profile is needed before claiming
a local improvement can close the50 acceptedIDs/s constraint.

Continue the entire transfer path: viable representation/C cost, all-bank
composition/changed routes, fresh donor-relative prediction/generation/tasks and
SAME artifact whole rate, physical DRAM/useful-n selection, actual families/~100B.
Qualified byte-exact components and measured failures change the next action;
they do not reduce the intended end state.
