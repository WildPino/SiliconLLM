# Proposed NEW455: decompose the cost before changing physical algebra

5 October 2026. NEW455 source/controller/protocol frozen db7c388; first execution pending.454 ALL8apparatus
PASS but direct/full-pair-LUT BOTH fail every fixed cost gate. Close their recipes;
retain reusable exact C operators/bank and all data. Goal ACTIVE/INCOMPLETE.

## New uncertainty and decision

Why did exact WO sparsity and lower stored coefficient precision produce almost
no direct speedup and a3.024x LUT slowdown? Whole FFN time is measured; component
time is not. Re-running454 without a NEW observable would add no evidence.
NEW455 should independently instrument components, preserving the same qualified
functions and inputs, so a physical-format decision has a measured bottleneck.

See [455 frozen protocol](METH_455_SWITCH_COMPONENT_COST_PROTOCOL_20261005.md).
No first import/compile/profile yet. Reuse complete454
bank/primal/trace/compiler/runtime/source hashes; preserve original C primitives.
No new coefficient roundings/fit/data/source masking or gate relaxation.

Measure ordered components at all336 actual inputs/IDs:

- Original WI quantizer/matrix and original WO quantizer/matrix.
- Candidate signed F64 Walsh/cast and WI A16 quantizer.
- Direct WI decode/block integer reduction/F64 weighted scaling.
- LUT table construction separately from lookups/block reduction/scaling.
- Private ReLU and original WO A16 quantizer.
- Nonzero scan/index construction; column integer accumulation; WO scaling/output.

Require all raw/down/basis/codes/scales still byte-exact454 before profiles count.
Instrumented sums and uninstrumented whole reference must be distinguished;
extra QPC boundaries/writes/checks are charged as profile overhead, not hidden.
Fixed whole-trace order/repetitions and numerical/resource bounds before first run.
Profiles are diagnosis; no new kernel promotion from changed timer boundaries.

## Algebraically different solutions if the diagnosis supports them

1. If repeated WI row/block reduction dominates: EXACT coefficient/scale permutation
   to output tiles, SIMD across output coordinates. F64 block contributions must
   still accumulate in ascending block order for EACH row. This changes physical
   tensor geometry, not the represented function. Tile size/formats must be frozen,
   inverse all coefficients/scales proven and new primal/cost gates required.
2. If sparse WO accumulations dominate: adjacent ACTIVE columns can be paired into
   signed I16 coefficient/activation products with AVX2 pair-add, preserving integer
   sum and512-term bound. Column selection remains exact actual zero codes, no
   omitted nonzeros. A new kernel/layout requires independent primal and cost.
3. If LUT indexed reads dominate: signed-pair symmetry can reduce tables without
   approximation: (a,b) and(-a,-b) share magnitude/table entry and opposite sign.
   For[-7,7]^2,225 pairs reduce to113 representatives including(0,0). That gives
   384*113*4=173,568B tables instead393,216B, plus explicit encoded sign/index costs.
   This is a NEW coefficient-pair format, not proof of faster lookup. All225 states,
   sentinels, signs/overflow/bytes/actual cost must qualify before use.
4. If WI precision overhead consumes WO savings: a NEW native-WI/native-column-WO
   method keeps100% stored I8 coefficients and skips zero WO products exactly.
   Its stored budget differs from453's78%; declare it before observations, retain
   454 failures and justify RAM/applicability/actual active cost. Source matrix
   storage is not enough to conclude efficient useful-n scaling. It cannot silently
   replace the whole goal or inherit another format's cost gate.

Do NOT implement all four as a sweep. Select one only after the diagnosis changes
the decision. Preserve closed398/400 vector-book controllers; different raw-pair
symmetry does not reopen them.

## Whole-project cost constraint

For measured full-model fraction f in the component and component speedup S,
under an additive nonoverlapped cost model:

    S_whole = 1 / ((1-f)+f/S)

This requires MATCHED whole-model context/workers, profiled overhead and actual
component fraction. Existing389 small profile contexts are not391 accepted S29
contexts, and454's one-worker FFN cannot supply that fraction. Reuse qualified
whole profiles only within their scope; if missing, a NEW matched whole cost
record is needed before claiming an improvement sufficient for50 acceptedIDs/s.

Keep dependency order: viable C format -> all-bank composition and new routes ->
fresh donor-relative prediction/generation/tasks + SAME accepted rate -> hardware
DRAM and two real useful-n increments -> actual another-family/~100B applicability.
No new all-bank export or mathematical partial result substitutes for these gates.
