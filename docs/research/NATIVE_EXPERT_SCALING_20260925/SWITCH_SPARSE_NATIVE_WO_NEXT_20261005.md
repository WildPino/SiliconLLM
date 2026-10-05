# Proposed NEW453: preserve WO fidelity through exact zero-code sparse execution

**5 October 2026; proposal only. No453 source/protocol/forward/zero-rate/cost yet.**
Retain447/448/451 compact FAIL,452 diagnostic qualified. Full objective unchanged:
useful RAM-scaled conditional capacity, actual CPU LUT/routing/DRAM, whole donor-
relative quality AND>=50 acceptedIDs/s on SAME artifact, families/scales/~100B.

## Why this NEW joint representation/execution variable

451's rotated uncompressed activation control0changes/KL1.21e-8 qualifies;
compact4changes fails.452 proves WO-only compression crosses all4 selected changed
pairs; WI-only crosses2 plus a different global event108. WI-only/full source-WO
rotated hybrid is expensive, exceeds old60% storage cap and is not a candidate.

Next uncertainty: can ORIGINAL-coordinate native I8 WO preserve those decision
margins while its actual zero A16 codes allow exact sparse computation and reduced
coefficient reads? Keep compact signed-basis WI; remove WO basis/compact rounding,
use original native I8 WO and its row scales. This changes an operator AND its
physical layout/cost; the451 source-I32 hybrid is not reused as a deployment model.
A separate new quality/primal/cost screen is mandatory; no inherited451/452 quality.

The60% storage cap remains authoritative for the failed BOTH-I4 pilots. A new
precision/layout method must disclose its different storage before observation;
it can be viable only through MEASURED whole active cost and RAM applicability,
not by relabeling a failed gate or claiming higher precision is sufficient.
Do not run a generic donor port or fit a smaller student instead of the full goal.

## Exact algebra and activation discreteness

For ORIGINAL I8 WO codes q[r,j], positive F32 row scale s[r], original A16 scale
alpha and actual I16 codes c[j], the native represented output is

    out[r]=F32((F64(sum_j I64(q[r,j])*I64(c[j]))*F64(s[r]))*F64(alpha)).

If c[j]==0, its integer product is EXACTLY zero. Sum only j with nonzero c[j],
using increasing original j and overflow-safe integer partials, then SAME original
F64 scales/order/F32 cast. Skipping float-small values or changing A16 rounding
would be approximation; neither is authorized under this exact operator.
Preserve all matrix coefficients and parameter identities. No low-rank/shared
readout, learned selector, altered source softmax mass or skipped head/core.

Transpose q to COLUMN-contiguous [3072,768] I8 storage so each active coefficient
column is768 contiguous bytes (12 cache lines of64 at alignment). All source
coefficients/row scales retained. No cache-residency/DRAM savings assumed before
actual reads/timing/counters; original row-major cache-line incidence differs.

Use I64 reference sums; a proposed C path can accumulate at most512 products
per I32 partial:512*127*32767=2,130,641,408<2^31; qualify this bound in the frozen
protocol BEFORE implementation, then I64 combine, no overflowing I32 full dot.
Keep original WO input neuron coordinates; Hadamard mixing would generally make
the ReLU zero structure dense. Source ReLU implies zeros; the charged zero set
is ACTUAL A16 codes after new WI, never a teacher mask. The same exact zero-code
identity applies to other activations only if their measured A16 sparsity is useful.

## Explicit stored/active accounting, not performance

At D768/M3072, per expert compact WI payload/block scales1,327,104B and native
I8 WO payload/row scales2,362,368B: sum3,689,472B.128-bank stored472,252,416B plus
768 shared WI signs before headers (about78% of source605,945,856B).
This is MORE stored precision than451. Actual bank creation/uniqueness/source
recovery and all metadata must be independently qualified before claiming an export.

If s actual nonzero WO A16 codes at a position, addressed coefficient/scale bytes
alone are1,327,104+768*s+3072. At hypothetical s=1536 they are2,509,824B versus
original4,733,952B/expert; this is algebraic screening, not a measured sparsity,
DRAM descriptor or latency. Charge sign/input basis, A16 zero scan/indices,
accumulators, output writes, full core/router/head/cache/prefill/glue separately.
Stored bytes grow withn; actual selected-function work may shrink with code zeros.
Neither validates RAM-only10x useful-n scaling or arbitrary100B applicability.

## NEW453 staged prospective test to make concrete next

Freeze ONE new math/controller/protocol, fixed shape/layout, all independent
bounds/gates/resource stops before any activation-code counts or new forward.
Same336 consumed prefixes/IDs/input/p and retained compact WI raw activations:

1. Fresh451/452raw/helpers/complete archives/source payload/runtime bindings.
   Replay retained WI states or bind their qualified exact computation explicitly.
2. Decode actual WO input ReLU in ORIGINAL coordinates, original A16 quantizer.
   Independent tiny zero/extrema/signed-I8/partial-bound/scale/cast tests; source
   WO native dense versus full I64 reference versus transposed zero-only skip
   outputs BYTE-EXACT at ALL336 actual activations. No coefficient omissions.
3. New full finalnorm/head predictions after compact WI/native ORIGINAL WO;
   explicit same source-relative mean/book/argmax/identity criteria must freeze,
   including original native-WO versus prior rotated-source diagnostic control.
   No quality inherited from a nearby function or consumed mean-KL statement.
4. Report actual zero-code distributions/per-book/expert, union/cache-line touched
   counts and complete hypothetical addressed descriptor BEFORE expensive C work.
   Do not replace hardware DRAM measurement with a logical byte count.
5. Only if primal/quality AND whole-cost applicability screen pass, NEW actual C
   cost/primal comparison: packed WI direct versus exact pair-activation LUT,
   COLUMN I8 zero-only WO with all scan/build/reduction/glue charged. Then complete
   all-bank export/fresh held-out/generation/tasks/SAME accepted rate/real DRAM.

Fixed source original p/routes are local controls; all-bank WI changes states and
routing and requires complete model validation. The source128/256 context/quality/
rate qualifications do not automatically transfer to the new representation.

## If this physical screen fails

Preserve failure, no sparsity threshold/clipping/precision sweep. Next alternative
is conditional information/readout-aware compact WO correction under full stored/
active budget.452's directional differences identify the constraint, not training
samples. Natural private ID exposure is sparse; justify development-only input/
activation metric and its conditioning before per-expert Hessian/fits. Source
native-primal STE derivatives are approximate424 components, not exact derivatives
through A16/hard selection. New qualified data/calibration/native validation is
required. No learning begins under this proposal.

Reassess the full loop: represent distinct functions, preserve decision information,
select with correct winner AND normalization, execute with measured physical cost,
and prove useful n/family/scale transfer. No453 numerical outcome/implementation yet.
