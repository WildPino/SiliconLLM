# Grouped saved-gradient displacement: frozen original C experiment

10 October 2026. NEW changed models, full goal ACTIVE/INCOMPLETE.
[Motivation/algebra](ORIGINAL_CATEGORICAL_TRUST_STEP_NEXT_20261010.md),
[qualified baseline/backward](ORIGINAL_CATEGORICAL_CAUSAL_PREFLIGHT_RESULT_20261010.md).
This is a test of actual discrete descent and step geometry, not a long campaign.

## Exact intervention and selection

Reuse the immutable actual27 source model (mmap8.652GB), qualified90 saved
gradient tensors (2.817GB), fixed canonical F32 head/norm, old same-input C
scores/routes, source BF16 logits and complete upstream adjudication receipts.
No new source, baseline forward/backward/native prefix, DEV or RESERVED query.
No Adam restore or optimizer execution. Three independently displaced models
from ONE saved gradient; each has one mathematical parameter displacement,
not three sequential accepted training steps. All three masters are retained.

Freeze alpha grid [.0001,.001,.01], ascending evaluation order. For each group G,

    rho_G=max(||P_G||_2,1e-5*sqrt(entries_G))
    Delta_G=-alpha*rho_G*g_G/||g_G||_2, if ||g_G||>0
    Pnew_G=F32(P_G+Delta_G).

Producer evaluates proposal/norms in F64; F32 master rounding is explicit.
Zero-gradient groups copy original F32 bits exactly. Head/norm remain frozen
canonical fields, not old source head/norm. Group definitions: each embedding
row; each bank expert/projection matrix separately; each router-weight row and
each router-bias scalar separately; every other core/norm tensor as one group.
The RMS floor is a declared trust-rule design choice, not an engine constraint.
No row-selection or optimizer state depends on observed candidate quality.

Save all group statistics: parameter norm, gradient norm, radius, actual F32
displacement norm, dot(actual displacement,gradient), moved coefficient count.
Record ideal and actual global directional prediction separately. Negative
STE directional derivative is NOT proof of actual quantized execution descent.

Export all three models through the original wide tensor exporter into the
qualified110-field E4BPv001 ABI. Preserve original header/table bytes. Run one
original C prefix58 per changed model/18 paired full-V source labels. Canonical
query is identical to prior; baseline metrics are read from existing scores.

Choose minimum observed weighted native KL, ties smaller alpha. Qualify local
discrete descent ONLY if all numeric gates for all three candidates pass, best
weighted native KL<=.99*baseline and best first-label KL<=baseline. First label
weight.5, remaining17 each.5/17. Absolute diagnostic quality gates: meanKL<=.05
and source argmax disagreement<=5%. Do not equate local descent with quality.
Frozen grid is a local line-search-like experiment with three parameter
proposals; optimizer_updates=0 specifically counts no iterative optimizer API.

## Complete quantization, routes and mathematical checks

Verify every F32 export field, every paired trit byte and scale from saved
masters, and18432 independent integer witness outputs for each candidate.
No sampled-field substitution. Compare all679477248 ternary symbols and all
row scales to original baseline; retain total changes and certified stable cells.
For actual old/new F32 exporter scales sold/snew, define u=2^-24,

    margin=min(abs(wold-.5*sold),abs(wold+.5*sold))
    B=abs(wnew-wold)+.5*abs(snew-sold)
      +u/(1-u)*max(sold,snew).

If margin>B the symbol is certified unchanged under the unexceptional F32
division rounding model, including a conservative allowance for both endpoints.
All certified cells must indeed retain their symbol. Uncertified does NOT mean
changed. Scale changes can change outputs without a symbol flip. This uses
actual scales, avoiding an unjustified real-mean Lipschitz claim about rounded
Torch reductions. Bound is F64 evaluated, not directed interval arithmetic.

All58x6 routing rows must have distinct valid IDs, nonnegative finite masses,
mass sum defect<=1e-6. Record changed ranked slots, changed sets, maximum mass
delta and expert unions. Mass delta by ranked slot does not match changed expert
identities; report that definition explicitly. No scalable-CPU-route claim.

## Numeric gates and independent stored audit

| Gate | Frozen requirement |
|---|---|
| Per-group actual relative F32 movement | <=alpha+1e-6 |
| Actual dot(displacement, saved gradient) | negative globally |
| Certified stable trits | zero changed among certified |
| Native route mass defect | <=1e-6 |
| Independent longdouble-norm proposal versus saved F32 values | max1 ULP |
| Group statistics | relative1e-9, denominator max(1,abs(reference)) |
| Aggregate proposal statistics | relative1e-9, same denominator |
| Independently shifted logaddexp native metrics | absolute1e-8 |
| All92 masters/all110 export fields/queries/witnesses/receipts | exact |

Audit checks all inputs/outputs, qualified source90 parameter and90 gradient
hash/stats, all three92 master states, every group update with independent
longdouble norm/dot evaluation, full original packing/trit certificate counts,
integer witnesses, actual native command/receipt and query bytes, all54 label
metrics/routes, frozen selection/counters/resource evidence. Deterministic
stored update arithmetic is verification, not another optimizer/model replay.
No model forward/backward/native/GPU call in the audit. Complete even catastrophic
quality or lack of descent; no Fail-Fast policy.

## Budget, provenance and stops

Local CPU6 threads/affinity0..5; CUDA hidden throughout. Producer held1800s,
worker reserve180s, union OS24GiB, output12GiB/log8MiB; native child100s each.
Full stored audit held1800s/OS20GiB/output2MiB/log8MiB, CPU1 thread/affinity1.
Expected output about10.3GB: three2.884GB master snapshots +three520MB packed
models +raw grouped statistics/full native scores/receipts. Complete custody
hashes and full quantizer scans included. Free disk observed190GB before prep.
No T4 or new external resource. Announce any later T4 allocation separately.

Freeze code/protocol/binding/holder before numeric consumption; seal actual
imported files/bytecode/DLLs behind package junctions. Holder seals its own
inputs, recursively retains all candidate output hashes, tracks exact process
creation/command and Win32 OS peaks through exit, allows only direct pinned
original --prefix children (three sequential), preserves foreign publishers.
No speed admission. Stop only actual resource/mechanical faults, kill only owned
worker/native children, retain first_fault/partials/terminal and full charges.
No consumed-code repair or completed history replay; repairs require a new
namespace/binding and a precise missing-evidence justification.

Tiny synthetic structural preflight checks zero-gradient bit preservation,
per-row geometry, independent norms and all9 byte symbols; AST PASS. It does
not consume actual model/gradients or replace the full experiment/audit.
