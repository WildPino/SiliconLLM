# Proposed NEW461: ONE exact shared-A16 fanout, complete original cost contrast

5 October2026. Proposal only; no461 source/protocol/compile/run yet.460 model-free
admission PASS; quality/time benefit still unmeasured. Full goal ACTIVE/INCOMPLETE.

## Uncertainty, variable and decision

Does eliminating repeated quantization/parallel-region setup at36 legal QKV/KV
sites improve whole inference while preserving all original model outputs?
460 counts remove38.7..38.9% dense A16 CALLS/~29.7% parallel regions; integer
products/code/scales/storage unchanged.458 dense share40..44% is not the measured
cost of these removable parts. Quantizer/launch cost and fusion scheduling/stack/
workspace/parallel granularity could dominate any savings. Measure one algorithm.

Keep the compact common-core operation tied to transfer: exact reusable fanout
for already qualified row-I8/A16 artifacts, no generic costly donor runtime.
If not useful, close this execution recipe and return to capacity/transfer work.
No shape/tile/precision sweep or arbitrary retry after seeing timings.

## Planned ONE implementation

Derive from unchanged qualified388 (same374 math, supports3/6workers). Preserve
ALL original math/source except one new primitive and three admitted callsites:
encoder selfQKV, crossKV preparation, decoder selfQKV. Check exact source reversal
after removing the primitive/replacing callsites/names before compile. No engine
edit/new weight layout/export; compile a separate opt-in entry with original
flags/runtime so original binaries remain immutable.

Primitive consumes2/3 Matrix views with common x/cols/tokens, separate outputs:
compute one ORIGINAL A16 code vector/scale per query, original head_integer_dot
per row/query, original F64(rowScale)*activationScale order and finalF32.
One OpenMP static row loop over disjoint logical map/row pairs. Keep integer dot
unchanged (bounds460), no coefficient repacking. tokens1 uses original-sized
stack codes; batch uses one original-sized heap buffer freed after the region.
Require output/input nonaliasing and disjoint output spans, original strides,
dimensions/I8/FE_TONEAREST. Keep original norms/nonlinearities/router mass/head.
Preserve logical calls/code/scales per phase/kind; new wall matrix timer can
include the single combined group, but compare whole profile0 costs primarily.

## Freeze before actual implementation observations

The actual461 protocol/controller/source must prospectively fix:

* Complete fresh payload/manifest/quality/baseline/binary/compiler/libomp/source
  bindings plus460/458 retained evidence; no source values newly fitted/rounded.
* Tiny primitive/native ORIGINAL comparator and independent scalar exact reference:
  maps2/3, token1/batch, below/above parallel threshold, nonmultiple16 cols,
  cols768 and4096, ±I8 extrema including-128, A16 zero/extrema/RNE ties,
  original scale product order. One expected nonalias negative exit2. Synthetic
  arithmetic controls are not expert-capacity/quality evidence.
* Both original source scales, same own cases/source29/cap64/closing32095/worker
  counts/masks/environment. Original qualified binaries versus one candidate
  binary, each warm1/measured3, rotating arm order by fixed case index.
* ALL warm/measured IDs AND full encoder/decoder/logit/route binary SHA exact
  original quality; logical counters/terminal criteria exact. All rejected
  case time charged, ordinary/prose numerators frozen. No fresh-context quality
  or improvement claim from repeating already consumed controls.
* Prospective WHOLE cost gates, suggested for each source: mean candidate/original
  <=.95, every book mean<=1.05, pooled measured whole-latency p95<=originalp95.
  Freeze exact weighting/p95 convention/thresholds in the actual protocol before
  observing candidate costs. Internal usefulness gate separate from final SAME50.
* Explicit compile/control/child/admission/total resource stops, working-set peaks,
  terminal queries, outputs/disk limits, no competing models/GPU, exact publisher
  allowance, copied argv records and immutable first failures. Expected bounded
  CPU minutes/fewGiB, not a new training budget. No repeated completed controller.

Only after ALL apparatus/whole gates pass consider a fresh full quality/rate
verification on the SAME candidate artifact under declared donor-relative tasks.
If gates fail, retain all states/times/tails and close ONE fanout execution recipe;
do not silently change5% into3%, remove a book/outlier or claim impossible fusion.

## Relationship to large useful n

Fusion changes a common-core cost term independent of stored E. It does NOT add
knowledge, make independent donor cores compatible, reduce stored bytes or remove
O(E) router winner/mass cost. A lower common overhead could leave more budget for
conditional capacity, which remains a hypothesis until a composed artifact is
validated. RAM feasibility/usefulness/actualDRAM are separate constraints.

After ONE native cycle, irrespective of result, reassess the transfer path using
the whole evidence: original same-core trained functions, stored/active geometry,
composition error and routing probabilities. Local WI-I4/sparse-WO math remains
available evidence with its closed cost recipes; no all-bank quality is inherited.
More families/~100B and fresh same-artifact50 remain final conditions.
