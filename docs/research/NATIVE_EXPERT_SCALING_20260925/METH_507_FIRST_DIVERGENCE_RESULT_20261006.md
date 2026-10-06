# 507 result: numerical error starts before the first router

6 October 2026. Goal ACTIVE/INCOMPLETE. Retained-data diagnosis COMPLETE,
both original main and independent audit exit0 ONCE; all6/8 apparatus gates
pass. No new inference, model/compiler/operator call or cohort. This is
consumed-state localization, not new quality, timing, DRAM or transfer evidence.

## What was verified

ALL96 cases/24 books from506. Original F32 generation NPZ and one candidate
generation wire per case:192 files/1,313,899,344B; actual small runtime/source
catalog2508 files/1,436,025,333B, no Torch/PyArrow imports. Complete original
source/official donor bridge and BYTE-qualified I8 candidate provenance retained
with506's known original timing/event/serializer exceptions.

Compare all29 source positions and only identical decoder token histories.
Include each first different choice as the last comparable step. ALL996 aligned
steps, ALL14 encoder and decoder snapshots, actual original expert IDs,
acceptance and selected normalized mass. Every first token rank flip closes
as an EXACT binary rational identity on the stored F32 logits, including
lowest-ID maximum ties. Independent audit rederives every vector/route/pair.

## Observations

| Quantity | Result |
| --- | ---: |
| Identical generated trajectories |65/96 |
| First differing choices |31/96 |
| Comparable decoder steps including first differences |996 |
| Encoder original-ID changes |165/16,704 routes |
| Aligned decoder original-ID changes |115/5,976 routes |
| Changed route acceptance |0 in both streams |
| Divergent cases with NO preceding/current changed expert ID |5/31 |
| Identical trajectories with at least one changed expert ID |62/65 |
| Donor winner-to-candidate-ID margin, min/median/max |.00129175/.05293131/.66001558 |

These counts show that changed discrete expert IDs are neither necessary nor
sufficient for a differing generated trajectory on these cases. This does NOT
exclude routing as a contributor: selected normalized masses differ in every
case, including those five with identical IDs. Do not collapse IDs and mass.

Both encoder and common-history decoder embedding snapshots are exact.
ALL96 encoder streams first differ after block0, a dense block before any
expert router (first sparse block is1). This localizes an initial numerical
discrepancy upstream of discrete expert routing. Recorded snapshots do not
separate attention, dense FFN, projection quantization and arithmetic.

| Encoder snapshot | Median per-vector relative L2, ALL96 | Maximum |
| --- | ---: | ---: |
| Embedding |0 |0 |
| After block0 |.00591803 |.00998140 |
| After block1 |.00828596 |.31393136 |
| After block5 |.01175447 |15.08197012 |
| After block11 |.01224843 |.61474106 |
| Final normalization |.01792699 |1.18191319 |

The extreme15.08197 is book2/case0, after block5/source position24. Relative
L2 divides by the donor state's norm; it is not by itself evidence of numerical
instability or output damage. All per-vector norms, absolute errors and cosines
are retained, not just this maximum. Decoder relative-L2 maximum.69279732 occurs
book9/case1/step0/final normalization; this trajectory remains identical.
At the last aligned final decoder state, median relative L2 is.00714925 over
ALL96, .00739015 over the31 divergent cases and.00690365 over65 identical cases.
Correlation or a large local maximum is not causal attribution.

The five unchanged-ID divergent cases, book/case/first step, are1/2/8,
2/3/5,10/2/6,15/0/3,23/1/4. Their maximum encoder selected-mass absolute
differences are.09408295,.02080733,.01323006,.01104355,.01021981;
decoder differences.01432130,.01026583,.00759143,.01621982,.00917876.
Remaining26 divergent cases have an expert-ID change somewhere in encoder or
aligned decoder history. Full router scores/mass distributions were not captured.

## Decision

Next: [508 source-head fixed-state decomposition](METH_508_SOURCE_HEAD_NEXT_20261006.md),
before another whole run. Apply the SAME original F32 readout to donor and
candidate captured final states on ALL996 aligned positions; check reproduction
of the donor logits/winners first. Count both recovered31 original differences
and newly introduced choices among965 originally matching positions.
Decompose the31 pairwise perturbations into upstream-state, readout and donor
arithmetic terms. This distinguishes a useful precision correction from merely
moving disagreement elsewhere. It does not roll out a repaired generation.

506's failed quality conjunction remains. Exact sparse column function remains
useful; adaptive WI certificates/256 expansion remain deferred. 499's global fit
and500..504 finite support/region failures are unchanged. Compact core, useful
larger n, LUT winner AND mass, actual DRAM and multiple families/scales remain open.

## Provenance, cost and limits

- Source freeze2930f83; binding executionc4678d0; independent audit9b7ae71.
- [Binding](meth507_binding.json), SHA59f6ebee73460b58745a6637499ef1eb3f35ecf150db5c31da2f46bc6c4afc09.
- [Main](meth507_main_result.json), SHA6b21f71416ffda30e9c5b8631aad3408515221b519590e09486a6f92463035ab.
- [Independent audit](meth507_audit_result.json), SHA439facabc4a201978065069f776f5355215330dcc8bde3d586f1f46616cd2476.
- [Protocol](METH_507_RETAINED_DIAGNOSIS_PROTOCOL_20261006.md), all original
  commands and terminal receipts in the completion record.
- Binding20.265s/92,409,856B recorded peak; main13.234s/101,933,056B;
  audit15.843s/107,925,504B. These resource fields are sampled before exclusive
  terminal JSON serialization; final guards also pass180s/2GiB/32MiB but the exact
  final process peak is not persisted. Do not present the pre-serialization peaks
  as complete lifetime maxima. Main RAW15,301,459B; audit RAW31,443B.
- One selected wire per case, no replay of all native repetitions; their prior
  BYTE equality is reused. Original F32 whole-model capture is not recomputed.

No fresh cohort confidence, semantic quality recovery, accepted-ID speed,
compact-core conversion, capacity addition or CPU-LUT scaling is demonstrated.
