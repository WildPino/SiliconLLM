# Stored route divergence observer: implemented, actual run pending

10 October2026. Resolves the stored-data portion of the
[causal attribution plan](ORIGINAL_CAUSAL_NUMERIC_ATTRIBUTION_NEXT_20261010.md).
No modification to consumed campaign code, criteria or original C engine.

## Purpose and required decision

Localize the first OBSERVED GPU/C route difference in position/layer order;
distinguish ranked-ID permutation from selected-set change; compare mass by
expert ID rather than only by slot. This determines what evidence an eventual
common-operand observation must capture. It cannot recover unstored router
scores, eighth/ninth margins, input states or activation quantizer coordinates.
Earlier unchanged routing does not prove earlier identical hidden states.

Tool: `benchmarks/native_expert_scaling/inspect_categorical_route_divergence.py`.
Standard library only; no Torch/NumPy/model, neural execution, native child,
GPU, optimizer, source, DEV or RESERVED query. Uses the24 FIT baseline histories
already produced by the frozen campaign, not the final model on those inputs.
Each evolving state retains its own ordinal/history identity.

## Custody prerequisites

Preparation requires actual completed campaign producer AND audit terminal
receipts, exit0/errornull/input extents exact, their externally supplied SHA256s
and the actual audit-result SHA256. Parent audit must cover complete24 campaign,
all inputs/outputs, independent integer-byte transitions, all fsum proposals,
all completed gradients/bridges and original packing. Scientific numeric or
quality failure does not prohibit this diagnostic when that custody is valid.

Cross-link audit result to producer result receipt; read the audited original
binding, match its hash to the producer receipt, and retrieve unique sealed
`step_01..step_24/step.json`. Match ordinal/case ID to the independently audited
list. Both original INPUTS and producer OUTPUTS are sealed sources: direction1
reuses qualified traces from its initial state. Shape/dtype, finite nonnegative
mass, ID bounds and distinct selected IDs are checked.

Freeze only these small input extents and observer source into a NEW binding.
Execution requires its exact SHA256, unchanged source/caps, all small input
hashes before/after. Independent standard-library struct decoding localizes
each route row and verifies ranked-slot count and maximum slot-mass error equal
the aggregate already adjudicated for that history. Binding itself is checked
again before writing the result. Full model hashing is not repeated.

## Prospective limits and outputs

Each preparation/execution:60s,OS peak256MiB, cumulative reads128MiB,
individual file<=8MiB, output<=2MiB. Windows peak working-set accounting and
wall-clock/input checks; no simultaneous owned benchmark. Expected small
stored-array workload, not an engine timing admission. Actual byte count/time/
peak recorded on execution. A cap exception remains a fault, not missing data
silently skipped. Exclusive output namespaces; fault capsule emitted on error.

Per-case outputs:

- ranked-slot disagreement count, selected-set mismatch rows and order-only rows;
- earliest ranked-ID/set/order mismatch with actual IDs;
- earliest threshold mass mismatch for identical ordered IDs;
- earliest threshold mass mismatch for common expert IDs, including permutations;
- slot-based and common-ID maximum mass differences and their locations;
- counts of threshold rows with matching ordered IDs/common IDs.

Positions/layers are zero-based. Original routing-mass tolerance is reused as
a diagnostic marker; no new acceptance threshold. Common-ID comparison is
meaningful at changed sets as a partial comparison; it does not measure missing
experts' function values or prove equivalent full mixtures.

## Reproducible synthetic verification

`inspect_categorical_route_divergence_preflight.py` checks AST, exact routes,
order-only permutation, true membership change, same-ID mass change, common-ID
mass hidden under permutation, malformed byte extent, invalid IDs/nonfinite
mass, and failed parent receipt rejection. Executed using isolated Python3.12:

    C:/Users/giosa/AppData/Local/Programs/Python/Python312/python.exe -I -S -B -X utf8 benchmarks/native_expert_scaling/inspect_categorical_route_divergence_preflight.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/categorical_route_divergence_preflight_20261010.json

[Fixture capsule](categorical_route_divergence_preflight_20261010.json): all6
fixture flags PASS. No actual campaign route observation, binding preparation
or Guard execution; synthetic format verification is not actual qualification.
The Windows accounting and receipt/input cross-links await real post-audit use.

## Exact next execution stage

After actual producer closure, run the CURRENT full stored audit under its
already frozen launcher/caps. Preserve all faults; do not substitute this small
observer for any part of that audit. Confirm both actual processes are terminal,
inspect resulting producer/audit receipts and hashes, then prepare the observer
binding using `--prepare`, `--audit`, `--audit-sha`, `--audit-terminal`,
`--audit-terminal-sha`, `--capture-terminal`, `--capture-terminal-sha`, `--out`.
Commit that new binding and exact invocation record before `--execute --binding
... --binding-sha ... --out ...` in a fresh namespace. These hashes are future
observations; no guessed values or speculative actual-result record exists.

If full audit fails mechanically, preserve it and fix its independent
adjudication in a separately frozen namespace first. If stored localization
identifies a route defect, design only the missing common-operand observation
needed to decide its correction. This tool adds observability; it does not
solve useful conversion, structured CPU lookup or same-artifact quality/speed.
