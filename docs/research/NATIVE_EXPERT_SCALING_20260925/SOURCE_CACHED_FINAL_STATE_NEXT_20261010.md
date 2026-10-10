# Next: align the observed source final state with its observed labels

10 October2026. PROSPECTIVE INVESTIGATION; no new source call selected yet.
Full original goal remains ACTIVE/INCOMPLETE. No compact codec fit or T4.

## Evidence and uncertainty

The complete source final readout has meanKL around.0005 and below1% mean
disagreement in both splits/readouts, but fails the frozen5% every-case ID gate
on3/53 positions in one DEV conversation. Complete stored audit passed and
the new margin diagnosis found positive gaps, identical BF16/F64 wrong choices
and no simple +/-1 temporal-offset correction. Exact teacher ties are excluded
for these three changes. These results do not identify an unobserved cached
state difference as the cause. [Source result](SOURCE_FINAL_READOUT_RESULT_20261010.md),
[margin result](SOURCE_READOUT_MARGIN_RESULT_20261010.md).

The relevant missing quantity is the actual pre-final-norm hidden state used
by cached generation for each label. Current h24 is from full-prefill with
teacher-forced token history. Algebraic equivalence of these causal operations
in exact arithmetic does not imply equal finite-precision output states.
An output-head precision change on the same current normalized feature does
not fix the three changes.

## Ordered work

1. Inspect pinned source generation settings, output_logits semantics, final
   layer hooks, positions and masks/cache updates versus the retained full-prefill
   capture. Verify actual code/runtime hashes already qualified; do not guess
   from a different installed Transformers version. Reading local source and
   stored records is the next safe action; no inference replay.
2. If this exposes a deterministic mapping/operator defect, document and repair
   that defect in a NEW instrument/namespace. Preserve consumed code and prior
   failed outputs. Freeze the smallest control that tests the changed variable.
3. If the distinction requires a new observation, capture only the actual
   cached pre-final-norm states paired with their SAME-call source logits for
   the already identified conversation. Count the new source history/model/
   generation/head calls honestly. Gate token/label reconstruction against the
   retained original source record, custody of paired state/logits, source norm
   and head reconstruction, and relevant changed-state differences. Do not
   replace old cached labels silently with full-prefill labels.
4. Price source loading/53 steps/hash/GPU/OS/output and stored audit before
   launch, using actual source implementation and prior costs. No training,
   RESERVED access or large family replay. This document is an investigation
   order, not an executed or fully priced capture protocol.

## Decision relevance

A reproducible mapping/precision defect changes how paired supervision must
be captured; a genuine cached/full-prefill difference requires matching each
target state to its execution format. Neither proves thatD256 is adequate or
that a compressed codec is useful. Preserve the original strict source-readout
FAIL. Only after resolving this prerequisite select one FIT-only paired compact
representation/readout in original RMS geometry, with prospectively frozen
unchanged DEV gates. Do not train away a DEV diagnostic by fitting to it.

The three positive-gap changes are far smaller in distribution than actual51's
0/16 tasks or h24P/actual-head KL13.567. They are a different failure to isolate;
their discovery does not explain those larger conversion defects by itself.
The main pipeline still requires useful SSM/SWA histories and sparsely consulted
ternary functions in engine.c, CPU LUT/routing ID-mass, physical DRAM and
same-artifact useful speed, followed by donor family/scale applicability.
