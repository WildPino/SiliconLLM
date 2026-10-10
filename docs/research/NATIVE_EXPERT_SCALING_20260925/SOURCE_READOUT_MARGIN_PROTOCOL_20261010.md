# Stored margin diagnosis of the source final readout

10 October2026. Prospective diagnostic, after the complete source-readout audit
only. Original frozen alignment FAIL remains unchanged. No compact codec fit.

## Question and decision

The completed source readout differs from cached labels on three of53 positions
in broad_dev_everyday_conversations_036, the sole case failing the5% gate in
both BF16 and F64. Case KL is about.0007. Determine whether those selected IDs
are members of the teacher's exact BF16 top tie or have a positive teacher-logit
gap. This isolates an output-ranking issue; it cannot identify a particular
unobserved recurrent state or justify changing the original thresholds.

Use only this already identified DEV case's two saved full-V score streams,
teacher logits and qualified metadata. This is post-result diagnosis, never
FIT training/calibration or a new quality admission. No source history, head
contraction, generation, optimizer, native call, RESERVED query or T4.

For every53 rows, decode BF16 exactly, keep NumPy's first-index argmax rule,
save top teacher ties and exact scores/gaps for mismatches. Recompute independent
logaddexp KL to <=1e-10 absolute of retained values. Original mismatch counts
must equal3 each. Repeat identity-level comparison across the two readouts.
The verdict is ALL_MISMATCHES_TEACHER_TOP_TIES iff no mismatch in either stream
has a positive teacher gap; otherwise POSITIVE_MARGIN_DRIFT_PRESENT.
Neither verdict upgrades the failed alignment experiment.

For delta=p-q, c=(max(delta)+min(delta))/2 minimizes the constant-shift invariant
Linf error epsilon=(max(delta)-min(delta))/2. A unique teacher argmax with gap
>2epsilon is guaranteed preserved. Save this certificate for every row and
assert no certified mismatch. For the actual selected contender verify exactly
that delta[pred]-delta[teacher]>=q[teacher]-q[pred] up to1e-12. This is an algebraic
consistency witness, not an empirical explanation by itself.

As an off-by-one control compare each prediction to the preceding and following
cached label,52 comparisons per direction/readout. Report meanKL, do not use
these shifted targets for fitting or gate relaxation. No alternative-scale search.

## Frozen resources and custody

One script source_readout_margin.py supplies binder, worker and held launcher.
Freeze script/protocol/binding before observations. Bind qualified source result,
terminal/audit/receipt, exact53-row teacher/BF16/F64 files, Python/helper and
qualified NumPy runtime identities. Check every input before and after; preserve
first faults. OS512MiB/60s/log1MiB/output256KiB; expected seconds plus modestIO.
Hold exact Win32 worker through exit. No concurrent owned benchmark. Audit of
all48 source cases must complete before this begins. It reads about42MiB of
three score streams, plus runtime/provenance bytes specified in its binding.

## Correction options after diagnosis

Exact top-tie flips suggest deterministic ranking/precision differences to
investigate, without dismissing their autoregressive consequences. Positive
gaps require inspecting observed drift rather than explaining everything by
ties. Either way, matching cached hidden-state capture to its labels is the
direct correction if history-format drift is confirmed. A source-history call
requires a new priced protocol; this diagnostic does not execute it.
