# Target F32 SSD storage qualification (frozen before observations)

9 October 2026. New storage variable; no geometry, precision, data or optimizer
change. Reuse the actual 58/1507-ID training costs and their first faults. No
unchanged full dense replay, source generation, update, native run or T4.

## Decision and algebra

The original interchunk multiplication has indices
`P[b,i,j,h,d,n]=decay[b,i,j,h]*state[b,i,h,d,n]` and
`new_state[b,j,h,d,n]=sum_i P[b,i,j,h,d,n]`.
Tile j, preserving the entire i reduction, then concatenate j outputs. This
is an identity over real arithmetic; floating point forward/adjoints require
measurement. At R96,H48,d16,N256 the product drops from 7,247,757,312 bytes
to 75,497,472 bytes per tile. Output/state size remains linear in R.
Three other C*B, B*x and C*state contractions tile the existing chunk axis;
their original state/channel/time reductions remain complete. No truncation,
lower precision, changed chunk size, kernel installation or source-file edit.

## Reused state and comparison scope

Load audited checkpoint286/Adam294, all211 tensors, moments2,039,461,888B.
Restore saved CPU/CUDA RNG after construction and before EACH case, as in
the original cost preflight. Original F32/TF32-off, nonreentrant block
checkpointing/preserve_rng_state=True, train-only AQ normalized dither .025,
case-mean full V65537 KL/temp1; zero optimizer steps.

Reobserve shortest58 and longest1507 FIT inputs under this CHANGED storage
schedule. Record losses, all12 explicit core/bank/norm gradient norms, all211
finite gradients, timing/allocator/held OS, and unchanged model/Adam tensors.
Paired recorded loss absolute difference <=1e-5; each gradient GROUP NORM
relative difference <=1e-4. Old full gradient directions were not saved: this
group check does not certify whole-model gradient direction equivalence.

On the initial forward of SSM block0 of the LONG case, save actual input
operands and the actual output adjoint from that case's backward for each of
the FOUR contractions. Replay each isolated original and tiled formula in
F32 on CPU with that same saved adjoint; compare every output and both input
VJP arrays. This exercises real target values/directions at the longest FIT
shape, without full dense model/Adam CUDA residency. Compare CPU tiled vs
saved CUDA tiled outputs as a separate device observation. Fixed limits for
each output/VJP comparison: relative RMS <=1e-5, max absolute difference
divided by max absolute reference <=1e-4; denominator floors1e-30. All arrays
finite; shapes exact. Persist operands/adjoints and comparison metrics.
Scope: four observed block0 contractions, two whole losses/group norms and
gradient connectivity, not all hidden directions/unseen inputs/quality.

## Resources and stops

One family <=1800s with60s reserve; CUDA allocated<=11GiB/reserved<=12GiB.
OS<=32GiB permits the isolated CPU dense reference and its adjoint workspaces
on the available80GiB host; this is a separately priced reference stage,
not a raised cap or passing reinterpretation of the failed16GiB cost family.
Output<=1GiB, log<=4MiB, six worker CPU cores, no overlapping model/native jobs.
Stop/preserve first failure. New successful aggregate requires every numerical,
finite/equality and resource gate; failure does not admit the schedule.

If qualified, use this SAME bound helper in one fixed24-update broader-data
pilot from Adam294. Freeze all48 before/after, old32 DEV retention, order,
criteria and budget before updating. Qualification alone gives no capacity,
chat quality, native numerical parity, accepted50, useful n or DRAM claim.
