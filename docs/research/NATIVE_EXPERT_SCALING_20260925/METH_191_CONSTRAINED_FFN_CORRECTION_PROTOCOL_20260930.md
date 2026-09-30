# METH-191: constrained B-only FFN correction, train-only selection

The single METH-189 joint A/B AdamW run worsened viewed-source ranking
by 16.110 points relative to the uncorrected core. This protocol tests
one mechanistic stabilization: keep each seeded rank-64 A fixed,
update only B, and project the composed correction `BA` to a
Frobenius norm no larger than 3% of its stored Q6 base matrix after
every optimizer step. The 3% bound is just above METH-186's observed
maximum 2.838% FFN quantization weight error. It limits update scale
without adding a runtime operation: exported A/B remain ordinary
linear residual factors. Keep the R8 head, stored Q6 core, BF16
attention, controls, centered E1280 bank and routers frozen.

Use the same hash-bound METH-136 raw and long-chat training data and
METH-88 KL+confident-top-1 hinge as METH-189. Use seed 189189 for A
and draw seed 136136. Train 64 raw+chat updates with AdamW B-only,
learning rate 1e-4, zero weight decay and global norm clipping at 1.
This one candidate is specified before running; do not sweep learning
rates or ranks against the viewed METH-121 development set.

Select a candidate with a fixed **training-corpus-only** validation
split: eight raw rows absent from all 256 METH-136 draws and eight
long-chat rows absent from the first 64 update draws, seeded 191191.
Evaluate mean per-sequence METH-88 objective at update 0, 16, 32 and
64. Save the best of 16/32/64 only if it improves the update-0 mean
by at least 10%; otherwise reject without reading development
sources. Repeat validation after BF16 factor serialization; reject
if rounded factors increase the selected validation objective by
more than 1%. The full 72-pair BF16 factor payload remains
53,084,160 bytes and the ideal addressed ledger 534,619,136
bytes/token. This is a cost calculation, not measured bandwidth.

Cap at 15 minutes, 10.5 GiB GPU allocation, 20 GiB RSS, 1 GB new
disk. At update 16 stop if projected full runtime exceeds 15 minutes;
stop on nonfinite loss/gradient, parity or binding failure. Only a
train-only selected and BF16-validated checkpoint may enter the same
viewed-source METH-187 development gates. Fresh source-disjoint
quality and native same-artifact rate are later gates. No T4 is
authorized by this protocol.
