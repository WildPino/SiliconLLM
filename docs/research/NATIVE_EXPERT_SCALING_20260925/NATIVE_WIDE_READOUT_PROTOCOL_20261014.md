# Frozen exact wide-lane readout and paired whole24 qualification

Before timing observations. Reuse canonical all24 p6/I16 packets, qualified
whole24 binding and all97 full-width bridge trajectories. New variable ONLY:
PMADDWD lanes widen after1024 columns, rather than horizontal int32 reduction
every128. Two independent inner accumulators, three wide chunks, one final
int64 reduction. Lane bound128*364*32767=1526680064<2^31; whole dot bound
3072*364*32767=36640321536<2^53. Ordered F64 scale/BF semantics unchanged.
No new rank/precision/learned weight/router/source forward/GPU/RESERVED.

Prepare a separate translation unit by replacing ONLY the readout include
in exact current row_state.c; assert no other text changes, save generated
source and compile with original O3/AVX2/FMA/znver2/ffp-contract=off flags.
Original code and qualified DLL remain untouched. Freeze all new code and
this protocol in Git, then input/runtime/compiler/source/DLL pins before
any controls/timing. All inputs hashed before/after every numerical phase.

64 adversarial dot pairs: first8 full saturation/sign/zero/alternating/lane
patterns, remaining56 seeded RNG0 bounded int16 pairs. Manual Python integer
sum independently checks every3072-term dot. Save inputs and outputs.
All24 q coefficients and continuation r are validated in the admitted domain.

Paired cost: same four histories and final16 positions from qualified
native_24_cost binding, both resident and parameter-evicted. Fresh own contexts
for both DLLs reconstruct every prefix. All prefixes/final snapshot outputs
must equal the qualified traces. At position index t, run original then wide
for even t, reverse for odd t. Calls are sequential sites0..23, batch1.
All24 parameters share stable array owners across arms. Evict every parameter
array before EACH evicted arm; never use the preceding arm's flush. Flush
time excluded and separately stored. Prebuild pointers outside timing.
Measure whole24, per-site and same-input readout-only calls for both arms;
128 same-ABI no-op whole24 controls, no overhead subtraction. Direct readout
controls execute after both whole24 arms, alternating order by t.

Every timed site's complete intermediate arrays and final state/ring/seen/last
must be bit-identical to qualified stored cost records in both arms. Reference
these sealed packets instead of duplicating463MB of identical arrays. The
auditor replays the candidate and independently checks exact readout dots,
scale/BF output, adversarial vectors, summary arithmetic, order and sample
coverage. Shared existing buffer/ABI helpers and recurrence DLL explicit.

Regardless of speed outcome, replay ALL97 qualified bridge trajectories with
the candidate DLL from fresh contexts and actual rows, checking every output
BF word and final state/ring/seen/last against the archived bridge. Auditor
repeats all trajectories. This qualifies stored native compatibility; it does
not rerun source GPU operators under the changed driver or create new held-out
chatbot evidence. Do not stop audits on disappointing speed.

Admission: all exactness/hash/domain checks mandatory. Resident paired ratio
of whole24 medians wide/original <=.95, and every split/length stratum <=1.05;
report evicted ratio separately with identical requirement for that mode.
Both modes must pass to admit general acceleration for this screen. Keep all
tails. Report remaining20ms/10ms budget; crossing20ms alone is not final useful50.
No ideal bandwidth inference. Physical parameter payload unchanged; DRAM
unresolved. Desktop sharing recorded; no concurrent heavy research worker
during timing, no user process termination. CPU affinity[0], single-threaded
kernel, OPENBLAS_NUM_THREADS=1. Per phase<=1800s/OS2GiB; new E family<=32MiB.

If rejected, preserve the exact candidate and result; do not start a minor
layout grid. Reassess active bytes/core/layer/shared or conditional generator
reduction tied to donor-derived parameters and own-model quality. Same final
CPU artifact quality/useful50, useful capacity n, families/scales and conversion
cost remain the original goal. Exclusive phase outputs and PID/create receipts;
closure verifies all successful workers absent and source/artifacts unchanged.
