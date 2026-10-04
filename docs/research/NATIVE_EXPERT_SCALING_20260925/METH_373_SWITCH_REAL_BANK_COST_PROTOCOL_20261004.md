# METH-373: CPU cost of actual nested 64/128/256 learned banks

Prospective protocol, frozen with the controller before timing. Uncertainty:
does a fourfold real candidate pool impose material full or cached decoder
cost at fixed active work, on the available CPU? Reuse exact356 executable,
338 full256 and370 physically compact64/128 artifacts,371 independent dynamic
contracts,363 full256 quality and372 complete subset outputs.372 first mixed
quality result FAIL retained039dc99:64 loses both primary outcomes;128 improves
masked NLL, with generated field harm not established at primary confidence.
No monotonic quality claim or changed subset promotion follows.

## Fixed observations and identities

ALL24 consumed362 books/96 cases. Exactly source29 and teacher decoder14 for
EACH n, same source and forced prefix IDs, no own-natural length differences.
Actual retained firstn original WI/WO pairs and F32 classifier rows in each
of12 banks, same common core/head/precision/capacity/top1. All258 routes must
be accepted (source29 < capacity64; each cached decoder call has one token).
Routes may differ naturally with n. All complete warmup/measured/profile
output SHA must equal prior full teacher363 (256) or372 (64/128), ALL96 each.

Fresh physical HEAD bindings for controller/dependencies/engine/protocol/raw;
fresh whole payload/spec/metadata hashes for all three artifacts, independent
manifest parser checks all names/offsets/scales/shapes/n/EOF. Same356 executable,
compiler/libomp hashes, fresh359 topology exact371. No concurrent model/native
timing/training/GPU/download. No source payload/runtime/engine edits/recompile.
CPU1, EVERY child affinity [0] read back; OMP_WAIT_POLICY=PASSIVE,
KMP_AFFINITY=none, OMP_PROC_BIND absent. Prespecified source/case/n order:
for book0..23/case0..3, cyclic order (64,128,256) shifted by (4*book+case)%3;
each of the three orders occurs32 times. One full warmup then three measured
repetitions/profile0 per child. After ALL primary cases, repeat same balanced
order with one warmup/one measured repetition/profile1. Every row and complete
output retained. No favorable subset/case/repetition selection.

## Measurement and decision

PRIMARY profile0 complete encoder + crossKV + cached forced decode wall time;
decode time separately. Load/startup, tokenization, serialization and teardown
excluded and load separately recorded. Per-case median of three measured
rows, then sum within each book (four cases). Primary n256 / n64 ratio of sums
and one-sided upper95 book-bootstrap10000draws/seed373373, for full AND decode.
Both upper bounds <=1.20 required: a prespecified <=20% cost increase for a
fourfold candidate bank at this scope. This is a CPU scaling gate, NOT the
accepted-generation >=50 gate. Further descriptive128/64 and phase ratios
use identical bootstrap. No causal cache/bandwidth attribution from a ratio.

For each n, full AND decode aggregate repeat ratios max/min <=1.10, summing
all96 rows separately by repetition0/1/2. Per-case timings remain visible;
aggregate repeat gate intentionally follows364 rather than claiming every
case repeats within10%. Profiles are secondary: matrix time for encoder/
decoder core, expert, router, head, with clock overhead; outputs exact primary.
Report phase wall time and matrix attribution separately.

Decode I8 matrix code123,764,736 and row scale534,016 logical bytes per position
are invariant. F32 router18,432*n bytes/position and6 classification calls;
expert6 pairs/position. Encoder similarly checks29 positions. Every route's
actual retained original function ID follows the qualified manifest. Record
accepted expert unions per bank/case and across cohort, unique consulted
expert code/scales, whole mapped payload file bytes and sampled child RSS.
These are logical address / observed consultation / sampled process memory
quantities; they are NOT physical DRAM traffic, exact weight residency,
cache-hit counters, or evidence that the whole mapped file is resident.
The unchanged runtime uses direct lookup, not a new hierarchical/ternary LUT.

Seven gates:288 primary+288 profile cases complete; all full output bridges;
all affinity readbacks; all dynamic lookup/routes/logical descriptors; all n
full/decode aggregate repeat ratios; full256/64 upper<=1.20; decode256/64
upper<=1.20. PASS supports bounded actual fourfold-bank CPU cost here. Failure
preserved before changed experiments; no optional repeat or relaxed limits.
High full/decode growth motivates routing/memory redesign before extension;
profile attribution directs the change without asserting DRAM causality.

MAIN30min/checked combined RSS16GiB after imports. Stop and retain first raw
failure/partials; no overwrite. Expected <=15min, <4GiB RSS, outputs/logs
approximately1GiB. Final useful RAM-scale n/LUT/real DRAM/quality/generalization,
real~100B/multiple families and >=50 accepted SAMEartifact goal remain active.
No reuse of373 timing as a rerun or promotion of failed364 or366 profiles.

```powershell
.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth373_switch_real_bank_cost.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth373_switch_real_bank_cost_result.json
```
