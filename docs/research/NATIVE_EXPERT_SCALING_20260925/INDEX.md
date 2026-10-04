# Native expert scaling: research control index

**4 October 2026. Branch:** `research/native-expert-scaling`.
**Goal ACTIVE. No artifact meets the full final goal. No live model/timing job.**

## Goal and constraints

Transfer pretrained capacity to a compact reusable core plus selectively
consulted useful functions in `benchmarks/phase60/engine.c`. Preserve donor
prediction/generation/task quality within frozen measured limits; >=50 accepted
batch1 tokens/s end-to-end on SAME artifact,100 stretch. Demonstrate multiple
families/scales, real~10B/~100B when resources permit. User priority: useful n
grows with RAM; CPU LUT/routing/real DRAM cost and quality remain viable.
Distinct functions, candidate labels, stored/active/read bytes are separate.
Synthetic/copied banks and component rates do not close the goal.

Local Ryzen53600X/80GiB RAM/RTX3060. Freeze criteria/code before observations;
no model job overlaps native timing. Retain first failure before changes.
T4 requires prior reason/budget/stop communication. Routine Graphify disabled.

## Two questions and latest decisive evidence

| Question | Established | Still missing |
| --- | --- | --- |
| Useful conditional target |369 real256 matching has predictive/generative usefulness;370/371 physically smaller64/128 complete artifacts/contracts;373 fourfold actual-bank CPU cost bounded |372 quality is mixed: no monotonic n gain; useful >256, hierarchical routing/LUT quality, physical DRAM, larger RAM/donors |
| Pretrained-to-compact transfer | Complete14.664B original Switch ->14.818GB I8/A16 target in engine C;363 whole donor-relative quality18/18 PASS |364 same-artifact accepted FULL rate FAIL; broader tasks/contexts and multiple-family/~100B proof |

[372 actual64/128/256 quality](METH_372_SWITCH_NESTED_BANK_USEFULNESS_RESULT_20261004.md):
first mixed result retained039dc99,5/7 gates PASS. Relative256,64 masked NLL
+0.837734 and generated-field accuracy-13.28125pp meet both harm bounds.
128 masked NLL improves-0.318440; generated-field accuracy-4.166667pp with
primary upper98.75=0 fails. Original masked teacher agreement38.28%/49.22%
versus full25696.48%. Changed subsets cannot inherit363 acceptance. All96
cases per subset consumed; no primary threshold change or optional repeat.

[373 actual bank CPU cost](METH_373_SWITCH_REAL_BANK_COST_RESULT_20261004.md):
ALL7 PASS retained537f01b,1,060.891s/max1.420GB. Same356/CPU1/affinity[0], ALL96
source29/forced14 at actual64/128/256, balanced order/warm1/rep3 and profile.
Every complete output exact363/372. Full256/64 grows2.91% (upper95 3.62%);
decode0.51% (upper1.27%). Router matrix profile grows3.83x,3.63% of OWN full
profile wall time at256. Aggregate repeats<=1.01363. No accepted-rate or
physical DRAM inference. File sizes3.904/7.542/14.818GB are actual compact banks.

## Reproducible path and closed alternatives

| Step | Evidence / current boundary |
| --- | --- |
|321 applicability +324 official isolated reference | Base128/256 have same123.765M decode matrix coefficients;324 pinned Transformers4.57.6, ordinary project5.13.1 unsupported |
|326/327 full original source |58.860GB acquired; all6392 finite original tensors/ties/12x256 distinct pairs;14.664B unique original parameters |
|328/334 source C contract |Tiny/nine faults and prescribed arithmetic exact; original-backend probability329/330/332 fails retained |
|335/338 compact export/recovery |14.818GB I8 weights/F32 controls exact; interruptions335/337 retained |
|356 all projection inputs A16 |All numerical/primitive/Tiny/fault/cache/full contracts PASS; same compact weights |
|358/359/360/361 threading/cost |CPU6 repeat failures retained; single actual core CPU1/affinity[0] qualified361 |
|362/363/364 whole quality and accepted rate |NEW24 books/96 cases,363 ALL18 quality PASS;36445.1114 ordinary IDs/s including markers, lower42.5362<50 FAIL; prose24.6979/s |
|365/366 exact encoder batches |365 ALL96 bytes exact363;366 repeat1.101982>1.10 FAIL;367 draft ineligible/unexecuted |
|368/369 learned bank identity |Two fixed mismatches independently qualified and both harm prediction/generation; no individual-expert/additional-n proof |
|370/371/372/373 actual n |Physical64/128 exact exports + dynamic contracts; mixed quality; fourfold CPU cost PASS, bounded to these real banks |
| GigaChat314/316/319 |Frozen donor-adaptation assets reused; full-mixture compact bound FAIL, full-width decoder38-56ms vs14/repeats FAIL; generic port paused |
| Dense276-297 |Scoped archive/reference exists; unchanged native297 unsupported41 vsE1280 40 FAIL |
| Granite4HTiny / StdMoE |Source-active-cost or precision gates fail; no unchanged promotion |

Detailed records and links: [history through373](INDEX_HISTORY_THROUGH373_20261004.md),
[prior evidence](PRIOR_EVIDENCE.md), [procedure](METHOD.md).

## Exact artifacts and resumption

Original source: `results/native_expert_scaling/meth326_switch_base256_source`.
Full target: `results/native_expert_scaling/meth335_switch_w8a8_export/weights.bin`,
SHA `e0e5a940b0150b78d0080815a1fddd2a6b50f011351012ed5b4a48a88ba49056`;
spec `results/native_expert_scaling/meth338_switch_tensor_recovery/manifest.bin`.
Qualified binary `results/native_expert_scaling/meth356_switch_all_a16_contract/meth356_switch_all_a16.exe`,
SHA `596410690230be9ec05f5baf93c60d4cab677e3e67225ca94b5f3ed9ea8e628d`.
Actual subsets: `results/native_expert_scaling/meth370_switch_nested_bank_export/n64`
and `n128` (weights/spec/metadata).353/356/363/371/372/373 outputs/raw retained.
All362 sources consumed. Isolated324 environment reusable; source128 full
weights absent. Default engine body byte-exact0ff9705; unrelated edits preserved.

[374 worker contract](METH_374_SWITCH_PHYSICAL_WORKERS_CONTRACT_RESULT_20261004.md)
ALL7 PASS retained0de8cf0,exec1062 exit0 fully consumed;307.844s/max1.404GB.
NEW explicit worker masks0,2,4,6,8,10/actual thread IDs/live readbacks plus
ACTIVE/infinite wait; exact356 math and ALL96 full teacher/own-natural SHA
EXACT363. Negative actual-affinity fault detected. Scoped363 quality only.
[375 cost](METH_375_SWITCH_PHYSICAL_WORKERS_COST_PROTOCOL_20261004.md) prepared
for freeze; SAME374 binary/338 target/new six-worker profile, source9/64
forced32, unchanged20ms/1.10 per-fixture gates. Freeze before observations.
Only PASS licenses separately frozen376 full accepted rate;376 draft remains
unresolved/ineligible until375 qualifies. No live model/timing job.
Whole useful greater n/LUT/realDRAM/multiple families/scales/~100B goal active.
