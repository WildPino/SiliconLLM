# Native expert scaling: research control index

**4 October 2026. Branch:** `research/native-expert-scaling`.
**Final goal incomplete. Two independent same-family source scales have bounded
whole-quality/SAMEartifact FULL-rate qualification. Next: real router tail mass.**
Goal tool currently reports paused despite user-authorized continuation;
no complete/blocked status was set. Research remains unfinished.

## Goal and constraints

Transfer pretrained capacity to a compact reusable core plus selectively
consulted useful functions in `benchmarks/phase60/engine.c`. Preserve donor
prediction/generation/task quality within frozen bounds; >=50 accepted batch1
tokens/s on SAMEartifact,100 stretch. Demonstrate multiple families/scales and
real~10B/~100B when resources permit. User priority: useful n grows with RAM;
CPU LUT/routing/real DRAM and quality remain viable as choices grow. Distinct
functions/candidate labels/stored/active/read bytes must stay separate.

Ryzen53600X/80GiB RAM/RTX3060. Freeze before observations, no model job overlaps
native timing, retain first failures. T4: prior reason/budget/stop communication.
Routine Graphify disabled. Preserve unrelated existing tracked/untracked work.

## Two questions and latest decisive evidence

| Question | Established | Still missing |
| --- | --- | --- |
| Useful conditional target |369 real bank matching useful;370/371 real64/128 subset exports/contracts;373 fourfold bank CPU cost bounded |372 quality mixed, no monotonic n gain; useful>256/10x, hierarchical LUT, physical DRAM |
| Pretrained-to-compact transfer |Independent original Switch7.415B/14.664B ->7.542/14.818GB complete I8/A16 C targets;387/363 whole original-relative quality18/18;391/376 SAMEartifact FULL rate PASS | Another family, broader contexts/tasks, useful larger choices and~100B |

| Qualified original source | Actual worker profile | Accepted ordinary IDs/s (lower95) | Prose IDs/s (lower95) |
| --- | --- | ---: | ---: |
|128 /7.415B |3 physical[0,2,4] |96.56385 (94.58827) |55.97659 (54.59729) |
|256 /14.664B |6 physical[0,2,4,6,8,10] |63.52543 (59.51178) |34.77928 (32.64971) |

[391 result](METH_391_SWITCH_BASE128_ACCEPTED_RATE_RESULT_20261004.md), retained
4764544: ALL5 PASS,166.313s/max1.335GB, all384 warm/measured natural hashes AND
IDs exact387,96/96 healthy/1142 ordinary IDs INCLUDING480 sentinels/662 prose.
ALLcase median FULL11.826372s, repeat1.012403. [376 result](METH_376_SWITCH_PHYSICAL_WORKERS_ACCEPTED_RATE_RESULT_20261004.md),
56922f6: ALL5 PASS,81/96 healthy/895 IDs INCLUDING405 sentinels/490 prose,
repeat1.045386. Warm fixed-ID short English four-span infilling FULL model
encoder/crossKV/decoder/head/argmax/stop; startup/initial team/load/tokenization/
serialization/cleanup excluded. No cold service/general-context claim. Different
sources/cohorts/profiles prevent causal n/quality/speed comparisons.

[392 real routing audit](METH_392_SWITCH_ROUTE_AUDIT_RESULT_20261004.md),62bde68:
ALL384 existing complete quality outputs freshly exact363/387/389,2.860s/55MB.
Real selected-probability medians .098/.106 at256 and .248/.258 at128,
teacher/natural. All256 routes would change selected scaling>1% if normalized
only on the best candidate. This is algebraic sensitivity, not whole-model
harm. Exact best-expert retrieval alone cannot preserve full softmax scaling.

## Reproducible path, failures and limits

| Step | Current evidence |
| --- | --- |
|321/324/325 applicability/reference |Same core geometry/top1/ReLU; isolated Transformers4.57.6 reference; ordinary project5.13.1 unsupported |
|326/327 and378/379 original sources |ALL6392/3320 finite original tensors/ties,12x256/128 distinct WI/WO pairs; distinctness is not individual usefulness |
|335/338 and380 export |Same row-I8/F32 recipe and full segment readback, actual14.818/7.542GB;335/337 interruptions retained |
|334/356 and381 target numerics |I64/A16 independent full state/logit/route/cache/greedy; original-backend329/330/332 failures retained |
|362/363 and382/387 whole quality |NEW source-specific24-book/96-case cohorts, ALL18 each; now consumed.386 mismatched original teacher-mode bridge FAIL retained |
|374/375/376 source256 runtime |Six actual physical workers, complete quality-byte identity/cost/FULL-rate PASS |
|389/390/391 source128 runtime |Three actual workers, both sources'192 teacher/natural bytes exact; source128 cost/rate PASS.388 early-launch HEAD-binding apparatus failure retained |
|383/390 and old threading paths |Source128 six-worker and source256 three-worker repeat FAIL; unchanged paths closed. Draft384/385 ineligible |
|368/369/370/371/372/373 actual bank interventions |Matching bank identity useful; physically smaller64/128 variants have mixed quality.256/64 full cost+2.91%/upper3.62%; no DRAM or causal n gain |
|GigaChat314/316/319 |Frozen donor-adaptation assets reused; compact latent bound FAIL; full-width additive I8 synthetic decoder38-56ms/repeats FAIL. Generic port paused |
|Dense276-297 / Granite4HTiny / StdMoE |Quality, active-cost or precision failures; no unchanged promotion |

[Method](METHOD.md), [history through373](INDEX_HISTORY_THROUGH373_20261004.md),
[prior evidence](PRIOR_EVIDENCE.md); immutable numbered protocols/results.
Source128378-391 reports document actual acquisition/export/quality costs and
first failures. Both scales preserve all original banks/core without training.

## Exact artifacts and next resumption

[Base256 reproduction](SWITCH_BASE256_REPRODUCTION_20261004.md) and
[base128 reproduction](SWITCH_BASE128_REPRODUCTION_20261004.md) give full identities,
commands, frozen checkouts and qualified source-specific profiles. Actual
original128 OWN artifact is NOT370 n128 subset. Isolated324 original environment
reusable. Qualified binaries374 and389 remain separately bound to their rates;
a changed engine/controller needs its own exact-source/full-output verification.

Next393: instrument actual normalized router inputs/full scores while retaining
ALL384 complete outputs exactly363/387/389; quantify an oracle ranked shortlist's
retained probability mass and selected scaling before choosing a hierarchy/LUT.
No rate polishing or automatic old GigaChat-port restart. Exact top1-only traces
cannot reconstruct ranked tail mass. The necessary candidate bound in392 is
not a sufficient shortlist size or changed-router quality result. More useful
choices, physical DRAM, other families/contexts/~100B remain open.
