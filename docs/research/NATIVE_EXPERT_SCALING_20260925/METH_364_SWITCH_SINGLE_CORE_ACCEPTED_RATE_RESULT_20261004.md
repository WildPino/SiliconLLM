# METH-364: SAME quality artifact accepted FULL generation rate FAIL

Freeze d16da82; authoritative exec17332 exit0 fully consumed. Raw SHA
`96fdecb5557161fed1849ba8322c83e55f685754d9cbba00886e07b8b58fb4ba`.
MAIN256.203s/max checked combinedRSS1,332,543,488B. SAME356 binary/338
14.818GB payload/363 ALL18 quality/PRIMARYCPU1 affinity[0]/ALL96 source362
cases, every actual affinity readback and complete full output SHA exact363.
No original teacher loaded and no model/timing overlap. Fresh whole payload,
manifest, binary/compiler/runtime identities exact. No recompile or rescore.

| Metric | Observed | Unchanged criterion | Result |
| --- | --- | --- | --- |
| Accepted generated token rate | 45.111438 tokens/s | one-sided95 lower>=50 | FAIL |
| Book-bootstrap one-sided95 lower | 42.536190 tokens/s | >=50 | FAIL |
| Aggregate three-repeat full-time max/min | 1.005005 | <=1.10 | PASS |
| ALL96 quality outputs / actual affinity | exact363 / [0] | exact | PASS |

81 healthy accepted cases supply895 generated IDs INCLUDING405 structural
markers, and490 prose IDs. ALL96 cases' median full times charged19.839758s;
rejected-case tokens contribute zero. Prose-only rate24.697882 tokens/s,
one-sided95 lower23.305219. Book-level10000 draws/seed364364 unchanged.
Natural lengths10-13 (median11), pretokenized encoder29. Each process one
complete warmup/three measured repeats, original quality acceptance fixed363.
The lower bound and marker-inclusive numerator must not be described as
>=50 prose tokens/s or open-ended chat capability.

## Phase diagnosis and decision

Sum of per-case median phase times: encoder10.448665s (52.67% of full),
cross-KV1.476011s (7.44%), cached decode/greedy7.873360s (39.69%). Phase medians
need not sum exactly to the sum of median whole times. Prefill dominates these
short responses; improving decoder-only rate does not establish accepted FULL
rate. Complete encoder still projects attention O and dense FF one token at a
time. Candidate NEW exact execution: batch those projections and F32 encoder
routers, preserve route order/capacity/per-token rounding, and share I8 weight
conversion over four-token integer batches. Freeze independent primitive/Tiny/
full/cached numeric and ALL96 byte-equivalence criteria before observations.
No alteration of frozen356/363/364, acceptance, data, or 50/1.10 thresholds.

363 original-relative whole quality PASS remains scoped valid. SAME unchanged
356 execution has failed accepted FULL >=50 and is not promoted. Preserve this
complete failure before any new execution candidate. Full measured model
includes encoder/cross-KV/cached decoder/argmax/stop, excludes process startup,
manifest load/tokenization/output/cleanup; load separately reported, startup/
tokenization not timed. Warm/hash-primed pretokenized scope only. Useful larger
n, physical DRAM/LUT and routing, causal bank usefulness, genuine real128 scale
comparison, cross-family/~100B final goal remain open.
Reproduction: [364 protocol](METH_364_SWITCH_SINGLE_CORE_ACCEPTED_RATE_PROTOCOL_20261004.md).
