# METH-343: complete NEW original-primary prediction, top1 gate FAIL

Freezeddd0107; raw49e0034, authoritative exec59475 exit0. Raw SHA
`a4838903958be5f6bb567c1dedd802a17e6b27e2d97c4f490210139f296cbadf`.
ALL24 previously unscored books/96 known8-token masks evaluated with ORIGINAL
unmodified official4.57.6 CPU1 PRIMARY and native341CPU6 SAME14.818GB artifact.
MAIN2341.609s after imports within40min; max checked combined RSS9,426,763,776B.
EndRSS7,245,824B after working-set trim. No GPU/download/training/timing overlap.

Fresh ALL6392 original coefficients match327 before scoring; namespace/shape/
CPU/F32/unique14.664B/tied aliases exact. Source-shard canonical checks
135.219/145.187/160.078/153.219/146.469/156.922s. Two fresh original complete
states/logits/routes byte-exact cached331; both native outputs SHA exact336.
Fresh whole target payload/manifest/executable/libomp identity exact341/338.
Every original array and actual native output/log/command retained. Original
source policy is canonical functional identity, no new ZIP-envelope claim.

## Frozen book-level paired results

10,000 bootstrap draws,24 BOOKS (four windows/book), seed343343.

| Metric | Native minus original | One-sided95% relevant bound | Fixed criterion | Result |
| --- | --- | --- | --- | --- |
| All11-target-token mean NLL | +.000454 nats | upper+.013893 | <=.05 | PASS |
| Eight masked-token mean NLL | +.023530 nats | upper+.034667 | <=.05 | PASS |
| Masked-token accuracy | +.130208 percentage points | lower-1.171875 points | >=-2 points | PASS |
| Whole8-mask-token teacher-forced exact accuracy | 0 points | lower0 | >=-5 points | PASS |
| Original top1 agreement | 1001/1056=94.791667% | fixed pooled point | >=95% | FAIL |

Masked-token NLL loss is positive (paired one-sided lower+.012106), within
the frozen allowance; do not call this loss zero because other metrics pass.
55 original-top1 differences,52 on masked-word positions,3 on EOS prediction.
Per-position counts[0,6,7,6,12,6,7,6,2,0,3]. Actual known masked-token correct
336/768 original,337/768 native. Whole8-token exact0/96 for BOTH arms: that
particular strict task has no observed sensitivity here. Teacher-forced
correctness is not free autoregressive reconstruction or semantic utility.

## Decision and limits

Seven of eight gates PASS; **unchanged W8A8 candidate NOT qualified**. Do not
round94.7917% to95%, relax the threshold or promote by the other metrics.
All342 sources now consumed; source/candidate changes require NEW quality.
Stop this fixed candidate before free-generation/task/accepted-rate promotion.
Its complete arithmetic336 and forced CPU margin341 remain scoped valid.

Next separately frozen fixed-state344 attribution: distinguish head activation
quantization, head weight coding and original/native upstream states using
actual saved vectors and exact retained original F32 embedding/head. Select a
smallest new precision variable BEFORE actual composition/cost and NEW quality.
Free generation C draft is unexecuted/unfrozen. No final>=50 accepted rate,
useful larger-n/LUT/physical DRAM/cross-family/~100B result.
Reproduction in [343 protocol](METH_343_SWITCH_FRESH_PREDICTION_PROTOCOL_20261003.md).
