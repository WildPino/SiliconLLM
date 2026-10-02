# METH-280: new-source complete I16/private128 prediction passes

Freeze `b9eacb7`;[protocol](METH_280_COMPLETE_I16_FRESH_PREDICTION_PROTOCOL_20261002.md).
Raw `meth280_complete_i16_fresh_prediction_result.json` SHA
`6419cecee701182475cbe236c53987a4732faa37685986e25a5cf4b0c3b72f11`.
Unchanged complete276 diagnostic artifact SHA
`4f9b9c7a76475d9b6e241947ee590884ea268d4bf7f02097df57ad4b2ac23fe9`;
278 sources/279 source-only answerability and annotations bound before scores.

## Actual result and decision

All prospective pooled/category BPB/top1,immutable archive/loader/source/
answerability and finite prompt K64 inclusion/exact-rerank gates pass.
All24 new sources scored for three arms,full-prefix/no-cache/full BF16 head.
No refitting,source replacement or new tolerances.

| Pooled measure | BF16 donor | BF16 E1280 | Complete I16/private128 |
| --- | --- | --- | --- |
| BPB | 1.1782803705 | 1.1732348289 | 1.1733131972 |
| Donor top1 agreement | 100% | 96.2089915% | 96.5006075% |

Candidate minus donor BPB-.00496717327;minus BF16 E1280+.00007836826;
agreement plus.291616points versus E1280. Descriptive10,000-source-bootstrap
seed280280,p05/p95 BPB deltas:donor[-.005780818,-.004209371],
E1280[-.000190658,+.000351262]. The E1280 interval includes zero;no
significant improvement is claimed. The source cohort differs from261,
so comparing its absolute scores to263 cannot isolate an artifact effect.

Category candidate BPB minus BF16 E1280:
code+.000114646,prose+.000015392,technical+.000105067.
Agreement changes versus E1280:-.071839/+.149589/+.793651points.
All below frozen BPB/top1 bounds;all K64 prompt misses/rerank mismatches zero.
Raw retains all source-level document/prompt rows and resource/hash evidence.

Session86930 exits0,112.156s;endRSS3,298,586,624bytes,
peakallocatedGPU4,100,675,584bytes. LocalRTX3060/six threads,no download/T4,
no apparatus repair or overlapping CPU performance job.

## Interpretation and next action

This is an independent project-transfer prediction screen at new source IDs,
not a new corpus or certified donor-pretraining exclusion. These sources are
now consumed for any future candidate development. No generated text has
been observed from them;source-only summaries/details remain frozen.
Pass licenses frozen281 full-prefix generation-health/K64 evaluation and
subsequent prospective PIQA regression and arm-anonymous semantic review.
Neither this result nor276 composition reopen fixed259 semantic267 or
274/275 component-cost stops. Actual full-model native quality/accepted>=50,
useful RAM-scale n/complex routing/LUT/DRAM and10B/100B/family transfer remain
open. The same artifact is still explicitly diagnostic/unpromoted.
