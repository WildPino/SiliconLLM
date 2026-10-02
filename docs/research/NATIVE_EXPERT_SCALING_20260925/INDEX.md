# Native expert scaling: research control index

**2 October 2026. Branch:** `research/native-expert-scaling`.
**Active research:** no artifact meets all final requirements.

## Goal and constraints

Transfer pretrained capacity to a compact reusable core plus selectively
consulted useful functions in `benchmarks/phase60/engine.c`. Preserve useful
donor-relative held-out/generation/task quality and measure >=50 accepted
batch1decode tokens/s on the SAME artifact. Demonstrate transfer across
families/scales,including real approximately10B/100B when resources permit.
User priority: useful n grows with RAM without proportional active cost or
quality loss from more complex CPU routing/LUT and real DRAM consultation.

Count distinct functions,route labels,stored/active/read bytes separately.
Synthetic/copied capacity or component rates cannot establish the goal.
Freeze gates before observations. LocalRTX3060/six threads;T4 needs prior
reason/budget/stop communication. No CPU performance job overlaps model work.

## Two research questions

| Question | Established | Still missing |
| --- | --- | --- |
| Useful large-n target | Small centered BF16 E1280 retains scoped quality;30,556 unique functions/164 aliases stored with1280 route labels/layer | Useful tenfold new capacity;parent-selection growth,large-RAM actual routing/LUT/DRAM cost and independent quality |
| Pretrained-to-compact transfer | Complete276 diagnostic1.329GB archive/own725-field loader;all6144 composition controls and277/280 consumed/new-source prediction pass | Actual full native arithmetic/cache/quality and accepted>=50 on this artifact;sparse/larger-donor variants |

This dense0.5B case still executes all4864 FFN features/layer;it does not
establish constant-active-cost conversion for10B/100B. GigaChat assets and
fidelity from frozen `research/donor-adaptation` are reusable;its scoped Q4
baseline charges1016MB active/token. The generic donor port stays paused.

## Current actual artifact and decisive stops

- [259 complete archive](METH_259_UNIQUE_BANK_CORE_RESULT_20261002.md) preserves
  all original conditional functions through unique dictionaries/alias maps.
  [267 strict semantic failure](METH_267_ANONYMOUS_SEMANTIC_RESULT_20261002.md)
  closes fixed259 promotion;prediction/generation-health/PIQA passes cannot
  override it. [268/269 diagnosis](METH_269_FFN_BF16_NODE_RESULT_20261002.md)
  motivates changed source features and induced-routing qualification.
- [271 fixed source128](METH_271_OUTPUT_AWARE_ROWS_RESULT_20261002.md) improves
  mean/reserve source error22.42%/21.87%,but component cost fails.
  [274 actual CPU I16](METH_274_I16_SHARED_INPUT_RESULT_20261002.md) qualifies
  all6144 numeric/source states and59,768,832 integer dots;10.550804ms fails
  the10ms allocation. [275 paired layout](METH_275_PAIRED_I16_LAYOUT_RESULT_20261002.md)
  also fails10ms/5% improvement. Do not reclassify or retime fixed recipes.
- [276 diagnostic composition](METH_276_DIAGNOSTIC_COMPLETE_CORE_RESULT_20261002.md)
  explicitly revises assembly order:the failed10ms component allocation is
  not the user whole-model50tok/s gate. Only72 private fields change;653
  others byte exact259. Actual archive1,329,447,260bytes SHA
  `4f9b9c7a76475d9b6e241947ee590884ea268d4bf7f02097df57ad4b2ac23fe9`.
  All725 fields/own loader/6144 FP32,BF16-return,route/alias/isolated
  conditional composition pass. Original zero-score CUDA launch failure
  preserved;environment-only repair134.422s. Diagnostic/unpromoted status.
- [277 development](METH_277_COMPLETE_I16_DEVELOPMENT_RESULT_20261002.md) passes
  existing consumed gates with exact194/260 controls. Versus259,BPB improves
  .000119123 but agreement drops.244771points;no semantic repair inferred.
- [278 new sources](METH_278_COMPLETE_I16_FRESH_MANIFEST_RESULT_20261002.md) and
  [279 answerability](METH_279_I16_SOURCE_ANSWERABILITY_RESULT_20261002.md)
  preserve24 new source IDs after3200 exclusions;all source-only annotations
  committed before inference. Same source pool,not donor-pretraining exclusion.
- [280 new prediction](METH_280_COMPLETE_I16_FRESH_PREDICTION_RESULT_20261002.md)
  passes all pooled/category/BPB/top1/K64 gates:BPB1.17331320 versus donor
  1.17828037/E1280 1.17323483;donor-top1 96.5006%,+.291616points versus E1280.
  Session86930 exits0,112.156s. These source IDs are now consumed for any
  future candidate development. Generation/semantic/task/rate remain due.

## Exact live resumption

[281 full-prefix generation](METH_281_COMPLETE_I16_GENERATION_RESULT_20261002.md)
passes all72 continuations/health/generatedK64 gates. Candidate EOS22/24,
early0/repeat1 versus donor22/0/1 andE1280 23/0/1. Session76893 exits0,
602.594s,no active model/benchmark job. Labeled texts have not been inspected.

[282 anonymous semantics](METH_282_COMPLETE_I16_SEMANTIC_RESULT_20261002.md)
passes all six fixed gates. All72 findings committed `70e4061` before map
inspection. Candidate unsupported37/severe17/missing0 versus donor42/21/0
andE1280 42/20/1. Same-agent anonymous finite screen,not broad quality proof.
Raw SHA `071e797942e9b225151a60b282f96c8d4ade0b561913e6098c5adc3b525906db`.

[283 full PIQA regression](METH_283_COMPLETE_I16_PIQA_RESULT_20261002.md)
passes all four unchanged gates. Candidate1289/1838 versus1291 for EACH
control,-.108814points;paired lower-.598477/-.544070points. All recomputed
control NLL/choice rows exactly266. Raw SHA
`2ed1e54df9722c32287445eeb2b8d40590742b1893656c374cff6490912a7725`.
Session97267 exits0,1101.906s,endRSS3.43GB/peakCUDA3.14GB. No active job.

[284 actual phase60 archive/operators](METH_284_NATIVE_ARCHIVE_RESULT_20261002.md)
passes all nine guards at freeze91923a5:all725 fields/direct original archive,
all6144 source bytes exactly274,parent/child/alias mismatches0/0/0,gate
error0,conditional relativeL2 median0/max.00385553 (not all BF16 bitwise).
Wrong actual B lookup preserves metadata/IDs but fails contribution checks.
Raw SHA `615e4854b8e9300f660c66f513152e4ed63b78e41f6d521400f784bede0ca18a`.
Session85190 exits0,68.422s (GPU52.328/CPU16.078),no active job.

[285 whole prefix/cache smoke](METH_285_WHOLE_PREFIX_RESULT_20261002.md)
FAILS compact-core-only logit maximum .054223787 above .05. Both arms
48/48 top1; complete-bank hidden/logit maxima .04757/.04791 pass. All12
CPU cache rebuilds byte exact and all12 erased-history faults detected.
Session72992 exits0,121.266s,no active job. Raw SHA
`c9b87bdaa42fed8e9313d6634671ded9f3fa31993eb9f46e23a43c95885f6eda`.
Fixed native recipe stops before quality/K64/rate; retain unchanged limits.

[286 same-input localization](METH_286_SAME_INPUT_RESULT_20261002.md)
observes actual MATH SDPA/all24 calls, exact forced-MATH control and RoPE.
Norm/qkv/current attention maxima<=.001674 with median0. BF16 probability
variant worsens error; frozen halving indicator FALSE,do not adopt it.
Session82145 exits0,56.750s,no active job. Raw SHA
`1c73b9fa032318f820de796b3e7743708846fcd4888bb414d4e75fd1750202bc`.

[287 residual trace](METH_287_LAYER_TRACE_RESULT_20261002.md) reproduces285
CPU final/GPU tail bytes exactly. Same-input o/postnorm maxima .001038/
.002008; sourceFFN FP32 maximum8.51e-7/BF16 .000578; both residual additions
exact. Propagated layer-residual maximum reaches .06735. Session92947
exits0,59.781s,no active job. Small local differences accumulate; no single
cause/repair proven. Raw SHA
`32adcf57d582270510bd689726b1e34c217310a213f19635a5f15056e99f1650`.

**Next exact action:** prospective changed norm-square accumulation only,
F64 sum then original F32 mean/inverse/BF16 boundaries; test entire285
prefix/cache assay against pinned original reference with unchanged gates.
No weights/routes/data fitting; changed
execution needs new record and same285 gates. Native BPB/generation/task/
semantic/K64 and accepted>=50 still due. Retain large-n/RAM/real DRAM/family
goal and original259/264/274-275 stops.

Prepared [native implementation plan](NATIVE_COMPLETE_I16_IMPLEMENTATION_PLAN_20261002.md)
binds actual725-field types,Python BF16 cast/rounding boundaries,unique-B
alias resolution and finite K64 head requirements. It is a source-based
plan,not executed C qualification or a prospective acceptance protocol.

## Method,history and workspace

[METHOD](METHOD.md) records reproducible tools/prerequisites/variant limits;
[PRIOR_EVIDENCE](PRIOR_EVIDENCE.md) is the initial native inventory.
History:[through216](HISTORY_THROUGH_METH216_20261001.md),
[217-264](HISTORY_METH217_THROUGH264_20261002.md),
[265-280](HISTORY_METH265_THROUGH280_20261002.md),
[former method](METHOD_HISTORY_THROUGH264_20261002.md). Historical resume
instructions are superseded. Closed hard-carve/weight-rank/static-bias/
affine/intercept/amplitude and weak-child recipes stay closed;parameter
hashes/balance alone do not prove useful learned capacity.

Preserve unrelated edits in `docs/research/RESEARCH_INDEX.md` and
`benchmarks/donor_adaptation/density/build_document_holdout.py`;stage exact
paths. Docs English/user updates Italian. Graphify explicit graph work only;
routine hooks removed with backups,GitLFS retained. No subagent delegation.
