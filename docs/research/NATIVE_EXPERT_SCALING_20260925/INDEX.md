# Native expert scaling: research control index

**3 October 2026. Branch:** `research/native-expert-scaling`.
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

## Current decisive evidence

- [276 complete archive](METH_276_DIAGNOSTIC_COMPLETE_CORE_RESULT_20261002.md)
  has725 fields/1.329GB and scoped GPU quality passes. Actual original
  C arithmetic fails285's numeric gate;288/289 repairs not adopted.
- [295/296 actual native prediction/generation](HISTORY_METH295_THROUGH299_20261003.md)
  pass their fixed scopes. [297 anonymous semantics](METH_297_NATIVE_SEMANTIC_RESULT_20261002.md)
  closes the unchanged native profile:unsupported41 versus E1280 40,
  despite five of six gates passing. Skip unchanged-profile PIQA/K64/rate
  promotion;no regrading or favorable source subset.
- [298 GigaChat full covariance](METH_298_GIGACHAT_FULL_COVARIANCE_RESULT_20261002.md)
  rank192 FAILS all four gates,including evaluation-domain oracle median
 87.454%<95%. Verified106/125-domain actual source input captures remain
  reusable:67,647/47,931 routed rows,394.70/279.66MB,exact moment controls.
- [299 original nonlinear32-channel omission](METH_299_GIGACHAT_NONLINEAR_BLOCK_RESULT_20261002.md)
  FAILS all24 gates;half-width greedy test output error median51.787%.
  No cheap-router continuation of the unchanged omission rule.
- **[300 full-feature pair LUT complete cost](METH_300_GIGACHAT_VECTOR_LUT_COST_RESULT_20261003.md)**
  rejects training/export of this fixed U8/two-coefficient format. Freeze
  `f2a8e38`,exit0,1.797s/endRSS23.59MB;all414 source/Q4 descriptors and
  every01/04 organ reconcile. Complete addressed weights829.378MB/token,
  optimistic palette sharing829.150MB,versus560MB.812.81M lookups,
  424.17-494.67MB table writes/token are LOGICAL counts,not DRAM/rate.
  Even zero-head/zero-palette leaves729.85MB. Raw SHA
  `9458aec3f6baafbe00eaa266a3ac7da84c43040153cbfee3fc24ea9fd64dcd75`.
  Allthree gates fail;independent shape recount exact. No palette training,
  new donor inference/GPU/T4/downloads or native performance claim.

## Exact resumption

No scientific job active. Two-coefficient U8 proposal is now CLOSED as a
complete candidate under the unchanged cost allotment;keep its original
[proposal](FULL_FEATURE_VECTOR_LUT_PROPOSAL_20261002.md) and300 failures.
Full-feature codes need<=2.676851bits/coefficient BEFORE palette overhead,
with the priced full rows/scales/controls. The next distinct LUT hypothesis
is FOUR original coefficients/U8 non-Cartesian index(2bits/coefficient).
This is unmeasured,not a retained quality recipe. First freeze a new
complete-organ descriptor cost/layout/control/native construction+lookup
preflight,BEFORE any palette optimization. Charge all MLA/shared/dense/head
queries and real selected reads;down and MLA head inputs differ. Then only
if that cost path is credible,freeze source-bound calibration,matched
Cartesian control,error/resource gates before observations using verified
298 captures. Do not repeat49min source capture or old34/130/132 factor-LUT
assays;06/08/09 low-bit whole-model quality failures remain relevant.

Useful RAM-scale n also needs quality-validated selective search:300's
analytical64->640 formula keeps tables/selected routed reads fixed but
raises flat router9.8304->98.304MB/token,encoded payload5.388->48.180GB.
Only64 pretrained experts are real;no synthetic/copy bank quality claim.
See [capacity and CPU routing boundary](NATIVE_CAPACITY_SCALING_STATUS_20261002.md).
Generic donor port stays paused. New small-core quality needs a changed
scientific variable and untouched sources;292 sources now consumed.
Actual phase60 same-artifact independent quality/K64/accepted>=50 plus
useful additional capacity and cross-family10B/100B remain missing.

## Method, history and workspace

[METHOD](METHOD.md) describes reproducible steps and applicability;
[PRIOR_EVIDENCE](PRIOR_EVIDENCE.md) is the initial native inventory.
History: [through216](HISTORY_THROUGH_METH216_20261001.md),
[217-264](HISTORY_METH217_THROUGH264_20261002.md),
[265-280](HISTORY_METH265_THROUGH280_20261002.md),
[281-294](HISTORY_METH281_THROUGH294_20261002.md),
[295-299](HISTORY_METH295_THROUGH299_20261003.md). Historical next steps
are superseded. Fixed259 strict semantic failure,264 cached-reference
failure,274/275 component-cost failures and earlier hard-carve/weight-rank/
static-bias/affine/intercept/amplitude/weak-child stops remain recorded.

Preserve unrelated edits in `docs/research/RESEARCH_INDEX.md` and
`benchmarks/donor_adaptation/density/build_document_holdout.py`;stage exact
paths. Docs English/user updates Italian. Graphify explicit graph work only;
routine hooks removed with backups,GitLFS retained. No subagent delegation.
