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
- [300–306 full/compact CPU cost history](HISTORY_METH300_THROUGH306_20261003.md)
  preserves full-source LUT failures and compact640-function/all9.437B routed
  coefficient fixtures. Exact-I8 kernel305 reproduces303 but FAILS14ms at22ms.
  Packed-I4/tile4 candidate306 retains coefficient capacity, own format controls
  PASS,310.572MB addressed weights/5.127GB peakRSS;PASSIVE20–22ms FAIL14ms.
  No compact teacher training/source-quality or accepted-token evidence yet.
- [307–311 recent cost/representation](HISTORY_METH307_THROUGH314_20261003.md):
  waiting and verified physical affinity do not yield stable14ms; stop tuning.
  PCA16/ten128D parent subspaces and shared256/residual128 both FAIL all gates,
  pooled test errors81.607%/63.968%. No unchanged student/export promotion.
- **[314 full source MoE target builder](METH_314_GIGACHAT_MIXTURE_TARGETS_RESULT_20261003.md)**
  all six cases PASS source/hash/router-slot/post-SwiGLU/archive controls.
  37,381 actual inputs,149,524 selected parent outputs,1.609GB archives;
  original GGML F32 routing/BF16 activation packing, max nonlinear error3.153e-7.
  69.5s/3.298GB maximum checkedRSS.312/313 failures/partial assets preserved.
  This is source-target reconstruction, not a trained compact artifact or rate.

## Exact resumption

No scientific job active. Pair/vector4 full-source LUT implementations are
CLOSED for unchanged candidate training;no table-only/expert-only rescue
licensed. Old rank192 and direct-channel omission remain failures. Native
fixture timing is enabling cost evidence,not donor quality,useful capacity,
physical DRAM measurement or accepted decode rate.

Next exact action: freeze the [COMPLETE mixture pilot](GIGACHAT_COMPLETE_MIXTURE_PILOT_PROPOSAL_20261003.md)
using qualified314 archives: shared output space plus the GATED SUM of four
regional source-parent residual projections, evaluated against full source
MoE output, with fit-only input routing/control/empty-region rules. No parent
error proxy or student training.20min/12GiB, all states and coverage retained.
Fit parent counts down to10 and one test parent absent already preclude
claiming all640 useful children; no silent exposure waiver. Define new data
needs only after this bounded complete-function diagnostic. No source recapture.
303–311 unchanged profiles/recipes failed or inconclusive; stable complete
native cost and independent whole-model quality still mandatory. No live job.

Useful RAM-scale n still needs selective search:flat router9.8304->98.304MB
for analytical64->640;only64 real pretrained experts exist. No duplicate/
synthetic bank quality claim. See [capacity and CPU routing boundary](NATIVE_CAPACITY_SCALING_STATUS_20261002.md).
Generic donor port stays paused. New small-core quality needs changed
scientific variable and untouched sources;292 sources consumed. Independent
whole native quality/K64/accepted>=50 on SAMEartifact,useful added capacity
and verified cross-family10B/100B applicability remain missing.

## Method, history and workspace

[METHOD](METHOD.md) describes reproducible steps and applicability;
[PRIOR_EVIDENCE](PRIOR_EVIDENCE.md) is the initial native inventory.
History: [through216](HISTORY_THROUGH_METH216_20261001.md),
[217-264](HISTORY_METH217_THROUGH264_20261002.md),
[265-280](HISTORY_METH265_THROUGH280_20261002.md),
[281-294](HISTORY_METH281_THROUGH294_20261002.md),
[295-299](HISTORY_METH295_THROUGH299_20261003.md),
[300-306](HISTORY_METH300_THROUGH306_20261003.md),
[307-314](HISTORY_METH307_THROUGH314_20261003.md). Historical next steps
are superseded. Fixed259 strict semantic failure,264 cached-reference
failure,274/275 component-cost failures and earlier hard-carve/weight-rank/
static-bias/affine/intercept/amplitude/weak-child stops remain recorded.

Preserve unrelated edits in `docs/research/RESEARCH_INDEX.md` and
`benchmarks/donor_adaptation/density/build_document_holdout.py`;stage exact
paths. Docs English/user updates Italian. Graphify explicit graph work only;
routine hooks removed with backups,GitLFS retained. No subagent delegation.
