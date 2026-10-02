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

## Current decisive evidence

- [276 archive](METH_276_DIAGNOSTIC_COMPLETE_CORE_RESULT_20261002.md):
  actual725 fields,1,329,447,260bytes; all6144 composition controls pass.
  SHA `4f9b9c7a76475d9b6e241947ee590884ea268d4bf7f02097df57ad4b2ac23fe9`.
  GPU277/280 prediction,281 health,282 anonymous semantics and283 PIQA pass
  in their limited scope. They do not establish native quality or rate.
- [284](METH_284_NATIVE_ARCHIVE_RESULT_20261002.md) verifies actual phase60
  direct archive/operator/route/alias controls.
  [285](METH_285_WHOLE_PREFIX_RESULT_20261002.md) numerical smoke FAILS
  bank-off logit5.4224%; all96 top1/12 cache/12 negative controls pass.
  [288](METH_288_NORM_REDUCTION_RESULT_20261002.md)/
  [289](METH_289_RESIDUAL32_RESULT_20261002.md) precision variants also fail;
  neither adopted.
- [290](METH_290_REFERENCE_PREFIX_RESULT_20261002.md) GPU causal reference
  is stable. [291](METH_291_OPERATOR_SURGERY_RESULT_20261002.md) all-CPU
  hybrid exactly reproduces actual285; no one-group GPU restoration
  meets the prospectively fixed rescue indicator.
- [292 explicit policy revision](METH_292_NATIVE_PRIMARY_EVALUATION_POLICY_20261002.md)
  licenses direct ORIGINAL-profile native quality against donor/E1280 on
  new sources. Numeric failures remain failures; quality/rate margins
  unchanged. [New manifest](METH_292_NATIVE_PRIMARY_MANIFEST_RESULT_20261002.md)
  has24 sources,8/category,3224 old-source exclusions;
  [293 answerability](METH_293_NATIVE_ANSWERABILITY_RESULT_20261002.md) passes
  all24 findings committed before inference. No model source selection.
- [294 scorer bridge](METH_294_NATIVE_SCORER_BRIDGE_RESULT_20261002.md) passes
  original forward-body proof, two actual CPU full-head rows byte exact,
  all16 top1 exact and NLL error<=6.76e-13.13.390s CPU assay, not rate.

## Exact resumption

No scientific job active. Next: implement and prospectively freeze METH-295
actual native full-head document BPB/prompt top1 on ALL292 sources against
new BF16 donor/E1280 controls. Relative-cache/global-RoPE window positions
must match original M17 metric. Native generation, anonymous semantics,
full1838 PIQA, actual CPU K64 and accepted>=50 remain due.

The new execution entry is real phase60 C, not a Python oracle. Do not
rebuild the197s model-free source pool or rerun failed precision variants.
Complete model still executes4864 features/layer. No useful large-n,
RAM-scale routing/DRAM cost or cross-family10B/100B claim.

## Method, history and workspace

[METHOD](METHOD.md) describes reproducible steps and applicability;
[PRIOR_EVIDENCE](PRIOR_EVIDENCE.md) is the initial native inventory.
History: [through216](HISTORY_THROUGH_METH216_20261001.md),
[217-264](HISTORY_METH217_THROUGH264_20261002.md),
[265-280](HISTORY_METH265_THROUGH280_20261002.md),
[281-294](HISTORY_METH281_THROUGH294_20261002.md). Historical next steps
are superseded. Fixed259 strict semantic failure,264 cached-reference
failure,274/275 component-cost failures and earlier hard-carve/weight-rank/
static-bias/affine/intercept/amplitude/weak-child stops remain recorded.

Preserve unrelated edits in `docs/research/RESEARCH_INDEX.md` and
`benchmarks/donor_adaptation/density/build_document_holdout.py`;stage exact
paths. Docs English/user updates Italian. Graphify explicit graph work only;
routine hooks removed with backups,GitLFS retained. No subagent delegation.
