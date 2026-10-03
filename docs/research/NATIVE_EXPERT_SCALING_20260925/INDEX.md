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

- [315/316 complete-mixture diagnostic](METH_316_GIGACHAT_COMPLETE_MIXTURE_BOUND_RESULT_20261003.md)
  closes shared256/four regional128 fixed fields: ALL four gates FAIL;
  test median errors53.790%/66.767%/21.560%, joint-span oracles also fail.
  37,381 states completed284.828s/3.252GB checkedRSS;315 wall failure retained.
  No coefficient-only training/export of unchanged fields.
- [317 full-width additive descriptor](METH_317_GIGACHAT_ADDITIVE_DESCRIPTOR_RESULT_20261003.md)
  preserves source32-head/1280 widths;2bits/code coefficient+I8 books,
  originalQ6 head.542.987MB addressed/token PASSES560MB accounting only.
  Flat router at640 grows to631.461MB; hypothetical hierarchy543.019MB,
  approximately24.84GB stored. Larger capacity/route quality unimplemented.

## Exact resumption

No scientific job active.315 partial failure committed before316 repair;
316 all-state equivalence controls pass, scientific recipe CLOSED. No more
unchanged shared/regional projections, coefficient training or affinity tuning.
Qualified314 complete source targets remain reusable, anchor-conditioned and
consumed; sparse exposure cannot establish640 useful learned children.

Next implement and freeze the [full-width additive native preflight](GIGACHAT_ADDITIVE_NATIVE_PREFLIGHT_PROPOSAL_20261003.md):
NEW opt-in C/controller, all64 source-sized banks/32-head MLA/shared1280/
routed1280/dense8960/full actualQ6 head/router/norm, exact scalar decoder
controls, complete14ms/stability gates. Proposed20min/12GiB baseline; no GPU.
Synthetic fixtures establish cost only. No book training before numeric/cost
pass; then separately test actual640-bank allocated/routing cost and learn
source-aware full-output books. Flat routing grows with n, hierarchy requires
its own source coverage/training/quality.6400 bank scenario exceeds local RAM.
Causal complete quality/accepted>=50 SAME artifact and genuine useful larger
capacity/cross-family100B remain mandatory and unverified. No live jobs.
