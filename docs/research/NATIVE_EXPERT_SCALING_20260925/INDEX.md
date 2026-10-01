# Native expert scaling: research control index

**Date:** 1 October 2026. **Branch:** `research/native-expert-scaling`.
**Status:** active research; no artifact meets all final requirements.

## Goal and constraints

Transfer a pretrained LLM into a compact reusable core plus selectively
consulted conditional capacity, execute it in `benchmarks/phase60/engine.c`,
retain useful donor-relative held-out/generative/task quality and measure
>=50 accepted batch-1 tokens/s on the same artifact. Demonstrate which
steps transfer across families/scales, including a real approximately-10B
case; 100B when resources permit. User priority: n can grow with RAM,
without proportional active cost or loss from the more complex routing.

Do not substitute synthetic pools, copied experts, component rate or a
generic donor runtime for transferred capacity. Count resident capacity,
active bytes, route cost and useful distinct functions separately. Do not
assume cache residency, ideal DRAM bandwidth or composition quality.
Freeze decisions before observing results. T4 requires prior explanation
of reason, budget and stop. None has been used.

Procedure and prerequisites: [METHOD.md](METHOD.md).
Initial native inventory: [PRIOR_EVIDENCE.md](PRIOR_EVIDENCE.md).
Detailed experiment register and historical decisions through METH-216:
[archived index](HISTORY_THROUGH_METH216_20261001.md). That snapshot preserves
all prior links/operational history; its old next actions are historical.

## 1. Geometry and useful expert-count scaling

- **Established small rung:** centered BF16 Qwen0.5B-Instruct E1280
  retains scoped fresh donor-relative quality ([METH-121/123](METH_121_123_ZERO_MEAN_CHILD_EXTERNAL_RESULT_20260928.md)).
  Its exact route beats shifted IDs on consumed sources
  ([METH-183](METH_183_E1280_CHILD_ROUTE_ALIGNMENT_RESULT_20260930.md)).
- **Useful tenfold growth unproven:** METH-175 E12800 training passes
  artifact/balance but METH-176 fresh gain fails; METH-178/179/201–203
  find little distinct child function. Do not repeat hash/shared-base B
  training without changing route/function coupling; details in archive.
- **Latest route result:** [METH-216](METH_216_LOCAL_CHILD_KEYS_RESULT_20260930.md)
  fits nine parent-local keys in existing child coordinates. All ChatML
  fit gates pass. Raw passes global load/content/coverage; hot-parent
  share fails in layers 12/16/17/20 (30.163% worst vs <=25%). Old/new
  reserved and source screens remain unopened. No B/native promotion.
- [METH-217/218](METH_218_HARD_PARENT_BIAS_RESULT_20261001.md) diagnoses
  and corrects five fit soft-to-hard gaps using only 45 bias changes.
  All fit gates pass, but old reserve fails max-load in 3 raw / 2 ChatML
  layers, all unchanged by that correction. New reserve/source stay closed.
  Stop targeted static calibration; useful route/function coupling is open.
- [METH-227](METH_227_CONDITIONAL_DONOR_TANGENT_RESULT_20261001.md)
  derives176 distinct full-input donor functions; all160 cells occupied.
  E160 reduces E16 consumed SSE24.3% and beats rotated choice, but59.70%
  absolute function error fails <=1%. This is local count gain in an
  inaccurate component, not accepted useful capacity or a full-model gain.
- [METH-230/231](METH_231_SOURCE_DERIVATIVE_CHILD_RESULT_20261001.md)
  fits selected source nonlinearity with anchored output functions.
  Source derivatives resolve one18-repeat input child whose weights copied
  its parent. All176 stored pairs then differ; E160 consumed error9.38%
  and only0.94% E16 gain fail. No full-model/independent-quality promotion.
- **CPU cost:** selected-factor synthetic component medians are below
  0.7 ms, but METH-198/199/200 fail repeatability for the tenfold pool
  ratio. No reliable large-n ratio or full accepted-token rate follows.
  Parent selection at larger count, real DRAM access, distinct learned
  capacity and complete native route/LUT cost remain required.

## 2. Pretrained-to-compact-core transfer

- The quality-valid BF16 E1280 native reference reaches 16.818 tok/s;
  selected-row head version reaches 20.791 (METH-127/129), below target.
- Q6 FFN corrections fail METH-187/190/191; grouped-Q8 scale refinement
  fails METH-195/196. Do not restart these unchanged recipes.
- [METH-211/212](METH_212_STORED_LOADER_RESULT_20260930.md) produces a real
  820.707 MB Q8-FFN/exact-BF16-tied-head core and loader: all 364 tensors
  supply all 290 config-only parameters, with exact consumed-source parity.
  [METH-214](METH_214_FRESH_PREDICTION_RESULT_20260930.md) rejects its fixed
  fresh quality: pooled donor-top1 loss 1.291 points vs <=1, despite BPB,
  category and finite K64 passes. Generation/task/blind stop; METH-215
  runner is unexecuted. METH-213/214 data are now consumed diagnostic data.
- [METH-219/220](METH_220_FFN_INTERMEDIATE_RESULT_20261001.md) finds
  exact FFN byte identity with METH-182, rejects the W8A8 GPU emulator
  at 24/384 component outliers, and validates a float-input C numerical
  alternative. Its 12.540 ms FFN component fails the <=10 ms budget.
  No new model quality is scored; compute-precision change alone has
  not produced an eligible complete saved-core path.
- [METH-221](METH_221_Q6_NATIVE_RESULT_20261001.md) validates actual
  Q6 code/scale layout and numerical parity; its 21.542 ms FFN kernel
  fails cost. Stop this kernel; fewer stored bits do not establish rate.
- [METH-222/223](METH_223_FUNCTION_FIT_RESULT_20261001.md) implements
  actual donor layer 12 output-function transfer with an affine common,
  PCA64 and E16/E160 learned affine cells. E160 fits better but validates
  worse; its 46.94% normalized output SSE rejects the geometry. The
  common is inaccurate already at fit, and BF16 rounding is negligible.
  Only one-layer training-window evidence, no full-model/new quality.
- [METH-224](METH_224_NONLINEAR_COMMON_NATIVE_RESULT_20261001.md)
  validates a BF16/FP32 SwiGLU common of hidden width768: native numerical
  parity passes and the24-layer component costs3.261ms/token, within10ms.
  Actual source subsets exercise the operator; they are not trained quality.
- [METH-225](METH_225_NONLINEAR_COMMON_DISTILLATION_RESULT_20261001.md)
  learns a BF16-effective nonlinear common for4096 fixed updates. Consumed
  validation error improves57.70%->37.30% but fails the <=10% intermediate
  gate. Trained C parity/cost pass; stop this recipe before descendants.
- [METH-226](METH_226_DONOR_TANGENT_NATIVE_RESULT_20261001.md) validates
  source Jacobians and a0.654ms24-layer affine C component.
  [METH-228](METH_228_TANGENT_BIAS_BOUND_RESULT_20261001.md) then replays
  all METH-227 scores exactly. Even the nondeployable best
  E160 bias oracle leaves39.48% error; bias-only repair is excluded for
  those fixed products. Nonlinear conditional response remains needed.
- [METH-229](METH_229_SOURCE_CURVATURE_NATIVE_RESULT_20261001.md)
  validates actual affine+H2048 BF16/FP32 combined arithmetic/serialization
  and8.875ms24-layer component cost. Its source-prefix fixtures are not
  a trained quality model; METH-231 rejects the corresponding function path.
- [METH-232](METH_232_FULL_SOURCE_ROW_Q8_RESULT_20261001.md) retains all4864
  nonlinear source units with per-row int8/FP32 operators and no full affine
  branch. Code/numeric/source-function fidelity pass (.01944% pooled error);
  11.023ms component fails<=10ms. Stop this operator before conditional fitting.
- [METH-233/234](METH_234_ROW_Q8_SILU_LUT_RESULT_20261001.md) diagnoses3.004ms
  scalar SiLU/product and replaces it with513 shared FP32 samples/interpolation.
  Every projection byte unchanged;numeric/source-function gates and8.522ms
  component cost pass. This qualifies the new full-feature operator only.
- GigaChat10B Q4 retains scoped quality but costs 1016 MB active/token.
  METH-180/181 global rank-192 variants fail energy proxies. Frozen
  donor-adaptation C fidelity work is reusable evidence, not a resumed
  generic port. StdMoE W4 quality/cost also fail; metadata-only Granite
  is not a transfer result. See METHOD for revisions/assets/variants.
- No compact full composition or multi-family/10B/100B method is validated.
  Core recovery needs a changed train-validated representation/adaptation
  plus new quality; eventual native precision/traffic must be measured.

## Current experiment and exact resume

**Latest evidence:** METH-232/233/234 sessions60444/53582/17088 complete,
exit0. METH-232 cost fails; METH-233 bitwise phase diagnosis motivates the
changed METH-234 SiLU lookup. All native prerequisites now pass:8.522ms,
.01944% source-function error,actual C/GPU lookup parity. No project job
active,conditional bank fitted or full model accepted. METH-231's distinct
176 functions remain quality-rejected.
METH-235 session18122 completes,exit0. Continuous QR/source/center controls
pass; all16 stored row-Q8 derivatives fail1% (worst1.178%). Fit-input source
error.02438% passes,but no conditional fit/validation is opened.
**Next exact action:** execute frozen
[METH-236 scale-only Jacobian bound](METH_236_OUTPUT_SCALE_JACOBIAN_BOUND_PROTOCOL_20261001.md).
Keep codes fixed,solve analytic optimal row scales against original source
Jacobians;any parent lower bound>.01 excludes scale-only repair. No new
snapshot,fit/new source or retiming. Local3060,six threads,10min,no T4.
Do not retry independent affine cells, global weight repair, weak-child
training or static-bias calibration. Full same-artifact independent
quality, C rate, large-RAM route/LUT and real multi-family/10B remain required.

## Workspace rules

Preserve unrelated tracked edits in `docs/research/RESEARCH_INDEX.md` and
`benchmarks/donor_adaptation/density/build_document_holdout.py`; stage exact
research paths only. Documentation is English, user updates Italian.
Graphify is optional only for explicit graph work. Routine Git/search
Graphify hooks were removed with adjacent backups; retain Git LFS.
No project inference job preceded METH-217 at this session's process audit.
