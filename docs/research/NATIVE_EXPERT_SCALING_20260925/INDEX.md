# Native expert scaling: research control index

**Date:** 2 October 2026. **Branch:** `research/native-expert-scaling`.
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
- [METH-235/236/237](METH_237_QUANTIZED_OUTPUT_PROJECTION_RESULT_20261001.md)
  transfers continuous source derivatives into full nonlinear output-only
  priors; all16 stored int8 derivatives fail1%. Optimal scales and four
  encoded-feedback cycles cannot pass those frozen recipes. No fitting/
  validation/full-model promotion; mixed precision is qualified below.
- [METH-238/239](METH_239_MIXED_OUTPUT_PRIOR_RESULT_20261001.md) implement
  fixed32 BF16 output escapes:actual native8.727ms and all16 stored
  source-prior derivative/value controls pass. Gate/up/LUT bytes unchanged.
- [METH-240](METH_240_MIXED_CONDITIONAL_PAIR_RESULT_20261001.md) fits176
  distinct readouts with .01566% E160 consumed output error,but tenfold
  count loses22.39% against E16. Accurate absolute functions alone are
  not useful additional capacity. No stored routed C/full-quality promotion.
- [METH-241](METH_241_SOURCE_PRECISION_REPLAY_RESULT_20261001.md) finds exact
  BF16 target replay,while FP32 source values differ. [METH-242](METH_242_BF16_VALUE_PRIOR_RESULT_20261001.md)
  changes only source values and loses24.03%;[METH-243](METH_243_PARENT_VALUE_PRIOR_RESULT_20261001.md)
  preserves fitted parent values and loses22.61%. Close these intercept
  recipes;the encoding uncertainty is resolved by METH-244 below.
- [METH-244](METH_244_READOUT_ENCODING_RESULT_20261001.md) separates176
  continuous fixed-fit solutions from actual encoding. Raw E160 fit gain
  is49.53%;quantization erases it. This motivates separately encoded small
  corrections while retaining the parent,first actual native cost/fidelity.
- [METH-245/246](METH_246_SINGLE_TEAM_RESIDUAL_RESULT_20261001.md) implement
  rank32 BF16 corrections without reencoding the base. First four-team
  kernel misses10ms;changed one-team kernel is bitwise equal and9.637ms.
  Source-derived fixtures only;learned factor preservation is METH-247.
- [METH-247/248](METH_248_RAW_VALIDATION_RESULT_20261001.md) implements
  151.681MB real factor bank. Fit gain survives11.72%,consumed count loses
  2.26%;raw final readouts lose27.78%. Preserve full parent function and
  change continuous hierarchy before codec-only/rank/strength attempts.
- [METH-249](METH_249_PARENT_ANCHORED_LATENT_RESULT_20261001.md) preserves
  the complete actual parent and adds tied-latent continuous child residuals.
  All176 functions are distinct; consumed count gain0.0829% is positive
  but fails10%. Diagnose actual-parent error output space before another fit.
- GigaChat10B Q4 retains scoped quality but costs 1016 MB active/token.
  METH-180/181 global rank-192 variants fail energy proxies. Frozen
  donor-adaptation C fidelity work is reusable evidence, not a resumed
  generic port. StdMoE W4 quality/cost also fail; metadata-only Granite
  is not a transfer result. See METHOD for revisions/assets/variants.
- No compact full composition or multi-family/10B/100B method is validated.
  Core recovery needs a changed train-validated representation/adaptation
  plus new quality; eventual native precision/traffic must be measured.

## Current experiment and exact resume

**Prior continuous coupling evidence:** METH-249 keeps the complete actual parent;
all176 decoded functions distinct, fit gain2.98%, consumed count gain
0.0829% and rotated gain0.2251% fail10%. Paired gain is positive. Two
aggregation apparatus stops precede a frozen no-refit recovery: all16
parent scores exact, Python `sum()` restored, original validation once.
Session86289 exits0. No codec/native bank or full quality/rate promotion.

[METH-250 fit-only diagnosis](METH_250_RESIDUAL_OUTPUT_SPACE_RESULT_20261001.md)
then passes: inherited space captures2.94% of centered parent error,new
stored residual-covariance space15.40%. All16 exact parent/eigen/readback
controls pass. Session61839 exits0. This is optimistic projection,not
learned count gain; no validation targets were read.

[METH-251](METH_251_MATCHED_RESIDUAL_BASIS_RESULT_20261001.md) then executes
the matched hierarchy: all controls/176 distinct functions pass, fit
count gain4.66%, but consumed E160 loses0.1826% and paired gain is negative.
Session84313 exits0; no job active. Close inherited/residual-left output
recipes before rank/tau/codec retries. No stored/native bank promotion.

[METH-252/253](METH_253_SPLIT_PRIVATE_FEATURE_RESULT_20261001.md) implements
actual private32 original BF16 gate/up rows before unchanged LUT/readout.
Source-prefix function error falls81.03%; branch-per-row10.299ms fails,
changed split-workshare kernel is bitwise equal and9.318ms passes. Actual
329.334MB fixture/all361 segments qualified. Session17039 exits0.

[METH-254](METH_254_PRIVATE_UNIT_PAIR_RESULT_20261001.md) stops at6 positive
source-unit promotions on first learned parent0, before32-unit bank or any
validation. Session86792 exits0; no job active. Source/native bindings and
first learned-parent controls pass. Close this fixed learned-readout pair;
incomplete raw fit-summary0.0 is not a quality score.

[METH-255](METH_255_FEATURE_READOUT_COMPOSITION_RESULT_20261001.md) passes
all16 old coefficient/score controls but diagnoses learned-private fit
loss28.02%; coherent-source private gains7.91% and remains4.56x less
accurate than learned shared. Both prospective coherence gates fail.
Session68832 exits0; no job active. Close fixed promotion/source-reset
composition, not all nonlinear source responses. No validation was read.

**Latest decisive evidence:** [METH-256](METH_256_BOUNDED_SOURCE_RESPONSE_RESULT_20261002.md)
completes all176 distinct actual functions, exact parent/solver/snapshot/
score controls and positive paired gain. Count gain0.08439%, rotated
gain0.03415% fail frozen10% gates. Session64120 exits0; no job active.
Close this fixed bounded nonlinear-amplitude recipe before native blending.

[METH-257](METH_257_FULL_SOURCE_CORE_STOP_20261002.md) assembles all361
source segments but stops on first-layer effective-BF16 sibling B
distinctness, before saving any core or reading quality targets.
Session16232 exits1; no job active.1280 routing labels are not guaranteed
to be1280 genuinely different existing functions; cause/count unmeasured.

[METH-258](METH_258_EFFECTIVE_BANK_DIVERSITY_RESULT_20261002.md) passes all24
read-only audits:1243..1280 effective codes/layer,30,556 total,164 route
aliases.94 groups already duplicate raw B; zero merges at centering or
BF16 casting. Minimum different-code real-state distance3.807804e-7.
Session28006 exits0; no job active. No quality targets or new fit.

**Next exact action:** execute frozen [METH-259 unique-bank full core](METH_259_UNIQUE_BANK_CORE_PROTOCOL_20261002.md),
725 physical fields plus original-leaf alias maps, complete saved loader
and original-source/conditional bitwise conservation. Distinguish1280
route labels from actual1243..1280 unique codes; no duplicate capacity.
Protocol/code present; no259 execution/result yet. No saved257 artifact.
Then frozen consumed-development and new independent full-model quality
before native alias lookup/LUT/DRAM/accepted-rate promotion.
Current253 fixed source operator remains9.318ms, not a full model/n result.
Full independent quality, large-RAM n/route-LUT/DRAM, same-artifact accepted
rate and multi-family/10B/100B remain required.
Do not retry closed independent affine cells,global weight repair,
weak-child training,static-bias calibration or fixed intercept variants.
Full same-artifact independent quality,C rate,large-RAM route/LUT and
real multi-family/10B remain required.

## Workspace rules

Preserve unrelated tracked edits in `docs/research/RESEARCH_INDEX.md` and
`benchmarks/donor_adaptation/density/build_document_holdout.py`; stage exact
research paths only. Documentation is English, user updates Italian.
Graphify is optional only for explicit graph work. Routine Git/search
Graphify hooks were removed with adjacent backups; retain Git LFS.
No project inference job preceded METH-217 at this session's process audit.
