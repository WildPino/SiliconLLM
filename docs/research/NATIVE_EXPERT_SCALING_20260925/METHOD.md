# Pretrained-to-native conditional-capacity method

12 October2026. **ACTIVE/INCOMPLETE**. This is the reproducible procedure under
construction. It currently produces a failed chatbot and validated analysis/
conversion components, not a capacity-preserving universal converter.
[All-label exact categorical factors/gradient/source audit](BROAD_EXACT_RESIDUAL_CORRECTION_RESULT_20261012.md)
now IMPLEMENTED/LOCALLY VERIFIED on4422 FIT/4386 DEV,all48570 full-V point rows,
all4422 explicit RMS Jacobians and parameter bits. Compact head/Z/radius factors
replace unnecessary dense score archives;independent direct2048 source reading
verifies all outputs/probabilities. One selected.0001 step improves FIT3.10%,DEV
~.97%,all qualityFAIL. [QR gauge/native scalar-norm export](GAUGED_NATIVE_READOUT_RESULT_20261012.md)
now IMPLEMENTED/LOCALLY VERIFIED on all8808 labels:577249896 actual F32 logits
bit exact with independent scalar FMA reference;all gauge/source maps audited.
Scalar epsilon includes perpendicular center energy;normalized head bias is an
explicit engine extension. Zero predictor argmax changes;source-relative DEV
case disagreement41.09%/all qualityFAIL. Teacher-injected code,not causal core.
[Fixed native head moment/cost preflight](NATIVE_HEAD_DOMAIN_PREFLIGHT_RESULT_20261012.md)
now IMPLEMENTED/INDEPENDENTLY VERIFIED for all8808 exact full-vocabulary moments,
initial losses/gradients/positive-sphere codes/support lowers. First precision
guard failure retained;V2 same1e-5 guard remains false. Complete source/native
score audit verifies equations and rounds within score-range KL bound,zero ID
changes. No positive floor,optimized code or causal core. CPU pass estimate28.2s.
[Affine-ray decision assay](NATIVE_HEAD_AFFINE_RAYS_RESULT_20261012.md) now
IMPLEMENTED/INDEPENDENTLY VERIFIED:all2361 full constraints/8808 associations,
2342 actual native target codes,153487654 logits bit exact with scalar reference.
DEV mean case oracle coverage98.6616%,but available KL contribution9.6007.
Winner availability does not preserve probabilities/establish causal encoding.
[GPUF64 initial objective/gradient backend](NATIVE_HEAD_GPU_PREFLIGHT_RESULT_20261012.md)
now IMPLEMENTED/INDEPENDENTLY VERIFIED for all8808 full-vocabulary losses/
257-gradients/IDs,CPU/GPU errors~2.5e-14. Approximate component pass4.3s;
clock15.625ms/desktop sharing,timing admission false. No model/code optimization.
Next [all-label probability-gap selected design](ALL_LABEL_NATIVE_PROBABILITY_GAP_DESIGN_20261012.md)
UNIMPLEMENTED/UNBOUND:128 max updates/1200s solver/1800s capture,half-ball
lower points and actual native sphere codes;full stored-point/native audit.
No repeated winner-only ray optimization or smaller successful cohort substitution.
[Exact categorical V/W/center correction and independent Jacobian audit](EXACT_CATEGORICAL_RESIDUAL_CORRECTION_RESULT_20261012.md)
now IMPLEMENTED/LOCALLY VERIFIED. Actual source FIT72 final states/full trained
head,complete RMS derivative and both center paths;full gradient/differences/
candidate parameter bits/selection/DEV audited. Original coarse nonacceptance
retained;separate smaller-scale check qualifies one FIT-selected step. FIT improves
42%,DEV mean6%,still26/72 wrong. Generic factors not idempotent (.328 defect).
This is a teacher-injected readout fit,not deployed causal encoding or core repair.
The [broader existing supervised labels](BROAD_EXACT_RESIDUAL_CATEGORICAL_NEXT_20261012.md)
follow-up is now COMPLETE as linked above;compact score custody has independent
direct full-V arithmetic qualification. Earlier prospective records are historical.
[Conditional RAW affine charts and all fixed-decoder choices](CONDITIONAL_RAW_CHARTS_AND_CHOICE_RESULT_20261012.md)
now COMPLETE/full audits PASS/quality proxies FAIL. FIT-only4/16 charts at0/12/24,
<=256 active directions,required norm cross/bias/skip offsets are implemented and
stored. Oracle source routing and injected coordinates remain nondeployable.
Every final fixed decoder tested:25/72 source decisions unavailable in any16
candidate. Selected next:exact categorical residual-factor gradient/correction,
[specified but UNIMPLEMENTED](EXACT_CATEGORICAL_RESIDUAL_CORRECTION_NEXT_20261012.md).
[Source-function residual read/write compiler and stored audit](SOURCE_FUNCTION_COORDINATES_RESULT_20261012.md)
now COMPLETE:actual normalized source readers/metric,seven distinct V/W/G bases,
six skips,all336 histories and DEV72 full-V outputs. FIT quadratic optimality
validated;quality proxies all FAIL. Tool can reproduce algebraic construction,
not acquire deployed causal coordinates. New finite-logit diagnostic proves
why source-local Fisher is no finite-loss bound. Selected next conditional RAW
charts is [specified but UNIMPLEMENTED](CONDITIONAL_RAW_RESIDUAL_CHARTS_NEXT_20261012.md).
[Seven residual bases and actual probability sensitivity](SOURCE_RESIDUAL_COORDINATE_AND_READOUT_RESULT_20261012.md)
now COMPLETE/full audits PASS/proxies FAIL. Angular256 optimal FIT reconstruction
still changes31/72 source-readout argmaxes with oracle norm. Function-sensitive
read/write or conditional encoding is selected;energy/normalization alone insufficient.
[Fully trained projected24-layer shadow](FALCON_PRETRAINED_PROJECTED_SHADOW_RESULT_20261012.md)
and full audit now COMPLETE:local F32 operators/coefficient maps PASS,chatbot0/16,
DEV KL22.6229/dis100%. Fixed global P does not preserve composed behavior; drift
already large at boundary4. New state encoding, not merely more dose, is selected.
[Latest restricted D256 trained-operator export/audit](FALCON_PROJECTED_S6_TRANSPORT_RESULT_20261012.md)
COMPLETE: local norm-folding identity and three F32 packages. Composed pretrained
shadow now also complete and failed; see the newer result above.
[Operational index](INDEX.md), [latest actual outcomes](RECOVERY_AND_OPERATOR_TRANSPORT_REASSESSMENT_20261012.md).
Earlier method/status history is byte-preserved in
[METHOD through V4 launch](METHOD_THROUGH_RECOVERED_V4_LIVE_20261011.md).

## Starting models and applicability

Actual useful canonical donor: pinned Falcon-H1-1.5B-Instruct; source tokenizer,
chat template/EOS, weights/config and primary runtime are recorded in experiments.
Source hidden2048/layers24, FFN4608 SiLU gating, hybrid48-head SSM3072/state256
with shared B/C and postgate RMS plus parallel attention. Donor canonical tasks
14/16 are a measured local baseline. Tiny90M/source scan is a separate qualified
component; Qwen and other donor analyses do not establish general conversion.

Current native anchor: original engine.c, D256/layers6,5 Mamba1 SSM+1 SWA,
ternary gated-ReLU banks, packed LUT, F32 sensitive organs; wider DN1024/DT48,
N96,EH128,k8,n1152,V65537 in the failed source-informed learner. Parameters,
selected active coefficients, useful preserved functions and stored bytes must
be counted separately. Arbitrary source architectures are not supported yet.

## Initial analysis and selecting a route

1. Identify actual nonlinearities, norm locations/epsilons/precision, state and
   attention generator dimensions, composition/depth, tokenizer/head and actual
   pretrained coefficient provenance. Do not infer transport from shape agreement.
2. Price compact core/state/head and conditional bank/read traffic/routing for
   fixed active budget and variable useful n. Converter dense storage grows O(n).
3. Qualify source behavior/operator/cache/readout pairing before fitting a codec.
4. Compile exact transforms where possible; name approximations and learning.
   Qualify every boundary and composed behavior on the SAME exported artifact.
5. Choose new experiments to resolve the current measured defect, preserving
   earlier valid controls and all execution/scientific failures.

Current decision:fixed-P,angular256,global source-function256 and tested conditional
RAW charts fail. Exact finite categorical factor/center gradient is now verified;
one FIT-only step improves loss modestly on DEV without restoring decisions.
All existing supervised labels/equal-case weighting now tested/audited;improvement
is small and quality still fails. Reconnect via exact decoder QR gauge/native
scalar-norm head export before another plain-gradient epoch.
Initial Fisher or more unchanged charts do not certify finite loss;available
causal compact routing and composed own-history remain mandatory.
[Exact prospective path](PROJECTED_PRETRAINED_OPERATOR_NEXT_20261012.md).
No blindly continued donor-adaptation or new T4 job.

## Available tools and actual scope

| Stage/tool | Inputs and outputs | Validated status/cost/limits |
|---|---|---|
| Source runtime/canonical behavior/internal capture | Pinned source, histories -> labels/residuals/recurrent operands | Actual source14/16; full48 operands separately adopted; original capture runtime/identity gaps remain |
| original_falcon_learner.initialize | Source weights+P -> projected embedding/head/FFNs, fresh sequence core | Implemented source-informed heuristic; no exact full source-function/sequence/depth transfer |
| Original tensor/packed exporter+chat client | CPU masters -> packed original C artifact+canonical traces | Native consumer real; prefix/split/reuse/IDs/masses qualified locally; learned chatbot quality FAIL |
| Categorical moment/entropy compiler | Source probabilities+fixed native head -> FIT-only supervision | All8808 numerical entries audited; payload18.144MB/63.628x; no capacity preservation claim |
| Causal adjoint/grouped steps | Actual histories+masters ->90 gradients and92-array proposals/export | Qualified58-ID initial+changed bridges; saved-gradient steps reduce local C loss,18/18 wrong |
| Lossless master-transition archive | Actual92 F32 arrays -> exact archive/reconstruction | 2.884GB->186.158MB, all721M words independently exact; future ratio not guaranteed, RAM still dense |
| Recovered complete categorical audit | Sealed24-direction failed producer outputs -> full mathematical report |24/48/48/4 milestones/115 receipts complete; worker6.866h/holder7.962h final isolation FAIL, not successful original custody |
| recovered_categorical_native_chat | Final SAME pack ->16 tasks+own-answer followup+full scores/routes | Actual capture41.015s/audit3.031s,0/16/followup false; raw incorrect-content rates not useful50 |
| inspect_recovered_categorical_route_divergence_v2 | Stored GPU/C ID/mass -> exact divergence locations |24 histories/.828s/49.795MB read;3 membership rows; hidden causal operands/margins unavailable |
| initializer_quadrant_algebra | Exact initializer code+synthetic scalars -> reproducible even-function identity |100 cases/.016s; pre-learning all-expert real mixture only, not learned model/top8/quantization |
| falcon_auxiliary_s6_transport | Actual source coefficients -> original-S6 field compiler+39 F64 operand frames |3 sites/all gates PASS,9.296s/440.009MB OS; independent stored fsum audit6.813s; not compact/native/BF16/chatbot qualification |
| falcon_projected_s6_transport | Actual source/P256/gamma -> restricted core and durable F32 fields |3 sites/all local gates PASS,13.047s/530.104MB OS; all-field audit8.094s; source feature/state widths costly, composed/native behavior untested |
| falcon_pretrained_projected_shadow_v3 | All411 actual source fields ->24-layer projected F32 model/16 tasks+followup/24 DEV/168 residual streams | Capture312.406s/all resource gates PASS;full audit165.078s;quality0/16,DEV KL22.6229/dis100%;not native/selective deployment |
| inspect_projected_shadow_trajectories | Source/shadow seven boundaries ->coordinate/cosine/energy localization |24 cases/168 pairs,25.110s,all hashes exact;initial.149% error,after4~194%;no information-capacity ceiling |
| source_residual_coordinate_feasibility | Raw source FIT/DEV boundaries ->7 angular eigensystems/F64+F32 bases/6 transports |336 comparisons/capture93.125s/audit60.015s;numeric PASS,all reconstruction proxies FAIL;no model/optimizer call |
| residual_basis_readout_probe | Actual source h24/head/gamma ->fixed DEV72 full-V reference+4 projection/norm modes |Capture17.735s/full dense audit19.719s;new31/72 wrong even oracle norm;no composed model or source generation |

Detailed revisions/bindings/invocations/hashes/results and scope are linked in
[latest report](RECOVERY_AND_OPERATOR_TRANSPORT_REASSESSMENT_20261012.md) and
[prior evidence](PRIOR_EVIDENCE.md). No repeated source/backward/donor/DEV runs
are implied by a stored audit. Full input/output custody is prospective; missing
original counters are UNKNOWN, not retroactively filled.

## Exact, approximate and learned parts

**Exact:** canonical byte/parameter boundaries, lossless master transitions,
compiled categorical supervision within declared arithmetic checks; real scalar
SSM head coordinate commutation under its prerequisites; auxiliary generator lift
uses SiLU(a)-SiLU(-a)=a, B/C selectors, copied convolutions/gates/head coefficients.
The lift additionally requires the source primary-only postgate denominator.
It currently uses D2048/DN3680/N256/DT48, not a cheap admitted native artifact.

**Approximate:** P256 residual projection, state96 dual transport, linear384
channel maps/readouts, weight/activation quantization, source24->target6 composition,
projected diagonal norms and the current signed FFN initializer. Separate assays
measure these losses; success separately is not success in composition.
Uniform real ALL-expert initialization recovers mean even parts of source FFNs;
actual top8 routing is different and can break that symmetry.

**Learning implemented:** source-informed joint history/bank updates, categorical
readout fits and24-direction grouped campaign. Report every accepted/rejected
proposal and held-out behavior. Current final chatbot remains0/16. No claim that
more dose, router correction or a free-feature head will recover useful capacity.

## Missing steps and selected experiment

Restricted pretrained-operator compiler is IMPLEMENTED/LOCALLY QUALIFIED on
three sites/39 injected F64 inputs; F32 coefficient export and independent all-field
audit complete. This is not F32 arithmetic/native/chatbot parity. Correct norm transfer
requires folding gamma into each source input projection and accounting for
G=P^T P and dimension-scaled epsilon; averaging a diagonal norm is insufficient
for an exact restriction identity. The costed24-layer trained shadow has now been
completed/audited and FAILED quality;new global/conditional raw state encodings
also completed/audited and failed their local proxies. Do not repeat those same
geometry objectives as if their qualification were still pending. Exact finite
categorical residual-factor correction is now implemented/verified locally;
broader actual labels/score-factor arithmetic also complete/audited. Exact QR
decoder gauge/scalar-norm native head export is the next variable to bind;
source map/normalization must stay equivalent to the fully audited factor point.

Then a cheap conditional nonlinear representation must replace costly source
features/FFNs while preserving useful composed behavior. Original packed LUT/
ternary execution, actual native normalization/extensions and state precision
need explicit source/C qualification. Structured/hierarchical retrieval must
preserve IDs AND masses and real DRAM cost at useful large n. Future sparse
conversion cannot assume the current dense masters/gradients become sparse.

## End-to-end admission and generality

Fresh donor-relative held-out labels AND generative tasks/own-history followup,
canonical tokenizer/EOS/context, on exported native artifact. >=50 accepted
batch1 tokens/s on THAT useful artifact (100 stretch), including actual head,
router/selected weights/core/state/prefill/interaction and declared cold/warm
host/thread conditions. Wrong-content raw IDs/s are not admitted throughput.

Distinct useful n/RAM-driven10B/100B-scale capacities, structured CPU routing
and normalization, physical DRAM and family/scale variants remain missing.
Qwen/Mamba2/attention-only families need their own costed operator and behavioral
branches; no universal compatibility. Donor-sized generic runtime is a diagnostic
or transport baseline and does not satisfy the conditional-capacity thesis.

## Operation and continuation

No owned run live. No T4 allocated; communicate exact reason/budget/stops before
allocation. Preserve original engine, foreign files/processes and all consumed
sources/bindings/results. Routine Graphify disabled; complete audits even quality
failure; exact PID/create/session authority. No Luna or delegated monitoring.
Hours-long job: announce bounded frozen invocation, launch, user-requested PAUSE,
stop until human resumes. [INDEX](INDEX.md) holds the next exact action.
