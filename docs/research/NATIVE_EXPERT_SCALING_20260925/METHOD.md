# Pretrained-to-native conditional-capacity method

13 October2026. **ACTIVE/INCOMPLETE**. This is the reproducible procedure under
construction. It currently produces a failed chatbot and validated analysis/
conversion components, not a capacity-preserving universal converter.
[Full trained3072 channels/state96](FULL_CHANNEL_NATIVE_RESULT_20261013.md)
now IMPLEMENTED/INDEPENDENTLY VERIFIED for all48 histories/all3 sites0/12/23.
Every CPU state timestep/all final states,all277,057,536 raw F32 score bits,
normalized vectors/source metrics/flags/hash controls qualified. Ownnorm DEV
mean.076342/.087177/.024904/all local gatesPASS;source-denominator controls
FAIL0/12,retained. Same-component batch1 scan~59us,total~1300us versusold384
~1249us across separate runs;8x state count does not impose8x whole cost here.
Keep wider trained functions as a shared-core candidate;full source hidden/gen/
conv/readout costs and all-layer/own-history quality remain missing.
[Gate input-operator geometry](GATE_INPUT_OPERATOR_RESULT_20261013.md) now
IMPLEMENTED/INDEPENDENTLY VERIFIED:all3 raw3072x2048 gate operators rank2048,
conditions27/71/55,blockGram/direct rectangular SVD qualification. No input
bit-supervision claim from matrix geometry alone.
[Approximate source input recovery](GATE_INPUT_RECOVERY_RESULT_20261013.md) now
IMPLEMENTED/INDEPENDENTLY VERIFIED:fixed H64 least-squares/BF16 candidate,actual
BF16 MuP sections/explicit rounded source projection/conv4/SiLU/delta. All144
histories/all5 fields pass frozen whole-history1% consistency,including independently
captured x/B/C/delta;all453,465,264 field words/138,528,768 H64+candidate words
per type/18,874,368 inverse coefficients qualified by rectangular QR and separate
replay. No actual source input bits or true-input error bound. Worst centered
site23 B1.381%/x1.070% retained. Approximate inputs may initialize calibration;
temporal propagation must qualify before they become usable fitting proxies;
the completed temporal counterexample below prevents that admission.
[Temporal propagated-input gate](RECOVERED_INPUT_TEMPORAL_RESULT_20261013.md)
now COMPLETE/full independent audit144 histories. Own-output1%PASSall144;
final-state1%FAILone DEV Numina/site23 at1.125798%. Donor-relative own-norm
local budgetsPASS/all3,mean.076416/.087351/.025683. Whole-operator numericFAIL
retains DIFFERENT FIT Smol/source-denom BF16 smooth1.4098e-4>1e-4;all normalized
BF16 vectors exact/all277,057,536 raw F32 score bits exact. All primary own-output
and state/projection controlsPASS. Coordinate-complete rounding witness111
adjacent crossings/10 cancellation discrepancies adjudicates without waiving.
All144 histories/metrics/flags/hashes/audits completed,not a failed subset.
Inferred inputs are NOT admitted for temporal calibration. Independent F64
forcing decomposition/survival-sum audit2,064,384 final state/component words
PASS:delta forcing largest ordered term,decay partially cancels;native arithmetic
error~5e-7 original-state units much smaller than1.125798%. Not unique causal
percentages or proof of unlimited future instability;whole numeric failure remains.
Next [TRUE generator inputs](TRUE_GENERATOR_INPUT_CAPTURE_NEXT_20261013.md)
PROPOSED/UNIMPLEMENTED/UNBOUND. Same48 source prefills capture BF16 H at ALL24
layers/full7 continuity fields at0/12/23,then actual-source replay qualifies
before [FIT response-rank compression](SOURCE_FUNCTION_RANK_NEXT_20261013.md).
Proposed local RTX3060 family1800s/OS12GiB/GPU10GiB allocated11GiB reserved/new
E6GiB;no source/GPU/T4 allocated now. All-layer input capture avoids future
source forwards solely for missing input data;it does not prove all-layer
operator or own-history capacity. Compact causal input/output/conditional
ternary functions still missing;offline inverse is not runtime encoding.
Numerical package alias custody gap in prior native input enumeration is explicit
in its report;new preparations bind resolved ABI/BLAS files directly.
[Actual C mixed384x96 operator](COMPOSED_OPERATOR_MIXED_RESULT_20261013.md) now
IMPLEMENTED/INDEPENDENTLY VERIFIED for all48 source-injected site23 histories,
including all rawF32 score bits (92,352,512) with separate scalar FMA reference.
F32 state/F64 decode-gate-RMS/BF16/AVX F32 output boundary;local gates PASS,
DEVmean.068813/worst.096949. Original V1 normalized-input failure fully retained;
no threshold relaxation/subset. Source generators/conv/hidden inputs still supplied.
Same qualified component/two batch1 passes:scan~10us,normalize+out~1157us,
total supplied-operand~1248us;source input generator kernel separately~2535us.
These are measured component costs,not complete layer/clean-system/expert-DRAM/
whole-chatbot speed. [Next state width/source-function investigation](SOURCE_FUNCTION_COST_AND_STATE_WIDTH_NEXT_20261013.md)
PROPOSED/UNIMPLEMENTED/UNBOUND. Preserve trained functions where actual active
cost permits;compact CAUSAL hidden/generator/readout transfer remains missing.
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
[All-label probability gap](ALL_LABEL_NATIVE_PROBABILITY_GAP_PROTOCOL_20261012.md)
now IMPLEMENTED/FROZEN;capture10400 CLOSED0 after1391.396s,budget36 completed/
partial37 updates. All8808 final native codes/source q computed;provisional
DEVcaseKL.0483465/dis2.3003%,all original budgetsFAIL,lower0. Full four stored-
point/native independent audit13729 COMPLETE/PASS:all35232 point rows/all577249896
native logits bit exact/direct source q;no image-optimum/convergence admission.
[Actual result](ALL_LABEL_NATIVE_PROBABILITY_GAP_RESULT_20261012.md).
[Entropy-mixture dual witnesses](DUAL_ENTROPY_MIXTURE_RESULT_20261012.md) now
IMPLEMENTED/INDEPENDENTLY VERIFIED:all140928 entropy/direct moment/support rows,
ten positive lowers below budgets;no relevant reader obstruction proved. No
causal/model-quality admission. [2048 source-head moments](SOURCE_HEAD_FEATURE_MOMENTS_RESULT_20261012.md)
now IMPLEMENTED/INDEPENDENTLY VERIFIED:all8808 source rows and three decoder
controls, full feature/RMS derivatives;144.31MB moments. Allows a source-derived
decoder to change while retaining full categorical supervision;fixed257-moments
cannot cover that change. Native F32 coefficient rounding remains separate.
[Recurrent-only channels/full skip](RECURRENT_ONLY_CHANNEL_SKIP_RESULT_20261012.md)
now IMPLEMENTED/INDEPENDENTLY VERIFIED:all144 records/1,108,230,144 controlled
BF16 output words/energies/cross terms;relative output difference0. Removing
projected skip loss improves0/23 markedly,12 modestly;all local qualityFAIL.
Original F32 precision bridge252/576FAIL retained,not precision admission.
Full generators/gates/conv buffers retained;no compact causal/native model.
[FIT recurrent-output bases](RECURRENT_OUTPUT_BASES_RESULT_20261012.md)
now IMPLEMENTED/INDEPENDENTLY VERIFIED:FIT y-D*x covariance with unchanged GM,
all288 new packed scans/51,948,288 values checked by full CPU state256 recurrence,
all1,108,230,144 output values and bases/gates/hashes verified. Site23 adaptive
ownnorm mean.06357/worst.08701,all local gates PASS/both norm modes. Sites0/12
and global local gatesFAIL;source nonlinear work retained,no native model.
[Coupled sensitivity](COUPLED_RECURRENCE_SENSITIVITY_RESULT_20261012.md)
now IMPLEMENTED/INDEPENDENTLY VERIFIED:all554,115,072 finite/J vectors,raw/head
statistics/FIT Grams/proxies/450 controls. Site12 adaptive finite remainder
.30044FIT/.33078DEV selects a finite-output objective;correcting mean Gram alone
not selected. Source-injected/F64 smooth operator,no BF16/native derivative.
[Finite backward/retraction/one bounded FIT update](FINITE_RECURRENCE_GRADIENT_RESULT_20261012.md)
now IMPLEMENTED/INDEPENDENTLY VERIFIED:24 full CPU autograd cases/all GY/parameter
gradients,independent tangent,all FD/candidate/selection/parameter bits/all48
actual compact scans. Alpha.0003 FITloss.537006->.533005,DEV ownnorm.792727->
.789261;all local qualityFAIL. Original algebra/control/metadata faults retained;
fine derivative qualification separate,original coarse flags false. Next
[exact coordinate whitening](WHITENED_RECURRENCE_PREFLIGHT_RESULT_20261012.md)
now IMPLEMENTED/INDEPENDENTLY VERIFIED:all physical/white outputs initially
identical~1e-14,dual autograd/chain/SPD/inverse/tangents/full matched-output curves.
Whitened best FIT.520252 versusreference.530466 fromsame.533005 point;no capacity
change,no installed update/new scan in that preflight.
[Fixed candidate qualification](WHITENED_CANDIDATE_RESULT_20261012.md) now
IMPLEMENTED/INDEPENDENTLY VERIFIED:all48 ideal outputs/new actual384 state256
scans/all BF16 outputs, every CPU timestep. Fixed FIT alpha.1;no DEV reselection.
All48 ideal losses improve,DEV actual mean.775702 versus.789261;all local quality
gatesFAIL. Numerical transport verified,quality repair remains research.
[Fixed site23 adaptive384/dual96 composition](RECURRENT_AXIS_COMPOSITION_RESULT_20261013.md)
now IMPLEMENTED/INDEPENDENTLY VERIFIED:all48 actualF32 scans,every CPU state
timestep/all six BF16 outputs per case,projection/commutation/energy controls.
Joint DEV ownnorm mean.068813/worst.096949,all local gates PASS/both modes;
21.33x smaller recurrent state. This local component retains donor generators,
3584 conv channels,3072 gates/fullskip/norm/out and source hidden inputs.
Scalar source head decay permits axis composition;it does not close the compact
causal generator or commute mixed maps through source SiLU/dynamic gates.
The [native operator qualification/cost pilot](COMPOSED_OPERATOR_NATIVE_COST_NEXT_20261013.md)
is subsequently COMPLETE as linked above. Input-generator GEMV measured
separately;full source convolution/hidden generation is not an autonomous
native path. No21.33x whole-model speed admission.
[Causal transfer boundary](CAUSAL_TRANSFER_AFTER_READOUT_GATE_20261012.md) states
history-to-code categorical supervision and full-state closure/future-information
requirements;oracle code regression is not a pretrained-to-engine converter.
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
