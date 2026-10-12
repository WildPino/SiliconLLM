# Conditional raw charts and exhaustive choice: numeric PASS, quality FAIL

12 October2026. Both experiments and complete independent audits CLOSED exit0.
Full objective ACTIVE/INCOMPLETE; no useful pretrained-to-engine admission.

## What was implemented

[Frozen chart protocol](CONDITIONAL_RAW_RESIDUAL_CHARTS_PROTOCOL_20261012.md),
[tool](../../../benchmarks/native_expert_scaling/conditional_raw_residual_charts.py).
Three actual source raw residual boundaries0/12/24, FIT24 histories11,148 states.
4/16 fixed slots, up to256 active coordinates. Equal-case source-RMS weighted
raw centers, deterministic farthest seeds,32 Lloyd rounds and fixed source reading
M from the prior independently audited global256 study. No labels or DEV fit.

Each chart's moment about its frozen final center gives source-sensitive distinct
V/W bases. Full C and support spectra, covariances, centers/assignments/33 histories,
all288 case encodings, source observables/head and544 affine skip pairs retained.
Exact norm uses mu^T mu +2(V^T mu)^T z+z^T V^T V z. Readers require A mu bias,
and skips require W_new^T(mu_old-mu_new) offset. No diagonal norm substitution.

All slots occupied; some supports low rank. At0,K4 active[3,23,256,86]; K16
active1..256 (min24/max2638 FIT positions). At12,K16 one slot has1 FIT position/
1 active coordinate; others include a29-coordinate slot and14 full256. At24,
K4 one87-coordinate slot, K16 one80/one179 and14 full256. Inactive columns exactly0;
no ridge, duplicated experts, widened active code or adaptive K after DEV.

The nearest source-center choice is an ORACLE ACCESS diagnostic (full h and M),
not the mathematically best chart choice and not a deployed compact router.

## All-history local reading measurements

DEV24/full11,399 positions at each boundary/K; normalized readers under fixed M.
Oracle denominator is actual source RMS; own is the affine reconstruction RMS.
All existing mean10%/worst20% proxy gates FAIL for every configuration/mode.

| Boundary | Global256 oracle/own | K4 oracle/own | K16 oracle/own |
|---|---:|---:|---:|
| 0 | 27.26% /41.37% | 26.60% /39.60% | 25.46% /39.02% |
| 12 | 40.76% /47.59% | 40.54% /45.44% | 46.83% /50.62% |
| 24 | 18.81% /23.23% | 19.08% /22.55% | 25.19% /26.65% |

More FIT-local bases need not improve unseen readers. This concerns the tested
global metric/weighted clustering/frozen coordinates, not a theorem about all
conditional state representations. Eight actual source projections at0/12 use
the same144 first/middle/last HISTORY positions; all packets/128 metrics audited.

## Actual source output probabilities, identical DEV72 LABEL states

Actual source BF16 coefficients decoded to F64, full65537 vocabulary. No cached
generation-logit equality or new generative task run implied.

| Choice | MeanKL oracle/own | WorstKL oracle/own | Source argmax wrong /72 |
|---|---:|---:|---:|
| Prior global256 | 2.4291 /2.7278 | 9.4617 /11.4366 | 26 |
| K4 nearest center | 2.3411 /2.6020 | 9.7703 /11.4985 | 27 |
| K16 nearest center | 2.2099 /2.3762 | 9.8405 /10.4377 | 27 |
| K4 BEST exact-KL candidate | 2.1266 /2.3242 | 8.7751 /10.0936 | 27 |
| K16 BEST exact-KL candidate | 1.6482 /1.7233 | 7.7395 /7.9783 | 25 |

All probability gates meanKL<=.01,worst<=.1,disagreement<=.01 FAIL.
Bias-free head argmax identical under positive oracle/own normalization for each
fixed reconstructed state; radius can improve confidence but cannot fix these IDs.

### Separate every-decoder test isolates the selector limitation

[Frozen oracle protocol](CONDITIONAL_CHART_READOUT_ORACLE_PROTOCOL_20261012.md),
[tool](../../../benchmarks/native_expert_scaling/conditional_chart_readout_oracle.py).
No basis/center update. Evaluate EVERY fixed final chart on EVERY source state,
with its fixed source encoder W and decoder V;20*72*2=2880 full-V candidate rows.
Choose minimum ACTUAL full categorical KL using teacher probabilities, and also
check existence of any chart retaining the source argmax. Full tensors retained.

K4:27 source decisions unavailable in ANY candidate. K16:25 unavailable in ANY
candidate. Thus a new selector over these frozen source encodings/decodings can
repair at most2 of the27 nearest-center errors at K16. Best-KL choice attains
those2 repairs here, but still fails badly. It needs full source states/probabilities
and computes every chart, so its selection is not a CPU routing implementation.

This bounds ONLY the current frozen predictions for THESE72 states. Arbitrarily
learned coordinates in the same decoder image, new directions or new chart
definitions need not share it. No general256 capacity or conditional-bank ceiling
proved. The complete classifier/per-state outcomes are stored, not just counts.

## Cost and decision

Increasing stored redundancy alone in the current quadratic fitting criterion
does not establish useful capacity. Do not invest in a router for these unchanged
fixed encoders expecting to recover the missing25 decisions. Direction/encoding
correction with finite categorical objective is now required. Next:
[exact categorical residual gradient plus one bounded correction](EXACT_CATEGORICAL_RESIDUAL_CORRECTION_NEXT_20261012.md)
PROPOSED/UNIMPLEMENTED. Fix/check full RMS/V/W/center Jacobian on FIT states,
then qualify an immutable FIT-selected candidate on DEV; no long dose yet.
This differs from the failed native learner's fresh core/global-P geometry.

Dense shape-derived estimates, worst-case256 active, F32: selected internal
input readers8240*(256+1) coefficients=8,470,720B/chart; selected final head
65537*(256+1)=67,372,036B/chart including required bias. K16 final heads would
store1,077,952,576B; active head remains16,777,472 MAC plus bias additions.
G/cross/center norm263,172B/chart; each skip matrix+offset263,168B/pair.
All544 representative pair maps143,163,392B F32. These are potential compiled
map counts, not produced native weights, physical DRAM traffic or speed tests.
Original source SSM3072/state256 feature costs remain unresolved. An index carries
2/4 bits but its computation/source-state information is not free. No useful n,
multiple-family conversion or same-artifact>=50 evidence added by these screens.

## Provenance and closure

Chart source/protocol d7df351; [binding](conditional_raw_residual_charts_binding_20261012.json)
166 inputs/SHA0042e9af9c6f7f9aa9584db28c6bfeb869943ec0fbcb9f7b0b8a754bc69a8d08;
prepare3.609s/exit0. [Invocation](conditional_raw_residual_charts_launch_20261012.json)
bc74160. Producer14276 CLOSED0/PID27800 creation1791805290.022833,
418.079s/result,418.297s terminal,OS1,196,109,824B;367 files6,910,203,862B.
[Result](conditional_raw_residual_charts_result_20261012.json)
SHA4fd44e9b032dc6c8bf1c4975f5d06c9f3100a0d3839ed027353deac14a1249e4;
terminal94aa8bc2c13beb5007b90c8fb55ae22c4604aacf004d7603a631d829faaf2976.

[Chart audit](../../../benchmarks/native_expert_scaling/conditional_raw_residual_charts_stored_audit.py),
[protocol](CONDITIONAL_RAW_RESIDUAL_CHARTS_STORED_AUDIT_PROTOCOL_20261012.md),
5bc9454/[invocation](conditional_raw_residual_charts_stored_audit_launch_20261012.json)
8678f86. Session19703 CLOSED0/PID14860 creation1791805815.2330475;
235.546s/OS1,503,277,056B. [Adjudication](conditional_raw_residual_charts_stored_adjudication_20261012.json)
SHAa06e9d8927dc3e49fe6ead3f8ff693d58665b24ab8d6d8a3813eaeb008b4611e.
All60 covariance/spectral/map charts,198 clustering states,288 full histories,
128 source projection metrics,288 probability rows and544 affine skips PASS.
Independent metric factor distances/weighted updates/covariance accumulation,
source-width head sums/direct decoded norm/quadratic expansion. Rounding-ambiguous
nearest ID rows0; max relative2.6401e-13/scalar1.3797e-11. No chart eigensolver refit.

Oracle source/protocol b04df52; [binding](conditional_chart_readout_oracle_binding_20261012.json)
SHA34de2d641d2bb29ecd50d00f92c4ebcdd7cabaf800be851cd9e59ef00131b7c1;
prepare4.625s. [Invocation](conditional_chart_readout_oracle_launch_20261012.json)
9822e71. Producer25395 CLOSED0,123.735s/OS1,577,476,096B;2 files1,535,986,960B.
[Result](conditional_chart_readout_oracle_result_20261012.json)
SHA34938b5d9b09f97f03307b40dc216f3f59d5b711216c562af604eff01390b8dd;
terminal4bf1d9e8d457dd8f8201eb5b7388defc7fea0a816aaa2228f43fbd8244f39cb2.

[Oracle audit](../../../benchmarks/native_expert_scaling/conditional_chart_readout_oracle_stored_audit.py),
[protocol](CONDITIONAL_CHART_READOUT_ORACLE_STORED_AUDIT_PROTOCOL_20261012.md),
freeze9822e71/[invocation](conditional_chart_readout_oracle_stored_audit_launch_20261012.json)
a437f1a. Session25362 CLOSED0; full20 decoded maps/all2880 full-V candidate
probabilities/all288 best choices and argmax-existence rows audited PASS.
See [complete adjudication](conditional_chart_readout_oracle_stored_adjudication_20261012.json)
for hashes/peak/time/deltas and [session closure](conditional_charts_session_closure_20261012.json)
for exact PIDs/creation/closed handles.

All resource limits and full before/after hashes PASS. No first execution fault
in these families. All original inputs/engine/foreign changes/processes retained.
Outputs ordered on D (space sufficient); no E copy/T4/GPU/model training/native
run, no RESERVED/longalign query. No own job remains live; full goal incomplete.
