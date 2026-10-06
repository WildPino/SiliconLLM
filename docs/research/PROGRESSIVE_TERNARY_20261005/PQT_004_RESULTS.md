# PQT-004: broader calibration helps but does not preserve the donor

5 October 2026. **Completed, independent audit passed, no arm promoted.**
[Prospective protocol](PQT_004_PROTOCOL.md), scientific source
`a4ec1777a0ecbeaebf3a2e3a73633a23a04f31fa`, bundle/dispatch `811ed15`.
Private acct1 kernel 137231297 v1, pinned qualified T4x2 image. Acknowledged
20:15:48 UTC, COMPLETE observed 20:27:10 UTC. Install/experiment/audit exited
zero. No numerical failure, retry, repair or gate relaxation.

## Matched results and decisions

Same original Qwen last down projection [896,4864], frozen core/readout.
C is the original sixteen contiguous train windows; S sixteen distributed
windows. Same 2048 calibration positions, 256 actual head updates, group-64
codes/scales/bias. All arms evaluated on the same eight new stratified test
windows and eight new prompts. No content/error-based context selection.

| Arm | Layer normalized SSE | Mean KL | NLL increase | Source argmax agreement | Rollout position agreement | Exact traces |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PR-B-C | .06384594 | .17096402 | .14844935 | 79.5898% | 20.3125% | 0/8 |
| HK-C | .08467592 | .14745014 | .12867199 | 80.0781% | 24.2188% | 0/8 |
| PR-B-S | .04883343 | .13856377 | .10963192 | 80.8594% | 19.9219% | 0/8 |
| HK-S | .06421239 | .12531357 | .09926664 | 80.0781% | 19.9219% | 0/8 |
| I4 | .00835301 | .01628614 | .02593182 | 89.7461% | 12.5000% | 0/8 |

[Independent adjudication](PQT_004_ADJUDICATION.json) confirms both prospective
coverage signals: PR-B-S lowers KL 18.9515% versus PR-B-C, improves argmax
agreement 1.2695 percentage points and reduces NLL increase .0388174.
HK-S lowers KL 15.0130% versus HK-C, leaves argmax agreement unchanged and
reduces NLL increase .0294054. All comparisons use identical evaluation contexts.
This is a bounded coverage benefit, not statistical confirmation across seeds,
donors or tasks. Neither matched method improves rollout positional agreement;
HK-S is 4.2969 percentage points below HK-C.

Every arm fails every absolute quality gate: >=99% argmax agreement, KL <=.01,
NLL increase <=.01, >=95% rollout agreement and >=7/8 exact continuations.
All storage/fit-time gates pass. Best predictive ternary PR-B-S still disagrees
on 196/1024 source choices. I4 has much better predictive fidelity but also
fails preservation. Its lower greedy positional match on these eight prompts
does not establish lower semantic/generative competence; all exact traces fail.
No arm is promoted. Wider calibration alone did not resolve preservation.

## Final calibration evidence

These metrics use the **final frozen representations**, not changing online
batch losses. All 2048 positions per arm were independently checked in FP64.

| Arm | Fitted-set mean KL | Fitted-set argmax agreement |
| --- | ---: | ---: |
| PR-B-C | .02186336 | 91.1621% |
| HK-C | .01743219 | 93.2617% |
| PR-B-S | .02373891 | 90.4785% |
| HK-S | .02114194 | 92.4805% |

Calibration fidelity is substantially better than held-out fidelity in both
coverage conditions, but even calibration remains below the 99% argmax reference. This
does not identify a unique cause or justify selecting more updates on the
observed evaluation. It weakens a claim that contiguous coverage was the sole
obstacle. Direct head KL fitting again improves likelihood while worsening
layer SSE relative to its parent. Local reconstruction remains an inadequate
standalone selection criterion.

## Actual data identities and consumed contexts

Pinned corpus tokenization: 2,517,233 train tokens / 19,513 complete windows;
298,939 test tokens / 2,317 complete windows. Sixteen S calibration IDs:
859,1572,2523,4044,5226,6272,8328,9273,10750,11874,12984,13612,15696,17073,
17078,19351. Test IDs: **92,396,690,1053,1279,1525,1754,2306**.
Prompt IDs: **251,449,664,1165,1293,1565,1854,2262**. These sixteen test windows
are now consumed in addition to 0..39. Preserve the explicit ID ledger; do not
assume they form a contiguous range. Selected tokens and IDs were rederived
independently by retokenizing the pinned corpus and applying the frozen hashes.
Public WikiText is not fresh project-wide or pretraining-independent validation.

## Costs, audit and repeatability

Both HK arms verify Adam state steps [256,256,256]; no core gradients.
Measured fitting operations 29.5742s, excluding capture/metric/serialization
costs counted in process elapsed time. Installation 16.5631s, experiment
192.5326s, audit 439.1407s. Main peak GPU allocation 5,617,715,200 bytes;
final RSS 4,385,017,856 bytes. Ternary+bias payload 1,365,504 bytes (15.6661%
of this projection's FP16 bytes); I4 2,451,456 bytes (28.125%). File headers
retained separately. No whole-model or native execution claim.

Independent cuda:1 audit imports no fitting modules, checks decoded outputs,
all metric aggregates and complete training permutations, and rederives data
selection. Source evaluation projection RMS 2.47293e-7, spread calibration
projection RMS 2.49305e-7. Maximum audited evaluation readout difference
6.98664e-6, within 2e-5 tolerance. All four ternary calibration representations
replayed through the independently loaded full nonlinear model; surrogate-state
RMS <=8.15e-7 and independent FP64 RMSNorm RMS <=7.92e-7. Full calibration
readouts and point aggregation pass. Every evaluation state matches independent
replay exactly; all 48 complete greedy traces / 1536 tokens match exactly.

[Control reproduction](PQT_004_CONTROL_REPRODUCTION.json): thirteen original,
calibration/schedule and C/I4 packed/scales/bias files reproduce PQT-003 exactly,
including HK-C versus HK's final learned bytes. This verifies the matched C
procedure; evaluation samples changed, so cross-run metric differences are not
method effects. [Retention](PQT_004_RETENTION.json): all 98 files / 264,244,564
bytes retrieved and hash/size checked, fifteen actual server embedded files
match the frozen source commit. Small raw files are in `pqt_004_evidence/`;
all arrays in `results/progressive_ternary/PQT-004/remote_001/`. Do not repush
or retrieve this completed reference again. Reaggregate with
`scripts/progressive_ternary/adjudicate_pqt004.py` and a new output path.

[Post-run quota](accounts_post_pqt_004.json): acct1 used .411526111h,
29.588473889h unreserved, zero reservation; acct2/3 unused, 30h each.
No local GPU/native benchmark, source-checkout/environment mutation, merge or
push. Source/input qualification used only the owned CPU environment.

## Research decision and next scope

Coverage improves the tested source-distribution approximation at a fixed
budget, but fails preservation on a broader diagnostic split. Keep this result
as a scoped negative, not a universal impossibility claim. Do not continue
adjacent Qwen tuning solely to improve observed validation numbers.

Next: freeze an original Switch expert-function comparison using admitted F32
wi/wo weights and explicitly consumed input distribution. Compute original F32
function references rather than inherit I8 outputs. Distinguish calibration,
independent algebra/codec checks, fresh robustness verification and natural
model-quality evidence. Prepare an actual packed CPU artifact, then coordinate
bounded native timing on that same artifact. Complete-model predictive/generative
preservation and native usefulness remain required; the goal stays active.
