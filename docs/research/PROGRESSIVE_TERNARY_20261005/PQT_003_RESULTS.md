# PQT-003: direct head fitting improves likelihood but fails preservation

5 October 2026. **Completed, independently audited, no arm promoted.**
The [prospective protocol](PQT_003_PROTOCOL.md) and scientific sources were
frozen at `ddab45308a431994af91a5c9c43bc9c4e3920fa7`; bundle/dispatch at
`d808934`. Private acct1 kernel 137228525, version 1, qualified pinned T4x2
image. Acknowledged 19:56:01 UTC, COMPLETE observed 20:01:18 UTC.
All installation/experiment/audit return codes are zero. No numerical repair.

## Absolute decision and all arms

One original Qwen last FFN down projection [896,4864], unchanged core/readout.
Calibration first 16 train windows; new test windows 24..31, 1024 predictions;
new rollout prompts from windows 32..39, 8 greedy continuations of 32 tokens.
All prior windows are consumed evidence. Public WikiText remains diagnostic.

| Arm | Layer normalized SSE | Mean KL | NLL increase | Source argmax agreement | Rollout position agreement | Exact traces |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PR-B | .05946294 | .14092070 | .11477722 | 78.5156% | 28.5156% | 1/8 |
| I4 | .00929668 | .01890426 | .01268058 | 89.5508% | 44.1406% | 1/8 |
| HK | .07935883 | .11517831 | .09449772 | 80.6641% | 27.7344% | 1/8 |
| HM | .07881803 | .11806287 | .09240773 | 79.4922% | 30.4688% | 1/8 |

Every arm fails every absolute quality gate: prediction agreement >=99%,
KL <=.01, NLL increase <=.01, rollout agreement >=95%, >=7/8 exact traces.
All storage/time gates pass. [Independent adjudication](PQT_003_ADJUDICATION.json)
reaggregates per-position records and raw tokens, reports per-window/prompt
results and int4 comparisons. Best predictive ternary arm HK disagrees on
198/1024 source decisions. Its packed payload is 55.70% of I4, with materially
worse quality. Int4 also fails the source-preservation gate on this split.

Head KL fitting improves KL/NLL and argmax agreement against PR-B, despite
worse layer SSE. The margin arm improves rollout positional agreement modestly
and has lower NLL increase than HK, without better source argmax agreement.
No metric supports deployment or complete-model preservation. Greedy string
divergence does not establish lost semantic/factual capability or reject every
ternary representation. This is a negative result for the frozen procedure,
calibration, budget and single-projection intervention.

## Applied fitting, cost and calibration limitation

HK/HM each applied 256 Adam updates; each of the three parameter state counters
equals 256. Frozen core: 494,032,768 parameters, no gradients. Optimized latent
codes/scales/bias: 4,427,136 fitting parameters; continuous codes and optimizer
state are discarded from the inference representation. Hard code changes:
28,003 HK (0.6425%), 23,042 HM (0.5287%). Both arms use exact hard ternary
forward weights with an identity surrogate derivative, not continuous inference
weights. Final group scales and 896-value bias are included in the payload.

Per-arm head fitting: HK 6.6043s, HM 6.5123s; entire fitting phase including
initial progressive/refinement and I4 21.7040s. Installation 17.1645s,
experiment 167.4522s, audit 112.1091s. Peak GPU allocation 6,841,972,736 bytes;
final main-process RSS 4,285,034,496 bytes. GPU processes are sequential on
separate devices, not pooled memory. Ternary+bias payload 1,365,504 bytes,
15.6661% of this projection's 8,716,288 FP16 bytes; I4 2,451,456 bytes, 28.125%.
Headers are retained separately; this is not whole-model compression or speed.

Post hoc [trace diagnostic](PQT_003_DIAGNOSTIC.json): first/last epoch mean
online calibration KL .02022/.01740 HK, .02052/.01643 HM. These are batch losses
on changing pre-update models, **not final frozen calibration metrics** and
not an independent validation set. Their distance from held-out KL motivates
a prospective calibration-coverage test; it does not prove a particular
overfitting mechanism. Repeating longer fitting alone would not resolve this
uncertainty. No evaluation-based hyperparameter/checkpoint choice was made.

## Independent verification and retained evidence

Independent cuda:1 process imports no fitting module. Original source projection
FP64 reconstruction relative RMS 2.53365e-7; maximum audited KL/NLL readout
difference 9.12036e-6, within frozen 2e-5 tolerance. All 1280 stored generation
state decisions checked, no argmax discrepancy. Full independently reloaded
nonlinear model replay matches every source/candidate evaluation state exactly
and all 40 greedy traces / 1280 generated tokens exactly. HK/HM surrogate
calibration states match actual full-model forwards within 8.15e-7 relative RMS;
independent FP64 RMSNorm reconstruction within 7.95e-7. Full calibration epoch
permutations and actual optimizer update traces pass. Source surrogate admission
RMS 8.45495e-7. These checks strengthen apparatus validity, not capacity claims.

Source weight, calibration inputs/outputs/tokens, PR-B packed codes/scales/bias
and I4 codes/scales reproduce PQT-002 byte-for-byte: nine control identities.
All 74 fetched files / 162,806,290 bytes are hash/size checked; 34 small raw
files / 781,006 bytes retained in `pqt_003_evidence/`. All twelve actual remote
embedded files match the frozen scientific commit. See
[retention](PQT_003_RETENTION.json), `pqt_003_fetch.json` and raw audit.
Complete arrays: `results/progressive_ternary/PQT-003/remote_001/`.
Reproduce adjudication using `scripts/progressive_ternary/adjudicate_pqt003.py`
with this directory and a new output path. Do not repush the completed reference.

[Post-run account snapshot](accounts_post_pqt_003.json): acct1 used .230226667h,
29.769773333h unreserved, zero reservation; acct2/3 each 30h unreserved, unused.
Counters are account quota observations, not proof of future idleness/reset.
No local GPU/native timing, main checkout/environment mutation, push or merge.

## Next evidence boundary

Overall goal remains active and incomplete. Original Switch expert tensors are
now independently admitted by PQT-SRC-001; useful expert-function preservation,
routes, whole-model generalization and packed native cost still require tests.
Next bounded Qwen question: change calibration coverage at a fixed token/update
budget, with frozen sampling and fresh evaluation/prompt windows before launch.
Separately freeze real-expert activation/reference roles using the original
float weights; inherited captures are consumed development evidence. Native
timing requires coordinated exclusive local resources, not merely an idle sample.
