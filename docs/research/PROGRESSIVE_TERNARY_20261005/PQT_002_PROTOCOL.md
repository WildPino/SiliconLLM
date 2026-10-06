# PQT-002: scale adaptation, explicit offset and int4 comparison

Prospective, 5 October 2026; no PQT-002 numerical observation. PQT-001-R1
demonstrated relative gains but only ~76% source argmax agreement. Preserve its
consumed evaluation; do not use a relative gate as proof of behavior preservation.

## Question and frozen arms

Can adapting existing scales and a cheap, explicitly counted calibration offset
restore predictive/generative fidelity at the ternary storage budget, compared
with a relevant int4 baseline? Ten arms, all reported: D, I4, DR, PR, DR-S, PR-S,
DR-B, PR-B, DR-SB, PR-SB. D/DR/PR use the original group-64 scales and procedures
of PQT-001-R1. S adds two fixed cycles: solve row/group scale coefficients then
one discrete coordinate pass. B adds one output offset estimated from calibration
after weight fitting. SB combines S and B. No hyperparameter selection on evaluation.

Same immutable donor, model/dataset revisions, SHA-256 identities, original
FP32 source values, SDPA and last FFN down projection [896,4864] as PQT-001-R1.
Technical inputs are fully specified in that protocol and reverified per run.
Runtime pins are those repaired by PQT-001-R1, including loaded NumPy 2.1.3.

## Data roles and generation

Keep the exact original training calibration: first 16 nonoverlapping 129-token
WikiText train windows, fit first 128 positions, 2048 activations. Evaluation:
test windows indexed 8..15, 1024 first-128-position predictions with next-token
labels. Original test windows 0..7 are consumed and excluded. Generation:
first 32 tokens of test windows 16..23, eight fixed prompts, at most 32 new
greedy tokens, stop on EOS, use each candidate's own generated tokens and KV
cache. No sampling. Keep complete source/candidate token IDs, decoded traces,
EOS status and lengths. Compare token positions up to the longer continuation,
missing tokens count as disagreement; also exact continuation count.
This tests changed-state rollout fidelity, not factual/semantic task competence.
Public WikiText is not fresh project-wide or pretraining-independent validation.

## Representation and fitting

All ternary weight codes are {-1,0,+1}, four per byte, group size 64, FP32 scales.
S: for each output row form group features Z[n,g]=sum_j_in_g X[n,j]*q[row,j].
Solve (Z^T Z/N + lambda I) alpha = Z^T y/N + lambda alpha_old, with
lambda=.01*mean(diag(Z^T Z/N)), panels of 32 output rows. Negative solved
scales are made positive by reversing corresponding group codes, preserving
the exact represented weight. Two solve/refine cycles, one code pass each;
record solve rows, cycles and calibration SSE. Reject increases exceeding
1e-5 relative +1e-12 absolute. No float weight residual or rank expansion.
B: bias=mean_calibration(y-X Q^T), 896 FP32 values (3584 bytes), added to the
projection output. This is an explicit mixed representation, not a pure
ternary model. No evaluation residual is used to fit this offset.

I4: signed levels -7..+7, two 4-bit two's-complement codes per byte, -8 reserved.
Per-row/group scale selected by original-weight SSE among factors
{.70,.80,.90,.95,1.0}*group_max_abs/7, earliest factor wins ties; nearest-even
rounding clipped to [-7,7]. This is a direct int4 baseline, not optimal int4 PTQ.
Payloads: ternary 1,361,920 bytes, ternary+B 1,365,504 bytes, int4 2,451,456
bytes. Include file headers separately. One converted projection in a mixed
model; no whole-model compression/speed claim.

## Metrics and fixed decisions

Report all-position layer normalized SSE, source→candidate KL, next-token NLL
increase, source argmax agreement; per-window metrics and raw per-position data.
Prediction preservation gate: agreement >=99%, mean KL <=.01 nats,
NLL increase <=.01 nats/token. Generation preservation: token-position agreement
>=95% pooled across eight prompts and at least 7/8 exact continuations.
Capacity/cost: ternary payload <=35% of projection FP16 bytes; entire fitting
phase <=45 min. A ternary arm must meet all these absolute gates to justify a
next expert/native test. Report I4 against the same quality gates. Separately
compare each ternary arm with I4; do not silently trade quality for smaller size.
No tested arm establishes full-model/expert feasibility without subsequent work.

## Qualification and audit

Before donor access, rerun existing independent synthetic controls plus scale
solve versus scalar inverse, offset versus scalar mean, int4 golden bytes and
FP64 cancellation controls. Freeze every representation before evaluation.
Retain scale/code/bias bytes, captures, states, token arrays, generation traces,
model/input identities, package/hardware admission, events and first failures.

Independent process on cuda:1 imports no fitting module. Decode every payload,
reload source weights/readout, reconstruct all projection outputs including
offset, check payload and metric aggregation. Check FP64 full-vocabulary readout
on first four positions of each evaluation window (32 positions), same 2e-5
NLL/KL and 1e-4 near-tie argmax tolerances as PQT-001. Source/candidate layer
reconstruction relative RMS <=1e-5. Check dtype and cancellation arithmetic
explicitly; record FP64 control result rather than infer precision from timing.
Generation: record last-normalized states at every step; independently recompute
every greedy next-token choice from its stored state/readout, allowing only
the same 1e-4 FP64 near ties. This audits token choice, not a second nonlinear
replay; any promoted method still requires independent replay/generalization.

## Resources, provenance and stops

acct1, fresh private `wildpino/pqt-002-scale-bias-20261005-001`, same pinned GPU
image as PQT-001-R1, actual T4x2 required; cuda:0 sequential fitting/evaluation,
cuda:1 separate audit, no unified memory assumption. Seed 20261005, deterministic
algorithms, TF32 disabled. Internet enabled for pinned packages/inputs. Live
identity, quota and full owned-session terminal-state admission before push.
Server 5400s, installation 900s, experiment 3300s, audit 900s; timeouts retain
partial evidence. Preserve first failure and use numbered repairs, never overwrite.
No local GPU/native benchmark. No other branch/environment modification.
Freeze source commit and source-byte manifest, generate and commit bundle before
dispatch. Retrieve once on terminal state; large arrays outside Git, small raw
evidence and hash manifests committed. Update TERNARY_INDEX.md after decisions.
