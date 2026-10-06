# PQT-003: direct head fidelity with hard ternary forward weights

Prospective, 5 October 2026. PQT-002 failed absolute source preservation;
local layer SSE and greedy string agreement do not establish semantic capacity.
No PQT-003 donor fitting or evaluation has occurred before this protocol.

## Question and fixed arms

Can bounded calibration-only optimization of the actual final source prediction
distribution restore fidelity where layer reconstruction did not? Four arms,
all reported: PR-B, I4, HK and HM. PR-B and I4 retain PQT-002's exact procedures.
HK minimizes source-to-candidate KL. HM adds .1 times a hinge enforcing the
original winning token's margin, capped at .25 nats. Both start from PR-B,
chosen for a minimal controlled extension rather than evaluation ranking.
No original core parameters are trained. This is limited post-training
distillation, not training from scratch, and not a no-gradient algorithm.

Same Qwen2.5-0.5B donor revision, file hashes, original FP32 source values,
last down projection [896,4864], SDPA and runtime pins as PQT-001-R1/PQT-002.
All immutable input identities are reverified. Only this projection is changed.
All previous evaluation/prompt windows 0..23 are consumed and excluded.

## Data roles

Calibration: original first sixteen 129-token WikiText train windows, 2048
first-128-position states. Evaluation: WikiText test windows 24..31, 1024
next-token positions. Generation: first 32 tokens of test windows 32..39,
eight prompts, at most 32 greedy new tokens, stop on EOS, own cache/prefix.
Tokenizer and corpus preparation match PQT-002. Public WikiText remains a
diagnostic rather than fresh project-wide or pretraining-independent validation.
Fit all final representations before any new evaluation metric or rollout.

## Exact intervention and fitting

Capture original final-decoder pre-normalization states H, original projection
output Y and input X. Keep residual R=H-Y fixed on calibration. Candidate states
are the frozen actual RMSNorm(R+X Q^T+b); the tied original readout is frozen.
Validate original Q=W against full donor post-normalization states with relative
RMS <=1e-5. This surrogate represents the actual last-projection intervention;
it does not optimize a layer SSE proxy. Audit independently against full forwards.

Hard Q contains {-1,0,+1} codes times positive group-64 FP32 scales. Latent
real-valued codes use exact hard forward values and an identity straight-through
derivative. Exact half-scale ties round to zero. Optimize latent codes, log scales
and 896-value FP32 output bias; no continuous weight residual is deployed.
Adam: latent lr .02, log-scale lr .001, bias lr .001, beta(.9,.999), eps1e-8,
no weight decay or AMP, global gradient norm clip 1.0. Clamp latent [-1.5,1.5]
and scale multipliers to [.25,4] of initial max(scale,1e-8). Eight calibration
epochs, batch 64: 256 actually applied updates per arm. CPU random generator
seed 20261005 produces eight full 2048-position permutations; identical schedule
for HK/HM. No early stopping or held-out checkpoint selection. Evaluate only
the final 256-update artifacts. Count optimizer state steps for each parameter,
record each loss/gradient norm and final hard code/scale/bias changes. Reject
missing/nonfinite gradients or parameters; verify frozen core has no gradients.

HK objective: mean full-vocabulary KL(source || candidate), source FP32 log
probabilities cached once from original normalized states/readout. HM additionally
uses max(0,min(original top-two logit gap,.25) - candidate source-winner logit
+ highest candidate non-winner logit). Report the KL and hinge separately.
Inference stores ternary packed bytes, scales and bias only: 1,365,504 bytes;
I4 stores 2,451,456 bytes. Count fitting shadow/optimizer memory and time separately.
Source is serialized 16-bit; fitting uses its original values in FP32, not a
previous quantized donor. One projection remains inside a mixed model.

## Decisions, controls and audit

Same absolute gates as PQT-002: all-position source argmax agreement >=99%,
mean KL <=.01 nats, next-token NLL increase <=.01 nats/token; generation pooled
token-position agreement >=95% and >=7/8 exact continuations. Missing tokens
count as disagreement. Payload <=35% of projection FP16 bytes and total fit
<=2700 seconds. No relaxation after observing results. Report all arms against
I4 and PR-B. Passing only admits further expert/generalization/native testing;
it cannot establish complete-model preservation, semantic usefulness or speed.

Before donor access, existing independent controls plus: exact hard STE forward
and tie behavior; specified identity surrogate derivative; head KL/active-hinge
gradient checked against independent scalar softmax arithmetic in FP64; actual
Adam step changes a hard code and reports step=1. The derivative control is not
a finite difference of the discontinuous hard objective.

Independent cuda:1 process imports no fitting module. Retain PQT-002 independent
decoder, FP64 projection/readout audit and all stored-state greedy choices,
same 1e-5 reconstruction, 2e-5 readout and 1e-4 near-tie tolerances. Independently
reload original model with frozen core; replay all 8 source/candidate evaluation
contexts and all 8 greedy continuations per arm, requiring exact token traces.
For HK/HM replay all calibration windows and check their saved surrogate states
against actual model states plus independently computed FP64 RMSNorm using
original norm weights. Verify complete calibration permutations and actual
optimizer step counts. Audit failure prevents scientific adjudication.

## Resources and provenance

acct1, fresh private `wildpino/pqt-003-head-fidelity-20261005-001`, pinned qualified
GPU image and runtime from PQT-002, actual T4x2. Seed/deterministic algorithms,
TF32 disabled. cuda:0 fitting/evaluation; cuda:1 separate audit. Each head arm
<=1200s, entire fit <=2700s; experiment subprocess <=3300s, install <=900s,
audit <=900s, server <=5400s. Hard timeouts preserve first failure and logs.
Live owner/quota/complete session inventory admission before dispatch. No
local GPU or native timing, no mutation of the other checkout/environment.
Commit protocol/source, generate committed-byte bundle, commit bundle then push.
Retain all raw artifacts and exact server source; retrieve once on terminal
state, immutable evidence namespaces, document numbered repairs without erasure.
Track quality/cost/limitations and next action in TERNARY_INDEX.md.
