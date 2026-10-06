# PQT-002: scalar adaptation and offsets fail preservation gates

5 October 2026. **EXPERIMENT COMPLETE / GOAL INCOMPLETE.** All ten arms executed;
independent audit passed and gates were reaggregated from retained points/tokens.
No ternary arm passes the declared absolute prediction or rollout gates.
This closes the tested fixed-group scalar/constant-offset recipe on this
projection and diagnostic split, not all behavior-aware ternarization.

Original Qwen2.5-0.5B last FFN projection, unchanged pretrained core, original
calibration windows; new teacher-forced test windows 8..15 and rollout prompts
16..23. Scientific source `2221391`, dispatch `61e058c`, private acct1 kernel
`wildpino/pqt-002-scale-bias-20261005-001`, version 1, ID 137225360.

| Arm | Normalized layer SSE | KL | NLL increase | Source argmax agreement | Rollout token-position agreement | Exact rollouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D | .210623 | .331252 | .299564 | 67.4805% | 20.7031% | 0/8 |
| I4 | .009100 | .017421 | .020858 | 91.4063% | 37.8906% | 2/8 |
| DR | .068831 | .182429 | .160941 | 79.1016% | 21.0938% | 0/8 |
| PR | .057786 | .147175 | .125591 | 79.8828% | 21.0938% | 0/8 |
| DR-S | .059637 | .163990 | .157503 | 79.8828% | 18.7500% | 0/8 |
| PR-S | .050554 | .128991 | .114408 | 81.6406% | 15.6250% | 0/8 |
| DR-B | .068495 | .181772 | .159882 | 79.1016% | 21.0938% | 0/8 |
| PR-B | .057224 | .144656 | .122170 | 79.7852% | 21.0938% | 0/8 |
| DR-SB | .059566 | .163877 | .157274 | 79.9805% | 18.7500% | 0/8 |
| PR-SB | .050408 | .128469 | .113646 | 81.7383% | 15.6250% | 0/8 |

Prediction gates were >=99% agreement, KL <=.01 and NLL increase <=.01;
rollout gates >=95% positional agreement and >=7/8 exact continuations.
I4 also fails these stringent source-preservation gates, although its predictive
metrics are substantially better than every ternary arm. Its NLL increase is
only .02086; divergent greedy text alone does not prove lost semantic capability.

Scale adaptation improves original-context metrics but decreases positional
rollout agreement on these eight prompts. A calibration output offset adds only
minor predictive improvement and leaves the corresponding rollout tokens
unchanged. PR-SB has the strongest predictive ternary metrics, but still has
187/1024 source argmax disagreements. Fitting projection SSE remains an
inadequate objective for preserving final model behavior.

Greedy positional agreement is sensitive to an early divergence and wording
changes. These are behavioral-fidelity diagnostics, not factual correctness,
semantic quality or broad generative-capacity measurements. No arm is promoted;
do not convert this limited negative into a universal or semantic conclusion.

## Representation, cost and checks

Ternary payload 1,361,920 bytes; with FP32 output offset 1,365,504 bytes.
I4 payload 2,451,456 bytes. Ternary uses 55.56% of I4's stored payload, at a
large quality loss. Scales, offsets and file headers are separately recorded.
These are one-projection representations in an otherwise FP32 model, evaluated
by dequantization on GPU; no native-speed claim.

All-arm fitting: 42.0203s. Installation/experiment/audit subprocesses:
16.0157/264.0470/63.6598s. Peak allocated GPU 2,662,531,072 bytes; final RSS
4,057,669,632 bytes. Run-specific cost, not target CPU timing.
Independent reconstruction and readout checks passed, including explicit FP64
cancellation. Source reconstruction relative RMS 2.534e-7; maximum audited
NLL/KL difference 6.475e-6. Every generated greedy choice was rechecked from
retained normalized states: 2804 states, zero precision disagreements.
This does not independently replay the nonlinear transformer for each rollout.

Calibration source inputs/outputs and D/DR/PR code payloads reproduce the
PQT-001-R1 hashes. Newly observed metrics therefore concern different contexts,
not changed initial algorithms or source activations.

Latest quota: acct1 .146435 h used, zero reserved, 29.853565 h unreserved;
acct2/3 each 30 h unused/unreserved. No doubling of API quota for T4x2.
All current jobs are terminal; no local GPU/native benchmarks were launched.

## Evidence and next decision

[Protocol](PQT_002_PROTOCOL.md), [adjudication](PQT_002_ADJUDICATION.json),
[retention](PQT_002_RETENTION.json). Small raw evidence in `pqt_002_evidence/`;
complete local retrieval `results/progressive_ternary/PQT-002/remote_001/`:
109 files, 191,764,273 bytes, all matching `pqt_002_fetch.json`.
Per-window aggregates and per-prompt comparisons are in adjudication, full
decoded traces and state/code/scale identities in retained raw files.

Next method question should target the final source distribution directly:
bounded head-aware distillation of ternary choices/scales, with no core
pretraining and complete update/cost accounting. This is different from
optimizing a local layer proxy. A counted low-rank correction is another
possible mixed-representation hypothesis, but is not yet tested or qualified.
Use new numerical evaluation contexts and include generation/semantic limits.

In parallel with that design, retain the actual native scope: original Switch
expert tensors, nonlinear expert-function preservation and deployment cost on
the same artifact. [Source admission note](NATIVE_TRANSFER_SOURCE_NOTE.md)
identifies the original archives and why the parent I8 export/captures cannot
substitute for float expert inputs or fresh validation. No further protocol is
frozen or job launched yet; the whole research objective remains active.
