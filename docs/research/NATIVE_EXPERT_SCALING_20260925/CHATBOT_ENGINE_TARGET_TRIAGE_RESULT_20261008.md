# Actual target-aware donor screening and selected bridge

8 October2026. Goal ACTIVE/INCOMPLETE. Tools/protocol frozen at
`03d65faeb0b5e2f80242e758002d7c37d14ee266` before observation.
Both commands exit0; no tensor libraries, weights, model or native calls.

## Actual inputs and observations

[Acquirer](../../../benchmarks/native_expert_scaling/chatbot_engine_target_acquire.py)
completed in.922s; creates [bound inputs](engine_target_triage_inputs_20261008/inputs.json),
SHA `aaccb86d02331060c0f42afec0295f713cd53b78fb975a343b3c5a6092d59488`.
Public pinned config/API bytes are retained; local configs/operator source and
existing Qwen/Giga qualified census bytes are bound. BitNet metadata revision
is in that manifest; no remote Python or weight download.

[Offline tool](../../../benchmarks/native_expert_scaling/chatbot_engine_target_triage.py)
produces [raw result](chatbot_engine_target_triage_20261008.json), SHA
`7696e8da481aedb7cc84f12b71a4434fe226d0694f671313d4f0c6b3a5556c81`,
93,243B,5 donor records/40 prospective target budgets. In-script final elapsed
.047s,OS peak25,821,184B; executor.328s. Resource samples are not a through-exit
family audit. Metadata-only60s/256MiB/5MiB bounds are satisfied in those scopes.
[Hand calculation](chatbot_engine_target_triage_handcheck_20261008.json) independently
checks seven decisive D256/L6/V32768/n128 integer terms; no arithmetic fault.

| Source | Complete counted matrix products/decode | Recurrent/attention structure | Starting advantage and missing step |
|---|---:|---|---|
| Qwen0.5B-Instruct |493,961,216|24 attention, no recurrence|Existing qualified interaction/supervision; core recurrence+compression and ternary transfer missing |
| Giga10B main |1,628,078,080|26 MLA, MoE|Actual pretrained large conditional donor;650M attention products and large active FFNs remain |
| Granite4.0-H-Small |8,800,960,512|36 SSM2/4 attention, MoE|Already recurrent/conditional;3.681B SSM projection products and3.775B routed FFN products remain |
| Falcon-H1-Tiny90M-Instruct |90,996,736|24 parallel SSM2+attention branches|Small actual instruction donor with recurrence; all attention branches and dense FFNs still active |
| BitNet2B4T master |2,412,380,160|30 attention, gated squared-ReLU/SubLN|Producer reports ternary training; recurrence/conditional geometry still missing |

Qwen/Giga counts are **reused qualified census**, not recaptured. Other counts
are new config/operator deductions, not complete tensor-header parameter counts.
Attention history, elementwise/normalization/nonlinear/selection and dispatch
costs are additional. BitNet BF16 master values and ternary cardinality are
unobserved. These products do not establish any source or target throughput.

## What the target budgets establish

The counted original-core target preserves each source's full vocabulary; it
does not obtain speed by silently changing token IDs. Separate F32 embedding/
head arrays are conservatively charged, including source-tied cases. Two core
shapes, D256/L6 and D512/L12, use original Mamba1/SWA/gated expert dimensions;
none has learned weights or behavior admission.

For Falcon's V32768 and n128, D256/L6 requires16,013,312 counted matrix products,
119,028,480B packed byte-pair model coefficients and48,705,536 logical coefficient
bytes/decode, using the proposed tree upper budget. F32 master+gradient+Adam
moments for all target coefficients cost1,533,078,528B before activations/teacher/
copies/workspace. This shows a costed candidate, not preserved donor knowledge.

**Selected first training candidate:** preserve donor width512, provisionally
use12 blocks,2 bounded attention sites,top8/h128/n32. The original-core accounting
gives56,307,712 matrix products,260,042,240B packed model coefficients,
163,737,600 logical coefficientB/decode and2,097,133,568B chosen dense-Adam arrays.
The actual source-compatible SSM projection variant still needs its own budget;
do not adopt these Mamba1 numbers as measurements of that variant. Width is kept
to preserve source embeddings/readout and avoid an immediate512->256 bottleneck.
Depth/attention/precision adaptations still require training and quality evidence.

For the D256/L6 accounting case with~100B expert coefficients, default packed-only
storage with the proposed tree is52,301,872,128B; the old reference-copy loader
would require553,211,386,536B in counted coefficient copies. Full F32 Adam arrays
would require1,609,214,914,560B. Thus large inference RAM and large conversion
optimizer storage are genuinely different problems. Month-plus T4 time alone
cannot allocate those arrays. Factor/adaptor or selective/lazy optimizer methods
will require explicit quality and storage evidence; none is admitted here.

## Algebraic connection to the original scan

Inspection of the bound Falcon fallback step gives, for head a/channel i/state j,

\[
H'_{aij}=\exp(\Delta_a A_a)H_{aij}+\Delta_a B_{aj}X_{ai},
\quad Y_{ai}=\sum_j C_{aj}H'_{aij}+D_aX_{ai}.
\]

Flatten c=a*head_dim+i. Replicate A and delta over the head's channels and A
over j; map B/C through their source groups. This has the original engine scan's
algebraic form. Tiny config has one group, so B/C are shared across heads, and
gated RMS is disabled. Its final SiLU gate also matches the scan organ's gate.
This is an **algebraic mapping**, not a finite-precision/operator parity result.
Source step casts, dt clamp, state dtype and chunked prefill need explicit checks.

The projection/conv layout differs: Falcon obtains X/B/C and delta directly
from its input projection, convolves X/B/C, and applies head-wise delta. Original
Mamba1 derives B/C/delta through its post-conv x_proj/dt_proj. Consequently,
unchanged original weights cannot be loaded as-is. A small projection/conv
adapter can reuse the scan mechanism, rather than porting a generic whole donor
runtime. Actual C compatibility is the next experiment, not established here.

## Decision and next action

Select FalconTiny as the **first real scan-bridge donor**, then test its usability
before choosing it for full chatbot adaptation. This choice is based on state/
operator affinity, reusable original scan algebra and tractable conversion memory.
It is not a claim that90M preserves10B/100B capability. Giga/Granite remain actual
large family variants, and Qwen remains a qualified Transformer comparison.

Next implement/freeze bounded Falcon source acquisition and canonical interaction
preflight, followed by real source-weight/state capture and original-scan bridge
comparison. Establish the teacher's absolute usefulness before long training;
if insufficient, use a stronger hybrid instruction donor with the same mapped
operator class. Then initialize target-shaped whole learner and recover ternary/
conditional/depth/attention changes progressively. See [bridge NEXT](CHATBOT_FALCON_SCAN_BRIDGE_NEXT_20261008.md).
No order-stability fit, old science replay, large generic port or T4 job starts.
