# GigaChat learned compact active core: proposal after native LUT failure

**UNMEASURED architecture hypothesis. No frozen training protocol, student
weights, source fidelity, useful additional n or native rate is established.**
301/302 establish that this direct full-source vector4 LUT implementation
cannot fit the native operator budget. MLA, routed and shared work dominate;
optimizing only table construction or only expert weights is insufficient.
298's linear rank192 and299's direct original-channel omission remain failures.

## New scientific variable: distill complete nonlinear functions

Retain source residual hidden1536, vocabulary128256, 26 layers, compressed
KV512 and source rope conventions. Propose learning smaller active attention
and FFN functions jointly against full teacher output, rather than approximating
each source matrix independently or retaining a static subset of its channels.
This changes the active geometry and therefore MUST earn independent composed
quality. It does not claim that discarded source features/heads are harmless.

Prospective geometry for the NEXT COST screen, not an adopted model:

- MLA8 heads instead of32, with qk-nope128/qk-rope64/value192 retained.
  Query/output projections become1536x1536;K-B/V-B use8 banks. KV-A576 and
  KV-rank512 stay as source anchors. A learned eight-head attention function
  would need full teacher contexts/output targets; projection SVD alone
  cannot qualify the changed softmax/head composition.
- Routed macro experts64/top4, nonlinear width128 instead of1280; shared
  nonlinear width256 instead of1280; dense first FFN width1024 instead of8960.
  Learn gate/up/down together against full teacher SwiGLU functions and the
  composed routed/shared mixture. Literal source channel dropping failed299;
  a trained nonlinear student is a different variable, not a rescue of299.
- Preserve original source router selection initially for the real n64 case.
  New or hierarchical selective routing needs a separately frozen retrieval/
  composed-quality audit. Do not combine a changed router with the first
  function-fit comparison or assume increasing RAM removes router cost.
- Cost row-I8 weights and dynamically scaled signed-I8 input vectors,
  integer dot products, explicit FP32 row/input scales and source nonlinear/
  control arithmetic. This avoids large per-query float coefficient tables.
  Input conversion and all integer/scale work MUST be measured in phase60;
  numerical controls compare independently decoded integer math. No QAT/
  conversion fidelity is assumed; training and precision must compose.
- The complete untied head is retained and priced at the source Q6_K format
  as a conservative starting treatment. Its native decode/math must also be
  measured. No implicit free shortlist/head caching. Any head change requires
  new exact-reference/source quality checks; BPB evaluation needs full logits.

## Transparent arithmetic only: what the geometry could save

Each MLA layer has2x(1536x1536)+1536x576+8x(128x512+512x192)
=6,914,048 coefficients. The full-source26-layer MLA has650,051,584;
this proposal has179,765,248. Source weights are not yet mapped to it.

| Proposed active organ | Coefficients/token | I8 weight+FP32 row scales, bytes/token |
| --- | ---: | ---: |
| MLA8heads |179,765,248 |180,730,368 |
| Routed4x128 channels,25 layers |58,982,400 |59,699,200 |
| Shared256 channels,25 layers |29,491,200 |29,696,000 |
| Dense first1024 channels |4,718,592 |4,732,928 |
| Flat F32 source router64 |2,457,600 |9,830,400 |
| Full original Q6 head |197,001,216 |161,602,560 |
| Norms/biases and one Q4 embedding row |97,856 |386,144 |
| **Complete addressed weight scenario** | |**446,677,600** |

These are proposal calculations, NOT an exported or scored artifact, physical
DRAM or timing. Additional input scales, format metadata/alignment, activation
conversion, output accumulation, nonlinear/attention/KV/routing work remain.
The new geometry's native cost gate must not borrow301's lookup times or old
276 integer component times. At n640 flat router/bias alone add88,531,200B;
addressed weight would approach535MB before the omitted costs. More n eventually
requires selective search even if the smaller functions preserve quality.

The encoded routed bank grows with n, with only four functions consulted
per layer. For this geometry25*3*1536*128=14,745,600 distinct I8 coefficients
per additional expert across layers, plus row scales/router/bias. Any n640
or larger is only a shape formula until new different functions are pretrained/
trained and independently quality-gated. Do not duplicate64 source experts or
claim a synthetic bank proves the user's useful RAM-scale capacity.

## Exact next action before teacher collection/training

First independently verify every arithmetic row above and freeze a source-bound
complete cost/controller/layout protocol for this proposed geometry, including
actual Q6 full-head decoding and activation conversion. Validate signed-I8
integer operations without saturation/overflow, explicit rounding/scales,
packed-field negative controls, selected bank offsets and actual RAM reads.
Run actual phase60 operators at n64; gates and stop before observations.
If that full native budget fails, do not collect/train this geometry.

Only a credible cost path can license a separately frozen joint nonlinear
function-fit pilot. Reuse298 real FFN input captures for calibration where
appropriate; they cover selected experts, not unbiased whole-MoE teacher
output or attention contexts. Identify the additional teacher targets needed
before any capture run. Reuse frozen donor source graph/instrumentation for
those targets without resuming its generic port. Communicate expected capture/
training time and stop conditions; new collection could cost minutes to hours.

Before any teacher/student quality observation, fix initialization, data fit/
validation split, source routing/gates, optimizer/loss/precision, exposure,
independent error and resource gates. Complete source projections/functions
are targets, not output-free oracles deployed at inference. Layer/function
passes cannot establish composed residual/attention/routing model quality.
Independent source-disjoint prediction/generation/tasks, actual native cache/
head/route fidelity and accepted>=50 on the same exported artifact remain
mandatory. Multiple families/scales remain a further verified applicability
requirement, not an assumption about this proposed sparse-source geometry.
