# GigaChat learned compact active core: proposal after native LUT failure

**UNTRAINED architecture hypothesis; specified303 native cost fails. No frozen training protocol, student
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
- Keep source macro experts64/top4, but learn TEN different nonlinear width128
  specializations per macro expert, selecting ONE per active parent. Thus640
  conditional functions/layer retain the routed coefficient capacity of64x1280
  while active FFN work shrinks. Query regions require different learned
  functions, balanced source exposure and usefulness checks, not copied weights.
  Shared nonlinear width256 replaces1280; dense first FFN width1024 replaces8960.
  Learn each specialization's gate/up/down together against the COMPLETE parent
  SwiGLU function in its query region, then validate the composed routed/shared
  mixture. Literal source channel dropping failed299; trained nonlinear query
  specialization is a different variable, not a rescue of299. Earlier weak
  low-rank child banks175/176 also remain failures: here the proposed children
  replace full parent FFNs with nonlinear functions, not small additive factors.
- Preserve original source router selection initially for the real n64 case.
  Within each selected parent, price an input1536->16 shared query projection
  and ten16-dimensional FP32 child keys, stable top1. Child routing/training
  requires a separately frozen source-exposure/function-fit/quality audit;
  retaining the teacher macro selection does not validate child selection.
  Larger source parent counts require additional selective macro search.
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
| Shared I8 child query projection,16/layer |614,400 |616,000 |
| Ten FP32 child keys for each of four selected parents |16,000 |64,000 |
| **Complete addressed weight scenario** | |**447,357,600** |

These are proposal calculations, NOT an exported or scored artifact, physical
DRAM or timing. Additional input scales, format metadata/alignment, activation
conversion, output accumulation, nonlinear/attention/KV/routing work remain.
The new geometry's native cost gate must not borrow301's lookup times or old
276 integer component times. Here parent64xchild10 means640 total functions.
For parent640xchild10=6400 functions, flat parent router/bias alone add
88,531,200B;addressed weight would approach536MB before omitted costs. More n
eventually requires selective macro search even if source quality is preserved.

The bank grows with total conditional n, with only four children consulted
per layer. At640 functions/layer,25*640*3*1536*128=9,437,184,000 routed
coefficients, EXACTLY the original64x1280 routed coefficient count. Row scales
increase because there are more down output rows. This retains an intended
large parameter capacity; it is not yet knowledge transfer or useful capacity.
Per extra function across layers there are14,745,600 I8 coefficients plus row
scales; macro router and child-key growth are priced separately. Parent640/
child10 bank alone would need94.37GB of I8 codes, more than this host's RAM:
a real100B case needs sufficient resources or independently quality-validated
storage precision. No such donor or parameters are instantiated here.

Only64 pretrained parent functions are currently real. Proposed640 trained
regional functions are not640 additional pretrained experts; measure their
distinct parameter/function responses, source region coverage, quality and
utility. Scaling to a larger donor must supply actual different teacher
capacity. Do not duplicate64 experts or count synthetic banks as useful n.

## Cost screen result and next action before teacher collection/training

[303 actual native I8/Q6 screen](METH_303_COMPACT_I8_PREFLIGHT_RESULT_20261003.md)
now passes controls but FAILS14ms with31.268–31.936ms medians/9.953GB peakRSS.
This unchanged implementation is closed before collection/training.
[304 unchanged attribution](METH_304_COMPACT_I8_PROFILE_RESULT_20261003.md)
retains exact outputs:projection work90.478%,MLA dominates. Its separately
compiled21.846–22.477ms remains a cost failure and does not regrade303.
[305 exact-I8 kernel](METH_305_BIASED_I8_PREFLIGHT_RESULT_20261003.md) now
passes original hashes but fails22ms. [306 packed-I4 candidate](METH_306_PACKED_I4_PREFLIGHT_RESULT_20261003.md)
retains coefficient capacity but changes precision;own controls pass,cost fails
20–22ms. [307 waiting profile](METH_307_OPENMP_WAIT_POLICY_RESULT_20261003.md)
has11.3–14.8ms ACTIVE medians but fails variation/EACH14ms.
308/309 affinity investigation fails or remains inconclusive; stop its tuning;
no teacher collection/training or learnedI4 quality is licensed yet. The following
requirements remain necessary for any cost-qualified replacement.

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

## Updated real-function boundary after310

[310](METH_310_GIGACHAT_REGIONAL_SUBSPACE_RESULT_20261003.md) tests actual parent outputs in ten input-selected128D spaces and fails all gates, unlike synthetic cost303–309. This does not include the shared core's compensating output space. [Next shared256/regional128 bound](GIGACHAT_SHARED_REGIONAL_BOUND_PROPOSAL_20261003.md) is unrun; no joint student training/capture or quality claim is licensed. Preserve exposure and generalization failures.

311 common256-plus-regional128 independent-parent output bound also fails all gates. The next uncertainty is complete routed-plus-shared MoE mixture preservation, with actual source targets and unchanged cost/quality requirements. This is not a claim that mixture compensation succeeds.
