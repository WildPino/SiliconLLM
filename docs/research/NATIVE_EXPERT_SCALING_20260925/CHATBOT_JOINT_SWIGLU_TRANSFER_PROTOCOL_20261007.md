# Finite joint SwiGLU transfer: selected representation, before fitting

7 October2026. Goal INCOMPLETE. [Canonical interaction](CHATBOT_INTERACTION_RESULT_20261007.md)
is now qualified. The [complete census](CHATBOT_OPERATOR_CENSUS_RESULT_20261007.md)
shows that current Qwen still activates ALL4864 source channels. This trial
changes that representation. It is not yet fitted/exported/native-qualified.

## Representation and exact uncertainty

Freeze ONE geometry: D896/L24, shared512 SwiGLU channels, top4 parents from16,
one128-channel FULL-input nonlinear function per selected parent, query32.
Compare1 versus10 children/parent, E16 versusE160. Every leaf has its own
trainable gate/up/down; shared gate/up/down are trained jointly with leaves.
The actual [Torch block](../../../benchmarks/native_expert_scaling/chatbot_compact_geometry.py)
implements this selected-only evaluation. It does NOT evaluate the original
4864-channel source basis, retain an affine896x896 branch, or constrain new
functions to old310/311/316 PCA/shared output spaces.

    s_512(x) + sum_(four parents p) w_p(x) B_pc[SiLU(G_pc x)*(U_pc x)].

Full normalized x enters each selected function. A frozen32-dimensional
projection is used ONLY to choose regions. Parent/child squared distances
use stored centers and norms; stable ascending-ID ties. Parent mass is exactly
the softmax over the four SELECTED parent negative distances. Child selects
one nearest function, with unit child mass. This is an explicit target router,
not preserved donor MoE routing, fullE160 softmax, or qualified large-n LUT.
The query-norm term cancels algebraically in choices/selected-parent softmax;
coefficient products and stored centroid norms are both charged.

Retain complete source attention and tied BF16 embedding/head in the initial
finite experiment. Keep actual C float32 KV allocation4096 explicitly priced.
The first encoding proposal is BF16 matrices, FP32 training/reference arithmetic;
native BF16 decode/reduction ordering needs its own future qualification.
Router centroid norm scalars are explicitly four-byte float32 in the byte ledger.
The executable score convention is2*q.dot(center)-stored_center_norm; omitted
common query norm cancels in real algebra. No bit-equality claim to a separately
rounded full-distance implementation. The block requires explicit source-row
initialization before evaluation, retaining no full source basis at runtime;
row selection is an unfrozen fit-stage choice and initialization is not fidelity.
No unused Q8 head proposal, old K64 exact-rerank recipe or speculative tokens
may silently enter quality or accepted rate. METH214's pooled fresh ranking
FAIL stays closed, despite its head shortlist passing within that scope.

## Complete necessary cost gate, BEFORE new observations

[Specification](../../../benchmarks/native_expert_scaling/chatbot_compact_spec.json)
fixes every dimension/slot/mass/storage convention. Derive new dimension costs
from that specification and SHA-bound existing Qwen census; never replay census
or values. Each arm must use<=3/5 source complete matrix terms AND<=3/5 source
logical coefficient bytes/token. These are necessary eligibility checks, not
physical DRAM/throughput gates. Include source attention/head, projected router,
all16 parent score products, selected child scores and centroid scalar bytes.
Context-dependent attention, nonlinear/reduction/control work remain explicit.
Stored parameters increase with E; active function width remains1024.
Greater slots alone cannot establish distinct trained/useful capacity.

Rationale: measured METH229's six-thread2048-source-channel plus affine component
cost8.875ms/token left little room within a20ms whole-token budget. That operator
and its nonlinear quality230/231 did not qualify a whole model. Halving its
gated-channel count while removing the affine branch is a concrete changed
variable; the complete head/attention floor is retained in the new budget.
No linear extrapolation of that component clock is claimed.

## Source exposure prerequisite and next actual work

METH125 has only256 token positions PER LAYER on augmented E128/E1280
trajectories, not6144 independent states/layer or original-donor support.
METH230's repeated-input child problem and231's9.381% validation error warn
against treating those states/derivatives as enough for a new joint converter.
METH256's amplitude-only count gain remains closed. Reuse their costs and
diagnoses, not their failed fitted functions or consumed cohorts as fresh gates.

First implement/freeze a bounded pure-original Qwen source-capture stage using
the qualified canonical serializer and BOTH EOS. Keep calibration conversations
disjoint by conversation from a reserved development partition. Capture full
normalized pre-MLP x AND complete original FFN response, with position/message/
own-history provenance; do not flatten24 layers into more samples for one layer.
Use the existing local source only, no T4/downloads. Source weights and actual
model/runtime/operator inputs require bindings before values. A GPU monitor
must retain actual process/allocated GPU/RSS and the worker peak through exit.
Do not launch unbound weights or infer resource success from earlier runtimes.

The training support/region construction/fit objectives, exact finite update
budget, held-out response criteria and stop rules MUST be separately frozen
before fitting. One layer12 pilot may reject this complete priced recipe
before24-layer conversion; it cannot promote the full pipeline. Freeze source-
informed initialization and joint optimization of shared AND leaf G/U/B,
with exposure/uniqueness/matched-choice and held-out complete-response checks.
No closed fixed-output/prior-only/amplitude/precision/width ladder is reopened.
If this fixed geometry fails, diagnose and change representation/information;
do not increase widths/steps/strengths silently after seeing the result.

## Remaining whole pipeline gates

Pass of the local pilot only permits all24-layer transformation and new export/
native profile. Bound Python/C dtype/shape/C-order/writability/byte sizes for
both inputs AND outputs. Then fresh donor-relative canonical dialogue, tasks,
own-history generation and >=50 accepted IDs/s on the SAME resulting artifact.
Use new held-out data AFTER candidate freeze; source capture/development cannot
be called fresh whole quality. Original295/296 controls and297 FAIL remain.
Price physical selected DRAM and CPU LUT winner AND mass at useful larger n;
transfer to actual Giga~10B/other scales only with family-specific whole contracts.
The~79.674B header census is not a trained100B conversion. No goal promotion.
