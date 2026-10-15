# Native expert scaling: research control index

15 October 2026. **Goal INCOMPLETE / active. Source capture and adoption CLOSED; pipeline reassessed.**
Current state supersedes the historical [handoff](../../../HANDOFF.md) and its
[preserved interruption observation](HANDOFF_SNAPSHOT_20261014/active_router_interruption_observation.json).
[Method](METHOD.md), [prior evidence](PRIOR_EVIDENCE.md), [original goal](HANDOFF_SNAPSHOT_20261014/goal-objective.md),
[latest owner scope clarification](SCOPE_CLARIFICATION_20261015.md).

## Goal and constraints

Reproducible pretrained CHATBOT-to-engine.c conversion: compact reusable
SSM/SWA plus useful selectively consulted capacity, CPU IDs AND masses,
precision suited to actual kernels, measured DRAM and donor-relative quality.
The SAME artifact must deliver >=50 useful accepted batch1 tokens/s; 100 is
an improvement target. Useful RAM-driven n, very-large-model scaling within
ONE family and conversion master/gradient/optimizer costs remain required.
Native MoE is a valid primary path, including finer expert partitioning;
dense70B conversion and unrelated-family support are not mandatory. A small
pilot alone does not demonstrate the large-model pipeline. See scope clarification.

## Latest decisive evidence

[Pipeline reassessment](../PIPELINE_REASSESSMENT_20261015/DECISION_REPORT.md):
stop unchanged compact replay and minor codec/thread grids. Preserve pretrained
functions before reducing active work; compare native learned MoE transfer with
exact dense FFN partitioning plus trained IDs/masses. Neither is admitted.
[Implemented instruct preflight](../PIPELINE_REASSESSMENT_20261015/NATIVE_DONOR_PREFLIGHT.md)
checks586 tensors at pinned revision with no weight values: all-I4 addresses
779.764MB/token; explicit source-router/top6-preserving mixed2/3/4 policies
554.586/559.563MB, plus state/KV and unpriced execution. These are analytical
unimplemented formats, not DRAM/rate/quality. Even the native-MoE alternative
needs a major representation change; no unchanged Granite port or source
download is selected. Historical head/readout and13.8ms claims are separated
in the linked source audit. Existing source capture remains reusable data.

[Source capture/adoption closed](CHATBOT_SOURCE_COVERAGE_RESULT_20261015.md):
432/432 cases,360 FIT/72 DEV,76,512 new labels; both cached controls exact.
Worker/holder/CPU auditor exit0; all5.032B score words,411 raw source tensors,
873 output files and8390 input pins verified. CPU audit485.047s/234MiB/no GPU.
Adopted Falcon supervision only; no learner, quality, rate or capacity admission.
[Independent review](../PIPELINE_REASSESSMENT_20261015/INDEPENDENT_REVIEW.md)
proposes source-preserving native MoE partitioning. Existing Q4_K blocks require
512->2x256;128-wide children need another format or padding. Large-family core
cost remains an explicit gate; a small donor is an enabling pilot only.

[Balanced replay final result](CHATBOT_BALANCED_REPLAY_RESULT_20261015.md) is
CLOSED:512 updates/Adam830;worker/holder/auditor exit0. All80 cases/9,911 labels/
649,537,207 saved score words,211 model/422 moment tensors,exposure/routes/
resource/current full seals PASS. New FIT KL.70077/disagreement16.418% passes;
old DEV2.56407/35.993% and new DEV7.11348/81.487% fail. No unchanged152-case
fit or promotion. GPU training/final forwards remain producer witnesses.

[Source coverage protocol](CHATBOT_SOURCE_COVERAGE_PROTOCOL_20261015.md) and
[binding](CHATBOT_SOURCE_COVERAGE_STATE_20261015.md) now execute the remaining
360 FIT/72 DEV from the adopted public
corpus,30/6 per12 sources;40 LongAlign cases deferred,RESERVED untouched.
Longest prefix1769 first as resource probe,then all remaining fixed IDs;two
known donor controls before new packets. No learner call or new training recipe
in this phase. Current runtime/source411/scalars and exact BF packet controls
must pass;source packets require a separate complete CPU adoption audit.

[Depth diagnostic](FIXED_DEPTH_DIAGNOSTIC_RESULT_20261015.md) is CLOSED:
all3/6/12 FIT fail;nonselected6 has0/16 tasks/DEV14.82421/99.1114%. Numerical
PASS,original posthash fault retained. No unchanged omission/runtime grids.
[I4 generator+readout](ENCODED_ORGANS_WAIT_RESULT_20261014.md) still exceeds
20ms before other organs;finite cost evidence,not a universal lower bound.

[Exact packed child layout](GRANITE_CHILD256_LAYOUT_RESULT_20261015.md) PASS:
all6144 Q4_K blocks copied exactly, Q8 segments exact, full synthetic SwiGLU
relative L2=5.58e-8 with parent mass inherited by both children. Two256-wide
children fit existing kernels; one addresses442,368 of884,736 expert bytes.
No source-weight quality, selector, bundle cost, DRAM or rate admission.

## State of the two research questions

**Transfer:** [full-width native quality bridge](FULL_WIDTH_NATIVE_CHAT_RESULT_20261013.md)
passes the known corpus:14/16 tasks, own-answer followup, DEV KL .02156 /
disagreement4.8847%, all prefix/domain gates. All97 trajectories/24 sites /
1.188B output/integer words audited. Source GPU FFN/attention/head and dense
contextual generators remain; historical transitive custody limits are explicit.
This is a reusable quality control, not final CPU conversion or new held-out proof.

**Low-cost useful conditional geometry:** [actual FFN operands](FFN_OPERAND_RESULT_20261013.md)
are qualified at all24 sites. [Affine/cross defect](FFN_AFFINE_DEFECT_RESULT_20261013.md),
[fixed subsets](FFN_EXACT_SUBSET_RESULT_20261013.md), [PCA](FFN_SKETCH_ROUTER_RESULT_20261014.md)
and source-active FIT rules yield no admitted candidate. [Per-token oracle](FFN_REMAINDER_ORACLE_RESULT_20261013.md)
passes local gates with counts512/2048/32 but computes ALL source projections
before choosing; it is not a cheap router or optimal sparsity bound.

## Available components and closed paths

- [All24 row/state ABI](ROW_STATE_RESULT_20261013.md): source-derived state96,
  causal conv, own norm, canonical p6/I16 packets; generic C interface qualified.
- [Token/F0 compiler](TOKEN_GENERATOR_COMPILER_RESULT_20261013.md) and
  [own-state branch](TOKEN_STATE_RESULT_20261013.md): actual uint32 IDs, complete
  source-only table, persistent state and prefix/continuation equality.
- [I16 readout](INTEGER_READOUT_RESULT_20261013.md) is exact; tested
  [byte-LUT](EXACT_INTEGER_LUT_RESULT_20261013.md) was slower and is rejected.
- [Common linear generator residual](TOKEN_CONTEXT_RANK_RESULT_20261013.md)
  needs FIT ranks>=1408/1857/1857 within its stated family. Do not repeat that
  grid, D256 failed shadow or fixed output-bank oracle failure unchanged.

## Exact next action

[Q4 expert bundle](GRANITE_CHILD256_COST_RESULT_20261015.md): full numeric
control PASS,12workers/288samples stableoutputhashes, cost/stability FAIL.
[Large-family protected path](../PIPELINE_REASSESSMENT_20261015/PROTECTED_CORE_120B_CPU_RESULT_20261015.md):
all controls/placement/seals PASS; three process medians34.39/50.95/53.88ms
without head, experts, RoPE or KV attention. Reject this Q4_K variant. No
unchanged runtime grids, full120B acquisition or selector fit follows.

Next substantive variable: actual-weight activation-aware Q/O rank256 on the
FIRST source attention block, with K/V/norm/bias/sink/RoPE/window preserved.
Layer0 permits authentic embedding-derived states without the whole donor;
Falcon states/logits are invalid calibration substitutes. Acquire only pinned
Q/K/V/O, auxiliary tensors and used embedding rows by exact HTTP ranges.
Freeze case IDs/splits/token cap, arithmetic/control and local pre-residual
attention-error/distribution gates BEFORE fit. Rank512 is diagnostic only;
no DEV rank selection. Quantized factors must use an actual export format.
A local failure stops the uniform cheap recipe; a local pass cannot establish
whole-model quality or transfer to internal layers. Only then consider the
same-family20B full-model pilot. No run is launched from this planning note.

No owned capture/audit run remains live. Do not restart successful generation.
The complete source coverage is adopted; no automatic D512 fit follows. Qualify
the actual-weight core intervention; packed layout and bounded cost screens are closed.
Pause only for expected>=1h numerical jobs; continue shorter checks here.

RESERVED104 new+64 older and LongAlign remain unqueried. Preserve foreign
changes,owner Git identity and attribution-free commits. Final same-artifact
CPU chatbot quality/useful50,charged IDs AND masses,physical DRAM,useful n,
one family spanning very large models and sustainable conversion costs remain required.

## History and primary evidence

[Previous index snapshot](INDEX_THROUGH_DEPTH_DIAGNOSTIC_20261015.md) preserves
old launch/recovery observations. [609-report inventory](EXPERIMENT_CATALOG_20261014.md)
is a fixed report-file snapshot,not a count of individual experiments.
[Primary readback](DECISIONS_FROM_CATALOG_20261014.md) separates Switch warm
infilling rates,Qwen/Falcon quality and synthetic capacity controls. Unrelated
quality and speed evidence do not establish a single admitted artifact.
