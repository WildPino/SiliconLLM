# Bounded backend diagnosis, then recovery of useful chatbot capacity

9 October2026. Historical plan: common-core diagnostic now COMPLETE/CLOSE;
balanced source capture and transport adoption COMPLETE. Current resumption is
[balanced recovery NEXT](CHATBOT_HYBRID_BALANCED_RECOVERY_NEXT_20261009.md).
The proposals below retain their original scope. Results and complete bindings are in
[native result](CHATBOT_HYBRID_NATIVE_RESULT_20261009.md). No active worker/T4.
This supersedes the completed export/build NEXT; goal remains ACTIVE/INCOMPLETE.

## Architectural anchor and priorities

Original engine LUT/ternary/compact recurrence is the selected deployment path.
The first full artifact now uses those original kernels,425.21MB packed-only
resident model and9.12MB recurrent/conv/KV state. Temporary flat E72 routing is
charged. Original selected expert work is independent of E at fixed k/h/D;
original flat router and physical storage/bandwidth still depend on E.

There are two different remaining problems:

1. Execute the learned target's own forward consistently:13/32 logit rows meet
   fixed1e-4 RMS,19 fail, all32 greedy IDs equal. This is a deployment-numerics
   prerequisite, not donor-preservation quality.
2. Transfer donor capacity:8 updates/6 calibration cases yield FIT10/20 and
   DEV12/12 donor-ID disagreements. This tiny feasibility run does not establish
   inadequate target capacity or sufficient long-training recovery.

Keep the first problem bounded; the substantive research is the second.
No continued source-local1% approximation ladder or generic donor runtime.

## First action: ONE common-core diagnostic

**Uncertainty:** are material core differences present before quantized banks,
when each backend receives the same stored C operands? New common banks agree
in3131/3132 rows, all router IDs agree. Their ONE material row is `fit_code`
position33, whose later original output rows pass; it cannot explain all19 failures.

**Reuse:** actual packed model/field manifest, native trace3132 records,
queries/case boundaries, source-compatible target config and frozen core code.
No teacher call, head/embedding/whole target evaluation or training replay.

**New observation:** evaluate each of the12 individual recurrent/SWA modules
on its stored C core-input sequence, separately for all6 cases, with zero initial
state per case. Preserve recurrence/conv/attention evolution within each module;
do not zero state at each token. Compare ALL3132 core outputs to saved C outputs.
Inputs differ from the old GPU whole trajectory, so this is a controlled new
scope. Record norm reconstruction from saved block inputs separately, since
the C core-input normalization has already been applied to common operands.

Freeze implementation/protocol/inputs before execution. Allocate source-shaped
core modules only, loading bound packed F32 fields; no head/master expert arrays.
Require exact shapes/dtypes, TF32 off, all-case coverage and output retention.
Proposed budget <=600s family/4GiB OS/2GiB allocated GPU/64MiB output, sequential
local execution. Fix numerical classifications before values. If apparatus cannot
meet this budget, retain the fault and narrow by an explicit new protocol.

**Decision:** material common-core error localizes core arithmetic/scan ordering;
close common cores plus close banks support investigating discontinuous AQ under
small upstream perturbations. Neither result proves all whole errors causally.
Use an intermediate quantizer/gate capture only if it distinguishes a concrete
fix; set a separate budget before doing so. Stop this diagnostic after one
complete comparison; do not enumerate arbitrary precisions or tolerances.

Original native FAIL and thresholds remain unchanged. A corrected forward is
a NEW method/artifact with its own frozen learner/native comparison; do not
retroactively promote the old artifact. Original integer LUT bodies remain the
anchor. Canonical scalar arithmetic or margin-aware QAT are hypotheses, not
available/qualified fixes.

## Main recovery problem: account for the information discarded

The actual initializer projects D2048->512, retaining about47.1991% of the
equal-trace embedding/readout Gram energy; this is not a bound on task quality.
It chooses12 of24 source core sites, slices16 of64 channels within retained
SSM heads, and replaces the source's parallel SSM+attention at every site with
10 SSM-only and2 SWA-only sites. Row-slot counts retain paired source FFN groups,
but top8/72 consults only part of their transformed functions. All these changes
are approximations. None has a standalone guaranteed information-preserving inverse.

Before a long fit, measure whole residual/core/bank energies on a balanced
calibration cohort and make initialization scales explicit. A dense FFN split
into groups computes their SUM; normalized top-k routing computes a weighted
average. Source row copying alone does not preserve their magnitude. For uniform
random k of E with mass1/k, `E * mean(selected functions)` is an unbiased SUM
estimator, but actual learned routing is nonuniform and variance/covariance matter.
This identity is a diagnostic, not authorization to multiply all down scales by72.
Any shared base, overlap, output rescaling or expanded expert variant needs its
own complete source-relative recovery evidence and active byte/product budget.

## Broader local adaptation before a long T4 commitment

After a deployable learner forward is qualified, create a balanced, reproducible
FIT/DEV corpus covering instructions, arithmetic, reading, code, rewrite and
multi-turn history, with longer prefixes and complete stop-aware source answers.
Use original source template/token IDs/BOTH EOS. The6 pilot and16 screening
cases are consumed calibration and excluded from final fresh gates. Existing
Qwen200-case tensors belong to another donor and are not Falcon supervision.

Freeze a finite whole-model distillation/QAT pilot: actual GPU and OS memory,
tokens/updates/throughput/checkpoint cost, per-domain recovery and expert exposure.
Mix domains across updates; retain complete final evaluation, not only online
loss. Teacher logits/whole trajectories train the connected compact model;
layerwise auxiliary targets may help but do not substitute final donor-relative
behavior. Teacher-forced prefixes alone do not qualify student-own-history.

Define the recovery criterion, plateau/compute stop and fresh excluded evaluation
before running. If broad held-out recovery is absent, identify whether lost core
capacity, initialization or expert selection binds before scaling compute/n.
If it improves, add student-prefix recovery and fresh own-history/task checks,
then measure native accepted batch1 IDs/s on that SAME packed artifact. A month
of T4 is permitted in scope but not yet priced: communicate reason/budget/stops
and measure FP16-compatible feasibility before allocating it.

## Later required stages

Replace flat E scan with structured CPU LUT selection AND normalized mass at
useful larger n; test distinct functions/removal interventions and fresh quality.
Measure physical DRAM and selected bytes with pools exceeding cache, context/
thread/hardware variation and full head/chat overhead. Require >=50 accepted
batch1 IDs/s (100 stretch) on the quality-qualified artifact. Show an actual
second family/~10B variant and ~100B when resources permit. RAM residency alone
does not establish useful capacity or constant token cost.

All completed source/export/C-prefix/pilot/audit/bank namespaces are terminal;
reuse saved bytes. Preserve first faults, foreign tracked hashes and publisher.
No goal completion, blocker or resource request follows from this NEXT.
