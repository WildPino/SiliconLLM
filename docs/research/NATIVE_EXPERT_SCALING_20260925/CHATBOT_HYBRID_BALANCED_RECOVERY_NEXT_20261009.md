# Next whole recovery pilot: balanced source supervision and quantizer robustness

9 October2026. PROPOSED pilot/UNEXECUTED learning. Common-core diagnostic
[complete/close](CHATBOT_HYBRID_COMMON_CORE_RESULT_20261009.md). Original C full
gate remains FAIL19/32; source quality remains poor. Independent new source
[capture/adoption](CHATBOT_HYBRID_TRANSFER_CAPTURE_RESULT_20261009.md) is COMPLETE:
128 FIT/32 DEV across8 domains,4643/1103 full-vocab labels; all saved transport
PASS.137 EOS stops/23 length96 truncations; no answer-truth admission. No live job.

## First available action

Implement/freeze the finite whole learner below. Reuse the adopted corpus and
actual final learner/optimizer checkpoint; no completed source/capture/adopter/
core/bank evaluation replay. Saved source JSON tokenizer and bit/ID/EOS/position
transport is qualified with a shared Rust dependency, not independent BPE.
64 RESERVED texts stay unqueried.23 partial answers supply prefix labels but not
full-answer targets; teacher/truncation/coverage limits stay explicit.

## New variable and relevance to the real goal

The existing compact254.93M target actually deploys through original packed LUT
and SSM machinery. Common core/norm errors are<3e-6, common banks mostly<2e-6,
but whole small drift can cross discontinuous AQ boundaries. Tiny6-case/8-update
source recovery is inadequate. Fixing algebraic fidelity indefinitely would not
establish a useful chatbot; a NEW finite connected recovery/robustness pilot can
test both problems on the same eventual packed artifact.

Reuse actual final learner/optimizer checkpoint, geometry/ternary pair format,
original F32 control and C integer LUT bodies. A NEW quality fit uses different
supervision, balanced domain updates and a declared training-only quantizer
robustness objective. It does not change the native inference geometry/precision
or retroactively promote old FAIL. No generic donor runtime or duplicate pool.

For real-valued AQ normalization let `a=max(absmax(x),1e-12)` and
`t=63*x/a`. If `||delta_x||_infinity<=epsilon*a`, epsilon<1, the max/clamp
operation gives `|delta_a|<=epsilon*a`, so coordinatewise
`|delta_t|<=126*epsilon/(1-epsilon)`. A minimum distance to half-integer thresholds
larger than this bound guarantees unchanged rounded codes in real arithmetic.
F32 evaluation errors need an additional bound. Conversely, small norm error
without a rounding margin supplies no such guarantee. This is an algebraic
motivation for margin/dither training, not evidence that full-trajectory epsilon
is bounded: common-input RMS cannot establish that missing global assumption.

## Required implementation and frozen contract before learning

1. Implement optional training-only AQ perturbation/margin term, inference
   round/absmax/clip identical to current packed C target. A possible normalized
   margin is `distance(t, Z+1/2)` with `t=x*(1/scale)` and detached AQ scale;
   penalize insufficient margin. Small normalized dithering tests nearby code
   decisions during training. These are currently hypotheses, not available or
   proved fixes. Fix epsilon/margin/weight/seed before outputs; preserve CUDA RNG
   across activation-checkpoint recomputation and save its state for continuation.
2. Reload actual pilot checkpoint and verify complete model/optimizer shapes.
   Measure initial ALL160 whole teacher-prefix outputs on NEW data. Equal case
   weighting and domain-interleaved updates avoid long responses dominating by
   label count; report both case-weighted and label-weighted metrics.
3. Prewrite a finite schedule, e.g4 complete balanced epochs/512 updates, with
   exact data order, KL objective/temperature, learning rate/clipping, trainable
   groups, STE and robustness term. This schedule is a proposal until frozen.
   Keep final checkpoint, not a favorable DEV-selected epoch. Measure first
   short/long backward/update, all core/bank/norm gradient groups, actual GPU/
   OS/RNG/optimizer storage and through-exit cost.
4. Set whole recovery/compute criteria BEFORE any pilot values. Include per-domain
   absolute KL/ID disagreement and improvement against NEW initial baseline,
   finiteness/support, fixed throughput/memory/deadline and plateau/stop rule.
   DEV is calibration; it cannot prove fresh own-history preservation.
5. Evaluate ALL160 final whole outputs once, then export actual changed packed
   weights and execute NEW native prefixes against those saved learner outputs.
   Keep old13/32 PASS/19FAIL record. Require a new frozen strict whole numerical
   check, not a posthoc loosened tolerance. Also check source-relative outputs
   directly on this actual native artifact; GPU behavior is not C quality proof.

Proposed local cap<=3600s family/16GiB OS/11GiB allocated GPU/12GiB reserved,
bounded checkpoints/full-vocab output extents priced before launch. Actual
throughput on longer cases must establish this feasibility; no T4 starts from
these estimates. If long sequences/workspace or recovery fail, retain the result
and identify core information loss/initialization/selection/robustness before
adding arbitrary compute. Dense-group SUM versus normalized-mixture initialization
is a declared open concern, not assumed fixed by row copying or multiplying by72.

## Return to actual chatbot, useful n and speed

On eligible recovery/native forward, add student-own-prefix recovery and fresh
original-template/BOTH-EOS own-history/generation/tasks, then accepted batch1
>=50 on the SAME packed quality-qualified model. Include full head/chat/routing
and actual DRAM/cold cost. Structured CPU LUT winners AND mass, distinct useful
larger pools/removal interventions and other family/~10B/~100B variants remain
required. T4 requires communicated reason/budget/stops and measured FP16-compatible
feasibility before allocation. Keep goal ACTIVE/INCOMPLETE.
