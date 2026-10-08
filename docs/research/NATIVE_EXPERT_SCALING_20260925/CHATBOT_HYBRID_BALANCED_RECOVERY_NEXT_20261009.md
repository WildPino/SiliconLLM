# Next whole recovery pilot: balanced source supervision and quantizer robustness

9 October2026. Pilot IMPLEMENTED/FROZEN, ONE local instance LIVE under
[protocol](CHATBOT_HYBRID_RECOVERY_PROTOCOL_20261009.md)/
[process record](CHATBOT_HYBRID_RECOVERY_LIVE_20261009.md). No final recovery.
Freeze1f2a30a,180-input bindinged9299ff,session76414/launcher20000/worker33244.
Common-core diagnostic
[complete/close](CHATBOT_HYBRID_COMMON_CORE_RESULT_20261009.md). Original C full
gate remains FAIL19/32; source quality remains poor. Independent new source
[capture/adoption](CHATBOT_HYBRID_TRANSFER_CAPTURE_RESULT_20261009.md) is COMPLETE:
128 FIT/32 DEV across8 domains,4643/1103 full-vocab labels; all saved transport
PASS.137 EOS stops/23 length96 truncations; no answer-truth admission.

## First available action

Observe SAME session76414 or authoritative launcher20000/worker33244. Do not
restart an expired observation. Actual final learner/optimizer is restored;
ALL160 NEW initial evaluations COMPLETE,at least30 update records durable at
recorded snapshot;first-longest183-input backward ALL12 finite positive groups.
Actual first steps10–12s including complete CPU state copy;3600s
completion uncertain,cap unchanged. Reuse adopted corpus; no
completed source/capture/adopter/core/bank evaluation replay. Saved source JSON tokenizer and bit/ID/EOS/position
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

## Implemented and frozen contract

1. Training-only normalized AQ uniform dither[-.025,.025],seed20261009,
   no margin penalty. Original no-grad AQ63 unchanged. Explicit worker-local
   hook; original target code untouched. Whole-block checkpointing preserves
   CPU/CUDA RNG; completed state records retain RNG/model/moments/order.
2. Restore complete actual prior checkpoint/Adam step8. ALL160 NEW initial
   teacher-prefix observations before updates; equal case-mean forward KL/temp1,
   eight interleaved domains. Report case AND label-weighted metrics per domain.
3. Fixed4 epochs/512 updates,exact binding order,lr5e-5/clip1/weight_decay0,
   original STE/all trainable groups. Final512,not DEV-best selection. First,
   first-longest and final updates record all core/bank/norm F64 gradient groups.
   Actual memory/complete CPU snapshot transfer/through-exit cost charged.
4. Predetermined exploratory gates:FIT caseKL<=.50 initial,DEV<=.75 initial
   and<=1.0/label-disagreement<=20%;every DEV domain<=2.0/<=35%. At least8
   visited banks/site/phase,finite complete512 updates. These are calibration
   eligibility,not final quality. No metric-driven early stop;late online plateau
   diagnostic only. Actual caps reserve60s for fault recovery within3600s.
5. ALL160 final outputs once after fixed final checkpoint. Inspect terminal,
   any fault,resource caps,checkpoint and independent saved-only metrics/state
   before promotion. Then freeze NEW export/C-prefix protocol against saved
   learner logits;old13/32PASS/19FAIL retained. Frozen strict native criterion,
   plus source-relative outputs on actual C artifact before long adaptation.

The [saved-only auditor](CHATBOT_HYBRID_RECOVERY_AUDIT_PROTOCOL_20261009.md)
is prepared/UNEXECUTED,including partial-state adoption. Compile/review and
extend launcher schema only AFTER actual original exit. Archived original
launcher bytes match its original bound SHA; new launcher binds separately.

Frozen local cap<=3600s family/16GiB OS/11GiB allocated GPU/12GiB reserved/
12GiB output. Initial+final logits3,012,604,816B plus~3.06GB final/~3.06GB
possible recovery checkpoints,small metadata within cap. Actual
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
