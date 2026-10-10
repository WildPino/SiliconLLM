# One stored-gradient router-support counterfactual

11 October2026. PROPOSED/UNIMPLEMENTED/UNEXECUTED. No concurrent numerical
job during recovery seal or full recovered audit. This is a possible next
diagnostic, not a chatbot conversion result or a replacement for missing tasks.

## Isolated mathematical question

At a fixed history point, top-k routing has a selected set S at each token and
layer. The exact selected normalized mass is

    m_i = exp(z_i) / sum_{j in S} exp(z_j), i in S.

For an inactive logit z_j outside S, its local derivative is exactly zero while
S remains fixed. A parameter row for expert j that is never selected anywhere
in the complete history has zero route-mass derivative at that piece. This
does not make the selection globally differentiable or certify future support.

The current finite implementation computes all-n softmax and then selected
renormalization. Its finite backward may leave cancellation residuals outside
the complete-history selected union. This is a conditional explanation; a
specific kernel's residual has not been located or proved causal.

The grouped update maps any nonzero row gradient g to

    delta = -alpha R g / ||g||,
    R = max(||p||, 1e-5 sqrt(d)).

The magnitude is alpha R independently of ||g||; the mapping is discontinuous
at g=0. Metadata for the first campaign direction reports5,529 never-selected
router rows, tiny nonzero gradients and full-radius movements in11,058
weight/bias groups. Coefficient-level campaign confirmation is pending the
complete recovered audit. The already separately qualified saved first-point
gradient and native baseline remain usable prerequisites.

## Fixed intervention and comparison

Use the actual first58-ID history, its already qualified changed92-master
initial state and saved90-gradient tensors. Use the entire GPU route union at
all58 tokens/layer, not only response labels, the final token or C unions.
For each layer set router weight and bias gradient rows outside that union to
exact F32 positive zero. Leave every other gradient coefficient unchanged.
This is a stored-gradient counterfactual: no new history/backward, donor,
DEV optimization, export of an unchanged baseline or broader epoch replay.

Fix alpha=.001, the original first direction's selected candidate. Reuse that
candidate and its native result as the reference. Construct exactly ONE new
candidate with the same original grouped proposal formula, scale floors,
F64-to-F32 rounding and exporter. Do not search a new alpha after observing
metrics. Save mask provenance and changed router arrays/delta metadata;
sharing immutable other saved gradient tensors is allowed, overwriting them
is not. The runner and new exact binding still need implementation.

Required causal invariants:

- Mask equals the complete original GPU history union; rank and mass files
  match the qualified parent hashes and forward/backward route bits.
- All non-router proposed parameter bits equal the original candidate.
- All selected router rows' proposed bits equal the original candidate.
- All excluded router rows' proposed bits equal the original baseline.
- Independent fsum checks verify radius, rounding and dot products. Report
  both dot products against the original and masked gradients.
- Independently reconstruct the transition, all packing fields/trits/scales,
  stable-cell certificates and integer witnesses; run and fully audit ONE
  new original C prefix on the same history and labels.
- Compare named expert identity/order/mass and all scores/label metrics with
  both existing baseline and existing candidate; do not infer hidden-state
  equality from matching IDs.

No GPU/C score check at the changed candidate is available without a new
candidate GPU history. The existing baseline bridge remains qualified; this
experiment measures the finite C consequence of a specific parameter
intervention. It must not claim a candidate-wide new causal bridge.

## Decisions fixed before observation

Always complete mechanical/byte/numerical audit, including quality failure.
Use original metric tolerances and route normalization criteria.

1. If mask/bit/packing invariants fail, the intervention is mechanically invalid;
   preserve the fault and repair the new adapter before interpreting quality.
2. If the new candidate improves weighted KL and does not regress first KL
   relative to the original candidate, it supports usefulness of removing those
   specific inactive-row movements on THIS case. It does not locate the
   backward kernel defect or establish long-history/general quality.
3. If it preserves those metrics within the original absolute tolerance, it
   supports equivalence of their finite C consequence on THIS case, not on all
   histories or n. Record removed movements and changed route exposure.
4. If it worsens the finite metrics, this counterfactual does not support using
   this mask to improve this case. Still inspect routes/quantization cells;
   ideal local derivative support alone is not a finite-step descent guarantee.

Implementation/resource budget must be frozen separately before running.
Expected one proposal/archive/export/native prefix plus a complete independent
audit, using the established individual-operation timings, no hours-long epoch.
No acceptance into a new training state is authorized by this diagnostic record.

## Why this addresses n without claiming scaling success

With larger n and fixed k/history length, the number of unconsulted router rows
can grow. The update rule must distinguish structurally zero information from
tiny numerical residuals before treating each row as a unit direction. This
counterfactual isolates that issue from donor transfer, core learning,
quantization and choice among many genuine expert functions. It does not reduce
the dense master/gradient O(n) RAM cost, implement structured CPU selection/mass,
preserve100B donor capacity, or demonstrate useful same-artifact50 token/s.

Canonical16 tasks/own-answer followup on the audited final campaign pack remain
the immediate missing scientific assessment. Select the diagnostic order only
after the recovered audit closes and the actual outstanding failure is known.
