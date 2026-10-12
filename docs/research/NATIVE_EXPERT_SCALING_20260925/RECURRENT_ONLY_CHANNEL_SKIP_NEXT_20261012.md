# Recurrent-only channel projection with full instantaneous skip

12 October 2026. **SELECTED / PROPOSED / UNIMPLEMENTED / UNBOUND.**

## Uncertainty and existing evidence

The balanced-head assay already retained full source generators, B/C, delta and
gate. Its 384-channel latent scans projected the entire ungated output, including
the instantaneous `D*x` skip. Existing fixed8/adaptive384 bases were fitted on
that entire output. No claim that those earlier generators were compressed.

For each source head, all 64 channels share scalar decay `a(t)`, delta and B/C.
Thus a static channel encoder `W^T` commutes with the within-head recurrence:

```
S(t) = a(t) S(t-1) + delta(t) x(t) B(t)^T
S'(t) = W^T S(t)
y = S C + D*x
ym = S' C + D*W^T x
y_old = V ym
y_new = V (ym - D*W^T x) + D*x
      = y_old + D*(x - V W^T x)
```

No commutation across different-decay heads is assumed. Retaining full stateless
skip can increase per-token work while leaving recurrent memory unchanged.
This is a hypothesis for compact memory plus rich static functions, not proof
of cheap/native selective computation.

## Finite first diagnostic

Reuse frozen fixed8/adaptive384 maps and actual stored 384-channel scans, all48
FIT/DEV cases and three sites0/12/23. No new basis, fit, scan, model forward or
optimization. Actual source x/gate/y/output and weights are already captured.
Compare old and corrected y, recurrent/skip errors and full gated output under
both existing modes: source-denominator oracle and reconstructed denominator.
The latter recomputes RMS jointly on the combined gated output; separate
normalization of recurrent and skip branches is mathematically invalid.

Freeze source, operands, tolerances and resource limits before computation.
Repeat every old output control at its original reconstruction tolerance;
independently audit all new stored outputs, statistics and original operands.
Retain existing local budgets: DEV equal-case mean output RMS<=.1,
worst case<=.2, worst domain mean<=.2, centered mean<=.1. These are local
operator gates, not chatbot quality admission. Complete audit even if they fail.

Initial bound: producer900s, audit1200s, OS6GiB, CUDA allocated2GiB/reserved3GiB,
new output<=3GiB. No T4/new source data/RESERVED/longalign/native speed claims.
If implementation reveals a need for hours, announce, bind/launch and pause
under the user's instruction. No delegation.

## Decisions enabled

If original budgets pass, qualify an actual causal/native pilot and cost before
long adaptation. If retained skip improves errors but budgets fail, quantify
remaining recurrent error and consider FIT-only recurrent-output basis fitting.
If it fails to help, preserve the decomposition/cross-term evidence and revise
the causal representation. No repeated old all-output basis fitting selected.
Combining this with the separate state256->96 projection is a later composition
test, not assumed by this diagnostic. Full source nonlinear generators/gates and
3072 skip features remain injected, so no compact chatbot exists yet.
