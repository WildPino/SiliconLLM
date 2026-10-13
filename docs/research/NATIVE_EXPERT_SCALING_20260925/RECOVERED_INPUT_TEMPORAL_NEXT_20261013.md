# Recovered source-function inputs: temporal propagation before calibration

13 October2026. **ORIGINAL PROPOSAL / NOW COMPLETE WITH FAILURES.**
[Full result](RECOVERED_INPUT_TEMPORAL_RESULT_20261013.md) supersedes operation:
all144 own-output flagsPASS,one final-state1%FAIL;one separate source-denom
BF16 numeric flag retained/full audit complete. State-mechanism auditPASS.
[True input capture next](TRUE_GENERATOR_INPUT_CAPTURE_NEXT_20261013.md) selected,
PROPOSED/UNIMPLEMENTED/UNBOUND. Original pre-observation rationale below retained.
Execute only after the full [input recovery audit](GATE_INPUT_RECOVERY_RESULT_20261013.md)
passes its frozen numerical controls. This is the next uncertainty to resolve,
before treating recovered inputs as a usable calibration proxy for cheaper
generators. It is not an autonomous encoder or a chatbot quality experiment.

## Question, reused evidence and decision

The fixed recovered BF16 inputs reproduce all five source generator fields
within 1% whole-history relative RMS on all 144 case/site pairs. Centered
variation is less well preserved: worst site23 B is 1.381%, x is 1.070%.
Small operand errors need not yield small recurrent-state or output errors.
Reuse the independently qualified native full3072/state96 operator, its fixed
FIT Ws/Vs and all48 original histories at sites0/12/23. Change ONLY supplied
x/B/C/delta/gate to the already frozen regenerated BF16 arrays. No refit,
new rank, DEV-selected recovery, native timing or new source-model execution.

If temporal fidelity passes, the data may initialize source-function fitting
with an explicit approximation boundary. It still supplies no bound on the
unknown true input or model-generated future histories. If it fails, retain
the full result and inspect which recurrent forcing/decay/readout term dominates;
true-input capture or an uncertainty-aware fitting objective then becomes a
separate bounded proposal. Do not select a different recovery on DEV.

## Algebra to inspect

For a channel in head h, with fixed source scalar A_h and the fixed state96 maps:

    a_t = exp(delta_t A_h)
    U_t = (delta_t x_t) (B_t Ws)
    S_t = a_t S_(t-1) + U_t
    y_t = S_t (C_t Vs) + D_h x_t

Using hats for regenerated operands, the real-arithmetic identity is

    Delta S_t = hat(a_t) Delta S_(t-1)
                + (hat(a_t)-a_t) S_(t-1) + (hat(U_t)-U_t)
    Delta y_t = Delta S_t (C_t Vs)
                + hat(S_t) ((hat(C_t)-C_t) Vs) + D_h Delta x_t.

These separate accumulated error, changed decay, input forcing, C readout and
the full skip. They are not exact identities for differently rounded F32/FMA
updates; numerical residuals must be measured under the original controls.
Negative A and positive delta imply each scalar decay lies in (0,1], but this
alone gives no small output-error bound: input forcing, C magnification,
cancellation, gate and RMS normalization remain involved.

Let u = y * SiLU(gate), r = sqrt(mean(u^2)+epsilon). The local differential
before BF16 rounding/output contraction is

    d[gamma*u/r] = gamma/r * [du - u*(u.dot(du))/(3072*r*r)].

Thus radial and angular perturbations differ; report both fixed original
source-denominator and actual reconstructed-denominator outputs, using the
SAME original source y/gate to define the diagnostic denominator. The fixed
source denominator is not a runtime claim.

## Fixed primary checks, before observation

- Every case/site, own-denominator BF16 physical output versus the corresponding
  qualified original-operand native output: whole-history relative RMS <=.01.
- Every case/site, full final F32 recurrent state versus corresponding qualified
  original-operand final state: relative RMS <=.01, reported separately from
  output consistency. Future-use qualification requires both.
- Retain centered/label/domain/mean/worst, every latent/projection/final-state
  metric, and diagnostic fixed-source-denominator differences. No admission
  based on a single aggregate or gate-only replay.
- Original donor-relative local own-norm gates remain mean<=.1, centered mean
  <=.1, worst<=.2, worst domain mean<=.2; do not relax after observation.
- Independent every-timestep F64 recurrence, fixed export bits, BF16 normalized
  vectors, complete AVX raw F32 output versus independent scalar FMA bits,
  all arrays/metrics/gates and before-after hashes. Original native numeric
  tolerances apply (projection1e-6, scan/state/control1e-4, scalar1e-8,
  raw-score bit mismatches0). Complete the audit even if temporal gates fail.

The new 1% temporal approximation gates are separate from native arithmetic
tolerances and whole-model quality. Only own-norm output/final state are the
primary approximation gates; fixed-source norm is diagnostic, never an
alternative selected after seeing DEV.

## Resources and route to the goal

Proposed CPU producer900s/audit1200s, OS6GiB, new E family<=6GiB, six BLAS
threads for audit/one actual native thread. Bind the exact source, protocol,
existing DLL/maps, recovery report/adjudication/arrays, corpus, resolved
runtime and launch before execution. No GPU/T4/RESERVED/source forward.
No hours-long allocation now; follow the user's announce/launch/PAUSE rule
if a later stage requires hours.

After this gate, focus conversion work on expensive input/output functions:
FIT-only calibration of a cheap generator/readout and selectively consulted
ternary functions, with causal information carried by the shared core.
Validate joined layers and own histories, then the same useful artifact's
>=50 batch1 tokens/s, actual expert DRAM, CPU IDs AND masses and useful n.
The full dense generator/readout is the current reference, not an established
target implementation. Multiple donor families/scales and conversion O(n)
masters/gradients remain unresolved.
