# METH490: separate normalized conditional root mass, ONE fixed recipe

Prospective, 6 October 2026. Goal ACTIVE/INCOMPLETE. No fit/control/value
observation has occurred in this namespace at this freeze.

## Uncertainty and decision

Can a separately learned affine sigmoid head reproduce the original root
conditional mass with a conservative path-error allowance, while avoiding
the max-support representation480 and the fixed second-moment formula481?
This is a necessary filter for ONE proposed hierarchical mass representation.
It does not fit the winner, refute other representations, or complete routing.

Reuse ALL qualified479 canonical inputs, original root mass/ID/max targets,
ownership, query links, balanced memberships and the independently admitted
R2/R3 evidence. Only n128 and its twelve roots are in scope. No capture, source
model inference, completed-main/audit replay, GPU, T4, install or download.
The original source is the same7,415,217,408-parameter artifact as489CPU n128.

## Frozen mathematical recipe

For root children L/R, original F32 exponential terms and sorted-ID F64 child
sums define `y=Z_R/(Z_L+Z_R)`. Preserve every target, including0/1; no epsilon
floor, rare-ID removal, development/validation mixing or ID/role/book selection.
The root of the true source winner always has positive selected-child mass.

Fit ONE affine logit `z=theta^T((x-mu)/sigma)+b` per bank. mu and sigma are
development-only book-balanced moments. A coordinate constant on development
has mu=its exact constant value and sigma=1; no variance epsilon or feature
selection. All768 F32 query coordinates are used; there are769 parameters.

Development has159414unique IDs (14848/bank0..5,11721/bank6..11); validation-only
has79458. Total238872, original387036occurrences. Books0..63 and128..191 only
train; books64..127 remain consumed diagnostics. Equal128book weight, uniform
over that book's distinct UIDs per bank, summed for UIDs shared across books,
exactly the preexisting480 ownership convention. No fresh-held-out claim.

Loss is stable soft-target binary cross entropy:

```
sum_i omega_i [ y_i*logaddexp(0,-z_i)+(1-y_i)*logaddexp(0,z_i) ]
gradient = Phi^T [ omega*(sigmoid(z)-y) ].
```

Initial theta/bias and moments are zero. EXACTLY32 full development updates
per root, Adam learning rate.03, beta1=.9,beta2=.999, epsilon1e-8, standard
bias corrections. No early stop/restart/grid/checkpoint choice or validation
feedback. All384updates, parameters before/after, gradients, both moments,
losses, standardization, development UID/weights and final loss are retained.
Finite-step Adam is not an optimum certificate, even though the ideal BCE
objective for fixed features is convex. A failure cannot refute the affine class.

Export raw coefficients `a=theta/sigma`; bias is b minus the ascending-coordinate
F64 sum of a_j*mu_j. Cast a/bias to F32. The finite C score uses original AVX
two four-lane F64 accumulators, pair addition, lane0..3 sum, F64 addition of
F32 bias, then one F32 cast. This is an explicit new affine bias operation.

For finite F32 z, use t=exp(-abs(z)), small=t/(1+t), large=1/(1+t) in F64.
For z>=0 emit qL=small,qR=large, otherwise swap, then independently cast both
to F32. No complementary `1-qR` shortcut, clipping or probability floor.
Ideal real outputs sum to1; the finite pair is allowed absolute sum error
<=2^-24 and is checked on every query. Finite normalization is not source fidelity.

## Controls, native codec and primary scientific gates

Before fit: soft-target gradient finite differences including y=0/1,
stable sigmoid extreme/zero controls; Decimal100 reference for nine native
logits[-1000,-80,-20,-1,0,1,20,80,1000]; two ordered768dot cancellation/scale
controls; actual RNE/FTZ/DAZ/CPU0 check. One bad-head-magic native invocation
must exit2 without a prediction file. No control sweep or replay.

One compile: O3/C11/x86-64-v3/no-fast-math/ffp-contract-off, bound existing
LLVM-MinGW/MSVCRT. No engine edit. One full native prediction invocation reads
ALL original unique inputs once and writes UID/F32logit/F32qL/F32qR. Compare
ALL ordered-dot logits BYTE, then preserve the actual native probabilities.

Scientific decision AFTER ALL12fits and ALL238872predictions, not a selected
passing bank. Let `eta=log(1.01)/7`. On the root child containing the true
native winner, require `abs(log(q_candidate/q_source))<=eta` for EVERY
development AND EVERY consumed-validation UID, with complete original
occurrence/book/source-ID reports. A zero candidate selected probability has
infinite error and fails; save it explicitly, never replace it by epsilon.
The JSON statistics are null when their maximum/mean is infinite, accompanied
by zero-probability/error counts. Binary diagnostics preserve infinities.

The ideal sum of seven such absolute log errors is sufficient for1% relative
leaf probability. A root pass alone does NOT admit the other126nodes/bank,
their composition, finite final rounding, winner geometry or end-to-end quality.
Those require further explicit gates. A failure closes THIS fixed32-update
uniform-budget recipe, not sigmoid normalization, all partitions, all training
or every nonuniform error budget. Time/resource failure is inconclusive.

## Resources, first data and independent admission

Main hard600s INCLUDING metadata, compile, waits and output accounting; combined
parent/native/observed descendants <=1GiB, outputs<=64MiB. Audit hard300s and
512MiB; no native or optimizer refit. CPU0/BLAS1. Streaming2048/1024records,
one bank at a time, avoids full resident unique/target mappings. Input byte
reads are charged; read-only hashes do not interpret model values.

Initial, each new child and each optimizer-step boundary checks foreign Python/
clang/native science; quiet waits<=240s each within the SAME total hard budget.
Only the exact authorized publisher daemon is exempt. Foreign work is never
killed or modified. Boundaries do not prove continuous machine exclusivity;
this inquiry makes no model/token-rate/DRAM or causal timing claim.

All scientific files/protocol/helpers must be committed before metadata binding;
binding is committed before sole main. Fresh full payload, data, runtime and
compiler/system catalogs are hashed before numerical imports and again by the
audit. Source479retention is a qualified prerequisite; the old6224capture files
and complete127-node audit are not replayed. Digest-linked direct inputs and
all original first faults remain retained. Unrelated three files/engine bytes
are checked before and after. No old namespace may rerun.

Operational-only context code is shared for hashing/clocks/process receipts;
the audit imports neither the inquiry nor its fitting/mathematics. It uses
independent wire offsets, source child sums, weights, gradient accumulation,
Adam state checks, raw export and scalar-lane native dot reconstruction. It
checks ALL384saved updates, not a newly fitted model. An independent stable
log-sigmoid reference checks both native F32probabilities within1ULP; actual
native probabilities determine the scientific error. All39unique/72occurrence/
4608book/9216source-ID views are independently aggregated. Full root schema,
predictions, normalized moments, physical arithmetic and scientific decision
must pass admission even if the recipe fails.

Each sole main/audit gets a literal tool/session/terminal/exit registration,
actual process PID/create-time, output hashes and Windows Event1000 query.
First fault/partials are retained before a separately numbered repair; never
reinvoke attempted main/audit/codec namespaces. The auditor has a metadata-only
retention path for a failed main and cannot promote incomplete fits.
Final admission links the actual registrations, RAW/RET and both Windows
instances. The full goal remains ACTIVE/INCOMPLETE for either result.
