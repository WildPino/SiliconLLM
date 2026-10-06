# Whole-project reassessment after497: fit identifiable information

6 October2026. Goal ACTIVE/INCOMPLETE. [497 result](METH_497_EXACT_FEATURE_RANK_RESULT_20261006.md)
supplies exact finite feature-rank evidence on all original source IDs. No
fitted candidate, model quality or large-n claim. Previous496 error accounting
and495 fixed learned/native failure remain qualified and unchanged.

## 1. The obstacle is now more precise

All127 exposed experts have exact full row rank on development and on ALL
17,540 saved feature states. All5,819 consumed features add independent row
directions to development. Thus every finite development target is representable
by this unconstrained real readout. Yet495's regularized fitted U still has
canonical development RMS2.4802%, consumed-validation8.8280% (496). Finite
sample representability is certified; accuracy of the learner/prior and physical
coefficient constraints remain unresolved. A floating singular-value test is
no longer needed to establish this particular exact rank.

The development design has exactly53,943 total coefficient null directions
across128experts, or41,428,224 with768 free output rows. Even ALL finite
source features leave48,124 directions, or36,959,232 output coefficients.
The small calibration set cannot identify the entire private readout. Source
weights and structure must carry the missing knowledge into this geometry.

## 2. Why zero training loss would be insufficient

For one expert let H be its development design. If delta_c is in ker(H),
C and C+delta_c have identical development outputs. Full rank of the stacked
development/consumed matrix additionally proves that any prescribed finite
consumed output change can be obtained by such a training-invisible shift.
This exact finite statement has no bound on the coefficient norm or encoding.
It establishes ambiguity, not unavoidable failure: a correct source-informed
prior can still select the appropriate values in those directions.

Excellent rare development fits in496 coexist with168%..183% consumed error.
Training interpolation alone cannot establish expert knowledge transfer. The
goal requires donor-relative fresh own-state/generation/task quality, and a
useful large expert bank cannot be inferred from byte count or finite rank.

## 3. One concrete learner change: minimum source-prior displacement

The495 ideal problem minimizes data loss plus lambda=.01 Gaussian/source-prior
penalty.497 permits a different uniquely defined question: among readouts
that exactly interpolate development targets, which moves least in the same
source prior metric? This is one equality-constrained algebraic method, not
a lambda/update/width/checkpoint sweep.

For each exposed expert use the SAME513-column physical feature design H,
SAME fixed physical L, development-only R=Y-physical_L, SAME serialized
weighted Cprior and positive definite frozen Kreg. Define

```
min_C ||(C-Cprior) T||_F² subject to H C^T=R
Kreg=T T^T
E=R-H Cprior^T
M=H Kreg^-1 H^T
Cinterp=Cprior + E^T M^-1 H Kreg^-1.
```

Exact row rank and positive Kreg imply M positive definite over the reals.
The strict convex metric gives one unique nearest-prior interpolant. Derive
and implement with triangular/Cholesky solves; never materialize an inverse.
The ideal correction changes only directions observable through development
in the Kreg metric; all metric-orthogonal data-invisible directions retain
their source-prior value. This preserves the current available donor prior
where the data cannot supply information. It does not prove the prior itself
correct, nor guarantee better consumed prediction or quantizability.

Using Z=H T^{-T} and D=(C-Cprior)T, the formula is the minimum-Frobenius D
such that Z D^T=E. The current regularized solution instead filters its singular
directions by sigma²/(sigma²+m*.01); exact interpolation removes that shrinkage
on represented training directions, while changing no prior metric or width.
The scale m follows this unnormalized Z definition. There is no scientific
reason to change prior/ID/precision at the same time.

## 4. Next498 pipeline and prospective eligibility

Implement/freeze ONE full minimum-prior-displacement learner for all127
exposed original experts; ID0 retains the marked source initialization.
Reuse qualified494 source C0/calibration/Kreg,495 physical feature/inputs/L,
source493 Y/UID/occurrences, and497 exact rank qualification. No additional
teacher capture, source/model invocation or resource. No target from consumed
validation enters a solve. All128original IDs/rare/empty cells remain.

Change only the readout learner from fixed penalized fit to exact development
interpolation in the same prior metric. Freeze A/L,512width, prior metric/
amplitude, current495 product keys and the I8 readout row codec. Preserve
current keys; no completed128 Adam replay. Reuse the qualified495 native
evaluator binary on the newly exported candidate bank in NEW498 output
namespaces; current dependencies include this binary/runtime/source code,
not unused compiler SDK. Existing features stay valid because A/L remain
byte-identical. One native prediction on ALL original states is a new changed-
artifact evaluation, not a completed495 prediction/control rerun.

Conversion arithmetic must check the equality residual, minimum-prior
stationarity, actual C32 serialization and complete physical B/scales/bias
bytes, not just an optimizer loss. Use new dyadic exact controls and a full
independent audit of all127solves/128cases, full native outputs, original
reports/source/control domains and physical/logical cost. Exact rank alone
does not bound numerical conditioning: if fixed Cholesky/equality checks fail,
retain the first fault/inconclusive result. No silent SVD cutoff, extra ridge,
regularization fallback or retry is allowed without a numbered protocol.

Predeclare quality gates before ANY new fit/result. Correct-ID unquantized and
physical weighted RMS<=1% on every one of the six original occurrence groups
remain local function gates. Full coupled weighted RMS<=1% andID>=99.9% remain
the separate full local recipe gates; unchanged failed routing means no whole
recipe promotion is expected from this readout-only step. Report this honestly.
Consumed metrics are diagnostics; they do not select another predictor or
qualify fresh model quality. Bound conversion/native/audit CPU/RAM/wall/output
cost before execution and charge actual dependencies/record writes.

This document selects the algebra and scope; complete new code, fixtures,
binding/resource/fault/decision protocol must be frozen before numerical use.
No498 controls, solve or candidate observation has occurred yet.

If Uinterp still misses consumed function gates, analyze source-prior transport
and weighted-function structure rather than choose another training checkpoint.
If Uinterp passes but physical output misses, address physical coefficient
encoding with its full active/storage cost;496 already separates that variable
from arithmetic. Even a passing local function needs reliable routing/mass.

## 5. Relation to the large-n goal

The ideal nearest-prior solve costs depend on each m_e<=308 rather than a
513-square direct solve per expert: factor Kreg once, form/factor each m_e-square
kernel and map its correction. Actual conversion time/memory must still be
measured; this is an algebraic opportunity, not a speed result. Stored private
parameters grow with n; active width and selected count must stay bounded.
Increasing n without donor-informed priors or adequate exposure cannot fill
the new data-invisible directions with useful knowledge.

The unweighted identity F=L0x+.5Vabs(Wx) does not become an exact identity for
p(x)F(x) after constant-amplitude weighting.496 derives its varying odd/even
parts. The minimum-prior method inherits that prior limitation; no global
parity lower bound on selected routing cells has been established.

Routing remains independently unresolved:495 ID56.0%/22.7%, finite Adam not
optimality certified; weighted choice-set normalization changes when n grows.
Four dictionary rows per ID would increase r=4n and active work, so494's
construction is not yet a RAM-only large-n method. Cheap product-key algebra
does not establish source decision geometry or causal useful n.

After local function/control transfer: composed/all-bank source476 contexts,
own-state fresh predictions/generation/tasks, SAME whole artifact>=50 accepted
batch1, physical DRAM, useful-n causal controls and actual other families/
scales/~10B/~100B as resources permit.489 source CPU128 speed is separate.
Completed495/496/497 numerical namespaces are terminal. No engine change or
whole quality/rate/family/goal promotion follows from497.
