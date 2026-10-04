# M433 protocol: output-benefit bound under preservation and fixed exposure

Freeze NEW433 controller/THIS protocol before FIRST compile/numeric outcome.
431 frozenb287b91/retained55e0cb5, complete6240-step additive pilot FAIL7/9.
Raw648f37d9b0708488d1491b6af91776efb0bf062084f6cffb317e4af580ccb9e4.
Close431/426 recipes; do not tune/retrain/gate-select or acquire new quality data.
Uncertainty: is the required prediction benefit feasible even for arbitrary
output distributions under frozen donor-preservation demands? How much fixed
per-ID coverage remains before/after431's final gate? A positive bound is NOT
capacity transferred or a deployable model, but can change the next diagnostic.

## Data and preserved bindings

ONLY consumed405/418/420 paired records,24books/all4cases/14positions each:
1008development18books and336validation6books. Full32128 native teacher logits,
not original F32 donor or genuinely fresh held-out quality. Natural cohorts
excluded. Bind committed418/420/431 raw and all recorded helpers; fresh full
4181936 outputs/420384 complete archives/43110 complete artifacts and original
374/389 binary hashes. Probability streams/data keys/classifier IDs/exposures
must match431. Extra source payload hashes inherited431, NOT freshly reread;
no actual source inference in this output-only diagnostic. Declare that scope.

Compute native-teacher p=probability(logits256), r=probability(logits128) with
immutable R422 probability function, exact ALL1344 stream hashes431. Independent
stable log-softmax logp/logr, exp consistency and mass norm<=5e-13. CE via
sum(-p*logq), not flooring tiny tails. Baseline CE compared both original
R422 loss and431 summaries within1e-10. Stable logaddexp mixtures avoid 0/0 or
clipping underflow tails. Keep full source logit provenance and recomputable
mixture weights; save per-position metrics, not a fictitious model artifact.

## Convex output oracle and derived bound

For validation allow an arbitrary vocabulary distribution q_i at every position:

    minimize F(q)=mean_i [ .5 CE(p_i,q_i) + .5 CE(r_i,q_i) ]
    subject to mean_i KL(p_i||q_i)<=.02
               every book mean_i KL(p_i||q_i)<=.05
               q_i normalized, nonnegative

This is the same original256 CE increase, since CE(p,q)-H(p)=KL(p||q).
Unconstrained optimum q_i=(p_i+r_i)/2, with F=mean entropy(q_i).
For equal56positions/book B=6, use Lagrangian

    L=F+lambda(mean KL-.02)+(1/B)sum_b mu_b(book KL-.05).

Our derivation: stationarity minimizes weighted CE at
q_i=(t_b p_i+r_i)/(t_b+1), t_b=1+2lambda+2mu_b>=1.
KL256 decreases monotonically as t increases. Compute minimum t_book,b>=1
that meets book cap, then minimum t_global>=1 such that mean KL at
t_b=max(t_global,t_book,b) meets global cap. Recover lambda=(t_global-1)/2,
mu_b=(t_b-t_global)/2. All constraints/multipliers complementary up to root
tolerance. This is a deterministic constraint root, not a model/gate sweep.

KKT/weak-duality foundation: [Boyd and Vandenberghe, Convex Optimization,
chapter5](https://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf). Problem-specific
weighted-mixture formula above is our derivation, not a quoted model result.
Feasible q=p is strictly within positive KL budgets. L minimum at the computed
weighted mixture gives dual lower bound D=F+lambda(meanKL-.02)+mean(mu*(bookKL-.05)).
Require primal feasibility<=budgets+1e-12, primal-dual gap>=-1e-12 and<=1e-8,
simplex stationarity relative<=1e-10, CE/KL identity<=1e-10, norm<=5e-13.
This is a numerical certificate at stated F64 tolerances, not an interval proof.

Root algorithm: evaluate t=1; if infeasible bracket by doubling t from2,
at most32doublings; at most60bisections, terminate interval width<=1e-12*max(1,t).
Always select feasible high endpoint. Fail if bracket/termination/bounds fail.
No objective thresholds or previous9 capacity criteria changed.

## Independent analytic fixture and negative controls BEFORE actual bound

Two books x two positions x vocabulary3. Fixed
p=[[.7,.2,.1],[.2,.7,.1],[.3,.3,.4],[.4,.4,.2]],
r=[[.1,.2,.7],[.1,.2,.7],[.35,.25,.4],[.35,.45,.2]].
Construct known t_global2/book[4,2] with independent Decimal80 scalar arithmetic,
define book/global caps from known q's first-book/overall KL. Require book2
below first-book cap. Solver must recover t within1e-8, q and CE within1e-10,
all general primal/dual constraints. Unconstrained entropy identity<=1e-12.
Normalized geometric mixture must worsen mean CE by>1e-3; incorrect missing
/2 in recovered multipliers must violate stationarity by>1e-3. Save all fixture
probabilities/caps/known/solved/incorrect values. No numeric preflight before freeze.

## Exposures and decisions

For ALL1344 paired positions recover source128 actual route labels from418
captures, classifier actual IDs/gate masks from431. Counts exactly match431;
factor update counts=3*development exposure for both arms. Record predicted-ID
and teacher-label dev/val counts, unseen actual IDs/teacher-label val positions,
classifier accuracy, gate-on counts and cardinality with>=2val positions before
and after gate. This is a necessary count bound, not an inferred per-ID gain.
Do not choose a subset, gate or training threshold from these numbers.

At primary336 validation points, attained q meeting preservation and BOTH
mixture/teacher128 relative gain>=1% establishes OUTPUT-level feasibility.
If baseline_mix-D <1%baseline_mix-1e-8, mixture target is numerically impossible
under these constraints even without architecture. Otherwise report unresolved
conditions, no false infeasibility. Positive output bound only licenses NEW
frozen-checkpoint function-versus-selector diagnostic, never431 retuning or
transfer/capacity/quality/rate claims. Existing ALL9 capacity criteria unchanged.

## Cost, stop and retention

<=120seconds total including bindings, RSS/Windows peak<=2GiB/output<=16MiB,
CPU0/Torch1/BLAS1/Torch2.6.0+cu124/NumPy2.4.6, no modeljob overlaps. Preserve
exact publisher daemons. No GPU/T4/network/new corpus/fitting/native C changes.
Stop at FIRST failed binding/fixture/root/certificate/resource gate, retain
failure before NEW numbered repair. Three archives: tiny fixture, ALL1344
per-position metrics and336 constrained metrics/multipliers; full hashes/sizes,
root brackets/evaluations, provenance/exposures/certificates/resources in raw.
No large probability checkpoint; q fully recomputable from immutable parent
full-logit archives and saved t_b. All previous source C binaries unchanged.
Goal active/incomplete, useful RAM-scale n/LUT/realDRAM/another family/~100B open.

After freeze:
`.venv/Scripts/python.exe benchmarks/native_expert_scaling/meth433_switch_output_bound.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth433_switch_output_bound_result.json`

## Sole procedural repair433 after retained432 first failure

432 frozenb5a21f8 analytic Decimal80 fixture passed, first actual book CE identity
failed before any actual output bound. Raw3f23146cf16a818ff442f1d0b34e3f1b805d188405e3f53b7412399d7d4c0aae.
Original comparator calls R422.numpy_loss with F32 logits, whereas immutable431
and new stable log-softmax use F64. NEW433 changes ONLY comparator caller to
explicit F64. Immutable432 bound/math/fixture, probabilities, masks/inputs,
solver/root/caps/thresholds/budget unchanged. No pilot/target change.

Bind432 raw/helpers/fixture; retain old F32 first-book ALL56x2 CEs and failed
maxerror>1e-10 alongside F64 comparator and independent logsoftmax CEs<=SAME1e-10.
First fixture SHA must remain unchanged; new output has fourth comparator
archive in addition to original three. Budget still120sec/2GiB/16MiB. Original
432 remainsFAIL; only actual completed433 result can resolve output feasibility.
