# M432 result: analytic fixture passes, first real CE comparator fails

Frozenb5a21f8; first/only run exit1 at paired_book0 before output bound/exposure.
Raw3f23146cf16a818ff442f1d0b34e3f1b805d188405e3f53b7412399d7d4c0aae.
The CE identity comparator receives F32 native logits; immutable R422.numpy_loss
does not promote dtype internally. Actual431 losses explicitly cast logits F64.
M432 stable log-softmax also uses F64. Missing caller promotion makes this
comparison test different numerical precision; do not loosen1e-10 bound.

Independent Decimal80 known-dual fixture PASS: recovered globalt2.000000000001819/
book[4,2.000000000001819], q maxerror1.010303e-14, primal/dual gap2.220446e-16,
stationarity1.776357e-16. Entropy identity2.220446e-16, incorrect geometric
mixture harm.0059402 and wrong-multiplier stationarity defect1.12853 detected.
No actual preservation-constrained output bound or exposure decision reached.

8.672s/max419123200B/3035611173B hashed. Fresh4181936 inventory/420384 full
archives/43110 outputs exact; source payload hashes inherited431, not reread.
Only fixture archive2624B SHA52cb066d56ed1b853b05fa8786f54e9dd80bb0b29bee9a9919da76e09db732e2.
CPU0/Torch1/BLAS1/no fitting/GPU/T4/new corpus/engine change. First failure
retained before repair. Final goal active/incomplete,431/426 recipes still closed.

Next NEW433: sole CE-comparator caller promotion to F64 as immutable431.
Retain old F32 first-book comparator arrays/failure and new F64 comparison;
SAME log-softmax/probabilities/solver/thresholds/protocol data and budget.
Math432 unchanged; freeze433 before actual bound/outcome, no432 promotion.
