# Whole-project reassessment after496: geometry and missing information

6 October2026. Goal ACTIVE/INCOMPLETE. [496 result](METH_496_READOUT_ERROR_RESULT_20261006.md)
is a complete independently admitted diagnosis of the unchanged495 artifact.
It selects analysis of fitting/information before a precision-only intervention.
This is progress on one local bottleneck, not a complete donor conversion.

## 1. Three separate obligations

| Obligation | Concrete evidence | Still required |
| --- | --- | --- |
| preserve conditional function |495 native bytes/equations correct;496 separates fit/coefficient/arithmetic error |Improve correct-ID weighted function and unseen-state prediction |
| choose useful function and mass |495 fixed key recipe admitted, ID56.0% development /22.7% consumed validation |Reliable decision/function-aware choice; finite128 Adam optimum not certified |
| grow useful n with bounded active work |small-donor123/183 learned128->1280;373 bounded local64->256 cost;489 source CPU128>=50 positive |SAME transformed whole artifact, fresh quality, causal useful n, physical DRAM, actual additional scales |

In495 canonical coupled RMS is71.08%/75.80%, whereas correct-ID RMS is
2.80%/8.92%. Routing is an independent large obstacle. Quantization changes
cannot fix wrong function identities. Correct identities alone also fail.
The respective errors and cross terms must remain separately measured.

## 2. What496 actually excludes

On all six declared occurrence groups, saved unquantized fitted U RMS is
2.38%..6.98%; coefficient Q-U RMS1.27%..1.32%; P-Q RMS ratio4.49e-8..4.59e-8.
The arithmetic term is small relative to the other measured terms. Increasing
precision alone leaves the saved fitted function above1%. Coefficient encoding
also needs improvement eventually; it exceeds1% itself and harms rare fitted
states severely. These are statements about this fixed artifact and domain.

The frozen readout solves are regularized by a weight/Gaussian prior. They are
not unrestricted least-squares projections. An observed regularized residual
cannot prove that the features lack representational capacity.

## 3. Data-only ambiguity: an exact bound from counts

Write H_e as m_e by513: the512 dequantized absolute features plus bias.
There are768 output rows in C_e. Each development equation constrains the
product H_e C_e^T. Every m_e<=308<513. Therefore, without any numerical rank
decision,

```
dim ker(H_e) >= 513-m_e
sum_e dim ker(H_e) >= 128*513-11721 = 53943
sum_e 768*dim ker(H_e) >= 41428224.
```

Adding a row coefficient v in ker(H_e) changes no development output. Its
prediction on a new feature h changes by h v. The data identify the function
only on their training row space. Regularization can choose among these
equivalent data fits, but its unique minimizer is not additional evidence.
Source weights offer additional information only to the extent that the
constructed prior transports their relevant function into these directions.

Even all consumed source UIDs have at most498 rows per expert. They do not
exhaust513 directions. This does not license fitting consumed validation or
declare a rank: it is an exact maximum possible rank bound. Increasing n
while holding calibration-state count fixed can worsen per-expert coverage.
RAM capacity by itself supplies no observations of newly added functions.

The rare groups illustrate the distinction: development fit error0.019%/
0.147% versus consumed-validation182.9%/168.6%. Few states, large discrepancy.
We have no certificate that any new interpolating coefficients would be
stable, quantizable or predictive.

## 4. The precise regularized problem

With R=Y-saved_physical_L and rows H, the495 ideal real objective is

```
J(C) = ||H C^T-R||_F²/m + lambda ||(C-Cprior)T||_F²
lambda=.01; Kreg=T T^T positive definite
A=H^T H/m + lambda Kreg
rhs=R^T H/m + lambda Cprior Kreg
C* A = rhs.
```

Kreg uses the frozen Gaussian prior geometry and its positive ridge; Cprior
uses the serialized source prior times the fixed empirical amplitude. C32 is
the serialized finite solve. Equation correctness and uniqueness of J's
minimizer do not certify the prediction optimum of unregularized data loss.

Whitening with D=(C-Cprior)T, Z=H T^{-T}/sqrt(m) and
S=(R-H Cprior^T)/sqrt(m) gives ||Z D^T-S||²+lambda||D||². If Z=U Sigma V^T,
the fitted correction responds to a singular direction by
sigma²/(sigma²+lambda); the residual retains lambda/(sigma²+lambda).
This algebra separates shrinkage from representation. Any numerical spectrum
would require its own prospective controls; a floating cutoff cannot be
silently promoted to an exact class floor. No spectrum was observed in496.

## 5. Weighting changes the odd/even problem

For the ideal real bias-free ReLU source with the qualified weight values,

```
F(x)=L0 x + E(x), L0=.5 V W, E(x)=.5 V abs(Wx), E(-x)=E(x).
p+(x)=(p(x)+p(-x))/2; p-(x)=(p(x)-p(-x))/2
(p F)_odd = p+ L0 x + p- E(x)
(p F)_even = p- L0 x + p+ E(x).
```

The ideal hybrid a_e L0 x+B_e abs(Ax)+c_e has fixed odd part a_e L0 x.
The weighted target's odd part also contains varying p+ and p- terms. Thus
the exact ReLU identity for unweighted F does not supply an exact transfer
identity for weighted pF. Holding weighted L fixed and fitting only even
features can miss source information even with an exact unweighted even prior.

This is a global real algebra observation. It is not a measured496 error
attribution or a lower bound on the selected-expert data: an expert's routing
cell need not contain both x and -x, and source/F32/A16 arithmetic differs
from the ideal formula. No new counterfactual source evaluation was run.
The empirical finite row geometry must be resolved before changing the class.

## 6. Next497: exact finite-feature rank certificates, no fitted candidate

The next narrow question is whether each development H_e has full row rank.
If so, every finite development target admits a real interpolating readout
in the same unconstrained feature class; its observed training error cannot
be explained as lack of finite-sample representability. This does not imply
that such a readout is small, quantizable, stable or predictive.

Use the already saved qphi I16/alpha F32 and original UID/occurrence domains.
These H entries are exact dyadic rationals: integer q times binary F32 alpha,
and bias1. Map them into one fixed odd prime field after exact exponent/
mantissa decoding. A nonzero m_e by m_e pivot minor modulo the prime proves
the same rational minor is nonzero. This yields exact real row-rank evidence
without a floating singular-value threshold. A deficient modular result is
only INCONCLUSIVE for full rational rank; do not label it a class failure.

Freeze one prime, deterministic pivot order, integer overflow bounds, the
rank/certificate wire, full domain, new exact controls, CPU/RAM/wall/output
limits and independent minor verification before ANY497 observation. Verify
prime admissibility and dyadic denominator inverses rather than assume them.
Retain all128 experts including the empty expert; count free directions and
which consumed-validation feature directions extend the development span.
Novelty from rank differences requires certified development rank (for example,
full row rank); differences between two inconclusive modular lower bounds
alone do not certify the real dimension gained.
No consumed target fitting, alternate predictor, coefficient export, optimizer
update, source FFN/native/model call or width/prior/ID sweep. Consumed features
may establish finite geometric novelty only; this does not establish utility.

Use only actual feature/UID/occurrence bytes, current qualification records,
helpers and runtime. The compiler, physical bank, target arrays and old full
source payloads are not necessary for this rank question. Full operational
protocol and helper freeze are required; this planning section alone does
not authorize an unfrozen numerical invocation.

If full row-rank certificates pass, prioritize regularization/prior transport
and development information before growing the dictionary. If they do not,
inspect the exact finite dependence/collision witnesses or report inconclusive.
Only after this distinction choose one concrete learner change with its
physical coefficient cost and independent fresh quality gates.

## 7. How this returns to large n

A useful large bank needs an informative shared map, private coefficients
that retain donor knowledge in unobserved directions, and reliable selection.
Our existing four rows per expert initialization makes r=4n, hence active
work grows with n; it is not the required RAM-only scaling rule. A future
shared dictionary must have its active width fixed or explicitly bounded.
Simply adding private byte blocks does not establish useful functions.

Product-key choice algebra can reduce score work to balanced O(sqrt(n)*(D+r))
with constant selected count, but adds structural constraints on source decision
regions and no measured normalization guarantee. For a growing donor softmax,
p_old_new=p_old/(1+Zextra/Zold); weighted functions must be reconsidered when
the choice set changes. There is no general algebraic implication
100B parameters=>10 times useful experts relative to10B: this is a product
target that must be measured on matched real capacity/quality/cost artifacts.

After a successful local transfer: all banks/composed source476 contexts,
own evolving states, fresh prediction/generation/tasks, SAME whole-artifact
accepted batch1>=50, physical DRAM and useful-n causal controls, actual other
families/~10B/~100B as resources permit. Source489CPU128 positive is separate;
no timing or quality inheritance from it. Completed495/496 namespaces stay
terminal. No engine integration or new resource is selected by496.
