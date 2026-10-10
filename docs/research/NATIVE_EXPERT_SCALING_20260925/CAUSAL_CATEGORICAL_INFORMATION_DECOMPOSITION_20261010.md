# What a shared categorical head does and does not identify

10 October2026. Derived algebra, no new dataset/model/optimizer experiment.
[Completed shared fit](CAUSAL_CATEGORICAL_READOUT_RESULT_20261010.md),
[fixed-head conditional identity](PAIRED_HEAD_IMAGE_ENCODER_ALGEBRA_20261010.md).
Native assessment pending; these identities do not declare its outcome.

## Separate three hypotheses

1. Vocabulary image: can each teacher distribution q_j be approximated somewhere
   in the fixed A code family softmax(A x), with its stated code domain?
2. Shared readout: is a single linear x_j=Theta t_j sufficient for these fixed
   causal native features? Its parameter ball is distinct from a per-label ball.
3. Causal construction: can the compact original SSM/SWA+ternary functions
   construct useful x from past tokens and generalize to its own generated IDs?

The72-label per-code experiment is evidence for1 on selected labels. The new
32-update shared fit evaluates a feasible candidate for2 on all4422 FIT labels.
A poor finite candidate with certified lower0 does not refute2. Neither local
experiment proves3, useful expert-n, CPU structured routing or physical DRAM.
The actual new-artifact original C assessment addresses numerical composition,
held-out behavior and own-history tasks; it cannot certify a convex optimum.

## Why the existing lower certificate can remain uninformative

Write A_v for centered vocabulary rows, t_j whitened native features,
x_j=Theta t_j, m_j=A^T q_j and c_j=sum_v q_jv log q_jv. Equal-case weights sum1.

    F(Theta)=sum_j w_j [c_j+logsumexp(A Theta t_j)-m_j^T Theta t_j].

The variational entropy formula logsumexp(z)=max_p[p^T z+H(p)] gives a dual
bound for ANY probability vectors p_j:

    L(p)=sum_j w_j[c_j+H(p_j)] - R ||G(p)||_F,
    G(p)=sum_j w_j[A^T(p_j-q_j)] t_j^T.

For the actual p_j=softmax(A Y t_j), this is exactly

    L=F(Y)-<gradient F(Y),Y>-R||gradient F(Y)||_F.

So the frozen tangent lower is a valid Fenchel dual certificate. However, a
large radiusR=16 penalizes a stationarity residual by16 times its norm. Finite
parameter values much smaller than16 do not make that penalty disappear.
Lower0 is the universal KL bound, not evidence for an attainable zero optimum.
Closing the gap requires better stationarity or a different feasible dual
witness. Merely calling a poor point an 'information ceiling' is unjustified.
Any new numerical solver would need a new frozen protocol/changed algorithm and
actual work accounting; it must not silently replay the completed32-update dose.

## Least-squares warm map measures a separate geometric quantity

Since E_w[t t^T]=I, let y_j=phi_j/a be the stored warm source codes. The unique
least-squares map on retained support is Theta0=E_w[y t^T]. Pythagoras gives

    min_Theta E_w ||y-Theta t||^2 = E_w||y||^2-||Theta0||_F^2.

This quantity would measure linear predictability of the CHOSEN warm code,
not categorical sufficiency of every code in the same vocabulary image. A high
residual can coexist with accurate probabilities through softmax gauge/kernel
freedom or another equivalent code. A low residual can still yield a large KL
when errors align with sensitive vocabulary directions. No new residual value
is measured in this note; the stored arrays permit a separately bound analysis.

## Exact loss decomposition keeps the nonstationary term

For any reference y_j and candidate x_j,

    KL(q_j||p_x)=KL(q_j||p_y)+KL(p_y||p_x)
                 + [A^T(p_y-q_j)]^T(x_j-y_j).

Dropping the last term is valid only with the corresponding stationarity/
constraint condition. The paired warm source projection has observed nonzero
categorical error; it is not an optimum merely because a quadratic criterion
was optimized. This identity explains why quadratic retained energy97.95%
did not guarantee categorical preservation in the earlier paired codec.

## A bound relating state separation and categorical distinguishability

For a shared Frobenius-ball map, ||Theta(t_i-t_j)||<=R||t_i-t_j||. For ANY two
vocabulary rows u,v:

    |log(p_i[u]/p_i[v])-log(p_j[u]/p_j[v])|
        <= R ||A_u-A_v||_2 ||t_i-t_j||_2.

Thus an opposite teacher preference cannot be copied with arbitrarily large
log-odds margins when two features are too close under this bound. This can
motivate a FIT-only separation witness without thousands of additional model
calls. It is conditional on the fixed state/decoder/parameter domain; it does
not rule out a new causal representation or another decoder.

A more global bound uses B=max_{u,v}||A_u-A_v|| (or conservative2max||A_v||).
For delta=R B ||t_i-t_j||, the likelihood-ratio span is at mostdelta, hence
TV(p_i,p_j)<=tanh(delta/4). Let d=TV(q_i,q_j), u=max(0,d-tanh(delta/4)).
For completeness, the TV bound follows by tilting p_j with likelihood weights
whose max/min ratio is k=exp(delta). The extremal variation for a fixed span
puts weights at both endpoints. If the high-weight endpoint has p_j mass a,
TV=a(1-a)(k-1)/(1+a(k-1)). Maximizing over a gives a=1/(1+sqrt(k)) and
TV=(sqrt(k)-1)/(sqrt(k)+1)=tanh(delta/4); the delta0 case follows by continuity.
This is an algebraic envelope, not a measured separation certificate.

Triangle inequality and Pinsker in nats yield the valid two-label contribution

    w_i KL(q_i||p_i)+w_j KL(q_j||p_j)
        >= 2 w_i w_j/(w_i+w_j) * u^2.

Overlapping pairs cannot be summed as independent certificates; use disjoint
pairs or an explicit valid fractional packing. Full-V teacher TV is required
for this witness. No nonzero bound is claimed before actually measuring it.
Large decoder rows/radius may make the bound vacuous, itself a recorded result.
Approximate recovered states require adding both state uncertainty contributions
to the feature distance before claiming a lower bound for the true C states.

## Decision order

First finish native full audit and SAME-artifact quality/rates. Then decide
whether the next necessary correction is a convex shared-map certificate,
new causal-code distillation/core/functions, or a qualified numerical observer.
Choose a new observable/algorithm that can change that decision. Preserve the
original LUT/ternary/SSM/SWA cost thesis and whole-chatbot gates throughout.
