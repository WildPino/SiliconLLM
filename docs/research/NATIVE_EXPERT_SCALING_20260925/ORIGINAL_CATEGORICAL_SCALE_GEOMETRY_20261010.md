# Relative expert steps, scale symmetries and surrogate-gradient limits

10 October2026. Mathematical derivation and two primary-source checks while
the grouped-step stored audit runs. No extra model/native/optimizer experiment.

## Published relation and scope

You, Gitman and Ginsburg's LARS uses a local step factor proportional to
||W||/||gradient|| (section4/equations4-6), controlling layer movement relative
to weight norm. Their evidence concerns large-batch CNN training. Our frozen
group rule has this algebraic form when its floor is inactive, with groups at
expert/projection, embedding-row and router-row granularity. Momentum, weight
decay, floor choices and actual quantized loss checks differ. That paper's CNN
accuracy is not evidence of LLM conversion or a universally safe step size.
[Primary paper](https://arxiv.org/pdf/1708.03888), v3,2017.

Yin et al. analyze appropriately chosen STEs for a two-linear-layer network,
binarized ReLU and Gaussian inputs; expected coarse-gradient alignment and
convergence are conditional. Poor STE choices can be unstable. Those conditions
do not cover this recurrent SSM/SWA/selected ternary-bank pipeline. Their result
supports explicitly testing the relationship between a surrogate direction and
actual loss. [Primary paper](https://arxiv.org/abs/1903.05662), v4,2019.

## Derivation for the original expert function

Define weight quantization Q(W)=s(W)*round_clip(W/s(W)) row-wise, with
s=max(mean(abs(W)),1e-5), and activation quantization A(x)=a(x)*q(x),
a=max(abs(x))/63. In real arithmetic, away from active scale floors and
unexceptional rounding boundaries, positive scaling preserves integer codes:

    Q(lambda*W)=lambda*Q(W), A(lambda*x)=lambda*A(x), lambda>0.

For gate G, up U and down D, the original expert is a quantized gated dReLU
map with ReLU(g)*ReLU(u), followed by a down projection. With the same input,
routing IDs and mass, independently scale matrices by lambda,mu,nu>0:

    F(lambda*G,mu*U,nu*D;x)=lambda*mu*nu*F(G,U,D;x).

Thus lambda*mu*nu=1 gives a two-dimensional family of function-equivalent
parameterizations in this ideal model. Matrix norms and ordinary coordinate
learning rates can differ despite identical expert functions. Scale floors,
F32 rounding/overflow and changed quantizer cells require actual validation;
this is not a bytewise C equivalence certificate. Sigmoid-gated donor SwiGLU
does not have this gate-scaling homogeneity, so it cannot be assumed upstream.

If a used gradient transforms inversely under a positive parameter rescaling,
g(lambda*W)=g(W)/lambda, a floor-inactive relative proposal transforms as

    -alpha*||lambda*W||*g(lambda*W)/||g(lambda*W)||
      =lambda*(-alpha*||W||*g(W)/||g(W)||).

This gives an algebraic reason to use relative group coordinates. It does not
prove inverse covariance for every surrogate, actual finite-precision path or
continuous core tensor, or actual loss decrease. Our frozen grid measures the
last question directly in original C.

## Explicit connection to expert-count scaling

If a whole bank contains n experts of similar norm and only a fixed subset
has nonzero gradient, its Frobenius norm grows as sqrt(n), while that sparse
gradient norm stays fixed. A bank-wide relative direction therefore increases
selected-expert movement by sqrt(n). Adding ten times as many inactive experts
can multiply that movement by sqrt(10), without adding active computation.
Per-expert/projection normalization keeps each existing selected group's
proposal unchanged; zero-gradient groups copy original bits. This is a
conditional algebraic property of the conversion update, not useful capacity
or actual100B training/inference evidence.

For this comparison assume the selected set, inputs and normalized selected
mass remain fixed. In real arithmetic that mass depends only on selected
logits, since the all-expert softmax denominator cancels when renormalizing.
New competitive experts can change the selected set; F32 reductions can change
rounding. Useful large-n choice/quality and structured CPU selection+mass
still need measurement. Dense gradient/master storage can grow with n even
when each group's trust-rule geometry is invariant.

## Function-neutral directions as a possible surrogate diagnostic

At one expert, tangent directions of the ideal scale family are

    t1=(G,0,-D), t2=(0,U,-D).

A true differentiable loss gradient on a function-invariant path satisfies

    <g_G,G>-<g_D,D>=0,
    <g_U,U>-<g_D,D>=0.

The identity weight STE differentiates through W where the forward uses Q(W).
These contractions need not vanish. A nonzero contraction would quantify
surrogate sensitivity to an ideal function-neutral direction; it would not
by itself show an implementation fault or failure of all surrogate directions.
Floors and F32 effects must be separated before assigning any measured defect.
Current stored contractions have NOT been computed in this note.

One proposed correction, if such bias matters to actual descent, is to remove
the neutral tangent component from a proposed displacement in relative metric
M=diag(1/||G||^2,1/||U||^2,1/||D||^2). Then

    T^T M T=[[2,1],[1,2]], T=[t1,t2],
    j=[<G,deltaG>/||G||^2-<D,deltaD>/||D||^2,
       <U,deltaU>/||U||^2-<D,deltaD>/||D||^2],
    c=(1/3)*[[2,-1],[-1,2]]*j,
    delta_bar=delta-T*c.

The joint relative-metric norm cannot increase under this orthogonal projection.
Individual group bounds can still increase; rescale the three projected blocks
together to restore their maximum relative-radius cap. Zero norms/floors need
a separately specified rule. This correction is UNIMPLEMENTED/UNTESTED and is
not part of the already frozen experiment; it is an optional diagnostic path
if actual descent later stalls or a directional prediction becomes misleading.

## Operational use

Keep first the actual changed-state GPU/backward prerequisite and finite
multi-case conversion plan. Local relative-step evidence is useful; held-out
own-history chatbot quality, same-artifact>=50, useful-n/CPU routing/DRAM and
multiple families remain missing. Scale geometry introduces no donor runtime
or new inference operator. Primary papers checked10October2026; this engine
derivation and proposed correction are separate from their tested results.
