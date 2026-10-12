# Source function coordinates: frozen protocol

12 October 2026. New variable: raw residual read/write coordinates selected by
source reading functions, rather than angular variance alone. Not a new donor
runtime, recurrent channel pruning, or another paired final-feature codec fit.
Previous angular covariance/audits and exact normalized projection algebra reused.

## Construction, derivation and scope

Seven boundaries 0/4/8/12/16/20/24; source dimension 2048, active coordinates 256.
Reuse independently verified C=mean_CASE mean_TOKEN u u^T on the 24 FIT cases,
u=h/sqrt(mean(h^2)+1e-5). No DEV fit, no centering or fitted gains.

At b<24 actual trained first input projections of block b read the input RMSNorm:
five SSM sections gate3072/x3072/B256/C256/dt48, Q1024/K256/V256. Fold source gamma,
SSM input .625 and five section MuP constants, attention input1 and key multiplier
.06905339660024876. Source has no projection bias. Define G_g=A_g^T A_g and
M=(1/8) sum_g G_g/tr(G_g C). Equal group FIT energy is a DECLARED surrogate choice,
not downstream nonlinear SSM/attention sensitivity. Scalar multipliers cancel
in balanced M but remain in the actual observable comparisons. FFN inputs occur
after the core residual and are not the saved state at these boundaries.

At b24 A=.01953125 W_head diag(final_gamma); M proportional to
A^T (diag(qbar)-qbar qbar^T) A, normalized by tr(M C).
Reuse the stored independently audited equal-case FIT qbar from the previous
paired codec. Its cached-generation logits and current full-prefill residuals
come from distinct source paths. It defines a chosen metric only; current actual
F64 source readout is the probability reference. Not sample-specific Fisher/KL.

Restrict C to eigenvalues>1e-12 lambda_max, discarded trace<=1e-8. No ridge.
Let S=C^(1/2), J=C^(+1/2), K=S M S. Leading256 eigenvectors U of K give
V=S U, W=J U. For column h: z=W^T h, hhat=V z; W^T V=I and L=V W^T is idempotent.
This is the rank256 truncated SVD optimum for the declared FIT quadratic
reading error ON THE RETAINED C SUPPORT. Derivation: whiten u=S y; the measured
matrix A S has right Gram K, and A S U U^T is its rank256 SVD truncation.
Spectral tail equals the empirical equal-case FIT oracle-norm error, within
1e-8 relative to total metric energy, including the negligible discarded support.
No claim that this is optimal for nonlinear functions, probabilities or tasks.

Deployment algebra requires norm metric G=V^T V. With rho=2048/256=8,
source normalization of reconstructed h is equivalently sqrt(8) V z divided by
sqrt(z^T G z/256+8e-5). Fold gamma into source readers. Six residual transports
are W_next^T V_previous; they are not identities. These coarse four-block maps
do not specify bases at every actual sublayer. Applying W to a full source h
is an injected encoding assay, not an affordable causal state acquisition.

## Evaluation fixed before observation

All 48 cases/all histories at all seven boundaries: relative RMS reading error
under M for (a) oracle true-source norm, (b) reconstructed norm, and (c) the
previous angular256 projection using oracle norm. Report numerator/denominator
per token, case metrics, and true/reconstructed radius. All 336 cases are used.
Proxy per boundary own norm: DEV mean<=.10 AND worst<=.20.

At b<24 first/middle/last HISTORY positions for each of 48 cases: 144 inputs;
save all eight actual source projection arrays and their two reconstructions,
before convolution, nonlinearities, RoPE, attention or recurrence. Report
FIT and DEV pooled relative RMS per group. No additional gates/optimization.

At b24 compare the SAME first/middle/last LABEL states of DEV24 (72 total) as
the preceding readout probe. Actual source full-V F64 outputs previously audited
are reference. Recompute V-projected actual head and full-V probabilities for
oracle and reconstructed norm. Proxy each mode meanKL<=.01, worstKL<=.1,
argmax disagreement<=.01. Positive normalization cannot change bias-free argmax.
No generative chatbot, own-history, C runtime or useful-speed admission.

Numeric gates: complete K eigen residual<=1e-10; W^T V defect<=1e-8;
FIT spectral/empirical error relative<=1e-8. Report F32-rounded dual defect,
but no F32 inference claim or implicit tolerance-driven refit.
All input hashes before AND after. Store complete K eigensystems, M/root/inverse,
V/W/F32 versions/G, observable packets, final projected head/scores/raw selected
states, and all six skips. Independently audit source coefficient maps,
full histories, observables, head/probability arrays and transports even failure.

## Budget and decisions

CPU six threads/affinity0..5; no children; 1200s, OS peak6GiB, output3GiB.
External publisher processes remain untouched; no clean speed timing admitted.
No source/shadow/model loader, optimizer, backward, GPU, native or T4 call.
No RESERVED or longalign query. Record PID/creation, exact argv, exit/terminal,
source/protocol freeze and binding. First mechanical fault freezes that namespace;
any repair needs separate source/namespace without changing scientific gates.

If reading/probability proxies pass, consider composition with genuinely compact
trained generators and explicit per-sublayer transports. If they fail, quantify
which groups and probability directions survive, then choose conditional charts,
another reading objective or width according to actual findings. A failure does
not prove a universal256-dimensional capacity bound. Do not launch long training
on a locally failed representation without a new correction. No useful engine
conversion, scalable useful experts or same-artifact>=50 conclusion from this run.
