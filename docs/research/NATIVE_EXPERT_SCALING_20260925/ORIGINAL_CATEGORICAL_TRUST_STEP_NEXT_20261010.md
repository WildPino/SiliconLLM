# Next causal transfer step: parameter geometry and actual discrete descent

10 October 2026. Algebra/plan implemented as [new frozen protocol](ORIGINAL_CATEGORICAL_TRUST_STEP_PROTOCOL_20261010.md): runner/holder AST and tiny structural preflight PASS, binding sealed c6500378. Actual updates/native/complete audit UNEXECUTED.
The [complete causal preflight](ORIGINAL_CATEGORICAL_CAUSAL_PREFLIGHT_RESULT_20261010.md)
qualifies the coherent readout/loss/backward bridge, not quality or a step size.
Do not replay its completed baseline forward/backward/native prefix.

## Separate the three mathematical problems

1. Output representation: fixed H and actual normalized f have a verified
   KL/state-gradient identity. Prior feasible72 codes show head image room;
   they do not show the original history function can produce these codes.
2. Continuous causal map: learn parameters taking token history to the useful
   normalized state. Conditioning depends on parameter units and RMS geometry.
3. Discrete execution: ternary code boundaries, activation quantization and
   top8 routing can change under weight movement. STE descent is a proposal;
   only new exported original C evaluation can establish actual descent.

For raw final state u, r=sqrt(||u||^2/D+epsilon), f=u/r,

    J_RMS=I/r - u u^T/(D*r^3).
    gradient_u L=J_RMS^T * H^T(p-q).

Tangential eigenvalues are1/r; the radial eigenvalue isepsilon/r^3. Small
radial loss sensitivity and strongly different tensor scales can make a common
coordinate step a poor choice. H's last column is zero, but RMSNorm still couples
that carrier coordinate to the other255: its raw-state derivative is generally
nonzero. Do not freeze it merely because the head column is zero.
For the fixed real readout, Hessian_f L=H^T(diag(p)-p p^T)H is PSD. This convex
state-space fact does not make the causal parameter/quantizer problem convex.

## Observed parameter conditioning, not a claim about old Adam

Numbers below are simple ratios of fully audited saved parameter/gradient
squared norms for THIS new first-balanced objective, not another model pass.

| Tensor | Parameter L2 | Gradient L2 | ||g||/||P|| |
|---|---:|---:|---:|
| layers.4.organs.out_proj | .21815608 | 5412.37213 | 24809.63244 |
| layers.2.organs.out_proj | .21652310 | 3300.13598 | 15241.49582 |
| layers.3.organs.out_proj | .21657288 | 3018.39783 | 13937.09958 |
| layers.1.organs.out_proj | .21507585 | 2497.97057 | 11614.37045 |
| layers.5.organs.o | .18837844 | 1839.64780 | 9765.70237 |

For UNCLIPPED SGD, relative tensor displacement is eta*||g||/||P||; eta5e-5
would move layer4 out_proj by124.05% of its current L2 norm. This is neither
the old clipped Adam update nor evidence that old Adam failed for this reason.
Global gradient clipping bounds raw SGD step norm but does not impose a tensor
relative-change bound on Adam. A FRESH first Adam step with no weight decay is
approximately -eta*g/(abs(g)+epsilon); its tensor L2 upper iseta*sqrt(entries).
For the262144-entry out_proj, eta5e-5 gives upper.0256, about11.73% of current
norm. This is an upper bound, not its actual displacement or a predicted loss.
Old Adam moments differ; they are not restored in the proposed experiment.

## Candidate direction and granularity

Construct a fresh direction from the already preserved gradient, then bound
each actual tensor/group displacement relative to its parameter norm (and an
explicit absolute floor for zero norm). Candidate general form:

    Delta_G=-alpha * rho_G * d_G/max(||d_G||,tiny).

Choose and freeze direction d, rho, zero-norm floors and alpha grid BEFORE
new quality observations. Report dot(g,Delta), actual F32 displacement after
rounding, per-group and whole norms; never call a negative STE dot product
proof of actual quantized loss reduction.

Bank geometry must be grouped by consulted expert/projection, not by the
whole1152-expert bank. A global bank norm can allocate excessive movement to
a sparse selected subset. Core matrices and router rows/biases have different
units; record each choice, not just one optimizer learning rate.

For a ternary row, s=max(mean(abs(w)),1e-5), q=clip(round(w/s),-1,1).
Since |Delta s|<=mean(abs(Delta w)), an unchanged ternary symbol is certified
if its distance to both +/-s/2 thresholds exceeds

    abs(Delta w_i) + .5*mean(abs(Delta w)).

This sufficient local condition can be conservative; exact F32 exporter
readback and full symbol/scale differences are still required. Quantizer
ties and floor cases need explicit treatment. Scale changes can alter outputs
even when all ternary symbols stay equal.

For fixed selected IDs, normalized routing mass equals softmax of the selected
logits: the all-expert softmax denominator cancels in renormalization. Stable
top8 needs the8th/9th logit gap to exceed twice the logit perturbation upper.
Ties/changed IDs are measured, not assigned a smooth derivative. These facts
connect eventual large-n structured CPU routing to function preservation;
they do not establish scalable selection or useful expert capacity.

## What the next experiment must decide

Reuse audited source/model/packed baseline, fixed H/norm, compiled FIT targets
and saved90 gradient tensors. First implement a stored-only group geometry
and fresh update/export tool; seal exact code, alpha grid, receipts and caps.
Then evaluate genuinely changed candidates on the SAME FIT history in actual
C, using the already retained full-V baseline. Full packed/quantization/routes
audits continue even a loss increase. No baseline forward/backward replay.

If actual C improves and numeric/packing gates pass, the result only admits
preparation of a finite multi-case causal campaign with explicit first-response
weight and DEV/generation checks. If it fails, attribute continuous step size,
changed ternary/activation cells or routing before increasing dose. A single
case gain cannot admit useful chatbot conversion or generalization.
No new optimizer/native/source/T4 call has been made for this plan. Set an
explicit budget/caps/stops in the NEW protocol before launching; preflight's
caps do not automatically authorize an unspecified longer campaign.
