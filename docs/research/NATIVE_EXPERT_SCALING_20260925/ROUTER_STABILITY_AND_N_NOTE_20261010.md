# Routing stability, normalized mass and expert-count scaling

10 October2026. Focused primary-paper follow-up and our algebra during the
held24-case campaign. No concurrent numerical job or consumed-rule change.
Useful chatbot/same-artifact50/large-n/DRAM/family evidence remains incomplete.

## Published stability mechanisms and their actual scope

Switch uses local float32 routing while retaining lower precision elsewhere;
its precision comparison shows stabilization relative to its bfloat16 run.
This motivates protecting routing arithmetic, but does not promise identical
CPU/GPU decisions. Our current router/master path is already F32 and TF32 is
disabled, so “use F32” alone does not identify a correction for the observed
cross-runtime discrepancy. [Fedus, Zoph, Shazeer, Switch Transformers,
section2.4/Table2](https://jmlr.org/papers/volume23/21-0998/21-0998.pdf).

ST-MoE finds selective precision insufficient at its largest scales and adds
a router penalty averaging squared logsumexp of all expert logits. In its
three-run comparison this improves stability without the quality damage of
aggressive update clipping. Sections3.3–3.4 discuss rounding and exponentials.
Its encoder-decoder, distributed training results do not establish numerical
parity or useful capacity in our ternary SSM/SWA CPU engine.
[Zoph et al., ST-MoE, equation5/Table4](https://arxiv.org/pdf/2202.08906).

## A z-loss is not a certificate of stable selection

Our real-arithmetic observation: for router logits s, set

    s'=s-logsumexp(s)*1.

Then logsumexp(s')=0, so that penalty is zero, while every pairwise difference,
ranking and normalized selected mass is unchanged. For example s=[-A,0]
can be shifted this way for arbitrarily large A. Zero penalty therefore does
not bound all individual logit magnitudes or guarantee a separation margin.
Training with the penalty can still help empirically; its gradient is not
restricted to that common-shift direction. This argument only rules out the
claimed implication, not the reported training benefit.

Applying a shift after scores have already been rounded cannot recover missing
score bits. We need the first causal-input/operator divergence and actual score
margins before choosing a penalty or precision repair.

## Fixed selected experts: mass sensitivity has an n-independent bound

Let total experts be n and selected IDs S have size k. In real arithmetic,
normalized selected weights satisfy

    m_i=softmax_n(s)_i / sum_(j in S) softmax_n(s)_j
       =exp(s_i)/sum_(j in S) exp(s_j),  i in S.

For FIXED S the Jacobian is J=diag(m)-m*m^T. Each absolute row sum is
2*m_i*(1-m_i)<=1/2. Integrating along a perturbation delta gives

    ||m(s+delta)-m(s)||_infinity <= ||delta||_infinity/2.

Subtract any common shift from delta first, since J*1=0. The optimal infinity
norm is half its oscillation: (max(delta)-min(delta))/2. Also
sum_i m_i*|delta_i-sum_j m_j*delta_j|<=||delta||_infinity, using the maximum
mean absolute deviation of a variable in that bounded interval. Thus

    ||m(s+delta)-m(s)||_1 <= ||delta||_infinity.

These real-arithmetic bounds depend on score perturbation, not n. They do NOT
cover a changed selected set, differing expert inputs, exponent/reduction
rounding, underflow, or another activation decision. The observed case3 mass
discrepancy alone cannot establish which of those conditions is responsible.

With identical expert values e_i at the same input and sum(delta_m)=0,

    ||sum_i delta_m_i*e_i||
      <= max_i ||e_i-c|| * ||delta_m||_1

for any output center c. Hence functional sensitivity also depends on expert
output spread. Small mass error is not automatically small final-logit error.
F32 normalization defects and changing activation decisions require additional
terms, not this fixed-value bound alone.

## Consequence for the conversion and CPU LUT goal

Three questions must be measured separately: selected-set/rank margins,
normalized selected mass, and the resulting function/history. More stored
experts can change competitors and margins; no distribution or margin scaling
law has been demonstrated for our learned bank. Constant k does not establish
constant retrieval cost or useful retained donor capacity.

One future addressing design can retrieve selected RAW scores and normalize
only those k scores, avoiding an all-n normalization work pass. Structured
retrieval still needs its own score-family/selection proof and real CPU cost.
The cancellation above is a real-arithmetic identity; changing the original
C probability-based sorting/rounding path is not automatically bit-equivalent.
This remains a proposed runtime/training variant, not the consumed campaign.

Current observations vary across histories: case2 changes4 ranked slots and
mass/selected-label scores; case3 preserves all IDs and supervised score/KL
limits but fails full-history mass; case4 full363 IDs reports ALL bridge gates
PASS and accepts alpha.0001, weighted83.890263->75.042610 with200/200 donor
argmax disagreements. These are worker observations awaiting full audit.
[Execution/custody](ORIGINAL_CATEGORICAL_CAMPAIGN_EXECUTION_20261010.md),
[attribution plan](ORIGINAL_CAUSAL_NUMERIC_ATTRIBUTION_NEXT_20261010.md).

We therefore do not select a z-loss, change precision or relax gates from these
papers alone. Complete the trajectory and audit, locate the first measured
defect, then choose a bounded correction that preserves the original engine
advantage and is evaluated on native held-out/own-history behavior.
