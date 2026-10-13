# Per-token exact remainder importance: qualified coverage diagnostic

13 October2026. **COMPLETE/FULL AUDIT PASS/DIAGNOSTIC ONLY.** Goal ACTIVE/INCOMPLETE.
No candidate admitted:selection evaluates ALL original source neuron functions.

## Question and reproducible scope

Does fixed per-cell selection lose useful conditional information? Reuse
all48 actual own histories/sites0,12,23/existing64 FIT centers/all2048 output
coordinates. At every token select source neurons by combined exact remainder
squared times beta-scaled down-column squared norm. Stable source-index ties.
This is a specified per-token energy heuristic with oracle access,not an
optimal subset or a lower bound on what sparse experts can express.

[Protocol](FFN_REMAINDER_ORACLE_PROTOCOL_20261013.md),sourcecb5f4fe/
bindingdc78161;[producer](ffn_remainder_oracle_result_20261013.json).
All budgets32/128/512/2048 plus full4608 control retained. No learned routing,
new centers,source model/GPU/RESERVED or optimizer. Source smooth proxy has
canonical BF coefficients/effective F32 alpha,beta,actual captured BF output
as reference. No output projection or teacher-state injection.

## All local error curves

Case-weighted relative FFN output errors against actual source BF. These
are local operator metrics,not whole-chatbot quality or deployed CPU speed.

| Site | Neurons | FIT mean | FIT worst | DEV mean | DEV worst | Fixed-cell DEV mean |
|---|---:|---:|---:|---:|---:|---:|
|0|32|0.196752|0.289325|0.198582|0.291909|0.575733|
|0|128|0.127341|0.186029|0.128853|0.186470|0.518641|
|0|512|0.061027|0.075778|0.062173|0.076069|0.400908|
|0|2048|0.012175|0.013354|0.012354|0.013821|0.243356|
|0|4608|0.004141|0.004290|0.004123|0.004256|0.004123|
|12|32|0.532752|0.567012|0.602978|0.690721|0.883059|
|12|128|0.372823|0.395031|0.425486|0.487288|0.820818|
|12|512|0.196643|0.213966|0.225681|0.256812|0.688920|
|12|2048|0.042296|0.048495|0.048711|0.054676|0.404537|
|12|4608|0.004672|0.004773|0.004669|0.004754|0.004669|
|23|32|0.066187|0.124634|0.072805|0.126221|0.113219|
|23|128|0.040353|0.075540|0.044127|0.080406|0.101091|
|23|512|0.017931|0.030730|0.019147|0.030867|0.077901|
|23|2048|0.004360|0.005593|0.004589|0.006089|0.036098|
|23|4608|0.003113|0.003207|0.003107|0.003210|0.003107|

At0,512 neurons reduce DEV mean.400908=>.062173 (6.45x smaller error).
At12,2048 reduce.404537=>.048711 (8.30x). At23,32 reduce.113219=>.072805
(1.56x). The corresponding FIT means/worst are.061027/.075778,
.042296/.048495 and.066187/.124634. Counts512/2048/32 are the smallest
TESTED budgets satisfying the existing mean<=.10/worst<=.20 on FIT oracle
curves;these budgets are chosen from FIT,not DEV. Their DEV curves and domain
means also meet those local thresholds. No cheap candidate admission follows.
At12,the512 oracle still has DEV mean.225681:substantial improvement over
fixed.688920,but not low enough under the same thresholds. A different
joint subset could behave differently;this heuristic is not optimal.

The observed improvement establishes that fixed-cell aggregation costs
coverage for this rule on these data. It does not establish that a cheap
predictor can obtain the same IDs or that higher n preserves chatbot quality.
Keeping active width fixed while increasing meaningful conditional choices
now has a source-derived target,not merely duplicated synthetic weights.

## Complete audit and the preserved runtime failure

Original audit expanded per-token selected weights to(N,r,2048). Its nominal
selected-weight tensor payload over the full assay is3,014,385,991,680B,
before copying/index/cache effects. This is algorithmic working traffic,
not PMU-measured DRAM or batch1 inference traffic. It completed240 site0
checks at1319.953s,then terminated1 by the1800s time guard at1800.141s/
OS2,985,328,640B. Source and [terminal](ffn_remainder_oracle_adjudication_20261013.terminal.json)
are preserved;no partial result admitted and producer was not restarted.

[Layout variant](FFN_REMAINDER_ORACLE_AUDIT_LAYOUT_20261013.md),source6a15e65/
closer4af9a80/preparationd93a999/binding7de0585,uses256-column projection
blocks and avoids the expanded tensor. Every original case/ID/mass/metric/
source/hash/numeric gate remains. [Audit V2](ffn_remainder_oracle_adjudication_v2_20261013.json)
PASS720 predictions/all144 controls/all15 curve and domain summaries.
Literal selected uint16 IDs exact;independent input-delta chain rule,
feature-minus-affine remainder and differently associated block projection
verify all saved output arrays. Shared source reader/BF decoder/literal ID
algorithm/NumPy BLAS primitive declared;no independent CUDA arithmetic claim.

Max F32 prediction relative difference2.663e-8;independent remainder feature
relative difference3.985e-16;full-neuron identity6.682e-16. Maximum absolute
Frobenius score difference1.116e-9 satisfies the frozen relative score
criterion (1e-10*literal score norm+1e-24);it is not a1e-9 relative score
error. Null-score tolerance was predeclared from the prior rank diagnostic.
Independent case/selection/domain metadata match within1e-9;no quality gate
changed. The faster audit is not a faster inference candidate.

## Cost and closed custody

Producer470.750s/OS1,242,435,584B;V2 auditor509.563s/OS1,549,312,000B.
Main preparation18.688s/OS43,425,792B;V2 preparation.032s/OS25,903,104B.
[Closure](ffn_remainder_oracle_session_closure_v2_20261013.json) all four
successful command receipts/PID-create pairs absent/original failed audit
absent/all source,input,output and protected hashes PASS. Complete720-file
E family3,143,325,180B<3GiB;selected IDs and full predictions retained.
Closure itself exit0/21.484s/OS30,806,016B. Original engine.c and foreign
tracked hashes preserved. No source GPU/T4/delegation or live owned run.
These are experiment costs,not the deployed active coefficient or DRAM cost.

## Decision and exact next action

Implement/freeze [source-linear sketch router](FFN_SOURCE_SKETCH_ROUTER_NEXT_20261013.md)
with ranks64/128/256/512 and fixed active counts512/2048/32 as determined
from FIT above. Fit input basis on the UNCENTERED second moment of FIT
input deltas;save the mean and centered-covariance identity,eigen/basis/
source projection witnesses. Approximate all source up/gate fields in the
small sketch only to choose IDs;evaluate true selected source functions and
retain the full-output affine operator. No full original U/G forward may be
used in the proposed deployed ID path. Select sketch rank on FIT,then check
that same choice on DEV;no promotion through DEV or RESERVED tuning.

This is a new routing approximation,not a retry of rejected fixed-output
banks or the token-only shared GG rank family. The middle-layer active count
is large,so physical cost must be screened before building all24/large banks.
A real compiled operator must include actual CPU IDs AND masses/precision/
loaded bytes/DRAM and leave room for shared core,contextual generators,
attention and head within20ms. Final same-artifact own-model quality/useful50,
useful RAM-driven n,multiple donor families/scales and conversion costs remain
open. No general10B/100B transfer or goal completion is claimed.
