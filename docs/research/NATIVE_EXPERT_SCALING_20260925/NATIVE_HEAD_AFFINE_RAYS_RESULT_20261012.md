# Fixed native head: decision rays available, probabilities still fail

12 October2026. COMPLETE/full independent audit PASS. Goal ACTIVE/INCOMPLETE;
no owned run live. Actual native oracle codes express many observed donor winners,
but preserve neither their probabilities nor causal code generation.

## Question and exact algebra

[Producer](../../../benchmarks/native_expert_scaling/native_head_affine_rays.py),
[frozen protocol](NATIVE_HEAD_AFFINE_RAYS_PROTOCOL_20261012.md),
[result](native_head_affine_rays_result_20261012.json).
Actual source-derived exported F32 C/beta/epsilon,promoted F64 for geometry.
All8808 existing source label states yield2361 distinct audited teacher winners.
Fixed d_j=(C_j-vocabulary_mean(C))/norm;no state-dependent direction fitting.
Positive ray t=s d_j. Positive scalar RMS cancels from argmax,so winner j requires
`(C_j-C_k)^T t >= beta_k-beta_j` for all k!=j. Full65537 constraints become scalar
lower/upper intervals in s>=0. Evaluate margins0 AND1 separately;zero coefficient
and active lower/upper witnesses retained. Empty intervals only reject this ray
at that margin;no general256-dimensional head infeasibility proof.

Margin1 intervals give finite candidate scalars with frozen1.01growth/.01add,
finite-upper midpoint fallback/cap1e6. Cast actual tF32 and execute original engine
head/scalar-norm extension;store every F32 score/norm/inverse word. Teacher winner
is an oracle input. Looking up these codes is not cheap autonomous routing or a
causal encoder that discovers the winner from the token history.

## Complete observed outcome

| Distinct teacher winners | Count |
|---|---:|
| Total | 2361 |
| Nonempty zero-margin ray interval | 2350 |
| Nonempty margin1 ray interval | 2342 |
| Finite candidate code within cap | 2342 |
| Actual native code chooses required target | 2342 |
| Native-versus-real argmax changes | 0 |

Eleven winners fail this ray even at zero margin;another eight fail margin1.
None discarded by scalar cap. Their58 label associations are unavailable in this
probe,not evidence that no other direction/code can express those winners.
Min actual raw margin1.00830999,min native normalized margin.19953156.
All frozen norm/inverse2e-5 and absolute score1e-4 precision flags PASS;
max score difference6.91019e-5. No kernel or code-cast failure found.

| Existing label coverage and probability loss | FIT | DEV |
|---|---:|---:|
| Native target labels | 4399/4422 | 4351/4386 |
| Mean per-case native target fraction | 98.7119% | 98.6616% |
| Available weighted real-head KL contribution | 7.82277588 | 9.60070268 |
| Available weighted native KL lower contribution | 7.82275137 | 9.60067710 |
| Available weighted native KL upper contribution | 7.82280040 | 9.60072826 |

Mean per-case coverage uses original equal-case weights,not global label fractions.
No missing-label loss is replaced by zero or renormalized away:table losses are
the available labels' contributions under original weights. Assigning any codes
to missing labels cannot remove the large observed contribution from available
labels. Both distributions and per-case decision coverage fail original budgets.
For comparison,existing native factor predictor DEVcaseKL2.93425 and41.0883%
disagreement;these rays improve oracle winners but worsen categorical loss.
Correct top1 alone loses distributional information and does not preserve a chatbot.

## Probabilities, sufficient moments and native error control

Use independently qualified full-vocabulary teacher moments m=A_aug^T q,H(q)
from [preflight](NATIVE_HEAD_DOMAIN_PREFLIGHT_RESULT_20261012.md). Current ray
code tF32 promoted F64 defines positive-sphere v and real-head probabilities.
Exact-real full KL is logsumexp(A_aug v)-m^T v-H(q);no truncated vocabulary.
This permits evaluating every label association without reloading the donor.

Native F32 score s_native differs from s_real by delta. Exact-real
`KL(q||p_native)-KL(q||p_real)=log E_p_real exp(delta)-E_q delta`,hence absolute
change<=max(delta)-min(delta). Store per-code span and every associated real loss/
numeric native-KL lower/upper. These are numerical evaluations of a real-arithmetic
bound,not interval certificates or direct exact native teacher KL. Previous
preflight false1e-5 precision flags remain unchanged and are not reused as PASS.
Actual native winner and full native bits are independently verified here.

## Full independent qualification

[Auditor](../../../benchmarks/native_expert_scaling/native_head_affine_rays_audit.py),
[frozen protocol](NATIVE_HEAD_AFFINE_RAYS_AUDIT_PROTOCOL_20261012.md),
[adjudication](native_head_affine_rays_adjudication_20261012.json).
All2361*65537 zero/one-margin constraints and endpoint witnesses recomputed with
coefficient differences before independent einsum summation;separate block center
and norms. All scalar rules/caps/directions/F64 codes/exact F32 casts checked.
Real head sums eight32-feature blocks,full logaddexp;all margins/IDs/radii/logZ/
score-error spans verified. Every8808 association/moment loss/native-KL interval/
unavailability/weight/summary verified using different moment-coordinate sums.

Separately compiled scalar explicit-FMA reference independently computes all
**153,487,654 native F32 scores**,**599,552 normalized words**,**2342 inverses**:
ALL bit exact. No producer DLL replay. Max geometric/moment relative2.1445e-13,
scalar1.31195e-10;full relevant input/output before/after hashes PASS.
Decision INDEPENDENT_STORED_PASS;scientific distribution/causal quality not admitted.

## Cost, frozen chain and closure

CPU6threads/affinity0..5,batch64;no source model/GPU/T4/RESERVED/longalign.
37 actual native calls;4 output files634,239,846B. Timing admission=false,
known external publishers retained;no accepted tokens/s inference.
Preparation.391s closed0;source/protocol a52e69e,
[binding](native_head_affine_rays_binding_20261012.json)
SHAf1966c68301769d94b3ebed80d61753700deeb0d1e90aa713289dc75dc862f55;
[launch](native_head_affine_rays_launch_20261012.json)a6acbdb.
Producer94685 CLOSED0,PID28272/create1791820396.6109145,result31.016s/
terminal31.031s,OS1,246,187,520B. ResultSHA
645bcee5f7cee2e29a42490e7d10bad9470e77a60f592e772cc4480ec007acd3;
terminal307f762c87a7a82821bd279056dca4089ab49ff4b5e866b92c55f2507bbea493.

Audit source/protocol da91f5c,[launch](native_head_affine_rays_audit_launch_20261012.json)
98f7686. Session75477 CLOSED0,PID18064/create1791820655.677856,
153.938s/OS1,143,922,688B. AdjudicationSHA
aeece96d073f2cbbd46671d8e196c55aeb6b17cef32801fe1cc8dd88fffa94de.
Producer900s/4GiB/1GiB and audit900s/4GiB caps PASS. Ordered D namespace;
original engine/foreign changes/publishers preserved.
[Session closure](native_head_affine_rays_session_closure_20261012.json).

## Decision toward the pipeline

Most observed winner decisions already have actual native codes,so current
41% case disagreement cannot be explained solely by unavailable winner regions
for those decisions. This is an oracle availability statement,not information
capacity of the head or learnability of a causal code. Same-winner distribution
loss still matters. Stop optimizing winner-only rays;no repeated ray variants.

Next [bounded full-probability image gap study](NATIVE_HEAD_PROBABILITY_GAP_NEXT_20261012.md)
PROPOSED/UNIMPLEMENTED. Reuse qualified sufficient moments to distinguish fixed
head/native-domain loss from current encoder loss. Price/qualify local GPUF64
against all existing CPU loss/gradients before freezing cohort/iterations/budget.
Head/domain/source reference differ from historical paired255 oracle experiment.
After bounded image diagnosis,return to causal SSM/SWA/ternary code generation,
not indefinite output-oracle tuning. Useful experts/CPU IDs+mass/physical DRAM/
multiple families/same useful artifact>=50 and preserved conversation still open.
