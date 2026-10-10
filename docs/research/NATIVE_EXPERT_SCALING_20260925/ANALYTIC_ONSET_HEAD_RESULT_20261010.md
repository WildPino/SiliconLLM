# Analytic onset correction: fully audited proxy quality failure

10 October 2026. **ANALYTIC_FIRST_CORRECTION_PROXY_QUALITY_FAIL**.
One new exported original packed head; full independent shifted audit PASS.
[Protocol](ANALYTIC_ONSET_HEAD_PROTOCOL_20261010.md),
[result](analytic_onset_head_result_20261010.json),
[complete adjudication](analytic_onset_head_shifted_adjudication_20261010.json).
Full goal incomplete. No new native execution or chatbot admission.

## Changed variable and scientific result

The [24-first-state control](CATEGORICAL_ONSET_CONTROL_RESULT_20261010.md)
found a preserved map of norm 31.330779696, above the earlier
introduced fit radius 16. Remove that radius for this ONE already computed
closed-form point. Same decoder, whitening, original core, routes and final norm.
No scale selection, code search, SVD, optimizer continuation or DEV fitting.
Fuse H = A Theta W offline and cast to the original ordinary F32 head.

| Stored forced-label proxy | Previous shared head | New analytic head |
|---|---:|---:|
| FIT case KL | 2.9208417425 | 49.3213438162 |
| DEV case KL | 6.2589668121 | 63.374191471 |
| FIT case disagreement | 56.8142% | 97.35225032% |
| DEV case disagreement | 87.0686% | 99.05862002% |
| FIT first-token KL | 4.5782547248 | 0.00412700483273 |
| DEV first-token KL | 4.6573244015 | 4.51188884083 |
| FIT continuation case KL | 2.8972929505 | 49.7588963387 |
| DEV continuation case KL | 6.2692290391 | 63.9644984241 |

24 FIT first codes: zero argmax disagreements, maximum reference KL delta
1.55475682089e-15; the full code interpolation identity
carries through the actual fusion. DEV first tokens: 19/24
argmax errors. All strict and original case/domain whole-proxy gates FAIL.
Per-case/per-label losses, IDs, entropy, uncertainty and domain summaries remain
in the result. DEV is reused development evidence, not untouched evaluation.

The selected first-code repair is feasible, but its global linear action badly
alters continuation predictions and does not generalize to DEV first states.
This rejects THIS global interpolation candidate. It does not prove an
irreducible KL floor, a D256 ceiling, or impossibility of a different nonlinear
causal/core/conditional-function construction. Actual exported C history and
own-history quality of this head were not run; proxy failure is not a claim
of newly measured task accuracy or speed.

## Packed bytes and numerical checks

One packed artifact 520029440 B,
SHA256 `5870828c933c2eb03e418949308783152dc61702ecf7d4ee5aad6f112ca01293`. Full parsed 110-field ABI,
head bytes exactly equal stored F32 cast; all 109 other fields/header/table and
full prefix/suffix extents identical to the previous packed artifact.
Head maximum row norm F64 13.2102213494,
F32 13.2102213257, cast row error upper
3.47945977957e-07. All 8808 old feature uncertainties propagated
through this larger head, including the exact original 39-rounding dot path.
Bounds remain conditional evaluated F64 arithmetic, not interval proofs or
own-history guarantees. Full fusion audit maximum adapter/head deltas zero.

The complete shifted audit checks all input/output hashes, all 8808 independent
full-V probabilities/KL/argmax/entropy/bounds, all 48 case/domain flags and all
24 reference first labels. Maximum per-label KL discrepancy
4.04725142289e-11 <= unchanged 1e-9;
maximum bound discrepancy 0.0;
first reference discrepancy 1.51558453287e-13.
All counters/resources pass. The new binding includes actual junction-resolved
NumPy/psutil modules and native runtime files, not merely their logical SITE path.

## First audit fault retained

Original audit completed all 8808 unshifted sequential-logaddexp rows, all48
case/ID/entropy comparisons and prior full hash/fusion/packed checks, then
terminated1 at its final max-label-KL/max-bound/aggregate conjunction (line114).
Stored aggregate equality independently reconstructed PASS. The program did
not persist either maximum, so the exact failed maximum and first-fault cause
are unresolved; no completed-prefix check is relabeled a full successful audit.
Original frozen code, [log](analytic_onset_head_stored_adjudication_20261010.worker.log)
and [terminal](analytic_onset_head_stored_adjudication_20261010.terminal.json) retained.

[New shifted audit protocol](ANALYTIC_ONSET_HEAD_SHIFTED_AUDIT_PROTOCOL_20261010.md)
subtracts row maxima before its independent sequential logaddexp reduction,
mathematically softmax-invariant. Producer uses exp/sum normalization. Same
1e-9 KL and 1e-8 bound tolerances and all quality gates; no threshold relaxation.
New normalizer, script, binding, freeze and namespace. It persists measured
maxima before the final assertion and fully passes. That supports the new
adjudication, not a retroactive success or measured cause for the old algorithm.
The complete extra 8808-label arithmetic pass is charged below. No producer,
head export, optimizer, model, SVD or native history was replayed.

## Reproducibility and measured cost

Producer freeze `b6e1aa8178f809e7eaee2d83e0a077eddc50ae37`,
binding `43732ba8fc07994b8fe0832267fef72cfa63d5a824b75aa3262e2b04c9ddcfaa`,
300 inputs/1870470874 B.
Separate reused holder binding `fd2348d4`, frozen with producer.
New shifted audit freeze `4af01befc778711655dfb9746bcf9a4d73a065a0`,
binding `4c1dd969c800e8fb9cded796de3fc0c13744c111eb57426c97c62dcb19a5af77`,
307 inputs/1872109934 B,
separate holder binding `effebc31`.
Exact held worker commands/PIDs/creation/limits/inputs are in terminal receipts;
worker1/holder11/one BLAS thread/CUDAoff/no children, CPU file arithmetic only.
No foreign overlap observed by holders, background publisher daemon preserved.
No isolated throughput admission is made by these arithmetic holders.

| Stage | Held seconds | Worker seconds | OS worker + holder peak snapshots |
|---|---:|---:|---:|
| Producer, exit0 | 88.641 | 81.547 | 673013760 B |
| First audit, exit1 | 86.218 | no final worker result | 606646272 B |
| Shifted full audit, exit0 | 90.203 | 84.734 | 606851072 B |

Combined held 265.062s.
Producer 5 output files/722403584 B,
plus result 1059889 B; new audit checkpoint241 B.
26424 stored full-V label probability products across producer/two audits;
one new head export, three fusion calculations including audits. No source,
native, history, optimizer, GPU, SVD, RESERVED or T4 call.
All owned sessions81330/95580/92911 terminal; final workers absent on process
inspection. Producer SHA `c592ba0b026056f074caf7ee68d501460ba3518d9efd21961665f7df91d3c7bf`;
complete shifted audit SHA `1bbbc231c7cda1a2fd0fb603928d2a14133aaa9cba157ce97a00c1ed3d21a08d`.

## Decision and next converter work

Do not send this failed global head through another native chatbot campaign
merely to reproduce its forced-label proxy collapse. Existing original-native
arithmetic authority remains with the previous separately assessed artifact;
this candidate has a real byte-qualified export, not new C quality authority.

[Next fixed-head categorical causal transfer](ORIGINAL_CATEGORICAL_CAUSAL_TRANSFER_NEXT_20261010.md)
is PROPOSED/UNIMPLEMENTED: compile source supervision to exact fixed-head
moments/entropy, qualify dense-versus-compiled loss/gradient, then a NEW original
core/bank campaign with coherent canonical head/norm and explicit onset weight.
No long training or T4 allocation yet. Useful same-artifact50, own-history chat,
RAM-driven useful experts, structured CPU IDs/mass, physical DRAM and another
family/scale remain required/open.
