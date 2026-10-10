# Shared categorical original C head: completed FIT fit and full audit

10 October2026. **SHARED_HEAD_FIT_QUALITY_FAIL**; full independent audit PASS.
Original goal incomplete. [Frozen protocol](CAUSAL_CATEGORICAL_READOUT_PROTOCOL_20261010.md),
[result](causal_categorical_readout_result_20261010.json),
[audit](causal_categorical_readout_stored_adjudication_20261010.json).

## Measured result and exact scope

| FIT24/4422 | Least-squares warm map | Best shared categorical map |
|---|---:|---:|
| Equal-case KL | 5.26002270249 | 2.9208417425 |
| Equal-case disagreement | 77.164962% | 56.814227% |
| Label-weighted KL | 5.77044572507 | 3.3379968851 |
| Wrong argmax labels | 3661 | 2857 |

KL reduction from the SAME warm-map initialization: 44.470929%. All five strict
quality flags FAIL; none of12 domains meets both domain gates. The old actual27
head has FIT caseKL7.83018812394; this is matched
cohort context, not a native comparison for the newly exported head yet. Earlier
source-feature paired codec .804121 and72-label oracle .0065297 concern different
features/datasets; no ratio across those experiments is claimed.

One weighted-feature thin SVD,rank256,retained covariance condition
676.750426977,discarded trace0,whitening relative
4.20171703254e-15. Theta0 norm1.50572158447,
best norm2.03547912593,Frobenius radius16. Shared65537-class probabilities
come from fixed255-column A and one255x256 adapter (65280 coefficients). No core,
expert,route,source/history/DEV/T4 update or query. Teacher moments on all4422.
32 accepted updates,75 full FG calls including rejected
backtracks/checkpoint Y/final certificate calls,331650
actual label FG evaluations. L rose to256 and then stayed fixed. Five checkpoints.

## Algebraic interpretation and limits

The map J is shared across causal native features. Offline H_new=A J gives an
ordinary65537x256 F32 original head; no additional runtime matrix or kernel.
The actual520029440-byte packed artifact is produced, SHA
`5e8160fb3b33bc2ff0276d32ab0bbe4a44f28ec1e30ac81b2d95a7bfd0f80fb6`. Complete prefix/suffix hashes and parsed
110-field ABI prove every byte outside the head unchanged (109 other fields).

Best feasible upper=2.9208417425, certified tangent lower=0,
gap=2.9208417425. This is a valid poor feasible point and an unfinished
optimization certificate. It **does not prove** a positive irreducible shared-map
floor, a D256 ceiling, information loss in all possible core representations,
or necessity of a nonlinear encoder. No unchanged extension of this32-update
dose is launched. Native/held-out/free-running behavior of this actual head is
still required before choosing a new core/history correction.

Per-label forced-native raw-score uncertainty under the adopted F32/FMA model
ranges0.0418007233364..
0.0420647276974; KL deviation at most
2epsilon. Largest real head row norm1.10512742209. Bounds are
conservative F64 evaluations, not interval certificates or own-history guarantees.
No new native/readout/head instance was evaluated during this fit.

## Complete independent audit

All141 inputs (5049340883 bytes),all27 outputs (789665088 bytes),
result/terminal/provenance sealed before/after. Full weighted SVD identity and
orthogonality, whitening/initializer, all4422 teacher probabilities/moments/
entropies/argmax IDs, all initial/best/lower objectives and gradients,4 math.fsum
gradient witnesses, all5 checkpoint Y gradients/objectives, feasible upper/
tangent lower/envelope, every case/domain/gate, full head fusion/cast/export and
all propagated uncertainty bounds PASS. No SVD/descent/core/source/native replay.
CPU stored audit reconstructed35376 linear
head rows in8 full stored FG checks.
Max gradient delta4.74620343027e-15,metric delta
3.90798504668e-14,full fusion delta1.38777878078e-16.

## Revisions, terminal cost and next decision

Freeze03977fb967d6af0561d2edf609ccaecbb8b9a169;
binding88c095e6980df2bbf89c20e4acfa8bfc15a50a867e39b832d69cb9ea4333665f.
Capture result5e35ade79058d74c4af7ddeeed2b54c0e7b7067067c4158ff71782741f89af13;
auditd05452168cd61940270c27e2fa4b82d671671b3ce744dec893dd15f5478614ce.
Session37190/launcher28496/worker27584 and audit76620/launcher8868/worker7208
CLOSED/exit0/no error. Capture held275.016s,worker257.219s;
audit held180.265s,worker169.750s;
combined455.281s. Capture OS held union
1810874368 bytes,
GPUallocated393388032/reserved425721856 bytes.
Audit OS held union953503744 bytes.
Initial binder/preparation costs are outside these held intervals.

Next [actual native assessment](CATEGORICAL_NATIVE_ASSESSMENT_PROTOCOL_20261010.md):
one new artifact, reuse qualified original executable,all48 forced histories/
8808 labels plus16 original chatbot tasks+own-answer followup. Verify old forced
routes/integer witnesses unchanged, compare all proxy/native scores under bounds,
held-out quality/own-history generation/raw rates on same artifact. Preserve full
audit regardless of quality outcome. Useful50/n/DRAM/CPU structured mass/families
remain unproved. No old inference/optimizer/checkpoint is restarted.
