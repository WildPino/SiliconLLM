# Token prototypes/shared context correction: audited failure and necessary rank

13 October2026. **COMPLETE / NUMERIC AUDIT PASS / COMPACT REPRESENTATION REJECTED.**
Goal ACTIVE/INCOMPLETE. No source model/GPU/SSM/optimizer/RESERVED query.

## Experiment and observations

Actual qualified chatbot H at sites1/12/23;all48 histories,22547 positions.
FIT24/11148 positions,2248 distinct tokens;DEV24/11399 positions. BF16 per-token
FIT means/global fallback plus a single output-aware linear residual operator.
Global mean control. Fixed ranks0/128/256/512/1024. Every case/every preconv
and causal field must have relative RMS<=1%,compact rank<=256. No DEV tuning.
Unknown DEV token positions24.730239% retained in every gate.

All6 variants fail even FIT at rank1024;no selected rank or compact candidate.
All dense canonical-source controls pass. Input residual energy/total energy
after token means17.0901%/48.6122%/45.0629% at1/12/23. This is energy,not a
fraction of knowledge or a Shannon-information estimate.

| Site | Rank256 DEV mean of each case's worst field | Rank1024 same measure | Rank1024 worst case field |
|---|---:|---:|---:|
|1|52.2241%|31.2499%|41.6385%|
|12|61.5189%|33.8047%|38.7978%|
|23|49.5557%|28.8181%|37.0597%|

Zero/24 DEV cases pass all fields in every listed row. Measures include both
preconv and causal fields;they are not whole-chatbot KL or task results.

## Independent numerical evidence

All1440 candidate curves/unknown counts/selection decisions checked. FIT means/
covariance exact;root relative residual<=3.16e-15,basis<=1.12e-14,source-factor
<=7.37e-15,weighted eigen identity<=1.06e-14. Shared NumPy BLAS is declared.
No numerical defect found;failed local accuracy is preserved. All inputs and
6 matrix files rehashed at closure. Producer672.969s/OS1,858,924,544B,
audit730.266s/OS2,113,642,496B;E888,591,804B<1GiB. Both terminal0.

[Binding](token_context_rank_binding_20261013.json),
[all curves](token_context_rank_result_20261013.json),
[independent audit](token_context_rank_adjudication_20261013.json),
[closure](token_context_rank_session_closure_20261013.json).
Source686fff5,binding450a746,result0a9c6f1. AuditSHA
1398090eade6a38eeaf1c3e8b40ace0507983ed7a75b426b9d00f41c86a50fe5.

## Algebraic family bound

Conditional FIT mean minimizes weighted residual variance for ANY fixed shared
linear correction. Changing token-only prototypes adds a PSD covariance term.
Full discarded weighted spectrum yields the optimal rank-k residual floor.
Subtract exact BF-mean offset energy and include measured source-reference
discrepancy plus conservative two-stage F32/BF16 relative/subnormal allowances.
Recompute FULL spectrum independently;do not rely on only1024 eigen identities.

| Site | Necessary shared rank for1% preconv fields | Rank256 weighted RMS floor | Feasible weighted RMS upper bound |
|---|---:|---:|---:|
|1|>=1408|.32903|.04487|
|12|>=1857|.59830|.04487|
|23|>=1857|.55137|.04488|

Numerical algebra witness,not interval-certified arithmetic. Necessary rank is
not sufficient. Scope:ANY token-only prototype and ONE common linear residual
operator on this FIT set/actual G under the stated field gate. Does not exclude
token/context-dependent operators,nonlinear transformations or a new whole-model
loss hypothesis. No universal dimensionality/impossibility claim.
[Derivation](TOKEN_CONTEXT_RANK_BOUND_PROTOCOL_20261013.md),
[bound](token_context_rank_bound_20261013.json),source16da4e9;
29.578s/OS883,474,432B/terminal0. BoundSHA
8e340a41fa37df8779a69ffba3b92e42b9d5a50909293d9f5d490f3a4de498c6.

## Decision

Do not compile all-vocabulary token means or retry shared rank256 to fix this
FIT failure. It would not change the family bound. Linear G has the same
Jacobian everywhere;duplicating G alone does not reduce active cost.
Keep the qualified wide native backbone. Next investigate full-size conditional
FFN operators where the Jacobian actually changes with input;see
[next expert contract](FULL_RANK_FFN_EXPERT_NEXT_20261013.md).
Final CPU engine/useful n/actual DRAM/same useful50/other donor families remain open.
