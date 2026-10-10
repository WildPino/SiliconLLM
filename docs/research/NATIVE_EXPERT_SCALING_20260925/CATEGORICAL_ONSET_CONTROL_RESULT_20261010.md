# 24-first-state code controllability: completed and fully audited

10 October2026. **FIRST_CODE_MATCH_ROBUSTLY_OUTSIDE_RADIUS**. All numeric flags
and complete stored audit PASS. [Protocol](CATEGORICAL_ONSET_CONTROL_PROTOCOL_20261010.md),
[result](categorical_onset_control_result_20261010.json),
[audit](categorical_onset_control_stored_adjudication_20261010.json).

## Measured algebraic result

24 FIT first-state vectors have numerical rank24/condition82.8160319463.
Approximate minimum singular value0.371092035513; uncertainty
matrix norm upper0.0418680150158; true minimum
singular value lower0.329224020498 under adopted F32
arithmetic model. Rank remains positive with stated uncertainty.

The existing24 good per-label reference codes have source-relative meanKL
0.00412700483273/max0.0311934616336,zero native argmax
errors,adopted unchanged from completed72-label probe. Matching them exactly
with one shared map requires minimum Frobenius norm31.269885992 on
approximate states. Minimum correction to current map norm31.2790787564;
preserved map31.330779696,base2.03547912593.
Minimum/preserved all24 code residuals5.55111512313e-15/
5.46784839628e-15. Full projector/Pythagoras/dual identities PASS.

Dual witness pairing977.805769953,denominator including native error
33.7831137614 gives true exact-reference-code norm lower
**28.9436248197>16**. The previous optimizer's
Frobenius16 excludes this exact reference-code interpolation,even under the
stated uncertainty model. It was an introduced fit constraint; original C has
no corresponding16 bound on F32 head weights. Neither this code constraint nor
these feasible per-label references prove a positive global categorical-KL
floor,information/rank ceiling or generalization. Bounds evaluated F64,
not interval certified; exact matching chosen codes stronger than acceptable KL.

## Complete audit/cost/provenance

One NEW24x256 SVD,no prior4422-row SVD/optimizer/head FG/source/native/GPU/DEV/
RESERVED/T4 call.199 inputs49076590B,junction-resolved actualNumPy2.4.6/psutil7.2.2
module sources/bytecode/native libraries. All input/output hashes,24 exact
source/native first positions/IDs,full factors/orthogonality,minimizers/projector/
null component/dual witness/native uncertainty/counters/caps PASS.24 independent
math.fsum dot witnesses,max scalar delta2.77555756156e-16.
Full SVD relative9.85170400996e-16,orthogonality1.31309154674e-14.
Audit does no SVD or optimization replay.

Numeric freeze6f3462cc5d0a3d016d2da9bd609366c9ae067e2d;
runtime binding4a60d8e6c68879a1b5f3154baa84fd6370b982a2b2e7a5db906ac28bac9cf301.
Separate holder freezee5ef334f62f9493a676231312f26adacd6ce3b8c/binding9e89e537.
Producer held2.187s/worker1.344s;
audit held1.234s/worker0.265s;
combined3.421s.
Producer OS union81162240B,
audit union80715776B;
output14 files3385856B.
Results SHA248a11c5e4810480b372bfe9462d30c8b4eedc05158a9c60900c858453d5e12e/audit3482b85bef799571c99a8d0ff3d4a632f2c8371e74094eef7aeca79174ca2675.
Both direct held commands terminal/exit0; exact PIDs/creation in receipts.
Initial binder omitted junction-resolved module paths; unconsumed preparation
and code retained. Original launcher rejected foreign uploader before any child/
namespace/SVD. Separate holder frozen; uploader had exited naturally before
binding,so no foreign uploader permitted/observed in executed holder. No foreign
work killed/messaged. Previous principal-runtime scopes are not retrospectively
converted into complete module custody by this new seal.

## Next new executable correction

[One analytic onset head](ANALYTIC_ONSET_HEAD_PROTOCOL_20261010.md) implemented/
AST PASS/bound,UNEXECUTED. Use already computed preserved map of norm31.3308;
remove only old optimizer radius for this closed-form point. Offline fuse H=A
Theta W as an ordinary original packed F32 head,109 other fields byte-identical.
Evaluate ALL8808 stored forced labels/first-vs-continuation/DEV and uncertainty,
then independent full audit. No new optimizer/SVD/model/native history. This
measures whether global correction spills into other states before choosing a
conditional-expert or causal-code intervention. Native/chatbot/useful50/n/DRAM/
CPU structured mass/family gates remain required and open.
