# Actual native head: complete moment/cost audit and precision-guard diagnosis

12 October2026. Goal ACTIVE/INCOMPLETE. No owned job live. Initial exact teacher
moments and loss/gradient are implemented and independently verified;no code
search, new learned model or native optimized-code upper has been produced.

## Scope and result

[Producer V2](../../../benchmarks/native_expert_scaling/native_head_domain_preflight_v2.py),
[frozen protocol](NATIVE_HEAD_DOMAIN_PREFLIGHT_PROTOCOL_V2_20261012.md),
[result](native_head_domain_preflight_result_v2_20261012.json),
[complete independent audit](native_head_domain_preflight_adjudication_v2_20261012.json).
All4422 FIT/all4386 DEV existing raw source final states,full65537 vocabulary.
Actual qualified F32 C/beta/epsilon promoted to F64 defines the new257-coordinate
head image;not the earlier paired255 image. Source q computed directly from
actual BF16 trained head/final gamma/source multiplier,source epsilon1e-5.

Compile m=A_aug^T q,H(q),teacher winner,current positive-sphere v,initial full
categorical L/gradient/supporting lower for every8808 rows. Stored two packets
54,825,964B total instead of dense teacher logits. Weights/splits unchanged;
no fitting from DEV and no untouched held-out claim.

| Initial promoted-F32 real-domain metric | FIT | DEV |
|---|---:|---:|
| Case meanKL | .99714032493 | 2.93424564790 |
| Case mean ID disagreement | 22.6093% | 41.0883% |
| Maximum sphere defect | 2.8422e-13 | 2.2737e-13 |
| Largest numerical support lower | -8.5424e-6 | -8.7385e-6 |
| Positive raw lower rows | 0 | 0 |

Initial supporting bounds prove no positive information floor. Clipping them to
zero would still not prove zero optimum,feasible good code or encoder capacity.
Source quality remains failed. No optimization or causal code generation here.

## First failure retained; complete diagnosis

Original frozen6505bae run79398 CLOSED1 after38.875s/OS1,749,098,496B at first
FIT max native-KL delta guard1e-5. No normal result and no numeric output packet;
original empty output namespace and
[terminal](native_head_domain_preflight_result_20261012.terminal.json) retained.
It was a mechanical acceptance failure,not a measured quality improvement.

V2 source/protocol01c746e keeps SAME equations/data/caps/1e-5 threshold;records
every delta and false flag instead of stopping there. Both fixed directional
derivatives2e-6/2e-7 pass2e-6 tolerance. KL guard remains FALSE on BOTH splits:
maxFIT1.32479e-5,maxDEV1.21752e-5. Decision remains
`COMPLETED_INITIAL_GATE_FAILURE_PENDING_AUDIT` in immutable producer result.
No retroactive tolerance change or original-run acceptance.

[Independent auditor V2](../../../benchmarks/native_expert_scaling/native_head_domain_preflight_audit_v2.py)
does not import producer equations. Actual2048-wide source logits accumulate
eight256-feature blocks;full logaddexp probabilities;moments/gradient accumulate
4096-vocabulary blocks. Direct full q*(logq-logp) verifies sufficient-moment loss.
Explicit scalar half-ball support minimizers verify every lower. Both finite
derivatives independently pass. Every moment/entropy/teacher ID/code/gradient/
loss/weight/summary verified,max relative1.7125e-11,scalar7.6864e-10.

Additionally compares ALL577,249,896 actual archived native F32 scores with
real-domain logits,full native teacher KL and stored per-label deltas/IDs.

| Real promoted head versus actual F32 kernel | FIT | DEV |
|---|---:|---:|
| Max absolute score difference | 5.47974e-5 | 5.78133e-5 |
| Max absolute source-relative KL difference | 1.32479e-5 | 1.21752e-5 |
| Changed argmax | 0/4422 | 0/4386 |
| Score-range bound violations | 0 | 0 |

For delta=s_real-s_native,the exact-real KL difference is
`log E_softmax(s_native) exp(delta)-E_q delta`. Both terms lie between the
minimum and maximum delta;absolute KL change<=max(delta)-min(delta).
Every row passes this bound with1e-9 numerical slack. Independent source/head/
moment checks and full score comparison isolate the failed1e-5 guard as stricter
than observed arithmetic rounding,not a data/formula drift. Numerical comparisons
are not interval certificates. Original native export numeric gates stay as
previously qualified;no new native optimized-code quality admitted.

Audit decision `INDEPENDENT_STORED_PASS`,initial_gates_pass=false. The moments
and real-domain equations are therefore qualified analysis inputs;the original
F64-versus-kernel gate is not promoted to PASS. Any new code claim must separately
execute actual C and verify probabilities/IDs,using a frozen justified precision
contract. A solver's F64 optimum alone is not a native upper certificate.

## Actual cost and custody

CPU6 threads/logical affinity0..5;no GPU/model loader/T4/subprocess. Two warmups,
five identical batch64 objective/gradient calls: .203,.281,.203,.187,.203 seconds;
median.203s. Nominal139-batch pass28.217s,not optimizer or chatbot throughput.
All-label source moment compilation39.613s;initial evaluation23.154s.
Timing admission=false,known external publishers retained.

V2 run79763 CLOSED0,PID28824/create1791819192.8571754,result75.094s/terminal75.125s,
OS1,749,000,192B. BindingSHA
e7d232c4cf55aebab3cbd0735a2723f2db49db0c9ce3f787b911ff179eb54f4a;
[launch](native_head_domain_preflight_launch_v2_20261012.json)1fb32d6.
ResultSHAa091f54cdd157fa6d97cec9422b66d5193afcea80da1ef368d64c891902e6251;
terminalSHA95b3b4043ce048a34f6d74807a8d169e76edd51d798bdc9f0f0783b3722e7142.

Audit37970 CLOSED0,PID22268/create1791819307.869484,143.281s,
OS3,067,551,744B;source/protocol e9a6342,
[launch](native_head_domain_preflight_audit_launch_v2_20261012.json)7cf7b30.
AdjudicationSHA962e031912d9adb9146142d34f90dd11d4d97f0d99825494dfff529428f062a2.
All900s/4GiB/96MiB producer and900s/4GiB audit caps PASS. Full relevant
input/output before/after hashes PASS; original engine/foreign files unchanged.
[Session closure](native_head_domain_preflight_session_closure_20261012.json).

## Decision and precise next

Next selected **bounded affine-ray decision feasibility probe**, still
UNIMPLEMENTED/UNBOUND. See
[head-domain mathematical specification](NATIVE_HEAD_DOMAIN_FEASIBILITY_NEXT_20261012.md).
Since positive RMS divides all logits,the teacher winner j requires finite t
satisfying (C_j-C_k)^T t>=beta_k-beta_j for every k. Freeze d_j from actual head
and t=s d_j:all inequalities reduce to an interval in s. Use existing audited
teacher IDs,deduplicate identical target winners to reduce cost;retain all8808
label associations and actual native verification for any candidate admitted.
An empty ray interval rejects only that ray,not256-dimensional head capacity.
This is an oracle decision test,not native routing or distribution preservation.

Its purpose is to separate decision availability from failed current linear
encoder before choosing full hemisphere/half-ball probability search. Full
probability quality/causal compact SSM/SWA/ternary acquisition/useful experts,
CPU IDs+mass/physical DRAM/multiple families/same useful artifact>=50 remain open.
No long epoch selected;no repeated unchanged old paired255 descent. Goal active.
