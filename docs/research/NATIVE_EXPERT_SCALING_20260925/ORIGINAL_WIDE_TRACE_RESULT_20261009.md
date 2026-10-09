# Actual26 original math: first discrete divergence localized to AQ

9 October2026. Goal ACTIVE/INCOMPLETE. All owned jobs terminal. No T4/source/
RESERVED/optimizer calls. Previous goal turn was progress: actual26 result and
zero-factor diagnosis. This turn implements and executes the selected diagnostic
replay, audits the first integer difference, and prepares a legal FIT seed helper.

## Decision toward the conversion pipeline

The original LUT arithmetic is consistent in the observed scope. GPU/C floating
trajectories cross an AQ rounding boundary, which changes discrete inputs and
then expert outputs. A small smooth error cannot be bounded through a discontinuous
rounding operation by applying the same relative tolerance to its final output.
This does not retroactively pass the declared GPU/C numerical gate.

Close the search for a LUT summation defect on these observations. Preserve the
native endpoint and explicitly treat GPU training as a surrogate. The actual26
native source-relative quality is still poor; the measured supervised GPU/C KL
delta .0010966 is small against native FIT KL6.312887. There is no evidence that
correcting floating parity alone would recover the donor's missing knowledge.
Do not spend an unbounded campaign making the surrogate bit-identical.

Next: apply the prepared FIT-response seed to unlock32 delta directions in an
explicit actual26 fork, measure function preservation/real new gradients/native
output after one new step, then select a finite joint source-history recovery
pilot. Original LUT/ternary/SSM/SWA operators remain the deployment endpoint.

## Frozen experiment and observer qualification

[Protocol](ORIGINAL_WIDE_TRACE_PROTOCOL_20261009.md),
[binding](original_wide_trace_binding_20261009.json),freeze
548c1902fa9692b8f12e6fad975830c332b63fbd. Binding SHA
933a15671409fe1d7d1b9fbf3d7112b2c6d805b3b556fa096a5c09c5457f0234/35 inputs.
Actual26 checkpoint/packed export from [bridge](ORIGINAL_WIDE_BRIDGE_RESULT_20261009.md),
one existing1507-position FIT history `broad_fit_smol_magpie_ultra_022`.
New inference is explicitly diagnostic; no training/before-checkpoint replay.

Thirteen marked observer insertions copy computed values. Removing them recovers
the prior extracted header text exactly,SHA62e0d8c33dc8dd284665be959821b06b05f5b9e985164ffad35498e7d3861e0c.
No change to original arithmetic or ternary byte-pair storage. Native single-FIT
consumer reads existing request index1. GPU uses actual masters and exact existing
stream-bank operations, with intermediate copies and a named integer temporary.

Instrumented GPU full65537 head rows are byte-identical to retained GPU_after.f32.
Instrumented native full heads match retained native FIT rows357..1864 bitwise;
all native IDs/masses also bit-identical. Observer identity is therefore measured
on98,764,259 head values per path. It is not inferred from source similarity.

Both persisted records have shape1507x6/itemsize46436/419,874,312B each:
pre/core norm/core/postcore/FF norm/MoE/postMoE,raw router scores/probabilities,
IDs/masses,input AQ codes/scales,gate/up/down integer sums,gate/up/hidden floats,
hidden AQ codes/scales. The previous GPU-route persistence gap is resolved by
this new diagnostic; it remains a gap in the old completed experiment.

## First observed discontinuity and exact integer identity

First input-code difference:token82/site0/coordinate43. At that point pre and
core norm are bit-identical; core relative RMS5.85334e-7,postcore7.22150e-9,
FF norm8.76056e-8. Its selected8 expert IDs are identical.

| Quantity | GPU | Native |
|---|---:|---:|
| FF input coordinate43 | -0.5851624011993408 | -0.5851624011993408 |
| Maximum absolute input,coordinate2 | 1.170324683189392 | 1.1703248023986816 |
| AQ scale | .01857658289372921 | .01857658475637436 |
| Coordinate43 after reciprocal multiplication | -31.500003814697266 | -31.499998092651367 |
| Rounded AQ code | -32 | -31 |

Both observed scales match correctly rounded division of their own maximum by63.
This witness does not show a division/rounding implementation bug. Slightly
different upstream inputs produce slightly different maxima and straddle a
half-bin at another, identically represented input coordinate.

Only coordinate43 differs in that input code vector. For each of the8 selected
experts andboth gate/up matrices,the actual integer-sum difference equals exactly
the ternary weight column43 times(-1). This is independently verified by decoding
the bound packed byte pairs. Sixteen128-element vector equalities prove the first
LUT difference comes from this input-code flip, not a corrupted accumulation.

At token82/site0,MoE relative RMS becomes .005034428;postMoE .000363507. The
quantizer can amplify sub-tolerance floating changes into a larger output
difference. No assumption of a globally smooth surrogate is justified.

## Complete retained trace scope

| Discrete field | Different coordinates | Different token/sites |
|---|---:|---:|
| Ordered expert IDs | 9 | 4 |
| Input AQ codes | 219 | 41 |
| Hidden AQ codes | 1977 | 47 |
| Gate integer sums | 31457 | 41 |
| Up integer sums | 31561 | 41 |
| Down integer sums | 60278 | 47 |

Ordered routing differs at(82,5),(1251,2),(1251,5),(1324,4). Only(1251,5)
changes the expert set;three calls only permute the same set. Maximum route mass
delta .001416116953 reproduces the earlier worker observation.

All observed gate/up integer sums agree when ordered expert IDs AND input codes
agree. All down sums agree when ordered IDs AND all hidden codes agree. These are
conditional checks over recorded selected products, not a proof of all bank
coefficients or of general n. Raw rank-indexed g/u/h float metrics include
different experts at routing mismatches; do not interpret those maxima as
same-expert arithmetic error.

The original26 failed head rows/worst RMS .07098787967/greedy mismatch339 reproduce
exactly.24 of26 failed rows have a same-token AQ orhidden-code difference.131/140
do not; earlier state/trajectory propagation is a possible explanation, not
independently proved by this trace (full recurrent states were not recorded).
Temporal cooccurrence alone is not a causal proof for every failed row.

No new source-relative quality scores,chatbot generation oraccepted-speed gate
were tested. The prior native three-case quality failure remains authoritative.

## Prepared correction: FIT directions outside existing delta responses

New [helper](../../../benchmarks/native_expert_scaling/original_delta_fit_seed.py)
implements a concrete linear initialization within the original x_proj operator.
It has not yet calibrated actual26 or changed a checkpoint.

Let X be FIT post-convolution/SiLU activations,T x1024;U0 existing16 delta rows.
D0=X U0^T. Solve A=D0^+ X andform R=X-D0 A. Principal32 right singular directions
W of R select variance not represented by existing delta responses. Fold the
residualization into legal weight rows:

```
Unew = W - (W A^T) U0
X Unew^T = R W^T
```

Scale each row toward median RMS of old FIT responses,with row L2 capped at10x
median old-row L2. Fix signs deterministically. Require full old response rank,
32 residual directions above1e-12 of original activation energy,finite rows,
cross-response relative<=1e-6 after F32 casting,and retained new response rank.
Reject numerical residual noise when old responses already span the inputs.
Vnew remains zero;new Adam moments remain zero. The mathematical initial delta
function is unchanged,while dVnew=g(Unew x)^T can become nonzero. New U does not
need a gradient in that first step. Native function preservation still needs
an actual measurement. FIT variance is not proof of donor information recovery.

[Algebra check](original_delta_fit_seed_algebra_20261009.json):deterministic192x48
features/oldrank4/newrank6,zero-read output bit identity,cross-response3.610e-9,
54 nonzero synthetic new-V derivatives,rank-deficient fixture rejected.
Helper raw SHA dfdadbd692b309dc8490909fa75f7b712a57b97c5d55716efd8bbb3e60d65c55;
no actual model gradient
orfit/native quality claim. Script elapsed .031s;no separately held family peaks.

## Resources, faults and independent audit

Launcher26868/worker28980/create_time1791568004.1199355/exit0/session56724 CLOSED.
Held family204.000s;worker result176.391s. Compile12852/create_time1791568007.542047/
exit0/held2.547s/direct peak4,771,840B. Native32788/create_time1791568052.866265/
exit0/held128.640s/direct peak528,306,176B. Heavy trace IO makes its duration a
diagnostic cost,not a speed benchmark. Observed nested compiler/linker separate
peak was not held. OS worker6,168,870,912B +launcher29,102,080B +direct max528,306,176B
=6,726,279,168B conservative sum,not simultaneous peak. GPU allocator598,123,520/
813,694,976B (not full device residency).13 outputs1,630,177,227B.
900s/reserve90/OS20GiB/GPU6/7GiB/output2GiB/log4MiB caps PASS.

Raw [result](original_wide_trace_result_20261009.json) SHA
e775a6cc4977f55c1afe826931b744d92f3636290c195c112d190e0282ea97a5,
[terminal](original_wide_trace_result_20261009.terminal.json),
[log](original_wide_trace_result_20261009.worker.log) retained.
[Independent stored audit](original_wide_trace_stored_adjudication_20261009.json)
rehashes35 inputs/13 outputs,verifies observer heads,all discrete counts/locations,
conditional gate/up/down sums,andfirst single-column difference frompacked trits.
Its script elapsed14.187s,not separately held runtime resources.
[First stored-helper fault](original_wide_trace_stored_audit_first_fault_20261009.json)
preserves exit1/session14578:rank/channel equality reduction shape error. Corrected
only the new unbound audit helper;stored-only rerun session63367 exit0. No main
worker/compiler/native/model replay was needed. Frozen experiment code unchanged.

Goal remains ACTIVE/INCOMPLETE:useful native chatbot and>=50 accepted batch1 on
same artifact,useful RAM-driven n/CPU routing/mass/physicalDRAM andmultiple
families/~10B/~100B remain missing. [Next](ORIGINAL_ENGINE_TRANSFER_NEXT_20261009.md).
