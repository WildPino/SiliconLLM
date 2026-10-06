# PQT-010: shared embedding/readout restoration results

6 October 2026. **Independent audit and standalone adjudication pass; no
preservation pass or promotion.** This is a consumed-development intervention
on the two frozen broad PQT-009-R1 payloads. All four float/int8 restoration
signals fail jointly and separately in each domain. Restoring the original
shared parameter worsens KL in every comparison. This does not identify a
unique failure cause or establish a lower bound for ternary representations.

## Inputs, intervention and preserved failures

[Protocol](PQT_010_PROTOCOL.md) and [exact binding](PQT_010_CONSUMED_BINDINGS.json)
precede outputs. T_ONE/T_STAGED are byte-identical B_ONE/B_STAGED archives.
F restores original F32 `model.embed_tokens.weight` and its tied readout alias;
I8 replaces that parameter with source-derived rowwise signed int8. All other
168 matrices and all float-vector members are byte-identical to their parent.
The tied parameter is stored once. The two functional uses are intervened on
jointly; the other trained parameters remain coadapted and frozen.

Original private kernel137260621 v1 completed evaluation but failed its cold
CUDA memory reset before the independent numerical audit. All168 files and
21 actual source bytes were retained; see [first failure](PQT_010_AUDIT_FAILURE.md).
The earlier unacknowledged dispatch001 and repeated exact404 reconciliations
remain immutable. Its HTTP rejection cause is unknown; compressed source
transport preserves exact scientific bytes and does not prove a platform limit.

R1 kernel137262743 v1 reproduced the cold-reset failure and qualified the
single `torch.cuda.init()` insertion. Its requested ERROR-kernel input was not
attached: push declared invalid_sources and server metadata kernel_sources was
empty. Preparation returned ValueError before original input/function access;
no more precise failing branch was logged. All 13 files/31 actual sources and
12 small raw files/13,941 bytes are retained. Do not restart either job.

[R2](PQT_010_R2_PROTOCOL.md) changes only transport to a private stored binary:
165 original files, 3,713,630,430 bytes, SHA256
`edcfbaad587adc45a056a2ca88dd78337100a4c015397a5e72bed24f99ee5b1f`.
Private dataset12394938 v1, acct3 kernel137264094 v1, COMPLETE observed
03:43:41 UTC by the delegated status-only monitor. Parent waited dormant.
One terminal fetch retained 181 files/3,714,609,729 bytes. All 40 embedded source
files match scientific commit `4a90b8cf88fd8310d2875ca52eb7d1d9e9aaae9f`;
original scientific source remains `91145c8d5cd99096974cf94d0529769dcbf59055`.
All 113 small raw files/18,002,479 bytes match fetch, working files and Git
`4883a267d49891c3e2b9d8561a8f026b4094a30a`; see
[retention](PQT_010_R2_RETENTION.json), [Git proof](PQT_010_R2_GIT_VERIFICATION.json).
No evaluation or archive construction repeated, **zero new optimizer updates**.
First failures are retained within and outside the repaired audit view.

## Predictive and own-prefix generative behavior

These are agreement with the original teacher, not labelled task accuracy.
Each domain has eight128-position windows and eight32-token prompts; unmatched
continuation tails count as disagreement. All arms have zero exact traces.

| Domain | Arm | Teacher argmax agreement | Mean KL | Mean NLL delta | Generation positions | Exact traces |
| --- | --- | --- | --- | --- | --- | --- |
| wiki | T_ONE | 270/1024 (26.367188%) | 2.940161042499 | +2.698603213375 | 15/256 | 0/8 |
| wiki | T_STAGED | 291/1024 (28.417969%) | 2.696645033377 | +2.472594463830 | 12/256 | 0/8 |
| wiki | F_ONE | 256/1024 (25.000000%) | 3.531413080419 | +3.364620275780 | 4/256 | 0/8 |
| wiki | F_STAGED | 285/1024 (27.832031%) | 3.214700345187 | +3.059881843289 | 4/256 | 0/8 |
| wiki | I8_ONE | 256/1024 (25.000000%) | 3.531188443635 | +3.364230787665 | 4/256 | 0/8 |
| wiki | I8_STAGED | 285/1024 (27.832031%) | 3.214896884735 | +3.060057016616 | 4/256 | 0/8 |
| news | T_ONE | 64/1024 (6.250000%) | 7.450929161571 | +7.462196464682 | 6/256 | 0/8 |
| news | T_STAGED | 59/1024 (5.761719%) | 6.937139949204 | +6.949289734039 | 1/256 | 0/8 |
| news | F_ONE | 65/1024 (6.347656%) | 7.534956275155 | +7.515540312229 | 3/256 | 0/8 |
| news | F_STAGED | 58/1024 (5.664062%) | 7.095645070787 | +7.078011044900 | 2/256 | 0/8 |
| news | I8_ONE | 65/1024 (6.347656%) | 7.535402511425 | +7.515868176977 | 3/256 | 0/8 |
| news | I8_STAGED | 58/1024 (5.664062%) | 7.096051574647 | +7.078595590902 | 2/256 | 0/8 |

Absolute gates: >=99% argmax, KL<=0.01, NLLdelta<=0.01, >=95% generation-position
agreement, >=7/8 exact traces, complete archive<=35% of988,065,536 FP16 bytes,
charged parent fit<=7200 s. Every quality clause fails for every arm/domain.
T/I8 meet storage; F fails storage and is only an excess-storage diagnostic.
All fitting-cost clauses pass. Consumed roles prohibit promotion even if a
descriptive gate had passed.

## Frozen relative criteria

Each F/I8 intervention vs its same-schedule T baseline required >=10% lower KL,
no worse NLL, argmax or generation-position agreement, separately both domains.
Negative reductions below indicate worse KL. Blank satisfied-clause entries
mean no clause passed.

| Domain | Comparison | KL reduction | Satisfied clauses | Signal |
| --- | --- | --- | --- | --- |
| wiki | F_ONE vs T | -20.109512% |  | Fail |
| wiki | F_STAGED vs T | -19.211105% |  | Fail |
| wiki | I8_ONE vs T | -20.101872% |  | Fail |
| wiki | I8_STAGED vs T | -19.218393% |  | Fail |
| news | F_ONE vs T | -1.127740% | argmax_no_worse | Fail |
| news | F_STAGED vs T | -2.284877% | generation_positions_no_worse | Fail |
| news | I8_ONE vs T | -1.133729% | argmax_no_worse | Fail |
| news | I8_STAGED vs T | -2.290737% | generation_positions_no_worse | Fail |

F and I8 produce identical argmax counts and generation-position counts in
this sample, with small KL/NLL differences. This is a descriptive observation,
not a precision equivalence claim. Restoring F32 raises Wiki KL by20.109512%
(ONE) and19.211105% (STAGED), news by1.127740% and2.284877%. Neither intervention
rescues these frozen payloads. A negative restoration can involve coadaptation;
embedding vs head or any other unique cause is not isolated.

## Entire deployed capacity

All scales, F32 vectors, descriptors, NPY headers and ZIP members are counted.
Overhead below is archive minus raw parameter bytes, not ZIP headers alone.
All archives are stored uncompressed. No optimizer/latent state or external
floating matrix cache is a deployed path. Shared parameter136,134,656
coefficients is counted once despite two aliases.

| Arm | Raw parameter bytes | Complete archive bytes | Total overhead bytes | Fraction of FP16 |
| --- | --- | --- | --- | --- |
| T_ONE | 154,649,088 | 154,883,496 | 234,408 | 15.675427% |
| T_STAGED | 154,649,088 | 154,887,561 | 238,473 | 15.675839% |
| F_ONE | 656,645,632 | 656,889,494 | 243,862 | 66.482381% |
| F_STAGED | 656,645,632 | 656,893,550 | 247,918 | 66.482791% |
| I8_ONE | 248,849,408 | 249,093,774 | 244,366 | 25.210248% |
| I8_STAGED | 248,849,408 | 249,097,842 | 248,434 | 25.210660% |

T contains 169 ternary matrices with group64 F32 scales and counted F32 vectors.
F/I8 are explicitly mixed. Float diagnostics occupy66.48% of FP16; I8 occupies
25.21%, so its quality failure is not caused by breaching the declared storage
limit. Whole-model GPU float reconstruction is not packed native deployment.

## Independent verification and limits

The separate audit imports no intervention writer or fitting/evaluation driver.
All 494,032,768 original parameters and 169 unique matrices are verified against
the immutable original checkpoint and inventory. It checks all six archives,
source-derived float/int8 rules, all 168 unchanged matrix members, all vectors,
original aliases, 165 binary members, 156 original audit-view files and nine
first-failure files. The original numerical audit source is unchanged.

All 2048 original and 12,288 candidate prediction hidden states replay exactly.
All 112 original/candidate own-prefix traces and 3584 actual choices replay exactly.
F32 point metrics and complete-position independent F64 readouts pass, maximum
relative RMS5.776318689e-7 against1e-5. Original/T point/state/generation output
identities reproduce 22 qualified R1 files exactly. Standalone standard-library
reaggregation independently recomputes every point, per-window metric, trace,
archive raw geometry/overhead and absolute/relative decision; see
[adjudication](PQT_010_ADJUDICATION.json). Optimization trajectories are not
replayed. Audit qualification establishes trustworthy measurements, not
successful preservation or broad semantic capabilities.

## Charged costs and memory

Parent fit/export 1,809.718722588 s is charged once, not retrained in this
experiment. The full original evaluation and both repair/failure attempts
remain charged. Bootstrap process wall time below includes subprocess startup;
nested control/audit internal elapsed times must not be added again.

| Attempt | Phase | Process seconds | Return code |
| --- | --- | --- | --- |
| Original 002 | install | 16.065978473 | 0 |
| Original 002 | experiment | 360.047353374 | 0 |
| Original 002 | audit | 6.233696130 | 1 |
| R1 | install | 17.565632062 | 0 |
| R1 | memory_control | 8.988001534 | 0 |
| R1 | repair | 2.120698790 | 1 |
| R2 | install | 16.154728994 | 0 |
| R2 | memory_control | 8.284477941 | 0 |
| R2 | repair | 119.930097509 | 0 |
| R2 | audit | 450.932123422 | 0 |

The R2 preparation includes input extraction/identity verification and original
checkpoint acquisition; its audit includes full numerical replay. Upload,
platform scheduling and terminal artifact transfer are additional operations;
no complete end-to-end compute attribution is inferred from these process times.
Original failed-audit memory is unknown. Historical parent audit peaks were
not instrumented and remain unknown.

| Process | CUDA peak allocated bytes (device0,1) | CUDA peak reserved bytes (device0,1) | RSS final bytes | RSS peak bytes |
| --- | --- | --- | --- | --- |
| Original evaluation | [3613846528, 0] | [4143972352, 0] | 4,559,036,416 | 6,457,192,448 |
| R2 independent audit | [0, 4921188864] | [0, 5431623680] | 3,276,783,616 | 5,442,334,720 |

CUDA statistics measure allocator counters in that process, not physical total
VRAM; Linux RSS peak uses process ru_maxrss KiB converted to bytes, not total
host memory. R1 startup-only control had no model allocations and cannot
substitute for a full audit peak. Evaluation/audit used frozen F32 SDPA runtime,
seed20261005, TF32false and two Tesla T4 devices; no local GPU or native timing.

## Decision and next dependency

Do not promote T, F or I8. This experiment closes the prescribed frozen shared
parameter restoration question; it does not close the research goal. A useful
next comparison can change trainable ternary scale amplitudes at the same
counted storage while matching data, update schedule and original initialization.
Freeze that hypothesis and control derivatives/tying/serialization before model
work. This direction is motivated by an untested restriction in the previous
method, not a diagnosed cause of failure. Natural original-expert contexts,
broader expert coverage, native timing/integration and broader capabilities
remain required. Native CPU timing still awaits the owner's explicit
uncontended reservation; an idle observation is not permission.
