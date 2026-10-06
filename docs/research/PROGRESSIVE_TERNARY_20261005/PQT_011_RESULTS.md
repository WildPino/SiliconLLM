# PQT-011: matched global adaptation with learned group amplitudes

6 October 2026. Qualified consumed-context comparison; no promotion. The tested
learned-scale procedure does not preserve predictive or generative behavior and
fails every preregistered per-domain and joint improvement signal. Wiki KL
improves only2.27–2.66% while continuation agreement worsens; news KL worsens
2.43–4.99%. This closes this bounded amplitude-adaptation test, not the whole
research goal or all possible ternary procedures.

## Qualification and scope

[Protocol](PQT_011_PROTOCOL.md), [input binding](PQT_011_CONSUMED_BINDINGS.json),
[bundle proof](PQT_011_BUNDLE_VERIFICATION.json),
[retention](PQT_011_RETENTION.json), [raw Git proof](PQT_011_GIT_VERIFICATION.json),
[standalone adjudication](PQT_011_ADJUDICATION.json),
[actual bootstrap/adjudicator source proof](PQT_011_BOOTSTRAP_SOURCE_VERIFICATION.json).

One private acct3 kernel137273085 v1, acknowledged05:05:13 UTC and COMPLETE
observed06:06:24 UTC. Status-only monitoring retained20 observations while the
coordinator waited dormant. Exactly one terminal retrieval:450 files,
1,969,737,112 bytes, every size/SHA256 exact;47 actual executable/input source
files match frozen Git9a47a85. Sixty-seven raw files/29,863,112 bytes match
retrieved/working/Git1abfd3d byte-for-byte. The actual server bootstrap differs
from the local bootstrap only by49 additional CR bytes; normalization of CRLF
to LF matches exactly. Embedded numerical/input source bytes remain exact.

The4,112,180-byte input binding remains byte-identical through private binary
transport. No new context selection: both evaluation domains and all calibration
identities were consumed before this trial. Original Qwen0.5B checkpoint,
tokenizer, corpora, seed20261005, pinned F32/SDPA T4 runtime and TF32-disabled
settings are verified again. No native CPU timing or whole-capability claim.

Four original-initialized arms each execute256 actual updates and32768 position
exposures on the same256x129 calibration windows, with a frozen original teacher.
ONE makes all169 unique matrices hard at the first update; STAGED makes them
all hard for the final64 updates. Original scalar vectors stay unchanged.
Learned arms add7,718,144 projected log-scale parameters to the latent matrices,
using the declared surrogate and joint clipping. These are joint-method
comparisons; weight-step normalization also changes when scale gradients join
the clipped norm. No optimizer trajectory replay is claimed.

The independent audit imports no writer/fitting modules. It verifies all169
unique matrices/494,032,768 original parameters, every256 original teacher
target, all matrix/scale counters, final archive/witness equations, original
vectors/aliases, every fitted-set F32 state/point and every consumed prediction/
own-prefix choice. All22 source/FIXED output identities reproduce qualified
PQT-009-R1 exactly, as do all fixed code/scale/vector member bytes. Independent
all-position evaluation F64 relative-RMS maximum5.776318689e-7; the separately
sampled128-position fitted-set F64 maximum7.351090190e-7, both below1e-5.
The latter is not an all-position fitted-set F64 bound. Separate stdlib
point/trace/capacity/gate reaggregation passes.

## Predictive and generative findings

Each domain has1024 next-token positions and eight32-token prompts. Prediction
agreement is agreement with the original model's argmax, not task accuracy.
Generation uses each candidate's own prefix/readout, with unmatched tails
counted as disagreements. Every arm has0/8 exact continuations in both domains.

| Domain | Arm | Argmax matches /1024 | Mean KL | Mean NLL increase | Continuation matches /256 |
| --- | --- | ---: | ---: | ---: | ---: |
| Wiki | FIXED_ONE | 270 | 2.940161 | 2.698603 | 15 |
| Wiki | FIXED_STAGED | 291 | 2.696645 | 2.472594 | 12 |
| Wiki | LEARNED_ONE | 297 | 2.861830 | 2.615444 | 9 |
| Wiki | LEARNED_STAGED | 299 | 2.635343 | 2.412673 | 9 |
| News | FIXED_ONE | 64 | 7.450929 | 7.462196 | 6 |
| News | FIXED_STAGED | 59 | 6.937140 | 6.949290 | 1 |
| News | LEARNED_ONE | 65 | 7.822753 | 7.866750 | 3 |
| News | LEARNED_STAGED | 59 | 7.105815 | 7.120394 | 4 |

All predictive/generative gates fail in every arm/domain: argmax>=99%, KL<=.01,
NLL increase<=.01, continuation-position agreement>=95%, exact traces>=7/8.
Storage and fitting-cost gates pass, without behavioral preservation.

Learned-versus-fixed same-schedule Wiki KL reductions are2.664169% ONE and
2.273261% STAGED, below the10% signal threshold. NLL/argmax improve but
continuation-position agreement worsens. News KL increases4.990298% ONE and
2.431471% STAGED, with worse NLL; ONE also worsens continuation agreement.
Every per-domain signal and both joint signals are false. Improved Wiki metrics
cannot substitute for the required two-domain behavior.

Descriptively, STAGED-versus-ONE KL reductions for the learned grid are7.914059%
Wiki and9.164782% news; fixed reductions are8.282404% and6.895639%. These do not
constitute a separately preregistered success decision or preservation result.

Final fitted-set results also fail preservation, so failure is not limited to
unseen calibration positions:

| Arm | Argmax matches /32768 | Mean KL | Mean NLL increase |
| --- | ---: | ---: | ---: |
| FIXED_ONE | 8685 | 2.898676 | 2.554108 |
| FIXED_STAGED | 9375 | 2.563065 | 2.286347 |
| LEARNED_ONE | 9293 | 2.807822 | 2.460960 |
| LEARNED_STAGED | 9572 | 2.498103 | 2.178597 |

## Capacity and actual scale/code changes

Every archive deploys493,961,216 two-bit ternary matrix codes, one F32 scale
per64 coefficients and71,552 original F32 vector coefficients. The shared
embedding/readout is counted once. Raw parameter bytes154,649,088 in all arms;
no latent float matrices, optimizer states or log parameters are deployed.
The no-update D archive supports independent change counts and is not an extra
evaluated or fitted arm.

| Arm | Complete archive bytes | Changed codes vs D | Changed scale groups | Audit-only witness bytes |
| --- | ---: | ---: | ---: | ---: |
| D | 154878220 | 0 | 0 | 0 |
| FIXED_ONE | 154889060 | 63715211 | 0 | 0 |
| FIXED_STAGED | 154893125 | 68679455 | 0 | 0 |
| LEARNED_ONE | 154891793 | 62820597 | 7718130 | 30894208 |
| LEARNED_STAGED | 154895858 | 67577881 | 7718136 | 30894208 |

Complete archive fractions15.675990–15.676679% of988,065,536 FP16 bytes all
meet the35% storage gate. Complete archive/member/header overhead is included.
Learned log witnesses are outside deployment and counted separately as research
overhead. Final learned alpha/alpha0 ranges are[.709163606,1.533678532] ONE
and[.667053103,1.497342348] STAGED. Neither endpoint saturates the declared
[.25,4] projection; original zero-scale groups=0. The final summaries match
independent witness reconstruction and all256 history zero-group counts.
The controls therefore rule out unchanged/missing learned scales for this run;
they do not establish a globally optimal scale solution.

## Charged resources

No remote failed attempt or repair in this trial. Installation17.378198s,
experiment process2922.530166s and audit process628.984975s; server elapsed
3568.983906s includes source preparation. Independent numerical audit timing
624.887037s is nested in its process time. Experiment includes setup170.839699s,
four-arm fit/export2544.834527s and evaluation187.605398s; do not add these
components again to experiment elapsed. Direct grid/export/all256 initial
teacher targets96.296098s are nested inside setup.

| Arm | Fit/export/final fitted-set verification seconds | Regenerated teacher-target seconds, included |
| --- | ---: | ---: |
| FIXED_ONE | 601.905030 | 88.093609 |
| FIXED_STAGED | 567.437118 | 87.801563 |
| LEARNED_ONE | 714.965798 | 89.718803 |
| LEARNED_STAGED | 658.636782 | 87.378115 |

Learned fitting/export takes18.783822% more time for ONE and16.072206% more
for STAGED in this matched run; these are process durations, not native inference
speed claims. Preparation controls and the first local transport admission
failure remain retained/charged separately. The first guard stopped in0.422s
before verification due to a concurrently owned Python preparation process;
sequential unchanged-source controls passed. No erased failure or silent rerun.

CUDA peaks are allocator measurements, not total physical VRAM. Experiment
allocated peaks[11,437,085,184,2,225,845,248] bytes, reserved peaks
[12,190,744,576,2,359,296,000]; reset per fitting/evaluation phase, overall
values independently checked as phase maxima. Evaluation allocated
[3,143,220,224,34,603,008], reserved[3,286,237,184,35,651,584]. Experiment
process RSS peak7,856,017,408 bytes. Audit allocated[0,5,144,406,528],
reserved[0,5,918,162,944], process RSS peak7,765,209,088. Linux ru_maxrss
is cumulative within each process; per-arm RSS entries are not isolated peaks.
Historical missing peaks remain unknown. Local GPU/native timing not performed.

## Decision and resumption

Do not promote the tested learned-scale procedure or extend identical runs
without a new prospectively justified hypothesis. The amplitude restriction was
relaxed with actual scale movement and matched input/update/storage scope;
neither same-schedule comparison satisfies the frozen signal, and full behavior
remains far from preservation even on the fitted set. This does not identify
a unique cause, prove a ternary approximation lower bound or reject different
discrete optimization, precision, grouping, budget or architecture choices.

The full goal remains active: original float-model natural expert contexts,
broader expert coverage, useful native integration/capacity and uncontended
CPU cost are still unresolved. Next prioritize those evidence gaps or freeze a
materially different output-guided discrete hypothesis before further fitting.
The ready native timing test still requires the owner's explicit CPU window;
machine idleness is not authorization. All current roles are consumed; any
future candidate needs separately frozen fresh validation before promotion.
