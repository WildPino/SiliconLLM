# PQT-009-R1: broader calibration helps locally, preservation still fails

6 October 2026. **Qualified bounded negative result; overall goal incomplete.**
Complete independent replay passed; standard-library reaggregation agrees with
the raw points, traces, audit and unchanged preregistered decisions. No arm
passes preservation, and none of the three joint relative signals passes.
Broader coverage is useful evidence for further diagnosis, not a promoted method.

## Immutable objects and data history

Parent PQT-009 source `6630edb8abbf3711530247d6dbb6c1380102c5d1` fitted four
arms and froze five complete archives before its retained selection failure.
R1 source `eebbba4a5af06e0def11317d9a518208f7ea2000`, bundle `dadf2cc`, kernel
137254285 v1 performed **zero new optimizer updates**. All 75 original files,
22 actual embedded parent sources, and 61 fitting-view files are byte exact.
The first failure remains distinct under `parent_provenance/`; original
runtime/input/cost/history records were not overwritten by repair records.
Narrow code/scale/vector identities reproduce the earlier PQT-008 artifacts.

`N` uses the first 16 WikiText train windows, 2,048 unique prediction positions;
`B` uses 256 prespecified distinct corpus-stratified windows, 32,768 positions.
Each fitted arm has 256 actual updates and 32,768 position exposures. ONE is
fully hard on all 256 updates; STAGED on 64. Original fixed group-64 scales,
counted F32 vectors, and shared embedding/readout tie are unchanged. This
compares coverage/repetition under matched update/exposure budgets, not every
possible learning rate, grid, fitting duration or corpus.

Both domains contain eight 129-token prediction windows and eight separate
32-token prompts, own-prefix greedy generation up to 32 new tokens. WikiText
selection is exactly the predefined token-only selection from the first attempt;
no output was evaluated then. News now uses the immutable provider train
partition **for evaluation only**, separately reconstructing the 32 previously
consumed provider test rows and excluding their token hashes across ID namespaces.
Complete permutations, accepted identities and rejections are retained and
independently reconstructed. All current outcomes are now consumed development
evidence. No claim of pretraining absence or project-wide untouched holdout.

## Separate domain outcomes

Argmax counts are out of 1,024; generation positions out of 256, with unmatched
tails counted as mismatches. Every arm has **0/8 exact continuations in both
domains**. KL and NLL delta are nats/token. All arms fail each absolute quality
gate: >=99% argmax, KL<=.01, NLL delta<=.01, >=95% generation positions, >=7/8
exact traces. Storage and fitting gates pass for all arms.

| Domain | Arm | Argmax count (%) | KL | NLL delta | Generation positions |
| --- | --- | ---: | ---: | ---: | ---: |
| Wiki | D | 4 (0.390625) | 9.045724 | +9.044168 | 0 |
| Wiki | N_ONE | 215 (20.996094) | 3.387028 | +3.169539 | 7 |
| Wiki | N_STAGED | 262 (25.585938) | 3.209164 | +3.013283 | 11 |
| Wiki | B_ONE | 270 (26.367188) | 2.940161 | +2.698603 | 15 |
| Wiki | B_STAGED | 291 (28.417969) | 2.696645 | +2.472594 | 12 |
| News | D | 0 (0) | 11.939169 | +12.055810 | 0 |
| News | N_ONE | 32 (3.125000) | 7.977051 | +7.976129 | 2 |
| News | N_STAGED | 45 (4.394531) | 7.733293 | +7.810742 | 3 |
| News | B_ONE | 64 (6.250000) | 7.450929 | +7.462196 | 6 |
| News | B_STAGED | 59 (5.761719) | 6.937140 | +6.949290 | 1 |

| Relative comparison | Wiki KL reduction | Wiki signal | News KL reduction | News signal / failed clauses |
| --- | ---: | --- | ---: | --- |
| B_ONE vs N_ONE | 13.193478% | Pass | 6.595442% | Fail: below >=10% KL |
| B_STAGED vs N_STAGED | 15.970472% | Pass | 10.295134% | Fail: generation positions 1 vs 3 |
| B_STAGED vs B_ONE | 8.282404% | Fail: KL and generation | 6.895639% | Fail: KL, argmax and generation |

Every relative comparison requires all its clauses separately in both domains;
there is no pooled rescue or threshold relaxation. More news argmax matches in
B_ONE do not compensate for failed KL, generation or absolute tolerances.
The same-domain benefits of wider coverage are measured. Different domain
outcomes are descriptive; this test does not isolate a universal domain cause,
prove representation impossibility, or justify unbounded fitting extensions.

## Final fitted-set diagnostics and capacity

| Arm | Fitted positions | Argmax (%) | KL | NLL delta |
| --- | ---: | ---: | ---: | ---: |
| D | 32,768 | 0.149536 | 9.324100 | +9.231501 |
| N_ONE | 2,048 | 69.531250 | .529333 | +.281006 |
| N_STAGED | 2,048 | 68.994141 | .480911 | +.106974 |
| B_ONE | 32,768 | 26.504517 | 2.898676 | +2.554108 |
| B_STAGED | 32,768 | 28.610229 | 2.563065 | +2.286347 |

Fitted-set metrics are descriptive and cannot be compared as identical-window
scores between N and B. B sees each window once versus repeated N windows;
neither has converged preservation on its own fitted set. Broader corpus
coverage alone under this fixed update schedule does not solve preservation.

All 169 unique matrices / 493,961,216 coefficients are finally hard ternary;
71,552 original F32 vector coefficients are counted. Unique original parameters
494,032,768; FP16 reference 988,065,536 bytes. Every arm deploys exactly
154,649,088 raw parameter bytes. Complete archives range 154,878,076–154,887,561
bytes (15.674879–15.675839% FP16), including descriptors/NPY headers/ZIP overhead.
No latent matrices or optimizer state deployed. Hard code changes vs D:
N_ONE 55,604,260; N_STAGED 56,742,841; B_ONE 63,715,211; B_STAGED 68,679,455.
GPU reconstructed F32 execution is not a native packed-memory or latency result.

## Independent verification, costs and exact retention

Separate process imports no fitting or repair-selector module. It reconstructs
all 256 exact teacher-target NPY identities, original parameters/tokenizer/
providers/runtime, calibration selections/arrays, actual four-arm counters and
step/target/stage histories. All final exported fitted-set states/F32-derived
points, all 12,288 new-domain prediction states, 96 own-prefix generation traces
and 3,033 actual pre-choice states/choices reproduce exactly. Original
wrapper/export calibration files are byte equal; no optimization trajectory
replay is claimed. All new-domain readout F64 diagnostics have maximum relative
RMS 5.776319e-7, no argmax differences. Calibration F64 is explicitly sampled
at 128 fixed positions per arm; maximum sampled RMS 7.675404e-7, no sampled
argmax differences, not an all-position F64 bound.

Parent fit/export 1809.718723 s is charged once, within its 7200 s gate. Arm
times: N_ONE 391.642968, N_STAGED 354.007489, B_ONE 550.227535, B_STAGED 512.009443 s;
each includes ~89 s target generation. Parent setup records grid/D export
8.914536 s and all 256 target generation 88.098473 s within 165.385795 s pre-fit.
Original failed experiment process 2038.600950 s and install 16.412078 s remain
charged; do not add nested arm/setup times to that total again.

Repair install 21.076297 s, evaluation process 368.134908 s, audit 715.311804 s,
all within declared bounds and all returncodes 0. Exact input extraction/view
preparation 21.728902 s is included in evaluation, not additional GPU fitting.
Evaluation CUDA allocation peaks 3,109,333,504 / 0 bytes, final RSS 4,504,014,848.
These are evaluation-process telemetry: audit GPU/RSS peaks and the failed
parent's final memory peaks were not recorded. Do not infer them or claim a
complete whole-process memory maximum. Local package/network operations and
timestamps remain in preparation/upload/fetch records; no isolated native CPU
timing was performed. Missing phase memory telemetry is a reporting limitation,
not numerical preservation evidence.

COMPLETE was observed 02:01:34 UTC (poll observation, not exact finish event).
Single retrieval retains **212 files / 3,110,946,354 bytes**, all SHA/size exact;
17 actual repair embedded sources match frozen Git bytes. **128 small raw files /
36,510,525 bytes** copied byte exactly to `pqt_009_r1_evidence/`. Large original
and repair artifacts remain in `results/progressive_ternary/PQT-009-R1/remote_001/`.
No repeated dispatch/retrieval. Source/retention, standalone adjudication and
raw Git verification are separate records. Commits unsigned; no merge/push.
At `87e7eef`, all 128 small raw files / 36,510,525 bytes match Git, working
bytes and immutable fetch exactly; PQT_009_R1_GIT_VERIFICATION_R1.json.
An initial verifier was dispatched before the evidence commit completed and
read the previous HEAD. Its first missing-path failure is preserved in
PQT_009_R1_GIT_VERIFICATION_FAILURE_001.json; only the metadata verification
was repeated after the commit completed. No scientific or remote rerun.

## Research consequence and resumption

These are eight prediction windows and eight prompts per domain at one fixed
seed, not a broad capability suite or replicated estimate of generalization.
The fixed original-grid, 256-update pure-ternary full-model procedure is not
qualified for preservation or useful native deployment. Wider calibration helps
Wiki transfer and some news metrics, but leaves failures far from the thresholds.
This is a bounded negative for the tested budget and grid, not every progressive
method. No causal shared-embedding, attention or individual-expert attribution.
Original expert-function negatives remain independently applicable.

Next separate structural sensitivity from an arbitrary fitting extension.
A prospective consumed-development ablation can restore only the original
shared embedding/readout of the frozen broad payload, and compare an explicitly
counted higher-precision shared parameter. Float restoration is a diagnostic
with excess capacity, never a ternary/promotion claim; any low-bit rescue is a
mixed representation, subject to complete storage accounting and later fresh
validation. Freeze its source/roles/criteria before any new output access.
This does not solve original expert-function preservation by itself. Native
CPU timing continues to await explicit owner coordination. Keep the full goal
active and preserve all newly consumed identities in subsequent protocols.
