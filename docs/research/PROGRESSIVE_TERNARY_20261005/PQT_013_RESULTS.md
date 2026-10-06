# PQT-013: matched row alphabets on original natural expert contexts

6 October 2026. **Qualified negative result: no arm advances.** All eight arms
meet complete-file storage limits; every arm fails the joint preservation
requirements. Independent packed-function and metric audits pass. This closes
the fixed row-alphabet/special-token hypothesis, without ruling out other
representations, training budgets or model architectures.

## Evidence and exact scope

[Preregistration](PQT_013_PREREGISTRATION.md),
[frozen binding](PQT_013_REMOTE_BINDING.json),
[source preflight](PQT_013_BUNDLE_VERIFICATION_001.json),
[raw retention002](PQT_013_REMOTE_RETENTION_002.json),
[actual committed-byte proof002](PQT_013_REMOTE_GIT_002.json),
[separate gate adjudication](PQT_013_ADJUDICATION_001.json).

Scientific source `e218e3e36d44ab4a34de091f5ccad05c0a9267db`; bundle `1afbb93a`.
The 20 actual source/protocol files equal their frozen Git bytes. Eight reused
scientific helpers match the separately passing CPU controls; packet transport
matches its qualified source. The independent numerical auditor checks scalar
qualification, actual large-matrix operation counts, endpoints and curvature;
it does not independently reimplement the complete large-matrix algorithm.

Donor `google/switch-base-128`, revision
`86c815ec05361a33a8b49fc717277da9c0a4e711`; original encoder layer11 experts
81,87,18,91, WI3072×768 and WO768×3072. Original routing/capacity64 and the
qualified natural encoder contexts remain unchanged. Calibration/development/
held-out document splits were disjoint when acquired, but these diagnostic
roles were already observed in preceding experiments. They are consumed
research evidence, not a fresh final confirmation set.

The original 117,362,605-byte input capsule and 12,071,742-byte auxiliary
capsule match their admitted hashes. Eight parent group64 packed baselines are
reused without refitting. All 32 labelled deployment artifacts are sealed before
any numerical development/held-out access. There are 96 packed F32 evaluations,
12 original F64 function witnesses and 2,000 fixed document-cluster resamples
per metric. Every candidate must pass every expert's development and held-out
point RMS ≤1%, bootstrap97.5th percentile ≤1%, and complete deployment bytes
≤35% of the original FP16 FFN pair (9,437,184 bytes).

Row SYM uses one fixed original-row maximum absolute magnitude; ASYM stores
two fixed positive magnitudes defining `{-a,0,b}`. PR applies scalar-qualified
activation-aware progressive compensation, damping and block128 with no act
ordering. WI uses original calibration X; WO uses that candidate's actual
quantized WI/ReLU context. Codes, levels, headers and metadata are all counted.
No optimizer, residual path, original dense deployment or hidden float rescue.

## Quality and storage

RMS entries below are percentages, relative to the original output norm.
Ranges span the four experts, not confidence intervals. The upper column is
the worst97.5th bootstrap percentile across development and held-out metrics.
Complete storage includes file framing, scales/levels and method metadata.

| Arm | Development RMS range | Held-out RMS range | Worst evaluation upper RMS | Maximum FP16 storage fraction |
| --- | ---: | ---: | ---: | ---: |
| G64-D | 48.13–73.09% | 47.78–74.08% | 75.38% | 15.65% |
| G64-PR | 7.22–54.09% | 7.47–55.18% | 56.94% | 15.65% |
| SYM-D | 62.40–126.25% | 62.50–129.44% | 136.68% | 12.68% |
| SYM-PR | 3.16–100.08% | 3.14–101.87% | 107.28% | 12.68% |
| ASYM-D | 60.06–155.10% | 60.17–159.36% | 169.07% | 12.85% |
| ASYM-PR | 2.92–95.97% | 2.96–98.14% | 103.27% | 12.85% |
| SYM-PR-F | 9.32–100.08% | 10.12–101.87% | 107.28% | 12.68% |
| ASYM-PR-F | 6.89–95.97% | 7.48–98.14% | 103.27% | 12.85% |

All individual development/held-out point errors also exceed1%, so this is
not merely a bootstrap-borderline rejection. Progressive compensation improves
some direct counterparts substantially, while remaining insufficient. The
best individual held-out row result is ASYM-PR expert18 at2.96%, with upper
bound3.38%; the method's worst expert remains98.14%. Group64 PR has a smaller
worst-expert error, but its best held-out expert still has7.47% RMS. Neither
an average nor an individually attractive expert satisfies the joint gate.

The paired calibration-only token filter excludes65 EOS positions from
expert18's236 original rows, retaining171 rows/90 documents. Experts81/87/91
have identical selectors, so12 equivalent matrix fits are explicitly reused;
their outputs are exactly equal between corresponding filtered/unfiltered
arms. All original evaluation rows, including EOS, remain present.

For expert18, filtering worsens SYM held-out RMS from3.14% to10.12% (paired
ratio3.2223, bootstrap interval2.8516–3.6860). ASYM-PR-F achieves7.48%, better
than filtered SYM's10.12%, but worse descriptively than unfiltered ASYM's2.96%.
The preregistered ASYM-PR-F reference is SYM-PR-F; retain that distinction rather
than silently replacing its paired reference. This is evidence about this
calibration filter under unchanged natural evaluation, not a universal claim
against special-token treatment.

The audit's maximum actual packed F32/F64 relative discrepancy is
`3.0699747598214204e-7`; original context/source discrepancy is
`2.6559168692461414e-7`, both below1e-5. Thus large quality errors are measured
properties of the admitted artifacts, rather than a detected decoding or
reference-function mismatch. Error mechanisms are not uniquely identified.

## Actual operations and resources

The private acct2 kernel `giggio253/pqt-013-row-alphabet-20261006-001`,
137371362v1, is COMPLETE, observed18:08:15 UTC. Both version1 inputs were
reverified private, owned and ready inside the push action. All11 addressable
owned kernels were terminal beforehand, zero GPU reservation, conservative
29.9339780556 hours remaining; the maximum1.5-hour run plus0.5-hour margin
passed. Interactive-session absence was not inferred.

Actual runtime: Python3.13.15, Torch2.11.0+cu128, NumPy2.1.3, two physical
Tesla T4 devices, one CPU thread, F32/noAMP/TF32 disabled. Frozen image identity
is in the binding/source manifest. There are20 new progressive matrix fits,
38,400 committed columns,40 Cholesky attempts/completions,20 inverses,
16 direct matrix quantizations,10 hidden-context forwards,12 equivalent
matrix-fit reuses and zero repeated baseline fits. Optimizer updates,
whole-model forwards, SDPA calls and native timing are all zero here.

| Complete server child | Elapsed seconds | Actual direct-child peak RSS bytes | Exit |
| --- | ---: | ---: | ---: |
| Dependency install | 5.231645 | 51,507,200 | 0 |
| Worker, including final raw hashing/report | 87.278961 | 1,643,671,552 | 0 |
| Independent numerical audit | 22.744500 | 1,065,623,552 | 0 |
| Verified lossless transport | 16.430923 | 257,380,352 | 0 |

Server before-final-report elapsed131.714817s; this is not a claim about
Kaggle queue/startup time, billed quota or complete notebook teardown. The
worker's nested65.657183s omits final hashing/report overhead; use the complete
87.278961s child cost. Actual device0/1 allocated peaks407,337,472/369,754,624B,
reserved450,887,680/408,944,640B. All declared caps pass; no resource stop.
RSS is actual wait4 for each direct child, not a measured full descendant-tree
peak. Source preparation took0.026977s; preparation/controls/publication/client
records remain separately preserved, not silently merged into fitting time.

One terminal fetch took183.604937s/exit0. No second fetch or scientific replay.
Server packets retain762 scientific/source members totaling439,298,392B in
three ZIP_STORED packets. Final retention002 preserves74 fetched/client/bundle/
first-failure entries totaling439,809,295B, with exact committed bytes proved
at `3bbc58fa7f36a5d4174718a69609f3036ec1160d`. Every scientific packet member
and executed source hash is verified. The server-pulled bootstrap has77 CRLF
pairs; only explicit CRLF→LF yields the submitted source, while the pulled
raw bytes are retained unchanged.

## First failures and numbered repairs

CPU controls001 failed an extra-member negative fixture's expected gate:
the reader correctly rejected the namespace earlier than the fixture expected.
CPU002 repairs that expectation, passing11 positives/22 rejection controls.
Local controls had separately stopped before numerical imports because the
other checkout was active. These are preserved apparatus/resource outcomes,
not candidate quality failures; no original fit was repeated.

Auxiliary publication001 stopped before upload/create because the installed
SDK rejects its75-character title. The private corrected44-character title
preserves identical capsule/ref/license. Exact status queries returned403,
which was never treated as proof of absence. Prospective admission003 instead
verified the deterministic pre-upload SDK guard, its actual source identity,
unchanged input and fresh complete owned inventory before first reachable
creation. Original responses/clocks/SDK fragment remain in retained raw.

Retention001 included its own currently open output stream, capturing zero
bytes before the log completed at891 bytes. The post-retention diagnostic
`PQT_013_RETENTION_FIRST_STOP_001.json` rejects that original-log identity;
the first archive/clock remain unchanged at `c01995f0`. Retention002 excludes
only its declared current open stream, retaining the completed001 log,
diagnostic and first costs. Scientific packet identities remain unchanged.
Complete retention0014.016585s, retention0023.945663s (see exact outer clock),
Git proof0025.467573s and adjudication1.146539s; all exits0 for the corresponding
completed operations. The first diagnostic is a failure, despite its retainer
having initially returned0. Do not cite001 as the exact-original-byte proof.

## Decision

No candidate meets the required behavior/storage conjunction. Close this
matched-alphabet experiment; do not weaken gates or silently change model
capacity, contexts, endpoints or precision. This adapted screen does not
reproduce QMoE's complete-model routing, propagated contexts, quality metric
or encoding/runtime. It neither contradicts its reported results nor
establishes whole-model perplexity from an expert RMS value. Broader/native
promotion is inadmissible. The [research closure](RESEARCH_CONCLUSION.md)
integrates this result with the preceding bounded tests and literature.
