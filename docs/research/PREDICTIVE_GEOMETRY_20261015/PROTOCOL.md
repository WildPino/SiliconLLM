# Predictive geometry, persistence and reuse

15 October 2026. Three explicitly authorized, bounded Kaggle experiments.
This protocol precedes their launches. It does not admit a model or a speedup.

## Scientific motivation

The preceding [context-program pilot](../CONTEXT_PROGRAM_PILOT_20261015/RESULT.md)
was exploratory negative. Its conditional arm had worse hidden reconstruction
but lower token NLL than the matched fixed arm; a prefix-additive control was
nearly as good in NLL and better in donor-relative KL. Its frozen negative gate
and held-out observations remain unchanged. More of the same fitting was not
selected. The present campaign tests the representation hypothesis underlying
that discrepancy with new articles and different measurements.

For source logits z(h), p=softmax(z), and a small intervention d in an internal
state, local donor-relative KL has quadratic form

`KL(p(h) || p(h+d)) = 0.5 d^T G(h) d + higher-order terms`,

where `G = J^T (diag(p) - p p^T) J` and J is the logit Jacobian. If v is sampled
from the full categorical p, the score gradient `g=grad_h log p(v)` satisfies
`E[g g^T]=G`. We estimate this uncentered second moment with independently sampled
categorical scores. No vocabulary truncation, confidence-based token filtering,
or resampling until enough unique tokens is allowed.

The speculative larger idea is to identify states with similar useful future
behavior and execute a compact dynamics over those equivalence classes. These
experiments do not construct that dynamics. They only measure local sensitivity,
its persistence, and its transfer across documents. Low sampled rank, local
invariance, or a larger archive alone cannot prove retained large-model capacity.

## Resource and operational scope

The owner explicitly authorized implementing and launching all three tests,
including automatic queuing if necessary. Available accounts are acct1/wildpino,
acct2/giggio253, acct3/sirwildpino. Each identity and its GPU quota must be checked
before launch; secrets remain in the existing external credential store.

Initial allocation: one new private script per account, with T4 acceleration,
at most 7,200 seconds per kernel. Count both allocated T4 devices conservatively:
4 T4-hours per arm, 12 T4-hours maximum for the three-arm wave. GPU 0 performs
the work; GPU 1 is recorded as unused. This is a cap, not an expected duration.
The owner's 80-hour envelope is treated conservatively as GPU-hours and is not
automatically spent. No successor, sweep or repair launch bypasses the attempt
ledger. An interrupted or rejected attempt remains a recorded allocation.

Bootstrap subprocess deadline: 6,900 seconds including dependency installation.
Numerical process cap: normally 6,000 seconds, shortened if setup consumed its
reserve. Record partial progress and an explicit fault on exhaustion. No model
calculations, tensor self-tests, or source-weight acquisition run locally.

The three tests are independent and may run in parallel. Kaggle's native batch
queue can hold a pushed job; observe queued/running state instead of pushing a
duplicate. Each job is below the documented 12-hour CPU/GPU session limit, so
no GitHub cron or cross-session relay is enabled. Do not reactivate historical
training workflows. Sources:
[Kaggle notebooks](https://www.kaggle.com/docs/notebooks) and
[efficient GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage).

## Fixed source, fresh documents and intervention

- Donor: `Qwen/Qwen2.5-0.5B-Instruct`, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Natural text: `Salesforce/wikitext`, configuration `wikitext-2-raw-v1`, revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`.
- Transformers `4.51.3`, Datasets `3.5.0`; record all effective dependencies.
- Frozen evaluation-mode FP32 donor, no weight gradients or optimizer.
- Seed 2441, 64 training articles, 12 validation articles and 12 test articles.
- Preserve dataset partitions; one deterministic 192-token input window per
  article plus the next target ID in the stored window. Exclude all 176 article
  hashes from the previous pilot, not merely its test articles. Reject duplicate
  article hashes between newly selected partitions. No synthetic fallback.
- Capture source entry-to-layer-8 inputs x and entry-to-layer-12 outputs y,
  with zero-based layer indices. Record selected article and token hashes.
- Differentiate an additive vector at one y position, initially index 63.
  The helper replaces that position's activation with `y + delta` and preserves
  the autograd connection. It removes the hook after every forward.
- Query positions 63, 67 and 79 measure the current and two fixed future-token
  distributions. Future inputs are the actual teacher-forced text. This is a
  single-position intervention, not a perturbation of the model's entire causal
  state, a free-running continuation test, or an all-futures equivalence test.

All three configurations and decision patterns are fixed before outputs exist.
Sharing the fresh cohort across this preplanned wave does not make the tests
independent replications. Confidence intervals are exploratory and unadjusted
for the multiple directions, ranks and contrasts.

## Experiment 1: geometry

Account acct1. Collect two independent categorical score gradients for each of
three query positions in each of 64 FIT articles: 384 FIT scores. Build a joint
uncentered empirical-Fisher basis from FIT only. Compare ranks 8, 32 and 128
against activation PCA fitted on FIT source-output states and seeded random
orthonormal bases. Unsupported numerical ranks are recorded, never invented.

Collect four fresh scores per query/article on each held-out partition: 144
held-out scores per partition. Measure projected score energy relative to total
energy, separately by query, with article-level paired bootstrap intervals.
FIT singular values alone are insufficient: sample count limits FIT rank, and
concentration must survive independent held-out scores.

For rank 32, apply signed equal-L2 perturbations along random, retained and
orthogonal-complement directions at norm ratios 0.005 and 0.02 of the source
activation norm. Compute exact full-vocabulary KL for the queried positions and
report the separate independent-score quadratic estimate. Small Monte Carlo
sample counts can produce noisy quadratic estimates; these are not certificates.
Two FIT directional finite-difference checks are mandatory; a numerical failure
prevents scientific admission. Preserve sampled token IDs, gradients, bases,
singular values, perturbation measurements and the derivative checks.

Prospective exploratory support pattern: at query 79, rank 32 captures at least
half the held-out score energy, paired Fisher-minus-random and Fisher-minus-PCA
intervals have positive lower endpoints, and equal-norm complement perturbations
produce at most one quarter the retained-direction KL at norm ratio 0.02.
Seal the validation decision before sampling test scores. The reserved test
confirms the frozen recipe and never changes rank or thresholds.

## Experiment 2: persistence

Account acct2. Use the first eight fresh articles of each partition, anchors
63/79/95, and query offsets 0/16. At each anchor/query build a rank-eight frame
from 16 categorical score samples and evaluate it using eight independent draws.
Compare a frame constructed at anchor 63 with current, seeded random and
cross-document training frames in the same donor coordinates. Report principal
cosines, projected energy, per-article contrasts, and paired bootstrap intervals.

The future-offset frame is an offline oracle: its construction sees the fixed
teacher-forced continuation to its query. It is not a causally available program
at the earlier anchor. At offset zero, construction does not inspect future text.
Sample-rank-deficient frames retain an orthonormal completion for matched storage,
but their principal angles have no identified semantic meaning; flag these rows
as unqualified. Do not draw a persistence conclusion from null-space completions.

A useful predeclared signal is that, on rank-qualified held-out rows, the old
offset-zero frame at anchor 79 captures at least 80% of the current frame's
energy and has positive paired advantage over both random and cross-document
frames. Anchor 95 and offset 16 are secondary diagnostics. If too few frames
are rank-qualified, report insufficient evidence rather than a positive result.
Save the exact derivative evidence and sampled IDs so no GPU rerun is required
merely to inspect a negative or rank-limited finding.

## Experiment 3: cross-context reuse

Account acct3. Build FIT-only rank-eight frames per article for queries 63 and
79, using 16 categorical scores per query/article. Construct nested archives
from the first 16 and all 64 FIT articles. Routing keys use only the normalized
mean of source-entry-to-layer-8 states at positions 31 through 62. They never
use source span outputs, derivative targets, the anchor token or future text.

Use cosine nearest-key selection and eight independent held-out score probes per
query/article. Compare selected local frames with matched-rank global FIT frames,
random frames and random local retrieval. A 64-slot archive made by duplicating
the first 16 entries must reproduce the 16-entry selection and projection energy.
Report actual effective ranks, selected IDs, keys, evidence and intervals.

A target-aware best-entry oracle may be reported on DEV only, explicitly as a
privileged finite-probe upper reference. TEST gradients evaluate frozen causal
selections and never select an entry. Full key scans are charged and explicitly
do not demonstrate bounded addressing at large scale.

A useful support pattern is a positive held-out paired advantage of nearest-64
over global-64 and random-local-64, together with a positive 64-minus-16 gain
at future query 79 and the duplicate control passing. If global frames dominate,
that may instead indicate common predictive structure; it is not evidence for
useful conditional capacity growth. Causal routing still uses expensive donor
entry states in this diagnostic, so no cheap online selector is established.

## Custody, completion and limits

Source-only private uploads include all modules, frozen config, article-exclusion
list, unchanged common acquisition utilities and this protocol. Hash both payload
and declared source files. Keep launch attempts, API responses, source readbacks
and downloaded results under
`E:/SiliconLLM_Artifacts/campaigns/predictive_geometry_20261015`, outside the
in-flight storage relocation. Never copy account credentials into a package.

Cloud tensor smoke tests precede donor acquisition. Numerical faults, nonfinite
values, hardware/revision mismatches and time exhaustion are explicit failures,
not weak scientific results. A COMPLETE kernel means only that execution ended;
inspect bootstrap, numerical status and each experiment's scope before reporting.

These are probes of local response geometry. Even a positive wave would leave
unproved: compact sufficient state, economical state updates, causally available
program construction, useful multi-step generation, retained large-donor ability,
physical DRAM behavior and the same-artifact 50 useful CPU tokens/s target.
