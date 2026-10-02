# METH-268: frozen consumed-prefix core/route replay diagnosis

## Uncertainty and decision

Same259 archive passes263 prediction,265 health/K64 and266 PIQA but fails
267 strict semantic preservation. Native promotion stays closed. Before
a changed core or route/function recipe, test whether source-core numerical
drift alters the existing learned router, and how much reference-route
replay removes next-choice/logit error. No fit, new artifact, source change,
quality reclassification or newly accepted capacity here.

Bind raw265 SHA f3581cefc4a7d758b141fc4b4fa62ca45f43d00fe6cc00a10af57895b4a5ec93,
266 SHA da571182fd1190c5d3cf634c634d63810a1f33c3f302a6848f13085202c0385e,
267 SHA bbcc22e31d599f1a6f4eb0212e4836f804161603504699053747cd0bf0f2d790,
and all original261/259/source/checkpoint/export/helper/tokenizer identities.
This protocol is after those observations, before any268 prefix scoring.

## Fixed consumed sample and intervention

Use every one of24 sources and both fixed BF16 E1280/saved265 trajectories.
Steps0,1,2,4,8,16,32,64,96,127 plus first differing generated ID-1/0/+1,
deduplicated within each trajectory,valid steps only. No error-row cherry
picking or source exclusion. First-divergence neighborhoods derive from
already consumed IDs; all results are diagnostic, not fresh evaluation.

For each same full prefix, run original BF16 E1280 and complete259, no cache,
capturing actual full-position parentIDs/scores/childIDs and last MLP input
states. Require each trajectory's own original265 next choice to match
exactly under instrumentation. Stop on mismatch/finite/resource failure;
do not widen this apparatus guard after observation.

On each reference last-input state, independently require original/saved
single-state routes and selected BF16 A/B values to match exactly. Measure
single-state source/candidate FFN relative squared error and ordinary
cross-model hidden-input drift/actual selected routes. Single-state FFN
measurement is separate from full-prefix BF16 operator arithmetic.

Then replay reference parentIDs/scores/childIDs at *every position/layer*
through the unchanged candidate. Conditional inputs and compact core remain
candidate; reference routing and softmax-score inputs are oracle injections.
Verify all injected values exact. Measure recovered/lost reference choices,
logit error and reference-choice deficit. This is nondeployable: it requires
another intact model. It isolates a routing/gating contribution at these
prefixes, not full free-generation semantic quality or a deployable remedy.

Report all categories/cases/layers, ordinary and replay choice disagreement,
any parent/child drift, FFN/input/logit errors. No prospective improvement
threshold: measurement diagnoses mechanism and cannot promote this closed
candidate. If same-input identity fails, resolve that implementation issue
before designing a new approximation. If oracle route replay helps materially,
investigate route/core robustness jointly; if it does not, prioritize source
representation/generative error. Neither result proves large-n useful capacity.

## Cost and stop

RTX3060/six threads,deterministic/highest/TF32off,20min after imports,20GiB
RSS/10.5GiB GPU. Both source and saved model resident for tracing. No T4,
downloads,CPU speed measurement,optimization or concurrent model job.
Persist per24-case progress/failure. Full goal still requires useful
RAM-scale n,CPU LUT/router/DRAM cost,same-artifact quality/accepted>=50tok/s
and multiple real donor families/scales.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth268_generation_route_replay.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth268_generation_route_replay_result.json
```
