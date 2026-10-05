# METH476: complete original context admitted

5 October2026. Branch research/native-expert-scaling. Goal ACTIVE/INCOMPLETE.
Current goal turn PROGRESS. All6 apparatus/all6 independent audit gates PASS.
No first numerical/compiler/native/audit fault, no repair or completed rerun.

## What was missing and what is now established

469/471 whole captures stored decoder layer outputs AFTER each FFN, not the
residual `pre` immediately before lastbank11. Router `norm(pre)` alone loses
amplitude/rounding information; subtraction from post does not exactly invert
rounded F32 multiplication/addition. That cached schema gap is retained here.

New476 reconstructs only last decoderlayer11 from cached previous-layer outputs
at ALL original positions and final encoder states. Those are sufficient for
lastlayer selfK/V/crossK/V; all original causal prefixes remain. Original native
norm/mv/attend/quant/head operations and original F32 residual order are used.
No whole source encoder/first11decoder capture, old controller or new generation.

The reconstructed pre is BYTE exact on ALL1344 cases where it was already
captured. On ALL6649 validation472 queries, the complete forward path matches
original FFN input/router128scores/winner/normalization probability/full FFN/
post/finalnorm/ALL32128logits BYTE exactly. No missing or narrowed cohort.

## Domain and controls

| Quantity | Actual coverage |
| --- | ---: |
| Known truepre goldens |1344positions/96teacher captures/24books0..23|
| Validation472 teacher |3584positions|
| Validation472 original natural |3065positions|
| Validation books/cases/trajectories |64books64..127/256cases/512trajectories|
| Total reconstruction batches/positions |608/7993|
| ALL original head logits compared and independently rederived |256,799,104|
| Original I8/A16 integer fixtures |13 BYTE exact|

Fixed original128 experts, bank11,D768/F3072/V32128/top1. All validation role1
queries equal the full saved472 query selection/order; ledger/book/case/mode/
position/input/code/alpha/probability/SHA32 joins exact. Fixed107factor/21fallback
policy of472 unchanged; no expert filtering or candidate quality comparison.
Source input IDs and original own-natural prefixes bound, embedding snapshots
match IDs. Native actual3workers physical logical0,2,4; masks1,4,16 verified.

Independent Python/NumPy audit imports no476main/C/model. Re-derives all7993
pre->FFNnorm/residual/source-down/finalnorm/head input/A16 codes/alpha and every
original head logit BYTE exactly. Scalar F32 coordinate squares, sequentialF64
accumulation, original scale/cast order, exact I8/A16 dot via F64 BLAS with
integer partial sums<2^53. Native last-layer attention reconstruction and full
WI/WO comparisons remain source-recorded byte-bound controls. Audit independently
checks knowntruepre/fixture13, all6649 ledger/digests/roles/order and stored output
rows/EOF/counts, preserved engine/unrelatedfiles, terminal/resources/runtime.

## Actual executions, resources and retention

Scientific freeze70c6a8f; sole first-command registration8d5e124; independent
auditor frozen4ca1536 before its first execution. ONE main session60373exit0,
ONE compiler and ONE native process, ONE audit session71464exit0. MainPID23900
creation1791212219.6388192; nativePID9484creation1791212256.003517; auditPID23504
creation1791212536.5633547. No model/scientific job overlap.

| Resource | Actual | Prospective limit |
| --- | ---: | ---: |
| Main before raw write |48.391s|600s|
| Byte admission |30.484s|150s|
| Compiler wall |3.656s|120s|
| Partial native wall |11.953s|420s|
| Conservative parent+child OSpeak |1,018,068,992B|2GiB|
| Native OSpeak |818,061,312B|included above|
| Audit terminal |60.531s|600s|
| Audit OSpeak |472,940,544B|2GiB|
| Context binary |36,991,620B|all7993rows*4628B+16B|
| New scientific OUT+RAW |38,990,991B|64MiB; upper47,690,884B|

Native wall includes partial reconstruction, cached I/O and diagnostics; it is
NOT accepted-token speed or whole-model performance. WindowsEvent1000 query
available, zero events/matches, literal window14:56:57.8106320..15:02:02.8220059UTC,
parent/compiler descendants/native PID+creation identities recorded.

Bound6567 prior files7,702,062,531B plus192golden whole/trace files and metadata
body; original full payload7,541,946,880B/manifest3320, actual runtime assets,
compiler/OpenMP, unchanged393 C/headers, original engine/unrelated3 preserved.
11 new output files38,976,650B; next retained inventory would be6578 files
7,741,039,181B before separately bound golden files/records. Existing native
traces and472functions remain immutable. Original engine is unchanged.

## Immutable identities

- [Protocol](METH_476_SWITCH_LAST_LAYER_CONTEXT_PROTOCOL_20261005.md),
  SHA14398f0aa2edb9ae95d73d9592bb8ebf9c4f40e56924ff9964adbd07b56c4462.
- [Binding](meth476_prospective_bindings.json),
  SHA2d3d5b225a0126c293ed716c99cc33125e36f8c7e64b628bf9b9c02db4991dc2.
- [Raw](meth476_switch_last_layer_context_result.json),
  SHA1f5402b1cf059458ad095c113070c6a5c99a8a6ccf29f280cdfb4c718dd0230f.
- [Retention](RETENTION_476_20261005.json),
  SHA3f4730698e4bbe3c44b96785379daf7d8bec18b2f2114e933ffcd466dbd9c508.
- Controller SHAb4b42bf8eefac0204ae7ae60f9fbb7ca64f061b7d44b9fde656a3b51ff8148e5;
  C SHAd298f6bcef111d6bef7f4b4d49b4104fb9a5108237ab29df68066f476e7e49f3.
- Auditor SHA7a6666f2590db71586fbba3c4c2c331ef41db86c1f1293ee87e420cebd36116a;
  Windows metadata SHA8eebf1232e8f2e0533b98d01da10431296da40d954148be1ede1deb725afce1a.

## Reassessment: the next problem is an information metric

For a fixed original prefix let f be originalFFN, f' saved472candidateFFN,
a the ORIGINAL route probability, r=pre+a f, and r'=pre+a f'. Let l/l' be
the actual finalnorm/rescale/A16 head logits; delta=l'-l and p=softmax(l).

    KL(p || softmax(l')) = log(sum_j p_j exp(delta_j)) - sum_j p_j delta_j.

This identity is invariant to delta+c*1. It measures prediction-distribution
change after the full downstream computation; per-expert latent RMS is a
different metric. For exact real softmax:

    KL <= (max(delta)-min(delta))^2 / 8.

For a UNIQUE source argmax, margin>range(delta) is a sufficient stability bound.
These real-analysis statements are not formal F32 certificates or statistical
confidence bounds. Stable argmax alone does not preserve the distribution.

The smooth normalization geometry also explains why Euclidean FFN error cannot
be read directly as semantic loss. Ignoring rounding and activation quantization,
let W=diag(finalnorm weights), s=(||r||²/D+epsilon)^(-1/2). Then:

    J_norm = s W [I - rr^T/(||r||² + D epsilon)].

At epsilon0 radial errors lie in its null direction; for positiveepsilon that
direction is attenuated, not exactly null. The small-delta information metric is
one half delta^T[diag(p)-pp^T]delta. Combined with route probability, normalization
and head, this gives a local geometry for useful retained information. The actual
A16 quantizer has thresholds; a smooth derivative is only an explanatory/local
approximation. Next evidence MUST compute the finite actual head outputs.

### Next NEW477, not executed or frozen

One descriptive observable diagnostic on ALL6649 admitted original prefixes,
using EXACT saved472candidate/source functions and admitted476pre. No refit,
changed rank/metric grid, expert removal, old472 rerun or candidate promotion.
Freeze source/protocol/runtime/input/budget before any candidate forward.

1. Qualify exact source residual/norm/head primal before candidate statistics.
2. Compute ALL32128candidate logits, stable full-vocabulary KL, source margins,
   delta range, actual argmax changes and distribution shift for every query.
3. Predetermine uniform-query, everybook, teacher/natural, ready/fallback,
   rare/dominant-ID and novel-coordinate summaries; retain complete denominators.
   Energy-weighted pooling cannot substitute for those views.
4. True-label deltaNLL only after exact label provenance admission; source argmax
   is not a ground-truth label. No label inference from source candidate outputs.
5. Independent complete head/metric audit, strict finite/rounding/softmax controls,
   immutable first faults. A cheap diagnostic must retain per-query witnesses,
   not duplicate854MB logits or narrow the6649 queries to fit a budget.

No acceptance threshold will be chosen after observing this diagnostic. It
calibrates existing local constraints; any new representation/criterion requires
a separately frozen prospective inquiry and fresh whole quality later. Old472
FAIL105/107/book107 and closed473/474/475 remain unchanged.

## Whole-goal implications and open problems

This turn removes a real context blocker and qualifies a downstream observable
path. It does not establish that472candidate retains predictions, that a new
geometry preserves pretrained knowledge, or that the goal is attained.

Foreground n evidence stays369..373: matching realfunctions useful; real nested
64/128/256banks exist,372qualitymixed;373fourfold fullcost+2.91%/upper+3.62% while
router~3.83x. No monotonic useful-n/physicalDRAM/winner-and-mass CPU LUT proof.
More stored pretrained parameters need measured causal usefulness, not just IDs.

After observable calibration: viable full/all-bank representation -> changed
own-state/fresh donor-relative prediction/generation/tasks -> accepted>=50batch1
on SAME realartifact -> physicalDRAM/large-n winner AND normalization mass ->
actual additional family/scale/~100B as resources permit. These remain joint
requirements. Existing original-source rates cannot be inherited by a candidate.
