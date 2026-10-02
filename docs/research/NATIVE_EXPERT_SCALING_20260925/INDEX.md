# Native expert scaling: research control index

**2 October 2026. Branch:** `research/native-expert-scaling`.
**Active research:** no artifact meets all final requirements.

## Goal and constraints

Transfer pretrained capacity into a compact reusable core plus selectively
consulted conditional functions in `benchmarks/phase60/engine.c`, preserve
useful donor-relative held-out/generation/task quality and measure >=50
accepted batch1tokens/s on the same artifact. Establish transfer across
families/scales, including real approximately10B and100B when resources
permit. User priority: useful n grows with RAM without proportional active
cost or quality loss from more complex routing.

Count distinct functions, route labels, active/stored bytes and actual
route/LUT/DRAM cost separately. Synthetic/copied capacity and component
rates cannot establish the goal. Freeze decisions before observation.
Local RTX3060/six threads; T4 needs prior reason/budget/stop communication.

## State of the two research questions

| Question | Established | Still missing |
| --- | --- | --- |
| Useful large-n target | Small centered BF16 E1280 retains scoped quality;259 stores30,556 unique functions with1280 labels/layer | Useful tenfold new capacity; parent-selection growth, real large-RAM route/LUT/DRAM cost and independent quality |
| Pretrained-to-compact transfer | Actual1.321GB complete259 archive; all725 fields/loader checks;260 development,263 independent prediction and265 generation-health pass | Task/semantic quality, actual native whole-model arithmetic and >=50 acceptedtok/s; sparse/larger-donor variants |

Current dense source still executes all4864 FFN features/layer. This small
case does not establish constant-active-cost conversion for10B/100B.
GigaChat assets/fidelity are reusable; its scoped-quality Q4 baseline costs
1016MB active/token. The frozen generic donor port remains paused.

## Latest decisive evidence and exact resume

- [METH-259](METH_259_UNIQUE_BANK_CORE_RESULT_20261002.md): one saved
  complete archive,1243..1280 unique functions/layer,164 preexisting aliases;
  all6144 component controls and complete archive/readback/loader pass.
- [METH-261/262](METH_262_SOURCE_ANSWERABILITY_RESULT_20261002.md):24
  newly selected project sources,3176 exclusions, source-only answerability
  fixed before inference. Different PG19 rows, not an independent corpus.
- [METH-263](METH_263_COMPLETE_CORE_FRESH_PREDICTION_RESULT_20261002.md):
  independent prediction passes; BPB-.000110010 versus BF16 E1280,
  -.005313449 versus donor; top1+.446429 percentage points versus E1280.
- [METH-264](METH_264_CACHED_REFERENCE_STOP_20261002.md): original donor
  differs on2/48 cached/full choices, before candidate continuation. Keep
  zero-mismatch cache guard and failed cached recipe unchanged.
- [METH-265](METH_265_FULL_PREFIX_GENERATION_RESULT_20261002.md): all72
  full-prefix continuations complete; health/K64 gates pass. Candidate
  EOS22/24,repetition1/24. Session43037 exited0,531.656s. This reference
  recomputes prefixes; production cache/CPU quality/rate remain unproven.

[METH-266 full PIQA](METH_266_COMPLETE_CORE_PIQA_RESULT_20261002.md) passes
all regression gates: candidate1290/1838 versus1291 for both controls,
-.054407percentage points. Session64975 exited0,1035.656s; no active job.

[METH-267 anonymous review](METH_267_ANONYMOUS_SEMANTIC_RESULT_20261002.md)
fails strict semantic gates after all72 findings were committed before
unblinding: unsupported38 versus39/41,severe16 versus15/14,missing-detail2
versus2/0 (donor/E1280). Fixed259 native promotion is closed;266 task pass
cannot override267. [METH-268](METH_268_GENERATION_ROUTE_REPLAY_RESULT_20261002.md)
passes all471 consumed-prefix apparatus guards. Nondeployable source-route/
score replay recovers15/40 choice differences,loses0;25 remain. Mean logit
error rises4.3858%,so top1 recovery is not overall fidelity. Same-input
routes/dictionaries exact. Initial shadowing stop preserved; mechanical
repair only. Session14458 exits0,375.921s; no active job.
[METH-269](METH_269_FFN_BF16_NODE_RESULT_20261002.md) completes all6144
same-state arithmetic controls. BF16-node-only mean error worsens2.22%;
oracle source features improve it80.40%. No node-only promotion or new
artifact. Session73517 exits0,8.438s; no active job;259 stays closed.
**Next exact action:** implement then freeze [METH-270](METH_270_PRIVATE_SOURCE_ROWS_PROTOCOL_20261002.md),
private source-input rows32->128,same source-only selection/readout/bank.
No270 code/result/fixture yet. Fixed component-error gates and <=10ms
native prerequisite before new full archive/fresh quality. This increases
source precision,not learned n; all final expert-count requirements remain.

After quality pass, load the same725 archived tensors into the native whole
model, qualify arithmetic/alias lookup/LUT/real DRAM, then accepted rate.
253 FFN median9.318ms is a source component result. No CPU speed benchmark
overlaps a model job. Useful large-RAM n and multi-family/10B/100B remain due.

## Open hypotheses and closed recipes

Useful additional capacity needs route/function coupling that generalizes;
balance and parameter hashes alone cannot prove it. Hierarchical routing
bounds consulted children, but larger parent count and useful trained n
remain unqualified. Do not retry unchanged hard carve, global weight-rank
repair, weak child B training, static-bias calibration, affine/intercept or
bounded source-amplitude recipes. Scoped failures remain in the history.

## Procedure, evidence and workspace

- [METHOD](METHOD.md): reproducible procedure, prerequisites and gaps.
- [PRIOR_EVIDENCE](PRIOR_EVIDENCE.md): initial native inventory.
- [History through216](HISTORY_THROUGH_METH216_20261001.md) and
  [217-264](HISTORY_METH217_THROUGH264_20261002.md): detailed experiment links;
  historical next actions are superseded.
- [Previous method history](METHOD_HISTORY_THROUGH264_20261002.md): retained
  donor/variant evidence and former decision diary.

Preserve unrelated tracked edits in `docs/research/RESEARCH_INDEX.md` and
`benchmarks/donor_adaptation/density/build_document_holdout.py`; stage exact
paths. Docs English/user updates Italian. Graphify optional for explicit
graph work only; routine hooks removed with backups, Git LFS retained.
