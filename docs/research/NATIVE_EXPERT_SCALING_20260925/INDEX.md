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
[METH-270](METH_270_PRIVATE_SOURCE_ROWS_RESULT_20261002.md) stops before native
timing: fixed128 coefficient-ranked rows improve mean error14.6685%, short
of15%; ratio .8533147 versus .85 gate. All unchanged/source/baseline guards
pass. Fixture337.596MB adds8.262MB;14.125s,session23631 exits0; no active job.
Do not loosen the gate or increase counts to rescue this fixed recipe.
[METH-271](METH_271_OUTPUT_AWARE_ROWS_RESULT_20261002.md) output-aware fixed128
selection passes mean/reserve/source and native numeric gates:22.4154% mean,
21.8746% reserve improvement,384 native median/max5.36e-7/1.24e-6. Cost fails:
11.397642ms versus10ms. Session5390 exits0; no active job/full archive.
[METH-272](METH_272_FUSED_INPUT_RESULT_20261002.md) shared input-dot fusion
conserves344,064 native values/checksum but fails cost/speedup:10.991488ms
versus contemporaneous control10.864646ms,ratio1.011675. Session30895 exits0;
no active job. This fixed optimization closes;271 cost and267 stops remain.
[METH-273](METH_273_PRIVATE128_PHASE_RESULT_20261002.md) phase diagnosis passes
all bitwise/checksum guards:shared+concurrent down/right92.1427%,private plus
team boundaries6.0249%. Diagnostic overhead retained,not a rate qualification.
Session74210 exits0; no active job. [METH-274](METH_274_I16_SHARED_INPUT_RESULT_20261002.md)
I16 shared input passes all6144 CPU source/reserve/GPU-native controls and
59,768,832 exact integer dots,but10.550804ms fails10ms. Session26089 exits0;
no active job/full archive.
[METH-275](METH_275_PAIRED_I16_LAYOUT_PROTOCOL_20261002.md),paired16 gate/up
Q8 layout/integer reader,same function/337.596MB,bitwise all6144 outputs,
contemporaneous control and absolute10ms/5% speedup gates.
[METH-275](METH_275_PAIRED_I16_LAYOUT_RESULT_20261002.md) conserves ALL6144
CPU outputs/quantizer/integer guards,but10.645058ms and3.5426% observed paired
gain fail10ms/5% gates. Session20495 exits0; no active job/full archive.
**Next exact action:** implement then freeze [METH-276](METH_276_DIAGNOSTIC_COMPLETE_CORE_PROTOCOL_20261002.md),
diagnostic-only complete I16/private128 archive/standalone loader/all725
fields and6144 composition controls. Explicit assembly-order revision;
prior component-cost/267 stops unchanged,no promotion. Actual whole quality/
same-artifact>=50 remain due; no276 code/artifact/score yet.

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
