# Progressive ternary conversion: bounded feasibility decision

6 October 2026. **Research complete with a scoped negative result.** No
investigated procedure has demonstrated pretrained behavior preservation and
the required storage/execution usefulness together. No candidate is qualified
for integration into `native-expert-scaling`. The initial idea is technically
viable as an optimization procedure, and behavior-aware progression can
improve direct rounding; these facts did not establish the requested useful
preserving conversion under the tested methods, thresholds and budgets.

The [Italian goal](GOAL.md) explicitly permits a rigorous negative conclusion
after an extensive literature review and a finite justified set of tests.
This closes that research decision. It does not claim mathematical
impossibility, exhaustion of algorithms or failure of untested budgets.

## What the literature establishes

The [review](literature_20261006/report.md) consolidates47 primary paper/project
clusters, with inspected source versions, source/evidence/claim ledgers,
counterevidence and reproducible search records. The printable report is a
15-page convenience copy; Markdown and evidence ledgers are authoritative.
Its original resumption appendix is a historical snapshot before PQT012/013;
the present conclusion and index supersede that operational state.

Progressive quantization of an existing checkpoint is established in the
literature. INQ freezes progressively quantized groups and retrains remaining
weights; GPTQ-family procedures compensate quantization error using calibration
curvature, without pretraining from zero. Thus the owner's proposal belongs
to a meaningful family of discrete conversion/behavior reconstruction
procedures. Uniformly random weight changes are one search proposal, not a
necessary definition of progressive conversion. A controlled behavior-aware
selection rule is more directly supported by the reviewed work, but no
experiment here proves random search cannot succeed.

Low-bit results must be read within their actual alphabet and objective.
Positive ternary-pretraining results do not show arbitrary pretrained float
conversion. Successful conversion papers may allow larger task/perplexity loss,
different group granularity, affine/asymmetric endpoints, mixed precision,
outlier/correction paths or substantial calibration training. These are
material differences from near-exact donor behavior and must be counted.

QMoE is directly relevant to Switch-base-128, but its reported float loss1.73,
ternary1.99 and1.93 with20k calibration examples are not the project's1% expert
output-RMS criterion. Rowwise asymmetric endpoints, special tokens,
capacity1024, propagated quantized contexts and specialized encoding/runtime
also differ. PQT013 tests two of these mechanisms under the preserved project
context, rather than reproducing its full system. PT²-LLM and CAT-Q motivate
other no-backpropagation or optimized conversion procedures; their settings
and actual fitting costs remain distinct. Primary citations and locators for
these claims are in the review; reported results are not local reproductions.

Native speed requires actual packed execution and a relevant cost comparison.
Theoretical bit counts, a compact archive and calibrated output improvement
cannot establish useful CPU expert capacity. Mixed precision and richer
codebooks are credible alternatives but change the representation question.

## Finite empirical coverage

This table groups the qualified experiment sequence by scientific question.
It does not count source admissions or numerical apparatus checks as quality
successes, and it does not treat repaired runs as independent replications.
Each linked outcome carries its own protocol, source/input identity, raw
retention, audit, metric gates and measured costs.

| Question | Evidence | Qualified conclusion |
| --- | --- | --- |
| Does progressive behavior-aware approximation beat direct rounding? | [PQT001-R1](PQT_001_R1_RESULTS.md) | Relative pilot improvement exists; inadequate absolute donor fidelity |
| Can bounded scale/offset or direct behavioral head fitting preserve predictions and own-prefix generation? | [PQT002](PQT_002_RESULTS.md), [PQT003](PQT_003_RESULTS.md) | No ternary absolute prediction/generation preservation pass |
| Does broader matched calibration resolve the head failure? | [PQT004](PQT_004_RESULTS.md) | Relative benefits exist; every absolute quality gate still fails |
| Do original expert functions survive calibrated ternarization? | [PQT005-R1](PQT_005_R1_RESULTS.md) | Tested ternary function errors38.84–50.83%; I8 preserves functions but exceeds storage budget |
| Can actual packed expert codes execute correctly on CPU? | [PQT006](PQT_006_QUALIFICATION_RESULTS.md) | Packed native correctness passes; no qualified timing or useful-capacity conclusion |
| Does complete progressive conversion preserve the model? | [PQT007](PQT_007_RESULTS.md) | All169 local objectives improve over direct rounding while full-model behavior fails |
| Does bounded global one-shot/staged adaptation help enough? | [PQT008](PQT_008_RESULTS.md), [PQT009-R1](PQT_009_R1_RESULTS.md) | Measured schedule/breadth effects fail joint and absolute gates |
| Is shared embedding/readout the isolated rescue? | [PQT010](PQT_010_RESULTS.md) | Floating/I8 restorations do not meet behavior gates; floating storage also exceeds budget |
| Do jointly learned group amplitudes resolve the remaining global failure? | [PQT011](PQT_011_RESULTS.md) | Original-initialized matched256-update arms fail all required preservation/joint signals |
| Can original natural Switch contexts plus counted low-rank compensation meet quality and storage? | [PQT012](PQT_012_RESULTS.md) |40 artifacts/120 metrics audited; no joint ternary/mixed pass, I8 too large |
| Do matched row alphabets/asymmetry and a calibration-only EOS filter rescue that result? | [PQT013](PQT_013_RESULTS.md) |32 artifacts/96 metrics audited; every arm fails1% fidelity, all pass storage |

Whole-model tests cover the qualified Qwen0.5B donor and the declared
prediction/continuation tasks; they are not exhaustive general-capability
evaluations. Original Switch expert tests cover selected functions and admitted
natural routing/context distributions, not the entire128-expert encoder/decoder
model. Earlier inherited I8-core contexts remain explicitly distinct from
original float-model contexts. No metric is substituted across those scopes.

Two examples explain why continuation without a promotion stop is unjustified.
In PQT011, learned staged Wiki argmax agreement is29.199219% and news5.761719%,
with KL2.635343/7.105815 and no exact continuations, far from the declared99%
prediction and continuation requirements. In PQT012, rank256 compensation
fits calibration below0.5% while held-out errors reach2.35–52.22%; I8 meets
fidelity at50.18% of FP16 bytes, exceeding35%. These are measured failures of
the conjunction, not fitting or compression accomplishments relabelled as
preservation.

PQT013 addresses the strongest materially different inexpensive mechanism
identified by the review. Its best individual unfiltered asymmetric held-out
expert error is2.96% and its worst98.14%, with all confidence/quality gates
failing. Filtering EOS worsens expert18's natural evaluation. Complete files
are12.68–15.65% of FP16, so storage is not the failing component. All762 raw
scientific/source members and74 final outer/client entries have exact retained
and actual Git proofs; no preservation conclusion depends on terminal status
alone. Scientific source, outputs, metrics, code, first failures and complete
child/resource costs are preserved.

## Why the investigated scope can close

The selected sequence spans direct/progressive rounding, calibrated error
compensation, scalar/group adaptation, behavioral head reconstruction,
calibration breadth, complete-model staged adaptation, shared-state rescue,
original nonlinear FFN residual compensation and rowwise asymmetric/filter
controls. Later tests respond to distinct observed or literature-supported
hypotheses, rather than repeatedly tuning a failed method on the same observed
outputs. Relative gains were retained, but never substituted for absolute
preservation or complete storage.

The remaining alternatives are meaningful and **untested**, with these reasons
for excluding them from this finite closure:

| Alternative | Reason for deferral and remaining uncertainty |
| --- | --- |
| Random coordinate/batch search | Not independently tested as an exhaustive discrete search; reviewed guided procedures provide the finite comparison. The large combinatorial space offers no declared bounded guarantee. This is an efficiency judgment, not an impossibility proof |
| Joint-FFN soft-to-hard reconstruction such as CAT-Q | A distinct local optimizer remains untested. Reviewed60-epoch training is substantially different from the admitted fixed budgets; existing global/head optimization does not reproduce or rule out this method. No promising absolute artifact requires another local tuning cycle to complete the finite decision |
| PT²-style affine correction/rotations or a faithful whole-system QMoE reproduction | Different representation or end-to-end routing/capacity/context/metric contract. Positive literature evidence remains valid; the matched screen does not rule out these settings |
| Multi-plane, vector/trellis/codebook or adaptive mixed precision | Changes single-plane ternary storage/compute semantics. It deserves a separately scoped comparison rather than retroactive success under the current alphabet |
| Much longer full-model QAT/distillation | Not demonstrated infeasible from hardware alone, but not admitted or tested within this bounded study. Three separate T4×2 sessions do not form one shared six-GPU training allocation; actual admission is per account/session |
| Broader source models, all experts, fresh confirmation data and native integration | No current quality-qualified artifact to promote. These remain future validation requirements if a materially changed procedure succeeds |

No existing failing gate is relaxed. The absence of a useful artifact is a
sufficient scientific stop for this investigation; native execution timing
would not repair its quality failure. The owner reservation was never granted,
so native latency/speedup remains **unmeasured**, rather than failed. Local
process guards preserved the other active checkout and never granted a timing
window from an idle sample. No claim of infeasibility from GPU quota exhaustion
is made: the latest run completed well within its admitted resources.

## Engineering decision and resumption

Do not integrate or promote any current ternary artifact into
`native-expert-scaling` as preserved pretrained capacity. Retain the independent
packed-code correctness implementation, original-context acquisition pipeline,
qualified codecs/auditors and literature dossier as reusable research assets.
Their reuse requires the same scientific scope and fresh resource checks.
No merge, push or modification of the other checkout was performed.

The research is complete; no benchmark, monitor, fetch, fit, source acquisition
or approval is pending. Keep private terminal kernels/datasets unchanged and
do not repeat qualified work to resume. A new positive-direction investigation
would need a materially changed question (for example task-quality tolerance,
joint local fitting or richer/mixed representation), prospective budgets/gates,
independent fresh final validation after selection, and native timing/integration
only for a quality-qualified artifact. This is a possible future study, not
unfinished required work under the accepted finite negative-result goal.

The [index](TERNARY_INDEX.md) is the entry point for provenance and historical
records. The conclusion is a bounded research decision, not proof that pretrained
models can never be ternarized.
