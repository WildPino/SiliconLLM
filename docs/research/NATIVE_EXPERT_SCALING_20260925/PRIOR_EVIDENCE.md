# Native expert-count scaling: prior evidence

**Purpose.** Evidence inventory for a native `benchmarks/phase60/engine.c` expert-count comparison, keeping top-k, model width, core count, and training recipe fixed. Donor-adaptation work is paused context, not evidence for this native question.

**25 September asset update:** the local E32 checkpoint, phase55 ids/tokenizer,
and E4 export were found and hash-verified; see [NES-00](NES_00_ASSET_AND_DISPATCH_20260925.md).
The earlier "availability unknown" statements below describe the pre-audit
inventory, not current workspace state. No trained native E128 checkpoint was
found in the reviewed native result directories.

## Readout

Native evidence establishes that a trained E32, top-8 MoE can retain the small-model quality advantage over its matched dense arms and can be executed by the phase60 C engine with parity and measured cost. It does **not** establish what quality, routing, or end-to-end cost does when expert count alone rises beyond E32. E128 and E256 are proposed/costed scale rungs, not completed trained comparisons.

The first candidate missing comparison in the reviewed evidence is a same-recipe native E32 versus E128 trained pair, with top-8, width/hidden expert size, core count, data and training budget held constant and the E128 arm adding experts only. The E32 checkpoint and native runner provide an anchor; an E128 trained checkpoint is not evidenced among the reviewed artifacts. Equal tokens do not equal equal total compute or exposure per expert: report each separately. An equal-compute comparison would be a separately specified experiment, not another reading of an equal-token run. Do not claim capacity-only scaling from duplicated weights.

**Anchor mismatch to resolve before a protocol:** the executed `phase59_moe.py` uses D256/N96/L6 and the historical sandbox data/tokenizer; the Phase-64 code ladder proposes L8 and a different vocabulary/data setup. Reusing that old E32 checkpoint as the direct baseline for a new L8 code E128 model would change more than E. Either retain the historical configuration for both arms or train a matched E32 baseline for the ladder configuration. Do not infer that the historical checkpoint is already the proposed S0 rung.

## Evidence ledger

| Evidence and link | Established at what scale | What it does not establish | Reusable runner/checkpoint if found | No-duplication boundary |
|---|---|---|---|---|
| [Scale-up architecture §3.5](../../SCALEUP_ARCHITECTURE.md#35-capacity--sparse-memory--fine-grained-ternary-moe-k1-validated--probe-4-2026-07-02) | Probe-4 trained matched arms for 4k steps at about 22.5M total parameters. E32×h128, top-8, active hidden 1024: BPB 0.8589, versus dense-1024 0.8799 and dense-4096 0.8674; E8×h512 top-2 was 0.8637. Switch-aux router: no dead experts, max/mean ≤1.47. | One seed/short run does not show expert-count scaling. No E128/E256 quality result. The E32 result does not imply larger total capacity itself is useful. | Probe-4 promoted checkpoint `results/phase57/moe_gran.pt` is named in [REPRODUCE.md](../../REPRODUCE.md); file is an off-repo release asset, not confirmed locally. Training runner is [phase59_moe.py](../../../benchmarks/phase57/phase59_moe.py). | E32 quality is an executed trained point; it is not evidence for larger E. Compare trained arms with matched recipe and suitable compute/token controls. |
| [Canonical evaluation](../../CANONICAL_EVAL.md#canonical-table-2026-07-03) | Canonical fp32 CPU evaluation preserves ordering: E32 MoE 0.8589; dense-large 0.8672 (training harness 0.8674); coarse E8 0.8636 (training 0.8637); base 0.8799. | Canonical re-evaluation is not a new training run, seed, or scale rung. Single-seed probe deltas under ~0.005 BPB are below cross-script reproducibility guidance. | `scripts/canonical_eval.py` and checkpoint manifest/release link in the document. | Keep absolute canonical values distinct from within-run paired deltas and training-harness values. |
| [Native phase60 E4 plan/results](../../ENGINE_PLAN.md#e4--moe--two-pool-layout-target-model-moe_granpt-probe-4s-promoted-arm-e32h128-top-8--active) | E4 executed the probe-4 E32 checkpoint: fp32 reference BPB 0.858856 (Δ+0.000002), top-1 100%; LUT+skip ΔBPB +0.000028 vs reference, top-1 99.7363%; counted pool 2400 KB/token; E4-full 701.7 tok/s, with the sandbox cache caveat. | E32 inference parity/performance is not an E-count sweep. At ~8.3M sandbox scale the pool fits cache, so streamed timing is priced, not directly measured at scale. | [phase60 engine.c](../../../benchmarks/phase60/engine.c), [e4_export.py](../../../benchmarks/phase60/e4_export.py), [e4_reference.py](../../../benchmarks/phase60/e4_reference.py), and `moe_gran.pt` if obtained from release assets. | Reuse the E4 path as the native E32 baseline; do not present an untrained synthetic E128 pool as a quality or learned-routing comparison. |
| [Phase64 decisions D1/D2](../../PHASE64_DECISIONS.md#1-d1--recipe-invariance-the-ladder-contract) | Defines invariant recipe and intended capacity dial: S0 E32 → S1 E128 → S2 E256, top-8×h128×L8, with active expert bytes held at 3.1 MB/token. | This is a design decision and property-gate plan, not evidence that E128/E256 were trained or passed gates. | Decision doc specifies BPB versus matched dense, router health, i.i.d.-union, and other rung probes. | Dims E alone may change in the intended ladder, but recipe, data/eval, and paired controls must remain auditable. |
| [Phase64 budget §3/§5](../../PHASE64_BUDGET.md#3-the-candidate-grid) | E128 (105M) and E256 (206M) are costed at D256/N96/L8/V2048, top-8; model predicts same speed bracket, 739–1309 tok/s. Budget prices training/data and expert-kernel overhead separately. | Modeled tok/s is not measured native E128/E256 performance or trained quality. Expert path is gather/kernel/dispatch-bound (integrated ~4.2 GB/s; pure kernel t6 17.0 GB/s), not a simple DRAM roofline. | [bench_64_1b.sh](../../../benchmarks/phase64/bench_64_1b.sh) is a reusable synthetic microbench for the budget anchors, not a trained-model runner. | Re-measure end-to-end timing at the target E and core count; keep engine overhead visible and use identical decoding/eval protocol. |
| [Native routing evidence, ENGINE_PLAN](../../ENGINE_PLAN.md#e4--moe--two-pool-layout-target-model-moe_granpt-probe-4s-promoted-arm-e32h128-top-8--active) and [scale architecture §3.5](../../SCALEUP_ARCHITECTURE.md#35-capacity--sparse-memory--fine-grained-ternary-moe-k1-validated--probe-4-2026-07-02) | Probe-4 working set over 8 positions touches 84–89% of E versus ~90% iid; near 100% by 16–32 positions; persistence is base-rate k/E. Native E4 reproduced mean 86.5% at N=8. | No warm expert pool/locality advantage at E32. Does not prove that routing stays iid at E128/E256, across code domains, or after a different training recipe. | E4 working-set/accounting instrumentation is described in ENGINE_PLAN; exact reusable standalone probe file was not identified in the reviewed file list. | Record observed unions against the iid expectation at each E. Treat any locality change as a measured result, not an assumption or failure. |
| [fp32 organ constraints, phase64 training plan §4](../../PHASE64_TRAINING_PLAN.md#4-the-curriculum-per-rung-dossier-skeleton-af-adapted-order-itself-is-mve-checked) and [ENGINE_PLAN phase61 result](../../ENGINE_PLAN.md#phase-61-closed-architect-2026-07-03-ssm-projection-ternarization-rejected--clean-double-fail-the-fp32-projections-stay) | The scale-up recipe keeps projections, SSM, head, and router fp32; earlier SSM projection ternarization failed its quality gate (0.8940/0.8972 versus ≤0.8899). | Expert weight format/precision does not license changing precision of control organs while scaling E. | Use the fixed precision map in the training recipe and phase60 native format. | A precision change is a separate variable and requires its own paired gate; do not bundle with E-count scaling. |
| [Phase64 training plan](../../PHASE64_TRAINING_PLAN.md#5-the-ladder-costed-desk-model-brackets-mfu-5-15-assumed-on-2t4-scan-heavy--measured-at-mve) and [budget §5](../../PHASE64_BUDGET.md#5-constraint-inversion--what-binds-now) | Training budget and licensed code-data supply, not inference footprint, bind the planned ladder. Plan requires roughly 6–10B licensed code tokens; measured MVE throughput and later rung costs are needed to reprice. | A rung design/cost sheet is not evidence that it ran. No verified E128/E256 trained checkpoint or native result was found in the reviewed artifacts. | The plan's A→F curriculum is reference only; sparse-slot rewrite is an explicit prerequisite. MVE scripts are apparatus, not an E128 runner/checkpoint. | Attribute quality to trained, data-backed runs. Keep data order, token budget, recipe, seeds and dense control visible in the comparison. |

## Comparisons that must stay separate

- **Native phase60 MoE probe-4:** E32×h128 top-8 trained from scratch in the project’s small native architecture family; C inference engine E4 validated this checkpoint. This is the only reviewed native learned-expert quality point.
- **Donor-adaptation E36/E63 and related work:** donor-model carve/adaptation experiments use different base models, parameters, routing/selection, objectives and engine paths. They may inform questions to ask, but their outcomes cannot fill the native E128/E256 evidence gap. Donor work remains paused and intact.
- **Rungs:** E32 is an executed trained result. E128 and E256 are specified/costed S1/S2 design rungs. Their projected capacity and speed are not measured outcomes.
- **Capacity accounting:** increasing expert count creates stored parameter capacity. Duplicating E32 weights into more slots only increases bytes on disk; it does not establish independent learned capacity, quality gain, useful routing, or efficient inference.

## Evidence interpretation by question

### Quality

- The useful prior is the matched probe-4 comparison, not the absolute BPB in isolation.
- The native trained E32 arm beat both the dense arm at matched active width and the larger dense upper-bound arm in that short run.
- E8 coarse experts performed worse than the granular E32 arm in the same probe.
- This supports retaining E32×h128/top-8 as the tested starting point.
- It does not show a monotonic quality curve in E.
- It does not show that E128 or E256 can be populated with diverse useful experts under the same token budget.
- The canonical re-evaluation checks harness consistency and preserves ordering.
- It does not add independent training seeds or resolve small-delta uncertainty.
- Scale-up decisions define BPB against a matched dense control as a per-rung read.
- That control matters because an E-count change can also change optimizer exposure and total training cost.
- No reviewed result supplies a native E32/E128 paired quality delta.

### Routing

- Probe-4 rejected the proposed hot-pool assumption at its tested scale.
- Nearly the whole E32 pool was visited across only eight positions.
- Persistence was approximately the base rate implied by k/E.
- The phase60 E4 engine reproduced the small-scale working-set observation.
- This gives an inference-side transfer check for E32, not for larger E.
- The phase64 rung plan calls for expert-union comparison with the independent-routing expectation.
- Larger E can alter both router competition and the number of experts selected across a window.
- Code-domain structure might differ from homogeneous TinyStories; that possibility is unknown, not a presumed locality benefit.
- Record dead experts, load skew, per-layer union, persistence, and workload/domain for each trained rung.
- If routing becomes local, treat it as a finding and remeasure cost; do not silently assume warm-cache savings.

### Inference cost

- Phase60 E4 establishes native dispatch correctness and one E32 end-to-end timing point.
- It also counts expert bytes touched per token and prices those bytes against a streaming floor.
- The sandbox pool fits cache, so the E4 time is not a direct large-pool DRAM measurement.
- Phase64 measured a synthetic expert kernel and decomposed the gap to integrated engine throughput.
- The integrated path includes selection, dispatch, dequantization, activation sparsity, and expert combination.
- Therefore expert-count scaling should report component timing as well as total tok/s.
- Hold core count and compiler/build configuration fixed across compared runs.
- Use the same token slice, decode settings, warmup, and repetition policy.
- Report pool size and bytes/token separately from quality and latency.
- Budget brackets are useful priors, not replacements for an E128/E256 engine measurement.

### Precision and training constraints

- The native recipe keeps the backbone/control organs in fp32 and ternarizes the designated MLP/expert weights.
- The earlier SSM projection quantization A/B failed the registered quality gate.
- An expert-count comparison that also changes the precision map would mix two questions.
- Phase64 training plan calls for dense pretraining/QAT followed by sparse MoE upcycling.
- Magnitude matching at upcycle is required to preserve transition BPB.
- The sparse-slot training rewrite is named as a prerequisite for the planned ladder.
- Its presence in a plan does not establish that it or the full ladder completed.
- Data provenance, license filtering, deduplication, and pinned validation are part of the evidence chain.
- Training throughput measured at the MVE is needed to revise the cost envelope.
- The training and licensed-data budget is a binding limitation before inference capacity is.

## Suggested evidence record for the first pair

This is a checklist for interpreting a future native comparison, not a newly registered threshold.

- Identify the exact E32 and E128 configs, source revision, checkpoint hashes, and exporter format.
- State the fixed top-k, D, expert hidden size, layer count, core count, and precision map.
- Record seed count and any seed pairing.
- Record optimizer, schedule, batch, sequence/context mix, tokens, steps, and total training compute.
- State whether the comparison equalizes tokens, compute, or both, and explain any remaining difference.
- Include the matched dense/control arm specified by the chosen comparison design.
- Report per-domain held-out BPB with the same harness for all arms.
- Record router dead-expert count, max/mean load, per-layer usage, persistence, and unions over stated window lengths.
- Compare observed unions to the iid expectation without assuming that either outcome is desirable.
- Measure native engine parity before interpreting speed or quality deltas.
- Report end-to-end tok/s, per-component time, pool footprint, bytes/token, and core configuration.
- Keep all design projections labeled as projections until the corresponding trained and engine runs exist.

## First comparison to resolve

The candidate question is whether a natively trained E128 model improves a predeclared held-out quality measure over E32 at fixed top-8, D256, h128, layer count, core count and training recipe, with the chosen matched controls and router/union telemetry. Use L8/code only with a matched L8/code E32 baseline; otherwise retain the historical configuration. The branch index leaves this choice open until asset binding and cost review. Report training tokens, steps, compute and seeds so extra capacity is not conflated with extra optimization. Measure end-to-end engine cost, pool bytes, dispatch cost and tok/s on the same reference protocol. E256 is a later rung, conditional on the E128 evidence and training/data budget.

Still unknown from prior evidence: availability and identity of the E32 checkpoint in the current workspace; whether a trained native E128/E256 checkpoint exists outside the reviewed paths; seed variance at these rungs; scaling of router health and locality on code data; and whether additional experts improve useful quality enough to offset training cost. No 10B/100B inference or quality claim follows from this inventory.
