# SiliconLLM research handoff

**Updated: 15 October 2026, Europe/Rome. Workspace: `D:/_THINGS/Progetti/SiliconLLM`. Branch: `research/native-expert-scaling`. Research HEAD before this documentation handoff: `7b93d10`.**

## 1. Exact stopping point

The owner has closed this session and requested documentation only. **The scientific objective is INCOMPLETE. The goal tool reports `paused`. Do not resume research, launch experiments or allocate compute merely because this file is opened.** Resume substantive work when the next user authorizes it. Older reports saying `active`, `running`, `next` or `prepared` are dated observations, not current process instructions.

The latest completed work is a complete original-Qwen recipient CPU comparison, not a donor-memory model:

| Candidate | Numerical result | Complete cost result | Decision |
| --- | --- | --- | --- |
| Original Qwen2.5-0.5B-Instruct, all-I8 matrices / F32 activations | All 32 saved vocabulary rows match decoded-reference top1; worst relative L2 5.833e-6 | All five case groups fail the frozen 18 ms/stability envelope; worst process mean 28.02 ms | Closed; no unchanged runtime grid or long recovery |
| Same source, W4G32 FFN / W8-A8 attention and head | Exact normalized states and whole-vocabulary logits at four selected positions after a documented reference-exp repair | All five groups fail; worst process mean 43.53 ms at context 2048 | Closed; fewer matrix bytes did not produce adequate physical speed with this kernel |

Both comparisons execute all 24 source blocks, actual causal attention, KV state, FFN and full vocabulary head. Neither contains donor-derived conditional memory. Neither has fresh donor-relative chatbot-quality admission. No final artifact reaches the goal.

**No numerical job is waiting. No T4 training has been launched or allocated.** A fresh handoff check found all **48 owned PID/create-time identities absent** across the six latest recipient result families. This checks explicitly owned receipts, not every unrelated process on the machine. Earlier research closures document their own workers and controller terminals.

The next selected step, after explicit resumption, is to pin and qualify a **substantially cheaper pretrained online core**, currently SmolLM2-135M-Instruct as a bounded cost candidate, before building and jointly recovering a large-donor conditional-memory artifact. Its weights have **not** been acquired, it has **not** been implemented in C, and it has **no** CPU or quality admission. A smaller standalone model is not the new objective.

There is no unfinished audit that must be resumed first. The interrupted FFN router audit described in the previous handoff was subsequently completed and closed.

### Preserved previous handoffs

- [Exact previous handoff, dated 14 October](docs/research/NATIVE_EXPERT_SCALING_20260925/HANDOFF_SNAPSHOT_20261015/HANDOFF_PREVIOUS_20261014.md):95,154 bytes, SHA256 `e03b36042ab9f82331f559eab37e8bd50c5310ccf89299c77266111edb895d73`.
- [Snapshot receipt](docs/research/NATIVE_EXPERT_SCALING_20260925/HANDOFF_SNAPSHOT_20261015/snapshot_receipt.json).
- [Fresh owned-process check](docs/research/NATIVE_EXPERT_SCALING_20260925/HANDOFF_SNAPSHOT_20261015/owned_process_check.json).
- The earlier 83,691-byte handoff remains at [HANDOFF_PREVIOUS_20260730.md](docs/research/NATIVE_EXPERT_SCALING_20260925/HANDOFF_SNAPSHOT_20261014/HANDOFF_PREVIOUS_20260730.md).

Those documents preserve the detailed older engine/Falcon history. Their old next-action, no-delegation and live-process statements are superseded here.

## 2. Owner instructions and accepted scope

Repository code and documentation must be **English**. Conversation may remain Italian. Preserve historical evidence verbatim when archiving it, including its original language.

Read root [AGENTS.md](AGENTS.md). Do not add automatic signatures, work attribution, service identities, coauthor trailers or assistant/vendor prefixes to commits, branches, descriptions or work products. Preserve the owner's configured Git identity. Technical donor, dependency and protocol names are allowed where necessary. Stage explicit files. Research commits have used `git -c commit.gpgSign=false commit ...` without changing the configured identity or global settings.

Latest owner clarifications:

1. **One family spanning genuinely very large models suffices.** Unrelated-family generality is an extension.
2. **Native pretrained MoE is allowed.** Dense-to-MoE conversion and a dense70B donor are not mandatory prerequisites. Further expert subdivision is allowed. A universal dense-impossibility proof is unnecessary before choosing MoE; untested routes remain untested.
3. **Months of T4 usage are acceptable.** Long joint adaptation must not be dismissed solely because a short local fit failed. This does not mean a GPU is allocated, cloud host RAM is sufficient, donor labels are free or concurrent GPU count is unlimited.
4. Pause the goal when waiting for a numerical run expected to take **at least one hour**; the owner wakes it manually. Shorter runs must not trigger a pause. The current pause reflects session closure, not an outstanding long run.
5. Delegation was explicitly authorized for cataloging, implementation, literature and checks. The old blanket prohibition in the archived handoff no longer applies. Reserve the most expensive independent strategic review for difficult, deep conceptual questions; use ordinary workers for routine implementation/audits. Completed session agents are not dependencies for a new session.
6. Stay orderly and periodically reconsider the complete pipeline. Avoid endless thread/codec/rank grids that cannot change the complete cost or quality decision.
7. Preserve first faults, fixed FIT/DEV rules and prescribed checks. The old fail-fast request was revoked. Poor model quality does not excuse skipping its planned numerical audit; a numerical implementation failure must stop subsequent cost admission.
8. No recurring automation or monitoring was requested. `graphify` is opt-in. The separate donor-adaptation roadmap is frozen; consulting its evidence does not restart that project.

Authorities: [scope clarification](docs/research/NATIVE_EXPERT_SCALING_20260925/SCOPE_CLARIFICATION_20261015.md) and [training-resource clarification](docs/research/PIPELINE_REASSESSMENT_20261015/TRAINING_RESOURCE_CLARIFICATION_20261015.md).

## 3. Goal and completion criteria

The preserved [original objective](docs/research/NATIVE_EXPERT_SCALING_20260925/HANDOFF_SNAPSHOT_20261014/goal-objective.md) asks for a reproducible pretrained-chatbot-to-C transformation. Later owner clarifications override its broader family interpretation. The architectural thesis remains: **separate useful total stored capacity from active computation and memory traffic per token**.

The target has a compact reusable contextual core and useful selectively consulted capacity, with routing IDs **and contextual masses actually computed on CPU**. SSM/SWA, LUT, ternary, low-bit weights, expert partitioning and changed representations are means, not proof of conversion. Joint recovery may change source internals if useful behavior is preserved within explicit measured limits.

| Requirement | Established evidence | Missing evidence |
| --- | --- | --- |
| Reproducible transformation | Source custody, captures, local operators, C formats, exact decompositions and complete small-core controls | Successful transfer into the selected conditional artifact |
| Useful chatbot relative to donor | Historical scoped controls and full-width Falcon bridge on a known corpus | Fresh own-history quality of the efficient final artifact, held-out formulations and declared contexts |
| At least50 useful accepted batch-one CPU tokens/s | Historical infilling rates and component timings | Same artifact that passes quality, with complete routing/core/head/state/frontend charged |
| Useful large conditional capacity | Exact child arithmetic and local source-neuron coverage | Causal usefulness of distinct entries and actual very-large-family scaling at contained active work |
| Physical memory behavior | Addressed-byte ledgers and resident/evicted fixtures | Final-model DRAM/cache evidence; logical payload is not measured DRAM |
| Sustainable conversion | Bounded tools and measured local runs | Donor acquisition/extraction, data, training state, GPU-hours, host memory, storage and resumable schedule |

The improvement target is 100 useful tokens/s. Fifty tokens/s permits 20 ms total per accepted token. Recent recipient screens prospectively require **18 ms core allowance and max/min rate spread<=1.10**, tentatively leaving2 ms for fusion/frontend. These are experimental allowances, not universal hardware bounds.

A small fast endpoint, synthetic experts, duplicated entries, metadata parameter counts, or quality from one model combined with speed from another cannot complete the goal. Historical internal-layer 1%/5% gates are recipe-specific evidence, not universal necessary conditions for another jointly trained geometry. Do not silently relabel their failures.

## 4. Reading order and evidence navigation

1. [Latest mixed-recipient result](docs/research/PIPELINE_REASSESSMENT_20261015/RECIPIENT_MIXED_SCALAR_RESULT_20261015.md), [all-I8 result](docs/research/PIPELINE_REASSESSMENT_20261015/RECIPIENT_INT8_BASELINE_RESULT_20261015.md), [recipient triage](docs/research/PIPELINE_REASSESSMENT_20261015/RECIPIENT_CORE_TRIAGE_20261015.md).
2. [Current architectural decision](docs/research/PIPELINE_REASSESSMENT_20261015/CARVING_DECISION_20261015.md) and the **final revised-priority section** of [strategy review](docs/research/PIPELINE_REASSESSMENT_20261015/CARVING_STRATEGY_REVIEW_20261015.md). Its initial codebook-first priority is historical and superseded.
3. [Carving research brief](docs/research/PIPELINE_REASSESSMENT_20261015/CARVING_RESEARCH_BRIEF_20261015.md), connecting positive and negative experiments to the representation problem.
4. [Index](docs/research/NATIVE_EXPERT_SCALING_20260925/INDEX.md), [method](docs/research/NATIVE_EXPERT_SCALING_20260925/METHOD.md), [prior evidence](docs/research/NATIVE_EXPERT_SCALING_20260925/PRIOR_EVIDENCE.md). Lower historical sections are not current launch orders.
5. [Experiment catalog](docs/research/NATIVE_EXPERT_SCALING_20260925/EXPERIMENT_CATALOG_20261014.md) and [primary decisions](docs/research/NATIVE_EXPERT_SCALING_20260925/DECISIONS_FROM_CATALOG_20261014.md). The609 catalog entries count report files in that snapshot, not 609 independent successful experiments.

Use the catalog to select primary reports that could change the next decision. Do not reread every old file before making progress. Follow a claim through protocol, binding, result, audit, terminal receipt, readback and closure. A producer result is not automatically independently admitted.

## 5. Work completed since the previous handoff

### 5.1 Interrupted FFN active-router audit was resolved

[FFN_ACTIVE_ROUTER_RESULT_20261014.md](docs/research/NATIVE_EXPERT_SCALING_20260925/FFN_ACTIVE_ROUTER_RESULT_20261014.md) is COMPLETE: full numerical audit PASS, no admitted FIT-selected candidate. It reused the frozen producer without repeating fitting, predictions, source queries, GPU work or reserved evaluation.

All 576 predictions, IDs/masses,144 source-proxy controls, derivative geometry, moments, factors and seals passed. The source-sensitive geometry improves some PCA errors, but site 0 and site 23 FIT-selected candidates fail DEV mean, while site 12 has no eligible tested FIT rank. Do not promote another rank using DEV. The interrupted worker's missing exit receipt remains missing; the successful resumed audit does not invent its exit code or OS cause.

### 5.2 Falcon compact/depth recipes and source coverage were closed

- [Fixed-depth diagnostic](docs/research/NATIVE_EXPERT_SCALING_20260925/FIXED_DEPTH_DIAGNOSTIC_RESULT_20261015.md): tested3/6/12 retained-block recipes fail. The nonselected6-block case has0/16 tasks and DEV KL14.82421/disagreement99.1114%; numerical checks pass. This rejects those omission/recovery recipes, not every long-trained compact model.
- [Balanced replay](docs/research/NATIVE_EXPERT_SCALING_20260925/CHATBOT_BALANCED_REPLAY_RESULT_20261015.md): fixed 512-update compact fit is fully audited. New FIT KL0.70077/disagreement16.418% passes its gate; old DEV2.56407/35.993% and new DEV7.11348/81.487% fail. No unchanged152-case fit or export follows.
- [Source coverage](docs/research/NATIVE_EXPERT_SCALING_20260925/CHATBOT_SOURCE_COVERAGE_RESULT_20261015.md):432/432 new source cases,360 FIT/72 DEV,76,512 labels. All 5.032B saved BF16 score words,411 raw tensors,873 output files and8,390 input pins pass complete CPU adoption audit. Source-only data are adopted; no learner is admitted.

Source capture is finished. Do not restart it, automatically retrain the failed D512 student, or make teacher states an uncharged deployment fallback.

### 5.3 Exact native-MoE children and complete projection cost

[Packed child layout](docs/research/NATIVE_EXPERT_SCALING_20260925/GRANITE_CHILD256_LAYOUT_RESULT_20261015.md): synthetic512-wide SwiGLU splits into two256-wide children by exact Q4_K block copying. All 6,144 blocks are preserved; both children with inherited parent mass reconstruct at relative L2 5.58e-8. One child addresses half the expert payload. Bias is counted once and masses must retain their contribution semantics. This fixture is not actual donor weights and does not prove quality after omitting a child.

[Projection-bundle cost](docs/research/NATIVE_EXPERT_SCALING_20260925/GRANITE_CHILD256_COST_RESULT_20261015.md): one-child expert payload is about 84.9 MB plus125.8 MB core/head, before complete attention/KV. Resident medians14.365/15.209/11.995 ms and evicted15.434/19.609/20.338 ms all fail the frozen 14 ms/stability allowance. Numerical controls pass. Its fastest sample does not license final throughput or a quality pilot.

### 5.4 Direct large-model core falsifiers

[Protected GPT-OSS120B core](docs/research/PIPELINE_REASSESSMENT_20261015/PROTECTED_CORE_120B_CPU_RESULT_20261015.md): about 557.38 MB Q4_K core plus53.08 MB F32 routers, with experts/head/attention-KV made free in a favorable synthetic fixture, yields three-process medians34.39/50.95/53.88 ms on three physical cores. This rejects that fixed variant, not every six-core implementation or transformation.

[Actual first-attention source slice](docs/research/PIPELINE_REASSESSMENT_20261015/FIRST_ATTENTION_SOURCE_RESULT_20261015.md) was acquired/audited: ten BF16 tensors,53.106 MB, plus1,501 embedding rows,8.646 MB;24 FIT/12 DEV cases and4,498 variable queries beyond their common prefix. No full120B checkpoint or source expert bank was acquired.

Two actual-source transformations were rejected:

- [Rank256 Q/O recipe](docs/research/PIPELINE_REASSESSMENT_20261015/FIRST_ATTENTION_RANK_RESULT_20261015.md):512 updates complete; branch L2 FIT0.211763/DEV0.292680 versus full-F32 control0.006635/0.006589. Independent CPU replay of all 36 source/candidate cases confirms rejection. Producer exit1 `FAILED_GPU_RELEASE`, with64 MiB still allocated at cleanup, remains explicit despite CPU numerical PASS. No GPU retry, rank512 or extra steps was selected.
- [Single source-head-group oracle](docs/research/PIPELINE_REASSESSMENT_20261015/FIRST_ATTENTION_SOURCE_GROUP_RESULT_20261015.md): teacher chooses the best of16 contiguous four-head groups and any real scalar mass, yet pooled L2 is0.518095 FIT/0.499741 DEV; all-group control passes. A learned selector cannot repair this fixed oracle class. Arbitrary learned head transformations remain open.

These results motivated changing the always-on representation rather than repeatedly shrinking only expert payload.

## 6. Selected direction and unimplemented hypotheses

### First priority: compact pretrained recipient plus frozen donor memory

The selected hypothesis moves expensive donor computation **offline**:

1. Build frozen values from canonical donor contexts, with exact key/layer/vector identity.
2. Obtain a bounded memory ID from available history, for example longest-suffix hashing or a bounded hierarchy. No full-bank scan or full donor evaluation online.
3. Read selected rows and compute contextual key/value projections, gates and masses through small charged maps.
4. Jointly recover the pretrained recipient, adapters and masses over substantial general text and interaction data.
5. Export and evaluate the same complete conditional artifact in C.

Large-donor QKV/O/FFN matrices become conversion cost rather than per-generated-token cost. Consequently, the protected-source-core failure does not directly lower-bound this different representation.

**No new grafting bank, joint training or large-donor transfer has been implemented or demonstrated.** Existing new code qualifies prospective recipient runtimes, whose measured cost is insufficient.

The main scientific risk: short-ngram values may preserve lexical associations while losing reasoning, instruction following and long-tail knowledge. Table size is not retained model capacity. A small average benchmark gain is insufficient. Useful donor knowledge must be shown on held-out donor-gap tasks, with causal dependence on distinct entries and better capacity at matched active cost.

### Other priorities

Second: long-trained fine-grained native-MoE carving. Exact compatible FFN children exist; cheap child IDs/masses, useful sparse omission, attention/core/head changes and whole-model recovery do not. A router must choose before loading/evaluating all children.

Third: additive activation-code tables as an optional cheap mechanism study. A proposed32-dimensional group/32-centroid code has only 5/32=0.15625 bits per dimension and may be far too coarse. Summed-vector arithmetic/cache cost and contextual coding accuracy are unproven. This is no longer the main prerequisite after the T4 clarification. No such fit was launched.

A proposal in the strategy report is not acquired weights, executed code or admission.

## 7. Literature and pinned metadata

The primary-source package is [ACTIVE_WORK_REVIEW_20261015](docs/research/PIPELINE_REASSESSMENT_20261015/ACTIVE_WORK_REVIEW_20261015/run_manifest.json), with [sources](docs/research/PIPELINE_REASSESSMENT_20261015/ACTIVE_WORK_REVIEW_20261015/sources.jsonl), [evidence](docs/research/PIPELINE_REASSESSMENT_20261015/ACTIVE_WORK_REVIEW_20261015/evidence.jsonl), acquisition/derivation scripts and raw pinned metadata. Its manifest describes that bounded review, not all subsequent experiments. Empty/unfinished claim ledgers are not a finished systematic report.

Primary works examined include Memory Grafting, Engram, STEM, STAPLE, MoLE, LongCat, Switch, MADDNESS, LUT-NN and LUT-LLM. Read retained versions before quoting details; paper hardware results do not establish local CPU performance.

- Memory Grafting improves matched recipients, **not donor equivalence**. Its smaller approximately 0.92B-total/0.29B-active configuration uses 50B training tokens and two frozen 3M-row BF16 memories; the larger setting uses 100B tokens. A100 bank-construction time is not recipient-training time or T4 extraction throughput for a much larger donor.
- Two3M-row2048-wide BF16 banks contain24.576 GB values. Two selected rows are8 KiB/token plus hash/index, projections, gates, misses and actual random-memory costs. Full-bank optimizer state disappears only if values remain frozen.
- Engram/STEM/MoLE involve substantial learning; STAPLE retains dense FFNs. None establishes cheap arbitrary-pretrained conversion here.
- LUT-LLM uses different code rates, training and hardware. FPGA results and extrapolated large-model rates are not Zen2 evidence for the proposed coarse activation code.

[Metadata screen](docs/research/PIPELINE_REASSESSMENT_20261015/ACTIVE_WORK_REVIEW_20261015/METADATA_SCREEN_RESULT_20261015.md):14 requests,1,109,168 retained bytes,7.8 s; no tensor values/headers, remote-code execution or model evaluation. LongCat-Lite revision `b62b68827ead0b7fef3ba98b57f18484acaaec06`; reviewed Transformers implementation `753d61104116eefc8ffc977327b441ee0c8d599f`.

LongCat's approximately 31.407B memory coefficients can select6 KiB BF16 rows/token, but its unchanged shared decoder still has a favorable analytical ledger of **1.523 GB/token before experts, KV and overhead**. Its table does not make its core cheap. Shapes are inferred from pinned code/config and indexed names, not actual tensor headers. An unchanged full-weight port was deprioritized.

Other family metadata, including Granite/GPT-OSS/Switch, live under `docs/research/PIPELINE_REASSESSMENT_20261015/`. Metadata presence is not weight acquisition. No final very-large donor/bank has been selected and acquired for grafting.

## 8. T4 horizon and remaining resource constraints

Months of T4 usage materially change strategy: a512-step failure does not rule out recovery over billions of tokens. Nevertheless:

- T4 device memory is typically16 GiB. Conventional mixed-precision Adam at roughly16 bytes/trainable parameter needs about 14.7 GB for0.92B parameters before activations/workspaces,44.8 GB for2.8B, and1.92TB for120B. Active count is not optimizer-state size.
- Frozen banks can stay in host RAM/sharded files, with only selected batch rows transferred. Local80 GiB host memory must not be assumed beside a remote T4.
- Ninety continuous days contain7.776M seconds. Fifty billion tokens require about 6,430 measured end-to-end tokens/s before interruptions/evaluation. This is a required rate, not measured T4 performance.
- Budget donor acquisition/extraction separately. Offloaded frozen inference still pays transfers, temporary decoding and expert unions. Full120B backprop is not the plan.
- A proposed first milestone is **1B tokens or14 T4-days**, whichever comes first, before conditional extension to months. It is advisory: not a frozen launched protocol, purchased quota, universal minimum or proof of sufficiency.

Before long training, measure throughput/memory, specify resumable checkpoints and budgets, retain general-chat quality and freeze causal capacity controls. Train recipient and adapters jointly; adapter-only failure around a fixed weak core does not settle that hypothesis. Comparative claims against ordinary continued training require a separately trained compute-matched no-memory arm.

## 9. Latest complete recipient tests

### 9.1 Original source and geometry

Local snapshot:

`C:/Users/giosa/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/`

Its `model.safetensors` symlinks into the HF blob cache:988,097,824 bytes, SHA256 `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.

There are290 BF16 tensors. Geometry: D896, FFN4864, L24, query heads14, KV heads2, head dimension64, vocabulary151936, RMS epsilon1e-6, RoPE theta1e6. Source embeddings/head are tied; both deployed archives materialize an explicit head. Canonical EOS IDs151645/151643 are handled.

Original F32 archive: `benchmarks/donor_adaptation/s1/results/native_expert_scaling/meth127_qwen05b_instruct_fp32.bin`,1,976,131,124 bytes, SHA256 `6b2be143303510f15785783542649026b719488407f48b350de67f429e206029`. METH127's original Instruct control is about 18.244 greedy decode/s with scoped C/Torch parity, not a solution.

### 9.2 All-I8 baseline: numerical PASS / cost FAIL

[Primary result](docs/research/PIPELINE_REASSESSMENT_20261015/RECIPIENT_INT8_BASELINE_RESULT_20261015.md). All 24 blocks, GQA/RoPE/KV, dense FFNs and full head execute in C using the original donor engine. R8 codes plus F32 row scales; F32 activations/embeddings/norms/biases. Matrix payload495,785,472 bytes/forward.

All 290 source roles,169 matrix derivations and460 archive extents pass. All 32 saved vocabulary rows match decoded-reference top1; worst L2 5.832934e-6, maximum absolute logit error0.00011158. This does not establish BF16 donor-chat quality; historical Instruct R8 quality failure remains.

| Case | Three native forwards/s | Median | Worst ms/forward | Rate max/min |
| --- | --- | ---: | ---: | ---: |
| English explanation |39.427 /47.387 /48.172 |47.387 |25.363 |1.222 |
| Italian planning |53.039 /58.788 /51.477 |53.039 |19.426 |1.142 |
| Python coding |38.117 /53.529 /48.719 |48.719 |26.235 |1.404 |
| Context512, cost only |43.870 /50.158 /58.075 |50.158 |22.795 |1.324 |
| Context2048, cost only |44.532 /44.800 /35.685 |44.532 |28.023 |1.255 |

Every group fails the frozen envelope. Do not say every individual process was below50/s or failed every separate criterion. There are930 charged forwards/933 emitted IDs; repeats reconcile. Sequence239.937 s after binding verification; largest child RSS5,374,119,936 bytes. Independent readback passes.

Retained directories under `results/native_expert_scaling/`:

| Directory | Outcome |
| --- | --- |
| `recipient_int8_baseline_20261015` | Export fails before model loading: isolated runpy lacks exporter directory on sys.path |
| `recipient_int8_baseline_repair1_20261015` | Export/compile/probe pass; reference fails because nonpersistent RoPE buffer remains meta |
| `recipient_int8_baseline_repair2_20261015` | Same sealed archive/executable adopted; byte-identical probe required; CPU RoPE buffer repaired; numerical PASS and15-process cost FAIL |

Artifact `recipient.i8.bin`:1,040,610,356 bytes, SHA256 `daf94964228636d8255bd20aacc4c88c6dd1e100afd05831c479345d4bf0a280`. Executable SHA256 `aae76015210f2b40cb20c45b6f3fab0441eb09f5c5ca225ab6659cbed3593ae1`.

### 9.3 Mixed format and original reference fault

[Initial mixed attempt](docs/research/PIPELINE_REASSESSMENT_20261015/RECIPIENT_MIXED_BASELINE_RESULT_20261015.md): W4G32 FFN, W8/A8 attention/head, BF16 embedding, F32 norms/biases and full causal F32 KV. AVX2 integer dots; grouped signed-Q8 FFN activations. The head scans every ID, then refines eight candidates using BF16 embedding rows; remaining approximate logits remain eligible in final argmax. It is not ANN or an exact source head.

All 291 packet roles pass source derivation, but the original NumPy-F32-exp reference diverges after position 0: at position 7 norm L2 is0.082598, whole-head L2 is0.092182, top1 native11/reference271. **No cost case ran after that failed audit.**

The [diagnostic](docs/research/PIPELINE_REASSESSMENT_20261015/RECIPIENT_MIXED_DIAGNOSTIC_RESULT_20261015.md) reproduces the exact original native-probe bytes and all four original reference-logit hashes. First float difference is position 0/layer 0 SiLU, with identical gate/up inputs. First activation-code difference is position 1/layer 21/down element 2805:

| Quantity | Native | Original reference |
| --- | ---: | ---: |
| Input |-0.09306909143924713 |-0.09306883066892624 |
| Group scale |0.006418542470782995 |0.006418542470782995 |
| Activation code |-15 |-14 |

The next layer Q output exceeds1e-3 relative error. Tiny primitive differences become discrete quantizer differences and propagate through later layers/KV.

A stored-vector check at all 192 native SiLU sites,933,888 values, matches **bit for bit** with scalar Python binary64 `math.exp`, rounded to F32 after every prescribed F32 operation. The original NumPy F32-exp path differs at814/4864 outputs at the first site. This check executes no model.

The explicit [scalar-reference repair](docs/research/PIPELINE_REASSESSMENT_20261015/RECIPIENT_MIXED_SCALAR_REPAIR_PROTOCOL_20261015.md) wraps the unchanged auditor and changes **only its exponential primitive**. Native executable, packet, quantizer, selected positions and numerical thresholds remain identical. Preserve the original failure. Mathematically similar FP32 nonlinear libraries can follow different activation-quantization trajectories.

### 9.4 Mixed continuation: exact numerical parity / cost FAIL

[Final result](docs/research/PIPELINE_REASSESSMENT_20261015/RECIPIENT_MIXED_SCALAR_RESULT_20261015.md). New probe is byte-identical to original C. All 291 roles are rechecked. Normalized states and all 151936 logits match the repaired explicit reference **exactly** at positions 0,7,15,31, while processing all 32 prefix tokens.

| Case | Three native forwards/s | Median | Worst ms/forward | Rate max/min |
| --- | --- | ---: | ---: | ---: |
| English explanation |40.896 /45.607 /33.380 |40.896 |29.958 |1.3663 |
| Italian planning |50.175 /58.507 /38.112 |50.175 |26.238 |1.5351 |
| Python coding |35.185 /47.627 /42.008 |42.008 |28.421 |1.3536 |
| Context512, cost only |41.040 /36.958 /48.535 |41.040 |27.058 |1.3133 |
| Context2048, cost only |37.552 /22.970 /35.729 |35.729 |43.534 |1.6348 |

All 17 commands exit0. All 15 cost cases forward64 own-history outputs:960 total, repeated IDs identical, no early EOS. Every case group fails both 18 ms and1.10 stability. Continuation251.469 s after initial verification; peak child RSS4,146,196,480 bytes. Independent readback reconciles requests, IDs, step times, masks, seals and cost arithmetic; all 17 owned identities are absent.

Packet630,039,552 bytes includes BF16 embedding. Active matrix payload357,484,032 bytes/forward versus495,785,472 for all-I8, only 1.387x smaller. This grouped-integer/runtime path is not fast enough. The test neither isolates a single cause nor proves all W4 kernels slow. It grants no physical-DRAM or useful-chat admission.

| Object | SHA256 |
| --- | --- |
| Mixed archive |`8bb77b499d76031a128cf8044889a7a772953e96dbe4c65b670e518ae95d00cb` |
| Compiled mixed executable |`df497c03349f752c582e95b0f4d61590c65e26625db276d9d6094c86b1614d42` |
| Native probe across initial/diagnostic/continuation |`9dfdc44494987db9116608d70261af9e6790f8b5856a0214260077dbf8090d12` |
| Final result |`1cabb4d329af9a62f4b48e2734f1cfbdfcc868b7760e47a39a96a891655ea42c` |
| Repaired numerical audit |`aee03b981e47a5337f1645eba679d44a5b8628f2781b02961bebbd89bbfea2ff` |
| Final closure |`a6d3c60053d160f34331df92e78e141475f78ce1d8f1de1280e9f7e5939081fd` |

Retained directories under `results/native_expert_scaling/`: `recipient_mixed_baseline_20261015`, `recipient_mixed_diagnostic_20261015`, `recipient_mixed_scalar_20261015`. Do not overwrite, silently replace artifacts or rerun exclusive commands there.

### 9.5 Interpretation of timing

Both complete baselines use six physical cores and three fresh processes per case, fixed ACTIVE/infinite wait, actual own-history greedy generation and source tokenizer prompts. Context512/2048 fixtures repeat real-token prose and are cost-only. Every non-EOS emitted output, including the final output, is forwarded and charged. EOS scans remain timed without inventing a forward. Load, initialization and prefill are reported separately. Mixed prefill omits unused intermediate heads; its last-token head is full. These are raw charged native-forward rates, not accepted useful-chat throughput. No cause of observed process dispersion was established.

## 10. Code map and binary contracts

Paths below are under `benchmarks/native_expert_scaling/` unless specified.

| File | Purpose |
| --- | --- |
| `recipient_int8_entry.c` | Original-Qwen wrapper over `benchmarks/donor_adaptation/engine/donor_engine.c`; complete forward, geometry/path guards, full decode charging |
| `recipient_int8_baseline.py` | Frozen prepare/run/export/compile/audit/cost controller; explicit adoption after reference fault |
| `recipient_int8_audit.py` | Independent source/R8 archive derivation and decoded HF eager CPU reference |
| `recipient_int8_readback.py` | Independent retained accounting/identity verification, no model |
| `recipient_qwen_entry.c` | Hard-coded complete Qwen mixed runtime, AVX2 dots, GQA/RoPE/KV/head, optional diagnostic guards |
| `recipient_mixed_export.py` | Local source291-role packet, generated catalog, zero padding and hashes |
| `recipient_mixed_baseline.py` | Original mixed controller; original failed attempt unchanged |
| `recipient_mixed_audit.py` | Frozen independent graph with activation quantizers and original NumPy F32 exp |
| `recipient_mixed_diagnose.py` | Exact-trajectory trace replay and first-divergence comparison |
| `recipient_mixed_silu_check.py` | Stored192-site scalar primitive reduction, no model |
| `recipient_mixed_scalar_audit.py` | Explicit scalar-exp wrapper over original auditor |
| `recipient_mixed_scalar_baseline.py` | Sealed continuation, same executable/archive, gate before15 cost jobs |
| `recipient_mixed_scalar_readback.py` | Independent final retained-result verification, no producer parser/model imports |
| `meth388_switch_thread_binding.h` | Windows physical-core discovery and actual affinity readback |

`benchmarks/phase60/engine.c` selects the profiles through `SILICON_QWEN_RECIPIENT_I8` and `SILICON_QWEN_RECIPIENT_ENTRY`. These benchmark entry profiles are not a finished general model-conversion API. `chatbot_compact_entry.c` remains tied to a failed learned conditional-Qwen prerequisite; it is not a qualified cheap recipient.

Mixed generated `RECIPIENT_CATALOG` defines `RcField {name,dtype,rows,cols,offset,bytes}`,291 roles, archive size/SHA. Fields start at64-byte-aligned offsets, padding zero, first offset0. Vectors `[1,N]`, matrices `[out,in]`:

- F32=1: exact widened norms/biases.
- BF16=2: original embedding.
- W8=3: **each row** stores F32 scale then signed I8 codes; maxabs/127, nearest-even[-127,127], zero row scale/codes0.
- W4G32=4: each32-input group stores FP16-rounded maxabs/7 scale plus16 bytes; low nibbles positions 0..15, high nibbles16..31; stored code=signed code+8. Quantization divides by decoded FP16 scale, clips[-7,7]. Zero block scale0/codes0x88.

W8 activation scales are whole-vector; W4 scales per32 group. Zero activation uses scale1/codes0. Integer sums are safe for declared geometry, including the unsigned-nibble AVX2 pair-sum bound.

**The old R8 wire is different:** it stores all matrix I8 codes followed by row scales, not interleaved rows. Headers, embedding types, activation semantics and request protocols differ. Do not substitute one serializer/loader for the other.

Mixed request `QWCR0001` carries input count/max-new/IDs; `QWCO0001` returns count/stop/elapsed/IDs. Probe `QWCP0001` carries prefix/selected positions/IDs; `QWCT0001` returns normalized states and full logits. Diagnostic trace records u32 position/layer/tag/count plus F32 arrays. Trace head-layer 24 is not another transformer block. The unused draft `recipient_mixed_readback.py` was removed; the final checker is `recipient_mixed_scalar_readback.py`.

## 11. Source assets, sealed data and historical traps

### Falcon source assets

Adopted432-case source coverage and full-width bridge remain reusable controls. Public prompt messages may be retokenized for a new donor, but **Falcon logits, states and hidden vectors are not another donor's labels or memory values**. The selected memory method needs canonical short-context construction and its own key/vector semantics; existing own-history captures are not already a grafting bank.

Reserved evaluation remains untouched:104 newer reserved cases plus64 older reserved cases; LongAlign was deferred. Do not open them for exploratory recipient choice or bank-ID design. Used FIT/DEV and public engineering prompts are not fresh blind quality evidence.

### Historical positive results and their limits

- METH253: approximately 9.317 ms for24 native Qwen FFNs, not complete generation.
- METH260/263: favorable BPB within scoped compressed Qwen recipes, not large-family capacity proof.
- METH265:72 responses pass generation-health checks, not unrestricted semantic quality.
- METH266: PIQA70.18% versus70.24%, one question among1838.
- METH267: blind severe errors16 versus donor15 fails the frozen rule. Nearness is not PASS; donor quality itself had limitations.
- METH510: Switch head full I8 scan plus top8 F32 refinement about 0.995 ms at V32128/D768; plain I8 about 0.847 ms. It is not ANN and not a universal sub-ms head for Qwen or Falcon dimensions.
- METH511: Switch7.4B warm bounded infilling89.98 generated IDs/s,52.56 prose IDs/s, scoped top1 around98.7–98.9%, edit error0.0344. One frozen economic criterion fails. It is encoder-decoder infilling, not an admitted causal chatbot or very-large pipeline.
- Legacy68.52/s one-byte Qwen belongs to a **base** model, not the original Instruct source tested now.

Use [catalog decisions](docs/research/NATIVE_EXPERT_SCALING_20260925/DECISIONS_FROM_CATALOG_20261014.md) and the [brief](docs/research/PIPELINE_REASSESSMENT_20261015/CARVING_RESEARCH_BRIEF_20261015.md) to reach primary reports.

### Falcon mechanisms and overinterpretations to avoid

The full-width bridge retains known-corpus behavior:14/16 tasks, DEV KL0.02156/disagreement4.8847%, own-answer followup and audited recurrent outputs. Source GPU attention/FFN/head and dense contextual generators remain; this is a quality control, not a cheap complete CPU artifact.

FFN remainder oracle counts 512/2048/32 at sites0/12/23 yield local DEV errors about 0.06217/0.04871/0.07281, **after computing all source neuron functions**. This is not a deployable cheap selector or global sparsity optimum. Fixed subsets/PCA/derivative-sensitive selectors produce no admitted FIT-selected replacement. A failed output-bank image oracle cannot be repaired by training a router inside that same image.

The claimed13.8 ms full contextual-generator shortcut was not qualified. Actual complete Falcon I4 generator medians are about 22.806/23.840 ms; generator plus internal SSM readout24.125/22.442 ms, already over 20 ms before other organs. Internal readout is not the vocabulary head. Local16.8 ms S6/LUT evidence and unrelated organ timings cannot be composed into a whole-model result.

Historical exact speculative verification exists, but expert-union/cold/accepted-token costs do not provide free source-weight amortization. Consult its actual evidence before proposing it as a complete escape.

## 12. Concrete plan after the next user resumes

### Step1: reestablish state

Read owner instructions, this handoff and current goal state; inspect Git and explicitly owned receipts. Preserve unrelated changes. Current outcome is closed numerical-PASS/cost-FAIL. No unfinished audit justifies replaying the latest tests unchanged.

### Step2: pin and qualify a materially cheaper complete recipient

Current bounded candidate: SmolLM2-135M-Instruct, **only an online processor for large-donor transfer**. Public config/card were inspected, but no immutable source acquisition, weights, C execution or quality admission exists. Freeze revision/license/tokenizer/config/EOS and byte/hash/resource limits before downloading weights.

Analytical counts include scales and full W8 head:

| Core | L/D/FFN | Query/KV heads | Vocab | All-W8 matrix MB/step | W4-FFN/W8-rest MB/step | Distinct F32 KV MB at2048 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Qwen |24/896/4864 |14/2 |151936 |495.785 |357.484 |50.332 |
| SmolLM2-360M |32/960/2560 |15/5 |49152 |363.061 |259.064 |167.772 |
| SmolLM2-135M |30/576/1536 |9/3 |49152 |135.299 |100.024 |94.372 |

These are not runtime/DRAM predictions. Smol135 uses biasless Llama-style attention/SwiGLU, RMS epsilon1e-5 and RoPE theta100000. Do not merely edit dimensions in the Qwen-biased serializer. Its weak source-reported reasoning/knowledge is a real initial-competence risk. Smol360 has substantially greater KV work at2048; smaller matrix payload alone does not select it.

Implement a complete model-aware C/export path with provenance, full head, GQA/RoPE/KV, tokenizer/EOS and own-history generation. Independently verify decoded serialization and execution, then complete-context physical cost. Reuse understood operators without importing failed learned banks. Final useful source-quality needs its own fresh evaluation.

No unchanged Qwen thread/wait grid, assumed W4 speedup or months-long recovery of a physically unqualified core. A future different serious kernel/representation is allowed if it changes a substantive complete-cost premise.

### Step3: define donor-memory transfer and a bounded source pilot

After a viable online core exists, select the actual large-family donor and canonical construction. Specify tokenization mapping, history keys, vector layers/precision, ambiguous keys, misses, bounded fallback, IDs/masses/adapters and every online cost. A full120B download is unnecessary merely to start a mechanism falsifier; final large-family evidence cannot be replaced by a small donor indefinitely.

Measure offline extraction throughput/resources before scaling rows. Freeze donor-gap tasks where donor materially exceeds recipient, plus general chat and held-out/no-hit formulations before training results. Existing Falcon source supervision is not automatically the new donor's canonical bank.

### Step4: joint recovery and causal capacity proof

Jointly train recipient/adapters/masses, initially freezing bank values, over substantial general recovery plus bounded donor supervision and interaction recovery. Define a measured feasible T4 milestone, checkpoints and stop conditions.

Require matched-frequency value shuffling, removal, masked distinct subsets at equal lookup count and increasing distinct banks at matched active work. Inference controls show causal use; a separately trained compute-matched no-memory arm supports comparative training claims. A proposed20% donor-gap recovery threshold is an example to freeze, not an existing gate, admission or final preservation standard.

Extend toward months only with causal donor contribution, growing useful distinct capacity, general-chat retention and credible CPU/training economics. Qualify>=50 useful accepted CPU tokens/s on the **same artifact**. Reconsider native-MoE carving after a bounded serious failure rather than indefinitely refining bookkeeping.

## 13. Runtime environment and safe operations

Hardware: Ryzen5 3600X,6 physical/12 SMT cores,80 GiB RAM; local RTX3060 12 GiB for bounded conversion/learning, not final inference. No T4 is allocated. Respect other user applications and announce local GPU budget/use before launching it.

Python: `C:/Users/giosa/AppData/Local/Programs/Python/Python312/python.exe`.

CPU scripts use `-I -S -B -X utf8` and explicitly insert `D:/_THINGS/Progetti/SiliconLLM/results/native_expert_scaling/chatbot_source_runtime/site`, a selected local site/junction view. Recorded versions: Torch2.6.0+cu124, Transformers5.13.1, NumPy2.4.6, psutil7.2.2. Version names are not substitutes for binding hashes. CPU jobs disable CUDA, use offline HF settings and explicit OpenMP/BLAS policy. Recent recipient work uses no network/GPU.

Compiler identity comes from `docs/research/NATIVE_EXPERT_SCALING_20260925/meth374_switch_physical_workers_contract_result.json`, compile argv[0], with adjacent `libomp.dll`: LLVM-MinGW MSVCRT. Do not silently substitute PATH compiler/math runtime. Latest mixed flags: `-O3 -std=c11 -march=x86-64-v3 -ffp-contract=off -fopenmp -Wl,--no-insert-timestamp`, generated catalog macro, `-lbcrypt -lm`.

Recent recipient runs use six physical workers, logical0,2,4,6,8,10, with actual start/end affinity readback. Older falsifiers often use three physical cores; do not conflate those bounds. ACTIVE/infinite is the fixed recent profile, not a universally optimal setting.

Use hidden noninteractive Windows subprocesses, exclusive namespaces/logs, limits and owned PID/create-time. Never kill by executable name or stale PID. Do not terminate unrelated games/publishers/helpers. Read the relevant historical GPU launcher/holder guards before changing them. Use literal paths and avoid recursive cleanup of research artifacts.

Read-only orientation commands:

```powershell
Set-Location 'D:/_THINGS/Progetti/SiliconLLM'
git status --short --untracked-files=no
git log -12 --oneline
Get-Content 'docs/research/PIPELINE_REASSESSMENT_20261015/RECIPIENT_MIXED_SCALAR_RESULT_20261015.md'
Get-Content 'docs/research/PIPELINE_REASSESSMENT_20261015/recipient_mixed_scalar_closure_20261015.json'
rg --files docs/research/PIPELINE_REASSESSMENT_20261015
```

Historical exact CLI arguments live in binding/terminal JSON. **Do not rerun `run` in completed directories**: exclusive started/output guards intentionally reject it. Any future justified replay needs a new prospective namespace/budget. Reading this file does not authorize a replay.

## 14. Git, artifacts and documentation closure

Only pre-existing tracked edits at handoff start:

- `benchmarks/donor_adaptation/configs/_manifest.json`;
- `benchmarks/donor_adaptation/density/build_document_holdout.py`;
- `docs/research/RESEARCH_INDEX.md`.

They belong to unrelated work and must not be staged or changed by the handoff. Many unrelated untracked artifacts exist too. No global staging/reset/clean, broad deletion or broad restoration.

| Commit | Reproducibility checkpoint |
| --- | --- |
| `5b52aab` | Freeze complete original all-I8 baseline |
| `2fa53a9` | Isolated exporter import repair |
| `b0116a8` | CPU rotary-buffer repair and sealed adoption |
| `45e865c` | All-I8 numerical-PASS/cost-FAIL closure |
| `4b78ac7` | Complete mixed encoding/kernel freeze; source for its retained executable |
| `954f15d` | Preserve initial mixed numerical rejection |
| `1d35abe` | Freeze exact-trajectory diagnostic |
| `52dda0a` | First divergence and scalar primitive repair/check |
| `ea96e58` | Freeze sealed scalar-reference continuation |
| `7b93d10` | Exact mixed parity and independent complete cost rejection |

The handoff documentation commit comes later; use Git log to identify it. Large binaries, traces and caches remain local. Reports and selected exact JSON evidence are committed under `docs/research/PIPELINE_REASSESSMENT_20261015/`.

Exact serialization is not donor quality; stored metric readback is not model execution; logical bytes are not measured DRAM; a proposal is not a run. Follow all custody and gate scopes before promoting any result.

The [handoff verification receipt](docs/research/NATIVE_EXPERT_SCALING_20260925/HANDOFF_SNAPSHOT_20261015/handoff_verification.json) records document/link/snapshot checks and the observed paused goal state. It is a documentation check, not a new scientific audit.

This handoff launches no numerical experiment and changes no experimental result or threshold. The session remains closed, goal paused/incomplete, and substantive work awaits the next user's resumption.
