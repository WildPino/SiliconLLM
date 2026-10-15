# Carving review: compile additive activation codes, then test the information loss

15 October 2026. Advisory only. No experiment, weight acquisition or training was performed for this review.

**Resource update:** the owner subsequently clarified that months of T4 usage are acceptable. The final section supersedes the initial priority order and short conversion horizon below. The initial analysis is preserved because its representation and inference-cost cautions remain relevant.

## Recommendation and limits

The next useful question is whether source-sensitive activation coding can replace matrix scans with additive table reads while retaining useful behavior. I would authorize one bounded real-FFN representation experiment before another large-family acquisition or CPU implementation. I would not represent this as an established route to 50 useful tokens/s.

The missing premise is unusually concrete: **can a contextual source operator tolerate the very low activation description rate required by a RAM-feasible, fast table representation?** Existing results do not answer this. They also provide substantial reasons to expect failure. If this particular experiment fails, stop this recipe rather than increasing tables, precision, training and exception paths until its original economic advantage disappears.

One native-MoE family containing genuinely very large models is sufficient. Dense conversion, a transformer-to-SSM conversion and success across unrelated families are not prerequisites. A Falcon site is proposed only as an inexpensive scientific falsifier using retained operands; its success would not establish transfer to GPT-OSS or qualify a small donor as the requested endpoint.

## What the failures actually establish

The [research brief](CARVING_RESEARCH_BRIEF_20261015.md) records the evidence and primary result links. The unchanged GPT-OSS protected core exceeds 20 ms in its tested three-worker implementation even with head, experts and attention free. [Uniform Q/O rank256](FIRST_ATTENTION_RANK_RESULT_20261015.md) has independently replayed DEV branch error about 0.293. [One unchanged source-head group](FIRST_ATTENTION_SOURCE_GROUP_RESULT_20261015.md), selected with an ideal signed mass, still has approximately 0.50 error. A better router cannot repair the latter fixed representation.

These failures justify stopping those recipes. They do not prove that useful chat requires 1% internal branch fidelity, that every transformed representation fails, or that six-core execution has a universal timing bound. Conversely, relaxing the internal gate does not establish useful chat: a prospectively frozen complete-model quality test is required.

The [Falcon remainder result](../NATIVE_EXPERT_SCALING_20260925/FFN_REMAINDER_ORACLE_RESULT_20261013.md) establishes conditional coverage when all source functions are evaluated to obtain selection scores. At site12 its selected width2048 gives DEV mean error0.0487, while width512 gives0.2257. Its shared affine contribution is not free. Cheap projected selectors have not recovered the favorable coverage.

The [conditional-output experiment](../NATIVE_EXPERT_SCALING_20260925/CONDITIONAL_OUTPUT_RESULT_20261013.md) rejects its frozen small collection of affine images even with target-aware selection. METH222/225 reject their fitted shared/conditional recipes; METH230 stopped on degenerate fitting data and supplies no broad nonlinear impossibility result. METH177/178/179 warn that adding banks without causal useful diversity is insufficient. The inventory of609 reports is not609 independently validated premises for this review.

## Hypothesis 1: source-derived additive activation-code tables

For a source projection W of shape m by d, partition x into C=d/g groups. For each group learn K prototypes p[c,k] of dimension g. Compile

`T[c,k,:] = W[:, group(c)] @ p[c,k]`

and evaluate

`y_hat = bias + sum_c T[c, encoder_c(x_group(c)), :]`.

This is exactly W applied to the reconstructed activation, before table quantization. Every group contributes; this is not choosing one source head or one low-rank output image. With g=32 and K=32, storage contains approximately as many table coefficients as W, but inference reads one thirty-second of them. A contiguous output vector is stored for each selected group/code. The source matrix can then be removed from the inference artifact; retaining a full-matrix residual correction would forfeit the central saving.

The encoder returns IDs from x before table reads. Start with exhaustive nearest-prototype search, costing roughly dK coordinate comparisons/distance arithmetic per encoded vector. This is cheap relative to a large matrix but must be measured. Each additive contribution has mass1; these masses must not be normalized across groups. Native expert IDs and their softmax masses remain separate, with the original router charged. Share the incoming activation codebook across experts in a layer to avoid encoding cost proportional to the number of stored experts. An expert's down-projection encoder operates on its own computed nonlinear intermediate, not the teacher's intermediate.

Small input-only decision trees could reduce search cost, but that is a subsequent encoding hypothesis. Original MADDNESS uses16-leaf structures; a32-code tree is a new design, not an inherited verified kernel. The [MADDNESS paper](https://proceedings.mlr.press/v139/blalock21a.html) provides the learned product-code/table mechanism. [LUT-NN](https://arxiv.org/abs/2302.03213) provides differentiable centroid learning and CPU implementation evidence, not a very-large autoregressive conversion result.

Ordinary input-space k-means is the wrong sole objective. For a linear projection the relevant error is `(x-q(x))^T W^T W (x-q(x))`, including cross-group terms. A block-diagonal output-weighted metric is an initialization, not the complete objective. Calibrate gate/up, source nonlinearity and down together against the real FFN output. Otherwise errors induced before the nonlinearity invalidate the down-table calibration distribution. Keep the source weights fixed in the first experiment; only the small prototype sets are fitted. No target/output value may enter the deployed encoder.

### Why this is different, and why it may still fail

For C groups there are K^C code combinations. For fixed-mass tables their affine span has dimension at most C(K-1), plus an offset; for d2880/g32/K32 this is at most2790, rather than a mandatory global rank256. This does not establish full rank, accurate interpolation or good generalization. The previous one-image oracle does not bound this compositional class. The FFN nonlinearity further couples the independently addressed contributions.

The severe risk is rate-distortion: K32/g32 provides only5/32=0.15625 discrete bits per activation coordinate. For an illustrative independent isotropic Gaussian source, the ideal squared-error rate-distortion expression `D/variance = 2^(-2R)` is about0.805 at this rate. Actual activations are neither assumed Gaussian nor evaluated in input MSE, so this is a warning, not a bound on model quality. Output sensitivity, correlations and repeated structure must do almost all the work.

Increasing K256 with g32 increases the table pool eightfold. INT8 tables double the Q4 pool. These are not harmless accuracy adjustments on an80GiB machine.

[LUT-LLM v2](https://arxiv.org/html/2511.06174v2), with [official code](https://github.com/LUT-FPGA/LUT-LLM), is positive adjacent evidence: it converts Qwen3-1.7B using activation/weight co-quantization and INT8 tables. Its vector length2 and64 activation centroids give3 activation bits/dimension,19.2 times our proposed discrete rate. It reports approximately10 A100-hours conversion; MMLU-Pro changes33.1 to30.8. Deployment is an AMD V80 FPGA; its32B performance comparison is extrapolated. It does not establish cheap120B conversion, Zen2 speed, or our much coarser code. Its co-quantized implementation also streams weight indices; it is not identical to directly selecting one long compiled output vector per activation group.

T-MAC-style exact weight-bit lookup is a different operation: it accelerates evaluation while still reading the weight codes. Activation-product tables precompute responses to input prototypes and introduce a new input approximation. Neither result can be substituted for the other.

## Coupled large-family ledger: necessary, not sufficient

The following is an analytical GPT-OSS-120B illustration, not a measured artifact. It uses the pinned source geometry described in the brief:36 layers, D2880, expert width2880,128 experts/top4,64 query heads of width64,8 KV heads, vocabulary201088. Use g32/K32, Q4_K tables with144 bytes per256 output coefficients, padded **output-vector** rows. This differs from pretending every table is ideal4bit. All source experts remain stored; all selected experts execute.

| Component | Resident payload, decimal GB | Selected/read payload per token, decimal MB |
| --- | ---: | ---: |
| All expert tables, fused gate/up and down | 66.886 | 65.318 |
| QKV/O tables | 0.553 | 17.294 |
| Complete vocabulary projection tables | 0.326 | 10.187 |
| Original F32 expert routers | 0.053 | 53.084 |
| BF16 token embedding | 1.158 | 0.0058 |
| Logical BF16 KV at context2048,18 full and18 window128 layers | 0.080 | approximately80.216 |

The table pool plus embedding/router is approximately68.98 GB, or64.24 GiB, before codebooks, KV, allocation overhead and workspaces. The logical per-token total is approximately226 MB including the stated KV reads. Expert output vectors pad5760 to5888 for fused gate/up and2880 to3072 for down. Original source matrices must not remain resident in addition to this pool. These numbers are conditional on uniformly adequate Q4 tables and shared bounded-size encoders; neither is established.

At an assumed sustainable40 GB/s,226 MB would consume5.65 ms of bandwidth time. This is not a latency prediction: scattered group/code addresses, paging, cache lines, intermediate buffers, execution overlap and source quantization scales matter. The output-vector layout gives long contiguous reads per selection, but jumps between vectors. Actual DRAM must be measured.

Each projection still needs `m*(d/g)` decoded additions, with block-specific Q4 scales/minima applied before summation. There are roughly160 million selected table entries per token across the listed expert/core/head projections, before padding. A naive output accumulator updated after every group causes roughly8 bytes of F32 read/write traffic per entry, about1.3 GB of **cache/work-buffer traffic**, not necessarily DRAM. Output tiling can keep partial sums in registers/L1; it introduces loop/layout tradeoffs and may repeat metadata access. Integer sums across independently scaled blocks cannot simply be treated as one exact integer accumulation.

Full causal attention is retained. At the stated context, QK and AV together require320,864,256 multiply-adds: `18*(2048+128)*2*64*64`. This includes both full and windowed layers; the earlier302 million estimate omitted the latter. Softmax, RoPE, sinks, norms, nonlinearities and writes are additional. The F32 routers still scan53 MB. Encode-search work, head accumulation and sampling are additional. Even this optimistic representation does not prove the20 ms budget; it merely removes the immediate multi-gigabyte matrix-stream obstruction. Speculation is not assumed, so accepted-token rate equals generated-token rate only after the same artifact passes quality gates.

### Conversion and capacity economics

Compile one layer/expert at a time from source weights, emit packed tables, and release temporary matrices. Do not allocate a full120B F32 master/gradient/Adam state. For fixed source W, learning only prototype sets is much cheaper in parameter storage; obtaining representative activations from rare experts can still dominate conversion. Layerwise calibration must eventually use candidate-distributed inputs or repair drift. The existing432-case Falcon capture cannot certify120B knowledge or calibrate another tokenizer/model.

This transformation retains a distinct source-derived table for every expert. That is a mechanism for retaining capacity, not evidence that the capacity remains useful. Matched-cost bank permutation, removal and rare-domain tests must cause meaningful quality losses before claiming useful large conditional capacity. Whole-model repair at LUT-LLM's reported scale cannot be extrapolated linearly to a3060, let alone promised to fit a sustainable conversion budget.

## Hypothesis 2, lower priority: a bounded shared value plus coded residual

Represent each group as `x_c = mu_c + rho_c p[c,id_c] + error_c`. Compile the common value once; compute a group amplitude rho from x before table access. Then `W x` is approximated by `W mu + sum_c rho_c T[c,id_c]`. This supplies continuous masses and may remove an easy radial/common component before coding directions. It costs norm/amplitude evaluation, scaling and additional calibration. Its descriptor is no longer only5 bits per group.

A global mean requires only one shared output vector. Token/ngram-conditioned common values require their own explicitly bounded addressable pool; per-token, per-layer, per-expert full-output caching can explode storage. A learned dense shared projection is not free and can recreate the protected-core obstruction. Source nonlinearities do not commute with mean subtraction, so fitting must remain a whole-FFN objective.

This refinement is plausible only if the residual has demonstrably lower source-relevant entropy. Earlier common-cell and conditional-map failures are direct reasons not to assume that. I would not launch it in parallel or add it silently after seeing DEV failure. It is a separate next hypothesis only if the first experiment identifies radial/common error as dominant on FIT.

## One bounded falsifier to freeze before any run

Use the retained real Falcon site12 FFN, authentic own-history operands and source coefficients bound by the [remainder protocol](../NATIVE_EXPERT_SCALING_20260925/FFN_REMAINDER_ORACLE_PROTOCOL_20261013.md). This is the difficult middle site, not the easy late-site example. Reuse its exact source activation/scaling semantics and all output coordinates. This tests a cross-operator composition with information actually retained; no additional large-model download is needed.

Freeze one g32/K32, fixed-mass1 recipe. Share the incoming code among gate/up projections; separately encode the candidate-generated nonlinear intermediate for down. Initialize prototypes on FIT only with source-output-weighted grouping, then fit only prototypes for512 fixed updates with a documented soft-backward/hard-forward rule, fixed seed and final checkpoint. Use complete FFN output loss. Preserve an unmodified source control. The encoder is input-only exhaustive nearest-prototype search; no teacher-selected IDs or test-time residual solve.

First replay **F32 tables**, to isolate activation coding. At this single site the F32 table pool is approximately113 MB for three D2048/F4608 projections; diagnostic storage need not satisfy a hypothetical whole-model F32 deployment budget. If warranted, replay the same frozen codes/prototypes with actual packed Q4_K tables. Do not refit after viewing DEV or compensate by retaining source matrices. This is one representation experiment with two precision controls, not a hyperparameter grid.

The consumed DEV cohort is a development screen. Record the old local mean0.10/worst0.20 thresholds for comparability, but do not make them universal usefulness gates. Before running, freeze a single-site substitution evaluation in the existing full Falcon recipient on a separate prospectively fixed prompt panel, without opening RESERVED. Suggested promotion gates are mean teacher-forced NLL increase at most0.02 nats/token, no more than0.10 in any prespecified domain, and no newly severe failure on the fixed generation rubric. These are proposed engineering admission margins, not established scientific constants or a substitute for the owner's final quality criteria. If no independent panel is available, report diagnostic evidence only.

Always evaluate the candidate's own FFN composition and, for generation, its own history. A local error above0.10 does not automatically reject a candidate that passes the frozen whole-model usefulness screen; a locally small error does not waive that screen. If the F32 version fails the functional screen, stop before Q4/kernel work. If F32 passes and Q4 fails, reject this memory-feasible table format; do not silently double the120B pool. If both pass, the next authorization would be a real table-kernel physical screen and then broader substitution, not full120B admission.

Suggested hard caps:2 GPU-hours,3 hours wall time,9 GiB GPU allocated/10.5 GiB reserved,8 GiB host working memory,2 GiB new output. Calibration alone should use substantially less; these caps include a bounded existing-Falcon substitution check. No numerical work is authorized by this advisory. No result is a certified best-code oracle: independent nearest-centroid choices, gradient fitting and coordinate refinement are finite recipes with interacting errors.

## Decision

Do not abandon an entire native family because one rank recipe failed, and do not continue optimizing an unchanged representation already missing both quality and cost. Test the smallest new premise that could change the economics: source-sensitive additive activation coding of a complete real operator at a fixed RAM-compatible code rate. A negative result would remove this specific cheap compilation route. A positive result would justify physical implementation and complete-model evaluation, while leaving large-family knowledge retention, conversion coverage and50 useful CPU tokens/s unresolved.

## Revised priority with months of single-T4 conversion available

This clarification changes my recommendation materially. I would now prioritize **a pretrained compact recipient, frozen donor-derived conditional memory, and prolonged joint recovery of the recipient and memory adapters**. Longer-trained fine-grained source-MoE carving is second. The additive-code experiment becomes an optional inexpensive side investigation, not a prerequisite for the project or my preferred main investment. The amplitude variant above belongs to that third direction, not an additional parallel project.

The reason is not that more time guarantees quality. It permits learning a new division of labor between a small contextual processor and a large cold bank, without requiring an extremely coarse quantizer to reproduce every contextual activation immediately. A512-update local imitation failure is weak evidence against that different training objective. Equally, months of wall time do not supply16GiB with120B optimizer capacity, make donor labels free, or establish useful capacity from bank size.

### First choice: recover a compact processor around frozen donor memory

For an input history h, use a deterministic suffix index `i = lookup(h)` and a frozen donor-derived memory vector `v_i`. A compact recipient computes a context-dependent mass and update, for example

`a = sigmoid(dot(norm(Q h), norm(K v_i)) / sqrt(d))`

`h_next = recipient_block(h) + a * V v_i`.

The ID is available before accessing the selected vector. The mass requires only that vector and the charged small projection/gating maps; it is computed before expensive downstream recipient work. No scan of the complete bank or donor is required. A bounded hierarchy or exact longest-suffix hash supplies IDs; misses use an explicit bounded fallback. The large donor runs offline to construct values, not once per recipient training token.

The recipient, adapters and masses should be trained together, starting from a pretrained initialization where possible. Freeze the large external bank initially. This is not merely adapter fitting around an immutable weak recipient, and not repeating the failed compact student on76,512 labels. Use a substantial general corpus for recovery, a bounded donor-supervised subset, and instruction/interaction recovery after the language objective. Avoid expecting sparse instruction examples to reconstruct broad pretrained competence.

The primary [Memory Grafting study](https://arxiv.org/html/2605.20948v1) now deserves serious consideration rather than dismissal for duration. Its smaller configuration trains a0.92B-total/0.29B-active recipient for50B tokens, with two frozen3M-row BF16 memories totaling about24.6GB for2048-wide vectors. The larger setting trains for100B tokens. It reports improving matched recipient baselines, not reproducing its donor's full capability. These are recipient-training experiments, not evidence that a frozen-recipient adapter-only shortcut works. The paper's measured construction cost on A100 does not predict a16GiB T4 with an offloaded much larger donor.

**Large-family path:** a native donor such as Qwen3-Next80B can construct the bank offline while its expensive mandatory projections are absent from the deployed recipient. The METH397 nonrouted-core obstruction therefore no longer applies to each generated recipient token; it remains a conversion cost. The existing pinned donor metadata are a starting point, not authorization to acquire that model. This method can also use GPT-OSS120B, but source acquisition, offloaded teacher throughput and useful transfer need separate qualification. A small donor is only a mechanism pilot. The final bank must demonstrably transfer useful information from the selected very-large donor.

For two3M-row banks, BF16 value payload is `2 * 3M * donor_width * 2` bytes:24.576GB at width2048, or34.56GB at2880. Reading one row at each of two sites touches8KiB or11.25KiB of value payload per token, plus hash/index traffic, projection weights, scales if quantized, gates and any fallback. Index storage and DRAM misses must be measured. At width2048 and recipient width768, separate key/value projections at two sites contain about6.29M coefficients: approximately3.54MB in unpadded Q4_K, not zero. The donor matrices are absent from the deployment artifact.

The recipient remains the dominant online cost. A sensible preliminary design allowance is at most150MB of packed projection/router/head payload per token, with explicit room for adapters, KV/state, attention arithmetic, norms and frontend inside20ms. This is a working allowance, not a measured guarantee: previous screens show that even modest payloads can miss the target. Charge the vocabulary head in that allowance. A complete C run on the selected recipient geometry must establish the necessary margin before spending months on recovery. Native source width/depth are not protected requirements; useful donor-derived behavior is.

**The central scientific risk is donor-relative capacity retention.** Short-ngram values may preserve lexical associations while losing contextual reasoning, instruction following and long-tail knowledge. A24GB table is not24GB of useful knowledge, and cannot be called120B retained capacity by counting its rows. Prespecify tasks on which the donor materially exceeds the recipient. Require bank-enabled recovery there, improvement when the bank gains distinct entries at matched active cost, and degradation under matched-frequency value shuffling/removal. Include held-out formulations and no-hit contexts. General chat quality must pass on the same complete CPU artifact. A small average benchmark gain alone would not complete the owner's goal.

### What months buy, and what16GiB still prevents

For90 days of continuous use there are7.776 million seconds. Training10B/50B/100B tokens requires approximately1,286/6,430/12,860 tokens/s respectively. Over180 days those rates halve. These are required observed end-to-end rates, not claimed T4 performance; interruptions, evaluation and donor preparation reduce usable time.

The rough transformer training proxy `6 * active_parameters * training_tokens` makes50B tokens at0.15B active approximately4.5e19 FLOPs, and at0.30B active approximately9e19 FLOPs. Over90 days these require5.8 or11.6 sustained TFLOP/s before omitted attention, embedding/head details, optimizer and transfer costs. Sparse execution and small batches may be inefficient. Thus months can make a compact recovery program plausible, but50B tokens is neither automatically feasible nor required as a universal minimum. Fit the schedule to measured progress per GPU-hour.

Optimizer memory depends on **total trainable** parameters, not active parameters. Conventional mixed-precision Adam can require roughly16 bytes per trainable parameter before activations/workspaces:0.92B parameters already imply about14.7GB;2.8B imply44.8GB;120B imply1.92TB. Even the paper's smaller recipient therefore needs a deliberate memory plan on16GiB: bounded adapters/frozen banks, an alternative optimizer, offloaded optimizer states, or a smaller trainable core. Each changes the recipe or throughput. Activation checkpointing does not remove optimizer state. The local80GiB host memory must not be assumed to exist beside a cloud T4; host RAM, storage and transfer access must be confirmed separately.

Frozen memory rows can stay in host RAM or a sharded file, with only batch-selected rows copied to GPU. At two2048-wide BF16 rows per token,6,430 tokens/s would imply about53MB/s of raw selected-vector transfer, before duplication, batching and index overhead. That is qualitatively different from paging gigabytes of source expert weights every training step. Precomputing corpus memory IDs and grouping transfers can help without changing the function. Persistent random disk access or remote network access could still dominate. There is no full-bank gradient/optimizer allocation in this initial plan.

A large teacher still needs to construct the memory. With short canonical ngrams, layerwise frozen inference, batching and streamed checkpoints avoid full-model backprop, but sparse expert unions, CPU-to-GPU traffic and temporary decoded weights must be charged. The existing Falcon captures are not already canonical short-ngram vectors from another donor. Budget and benchmark offline extraction separately before committing to tens of millions of rows. Continued training on ordinary text does not require donor labels at every token; a bounded donor-labeled recovery/evaluation corpus remains necessary to measure actual transfer.

### Second choice: train a finer-grained source-MoE representation

More time makes jointly learned selectors, masses, head reorganization and staged recovery scientifically legitimate. Retain all source experts, partition their FFN channels into physically compatible children, and progressively reduce active children while optimizing end-to-end language/distillation objectives. A compact hierarchical router can produce parent/child IDs and normalized masses from the incoming state before loading children. It must not evaluate full source gates to select them. Frozen source banks with bounded trainable routers/adapters are much more memory-feasible than full expert updates; they also impose a real recovery limitation.

This route still needs a different attention/core and head, not only finer experts. For illustration, replacing64 query heads by8 learned heads while retaining8 KV heads reduces ideal QKV/O coefficient work to about119MB of Q4_K payload across36 layers before padding. It reduces the earlier320.9M attention MACs to about40.1M. This is a new trained representation, not selecting one unchanged source group. Selecting one256-wide child per original top4 parent would reduce ideal expert work by approximately11.25 times, leaving about179MB in Q4_K before final-child padding and selectors. Together with53MB routers, KV and the unreduced head, that is still not an admitted20ms solution. Output retrieval or another trained head transformation remains necessary.

The [MoH paper](https://arxiv.org/abs/2410.11842) supports continued-training conversion to learned head selection, but its reported pretrained Llama result uses75% of heads, not the12.5% illustrative target above. [Sparse Upcycling](https://arxiv.org/abs/2212.05055) supports reuse of pretrained weights, but also spends substantial continued-training compute and does not itself reduce the activated original function enough for this CPU target. Neither licenses an extrapolation to extreme carving. Previous source-head and rank failures remain relevant initialization warnings; whole-model recovery can change the function, but must prove usefulness afterward.

Full120B optimization remains the wrong resource assumption. Layerwise or alternating recovery can train bounded modules and revisit earlier ones, but a successful local fit may fail after downstream drift. End-to-end adapter training through frozen, offloaded experts still requires their forward/backward-input execution and can be transfer-bound. Months help this cost; they do not erase it. This is why I rank memory grafting first under the updated constraint.

### Revised first commitment and stopping rules

I would replace the immediate codebook-first commitment with a **time-bounded joint-recipient recovery pilot**, after qualifying the complete recipient cost envelope. Freeze the selected recipient architecture, a small donor-derived memory built from canonical contexts, the general recovery corpus, donor-gap task panel and bank controls before training. Use a1B-token or14-T4-day initial cap, whichever occurs first, with predeclared checkpoints. This is a first milestone toward a possible multi-month program, not a claim that1B tokens settles recoverability. Existing Falcon assets can supply a low-cost mechanism pilot; a positive result still requires the chosen large family's real memory.

Train the recipient and adapters, not adapters alone. Evaluate the same checkpoint with its real bank, matched-frequency shuffled values, and masked entry subsets at equal lookup count; these inference controls test causal bank use without silently granting another full training budget. For a claim that grafting outperforms equally funded recovery without memory, a separately trained compute-matched control must also be funded and included. Predeclare a meaningful donor-gap recovery threshold, for example recovering at least20% of the donor-minus-recipient score gap on the fixed targeted panel while retaining the general chat gate; the exact task/scoring margin must be frozen before results. Exclude tasks with negligible or negative donor gap from that particular calculation, not from general quality reporting.

Continue toward the multi-month budget only if useful bank contribution grows with recovery, larger distinct subsets help at fixed active cost, and measured training/token-extraction rates make the remaining program feasible. Stop this recipe for bank collapse, no meaningful causal donor-memory benefit across the frozen milestones, loss of chat usefulness, failure of the complete CPU budget, or an economic projection outside the accepted months. A failed frozen-recipient or short adapter-only experiment is not a valid stop rule for the jointly recovered architecture. A failed1B-token milestone is a bounded investment decision, not proof that50B tokens could never work.

The revised recommendation remains conditional: **months make a trained compact processor plus large frozen donor memory a credible research program, not an established solution.** Its smallest missing premise is meaningful donor-relative knowledge transfer into cheaply addressed memory. The earlier additive-code route instead asks for exceptionally low contextual activation entropy. I now prefer testing the former because the published mechanism is closer to the desired asymmetry between large stored capacity and small active work, and the owner has made its recovery horizon admissible.
