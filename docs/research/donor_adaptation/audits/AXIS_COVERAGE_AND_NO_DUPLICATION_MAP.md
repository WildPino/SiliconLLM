# SiliconLLM research axes — coverage and no-duplication map

**Snapshot:** 2026-09-15. **Purpose:** answer “has this already been tried?” before opening a
brief, writing a runner, or spending CPU/GPU time. This is an index of project evidence, not a new
scientific result. The chronological source of truth for donor adaptation remains `../INDEX.md`; rate
arithmetic remains `../SPEED_LEDGER.md`; exact experiment status remains in the cited probe/result.
For donor selection, use `TARGET_DONOR_LOCAL_INVENTORY.md` for repository evidence and
`TARGET_DONOR_DECISION.md` for the frozen Qwen2.5-Coder-14B/Qwen3-30B-A3B-Base disposition.

## 1. How to read this map

Evidence classes are intentionally different:

- **CLOSED:** the registered question has a deciding measurement. Reopen only by changing the
  estimand or a named scope coordinate (donor, model nature, scale, format, engine path, domain).
- **MEASURED-SCOPED:** a useful number exists, but only inside the stated scope. It is not a scale
  law or a result for another lineage.
- **OPEN / CONDITIONAL:** no deciding measurement exists, or the next action is armed only after a
  prior gate passes.
- **PARKED:** valid unfinished work outside the current donor-adaptation priority.
- **PRIOR-ART / DESIGN ONLY:** literature, arithmetic or a proposal; never cite it as a project
  measurement.
- **VOID:** apparatus or controls failed before the registered estimand was measured. A repaired run
  may be owed, but the void numbers are not evidence.

When status text disagrees across documents, use this precedence:

1. newest canonical probe/result and its audit;
2. `.planning/codebase/RESEARCH_CATALOG.md` and this map;
3. dedicated run records such as `docs/PHASE64_RUNG1_RUN_RECORD.md`;
4. `README.md` / `HANDOFF.md` summaries;
5. briefs, proposals, old plans and archived documents.

The lower item may describe a plan that a later item completed, voided, superseded or parked.

## 2. The four research lineages — do not merge their claims

| lineage | object actually studied | strongest surviving result | present status | canonical navigation |
|---|---|---|---|---|
| Frozen / reservoir lineage, phases 1–54 | random or fixed silicon-native substrate, learned readouts, local plasticity, O(1) superposition memory, on-policy/DAgger and n-gram assists | local/medium-range similarity is cheap; long-range relations and selection are not recoverable from the frozen substrate. Static per-step and O(1)-superposition routes closed; BPE-1024 produced recognizable but structurally weak English | **CLOSED as a route to the target LLM**; keep its engine and instrumentation lessons | `HANDOFF.md` §§Phase 47–54; `archive/docs/phases/` |
| Trained CPU-native lineage, phases 55–64 | selective SSM + local attention, dReLU/ternary MLP, granular MoE, thin retrieval tier, trained from scratch | sandbox engine and properties validated; Phase-64 MVE completed; S0 30M reached 0.9277 code-val BPB, but its registered offline property gates were never read | **PARKED** while donor adaptation consumes quota; no quality-at-scale claim | `README.md`; `docs/PHASE64_*`; `docs/PHASE64_RUNG1_RUN_RECORD.md` |
| Pretrained-donor adaptation, P/R/T/D/F/E/H series | Qwen2.5/Qwen2.5-Coder Transformer weights converted, factorized, carved and/or healed for `donor_engine.c` | one-byte 7B score is nearly fp32; post-hoc carve/rank remain the structural wall; H2I v4 trained the mixed object but closed `SCORE-ONLY` | **ACTIVE**; chronology through E66 plus terminal H1/H2I | `../INDEX.md`; `../SPEED_LEDGER.md`; `.planning/codebase/RESEARCH_CATALOG.md` |
| External research / architecture exploration | SSM+retrieval, CPU bandwidth, linear attention, donor methods, CPU-native alternatives | candidate mechanisms, counterevidence and source trails | **PRIOR-ART / DESIGN ONLY** unless a row above cites a local probe | `docs/research/*Research*`; `docs/research/ATTENTION_LINEARISATION_PRIOR_ART.md`; `../prior_art/`; `docs/research/LLM_Reimagined_CPU_Native_Exploration_20260626/` |

The most dangerous false equivalence is “ternary works” versus “ternary fails.” It works when a
small native model is **trained into** its ternary MLP; naive/post-hoc ternarization of a pretrained
Transformer destroys ranking. Those are different objects and both measurements stand.

## 3. Coverage matrix by attackable axis

| axis | variants actually attacked | decisive evidence and scope | status now | do not repeat / legitimate reopening |
|---|---|---|---|---|
| **Model nature / recurrence** | frozen reservoir; trained selective SSM; dense Transformer donor; SSM+retrieval and hybrid alternatives in literature | phases 47–54 close frozen-substrate long-range semantics; phases 55–64 validate a trained SSM only at sandbox/S0 scale; donor E-series executes real Qwen Transformer weights | **CLOSED** frozen route; **MEASURED-SCOPED** native SSM and donor Transformer; hybrids mostly **PRIOR-ART** | Do not propose another readout/local-memory trick on the same frozen state. A new recurrent/hybrid architecture needs a new trained object and cannot inherit donor or sandbox quality |
| **Family and domain** | TinyStories/prose; code; logs; Qwen2.5 general and Coder donors | native properties transfer to code; logs are a measured LZ-favouring boundary. Donor quality uses pinned slices and Qwen scales; E15/E66 establish the real Coder-7B endpoints | **MEASURED-SCOPED** | Change family/domain only with a reason and new controls. Never compare BPB across tokenizers/slices as if paired; Phase-64 §4.2 explicitly forbids two tempting comparisons |
| **Parameter scale** | native ~8–22M, MVE/S0 ~30M; donors 0.5/1.5/3/7B; synthetic/constructed ~10B speed cells | E60/E62/E66 measure one-byte fidelity through 7B but E62 is non-monotone. E36/E39/E40 measure 10B **shape economics**, not pretrained quality. E63 constructs an exact 10B-format cell, with clean rate still owed | **OPEN at the joint 10B pretrained quality+rate claim** | Do not fit a scale exponent from 0.5/1.5/3B or promote synthetic 10B weights. A 10B quality claim requires target-scale pretrained/trained weights; a rate claim requires `G-E63d` or a separately registered successor |
| **Width, depth, heads, KV and rank shape** | depth reduction/redistribution; D/F/L shape budgets; GQA/KV/context; low-rank q/o/head; R128 attention; residual-layer rank | E24/E35 price depth; E34 prices attention+head floor; E21/E25/E29/E39/E40 price low-rank shape and kernels; E65 measures r/D=1/32 quality cost at D=1536 | mostly **CLOSED per registered shape**, no width law | Do not extrapolate E65 to D=4096 or call R128 free. Reopen with an explicit target shape and a target-quality measurement, not parameter-count equivalence alone |
| **Vocabulary, tokenizer, embedding and head** | BPE-1024/2048 native; Qwen vocab; tied/untied and fp32/ternary head; vocabulary-size speed/token-worth | Phase 50 makes unit choice load-bearing; E17 shows the donor head is not the sole ranking mechanism; E18 organ ladder; E43 measures token units but its speed half is permanently void | **MEASURED-SCOPED** | Do not rerun E43 as the same speed question. A smaller-vocab claim must report bytes/text as well as tok/s and re-establish quality; changing tokenizer breaks direct BPB/rank comparability |
| **Context and explicit memory** | SSM state; O(1) superposition; local attention; IVF-PQ/two-stage recall; decode context slopes | O(1) superposition/addressing closed negative; Phase 56 validates learned recall representation and two-stage cost. 128K query **cost** is measured, while direct recall quality is at shorter footing plus proxy/argument. E46/E48 price Transformer context slope | cost **MEASURED**; fused target-length benefit **OPEN** | Do not quote “128K recall validated” without separating cost from quality. Reopen only with target-length generated-rollout recall or an integrated model benefit gate |
| **Weight precision / representation** | fp32; naive and activation-aware ternary; 4-bit packed ternary storage; 1 byte/weight int8/R8; mixed tagged-v2 | T1/T2/T2b/E12/E15/E16/E20 close post-hoc ternary rule search for this donor; E60/E62/E66 show the one-byte rung is faithful through 7B; E63 proves tagged 10B arithmetic/path exact | ternary PTQ **CLOSED**; one-byte fidelity **CLOSED through 7B**; mixed 10B quality **OPEN** | No more ternary-rule sweep absent a genuinely new estimand. Do not call 4-bit storage “1.58 bits on disk.” Change donor, quantizer or trained format explicitly to reopen E66 |
| **Activation precision** | fp32 activations; int8 block/LUT activation quantization; trained fast-LUT path | E14/E32 price BPB cost; E59 gains ~1.25–1.33× but changes 42–74% of trained tokens | **CLOSED for blind enablement** | Never enable `--lutblk` from speed alone. Any successor must pair same-format quality/rank with speed and must identify activation quantization separately from weight format |
| **Folding, scales and normalization** | layer RMSNorm fold; final/head fold; row/group scales; magnitude matching at upcycle | E2 confirms layer fold; `--fold all` hurts when final gain is folded into ternary head. T3 isolates the useful fold from harmful rotation. Phase-64 ε-identity/magnitude matching records a residual D→E shock | **CLOSED per construction**; upcycle transition mechanism **MEASURED-SCOPED** | Reuse the adopted layer fold. Do not assume a ternary-specific fold helps int8 (E66 says its contribution is tiny/opposite in that construction) or hide an architecture switch behind “same checkpoint” |
| **Rotation, basis change and permutation** | dense orthogonal/Hadamard rotation; weight sparsifying basis; coactivation permutation; layout transpose | D2 says affordable orthogonal bases do not create useful donor sparsity; T3 rotations hurt ternarization; D0/D0c show finer coactivation granularity helps but random null helps more; E13's tile-major transpose is a kernel layout, not a model transformation | model-transform route **CLOSED / negative**; layout **ADOPTED** | Do not rerun generic Hadamard/orthogonal rotation. A new transform needs a different mechanism and must keep model-space transforms separate from bit-exact storage/layout transforms |
| **Low-rank factorization** | SVD/weight-only; activation-weighted q/o and head; residual layers; engine factored matvec; rank moved across organs | R2A/E21 find activation-weighted q/o geometry; E25 proves engine path/parity; E29 narrows which layers rank; E39/E40 buy speed; E65 shows the fast rank fraction consumes 51.66% of dense→chance gap at D=1536 | speed path **CLOSED/measured**; post-hoc quality at target width **OPEN but constrained** | Do not repeat rank solely because it speeds the shape. A new rank scheme must beat E65 on quality at a declared fraction and preserve ranking, or be trained rather than post-hoc |
| **Sparsity, pruning and carve granularity** | magnitude/block pruning; coactivation/null; neuron/expert granularity; static and oracle carve; 1.17% FFN; coarse locality | D1/D0/D0c/E19/E27/E33/E37/E38 exhaust post-hoc sparsity at the target active budget; E38's perfect per-token oracle is already above chance. E64 shows carve is even dearer on int8 | post-hoc target-budget route **CLOSED** | Do not search another router/selector on the same frozen carved FFN: the oracle ceiling failed. Legitimate reopening is training the structure (H1/H2I), changing the active budget, or changing the model/estimand |
| **Routing and selection** | learned gate predictors; static/oracle routers; trained routers; granular MoE | F1 characterizes predictability; E23/E37 price real router and show it is load-bearing; H1 S3 closes ternary 8-layer joint trainability; H2I v4 learns held-out score and beats STATIC routing, but fails its strict combined rank gate | post-hoc **CLOSED**; ternary 8-layer trainability **CLOSED/MEASURED**; one-byte 8-layer score+rank cell **CLOSED `SCORE-ONLY`** | Do not treat teacher-forced router behavior as free-running success. Do not call H1 a post-hoc-carve revival or a one-byte/rank success. H2I's BPB/router win is not an export, rank, rate, scale or target-donor license |
| **Organ allocation / composition** | FFN, q/k/v/o, head, norms; per-organ ternary/fp32; cheap components combined; parameter mass moved from FFN to attention | T2b/E18/E21/E22/E34/E36/E39/E40 price organs and composition. Cheap isolated changes can compose super-additively; attention+head set a floor; R128 shape can be fast with live synthetic FFN | economics **CLOSED for current engine/shape family** | Do not add isolated deltas to predict a mixed artifact. Export and measure the exact composition, with one same-binary control and explicit `CONFIG` |
| **Training regime** | readout-only, on-policy/DAgger, CE, cross-tokenizer span-KD, STE/QAT, alpha-QAT, MoE upcycle, reverse-KL, resume/DDP | native MVE: CE beats off-domain span-KD; QAT shock recovers; A→F and resume pass. H0/H1 show donor format/carve can train. H2I v4 shows trained one-byte score/routing but misses its combined rank gate. Phase-64 S0 final number lacks its offline gate verdict | donor H2I **CLOSED `SCORE-ONLY`**; native S0 **PARKED** | Do not launch old H2T or rerun H2I as written. Do not generalize off-domain KD failure to on-domain KD. A later checkpoint is not automatically better: use the predeclared adjudication, no checkpoint selection |
| **Training order / inversion** | QAT before/after structural switch; magnitude-matched dense→MoE surgery; reverse-KL final stage; CE versus KD | Phase-64 records an arm-independent D→E discontinuity and the ε-identity law; S0 applied alpha-QAT then upcycle and gained most in F, but no causal arm isolates why | **MEASURED, mechanism OPEN/PARKED** | Do not infer that reverse-KL caused the S0 gain or that ordering is solved. Reopen only as a one-variable order A/B with transition-BPB continuity |
| **Conversion and export** | PyTorch↔engine parity; >2 GB loading; RMSNorm fold; packed/int8/tagged-v2; per-matrix/per-layer kinds | E1/E2 establish parity; E63 adds exact mixed container mechanics; E66 runs a real 7B one-byte model. Tagged-v2 can represent H2I, but current exporter cannot emit the exact mixed trained object | base path **ADOPTED**; H2I seam **CLOSED (combined gate failed)** | Do not implement H2I export: v4 did not pass score+rank. Do not patch E63d's hash-pinned binary |
| **Kernel, packing and memory layout** | nibble/trit packing; streamed vs gathered rows; tile-major LUT; factored/int8 matvec; FMA chains; cache/L3/DRAM regimes | P1/P2/E9–E13/E25/E26/E28/E30/E31/E58/E60/E61 quantify exact paths. Gathered bytes are not streamed bytes; one host has format-specific bandwidths; FMA chain is not the binder | major paths **CLOSED/measured** | Do not transfer a bandwidth denominator across formats/layouts. A kernel change requires parity plus same-format trained-model fidelity, not only synthetic GEMV speed |
| **Attention, softmax and context kernel** | serial reduction; independent accumulators/AVX4; GQA/KV traffic; real-exp/libm/errno; full attention lever stack | E4 and E48–E54 isolate and fix the earlier disabled/serial path; E40 explicitly exhausts the registered attention levers | **CLOSED for “next easy attention lever”** | Read E40 before proposing another accumulator/KV/softmax pass. Reopen only for a changed algorithm, hardware or correctness contract, and preserve `CONFIG` provenance |
| **Threads, occupancy and measurement environment** | output parallelism; OpenMP region/wake policy; clean/busy boxes; page-cache/occupancy guards; paired order reversal | native Phase 63 adopts bit-identical threading. E44/E52 establish interval/occupancy discipline. E63d preflight refusal measured no rate | protocol **ADOPTED**; absolute E63d rate **OPEN** | Never count the refusal as a timing attempt. Do not pollute the quiet window. Use the pinned runner only when all 45 occupancy samples pass; report intervals, not point-estimate precision |
| **Tokens per weight pass / decoding** | block verify; state look-ahead; speculative/self-drafting MTP in prior art; vocabulary token-worth | native block-verify is exact but does not pay at 8.3M; state carries ~1 reliable future step. External MTP gains are prior art, not a local target-scale result | local sandbox question **CLOSED**; target-scale composition **UNMEASURED** | Do not quote literature MTP multipliers as SiliconLLM speed. Reopen only when weight streaming dominates enough to amortize verification and include acceptance-rate measurement |
| **Metrics and adjudication** | BPB; chance distance; free-running and teacher-forced rank; full trajectories; parity/logits; tok/s intervals; context/token value | E16–E20 prove BPB and greedy rank can disagree or invert; E66 passes score but misses 150/160 rank; H2I freezes both rank modes and score. E43 proves a token is not a fixed amount of text | protocol **ADOPTED** | No quality promotion from BPB alone; no speed promotion from kernel rate alone; no free-running rescue by teacher-forced rank. Score gate runs before H2I rank, exactly once |

## 4. Shape and scale ledger — what “large” currently means

| object | real trained/pretrained weights? | quality measured? | engine arithmetic/parity? | rate measured? | licensed statement |
|---|---:|---:|---:|---:|---|
| Native ~8–22M sandbox | yes, trained from scratch | yes | yes | yes, cache-resident | validates architecture properties and engine kernels, not billion-scale streaming |
| Phase-64 MVE/S0 ~30M | yes, trained from scratch | MVE gates yes; S0 BPB yes but property gates absent | not exact S0 export | GPU training throughput, not target CPU decode | machinery works; S0's 0.9277 is a measurement without its registered verdict |
| Qwen 0.5/1.5/3B donor | yes, pretrained | yes across multiple formats | yes | selected format rates, scoped | anchors format/rule trends; E62 forbids a smooth scale law |
| Qwen2.5-Coder-7B donor | yes, pretrained | fp32, ternary and one-byte score/rank | yes | no valid one-byte target-rate result | one-byte fidelity survives; strict rank bar does not |
| Qwen2.5-Coder-14B candidate | no local weights; official metadata plus derived arithmetic only | no | family-level format compatibility only; no artifact parity | no | dense one-byte upper bound 2.958 tok/s; do not download before a separately gated head+body transformation |
| Qwen3-30B-A3B-Base candidate | no local weights; official metadata plus derived arithmetic only | no | current carved path is not exact Qwen3 MoE | no | one-byte upper bound 13.607 tok/s; even zero-expert fixed floor caps at 33.657, so defer exact support until head+attention also change |
| E36/E39/E40 ~10B shapes | no: synthetic/noise weights | no donor-quality result | yes for their engine paths | yes, bands/controls as documented | the CPU can execute a suitable 10B **shape** at/above the target rate |
| E63 carved-int8 ~10B artifact | constructed exact-format artifact, not a pretrained 10B quality object | no | yes | clean A10B `G-E63d` 49.37 tok/s, CI [49.18, 50.49]; `G-E63e` ratio 0.9952 | synthetic-shape rate gate closed; no trained 10B quality or strict 50-tok/s pass; E61c/3B ancillary sweeps incomplete |
| H2I 8-layer mixed donor object | donor weights plus trained factors/router/carved layers | Phase A and Phase B score measured; strict combined rank failed | container capable, exporter gap audited but closed by verdict | no | trained one-byte carve improves held-out BPB and routing, but is terminal `SCORE-ONLY`; no export/scale claim |
| H4 rank48 ternary q/o, 1.5B donor | yes, pretrained body plus trained low-rank attention factors | CPU fp32 terminal 1.1537375 BPB; 66/160 teacher-forced, 6/160 free; `TRAINING_HELPS`/`BEATS_QO96`/`BEATS_QO192` true, `GENERATOR_PARTIAL` false | no engine export | no | training recovers score/rank beyond QO192 post-hoc but generation stays AT-FLOOR; any Stage B requires a new brief, with no 10B/format/rate transfer |

## 5. Apparent contradictions and measured inversions

These are not invitations to average the two sides; they are boundaries that future work must carry.

| apparent contradiction / inversion | correct resolution |
|---|---|
| Native ternary costs little, while donor ternary collapses | QAT/from-scratch ternary MLP and post-hoc Transformer ternarization are different estimands. Training into the format survives; post-hoc conversion does not preserve donor rank |
| One-byte conversion is nearly lossless, but one-byte carve is worse than ternary carve | E66 isolates the uncarved format; E64 isolates post-hoc structure on that format. The byte conversion is not the remaining quality wall; selection/carving is |
| 10B exceeds 100 tok/s, but no working 10B is demonstrated | E40 is a real rate on a synthetic shape. It proves hardware/shape feasibility, not pretrained quality. The joint claim needs both target weights and target rate |
| Lower BPB can have floor-level greedy agreement | E16–E20 measure discontinuous rank and exposure drift. BPB, teacher-forced rank, free rank and trajectories are separate gates |
| Fewer active weights should be faster, yet gather can lose | E26/E31 show gathered traffic and per-expert overhead; the cost depends on layout/granularity, not byte count alone |
| Attention was the bottleneck, then attention levers were exhausted | earlier runs used or priced different kernels. E49/E50 correct disabled fast-attention provenance; E40 is the terminal registered sweep |
| Recall is “validated at 128K,” yet target-length recall is open | 128K **CPU query cost** is measured. Direct quality is established at shorter context plus proxies/structure; integrated target-length benefit remains unmeasured |
| Phase-64 S0 improves to 0.9277 but “has no verdict” | the scalar is valid, but the pre-registered matched-dense/property gates were never run. A number without its comparator cannot decide the rung |
| Cross-tokenizer KD failed, yet KD is not globally dead | the measured MVE teacher/domain was off-domain and CE won. The recipe correctly flips to CE-primary; only on-domain KD remains a legitimate challenger |
| Format quality varies non-monotonically with scale | E62 measures exactly that. Do not smooth, fit an exponent or extrapolate to 10B |

## 6. Unmeasured cells that are real — and their priority

| priority | cell | why still open | action rule |
|---:|---|---|---|
| 1 | `G-E63d` clean absolute 10B carved-int8 rate | exact path exists, but preflight refused before any cell | use the pinned runner in a genuinely quiet 10–15 minute local window; do not substitute interim ratios |
| parked | Phase-64 S0 offline property gates | checkpoints exist; GPU training is complete, but dense/sparsity/router/recall/QAT gates were not read | CPU-only if the native branch resumes; do not spend donor quota or treat 0.9277 as the missing verdict |
| prior-art gap | direct integrated target-length recall benefit | cost and representation pieces exist separately | requires a new fused-model quality experiment, not another ANN microbench |
| future target | joint pretrained ~10B quality + >=50/100 tok/s | every component claim currently misses either real target weights, target quality or clean rate | compose only after trained structure survives; preregister exact artifact, engine and both metrics |

H1 closes its ternary 8-layer cell, and H2I v4 closes the corresponding one-byte score+rank cell
as `SCORE-ONLY`: score/routing learn, strict rank does not. Neither licenses an extra GPU launch,
a post-hoc carve rerun, a one-byte/rank promotion or H2I export. The remaining immediate measured
work is the quiet local `G-E63d` measurement.

## 7. Mandatory no-duplication checklist for every new brief

Before assigning a new ID, write all of these into the brief:

1. **Nearest prior cell:** experiment ID and exact cited artifact.
2. **Changed coordinate:** donor/family, model nature, scale/shape, format, training regime, domain,
   engine path or estimand. “Try another value” is insufficient when an oracle or terminal sweep
   already closed the mechanism.
3. **Why the prior result does not answer it:** one falsifiable sentence.
4. **Paired controls:** same binary, same slice, same ordering policy, explicit engine `CONFIG`.
5. **Metric separation:** quality, rank, trajectory, arithmetic parity and rate are not proxies for
   one another.
6. **Scale discipline:** mark every number as measured, derived or literature; never promote a
   synthetic shape or a non-monotone ladder into a scale law.
7. **Void rule:** failed controls/apparatus produce no scientific cell. Repair narrowly and preserve
   the void record.
8. **Stop rule:** state what failure prevents the next export, rank run, training stage or scale-up.

If the changed coordinate cannot be named, the experiment is a duplicate and should not run.
