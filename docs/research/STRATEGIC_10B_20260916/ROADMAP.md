# Strategic roadmap: pretrained ~10B, 50–100 tok/s, AVX2 CPU

**Rolling status through 23 September 2026:** [continuation status](STATUS_20260923.md). The [17 September `engine.c` scope decision](STATUS_20260917.md) remains valid. The text below remains the 16 September design snapshot; use the rolling status and [research control index](../RESEARCH_INDEX.md) for execution state and no-duplication decisions.

**Current execution head (24 September):** the accepted donor is exact through
layer 1; layer 2 first fails at attention output, and whole-KV cross-input has
localized sufficiency to the compact KV row. The separately frozen
[KV-partition protocol](../donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_KV_PARTITION_CROSS_INPUT_PROTOCOL_20260924.md)
has now closed `LAYER2_LATENT_VALUE_RESIDUAL_SUFFICIENT`: reference-query with
C prefix/reference positional tail fails, while reference prefix/C tail
passes. See the canonical
[result](../donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_KV_PARTITION_CROSS_INPUT_RESULT_20260924.md).
The only next fidelity coordinate is production of the 512-value
`kv_cmpr-2` prefix: immutable projected input versus KV RMSNorm operator;
no new donor/reference graph is justified. That distinction is now frozen by
the [KV RMSNorm cross-input protocol](../donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_KV_RMSNORM_CROSS_INPUT_PROTOCOL_20260924.md);
its [model-free apparatus](../donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_KV_RMSNORM_CROSS_INPUT_APPARATUS_RESULT_20260924.md)
was qualified. The sole scientific invocation produced every output but is raw
VOID because descriptive prefix metrics used a `[8,6144]` reshape on `[8,512]`.
The [offline-recovery protocol](../donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_KV_RMSNORM_OFFLINE_RECOVERY_PROTOCOL_20260924.md)
has closed [the result](../donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_KV_RMSNORM_CROSS_INPUT_RESULT_20260924.md)
as `LAYER2_PROJECTED_KV_PREFIX_RESIDUAL_SUFFICIENT_RECOVERED` with zero new
model/graph execution. KV RMSNorm is byte-exact; split only the upstream
`attn_norm-2` input from the KV-A projection. The
[KV-A projection protocol](../donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_KV_A_PROJECTION_CROSS_INPUT_PROTOCOL_20260924.md)
now freezes that zero-graph successor; qualify model-free next.

**Date: 16 September 2026. Analysis and design only. No code, tests, benchmarks, weight downloads, or training performed.** Reading baseline: HEAD `f1f9cfd59d34071a004248bc04c9d56e6038c596`. Register of sources, scope, and results: [EVIDENCE.md](EVIDENCE.md). The L/W codes refer to that register; further primary sources are linked in the text.

## Decision summary

**The most plausible path is to retain pretraining and change the minimum necessary, preferably starting from an already sparse/recurrent donor.** The conversion must be trained on the geometry that is actually executable: per-organ precision, mixer, rank, shared path, and routing. If retaining the already studied dense donor, the next informative step is a joint-adaptation pilot, not another composition of separately trained components. Distillation toward the native SSM remains a third strategy, consistent with the engine but more economically uncertain.

The repository currently contains no joint demonstration of pretrained ~10B, quality within +0.02 BPB, and 50 tok/s. E63 measures 49.37 tok/s on synthetic mixed-format weights; E66 demonstrates a real 7B donor with nearly unchanged BPB but fails its generative-agreement gate. H4/H2I show that training recovers score; H5 shows that assembling their checkpoints does not preserve that recovery. The dominant constraint is therefore **quality at the geometry that fits within 10–20 ms/token**, not merely the ability to move a 10B file. [L10–L18]

I propose three roadmaps with separate budgets: R1 retain a sparse/hybrid donor; R2 jointly transform a dense donor; R3 distill into a native SSM with conditional capacity. Before committing weeks of T4 time, a common gate must establish model identity, effective bytes, engine semantics, quality, and throughput units. An initial **24–72 effective hours of T4×2**, after documentary screening, can decide between directions; it is not a promise of complete conversion. Later tranches are conditional on new results, without automatically reopening closed experiments.

Success requires **the same artifact** for quality and speed, genuinely learned distinct parameters, declared context, and no silent substitution of the target with a small student or duplicated weights. 50 tok/s is the first objective; 100 remains a second frontier, to be pursued only while preserving the quality gate.

## 1. What changes after reading the repository

| Item | Current evidence | Consequence |
|---|---|---|
| Architecture | `phase60/engine.c` is an SSM with compile-time dimensions; `donor_adaptation/engine/donor_engine.c` is a separate Transformer runtime | “Runs on our C” and “is converted to the native SSM” are two different milestones [L01,L02] |
| Packing | The historical default uses 4 bit/weight; P1 has already added `--pack nibble` at **2 bit/weight** | The next lossless 2→1.6 bit step offers at most 1.25× on code traffic, not 2.5× [L07] |
| Quality | +0.00004 BPB concerns sandbox inference optimizations | It is not the cost of ternarizing a pretrained model; nor the cost of the native recipe [L06] |
| Cache | t6 exploits aggregate LLC; gradual spill and pollution | 16 MB is not a physical prohibition on running at 10B; it is a hot-pool objective [L03] |
| 10B performance | E36 ~50; E40 R128 above100; E63 mixed 49.37, CI[49.18,50.49] | All synthetic artifacts: no quality to inherit [L08–L10] |
| Donor7B | E66 A2 +0.000378 BPB; 137/160 agreements, gate150 failed | A faithful precision exists; it is not yet the quality+rate target [L12] |
| Structural compression | Aggressive rank, carve, and post-hoc composition degrade | Joint learning or a different donor nature is required [L13–L18] |

The native loader also reads complete fp32 copies of the MLPs and int8 control representations before the packed kernels. At 10B, the fp32 copy alone would be about 40 GB: **a compact ternary file does not imply compact RAM use**. A deployment export must separate the debug reference from the executable representation, without eliminating the reference as a verification tool. This is a future engineering requirement, not code written here. [L01]

I do not treat the old index as evidence superior to more recent probes. For example, T3 formally remains `VOID`; I do not turn it into universal evidence against Hadamard. Likewise, E38/E67 are not oracles of the best possible selector. [L19,L20,L16]

## 2. Quantitative contract and Pareto frontier

### 2.1 Definition of success

Establish before any new test: checkpoint and revision; separate **distinct**, total/active/stored parameter counts; tokenizer; identical UTF-8 corpus and immutable split; primary code domain plus general-regression suite; 2K/8K/32K contexts and, separately, any128K; batch1; greedy decode/hygiene consistent with the branch, with output at least256 tokens per prompt. These are proposed protocols, not existing results.

Proposed final gate: ΔBPB relative to the teacher on the same corpus ≤+0.01 preferred, **upper CI95% limit≤+0.02** mandatory; bootstrap by document, not correlated tokens. Do not transfer σ≈0.005 from the sandbox to a donor or different domains. Add executable-code and completion tasks, proposed relative regression≤2% on primary tasks with stated intervals, syntax errors, and rollout degeneration. Tasks and thresholds must be fixed before the numbers. Greedy/rank agreement is diagnostic and, if preregistered, has its own gates; it does not equal general capability. Its historical failure remains a failure. [L06,L12,L18]

For speed: end-to-end tok/s of **emitted and accepted tokens**, UTF-8 bytes/s and time on the same quantity of text; separate TTFT/prefill; p50/p95 latency, peak RAM, working set, context, threads, frequencies, occupancy, compiler, and flags. Propose lower-CI95%≥50, then≥100: a central value49.37 with a CI crossing50 does not pass. No credit for reducing the vocabulary while producing more tokens for the same text.

### 2.2 Equation to use

For each organ i and format f, record weight bytes read B_i, reuse, effective FLOPs/operations, and the rate of **the same kernel/format/layout**. A conservative initial model is:

\[
t_{AR}\approx\sum_i\max\{B_i/\beta_i,\ O_i/\pi_i\}+t_{routing}+t_{sync}+t_{state}+t_{sampling}.
\]

State/glue terms must be excluded from the sum if already incorporated in a kernel. Overlap among organs can make the sum conservative; do not add an integrated measurement and its DRAM cost a second time. `β_i` is not peak memory bandwidth: it includes granularity, cache, code decoding, and contention. Use the roofline to reject implausible proposals, not to promote them.

**Dense 10B case:** if every weight is read once per token, weights alone require 10 GB at int8, 5 GB at 4 bit, 2.5 GB at 2 bit, and 2 GB at 1.6 bit. Even at an ideal 40 GB/s, the 1.6-bit case stops at 20 tok/s before any other cost; at 28 GB/s, at 14. Packing alone does not produce 50–100 with 10B dense active weights. The exits are: less weight actually touched, exact reuse across multiple tokens, or a different architecture/capacity. This is traffic-conditional arithmetic, not a benchmark.

**Design budget, not a measurement:** reserving 30% of time for compute/glue/margin leaves 14 ms at 50 tok/s and 7 ms at 100 for streaming. The budget is shared by experts, shared projections, and head.

| Assumed rate | Byte/token at50 | Byte/token at100 |
|---|---:|---:|
| 28 GB/s |392 MB|196 MB|
| 40 GB/s |560 MB|280 MB|

At 28 GB/s this corresponds, if all bytes were weights, to 392M/196M int8 weights or 1.96B/0.98B at 0.2 B/weight. Metadata, padding, scales, read amplification, and state consume part of the budget. With the historical expert kernel at 4.2 GB/s, the same 14 ms buys only 58.8 MB; with 17 GB/s kernel-pure, 238 MB. Do not choose all best assumptions simultaneously. [L03]

### 2.3 Two pools, three footprints, and a hidden cost

Separate **total file/RAM**, **hot set reused between tokens**, and **traffic/token**. Not all experts visited throughout the forward need reside inL3 at the same time. The local working set and declared resident pool must meet the budget, or spills must be paid for. The model must include SSM state, buffers, lookup tables, scales, and cache pollution.

For a homogeneous gated MoE:

\[
P_{experts}=3LDhE,\quad P_{selected}=3LDhk,\quad B_{selected}=b_eP_{selected}+B_{scales}+B_{padding}.
\]

The gate and part of the up work must be computed before knowing what to skip: 92% hidden sparsity **does not mean** 92% of all reads avoided. The sandbox 2.12× already results from this asymmetry; do not automatically multiply it by routing and other sparsities. [L06]

**Router:** a flat fp32 router costs `4LDE = 4P_experts/(3h)` bytes. For 10B experts and h=128: 104.2 MB of router; h=1024: 13.0 MB, before backbone/head. Thus, increasing E at constant active cost is not free indefinitely. A hierarchical/product-key/factorized router is a new architecture to train and verify; lookups must remain cached and selected experts must be streamed in blocks.

**Head:** `4DV` bytes; D=4096, V=32768 ⇒ 536.9 MB fp32. Weight tying removes the second **stored** matrix, not the head product read for every token. A low-rank head with D=4096, V=32768, r=256 has `4r(D+V)=37.75MB`, still above 16 MB. The rank needed to fit in 16 MiB would be approximately ≤113 without including anything else: this is a severe quality restriction, not an automatic recommendation.

**Purely design-level SSM example:** D=512, L=16, E=400, h=1024 ⇒ 10.066B expert-only parameters. With top-2, 50.33M expert weights/token: 25.17 MB at 0.5 B/weight or 10.07 MB at 0.2 B/weight. But projections, using the historical extrapolation, are about 117 MB, V=32768 head about 67 MB, router 13 MB: the hot pool already does not fit. The low active cost is credible as accounting, **top-2/400 quality is entirely unmeasured** and time cannot be derived from the MLPs alone. [L03; derivation]

**Training coverage:** uniform routing provides about `T·k/E` assignments per expert/layer over T tokens. Reducing k/E reduces signal per expert; with skew, some do not learn. The 10B capacity must be useful and learned, not merely allocated.

### 2.4 Multi-constraint objective

Optimize the tuple `(ΔBPB, task quality, t_token, RAM, GPU-hours, risk)` subject to executable-format constraints. Every Pareto point must have an identifiable checkpoint. Do not add improvements from incompatible checkpoints; H5 concretely demonstrates why. For50, seek margin (design16ms, verify≤20); for100, design8ms and verify≤10. These are proposed margins, not changes to historical gates. [L15]

## A. Three distinct roadmaps

### R1 — Retain an already sparse/hybrid pretrained model

**Guiding idea:** avoid having to teach a dense FFN from scratch to function with1–5% activity. Initially retain the donor tokenizer, experts, routing, and mixer; compress selectively, then convert toward target primitives only where the gate permits.

Two candidates new relative to the local screen merit metadata evaluation. **Granite-4.0-H-Tiny-Base**, Apache 2.0, has 7B total/~1B active and a Mamba2+attention mixer with shared experts: it is a below-target pilot, not a 10B. **LFM2.5-8B-A1B-Base** declares 8.3B total/**1.5B active**, gated conv+GQA, and LFM Open License 1.0: near the requested scale, but not Mamba1 SSM. These are producer metadata, not local evidence of quality/rate. The A1B suffix does not replace accounting. [IBM model card](https://huggingface.co/ibm-granite/granite-4.0-h-tiny-base), [Liquid model card](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-Base).

Do not download every donor out of curiosity. The existing OLMoE, StdMoE14B, and Qwen3-30B-A3B screen remains useful; it corrects precisely the illegitimate transfer of E63 mixed rate to full-int8. New candidates must undergo the same accounting, including shared experts, head, routing, and mixer. [L11]

| Phase and dependency | Output | Gate / pivot | Proposed resource |
|---|---|---|---|
| R1.0 metadata screen | Exact config, license/revision, operator inventory, bytes per organ | Reject incompatibility or traffic not plausibly reducible within20ms; no verdict from nominal active params |1–2 analysis days,0T4 |
| R1.1 after0: baseline and calibration | Native teacher, activations and sensitivity; int8/4bit/fp32 precision map | Baseline fidelity; if format loses>0.02 before surgery, stop that combination |12–24 pair-hours |
| R1.2 after1: minimum compression | Checkpoint with groups/outliers or adapters, same donor structure | ΔBPB and task pass; kernel affinity and bytes pass; prefer expressive4bit to ternary if quality requires it |24–96 pair-hours |
| R1.3 after2: bridge toward target SSM | Progressive mixer/attention replacements; separate dReLU/QAT | Per-block and cumulative gates; retain original mixer if replacement fails, marking bridge incomplete |12–72 pilot pair-hours; complete conversion to be repriced |
| R1.4 after3: export and future measurement | Same checkpoint inC, correct formats, rate at declared contexts | Parity + quality +50; attempt100 only afterward |3–10 CPU engineering days;2–8 final-calibration pair-hours |

**Scope boundary:** retaining Mamba2/conv/GQA requires extending the C engine; it is not a conversion already supported by `phase60/engine.c`. If the constraint is the exact SSM v1 architecture, R1.3 is mandatory and a hybrid bridge does not count as completion. If no replacement passes the gates, R1 delivers only a useful baseline for R3. Artificially enlarging7B to10B by cloning experts does not satisfy the required capacity.

**Resources/risks:** about 48–192 pair-hours for feasibility, plus export/calibration; at 12 h/day, 4–16 active days; at a 90 pair-hours/week quota, 0.5–2.2 GPU-only weeks. This is not an estimate for completing 10B. Main risks: mixer conversion, domain quality, vocabulary/head, license, and custom code. Mitigation: one coordinate at a time, baseline checkpoint, exact weighted router before its compression. **R1 is the first choice for reducing the knowledge that must be relearned; it is not currently a promise of the target.**

### R2 — Joint surgery on the dense donor: shared path + residual experts

**Guiding idea:** the dense FFN is distributed; selecting very few groups removes functions that cannot be recovered with a simple selector. Construct a shared path covering the common function and a few experts that model the residual, jointly training the compressed projections as well. The proposed operator is:

\[
f_l(x)\approx g_{shared,l}(x)+\sum_{e\in S_l(x)}a_{l,e}(x)f_{l,e}(x).
\]

`g_shared` can begin low-rank, but must be compared with a small nonlinear MLP. The router sum and weight must match deployment; the selector cannot require the hidden dense FFN to select groups. E68 motivates this operator change, but its signal is local, concentrated in one layer, and still dependent on dense-z; it is not evidence that this solution will work. [L17]

| Phase and dependency | Output | Gate / pivot | Proposed resource |
|---|---|---|---|
| R2.0 new-geometry baseline | Donorfp32/int8, exact shape, shared+router+experts+rank cost | Reuse valid evidence, no rerunH2I/H4/H5; step zero of the new geometry |8–16 pair-hours |
| R2.1 after0: shared/residual distillation | Layer/block fit, linear/nonlinear comparison, and genuinely economical router | Local error per layer/p95 improves; no layer hidden by the mean; cost within envelope |16–48 pair-hours |
| R2.2 after1: joint curriculum | Rank and sparsity decrease gradually; initially faithful precision; CE+KD+feature loss | At least a pilot1.5B passes ΔBPB≤0.02 and generative suite; if the curve flattens outside budget, stop |48–176 pair-hours |
| R2.3 after2: target-scale confirmation | Identical recipe on the donor~10B, sensitivity tuning; genuinely learned distinct experts | No exponent transfer1.5→10B; quality gate on the target, updated accounting |240–960 reservable pair-hours, no convergence guarantee |
| R2.4 after3: export/rate | Actual artifactC | Joint gate;50 before100 |5–15 engineering days,4–12 calibration pair-hours |

To isolate causes while respecting “one variable per stage,” compare the **training-protocol factor**: same final geometry, same data/tokens, separate training versus joint training. Do not launch a combinatorial grid. Within the curriculum, fix only one transition at a time: precision, rank, sparsity, mixer. The final model must nevertheless be retrained with all operators present, because errors interact.

The proposed order is: faithful baseline → shared/residual → moderate rank → progressive routing/sparsity → selective QAT → joint recovery. The reversal “QAT before sparsity” is a subsequent A/B only if the first pilot demonstrates learnability. Reverse-KL can refine trajectories, but it is mode-seeking and does not replace CE/forward-KL coverage. No mathematical inversion restores information eliminated by rank/quantization.

**Decisive risk:** meeting the budget may require sparsity that the donor does not support at that quality. H4 shows score recovery without generation; H5 closes the simple assembly path. R2 is the most informative continuation of the local line, but has greater risk thanR1. The SSM mixer phase, if strictly required, remains an additional conversion: the first R2 success in the donor runtime does not automatically satisfy the native architecture. [L14,L15,L02]

### R3 — Native SSM distillation with conditional capacity

**Guiding idea:** optimize the CPU architecture directly: selective SSM, small shared blocks, ternary/QAT experts, dReLU, and controlled head. Acquire pretrained knowledge through progressive/sequence-level distillation instead of demanding a near-algebraic conversion.

The mixer→blocks→logits progression follows a line documented by [MOHAWK](https://arxiv.org/abs/2408.10189); [Mamba in the Llama](https://arxiv.org/abs/2408.15237) demonstrates hybrid distillation and weight reuse, but not the specific equivalence required on Zen2. The literature entails billions of adaptation tokens, not simple calibration. The natively ternary BitNet case was pretrained from scratch: it is not evidence of economical conversion of a 10B. [BitNet2B4T](https://arxiv.org/html/2504.12285v2).

| Phase and dependency | Output | Gate / pivot | Proposed resource |
|---|---|---|---|
| R3.0 teacher/tokenizer selection | On-domain baseline; byte likelihood and deduplicated data | Retain tokenizer if possible; changeV only with measurable tradeoff |8–24 pair-hours |
| R3.1 after0: native pilot | Student100–500M or distilled blocks, CE-primary with KD challenger | Signal relative to CE at equal tokens; transition without unrecovered shock; do not call it a pilot10B |120–360 pair-hours |
| R3.2 after1: capacity growth | Experts from reused weights/upcycling, routing curriculum, coverage | Useful unique capacity, no systematic dead expert, quality/active-budget curve |Cost to measure: E/k and optimizer dominate feasibility |
| R3.3 after2: target~10B | Distillation+QAT with final geometry and few dense parts | ΔBPB≤0.02 **against teacher**, not only against initial student; generation/task |Order of magnitude1000–10000+ pair-hours possible; stop/reprice after pilot |
| R3.4 after3: native export | New deployment format, golden, end-to-end decode | All gates on the target |5–15 engineering days,4–12 final pair-hours |

**Risk:** a small core might not represent the teacher’s capabilities, even with many experts. Distilling 10B→200M produces a different objective; it can be a useful product and pilot, but does not satisfy learned 10B total capacity. Weight tying and depth reuse reduce distinct parameters: do not count the same matrix N times as 10B capacity. Upcycling copies an initial function; diversity must be learned, and the costs reported in [Sparse Upcycling](https://arxiv.org/abs/2212.05055) do not demonstrate completion in a few T4 weeks.

**Decision:** R3 is the path most faithful to SSM v1 and least compatible with a promise of conversion in short sessions. Start it at full scale only with pilot gates and a measured cost estimate. A very fast small result must not mask failure of the10B/quality constraint.

## B. Option matrix

“Impact” indicates the mechanism and a theoretical limit where available, **not expected speedup**. B/M/A = low/medium/high. Priorities do not reopen historical gates.

| Lever | tok/s impact | Quality impact | Complexity | Risk | Decision note |
|---|---|---|---|---|---|
| Already MoE/recurrent donor | High potential: avoids dense-active | Retains more pretraining |A|M/A|R1, new semantics to support |
| W8 with fp32 activations | More bytes than ternary, but costly recovery avoided | Best local donor baseline |M|B/M|E66 is neither full-int8 nor 50 tok/s |
| Multi-level INT4, mixed 3/4/8 bit | Up to 2× byte reduction over W8, before overhead | Calibrate organ by organ |A|M|4 bits of ternary storage ≠ a 16-level quantizer |
| Ternary QAT | Favorable bytes and LUT | High risk on pretrained |A|A|Selective; prohibited from inheriting sandbox cost |
| 4→2-bit packing |2× codes; P1 already measured | Lossless |M|B|Reuse, not a new experiment |
| 2→1.6-bit packing |≤1.25× bandwidth portion | Lossless on ternary |A|M|Decoder can cancel the 20% bytes saved |
| Per-group/outlier activation scales | Ambiguous speed | Can improve fidelity |M|M|Additional scales/LUT and overflow to account for |
| Per-organ whitened low rank | Reduces bytes/FLOP if r<mn/(m+n) | Growing risk at low rank |M/A|A|E65/H4 preclude promises at r/D=1/32 |
| Residual low-rank adapters | Additional cost | Selective recovery possible |M|M|Dense fusion can lose the advantage |
| Block/layer pruning | Reduces executable cost | Possible injury to functions |M|A|Hessian/output-aware; no unsupported sparsity |
| Nonlinear shared + sparse residual | Can reduce required activity | New R2 hypothesis |A|A|E68 only local-linear; requires router without dense oracle |
| Hierarchical/factorized router | Reduces the flat E·D term and dispatch | New partition/routing |A|A|Account for misrouting and expert coverage |
| Hadamard/Fourier | Can help the quantizer, does not reduce rank | Possible benefit/harm |M/A|A|No free diagonals; T3 VOID; do not repeat generic sweep |
| Weight tying/cross-block reuse | Saves resident bytes; compute remains | Must be trained |M/A|A|No counting duplicated parameters |
| Early exit/adaptive halting | Reduces executed positions/layers | Loss and inconsistent state possible |A|A|Safer as a verified drafter |
| Exact dReLU SKIP | Avoids inactive up/down | No change if zeros are exact |M|B|Sparsity to measure on the target |
| Predicted SKIP | Potentially high | False negatives alter output |A|A|Do not call it lossless |
| Predictive prefetch | Hides latency, does not create bandwidth | Neutral if nothing skips |M|M|Miss/cache pollution; temporal locality not assumed |
| Co-training predictability | Can yield stable blocks | Can sacrifice specialization |A|A|Regularize with quality budget and router load |
| Low-rank/PQ/RVQ head | Lowers DV | Sensitive head, risky approximate search |A|A|Cache-resident codebook; E17 already limits low-rank head |
| Reduced vocab/adaptive softmax | Reduces head cost | Tokenizer/normalization change |A|A|Report bytes/s; hierarchical softmax requires training |
| Per-block entropy coding | Reduces traffic if decoder is economical | Lossless |A|M/A|Metadata and independent access; no whole-file/token decoding |
| SSM state compression | Smaller state | Long-term errors |A|A|State does not grow with context; cost depends on D·N·L |
| Scan poly/LUT | Reduces residual exp cost | Recurrent error |M|M|E3.5 already implemented; new test only on target range |
| Adaptive SWA | Reduces KV/local attention | Can worsen retrieval/dependencies |M/A|A|Measure target length, not only short |
| Threading/dispatch/layout | Recovers overhead | Lossless if order is preserved |M|B/M|Many axes already closed; no reused historical multipliers |
| Exact block-verify | Amortizes shared weights | Distribution can be preserved |A|M|MoE unions, draft, replay, KV included; conditional C/T |

Matrix references: local evidence L03–L23; external methods W01–W08. AQLM/QuIP# are bitrate and quality candidates, not paths already ρ-safe in the engine: resident codebooks and contiguous traversal must be demonstrated. [W02,W03,W05]

## C. Priority subproblems

Scale 1–5: 5 means maximum leverage or greatest tractability. These are design judgments, not measurements.

| Priority | Subproblem | Urgency | Leverage | Tractability | Concrete decision |
|---|---|---|---:|---:|---|
|0|SP0 — identity, semantics, metrics|Blocking|5|5|Which donor, which runtime, what counts as10B and tok/s; exact reference |
|1|SP4 — joint quality preservation|Blocking|5|2|Pilot of executable geometry; BPB+task+rollout |
|2|SP6 — training memory and cost|Blocking|5|3|Master/optimizer/activations/offload; real throughput and token budget |
|3|SP2 — streamed budget|Blocking|5|4|Bytes per organ, precision, k/h/E; shared+expert together |
|4|SP3 — compute floor|Blocking for100|5|3|Per-organ profile; rank/mixer/head before another exp-LUT |
|5|SP1 — resident budget|Blocking for hot path|4|3|Head/router/state beyond projections; explicit spill if allowed |
|6|SP7 — export/ABI/parity|Blocking at deployment|4|4|Per-matrix format, scales/layout, and identical checkpoint data |
|7|SP8 — context/tokenization|Blocking for the final claim|4|3|Quality and throughput on the same text and context |
|8|SP5 — engine throughput|Optimize after quality plausibility|3|4|New measured bottlenecks; test only new layouts/operators |

SP1 is not solved by imposing a perfectly shared 32 MB “virtual” L3: t6 and row-partition matrices are a specific condition, with concurrent state and streaming. SP2 includes down granularity, not only the number of experts; SP3 includes O(D²) linear projections, O(DN) state, local attention, and O(DV) output head, which scale differently. [L03,L04]

## Operational mathematical analysis of the levers

### Algebra and geometry: compress important functions, not only matrices

For `W∈R^(m×n)`, rank-r SVD reduces parameters from `mn` to `r(m+n)` only if `r<mn/(m+n)`. The minimum **squared** Frobenius error is `Σ_{j>r}σ_j²`, not the loss delta. The operational metric is:

\[
E\| (W-\widehat W)x\|^2=\mathrm{tr}[(W-\widehat W)C_x(W-\widehat W)^T].
\]

Whitening/regularization of `C_x` and downstream sensitivity prioritize directions; a local Hessian/Fisher approximation, neglecting the first-order term only when justified, yields `ΔL≈½δwᵀHδw`. For rectangular or non-normal matrices, W’s eigenvalues do not identify “dead directions”: use singular values of `W C_x^(1/2)` and observed Jacobians. Validate on heldout and rollout, because calibrated and generated covariance can differ. [W01; derivation]

The manifold hypothesis is a hypothesis, not a compression certificate. Measure effective rank, weighted energy tail, subspace stability across domains, and principal angles `cosθ_i=σ_i(U_aᵀU_b)`. Aligned gradients across layers suggest tying experiments only after normalization and basis alignment; they do not prove layers are substitutable. Compare neuron/weight clustering with random partitions of identical cost/layout and with output error, not silhouette alone. [L19]

An orthogonal basis change `Q` preserves singular values: it does not create low rank. `Wx=(WQᵀ)(Qx)` can improve outlier quantization, but `φ(Qx)≠Qφ(x)` for generic nonlinearities, and a dense rotation can destroy sparsity or recurrence diagonality. Fourier diagonalizes operators with appropriate structure, such as circular convolutions, not arbitrary learned matrices. For MLPs with dReLU/SwiGLU, compatible permutations and positive rescalings can be absorbed exactly with verified algebra; dense rotations must be limited to linear boundaries. [W02; derivation]

**Useful inversions:** change the order of transformations, not invert a map that has lost information. Quantization Q and pruning P generally do not commute: `Q(P(W))≠P(Q(W))`. A continuous curriculum toward the final operator is more controllable than two simultaneous jumps. Pseudoinverse and residuals recover components observable from data, not arbitrarily eliminated knowledge.

### Analysis and information: optimal bitrate is per organ

For small perturbations, `δh_(l+1)≈J_lδh_l+ε_l`; therefore final error contains `Σ_l(Π_{j>l}J_j)ε_l`. Correlated errors do not sum like independent noise. Measure amplification on sequences, logit margins, and routing in addition to local MSE. This is why a sum of ΔBPB from individual interventions does not predict their composition. [L15; derivation]

For logits with top1-top2 margin `m`, a perturbation with ∞-norm below`m/2` preserves the argmax at that point; this does not guarantee whole trajectories. For a top-k router, the margin between the k-th and (k+1)-th gives an analogous criterion. A small continuous perturbation can discretely change the path; stable tie-breaking and explicit format are part of the contract.

The rate-distortion to optimize is practical: minimize `Σ_i t_i(b_i,r_i,k_i)` subject to `ΔBPB≤ε` and memory, estimating empirical distortions and interactions. The Lagrangian `L_quality+λ_t t+λ_R RAM` guides selection, but does not replace final gates. Initially assign, for management purposes,0.005 BPB to precision,0.005 to structure,0.005 to mixer, and0.005 to margin; **these are reserves, not guaranteed additivity**. If interaction consumes all margin, return to a prior Pareto point.

A ternary has at most `log2(3)=1.585` bits of entropy per uniform independent weight. If `p0` is high, `H=-Σp_i log2p_i` can be lower, but scales, correlations, padding, and decoding remain. Five trits in one byte yield1.6bit/weight: this does not imply an economical direct kernel. Entropy compression is useful only if:

\[
B_c/\beta+t_{decode}+t_{metadata}<B_{raw}/\beta.
\]

Independent blocks, small resident offsets, codebooks in L1/L2/L3, and contiguous reads preserve the ρ-law. Expanding everything into DRAM before inference saves disk, not traffic/token. PQ/RVQ can compress head/embedding, but lookup for every weight from DRAM would be the wrong direction; prefer tile decoding or resident codebooks. [W03]

The minimum required bitrate cannot be deduced from parameter count or an entropy histogram. The important quantity is information useful to the target distribution, estimated through heldout rate-quality curves and ablation. Conversion from ΔBPB to perplexity depends on bytes/token: at 4 bytes/token, +0.02 BPB means multiplying PPL by `2^(0.08)≈1.057`; the required ceiling is very severe. Do not translate a paper’s PPL improvement with a different tokenizer directly into BPB.

### Systems and control: recurrence and routing stability

A local recurrence `h_t=A_t h_(t-1)+B_t x_t` with `||A_t||≤a<1` has perturbation uniformly bounded by `ε/(1-a)` under a bounded-per-step-error assumption. In the real selective SSM, A/B and projections depend on inputs: stability of the `exp(ΔA)` diagonal alone does not certify the closed system. Measure empirical Jacobians, norms, saturation, and drift along rollout, especially when approximatingexp or quantizing the state. The parameterization`A<0,Δ>0` prevents one class of instability; it does not guarantee quality. [L01,W08; derivation]

State compression: estimate component observability on future loss/recall, not only state variance. Balanced truncation provides intuition for stable linear systems; the selective SSM requires local linearizations and verification of the transition product. O(1) denotes independence from history length, not zero cost or unlimited memory. With b_s bytes, state alone costs about `b_s L_SSM·2D·N`.

Depth reuse retains weights but repeats compute. Adaptive halting and early exit must keep the state required by the next token up to date; skipping recurrent layers can leave stale states. Train an economical state update or use early exit as a draft then verify it. High softmax confidence is insufficient, as it can be miscalibrated.

Prefetch as predictive control: observed state generates a distribution over future experts; choose prefetch with utility `p_use·latency_saved-(1-p_use)·pollution_cost`, subject to a bytes-in-flight budget. Do not save the correct reads on a miss. For predicted skip, false negatives enter the loss and fallback must be explicit. Regularizing temporality and load balance can reduce diversity and create expert starvation: monitor occupancy, routing entropy, worst-expert load, and hysteresis. Local i.i.d.-like routing precludes assuming a hot pool. [L04]

Granularity: first model `t_expert≈B/β_chunk+Lk·τ_dispatch`, **only if β is kernel-pure**. With 64.1b, `τ≈8.4 µs` is a historical anchor, not a universal constant. At L=32, k=8, this is 2.15 ms of assumed overhead alone. At constant capacity and active fraction, reducing h increases E and k, hence dispatch; increasing h reduces selection flexibility. Optimal granularity minimizes time at equal quality, not abstract byte count. [L03]

### Block verification and resource allocation

For K verified positions and a mean tokens emitted per cycle, with any bonus token accounted for in the work:

\[
t_{emit}=\frac{t_{draft}+t_{shared,once}+K C_{position}+t_{expert,union}+t_{commit}}{a}.
\]

With independent routing, expected union per layer is `U=E[1-(1-k/E)^K]`. If E is large and k/E small, `U≈Kk`; experts therefore do not amortize and rejected proposals cost time. Shared weights can amortize. The historical simplified formula and C/T≈0.15–0.20 are feasibility filters, not universal gates. Free draft does not mean free verify. [L04,W04]

For sampling, acceptance/rejection must be correct relative to teacher and draft; greedy matching suffices only for the fixed greedy case. SSM/KV state must be committed only for the accepted sequence, with correct penalty semantics. Distributional parity does not imply bit-identical output with different RNG streams.

**Application of the six frameworks to each lever family:**

| Family | Algebra/geometry to measure | Analysis/control/systems | Information / final criterion |
|---|---|---|---|
| Rank, tying, pruning | Whitened spectra, principal angles, matched null | Layer-wise amplification, reconstruction, and rollout | Distortion per byte/compute saved |
| Quantization, basis, head/PQ | Outliers, covariance, lawful symmetries, codebook geometry | Logit/router margins; decoder in the loop | Rate-distortion with metadata and decode cost |
| Shared+MoE, predictability | Output-contribution errors, residual geometry | Collapse, starvation, controller budget, and routing stability | Entropy/coverage and quality per active byte |
| SSM/mixer/SWA | Observability, temporal rank, transition structure | Sequential stability and long-context recovery | Useful information remembered, not variance alone |
| Reuse, exit, distillation | Function redundancy, feature alignment | Consistent state, confidence calibration, recovery curve | Teacher information preserved at fixed cost |
| Packing, threading, verify | Operator equivalence, reduction order | Pipeline, contention, exact commit | Effective bitrate and accepted tokens/s |

Convergence of QAT or FT is not known a priori. Record ΔBPB and tasks relative to tokens/hours, learning rate, applied updates, clipping, and nonfinite. An exponential fit to a short curve can help set the time cap, but does not prove the reachable floor: no extrapolation from a few hours to total recovery.

## D. Ten high-information experiments — all proposed, none performed

The IDs `STRAT-01…10` are proposal names, not new E/H of the program. Before implementation, the successor must compare them with the updated axis map and assign a brief only to the actually new cell. “No-regret” means useful information within a limited budget; it does not guarantee a positive outcome.

| ID | What and how to measure | Meaning of outcomes | Proposed time/cap |
|---|---|---|---|
|01|Exact screen of new sparse/hybrid donors: bytes and operators per organ, head, shared path, state, router, loader RAM|If budget is plausible, select a donor; otherwise avoid port and training. No rejection based on another format’s rate|1–2 analysis days, zero GPU|
|02|Expressive 4/8-bit quantization on the selected donor, same activation and structure; paired ΔBPB, margins, tasks, and scale cost|If 4 bit passes, it opens a less destructive path than ternary; if only 8 bit passes, reduction must come from structure. Does not repeat the ternary-rule sweep|12–24 pair-hours|
|03|E68 successor: nonlinear shared path plus residuals and an x-based genuinely executable router; heldout error per layer/p95, diagnostic comparison with oracle|Good oracle/bad router: selection problem; both bad: capacity/operator; both good: authorizes end-to-end evaluation, not direct promotion|16–48 pair-hours|
|04|Joint versus separate training, **same new final geometry**, same tokens/seed; begin from the donor, not frozen H5 composition|If BPB+task passes, the new cell is crossed; score alone indicates another SCORE-ONLY, without export; none pass: R1/R3 pivot|48–120 pair-hours|
|05|Sensitivity rank allocation at fixed byte budget; shared basis/adapters; comparison with uniform rank, without repeating E65|If it beats uniform but remains outside +0.02, it is local progress; if it enters, it recovers budget for FFN activity|12–36 pair-hours|
|06|Replace one mixer block toward SSM, then a small group, retaining FFN and tokenizer; feature/logit distillation and increasing contexts|If one block already fails, no total conversion; if short passes but long fails, retain attention or memory tier. Do not infer composition from one block|24–72 pair-hours|
|07|Quality curve against **executable bytes** and sparsity on the first valid trained pilot, with three preregistered points; router margins and coverage|If the quality point requires >20 ms, change shape; if an intersection exists, confirm on the target, without declaring 10B from the pilot|16–48 pair-hours|
|08|Only after quality pass: profile and end-to-end of the **new** geometry/format; RAM without debug copies, traffic amplification, thread/dispatch, and 2→1.6 bit where relevant|If decoding costs more than avoided traffic, retain 2 bit; if overhead dominates, optimize it. Do not repeat P1/E40/E63|2–5 CPU days, zero GPU|
|09|Only if C/T and shared share are favorable: K2/K4 verify on a valid target; accepted tokens, expert union, draft/replay, parity, and bytes/s|High acceptance without speedup: compute/unions limit; net speedup and parity pass: adopt. No assumed multiplier|1–3 CPU days; zero GPU with n-gram|
|10|Confirm the same checkpoint on new holdout, more seeds where sustainable, 2K/8K/32K contexts, and second AVX2 CPU|Only OOD/long-context degradation: explicitly limit the product; target quality+rate: sole valid evidence of the objective|2–5 CPU days, 8–24 evaluation pair-hours|

Experiments 02–07 are a conditional menu, not a queue to run in full. The first tranche can stop at 02 and one branch among 03/06. Block verification is not a priority if useful sparsity does not yet exist.

## E. T4×2 sessions, A–F pipeline, and costs

### Physical limits and cost-estimate units

**One pair-hour = both T4s occupied for one hour = 2 GPU-hours.** I assume no cloud quota currently available: to convert hours to calendar time, I use two hypothetical scenarios, 12 pair-hours/day or 90 pair-hours/week. A session on one T4 economically equals 0.5 pair-hours, but does not demonstrate DDP scaling. The 3860 tok/s measured on the small pilot are not an estimate for 10B. [L21]

The T4 has 16 GB and 65 TFLOPS FP16 peak; two T4s do not constitute 32 GB of unified memory. [NVIDIA specifications](https://www.nvidia.com/en-us/data-center/tesla-t4/). Use loss-scaled FP16 with FP32-sensitive parts; make no native-hardware BF16 assumption on Turing. Master weights and Adam do not become ternary because the forward uses 1.58 bits.

Indicative accounting for 10B full finetuning: 20 GB FP16 weights, 20 GB FP16 gradients, 40 GB FP32 masters, 80 GB for the two Adam moments = 160 GB **before** activations, implementation-dependent. FP32 gradients increase this further. Two T4s require a frozen/quantized base with adapters, blockwise training, optimizer offload, or aggressive sharding; PCIe/CPU traffic and host memory must enter the estimate. DDP replicates the model and does not solve OOM. A quantized teacher base itself introduces error: the final reference remains the original teacher or a reference validated within a separate budget.

Analytic order of magnitude for dense training: `F≈6PT`. For 10B parameters and 1B tokens, this is 6×10^19 FLOP; with 130 TFLOPS aggregate peak, **about 128 pair-hours at theoretical 100%**, about 427–1282 at assumed 30–10% utilization, before teacher/offload. For MoE use effective active operations, but total parameters/optimizer remain to be hosted. This estimate is neither a benchmark nor a tight LoRA limit, where cost depends on what is frozen.

The main estimate comes from `H=T/(3600·q_train) + H_teacher + H_eval + H_IO`. With q_train of 100/300/1000 tok/s, 1B tokens require 2778/926/278 pair-hours. At 90 pair-hours/week: 30.9/10.3/3.1 weeks, before other costs. These are scenarios, not expected rates. Literature distillations of 3–20B tokens can therefore mean months or years of quota, while remaining small relative to original pretraining. [MOHAWK](https://arxiv.org/abs/2408.10189), [Mamba in the Llama](https://arxiv.org/abs/2408.15237).

### Offline preparation phases

This table is an input/output contract. B–D are ordered differently in the three roadmaps; they must not be concatenated automatically. GPU caps are for pilots, not promises of 10B convergence.

| Phase | Input → output | Metric/gate | Estimated T4×2 | Risk and fallback |
|---|---|---|---|---|
|A — source analysis|Config, revision/license, tokenizer, data → baseline, sensitivity, shapes, and byte accounting|Reference parity; reproducible corpus/scoring; operator compatibility|8–24 pair-hours plus 1–2 desk days|Out-of-domain/incompatible donor → next candidate, no surgery|
|B — progressive pruning/factorization|Baseline + sensitivity → blocks/rank/shared path with executable mask|Loss continuity at every transition; projected quality and performance with margin|16–72 pair-hours|Magnitude pruning fails → Hessian/output-aware or shared path; if still outside, stop|
|C — distillation/adaptation|Pinned teacher + new geometry → recovered checkpoint|CE control at equal tokens; BPB/rollout/task; feature matching is insufficient|48–240 R1/R2 pilot pair-hours; 120–360 R3 pilot|Costly logits or unhelpful KD → CE-primary; KD as on-domain challenger|
|D — QAT and sparsity|Checkpoint C, precision map, optimizer resume → checkpoint with final forward|Nonfinite = failure; cumulative quality ≤0.02; logit/router/task margins|24–168 pair-hours on pilot, if needed|Ternary fails → 4/8 bit per critical organ; accept more bytes or abandon geometry|
|E — calibration|Checkpoint D + disjoint calibration → final scale/bias/outlier map|No fit on val/test; heldout improvement; stable long rollout|4–12 pair-hours|Local recovery worsens globally → return to D, do not retouch gate|
|F — export|Final checkpoint, tokenizer, exact operator → artifact C + manifest + golden|fp32 reference, same quantized operator, expected output/token/BPB and RAM; then rate|1–3 CPU days if conversion supported; 5–15 with new operators; 0–4 control pair-hours|Missing ABI → Sol prerequisite; no claim before real conversion|

For C, retain the tokenizer where possible. If it changes, use likelihood on the same bytes and aligned context; PPL/token is not comparable. Cross-tokenizer KD remains challenger: off-domain MVE lost to CE, not demonstrating general impossibility. Teacher logits must include normalization and tail residual if truncated; teacher storage and scoring time are explicit. Randomize chunk membership before streaming to avoid the already measured harmful blocked order. [L04,L06,L21]

For F, specify header/version, endianness, dimensions, tokenizer identity, operator, bias/norm, tied weights, effective rank, factor orientation, per-matrix precision, packing, scales/zero-points, rounding/clipping, expert order, routing/top-k weights, and checksum. E1M1/E4M1 are not generic containers for these choices. Tagged-v2 from the donor branch does not make the exporter automatically compatible. Retain only representations needed in deployment, with a separate offline reference. [L01,L02,L18]

### Proposed sessions, in decision order

1. **Session S1: 24–72 pair-hours, 2–6 days at 12 h/day.** Input: a selected donor, data/licenses, proposed baseline and precision map. Output: precision ladder and a structural pilot; R1/R2/stop decision. Not a 10B export.
2. **Session S2: 120–360 pair-hours, 10–30 active days or 1.3–4 weeks of 90 h quota.** Only if S1 provides a new signal. Input: exact geometry, optimizer resume, measured cost per step. Output: joint pilot checkpoint, comparison with CE/separate training, learning curves, and target-budget proposal. If only BPB improves without tasks, stop promotion.
3. **Session S3: 240–960 pair-hours, 20–80 active days or 2.7–10.7 weeks of 90 h quota.** Only for target adaptation compatible with memory and cost. Output: qualified target ~10B or documented failure. Full SSM distillation can greatly exceed this tranche: R3 repricing is required, not an optimistic sum of prior hours.

Add 10–25% organizational reserve for scoring/checkpoints/restarts **if not already included in effective rates**. Do not account for it twice. Each cap replaces a promise of finite duration: at expiry, adjudicate the predefined checkpoint, not one selected ex post.

**Monetary cost:** without a current provider offer, use `c_pair` in €/pair-hour. To make options comparable, a **purely hypothetical planning** range of 0.50–1.50 €/pair-hour yields: S1 €12–108; S2 €60–540; S3 €120–1440. These are not market rates or verified quotes; storage/egress excluded. Authorized free quota reduces outlay, not hours/calendar. R3 at 1000–10000 pair-hours would equal €500–15000 under those assumptions. **Short sessions reduce uncertainty; we have no evidence they suffice to complete 10B.**

## Handoff for Sol and documentation discipline

For every new experiment: a pre-measurement brief with question, baseline, counterhypothesis, changed scope, gate, budget, stopping rule, data/revisions/hash, and protocol; then immutable raw logs, per-document metrics, effective configuration printed by runtime, final checkpoint and optimizer/RNG resume, file manifests, and decision. Link brief→run→checkpoint→export→result→ledger, also recording failures, VOID, and planted controls. Update the canonical index by axes; do not rewrite the past.

**Three parity levels:** (1) lossless layout/packing/threading: bit-exact against the same operator; (2) quantized kernel: parity against scalar reference with identical weights/scales/rounding; (3) surgery/QAT: statistical comparison with fp32 teacher, but not bit-exact by definition. An fp32 reference of already quantized weights verifies implementation, not fidelity to the teacher. Kernel tolerance does not authorize modification of ΔBPB or task gate. [L05]

Do not require PyTorch GPU and C fp32 to always be bit-identical across different exp/reduction implementations: preregister appropriate tolerances and retain local deterministic reductions. No indiscriminate `fast-math`, no hidden VNNI, generic AVX2 fallback. Any second CPU will have specific results; it will not inherit Zen2’s 185 GB/s. [L05]

Future commits: messages such as `docs(strat-03): preregister deployable shared-residual pilot` or `research(strat-03): record failed quality gate and artifacts`, with IDs and result links. **No assistant signature, no assistant Co-authored-by.** This work creates no commits. An unperformed experiment remains `PROPOSED`, not `PASS` or `OWED`.

Sol’s first activity: read this dossier and the latest axis map; propose one brief for STRAT-01/02 or the selected pilot, starting from valid evidence. Do not implement the failed H2I export or rerun E63 A10B, terminal H4, or H5. The priority is to close the joint-checkpoint gap, not generate more synthetic speed tests.

## Limits and methodology

I consulted the graph with scoped queries, read the five requested references, distinguished the native runtime from the donor runtime, and checked the current program through E68/H5. Superseded summaries were resolved by prioritizing the updated probe; the T3 VOID case remains explicit. External research used primary sources for compression, distillation, donors, and T4. The two parallel deep-research skill analyses covered historical evidence and literature, with no code intervention.

Local results are evidence from a single program, often one host/one seed: they do not become three independent proofs because they are cited in README, INDEX, and probe. No new experiment, automated verifier, or test run was performed, in accordance with the constraint. The report is a Markdown handoff with an evidence register, not a paper with results reproduced in this session.

The new hypotheses — nonlinear shared path, new-donor conversion, alternative precision, rank allocation, and joint curriculum — have proposed measurements, not numerical success probabilities. Model fields of every future revision, cloud rates, account quotas, 10B performance, or recovery within +0.02 BPB have not been verified. The operational conclusion remains: **minimize distance from the pretrained model, then demonstrate quality and budget in the same geometry; commit weeks only after that signal.**

## Additional external bibliography

Consulted: 16 September 2026. W01–W08 and all local sources are in the [register](EVIDENCE.md). These sources add methodological/metadata evidence; none certifies the composite objective.

- W09 — IBM Granite Team (2025). [Granite-4.0-H-Tiny-Base, model card](https://huggingface.co/ibm-granite/granite-4.0-h-tiny-base). Sparse/hybrid pretrained e architettura/licenza dichiarate.
- W10 — LiquidAI (2026). [LFM2.5-8B-A1B-Base, model card](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-Base). 8.3B total/1.5B active, conv+attention; do not replace accounting with the suffix.
- W11 — Bick et al. (2024). [Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models](https://arxiv.org/abs/2408.10189). MOHAWK.
- W12 — Wang et al. (2024/2025). [The Mamba in the Llama: Distilling and Accelerating Hybrid Models](https://arxiv.org/abs/2408.15237). Progressive distillation, not T4 time.
- W13 — Microsoft/BitNet authors (2025). [BitNet b1.58 2B4T Technical Report](https://arxiv.org/html/2504.12285v2). Natively ternary model trained from scratch.
- W14 — Komatsuzaki et al. (2022/2023). [Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints](https://arxiv.org/abs/2212.05055). Pretraining reuse, not automatic active-compute reduction.
- W15 — NVIDIA. [T4 Tensor Core GPU, specifiche](https://www.nvidia.com/en-us/data-center/tesla-t4/). 16 GB, 65 TFLOPS FP16, PCIe Gen3.
- W16 — NVIDIA. [CUDA GPU compute capability](https://developer.nvidia.com/cuda/gpus) and [CUDA math/type support](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html). Turing/T4 CC7.5 and hardware-type constraints.
