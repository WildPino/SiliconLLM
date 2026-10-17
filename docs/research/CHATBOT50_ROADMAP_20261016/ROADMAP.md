# CHATBOT50 roadmap: a pretrained sparse chatbot at >=50 tok/s in engine.c, then capacity at fixed per-token cost

16 October 2026. **Status: APPROVED by the owner on 16 October 2026** (directive:
follow the phases and rules of this document and record every decision,
measurement and step; see the [decision log](#12-decision-and-progress-log)).
When it was written, apart from the desk ledger in
[`benchmarks/chatbot50/`](../../../benchmarks/chatbot50/byte_budget_ledger.py)
(which read one local GGUF header and no tensor values), nothing had been
downloaded, compiled, trained or timed. It replaces the "next selected step" of
[`HANDOFF.md`](../../../HANDOFF.md) §1/§12 (SmolLM2-135M recipient plus donor
memory). It changes no historical result or verdict. Sections 0-11 are the
approved text; later changes are recorded only in §12.

Labels used throughout: **MEASURED** (a canonical local result), **SOURCE-DERIVED**
(headers, configs, arithmetic), **DESK-MODEL** (rates derived from bytes; never
quoted as a rate), **PROPOSED**, **[L?]** (literature not re-read in this
session; read it in its own tables before it enters a decision).

---

## 0. The decision in one page

**What went wrong so far.** Since 21 September the programme closed many
component cells but never ran a pretrained chatbot end to end in `engine.c`.
Two patterns account for most of it:

1. *Per-tensor bit-exact parity used as a system gate.* STRAT-01 produced 57
   protocols and 55 scientific engine results in five days (21-25 September)
   and reached layer 2 of 26 of GigaChat 3.1. MoE routing turns a last-bit
   reduction-order difference into a different expert set at near-ties, so the
   cascade has no natural end point. The project's own laws already say why:
   kernel bit-exactness does not compose to system correctness, and a gate
   written to be severe is not thereby valid.
2. *Cost screens of new representations before any complete baseline path.*
   The native-expert-scaling campaign indexes 609 report files. The cost
   screens of new representations (additive codebooks, LUTs, palettes, child
   experts, protected cores, compact recipients) all failed their envelopes;
   the only passing rates (Switch, METH391) belong to an encoder-decoder
   infilling model, not a chatbot.

**The physics.** Batch-1 decode on the reference Ryzen 5 3600X is
bandwidth-bound: rate ~= BW_eff / bytes per token. With the measured 34-36 GB/s
(32 GB/s used for planning), 50 accepted tok/s at a declared 2048-token context leaves a budget of
**B ~= 0.45-0.55 GB per token, KV included** (§3). "Independent of model size"
can therefore only mean: stored capacity grows, bounded by 80 GB of RAM, while
bytes per token stay below B.

**What the ledger says (DESK-MODEL, §3).** Among header-verified donors only
**Ling-mini-2.0** (16.26B total, 1.43B active, MIT, chat post-trained) reaches B
at a plausibly quality-safe precision: 826 MB/token with a plain Q4 policy,
**461-545 MB/token** once the KV cache is int8 and the 157k-row head is scored
by an approximate scan plus exact rerank, i.e. 53-61 desk tok/s at context 2048.
GigaChat 3.1 tops out near 38 desk tok/s even with every lever (its MLA
attention alone streams 366 MB/token at Q4). Qwen3-30B-A3B, gpt-oss-120b and
Ling-flash-2.0 exceed B with their always-on path alone; gpt-oss-20b exceeds it
through its routed experts. **In every family screened, the large members miss
B by 3-5x under every post-hoc policy priced here.**

**Consequence.** The scalable claim has to be built, not found: a narrow, fast,
pretrained MoE core, plus capacity added at fixed per-token cost and
transferred from the large member of the same family.

**Proposal: three phases, each with a deliverable that stands alone.**

| Phase | Deliverable | Horizon (ESTIMATE) | Main risk |
| --- | --- | --- | --- |
| **A** | Ling-mini-2.0 running in `engine.c`, >=50 accepted tok/s (lower CI95, context 2048) on the same artifact that passes donor-relative quality gates; an interactive chatbot | 3-6 weeks, CPU only | complete 4-bit runtime efficiency (the only complete 4-bit C artifact ran at 0.53 of desk) and timing stability |
| **B** | The conversion pipeline: triage ledger -> precision policy -> conversion -> fidelity/quality/rate gates -> optional T4 healing, applied to >=2 donors of other families, each ending in a documented applicability class | +1-2 months | operator coverage per architecture |
| **C** | Capacity at fixed cost: the Phase-A artifact grows its stored capacity (extra experts or memory layers) by distillation from Ling-flash-2.0 (same family, ~6x larger) while keeping >=50 tok/s | months, T4 | teacher-signal throughput; whether added sparse capacity is useful at all |

**Asked of the owner now:** approve the direction; approve the pinned
Ling-mini-2.0 download (~32.5 GB BF16); accept the programme rules in §6
(statistical end-to-end fidelity calibrated on the reference's own noise
instead of per-tensor bit-exact cascades; a diagnostic budget per failure).
**Later (Phase C):** the teacher-signal budget (§8, C1).

---

## 1. The contract

Inherited from [RESEARCH_INDEX §2](../RESEARCH_INDEX.md), the
[scope clarification](../NATIVE_EXPERT_SCALING_20260925/SCOPE_CLARIFICATION_20261015.md)
and the [original objective](../NATIVE_EXPERT_SCALING_20260925/HANDOFF_SNAPSHOT_20261014/goal-objective.md);
restated so that every clause has an instrument.

| Gate | Requirement |
| --- | --- |
| Rate | Accepted tokens/s, batch 1, declared decode mode (greedy + the project repetition penalty, or a declared sampler), context 2048 primary with 512 and 4096 reported; clean box (foreign occupancy <= 4.39%, E52); >=10 fresh processes interleaved; **lower CI95 >= 50**. TTFT/prefill, RAM and dispersion reported separately. 100 tok/s remains the stretch target. |
| Same artifact | One container hash and one printed-and-asserted `CONFIG` line for every quality and rate cell. |
| Engine fidelity | Engine versus a pinned reference runtime on the identical quantized artifact, within a tolerance derived from the reference's own measured noise floor (§6, rule 2). |
| Quality | Against the BF16 source: paired document-level dBPB with upper CI95 <= +0.02 (preferred <= +0.01) on fresh documents; donor-relative non-inferiority on a frozen task and rollout panel. |
| Size-insensitivity (Phase C) | The rate gate still holds after distinct stored capacity grows >=2x and >=4x over the Phase-A artifact; held-out quality improves monotonically with capacity and degrades when the added capacity is removed or shuffled; every cost that grows with capacity (router rows, key scoring, page/TLB effects) is measured. |

---

## 2. What the evidence already settles (reused, not rerun)

| Fact | Status | Source |
| --- | --- | --- |
| DRAM read ceiling 36.30 GB/s median, 40.81 best; L3 cliff at 16 MB | MEASURED | [SPEED_LEDGER §40](../donor_adaptation/SPEED_LEDGER.md) (E30) |
| int8 weight stream 34.0-34.9 GB/s in the donor engine; one box has different bandwidths per format, so never divide one format's bytes by another's bandwidth | MEASURED | SPEED_LEDGER §59-60 (E60/E61) |
| A GGML-style Q6_K matvec reached 34-39 GB/s at six workers (264.07 MB head in 6.695-7.733 ms) | MEASURED | [METH409](../NATIVE_EXPERT_SCALING_20260925/METH_409_LING_ADDITIVE_COST_PROFILE_RESULT_20261004.md) |
| Gathered rows below ~2 KB lose up to 42% of bandwidth; contiguous blocks >=32 KB lose ~6% | MEASURED | SPEED_LEDGER §41 (E31) |
| Per-token cost is linear in context; KV must sit inside the byte budget | MEASURED | E46 (COMMUNICATION history) |
| int8 per-row is near-lossless at 0.5-7B (+0.000066 to +0.0013 BPB) | MEASURED | E60/E62/E66 |
| Producer Q4_K_M of GigaChat 3.1: fresh dBPB +0.0123 (upper CI +0.0136), PIQA and rollout pass | MEASURED | [RESEARCH_INDEX §5](../RESEARCH_INDEX.md) |
| W4 plus F32 router on StdMoE fails the upper-CI gate by +0.0004 | MEASURED | STRAT-02F |
| Post-hoc ternary sits at chance; post-hoc dense-to-MoE carving is dead; post-hoc attention rank cuts are expensive (E65, gpt-oss rank-256 Q/O, head-group oracle) | MEASURED | E12, E37/E38/E64, E65, [first-attention results](../PIPELINE_REASSESSMENT_20261015/FIRST_ATTENTION_RANK_RESULT_20261015.md) |
| A router trained jointly with experts beats static selection; in-format training recovers score more than rank at 1.5B | MEASURED | H1, H2I, H4 |
| Block verification amortizes shared weights only; routed pools are about break-even under the measured i.i.d. routing | MEASURED | Phase 63, [SCALEUP_ARCHITECTURE §3.8](../../SCALEUP_ARCHITECTURE.md) |
| GigaChat MTP pilot: 25/38 accepted on one prompt; llama.cpp 12.78 -> 11.01 tok/s with a BF16 draft | MEASURED (pilot, n=1) | [MTP audit](../donor_adaptation/audits/STRAT_01_GIGACHAT31_MTP_PILOT_20260918.md) |
| Complete Qwen2.5-0.5B-Instruct runtimes: all-I8 (496 MB/token) medians 44.5-53.0 tok/s, spreads to 1.40; W4G32 mixed (357 MB/token) medians 35.7-50.2, spreads to 1.63 | MEASURED | [all-I8](../PIPELINE_REASSESSMENT_20261015/RECIPIENT_INT8_BASELINE_RESULT_20261015.md), [mixed](../PIPELINE_REASSESSMENT_20261015/RECIPIENT_MIXED_SCALAR_RESULT_20261015.md) |
| Selected-expert cost is not constant at extreme pool sizes (E128000 vs E12800: 1.31x) | MEASURED | METH177 |
| Ling-mini-2.0 geometry, 14,813 tensors, 16.256B positions | SOURCE-DERIVED (headers) | [METH407](../NATIVE_EXPERT_SCALING_20260925/METH_407_LING_MINI_SOURCE_HEADERS_RESULT_20261004.md) |
| Resources: 80 GiB RAM; RTX 3060 12 GB; Kaggle 3 accounts x 30 GPU-h/week of T4x2 (granted 90 h/week), ~30 GiB host RAM per session; D: 992 GB and E: 5.2 TB free | MEASURED/RECORDED | HANDOFF §13, COMMUNICATION history, `df` 2026-10-16 |

Two of these deserve emphasis. The engine **can** stream at the wall (int8 and
Q6_K both did). But the **only complete 4-bit C runtime ran at about half of its
desk rate** (§3), and complete runtimes show process-to-process spreads up to
1.63. Neither is a hardware limit; both are the critical path of Phase A.

---

## 3. The byte budget and the donor ledger

**Model (DESK-MODEL).** `t_token = bytes_per_token / BW_PLAN + T_FIXED`, with
`BW_PLAN = 32 GB/s` (0.88 of the E30 median; E61 and METH409 sit above it) and
`T_FIXED = 2 ms` for norms, routing, softmax, barriers and sampling (an
assumption that cell A1 measures; R1 measured 1.20 ms at 0.5B). For a lower
CI95 at 50 tok/s with a ~5% band, the central time must be ~18.5 ms, so
`bytes_per_token <= (18.5 - 2) ms x 32 GB/s ~= 0.53 GB`, KV included.

**Policies priced** (block sizes from ggml: Q4_K 4.5 bit, Q6_K 6.56, IQ3_S 3.44):

| Policy | Matrices | Routed experts | Head | Router | KV |
| --- | --- | --- | --- | --- | --- |
| P0 | Q4_K | native (Q4_K; gpt-oss MXFP4) | Q6_K full scan | F32 | F16 |
| P1 | Q4_K | native | Q6_K full scan | F32 | int8 + F16 scale |
| P2 | Q4_K | native | approximate scan (placeholder: rank-256 Q4_K preview) + exact Q6_K rerank of 64 rows | F32 | int8 |
| P3 | Q4_K | IQ3_S | as P2 | F32 | int8 |
| P4 | Q4_K | IQ3_S | as P2 | BF16 | int8 |

P2-P4 are hypotheses whose quality and fidelity are unmeasured; the rank-256
preview is a byte placeholder until cell A5 chooses the actual scheme.

**Ledger at context 2048** (full table, 512/4096 columns and provenance:
[`results/byte_budget_ledger_20261016.txt`](../../../benchmarks/chatbot50/results/byte_budget_ledger_20261016.txt)):

| Donor (evidence) | P0 MB/token | P0 desk tok/s | P4 MB/token | P4 desk tok/s | P4 at x0.85 / x0.53 |
| --- | ---: | ---: | ---: | ---: | --- |
| **Ling-mini-2.0** 16.3B/A1.43B (headers) | 826.0 | 36.0 | **461.0** | **60.9** | 51.8 / 32.3 |
| Trinity-Nano-Preview 6B/A1B (config) | 655.1 | 44.5 | 399.3 | 69.1 | 58.7 / 36.6 |
| OLMoE-1B-7B (config) | 965.3 | 31.1 | 645.9 | 45.1 | 38.3 / 23.9 |
| LFM2-8B-A1B (config) | 940.8 | 31.8 | 697.4 | 42.0 | 35.7 / 22.3 |
| GigaChat3.1-10B-A1.8B (headers) | 1,036.4 | 29.1 | 780.7 | 37.9 | 32.2 / 20.1 |
| Granite-4.0-H-Tiny (headers) | 1,011.0 | 29.8 | 808.5 | 36.7 | 31.2 / 19.4 |
| Qwen3-30B-A3B (config) | 2,035.7 | 15.2 | 1,438.6 | 21.3 | 18.1 / 11.3 |
| gpt-oss-20b (config) | 2,164.8 | 14.4 | 1,446.9 | 21.2 | 18.0 / 11.2 |
| gpt-oss-120b (config) | 3,049.4 | 10.3 | 2,175.0 | 14.3 | 12.1 / 7.6 |
| Ling-flash-2.0 103B/A6.1B (config) | 3,510.3 | 9.0 | 2,461.6 | 12.7 | 10.8 / 6.7 |

The actual producer GGUF of GigaChat 3.1 (Q4_K_M, read from its header) streams
1,016.2 MB of weights per token plus 61.3 MB of F16 latent KV at 2048, within 4%
of its P0 row.

**Scoring the desk model against reality.** On the two complete C artifacts
measured on this machine the desk model is optimistic: the all-I8 Qwen runtime
reached **0.85** of its desk rate at context 2048, the W4G32 runtime **0.53**.
The last column applies both factors. **Ling-mini P4 clears 50 only if the
complete 4-bit runtime reaches >= 0.83 of desk** (50/60.9). Closing that gap is
engineering with known precedents (the Q6_K head kernel already reached the
wall), and it is the first thing Phase A measures.

**What the ledger decides.**

1. For Ling-mini the head is the single largest lever: an approximate scan cuts
   240 MB/token. Without it no policy reaches 50 at 2048 (P1 with a Q4_K head:
   ~702 MB, 41 desk tok/s).
2. Speculative decoding is not a primary lever for Ling-mini. Its routed
   experts (45% of P4 bytes) do not amortize under the measured i.i.d. routing,
   so a two-position verify costs 1.43x a single step and a draft must be
   accepted more than 43% of the time merely to break even; the chat checkpoint
   ships no MTP head. It remains a lever for always-on-dominated donors
   (GigaChat: 68% of P0 bytes always-on), consistent with Phase 63.
3. Large members are excluded by the always-on core, not by expert payload:
   Ling-flash's attention, dense, shared, router and approximate head already
   cost ~1.05 GB/token at P4 before KV; gpt-oss-120b's ~0.59 GB. Shrinking
   experts cannot fix that, which is why the gpt-oss protected-core screen
   failed.
4. Trinity-Nano is even cheaper per token but is a 6B preview; it is the natural
   candidate for the second, independent family in Phase B.

---

## 4. Why this route and not the documented restart points

| Restart point in the record | What it would deliver | Why it is not first |
| --- | --- | --- |
| STRAT-01 normalization-parity cell (PAUSE record) | Bit-exact layer 1-2 of GigaChat 3.1 | The ledger caps GigaChat at ~29 desk tok/s as shipped and ~38 with every post-hoc lever; the cascade has no end point. Its kernels (Q4_K/Q6_K/Q5_0 x Q8, MLA, router, SwiGLU) are reused. |
| SmolLM2-135M recipient + frozen donor memory (HANDOFF §12) | A transfer falsifier on a cheap core | The core is not a useful chatbot (source-reported MMLU-cloze 29.3, GSM8K 1.4), so any failure or success is dominated by it. The memory mechanism is kept for Phase C on a strong core. |
| Native-MoE carving of gpt-oss or Granite | Finer experts with learned masses | Their always-on cores exceed the budget before any expert is read (protected core 34-54 ms). |
| New codebook/LUT activation formats | Fewer stored bits | Every complete cost screen failed; on Zen2 without VNNI, byte streams with cheap decode win. |
| Phase-64 native SSM ladder | A from-scratch CPU-native model | Not a pretrained chatbot within the T4 budget; remains parked. |
| Compress a large member (e.g. Ling-flash) into B | One big model at 50 tok/s | Its always-on core alone is ~2x B; reducing it needs retraining a 100B core, infeasible on T4. |

---

## 5. Phase A: Ling-mini-2.0 at >=50 tok/s in engine.c

Experiment IDs use the prefix `C50-`. Each cell gets its own protocol pushed
before its run (§6); the outlines below fix the question, the gate and the stop
rule. Compact no-duplication cards are in Appendix A.

**A0. Pin and screen (desk, ~1-2 days).** Run
[`conditional_donor_preflight.py`](../../../benchmarks/native_expert_scaling/conditional_donor_preflight.py)
and the ledger on pinned revisions of Ling-mini-2.0, Trinity-Nano, LFM2.5-8B-A1B
and any newer instruct MoE with <=1.5B active found at screen time. Verify
licence, chat template, EOS/pad IDs, tokenizer files, and **tokenizer identity
between Ling-mini and Ling-flash** (needed for Phase C). *Decision rule, fixed
now:* the Phase-A donor is the MIT/Apache instruct checkpoint with the lowest P2
bytes at 2048 among those whose producer-reported scores are not lower than
Ling-mini's on every benchmark the two cards share (at least two shared
benchmarks are required for eligibility); default Ling-mini-2.0. No weights
acquired.

**A1. Kernel and layer roofline (synthetic values, real shapes).** Measure
Q4_K, Q6_K, IQ4_XS, IQ3_S and Q8_0 matvec at six physical workers on
[3072x2048], [2048x2048], the fused 8x[512x2048] gate/up/down of one MoE layer
and [157184x2048], then one complete synthetic Ling-mini layer with its
barriers, and a 20-layer synthetic token without head. Start from the existing
llama.cpp-derived AVX2 kernels (`benchmarks/phase60/strat01_q4k_q8k.h`,
`strat01_q6k_q8k_avx2.h`). **Gate:** every streamed class >= 30 GB/s, complete
layer <= bytes/30 GB/s + 80 us. **Stop:** if Q4_K stays < 25 GB/s after one
tuning iteration, Phase A is re-planned (P-policies with an int8 core do not
fit B) before any download.

**A1b. Timing stability.** Root-cause the 1.14-1.63 process spreads: physical-
core affinity across the two CCXs, wait policy, pre-faulted and locked weight
pages, large pages if the privilege is available, priority. **Gate:** across 10
fresh processes the max/min of per-process median tok/s <= 1.06. A rate gate on
a lower CI is meaningless until this passes.

**A2. Acquisition and references (needs owner approval).** Download the pinned
BF16 checkpoint; record hashes; convert with pinned llama.cpp to BF16 and
Q4_K_M GGUF (imatrix from FIT documents only). Build a fresh document holdout
with FIT/DEV/RESERVED roles (`build_document_holdout.py`). Feed **source-
tokenizer IDs** to every runtime: STRAT-01 found the llama.cpp text tokenizer
mismatching on 95/96 documents for GigaChat.

**A2b. Reference noise floor.** On the Q4_K_M artifact, compare llama.cpp with
itself (generic vs AVX2 build, 1 vs 6 threads) on FIT positions: per-position
top-1 disagreement, mean KL, dBPB. This sets the engine tolerance before any
engine output exists. Also record llama.cpp's own decode rate at six threads as
the external reference rate.

**A3. Engine port, P0 artifact.** A clean bailingmoe2 decode path behind a new
`engine.c` profile: GGUF loader, QK-norm, partial RoPE (theta 600000), GQA
16/4, sigmoid router with expert bias, 8-group/top-4-group selection, top-8,
normalization and scaling 2.5 exactly as the pinned source code specifies,
shared expert, dense layer 0, KV cache, exact head, greedy/sampling, `--bpb`
scoring. Kernels are unit-tested against scalar oracles (existing practice).
**Gate (system fidelity vs llama.cpp on the same GGUF):** top-1 disagreement and
mean KL each <= max(2 x floor, 0.2% / 1e-4 nats); |dBPB| upper CI <= 0.002.

**A4. Quality of P0 against BF16.** Fresh-document dBPB per category; a frozen
task panel scored on both arms (PIQA reuse, knowledge multiple choice, short
math with exact match, verifiable instructions, an Italian/English chat panel
with automatic checks) with non-inferiority margins frozen in the protocol; the
96-document rollout degeneration test reused.

**A5. Approximate head (offline first).** On captured FIT hidden states compare
low-rank preview heads (rank 128/256/512) and IVF partitions (centroids C,
probes P; the Phase-56 recall-tier machinery is reusable) with exact rerank.
Candidate sets always include the tokens touched by the repetition penalty.
*Selection rule, fixed now:* the cheapest configuration with top-1 recall
>= 99.9% on FIT; confirmed once on DEV. Then C implementation. **Gate:**
teacher-forced top-1 agreement with the exact head >= 99.9% on DEV positions;
likelihood scoring (BPB, PIQA) always uses the exact head of the same weights.
Literature anchor: SVD-softmax [L?].

**A6. Precision ladder P1 -> P4.** int8 KV, IQ3_S or IQ4_XS routed experts
(imatrix from FIT), BF16 router; each step priced in bytes and in quality on FIT,
the chosen policy confirmed once on DEV. *Selection rule:* the cheapest policy
whose DEV dBPB upper CI <= +0.015 (leaving margin to the +0.02 gate) and whose
task deltas stay inside the non-inferiority margins. If no step beyond P2
qualifies, stop at P2 and take the rate consequence honestly.

**A7. Final adjudication on one artifact.** RESERVED documents and the frozen
task/rollout panel for quality; clean-box rate at 512/2048/4096 with >=10
interleaved processes; RAM, TTFT and prefill rate recorded.

**A8. Usable chatbot (deliverable, not a gate).** C tokenizer verified against
the source tokenizer, chat template, interactive loop.

**Registered predictions (to be scored, not gates).**

| Cell | Prediction |
| --- | --- |
| A1 | Q4_K on the fused expert and attention shapes reaches 30-36 GB/s at six workers |
| A2b | llama.cpp's own build/thread floor is >0 (top-1 disagreement of order 0.1-1%) |
| A2b | llama.cpp decodes Ling-mini Q4_K_M at <= 30 tok/s at context 2048 |
| A4 | P0 fresh dBPB lies in +0.005..+0.020 |
| A5 | IVF with exact rerank needs < 15 MB/token for 99.9% recall; a rank-256 preview alone may not reach it |
| A6 | IQ3_S routed experts cost +0.01..+0.04 BPB over P2 |
| A7 | Final central rate at 2048 is 48-58 tok/s |

---

## 6. Programme rules (proposed changes are marked NEW)

1. **NEW: complete path first.** No component optimization before a complete
   P0 forward runs with its full per-organ cost breakdown.
2. **NEW: system fidelity is statistical and calibrated.** The engine is judged
   against a pinned reference on the same artifact with a tolerance derived from
   the reference's own measured disagreement across builds/threads (A2b), set
   before the engine is measured. Per-kernel bit-exactness against a scalar
   oracle stays a unit test; it is never a cascade gate.
3. **NEW: diagnostic budget.** After any gate failure, at most three diagnostic
   cells or three working days, then a written design review that changes the
   coordinate or stops the line.
4. FIT selects, DEV confirms once, RESERVED adjudicates the final artifact. No
   threshold moves after DEV is read.
5. Protocols of discretionary gates are pushed before their run; `CONFIG` is
   printed and asserted; VOID runs are kept and never relabelled.
6. Desk rates are never quoted as rates; every claim carries its label.
7. Runs expected to exceed one hour are launched or explicitly authorized by
   the owner; T4 work is announced with reason, budget and stop rule first.

---

## 7. Phase B: the pipeline to other LLMs

**Stages** (each reusing Phase-A tools):

1. *Triage, metadata only:* ledger at P0-P4 against B; licence; operator
   coverage. Output: a class.
2. *Conversion:* pinned HF checkpoint -> pinned llama.cpp converter -> imatrix
   quantization per the chosen policy -> engine container with an architecture
   descriptor.
3. *Operator registry:* GQA/MLA/SWA, QK-norm, partial RoPE/YaRN, dense and MoE
   FFN, router variants (softmax/sigmoid, bias, groups, scaling), shared
   experts, short-conv or SSM mixers only when a selected donor needs them.
4. *Gates:* A2b/A3 fidelity, A4 quality, A7 rate protocols, unchanged.
5. *Lossy levers and healing,* only for over-budget donors: IQ2/IQ3 experts,
   top-k reduction, attention low-rank, head approximation; QLoRA plus logit
   distillation from the BF16 teacher on T4. Every lever is reported as a
   measured rate-distortion point.

**Applicability classes** (documented in the method record):
*Direct* (passes at P0-P4), *Lossy* (passes only with healed lossy levers),
*Capacity-transfer* (the donor serves as a teacher for a fixed-active
recipient, Phase C), *Unsupported* (with the measured reason). A dense donor
is Direct only if all its weights fit B at a quality-safe policy (on this
machine roughly <=0.8B parameters at 4-bit with an approximate head; the ledger
decides per donor); larger dense donors are Capacity-transfer, because
post-hoc dense-to-MoE carving is closed.

**Donors.** B1: the best second-family Direct candidate from A0 (currently
Trinity-Nano or LFM2.5). B2: GigaChat 3.1 as the Lossy test case (MLA plus MTP,
always-on dominated; it exercises speculation, attention compression and
healing, and its quality evidence already exists).

**Gate.** At least two donors of different families processed end to end with
a documented class; at least one besides Ling-mini passes the full contract.

---

## 8. Phase C: capacity at fixed per-token cost

The thesis to test: **at fixed bytes per token, held-out quality rises with
stored capacity transferred from the family's large member.**

**C0. Gap map.** Ling-mini artifact versus Ling-flash-2.0 (local llama.cpp; an
IQ3/Q4 GGUF of 44-58 GB fits in RAM) on frozen panels: closed-book knowledge,
reasoning, chat, fresh-document BPB. If the gap is mostly knowledge, sparse
capacity is the right tool; if it is mostly reasoning, active compute is, and
Phase C is reconsidered before spending T4 time.

**C1. Teacher signal (owner decision).** Ling-flash does not fit a T4x2 at 4
bits. Options: top-k logits from the local CPU (slow, and it occupies the
benchmark machine), sequence-level outputs from a hosted endpoint (cheap;
check terms of use and spend), or a smaller same-tokenizer teacher. The choice
fixes the realistic KD volume.

**C2. Falsifier with matched trainable parameters and data (T4).** On the
frozen 4-bit Ling-mini base: (a) extra experts per MoE layer with new router
rows; (b) one product-key memory layer (~1M rows, sparse updates); (c) a LoRA
control with the same trainable-parameter count. **Gate:** (a) or (b) beats (c)
on the held-out Ling-flash-gap panel by a margin frozen in the protocol, with no
regression on the general panel; removing or shuffling the new parameters
erases the gain; the engine still meets the rate gate. Literature anchors:
Lifelong-MoE, Memory Layers at Scale, Engram, Memory Grafting [L?, the last two
already examined in ACTIVE_WORK_REVIEW].

**C3. Scaling.** The winner at 2x and 4x capacity; monotone held-out gains, rate
gate still met, and all capacity-growing costs reported (router rows grow as
E x D per layer; product keys as sqrt(N)).

**Stop.** If C2 shows no gain over the LoRA control beyond its pre-registered
margin, capacity transfer is closed at that budget and recorded as such, not as
a general impossibility.

---

## 9. Risk register

| Risk | Where it shows | Fallback |
| --- | --- | --- |
| 4-bit kernels below the wall | A1 | Tune once; else re-plan before download (Trinity-class donor, shorter declared context) |
| Process-to-process dispersion | A1b | Placement/paging fixes; no rate claim until solved |
| Approximate head recall | A5 | Larger probe set; at P1 with a Q4_K head Ling-mini is ~41 desk tok/s, so a failed A5 means a cheaper donor |
| IQ3 experts cost too much quality | A6 | IQ4_XS (+~15 MB/token) or Phase-B healing |
| Chat quality insufficient for the owner's use (Italian) | A4/A7 | Not a gate; triggers a donor switch through the A0 rule |
| Teacher-signal throughput | C1 | Sequence-level KD; smaller same-tokenizer teacher |
| Custom MoE training code too slow on T4 | C2 | Grouped-expert kernels; smaller ΔE |

---

## 10. Resources and timeline (ESTIMATES)

Phase A: 3-6 weeks, CPU only, ~35 GB of new downloads plus ~20 GB of GGUF
variants. Phase B: 1-2 months, one or two further downloads of 15-25 GB.
Phase C: months of T4 within the granted 90 h/week, plus the C1 decision;
a Ling-flash GGUF (44-58 GB) for the gap map and teacher. Disk is ample.

## 11. Immediate next steps after approval

1. A0 and A1 need no model download and can start at once.
2. A2 starts on the owner's download approval.
3. Each cell's protocol is pushed before its run.

---

## Appendix A: compact no-duplication cards

| Cell | Nearest prior cell | Changed coordinate | Why the prior cell does not answer |
| --- | --- | --- | --- |
| A1 | METH409 (Q6_K head, additive codes), Granite child-256 cost, gpt-oss protected core | Standard k-quants at Ling-mini shapes, six workers, fused 8-expert layer, kernel-level gate | None measured Q4_K/IQ3_S on these shapes with six workers in a fused MoE layer |
| A1b | Recipient int8/mixed baselines | Estimand: dispersion causes, not rate | They report spreads up to 1.63 without a cause |
| A3 | STRAT-01 Rung 2C and successors | Donor/architecture and estimand: calibrated distributional fidelity of a complete forward | Per-tensor NRMSE cascades never certified a complete forward |
| A4 | STRAT-01 fresh BPB | Donor and engine path | Different model, different runtime |
| A5 | METH509/510 shortlist head, Phase-56 IVF recall tier | Sub-linear-byte approximate scan of a 157k x 2048 head with exact rerank | METH510 scanned the full head in I8; Phase 56 indexed context, not a vocabulary |
| A6 | STRAT-02 W4, E66 one-byte 7B | Donor and per-organ k-quant policy | Precision damage is donor-specific (STRAT-02F vs STRAT-01) |
| A7 | E63 Part B (synthetic 10B), recipient baselines | A pretrained chatbot artifact | Synthetic weights or non-qualifying artifacts |

## Appendix B: reproduce the ledger

```
python -I benchmarks/chatbot50/byte_budget_ledger.py --show-sources \
  --gguf benchmarks/donor_adaptation/density/results/strat01_gigachat_q4_97045b2/GigaChat3.1-10B-A1.8B-q4_K_M.gguf \
  --json benchmarks/chatbot50/results/byte_budget_ledger_20261016.json
```

Standard library only; reads configs/headers, never tensor values. External
facts used here (Ling-mini card: MIT, 16.26B total, 1.43B activated, 789M
non-embedding, MTP only for the base model; Ling-flash, LFM2 and Trinity-Nano
configs) were read from official Hugging Face pages on 2026-10-16 and must be
re-pinned by revision in A0. Landscape items found only through secondary
sources (Qwen3.5/3.6 MoE sizes, Ling 2.5/3.0, Trinity Mini/Large) are not used
in any decision here.

---

## 12. Decision and progress log

Append-only. Each entry: date, decision or measurement, evidence link, label.

| Date | Entry | Evidence | Label |
| --- | --- | --- | --- |
| 2026-10-16 | Owner approves the roadmap: direction (§0), programme rules (§6) and the phase order. The owner's directive asks to follow the phases without pausing for questions; the pinned Ling-mini-2.0 download (A2) is read as covered by it and is recorded here as such. | owner directive in session, 2026-10-16 | DECISION |
| 2026-10-16 | Work branch `research/chatbot50` opened from `research/native-expert-scaling` at `e21adbc`. | git | DECISION |
| 2026-10-16 | **C50-A0 closed.** Phase-A donor = Ling-mini-2.0 @ `a810f64` (no other candidate eligible under the §5 rule). Ling-flash-2.0 tokenizer identical; Ling-3.x flash id-compatible except 9 special tokens. Deviation: the roadmap was pushed after the screen ran. | [A0 result](C50_A0_DONOR_SCREEN_RESULT_20261016.md) | SOURCE-DERIVED |
| 2026-10-16 | A2 started: Ling-mini-2.0 @ `a810f64` downloaded to `E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/sources/`; all 14 files match the API hashes (LFS sha256 / git blob). llama.cpp pinned at `5b335f4` (the STRAT-01 pin; it has `bailingmoe2`); AVX2 and generic builds started (correction 2026-10-16: an earlier wording of this row said "built" before the build had finished). | `benchmarks/chatbot50/results/c50_a2_ling_mini_source_manifest.json` | MEASURED |
| 2026-10-16 | C50-A1 protocol pushed before any timed run. | [A1 protocol](C50_A1_KERNEL_ROOFLINE_PROTOCOL_20261016.md) | PROPOSED |
| 2026-10-16 | A1 instrument built (kernels unit-tested with planted controls, PASS); addendum 1 pushed before any admissible timed run. The box is not clean (foreign occupancy median 5.3%, max 24.7% over 30 s: Wallpaper Engine, Antigravity IDE, Chrome), so the timed run waits for the owner (COMMUNICATION.md). | [A1 protocol, addendum 1](C50_A1_KERNEL_ROOFLINE_PROTOCOL_20261016.md#addendum-1-pre-run-16-october-2026-pushed-before-any-admissible-timed-run) | DECISION |
| 2026-10-16 | llama.cpp `5b335f4` builds finished (AVX2 and generic, static, OpenMP). Every tool A2/A2b needs links; only `llama.exe` fails to link because the server is disabled, and nothing here uses it. | `E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/tools/` | MEASURED |
| 2026-10-16 | **Holdout `c50_holdout_v1` built:** FIT/DEV/RESERVED = 144/96/96 spans of 8192 bytes. Roles come from a salted content hash across all three categories. The 230 documents of earlier project holdouts are excluded. 2 spans that NFC would change are skipped (the source tokenizer normalises to NFC). Source-tokenizer IDs round-trip on every span; `verify` PASS. The document machinery is loaded from the committed blob `44cda68` of `build_document_holdout.py`, not from the uncommitted local edits. Decision: "fresh" means no project decision has used these documents. It does not mean they postdate the model (pretraining exposure is probable; A4 carries that). | [A2/A2b protocol](C50_A2B_REFERENCE_FLOOR_PROTOCOL_20261016.md), `benchmarks/chatbot50/corpus/c50_holdout_v1/manifest.json` | MEASURED / DECISION |
| 2026-10-16 | A2/A2b protocol pushed before the imatrix, the quantization and any A2b run. It fixes the imatrix input (all FIT, `-c 512`), four llama.cpp arms (AVX2 t6 ub512 reference; t1; generic; ub1 decode path), the floor (the largest pairwise point estimate) and the external-rate run (clean box, with A1). | [A2/A2b protocol](C50_A2B_REFERENCE_FLOOR_PROTOCOL_20261016.md) | PROPOSED |
| 2026-10-16 | BF16 GGUF converted (32.54 GB, sha256 `2b1c8f72ff25d152…`; 278 tensors). The GGUF vocabulary equals the source tokenizer on every source ID; the GGUF has 31 extra padding IDs that the source tokenizer never emits. BF16 per-token weight bytes: 2,242 MB. | `benchmarks/chatbot50/results/c50_a2_gguf_check_bf16.json` | MEASURED |
| 2026-10-16 | The imatrix stop rule fired: all of FIT projected to ~95 min on BF16 and ~68 min on Q8_0. **Decision (addendum 1, pushed before the run):** keep the BF16 source and halve the input by rule (the first 24 FIT items per category, 161,790 tokens, ~47 min). llama.cpp's text tokenizer differs from the source on 0.07% of that input's tokens. | [A2/A2b addendum 1](C50_A2B_REFERENCE_FLOOR_PROTOCOL_20261016.md#addendum-1-16-october-2026-before-the-imatrix-run) | DECISION / MEASURED |
| 2026-10-16 | A2b addendum 2 pushed before any arm: a runner and floor script that are separate, a negative control (a repeat of R0 that must be bit-identical), and the one-hour stop rule moved into the scorer. Smoke on 159 positions (not evidence): llama.cpp's single-token path differs from its batched path by a mean KL of ~4e-3 nats with a heavy tail. Batched paths of different sizes are bit-identical. | [A2/A2b addendum 2](C50_A2B_REFERENCE_FLOOR_PROTOCOL_20261016.md#addendum-2-16-october-2026-before-any-a2b-arm) | PROPOSED / SMOKE |
| 2026-10-16 | A3 engine `c50_engine.c` written: a GGUF loader and the complete bailingmoe2 decode, on the A1 kernels plus new Q5_K/Q8_0 dots and dequantizers. The Q4_K_M mix needs Q5_K (`attn_qkv`) and Q6_K (half of `ffn_down`, the head). Self-test PASS: AVX2 vs generic vs a dequantize oracle ≤ 3.2e-7, planted 193-199/200. The first self-test version is kept, failing only on its undersized planted control. A3 protocol pushed before any engine output on the real model; the reference is llama.cpp's single-token path (A2b R3). | [A3 protocol](C50_A3_ENGINE_PORT_PROTOCOL_20261016.md), `benchmarks/chatbot50/results/c50_a3_selftest_20261016.txt` | MEASURED / PROPOSED |
| 2026-10-16 | imatrix finished on the BF16 GGUF: 302 chunks, 40 min 34 s, exit 0 (sha256 `77d9a409…`). Two expert tensors have partial coverage: layer 10 (255/256 experts seen) and layer 13 (254/256). `llama-quantize` gives the three unseen experts uniform importance, so they are quantized as without an imatrix. | [A2 result](C50_A2_ACQUISITION_RESULT_20261016.md) | MEASURED |
| 2026-10-16 | **C50-A2 closed.** P0 artifact `ling-mini-2.0-a810f64-q4_k_m.gguf`: 9.91 GB, 4.87 bits per weight, sha256 `44de7315…`. Vocabulary identical to the source on every source ID. Per-token weight bytes: 783.4 MB, against the ledger's 742.2 MB (+5.5%). The difference is that producer Q4_K_M puts half of `attn_qkv` and half of `ffn_down` in Q6_K, while the ledger priced them at Q4_K. With F16 KV, P0 is 867.3 MB/token at 2048. **Correction to the A3 row above:** the mix has no Q5_K tensor (`attn_qkv` is Q4_K on 10 layers and Q6_K on 10). The engine dispatches on the GGUF type, so A3 is unaffected. | [A2 result](C50_A2_ACQUISITION_RESULT_20261016.md), `benchmarks/chatbot50/results/c50_a2_gguf_check_q4_k_m.json` | MEASURED |
| 2026-10-16 | A2b arms started on the P0 artifact (R0, R3, R1, R2, then R0b), each under the one-hour stop rule. | `benchmarks/chatbot50/results/c50_a2b_floor_run.json` (when finished) | RUNNING |
| 2026-10-17 | A2b arms complete (CONFIG confirmed on all five). The first comparison step stopped on arm metadata holding an unescaped Windows path; no pair was computed. **Addendum 3** (pushed before any pair): a tolerant metadata reader and `--compare-only`. The failed attempt is kept. | [A2/A2b addendum 3](C50_A2B_REFERENCE_FLOOR_PROTOCOL_20261016.md#addendum-3-17-october-2026-after-the-arms-before-any-pair-was-computed) | MEASURED / DECISION |
| 2026-10-17 | **C50-A2b floor measured.** Thread count: bit-identical (0). Generic build: top-1 4.10%, KL 0.0149 nats. Single-token path: top-1 3.84%, KL 0.0139. Negative control bit-identical. Floor = 0.0410 / 0.0149, so the A3 limits are top-1 ≤ 0.082 and KL ≤ 0.030. The prediction "0.1-1%" was wrong in size (~4× above its top). The external rate waits for the clean box. | [A2b result](C50_A2B_REFERENCE_FLOOR_RESULT_20261017.md), `benchmarks/chatbot50/results/c50_a2b_floor.json` | MEASURED |
| 2026-10-17 | **Decision (A3 addendum 2, before any engine output on P0):** the registered dBPB form (CI inside ±0.002) fails llama.cpp against itself, contrary to rule 2. It becomes \|dBPB\| point ≤ max(2 × floor_dBPB, 0.002) = 0.00444, with the CI reported. Both floor pairs pass the revised verdict in the engine's place. | [A3 addendum 2](C50_A3_ENGINE_PORT_PROTOCOL_20261016.md#addendum-2-17-october-2026-after-the-a2b-floor-before-any-engine-output-on-p0) | DECISION |
| 2026-10-17 | **C50-A3 PASS.** The engine runs the P0 artifact and, against llama.cpp's single-token path, scores top-1 3.99% (limit 8.20%), KL 0.0144 nats (limit 0.0299) and dBPB +0.0005 (limit 0.0044). The self-test was re-run on the same binary (`d14ea9a5…`). **Disclosure:** under the dBPB form registered first (CI inside ±0.002) the run would have failed, upper CI +0.0027. The form changed by addendum 2, before any engine output and because llama.cpp fails it against itself. The engine is as far from R3 as from R0, the same distance as llama.cpp's own paths from each other. No planted fault has been run through the system gate yet. | [A3 result](C50_A3_ENGINE_PORT_RESULT_20261017.md), `benchmarks/chatbot50/results/c50_a3_run.json`, `c50_a3_verdict.json` | MEASURED |
| 2026-10-17 | A3 addendum 3 pushed before any planted run: two planted-fault engines (routed scale ×0.9; RoPE base 500000) go through the unchanged A3 verdict. Expected: the scale fault FAILS. The RoPE fault has no confident expectation. | [A3 addendum 3](C50_A3_ENGINE_PORT_PROTOCOL_20261016.md#addendum-3-17-october-2026-after-the-a3-verdict-before-any-planted-run) | PROPOSED |
| 2026-10-17 | A3 planted control. The routed scale ×0.9 **FAILS** the gate (KL 0.044 > 0.030, dBPB −0.0084), as registered. The RoPE base 500000 **PASSES** (KL 0.024): a blind spot of the 2×floor gate, though its KL is 1.65× the healthy engine's with non-overlapping CIs. A3 PASS stands, scoped to errors larger than the routed-scale plant. | [A3 result](C50_A3_ENGINE_PORT_RESULT_20261017.md#planted-fault-control-a3-addendum-3-outside-the-verdict), `benchmarks/chatbot50/results/c50_a3_planted.json` | MEASURED |
| 2026-10-17 | **C50-A4 protocol pushed before any registered A4 run.** Three arms on pinned llama.cpp with the same source IDs: P0, BF16, and a planted-positive Q2_K from the same imatrix. Panels: 72 non-imatrix FIT documents (dBPB, roadmap gate +0.02), PIQA 1,838, ARC-Challenge 600, GSM8K 100, IFEval single-instruction subset 173, IT/EN chat 40, rollout 72. Task gate: net loss ≤ 2√(b+c) and P0 ≥ 0.90 × BF16. The imatrix documents are excluded, so the rollout uses 72 documents instead of 96 (deviation). Apparatus smoke done; the P0-vs-P0 null control gives exact zeros. ~13 h projected, so it waits for the owner (COMMUNICATION item 2). | [A4 protocol](C50_A4_QUALITY_PROTOCOL_20261017.md), `benchmarks/chatbot50/results/c50_a4_inputs_manifest.json` | PROPOSED |
| 2026-10-17 | **A5 part 1 protocol pushed before the offline evaluation.** Hidden states are captured by the engine (`--dump-hidden`, records bit-identical to the A3 binary) on the first 12 FIT items per category (79,709 positions). Candidates: low-rank preview (r 128/256/512, 4-bit storage simulated) and IVF (C 256/1024/4096), each with exact rerank. Selection rule from the roadmap: the fewest bytes per token at FIT top-1 recall ≥ 0.999, then confirmed once on DEV. Controls: gguf-py oracle and rank code. | [A5 protocol part 1](C50_A5_APPROXIMATE_HEAD_PROTOCOL_20261017.md) | PROPOSED |
| 2026-10-17 | **A5 part 1 on FIT: FAIL, and the registered prediction is wrong.** Attempt 1 is kept as VOID: its float32 oracle limit ignored the engine's Q8_K activation (addendum 1); the Q8_K-emulated oracle agrees 1.00000 with lse median error 1.6e-6. No configuration out of 48 reaches top-1 recall 0.999. The best preview is r512 4-bit at K=1,024: 0.9348 at 51.2 MB/token. The best IVF is C256 at P=128: 0.7074 at 104.3 MB/token. The known positive fires (full-rank preview 0.99005 at K=1, 1.0 at K=8). The head is not low-rank (r512 holds 44% of ‖W‖²). The preview misses 5% even where p(top1) ≥ 0.9. Reaching 0.999 with r512 needs K ~28k, ~96 MB/token, which puts Ling-mini P4 at ~533 MB, right at B. Next: grid extension part 1b, pushed before it runs; if it fails, the cheaper-donor fallback goes to the owner. | [A5 part 1 result](C50_A5_PART1_FIT_RESULT_20261017.md), `benchmarks/chatbot50/results/c50_a5_select_fit.json`, `c50_a5_diag_fit.json` | MEASURED |
| 2026-10-17 | **A5 part 1b on FIT: SELECTED rot r512 3.5-bit preview, K = 32,768.** Recall 0.99921 at 94.45 MB/token, against 264.1 for the exact head; 33/91 configurations qualify. The prediction (80-110 MB) is right. Runner-up: a full-rank ternary preview at K = 256, 101.0 MB. Ternary storage helps only at full rank. Budget: Ling-mini P4 = 461.0 − 23.8 + 94.5 = **531.7 MB, above B = 530 by 1.65 MB**. Under the registered branch, part 2 still runs and the breach goes to the owner (no action before A6). The selection clears 0.999 by 0.00021 (SE ~0.00011), so the DEV confirmation can fail on sampling alone. | [A5 part 1b result](C50_A5_PART1B_FIT_RESULT_20261017.md), `benchmarks/chatbot50/results/c50_a5_select_1b_fit.json` | MEASURED |
| 2026-10-17 | **A5 part 2: DEV confirmation FAIL, so A5 FAILS.** DEV was read once. rot r512 3.5-bit at K = 32,768 reaches top-1 recall 0.99874 against the 0.999 bar: 98 misses in 78,079 positions, and the Q8_K oracle control passed at 1.0. The predicted range 0.9985-0.9995 was right. The misses fall on code (85) and technical text (13), none on prose. They are confident tokens (median p(top1) 1.00), so this is not tie noise. Per the registered rule there is no automatic move to the runner-up, and DEV is spent. Ledger: P4 with the exact head is 701 MB, 41.8 desk tok/s. Even a passing head would have needed ≥ 0.93 of desk. **Decision to the owner (COMMUNICATION item 3):** bar, donor, or a new FIT protocol with a fresh confirmation set. | [A5 part 2 result](C50_A5_PART2_DEV_RESULT_20261017.md), `benchmarks/chatbot50/results/c50_a5_gate_dev.json` | MEASURED / DECISION PENDING |
| 2026-10-17 | **A5 part 4 (exploratory, FIT only; it decides nothing): head restricted by a script rule fixed from the tokenizer.** Keeping 99,140 of 157,184 rows (Latin/Greek/Cyrillic, symbols, emoji, fragments, specials) makes the exact scan 166.6 MB. FIT recall is 0.9596; every miss is Chinese and comes from five zh-cn Kubernetes documents in FIT, which the protocol had wrongly described as English (prediction scored wrong). On the 30 Latin-script documents the exact restricted scan never misses, and the r512 preview over the kept rows reaches 0.99933 at 81.5 MB. Dropping CJK is a product-scope choice for the owner. It does not fix the budget (P4 ~519 MB, needs ≥ 0.91 of desk). | [A5 part 4 result](C50_A5_PART4_SCRIPT_HEAD_RESULT_20261017.md) | MEASURED |
