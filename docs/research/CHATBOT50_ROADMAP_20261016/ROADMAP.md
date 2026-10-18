# CHATBOT50 roadmap: a speed that does not depend on model size, then a faster base speed, then hardware tiers

16 October 2026, **restructured on 17 October 2026 by the owner's directive** (see the
[decision log](#12-decision-and-progress-log)). The plan approved on 16 October is kept
unchanged in [`ROADMAP_V1_20261016.md`](ROADMAP_V1_20261016.md). Protocols written before the
restructuring cite its section numbers (for example "§5 A1b" or "§6 rule 2"). No historical
result or verdict changes. Sections 0-11 are the 17 October text. Later amendments are made in
place, and each one is recorded in §12, which continues unchanged.

**Status: Phase 1 under way** (the owner's go, 17 October; §12). The A4 → A6 chain launched at
09:17 is still running, and the S0 protocol is pushed (§11).

Labels used throughout: **MEASURED** (a canonical local result), **SOURCE-DERIVED** (headers,
configs, arithmetic), **DESK-MODEL** (rates derived from bytes; never quoted as a rate),
**ESTIMATE** (times, costs or sizes not yet measured), **PROPOSED**, **[L?]** (literature not
re-read in this session; read it in its own tables before it enters a decision).

---

## 0. The decision in one page

**Two problems, decoupled.** The 16 October plan coupled two questions: (1) does the decode
speed stay the same when the model's stored capacity grows, and (2) how high is that speed,
on this machine and on others. Phase A made (2) a precondition of everything (≥ 50 tok/s at
context 2048 before any pipeline work), so (1), the project's actual thesis, had no cell
running. The restructuring orders them:

| Phase | Question | Deliverable | Horizon (ESTIMATE) |
| --- | --- | --- | --- |
| **1** | Does the speed stay at v0 while stored capacity grows? | A reproducible pipeline: a reference artifact T0 at a measured speed v0; the same model at 2× and 4× stored capacity running at ≈ v0, with the routing/lookup overhead priced and measured, and held-out quality rising; then the same pipeline on another family | weeks for the speed half (CPU); months for the quality half (T4) |
| **2** | How high can v0 go? | Organ-by-organ upgrades of the active skeleton (head, KV, expert precision, router, attention, speculation, kernels), each gated for quality. Targets: ≥ 50 tok/s (lower CI95, context 2048), then 100 | days to weeks per organ |
| **3** | What does each machine get? | Hardware tiers: the v0 (bytes per token over bandwidth) and the capacity ceiling N_tot (GB of experts or memory) per CPU and RAM class | per available machine |

**Why now.**
- **A5 took six cells in one day** (parts 1, 1b, 2, 4, 5, 6) to remove ~158 MB/token of head
  bytes. It failed three times: parts 1 and 5 on FIT, part 2 on DEV. The pipeline waited
  behind it.
- **Fifty at 2048 was never one lever away.** Even if every Phase-A lever passed, the desk
  model puts Ling-mini at ~47 tok/s at context 2048 (§3).
- **The engine already runs the exact-head artifact in C** (A3 PASS, A8 chat).

**The physics is unchanged.** Batch-1 decode on the reference Ryzen 5 3600X is
bandwidth-bound: rate ≈ BW_eff / bytes per token.
- **Size independence** therefore means fixed bytes per token while the stored bytes grow.
- **Two separate limits:** capacity is bounded by RAM (80 GB here), speed by bandwidth
  (36.3 GB/s measured here).
- Phase 3 turns these two axes into a product table.

**Where things stand (17 October).**
- **T0 = the P0 artifact:** Ling-mini-2.0 Q4_K_M with the exact Q6_K head, F16 KV and an F32
  router (sha256 `44de7315…`). It is 867.3 MB/token at context 2048, so **34.4 desk tok/s**
  (DESK-MODEL, not a rate).
- **Engine and chat:** the engine matches llama.cpp within llama.cpp's own floor (A3 PASS),
  and Ling-mini chats in C (A8).
- **Quality against BF16 (A4)** is being measured now.
- **llama.cpp** decodes T0 at **14.39 tok/s** at depth 2048 (MEASURED, A2b).
- **The engine's own rate on a clean box is not measured yet.** A1b gives the first engine
  rates; its registered prediction is 24-30 tok/s at context 2048.
  - The "~38-41 tok/s, ~3× llama.cpp" of the directive is the desk rate of the A6 candidates
    (E4B 38.2; P4 with the exact head 41.8 before IQ3_S was repriced), not a measurement.

**What is frozen, what continues.**
- **A5 is frozen as a requirement.** Its state is kept for Phase 2:
  - the part-6 FIT selection (r1024, c = 0, K = 16,384, 106.3 MB/token, recall 0.99967, with
    its arrays frozen by sha256);
  - DEV-B (DEV items 33-44), unread and reserved for its single confirmation. The DEV-B
    capture was cancelled before it started.
- **The A4 → A6 chain continues.** A4 qualifies T0's quality (the 1× baseline). A6 ranks the
  exact-head precision policies, which feeds Phase 2.
- **Moved:**
  - A6 part 2 goes to Phase 2;
  - the rate half of A7 becomes S0 (v0 on T0);
  - A7 itself is the final rate adjudication at the end of Phase 2.

**Asked of the owner:**
- now: the go for Phase 1;
- after the chain: an empty machine for A1b and S0;
- when Phase 1 reaches them: the Ling-flash-2.0 download (S2) and the teacher-signal decision
  (S3);
- for Phase 3: the list of available hardware.

---

## 1. The contract

Inherited, as on 16 October, from
- [RESEARCH_INDEX §2](../RESEARCH_INDEX.md),
- the [scope clarification](../NATIVE_EXPERT_SCALING_20260925/SCOPE_CLARIFICATION_20261015.md),
- and the [original objective](../NATIVE_EXPERT_SCALING_20260925/HANDOFF_SNAPSHOT_20261014/goal-objective.md).

It is regrouped by phase so that every clause has an instrument.

**Phase 1: scale invariance.**

| Gate | Requirement |
| --- | --- |
| Reference artifact | T0 = P0 (`44de7315…`); one container hash and one printed-and-asserted `CONFIG` line in every cell. Engine fidelity: A3 (PASS). Quality 1×: A4 against BF16 (paired document-level dBPB upper CI95 ≤ +0.02 on fresh FIT documents; task and rollout panels with frozen non-inferiority margins). |
| v0 | Accepted tok/s of T0 in the engine: batch 1, declared decode mode, context 2048 primary with 512 and 4096 reported; clean box (foreign occupancy below 4.39%, E52); ≥ 10 fresh processes interleaved; A1b passed first. Reported as central value, lower CI95 and dispersion, with a per-organ time breakdown; never as a desk number. |
| Speed invariance | At stored capacity m ≈ 2 and m ≈ 4 times T0's, with the same top-k and expert size, in both router formats (T0's F32 and the lossless BF16). The bytes that grow with capacity (router rows, keys, lookups) are priced before the run. The measured rate must be explained by v0 and those bytes: the CI95 of the residual ratio v(m) / v_pred(m) lies inside 1 ± τ, with τ fixed before the run. This is an equivalence test, so noise widens the CI and fails it rather than passing it. Reported at each m, not gated: the routing/lookup overhead and the raw ratio v(m) / v(1). The raw ratio always uses a baseline with the same router format; a ratio against the F32 T0 is labelled as including the router upgrade. |
| Quality growth | Held-out quality at m ≈ 2 and m ≈ 4 is better than at m = 1 (CI excluding zero), and monotone in m. It beats a LoRA control with matched trainable parameters and data, by a margin frozen in the protocol. Removing the added capacity, or shuffling it at matched frequency, erases the gain. No regression on the general panel. |
| Pipeline | One reproducible recipe (pinned sources, scripts, manifests) takes a donor to a qualified T0, measures its v0 and grows its capacity. It is applied to T0's family and to ≥ 1 other family, each ending in a documented applicability class. |

**Phase 2: absolute speed.**

| Gate | Requirement |
| --- | --- |
| Rate | The 16 October rate gate, unchanged: accepted tok/s, batch 1, declared decode mode, context 2048 primary with 512 and 4096 reported; clean box; ≥ 10 fresh interleaved processes; **lower CI95 ≥ 50**, with 100 tok/s as the stretch target. TTFT/prefill, RAM and dispersion are reported separately. |
| Quality | Cumulative, against BF16: paired document-level dBPB upper CI95 ≤ +0.02 (preferred ≤ +0.01) on fresh documents, plus donor-relative non-inferiority on the frozen task and rollout panel. Each organ change is also gated against the artifact it replaces. |
| Engine fidelity | The engine against a pinned reference runtime on the identical artifact, within a tolerance derived from that runtime's own measured noise (§6, rule 2). A new runtime configuration (for example flash attention with q8_0 KV) gets its own floor. |
| Invariance kept | Every adopted upgrade re-runs the S1 sweep, so the Phase-1 claim still holds on the faster skeleton. |

**Phase 3: hardware tiers.**

| Gate | Requirement |
| --- | --- |
| Tier record | For each machine measured: the CPU and ISA class; memory type, channels and speed; measured bandwidth; the A1 kernel roofline; A1b stability; the measured v0 of T0 (and of the Phase-2 artifact); the capacity ceiling N_tot (GB) and m_max; and the per-token budgets for 50 and 100 tok/s. |
| Portability | The same source tree builds and runs on every tier. Thread count, core map, cache sizes and memory mode are runtime parameters, never the 3600X's constants (§6, rule 10). |

---

## 2. What the evidence already settles (reused, not rerun)

The 16 October rows are unchanged; the rows from the first Phase-A results (16-17 October)
follow them.

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
| Reference box: Ryzen 5 3600X (Zen 2, 6 cores, AVX2/FMA, no AVX-512/VNNI, L3 2 × 16 MB), DDR4-3200 on two channels (four DIMMs: 2 × 8 + 2 × 32 GB). Its theoretical peak is 51.2 GB/s, so the E30 median is 0.71 of peak | SOURCE-DERIVED (SMBIOS, 2026-10-17) / MEASURED | `Win32_PhysicalMemory`, E30 |
| Kernels at Ling-mini shapes, six workers: Q4_K 35.4-40.4 GB/s, Q6_K head 39.6, Q8_0 38.9-40.2, IQ4_XS 37.7-40.7, **IQ3_S 12.45 (compute-bound)**. One P0 layer takes 903 µs. A 20-layer synthetic token takes 18.54 ms for 562 MB, a non-streaming residual of 2.70 ms (the desk model assumes 2) | MEASURED | [A1 + A2b rate](C50_A1_A2B_RATE_RESULT_20261017.md) |
| llama.cpp against itself on T0: top-1 4.10%, KL 0.0149 nats (generic build); thread counts are bit-identical | MEASURED | [A2b](C50_A2B_REFERENCE_FLOOR_RESULT_20261017.md) |
| llama.cpp decodes T0 at 15.15 / 14.39 / 13.56 tok/s at depth 512 / 2048 / 4096 | MEASURED | A2b rate |
| The engine runs T0 within llama.cpp's floor (top-1 3.99%, KL 0.0144, dBPB +0.0005). A planted routed-scale fault fails the gate; a planted RoPE-base fault does not | MEASURED | [A3](C50_A3_ENGINE_PORT_RESULT_20261017.md) |
| Ling-mini chats in C: the C tokenizer and template match the source with 0 mismatches on five test sets | MEASURED | [A8](C50_A8_CHAT_DELIVERABLE_RESULT_20261017.md) |
| Ling-mini's head is not low-rank (r512 holds 44% of ‖W‖²). No approximate head has been confirmed on DEV at 99.9% recall, and a frequency-selected head cannot reach the bar | MEASURED | A5 parts 1-6 (§12) |
| `llama-quantize` silently ignores an `ffn_gate_inp` override. The F32 router is exactly BF16-representable, so a BF16 router halves its bytes without changing a weight | MEASURED | [A6 protocol](C50_A6_PRECISION_LADDER_PROTOCOL_20261017.md) |
| Ling-flash-2.0's tokenizer is identical to Ling-mini's; Ling-3.x flash is id-compatible except for 9 special tokens | SOURCE-DERIVED | [A0](C50_A0_DONOR_SCREEN_RESULT_20261016.md) |

**What is not settled.** The engine streams at the wall kernel by kernel (A1). What remains
unmeasured is the rate of the complete engine on a clean box (A1b, S0). Two earlier
measurements are the reason to measure it rather than assume it:
- the only earlier complete 4-bit C runtime ran at about half its desk rate;
- complete runtimes showed process-to-process spreads up to 1.63.

---

## 3. The byte budget, T0's organs and capacity accounting

**Model (DESK-MODEL), unchanged.**
- **Formula:** `t_token = bytes_per_token / BW_PLAN + T_FIXED`.
- **Constants:**
  - `BW_PLAN = 32 GB/s`: 0.88 of the E30 median, or 0.625 of the box's theoretical peak;
  - `T_FIXED = 2 ms`; A1 measured a 2.70 ms non-streaming residual on a 20-layer synthetic
    token.
- **Budgets, KV included:** a lower CI95 at 50 tok/s with a ~5% band needs a central time of
  ~18.5 ms, so `bytes_per_token ≤ (18.5 − 2) ms × 32 GB/s ≈ 0.53 GB`. At 100 tok/s the budget
  is ≈ 0.23 GB.

**T0 per token at context 2048** (SOURCE-DERIVED from the GGUF header,
`benchmarks/chatbot50/results/c50_a2_gguf_check_q4_k_m.json`):

| Organ | MB/token | Share | Phase-2 cell |
| --- | ---: | ---: | --- |
| Routed experts (8 of 256 per MoE layer, Q4_K/Q6_K mix) | 288.4 | 33.3% | O3 |
| Head (157,184 × 2048, Q6_K, exact) | 264.1 | 30.4% | O1 (A5, frozen) |
| Attention (QKV and output) | 134.4 | 15.5% | O5 |
| KV cache (F16) | 83.9 | 9.7% | O2 |
| Router (F32) | 39.9 | 4.6% | S1 (BF16, lossless here); O4 (int8, two-level) |
| Shared expert | 36.1 | 4.2% | O3 |
| Dense FFN (layer 0) | 20.4 | 2.4% | O3 |
| Norms and biases | 0.2 | 0.0% | |
| **Total** | **867.3** | | **29.1 ms, 34.4 desk tok/s** |

T0 stores 9.91 GB. Of that, 9.23 GB is routed experts: 4,864 experts in 19 MoE layers, 1.90 MB
each.

**What the Phase-A levers add up to (DESK-MODEL, context 2048).** Each row adds one lever to the
rows above it:

| Step | MB/token | Desk tok/s |
| --- | ---: | ---: |
| T0 | 867.3 | 34.4 |
| + the part-6 head (106.3 MB, selected on FIT only) | 709.5 | 41.4 |
| + q8_0 KV | 670.2 | 43.6 |
| + BF16 router | 650.2 | 44.8 |
| + IQ4_XS routed experts (A6 arm E4B) | 615.8 | 47.1 |

- **With the measured residual:** if the fixed term is A1's 2.70 ms instead of 2 ms, the last
  row is 45.6 desk tok/s.
- **What 50 needs:** ≤ ~528 MB/token at 2048, roughly 90 MB/token more than every Phase-A lever
  removes. That has to come from attention, IQ3-class experts with a faster kernel, or a
  shorter declared context.
- That is Phase 2's problem, not Phase 1's.

**Capacity accounting (SOURCE-DERIVED from T0's header; the S1 protocol re-derives it from the
built artifacts).**
- **Expert bytes stay flat.** Adding routed experts of T0's size and keeping top-8 leaves the
  expert bytes per token unchanged.
- **The router grows.** A flat router reads E × 2048 coefficients per MoE layer for every token.
- **Group sizes.** Expert counts are multiples of 8, so that Ling's eight routing groups stay
  equal.

The raw ratio is the desk speed relative to m = 1 with the same router format. The last column
compares the BF16 expansions with the F32 T0, so it includes the router upgrade.

| Routed experts per layer | Stored GB | m (× T0 stored) | Router MB/token, F32 / BF16 | MB/token at 2048, F32 / BF16 | Raw ratio, F32 router | Raw ratio, BF16 router | BF16 router against the F32 T0 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 (T0) | 9.91 | 1.00 | 39.9 / 19.9 | 867.2 / 847.3 | 1.000 | 1.000 | 1.022 |
| 536 | 20.01 | 2.02 | 83.4 / 41.7 | 910.8 / 869.1 | 0.955 | 0.977 | 0.998 |
| 1,088 | 39.91 | 4.03 | 169.3 / 84.7 | 996.7 / 912.0 | 0.878 | 0.934 | 0.954 |
| 2,048 | 74.52 | 7.52 | 318.8 / 159.4 | 1,146.1 / 986.8 | 0.770 | 0.867 | 0.886 |

- **With T0's F32 router**, capacity costs speed: −4.5% at 2× and −12% at 4× (DESK-MODEL).
- **A BF16 router** is lossless for this checkpoint (§2) and halves the slope: −2.3% at 2× and
  −6.6% at 4× against a BF16 T0. It does not remove the slope: a flat router still loses ~2.35%
  per extra T0-sized pool. S1 measures both formats.
- **Correction (17 October).** The first version of this section said that a BF16 router makes
  2× free relative to T0. That compares a BF16 expansion with the F32 T0, so the +2.2% of the
  format change hides half of the slope. Raw ratios are therefore always taken against a
  baseline with the same router format (rule 9).
- **A sublinear router** (two-level, product keys) bounds the term at larger m. An int8 router
  would cost −3.5% at 4× against an int8 T0, but it is lossy and needs a quality cell (O4).
- **Conditional memory banks with deterministic addressing** read a fixed number of rows per
  token whatever the table size, so their per-token cost is O(1) in capacity. Index, TLB and
  page effects remain to be measured: in METH177 the selected-expert cost rose 1.31× from
  12,800 to 128,000 experts.
- **On this box,** T0's policy hosts m ≈ 7.6: 75.9 GB of the 85.9 GB, leaving ~10 GB for the
  OS, KV and buffers (ESTIMATE). 2× and 4× fit; 8× does not.

**The 16 October donor ledger** (policies P0-P4 over ten donors) is unchanged in
[ROADMAP_V1 §3](ROADMAP_V1_20261016.md#3-the-byte-budget-and-the-donor-ledger) and
[`results/byte_budget_ledger_20261016.txt`](../../../benchmarks/chatbot50/results/byte_budget_ledger_20261016.txt).
Its conclusion still holds: in every family screened, the large members miss the budget
because of their always-on core. Capacity therefore has to be added at a fixed per-token cost,
not found in a large model.

---

## 4. What changed on 17 October

The directive and its reasons are in §0. This table keeps every cell traceable.

| Cell (16 October) | State on 17 October | Place now |
| --- | --- | --- |
| A0 donor screen | closed: Ling-mini-2.0 | T0's donor; its second-family notes feed S6 |
| A1 kernel roofline | closed: FAIL on IQ3_S only | evidence; the IQ3_S kernel is Phase 2 (O7) |
| A1b timing stability | protocol registered, not run (needs an empty box) | Phase 1, prerequisite of S0 (protocol unchanged) |
| A2, A2b | closed | T0, its reference floor, llama.cpp's rate |
| A3 engine port | PASS | qualifies the engine on T0 |
| A4 quality | running (chain) | T0's quality baseline (1×) |
| A5 approximate head | frozen: part 2 FAIL on DEV, part 5 FAIL on FIT, part-6 FIT selection frozen, DEV-B unread, its capture cancelled | Phase 2, O1 |
| A6 precision ladder (FIT) | queued in the chain after A4 | information for Phase 2 (O2-O4); T0 is not replaced in Phase 1 |
| A6 part 2 (DEV and tasks) | protocol registered, not run | Phase 2, O3 |
| A7 final adjudication | not started | its rate half on T0 → S0 (v0); the RESERVED adjudication → end of Phase 2 |
| A8 chat deliverable | done | T0's chat |
| Phase B (pipeline to other LLMs) | not started | Phase 1, S6 |
| Phase C (C0-C3) | not started | Phase 1, S2-S5 |
| `--kv q8_0` engine draft | written after the A1b commit, never built or run | saved outside the repository; Phase 2, O2 |

**Why T0 is P0 and not the A6 winner.**
- **The engine runs only P0's tensor types today:** Q4_K, Q6_K and Q8_0 weights, an F32
  router, F16 KV.
- **The A6 candidates need more:**
  - q8_0 KV, IQ4_XS or IQ3_S experts and a BF16 router in the engine;
  - a new reference floor for llama.cpp's flash-attention path.

  That is days of organ work, which is what Phase 2 is for.
- **Phase 1's claim is a ratio,** v(m) / v0, and it does not need the fastest T0.
- **Every Phase-2 upgrade that passes then raises v0** for T0 and for its expansions alike
  (§7).

---

## 5. Phase 1: the scale-invariance pipeline

Experiment IDs keep the prefix `C50-`; the Phase-1 cells are S0-S6.
- **Protocols:** each cell gets its own protocol, pushed before its run (§6). The outlines below
  fix the question, the gate and the stop rule.
- **Order:** S1 and S2 are independent and can be prepared in parallel. They run one at a time,
  because S1 needs an empty machine and S2 is a long CPU job.

**S0. T0 and its measured speed v0** (CPU; ~1.5 h of empty machine plus ~1 day of tooling,
ESTIMATE).
- **Qualification, existing or running:**
  - A3 PASS: engine fidelity on T0;
  - A8: chat in C;
  - the A4 verdict: quality against BF16, the 1× baseline;
  - the A6 verdict: the FIT ranking of exact-head precision policies, recorded for Phase 2.
- **If A4 fails,** T0 becomes the cheapest higher-precision exact-head artifact that passes
  (Q5_K_M, then Q6_K), and the rest of Phase 1 is unchanged.
- **Stability:** A1b, as registered. Gate: max/min of the 10 per-process medians ≤ 1.06. The
  adopted arm fixes the memory mode and priority of every later rate cell.
- **v0, the rate half of the old A7, on T0:**
  - ≥ 10 fresh interleaved processes per context, at 512, 2048 and 4096, on a clean box;
  - reported: central value, lower CI95, dispersion, TTFT/prefill and RAM.
- **Organ breakdown:** an instrumented engine option, default off and with byte-identical
  output, times each organ per token: attention, KV, dense, shared, router, routed experts,
  head, and the residual. This list is Phase 2's work order.
- **Outcome:** v0 is a measurement, not a target. Without an A1b PASS it is reported as medians
  and spread, and no invariance gate can rest on its CI.

**S1. Speed invariance with capacity only** (CPU, no training; ~1 week, ESTIMATE).
- **Question:** does the engine's time per token stay at v0, apart from the priced
  routing/lookup bytes, when stored capacity grows at fixed active bytes?
- **Engine changes**, each default off and printed on the `CONFIG` line:
  - the expert count, today a compile-time 256 asserted against the header, becomes a header
    value. At E = 256 the output stays byte-identical to A3's;
  - a BF16 router. The F32 router rows are converted to BF16 at load time, and the load exits
    if any row is not exactly representable. The dot expands them back to F32 in the F32
    kernel's accumulation order (one 8-lane FMA chain), so the logits are identical to the bit
    by construction. No extra GGUF is needed.
- **Known positive:** at m = 1 the BF16 router's output is byte-identical to T0's (`cmp` on the
  A3 positions). If it is not, the BF16 arm is VOID and S1 runs on F32 only.
- **Arm E (expert pool),** in both router formats (F32 and BF16), interleaved in one session:
  - expanded GGUFs at m ≈ 2, at m ≈ 4, and at the largest m that stays resident with ~10 GB
    free (≈ 7 here; **corrected 18 October to m ≈ 6**, E = 1,600 and 58.6 GB, because 71.7 GB of the 80 GB are free at idle; [S1 protocol](C50_S1_CAPACITY_INVARIANCE_PROTOCOL_20261018.md));
  - the extra experts are byte copies of T0's experts at new addresses. Their router rows are
    copies with a fixed perturbation, exactly representable in BF16, so that selections spread
    over the whole pool;
  - quality is not read, because the copies add no knowledge.
- **Arm H (hashed copies):** the same expanded GGUFs, with the router kept at T0's 256 rows.
  Each selected expert is sent to one of its byte-identical copies by a fixed hash. The router
  bytes do not grow, so arm H measures the pool effect alone (TLB, page and L3). Its output
  must be byte-identical to T0's at every m: a second known positive.
- **Arm M (memory bank):** a lookup microbenchmark (k rows per token from tables of 1-60 GB,
  deterministic addresses, synthetic values). It prices the conditional-memory route before S4
  chooses between the two.
- **Measured:**
  - v(m) with its CI, per arm and format;
  - the desk prediction from the built artifacts' bytes, and the router's own time;
  - the pool effect, from arm H;
  - the router's real cost per byte in each format, from arm E minus arm H. This box streams
    different formats at different bandwidths (§2, E60/E61), so the two formats are not assumed
    to cost the same per byte. Without arm H they would measure only a combination of their two
    costs, which a pool effect growing with E could also absorb;
  - consistency: each format's router cost must come out constant across m.
- **Gate:** the §1 equivalence test in each format, with τ fixed in the protocol before the
  run (default 3%). It is passed or failed at m ≈ 2 and m ≈ 4.
- **Reported, not gated:**
  - the routing overhead at each m;
  - the raw ratio v(m) / v(1) in each format, against the m = 1 arm of the same format in the
    same session;
  - the BF16 expansions against the F32 T0, labelled "includes the router upgrade".

  No product threshold on the raw ratio is set; if one is wanted, it is fixed before S1 runs.
  For example, ≥ 0.95 at 4× would already exclude the flat BF16 router on desk (0.934).
- **Resolution.** At 2× the BF16 effect (desk −2.3%) is close to the CI half-width that process
  dispersion alone may produce.
  - The number of processes per arm is set after A1b, from its measured dispersion.
  - The format effect can also be measured paired inside one process, by alternating blocks of
    F32 and BF16 steps; the protocol decides which.
- **Stop:** if the residual fails the test, the diagnostic budget (rule 3) applies before any
  T4 spend. The quality half would otherwise rest on a speed claim that does not hold.
- **After S1:** if the BF16 known positive passes, the later Phase-1 artifacts (S4, S5) use the BF16
  router. T0 stays P0, the qualified container, and v0 with the BF16 router is reported beside
  v0.

**S2. Gap map** (the old C0; CPU; needs the Ling-flash-2.0 download, which the owner approves).
- **Arms:** T0 against Ling-flash-2.0.
  - Ling-flash is pinned by revision and converted with the pinned llama.cpp. A Q4_K_M is
    ~63 GB at Ling-mini's 4.87 bits per weight (ESTIMATE) and fits in RAM.
  - Both arms are fed the same source-tokenizer IDs; the tokenizers are identical.
- **Panels, frozen in the protocol:** closed-book knowledge, reasoning (short math, ARC), chat,
  and fresh-document BPB. The documents are held out and never used for training or selection.
- **Decision rule:**
  - if the gap is mostly knowledge, stored capacity is the right tool, and S3-S5 proceed;
  - if it is mostly reasoning, active compute is the right tool, and the quality half of
    Phase 1 is re-planned before any T4 spend.
- **Outputs:**
  - a donor-gap panel for S4-S5, restricted to items where Ling-flash clearly beats Ling-mini
    (the margin is set in the protocol);
  - a general panel.

**S3. Teacher signal** (the old C1; owner decision).
- **The constraint:** Ling-flash does not fit a T4x2 at 4 bits.
- **The options:**
  - top-k logits from the local CPU, which is slow and occupies the benchmark machine;
  - sequence-level outputs from a hosted endpoint, after checking its terms of use and cost;
  - an id-compatible Ling-3.x teacher;
  - offline construction of memory values, where the teacher runs once per bank row and not
    once per training token (the project's
    [memory-grafting review](../PIPELINE_REASSESSMENT_20261015/CARVING_STRATEGY_REVIEW_20261015.md)).
- **What the choice fixes:** the realistic distillation volume, and which S4 arms are
  affordable.
- **Before the owner decides,** S3 is sized in tokens/s and in GB.

**S4. Falsifier at m ≈ 2** (the old C2; T4). On the frozen T0 base, with matched data.
- **Arms:**
  - (a) extra routed experts, with new router rows;
  - (b) a conditional memory bank: deterministic addressing, values built offline from the
    teacher and frozen, plus a trained read-out and gate;
  - (c) a LoRA control with the same trainable-parameter count.
- **Memory plan, shown before any spend.**
  - Optimizer state is charged on total trainable parameters: about 16 B each under
    mixed-precision Adam.
  - Arm (a) at the full m = 2 would train ~15B new parameters. Its protocol must therefore use
    a recipe that fits 2 × 16 GiB of GPU and ~30 GiB of host RAM: staged per-layer training,
    frozen banks, or experts taken from the teacher through trained adapters.
- **Gate:**
  - (a) or (b) beats (c) on the S2 donor-gap panel by a margin frozen in the protocol. One
    form, from the review: recovering ≥ 20% of the donor-minus-T0 gap, with the number fixed
    before results.
  - No regression on the general panel.
  - On the same checkpoint, removing the capacity, shuffling it at matched frequency, or masking
    entries at equal lookup count erases the gain.
  - The engine passes S1's invariance gate on the trained artifact.
- **Literature anchors** [L?]: Lifelong-MoE, Memory Layers at Scale, Engram and Memory Grafting.
  - The last two were examined in the project's
    [large-family gate](../PIPELINE_REASSESSMENT_20261015/LARGE_FAMILY_GATE.md) and
    memory-grafting review.
  - Their warning: a large table is not the same amount of useful knowledge, so the controls
    above are mandatory.
- **Stop:** if neither (a) nor (b) beats (c) beyond the margin, capacity transfer is closed at
  that budget. It is recorded as such, not as a general impossibility.

**S5. Scaling to m ≈ 4** (the old C3).
- **What runs:** the S4 winner, at m ≈ 2 and m ≈ 4.
- **Gate:**
  - held-out gains are monotone in m, and the controls still erase them;
  - the speed-invariance gate holds at each m, with the routing/lookup overhead reported;
  - every cost that grows with capacity is measured: router rows grow as E × D per layer, keys
    as ~√N, and index and page effects as measured.
- This is the first point where "quality up at ≈ v0" becomes a measured claim.

**S6. The pipeline as a recipe, and a second family** (the old Phase B).
- **Recipe:** pinned HF checkpoint → pinned converter → precision policy → engine container with
  an architecture descriptor → reference floor (A2b) → engine fidelity (A3) → quality (A4) →
  v0 (S0) → capacity sweep (S1) → capacity training (the S4/S5 recipe). Each stage has one
  script and one manifest.
- **Operator registry,** grown only as donors need it: GQA/MLA/SWA, QK-norm, partial RoPE/YaRN,
  dense and MoE FFN, router variants, shared experts, and short-conv or SSM mixers.
- **Second family,** chosen by a rule in the S6 protocol. A0's notes:
  - Trinity-Nano-Preview is the cheapest with reported scores (OpenMDW-1.1 licence);
  - EuroMoE-2.6B-A0.6B is the cheapest per token but reports no scores;
  - Ling-3.0-tiny would exercise KDA/MLA;
  - GigaChat 3.1 stays the test case for the Lossy class.
- **Applicability classes:**
  - **Direct:** passes the Phase-1 gates at a quality-safe policy.
  - **Lossy:** passes only with healed lossy levers, needed when the donor misses the tier's RAM
    or a declared speed floor.
  - **Capacity-transfer:** serves as the teacher of a fixed-active recipient.
  - **Unsupported:** with the measured reason.

  Unlike 16 October, the classes no longer depend on the 50 tok/s budget, which becomes a
  per-tier property (Phases 2-3).
- **Gate:** at least one other family goes end to end through the recipe, including S0 and S1,
  with a documented class.

**Phase 1 exit (§1):**
- v0 measured (S0);
- speed invariance at m ≈ 2 and m ≈ 4 (S1, re-checked on S5's trained artifacts);
- quality growth (S4, S5);
- the recipe reproduced on ≥ 1 other family (S6).

**Registered predictions (to be scored, not gates).**

| Cell | Prediction |
| --- | --- |
| A1b | As registered: B0 max/min 1.04-1.15, B1 ≤ 1.06, medians 24-30 tok/s at 2048 |
| S0 | v0 at 2048 is 0.70-0.90 of T0's desk (24-31 tok/s), and v0 at 512 is 1.05-1.20 × v0 at 2048 |
| S1 | At m ≈ 4 the raw slowdown is 10-16% with the F32 router and 5-9% with the BF16 router; the residual that bytes do not explain is ≤ 3% in both formats; the BF16 router is byte-identical to T0 at m = 1 |
| S2 | The Ling-mini → Ling-flash gap is larger on closed-book knowledge than on short math |

---

## 6. Programme rules

1. **Complete path first.** No component optimization before a complete P0 forward runs with
   its full per-organ cost breakdown. T0 satisfies the first half (A3, A8); S0's breakdown
   completes it.
2. **System fidelity is statistical and calibrated.** The engine is judged against a pinned
   reference on the same artifact.
   - The tolerance is derived from the reference's own measured disagreement across builds and
     threads (A2b), and is set before the engine is measured.
   - Per-kernel bit-exactness against a scalar oracle stays a unit test; it is never a cascade
     gate.
3. **Diagnostic budget.** After any gate failure: at most three diagnostic cells or three working
   days, then a written design review that changes the coordinate or stops the line.
4. FIT selects, DEV confirms once, RESERVED adjudicates the final artifact. No threshold moves
   after DEV is read.
5. Protocols of discretionary gates are pushed before their run; `CONFIG` is printed and
   asserted; VOID runs are kept and never relabelled.
6. Desk rates are never quoted as rates; every claim carries its label.
7. **Amended 18 October (owner):** a run expected to exceed one hour needs the owner's go only
   when it needs an empty machine, that is when it reads a rate or a timing. Other long runs
   (quality cells, builds, corpora) are launched without asking, one long job at a time, each
   recorded in §12. T4 work is announced first, with its reason, budget and stop rule.
8. **NEW (17 October): no organ blocks the pipeline.** A component that fails its gate is frozen
   with its state recorded (the selection, the unread sets, the artifacts and their hashes),
   and the pipeline continues on the exact component.
   - Resuming it is a Phase-2 cell with its own protocol.
   - The lesson is A5: six cells in one day on ~18% of T0's bytes.
9. **NEW: invariance is a priced decomposition.** Every capacity cell prices, before its run, the
   bytes that grow with capacity (router rows, keys, indices, page tables), and reports the
   measured residual.
   - "≈ v0" is never written without both numbers.
   - A raw ratio is always taken against a baseline with the same skeleton (for example the same
     router format), so that an organ upgrade cannot pass for invariance.
10. **NEW: hardware numbers are runtime parameters.** Thread count, core map, cache sizes and
    memory mode are read or configured per machine. The engine's compiled-in core map
    (`cores=0,2,4,6,8,10`) becomes a parameter in Phase 3, with a default that stays
    byte-identical on the 3600X.

---

## 7. Phase 2: modular optimization of the organs, to raise v0

**Principle.** T0's skeleton is a list of organs with measured bytes and times (S0).
- An upgrade to one organ applies to T0 and to every expansion alike, because the expansions
  share the skeleton and change only the stored capacity.
- This is checked, not assumed: re-running S1 on the upgraded engine or artifact is part of
  every adopted upgrade.

**Rules per organ cell:**
- FIT selects, DEV confirms once, and RESERVED is not read before A7.
- The engine is checked against the matching reference runtime, with a new floor for each new
  runtime configuration.
- The rate is measured on a clean box.
- The quality gate is set against the artifact it replaces and, cumulatively, against BF16.

| Cell | Organ (share of T0 at 2048) | State at the freeze | Next step | Desk gain at 2048 |
| --- | --- | --- | --- | --- |
| O1 | Head (30.4%) | A5 frozen. Part-6 FIT selection: r1024, c 0, K 16,384, 106.3 MB/token, FIT12 recall 0.99967, with `part6_r1024_{vr,preview,norms}.npy` frozen by sha256. DEV-B (DEV 33-44) unread. Parts 1, 1b, 2, 4 and 5 recorded | DEV-B capture (~40 min) and its registered confirmation; then the C head (A5 part 3) | −157.8 MB (T0 → 41.4 desk tok/s) |
| O2 | KV cache (9.7%) | A6 arm K8 scored on FIT (chain); engine `--kv q8_0` draft saved, not built | port; a floor for llama.cpp's flash-attention q8_0 path; an A3-style gate | −39.3 MB |
| O3 | Routed, shared and dense experts (39.9%) | A6 FIT ranking (chain); A6 part 2 registered; IQ3_S compute-bound at 12.45 GB/s | A6 part 2 (~7-8 h); IQ4_XS and IQ3_S in the engine; a faster IQ3_S kernel | E4: −34.5 MB; E3 needs the kernel |
| O4 | Router (4.6%) | exactly BF16-representable for this checkpoint; the lossless BF16 router moved into S1 on 17 October | an int8 router (lossy, so a quality cell) or a two-level router for large m | BF16 (in S1): −19.9 MB and half the capacity slope; int8: −29.9 MB |
| O5 | Attention (15.5%) | post-hoc rank cuts were expensive (E65, first-attention results) | a protocol that prices quality per MB saved | open |
| O6 | Speculation / MTP | not a primary lever: routed experts do not amortize under i.i.d. routing (a 2-position verify costs 1.43×), and the chat checkpoint has no MTP head | only if S0's breakdown or a routing-locality measurement changes this | open |
| O7 | Kernels, threading, prefill | non-streaming residual 2.70 ms (desk 2); prefill token by token | barriers and synchronization; batched prefill (TTFT) | ≤ 0.7 ms/token |

**A7, the final adjudication, moves here.**
- **Quality:** the RESERVED documents and the frozen task and rollout panel.
- **Rate:** clean box at 512/2048/4096 with ≥ 10 interleaved processes; RAM, TTFT and prefill
  rate recorded.
- **Artifact:** the final one, i.e. the upgraded T0 or, if Phase 1's quality half passed, its
  m ≈ 4 expansion.

**Targets.** ≥ 50 tok/s lower CI95 at 2048 (the Phase-2 rate gate), then 100. The desk sum of
the Phase-A levers is 47.1 tok/s (§3), so 50 needs O5, or O3 with a faster IQ3_S kernel.

---

## 8. Phase 3: hardware tiers

**Question.** Memory bandwidth sets v0, and past some point the CPU's compute does. RAM size
sets capacity. For each class of machine: what v0 does T0 reach, what N_tot (GB of experts or
memory) fits, and what does the Phase-2 artifact reach?

**H0. Inventory (owner).** The machines available and those worth renting: CPU model; memory
type, speed, channels and size; OS.

**H0 inventory, given by the owner on 17 October** (besides R0; rentals if needed):

| Machine | CPU: cores/threads, ISA | RAM as given | Peak GB/s if dual channel | T0 (9.91 GB) resident |
| --- | --- | --- | ---: | --- |
| ASUS VivoBook Max X541U | Core i5-7200U, Kaby Lake, 2/4, AVX2 | 8 GB DDR4-2133 | 34.1 | no |
| HP 15s-eq0020nl | Ryzen 5 3500U, Zen+, 4/8, AVX2 on a 128-bit datapath | 8 GB DDR4-2400 | 38.4 | no |
| Lenovo ThinkPad T490s | Core i5-8265U, Whiskey Lake, 4/8, AVX2 | 16 GB DDR4-2400 | 38.4 | yes, ~6 GB left for the OS |
| Microsoft Surface Pro 11 | Snapdragon X Elite, 12 Oryon cores, ARM64 (NEON, no AVX2) | 16 GB LPDDR5x | 135 (LPDDR5x-8448, 128-bit bus) | yes, ~6 GB left for the OS |

- **The peaks are upper bounds (SOURCE-DERIVED from the vendors' specifications).** The channel
  and module layout of the laptops is not known. H1's read ceiling measures the bandwidth
  instead of assuming dual channel.
- **The two 8 GB machines cannot hold T0 resident.** They form a tier below the planning
  table's smallest row. Serving them needs a smaller resident artifact or experts streamed from
  the SSD, and Phase 1 builds neither. Until that changes, their H1 pack is the read ceiling,
  the A1 roofline and A1b.
- **The Surface is the list's only ARM64 machine and its highest peak** (about 2.6× R0's). The
  A3 engine is x86 AVX2 C, so the portability gate needs an ARM64 kernel path before our engine
  can measure that tier's v0. An x64 build run under emulation is not a measurement of the tier.
  A native ARM64 build of the pinned llama.cpp can give the reference runtime's number.

**H1. Probe pack per machine** (~2-3 h of machine time each, ESTIMATE):
- E30-style read ceiling;
- A1 kernel roofline at Ling-mini shapes;
- A1b stability;
- S0 v0 of T0, and of the Phase-2 artifact once it exists;
- S1 arm E at the machine's largest resident m.

Every number keeps its label; until a tier is measured, its row is DESK-MODEL.

**H2. Runtime parameters.**
- Topology detection: physical cores, CCX/CCD and NUMA nodes, L3 size.
- Thread map and memory mode chosen at run time (rule 10).
- The 3600X defaults stay byte-identical.

**H3. Tier table.** The measured rows, and the product rule: speed is bought with bandwidth,
capacity with gigabytes. A tier is a (memory class, RAM size, CPU class) triple. Rows of
unmeasured tiers are planning numbers.

**Planning table (DESK-MODEL).**
- **Bandwidth only:** compute and NUMA are not modelled.
- **BW_PLAN = 0.625 × theoretical peak,** calibrated on the reference box only.
- **Capacity** at T0's policy, with 6-16 GB left for the OS (ESTIMATE).

| Tier (example) | Peak GB/s | BW_PLAN GB/s | T0 desk tok/s at 2048 | Budget at 50 / 100 tok/s (MB/token) | N_tot (GB) | m_max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DDR4-3200 dual channel, 16 GB | 51.2 | 32 | 34.4 | 528 / 232 | ~11 | 1 |
| DDR4-3200 dual channel, 32 GB | 51.2 | 32 | 34.4 | 528 / 232 | ~26 | 2 |
| **R0, this box: DDR4-3200 dual channel, 80 GB** | 51.2 | 32 (E30 median 36.3) | 34.4 | 528 / 232 | ~76 | 7 |
| DDR5-5600 dual channel, 64 GB | 89.6 | 56 | 57.2 | 924 / 406 | ~59 | 5 |
| DDR5-6400 dual channel, 96 GB | 102.4 | 64 | 64.3 | 1,056 / 464 | ~93 | 9 |
| Workstation, DDR4-3200 4 channels, 128 GB | 102.4 | 64 | 64.3 | 1,056 / 464 | ~125 | 12 |
| Server, DDR4-3200 8 channels, 256 GB | 204.8 | 128 | 114.0 | 2,112 / 928 | ~259 | 26 |
| Server, DDR5-4800 12 channels, 384 GB | 460.8 | 288 | 199.6 | 4,752 / 2,088 | ~396 | 39 |

- **Peak** = channels × 8 bytes × MT/s (SOURCE-DERIVED). The 0.625 factor is the reference box's
  ratio of planning bandwidth to peak, not a law; other memory controllers differ.
- **At high bandwidth the fixed term and the kernels' compute limit dominate.** IQ3_S is already
  compute-bound at 12.45 GB/s on six Zen 2 cores. The high-bandwidth rows are upper bounds until
  measured.
- **Multi-socket and NUMA machines** need per-node placement of the experts, which is not
  modelled.
- **Capacity rows** count 9.91 GB per m at T0's policy. Denser storage of the added capacity
  (IQ3/IQ4 experts, int8 memory rows) raises m_max for the same GB.

**Phase 3 does not wait for Phase 2.** H0 and the probe pack can run on any machine as soon as
it is available, and the table fills as tiers are measured.

---

## 9. Risk register

| Risk | Where it shows | Fallback |
| --- | --- | --- |
| A4 fails (T0 quality above +0.02) | S0 | T0 = Q5_K_M or Q6_K with the exact head; v0 re-measured; the invariance cells are unchanged |
| Process-to-process dispersion | A1b | placement and paging fixes within the diagnostic budget; no CI-based gate until solved |
| The router grows linearly with E: at 4×, −12% desk with F32 and −6.6% with BF16 (same-format baseline) | S1 | report it (rule 9); the BF16 router measured in S1; an int8 or two-level router (O4) at large m |
| Residual slowdown at large pools (TLB and pages; METH177 1.31×) | S1 | large pages (A1b arm B2) if the privilege is granted; expert placement |
| The Ling-mini → Ling-flash gap is reasoning, not knowledge | S2 | stored capacity is the wrong tool; the quality half of Phase 1 is re-planned before T4 spend |
| Teacher-signal throughput | S3 | sequence-level KD; offline bank construction in batches; an id-compatible Ling-3.x teacher |
| Training 2× capacity on T4: ~15B new trainable parameters at m = 2 (Adam ~16 B per parameter) do not fit 2 × 16 GiB | S4 | frozen banks with a trained read-out; staged per-layer training; experts taken from the teacher through trained adapters; a smaller m first |
| The training stack for bailingmoe2 (custom modelling code shipped with the checkpoint) | S4 | review it before use, or re-implement it from the A3 engine's verified semantics |
| The added capacity is not useful (S4 no better than LoRA) | S4 | capacity transfer closed at that budget, recorded as such |
| Phase 2: the head's recall bar, compute-bound IQ3 kernels, attention cost | O1-O5 | each organ stays frozen until its own cell passes; the Phase-1 claim does not depend on them |
| Access to hardware for the tiers | H0 | the owner's machines, then rentals |

---

## 10. Resources and timeline (ESTIMATES)

**Phase 1:**
- S0: ~1.5 h of empty machine plus ~1 day of tooling.
- S1: ~1 week. That covers the engine generalization, the BF16 router option (~half a day), an
  expander, 20-75 GB artifacts on E:, and ~2-4 h of empty machine (both router formats; the
  process count is set after A1b).
- S2: several days of contended CPU, for the Ling-flash-2.0 download (~206 GB in BF16, then a
  ~63 GB Q4_K_M), its conversion and the panels.
- S4-S5: weeks to months of T4, within the granted 90 h/week.
- S6: 1-2 months, with one or two further downloads of 15-25 GB.

**Phase 2:** days to weeks per organ. O1 resumes with ~1 h of CPU; A6 part 2 takes ~7-8 h.

**Phase 3:** ~2-3 h of machine time per tier, plus access to the machine. Disk is ample (E:
5.2 TB free on 16 October).

---

## 11. State at the restructuring, and next steps

**Running:** the A4 → A6 chain.
- A4 runs with `--budget-h 14`, then A6 with `--budget-h 6`, never two long runs at once.
- At 10:20 the P0 document panel had finished, in 2,714 s.
- Nothing else runs; the DEV-B capture was cancelled.

**The owner gave the go for Phase 1 on 17 October.** The steps, in order:
1. The A4 and A6 verdicts. **A4: NOT PASSED, with an uninformative planted control**
   ([result](C50_A4_QUALITY_RESULT_20261017.md)); an A4-bis design follows (rule 3); **its protocol is pushed (18 October)** ([A4-bis](C50_A4BIS_POSTRELEASE_PROTOCOL_20261018.md)); **A4-bis: A4 PASS, T0 = P0 qualified** ([result](C50_A4BIS_POSTRELEASE_RESULT_20261018.md)). A6: FIT candidate E4B.
2. The S0 protocol, pushed before any rate run: **done**
   ([C50_S0_V0_PROTOCOL](C50_S0_V0_PROTOCOL_20261017.md)). **A1b: PASS, B0 adopted**
   ([result](C50_A1B_RESULT_20261018.md)). S0 follows A4-bis.
3. The S1 protocol, the engine generalization (byte-identical at E = 256), the BF16 router option
   (byte-identical at m = 1) and the expander; then the S1 sweep in both router formats. **The protocol, its runner, report and arm-M bench are pushed (18 October)**
   ([C50_S1 protocol](C50_S1_CAPACITY_INVARIANCE_PROTOCOL_20261018.md)); the expansions are built and pinned
   by an addendum; the session (~4.5-5 h of empty machine) follows S0 and S2.
4. The S2 protocol and the Ling-flash-2.0 download (owner approval): **done**. The owner gave
   "ok S2" on 17 October. M0 and F0 are built and pinned (F0 by the owner, 18 October); **the S2 arms run from 18 October ~14:00**, before S0 (row of 18 October: S0 waits for A4-bis).
5. S3 (an owner decision), then S4.

The owner's actions are listed in `docs/COMMUNICATION.md`.

---

## Appendix A: compact no-duplication cards

The 16 October cards (A1-A7) are unchanged in
[ROADMAP_V1, Appendix A](ROADMAP_V1_20261016.md#appendix-a-compact-no-duplication-cards). The
new cells:

| Cell | Nearest prior cell | Changed coordinate | Why the prior cell does not answer |
| --- | --- | --- | --- |
| S0 | A2b rate (llama.cpp on T0); the recipient int8/mixed baselines | The engine's own rate on T0, with an organ breakdown | llama.cpp is a different runtime; the recipients are different models |
| S1 | METH177 (selected-expert cost at extreme pool sizes); E63 Part B (synthetic 10B) | A pretrained artifact's pool grown 2-7× at fixed top-k, in two router formats plus a hashed-copy arm, so that the pool effect and the router's cost per byte are measured separately | METH177 used synthetic shapes at 12,800-128,000 experts; E63 had no growing pool |
| S2 | none on this family | Ling-mini against Ling-flash on frozen panels | no prior cell compared the two |
| S4 | H1/H2I/H4 (router training at 1.5B); the memory-grafting review | Capacity added to a frozen pretrained MoE, against a matched LoRA control | the earlier cells trained routers or small recipients, not capacity at a fixed cost |
| S6 | the 16 October Phase B outline | A recipe that includes v0 and capacity growth | the outline stopped at rate and quality |
| H1 | E30, A1, A1b | The same probes on other machines | they measured one machine |

## Appendix B: reproduce the ledger

```
python -I benchmarks/chatbot50/byte_budget_ledger.py --show-sources \
  --gguf benchmarks/donor_adaptation/density/results/strat01_gigachat_q4_97045b2/GigaChat3.1-10B-A1.8B-q4_K_M.gguf \
  --json benchmarks/chatbot50/results/byte_budget_ledger_20261016.json
```

It uses the standard library only and reads configs and headers, never tensor values.
- **External facts it uses,** read from official Hugging Face pages on 2026-10-16 and to be
  re-pinned by revision in A0:
  - the Ling-mini card: MIT, 16.26B total, 1.43B activated, 789M non-embedding, MTP only for the
    base model;
  - the Ling-flash, LFM2 and Trinity-Nano configs.
- **Excluded:** landscape items found only through secondary sources (Qwen3.5/3.6 MoE sizes,
  Ling 2.5/3.0, Trinity Mini/Large) are not used in any decision here.

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
| 2026-10-17 | **C50-A6 protocol pushed before any arm is scored; the four GGUFs are built and checked.** Six arms on one fixed runtime (A6 scorer, flash attention on): R (P0, KV f16), K8 (P0, KV q8_0), E3/E4 (routed experts IQ3_S/IQ4_XS), E3B/E4B (plus a BF16 router). The head stays exact because A5 failed. Addendum 1: `llama-quantize` silently ignores a `ffn_gate_inp` override, so the first E3B/E4B builds were byte-identical to E3/E4 and are kept as VOID. They are rebuilt by a checked F32→BF16 router rewrite. The routers are exactly BF16-representable, so P4 halves router bytes (39.8 → 19.9 MB) without changing a weight. Weights per token: P0 783.4, E3 700.4, E4 748.9, E3B 680.5, E4B 729.0 MB. With KV at context 2048, no arm reaches B = 530 (best E3B, 725.0 MB, 40.6 desk tok/s). The FIT scoring run (~4.6 h) waits for the owner's A5 decision (COMMUNICATION item 3). | [A6 protocol](C50_A6_PRECISION_LADDER_PROTOCOL_20261017.md), `benchmarks/chatbot50/results/c50_a6_run.json` | PROPOSED / MEASURED (bytes) |
| 2026-10-17 | **C50-A8 protocol pushed before any verification is run.** A C tokenizer reads vocabulary, token types and merges from the GGUF (the 156,635 GGUF merges equal the source list). It covers added tokens leftmost-longest, NFC (tables from Unicode 15.0), a hand-written matcher of the source Split regex, byte-level BPE and decode. It comes with the chat template (system/user/assistant, thinking off, no tools) and `c50_chat.exe` on the A3 engine. Verification against HF `tokenizers` 0.22.2 and `apply_chat_template`: FIT texts, A4 chat prompts, a seeded 4,500-string stress set, decode of 2,000 random ID sequences, 12 conversations. The bar is 0 mismatches. A8 needs no long run and no owner action. | [A8 protocol](C50_A8_CHAT_DELIVERABLE_PROTOCOL_20261017.md) | PROPOSED |
| 2026-10-17 | **A1 addendum 2 (after a VOID pre-flight, before any timed process).** The owner's empty box failed the pre-flight on a single one-second sample (4.49%; mean 0.56%). The rule compared a 45-sample max with a bar calibrated on a window mean, so it could not be met on idle Windows. The pre-flight is now judged on its mean, like the per-process gate, which is unchanged. Attempt 1 is kept as VOID. | [A1 protocol, addendum 2](C50_A1_KERNEL_ROOFLINE_PROTOCOL_20261016.md), `benchmarks/chatbot50/results/c50_a1_run_1.json` | DECISION |
| 2026-10-17 | **C50-A1: FAIL on IQ3_S only.** 5 admissible processes (4 VOID replaced). Q4_K streams 35.4-40.4 GB/s on every shape; Q6_K head 39.6; Q8_0 38.9-40.2; IQ4_XS 37.7-40.7. IQ3_S on the eight-expert shapes reaches 12.45 GB/s (compute-bound). G-A1b PASS: one P0 layer 903 µs against 1,017. T20: 18.54 ms for 562 MB, non-streaming residual 2.70 ms (desk assumes 2). The Q4_K prediction 30-36 is partly wrong (the attention shapes are faster). Consequence fixed before the run: no re-plan; A6 prices IQ3_S at 12.45 GB/s (A6 addendum 2), so E3/E3B fall to ~28 desk tok/s and E4B (38.2) is the fastest arm on paper. **A2b rate:** llama.cpp decodes P0 at 15.15 / **14.39** / 13.56 tok/s at depth 512/2048/4096 (5/5 processes, spread < 1%), i.e. 12.5 GB/s effective at 2048; the prediction ≤ 30 is right. | [A1 + A2b rate result](C50_A1_A2B_RATE_RESULT_20261017.md), `benchmarks/chatbot50/results/c50_a1_verdict_2.json`, `c50_a2b_rate_1.json` | MEASURED |
| 2026-10-17 | **C50-A8 verification attempt 1: PASS.** The C tokenizer and template match the source with 0 mismatches on T1 (144 FIT texts), T2 (40 chat prompts), T3 (4,500 stress strings), T4 (6,684 decodes, 2,000 of them random ID sequences) and T5 (12 conversations, text and IDs). The 500 unassigned-code-point strings also match (reported, not gated). The registered prediction (≥ 1 T3 mismatch) is wrong. Planted controls, added after the pass and outside the gate, each fire: no NFC gives 1,046 T3 mismatches but 0 on FIT, because FIT texts are NFC-stable by construction; removing `\s+(?!\S)` gives 123 FIT and 280 T3 mismatches; a wrong lossy-UTF-8 rule gives 327 T4 mismatches. Loop smoke (T6) pending. | [A8 protocol](C50_A8_CHAT_DELIVERABLE_PROTOCOL_20261017.md), `benchmarks/chatbot50/results/c50_a8_verify_attempt1.json`, `c50_a8_verify_plant_*.json` | MEASURED |
| 2026-10-17 | **Owner decisions:** A4 authorized; after the A5 FAIL, option A: keep Ling-mini, measure first, which authorizes the A6 FIT scoring run. Launched detached as one chain: A4 (`--budget-h 14`), then A6 (`--budget-h 6`), never two long runs at once. Under option A, two new A5 routes get their own FIT protocols before they run: a head restricted to the vocabulary actually used, and a confirmation on a set not yet read. | COMMUNICATION (owner), chain `E:/…/a4_then_a6.cmd` | DECISION |
| 2026-10-17 | **C50-A8 deliverable done: Ling-mini-2.0 chats in C** (`bin/c50_chat.exe`, A3 engine unchanged, C tokenizer, template). T1-T5 pass again after the streaming refactor (attempt 2). T6 smoke on 3 fixed prompts: it runs, stops at EOS (2) or the limit (1), and prints CONFIG. Answers: 'La capitale del Portogallo è Lisbona.', a correct Rayleigh explanation, and a Fibonacci answer cut at 64 tokens. Reported against `c50_llama_generate` on the same prompt IDs: 2 of 3 greedy continuations are identical; one diverges at token 42 of 52. Prefill is token by token (correct, slow; TTFT is A7). | [A8 result](C50_A8_CHAT_DELIVERABLE_RESULT_20261017.md), `benchmarks/chatbot50/results/c50_a8_t6.json` | MEASURED |
| 2026-10-17 | **A5 part 5 protocol pushed before any statistic** (owner option A). The head is restricted to the N most frequent target IDs of a selection corpus no evaluation reads: FIT-role items ranked 49-248 per category, built by the pinned holdout machinery with a proof check that ranks 1-48 and 1-32 are unchanged. The scan over the kept rows is exact. Selection: the smallest N on a 9-point grid with FIT12 engine top-1 recall ≥ 0.9995 (margin fixed now). Confirmation, read once: DEV-B (DEV items ranked 33-44) at ≥ 0.999. Prediction: N 32,768-49,152, about 56-83 MB. | [A5 part 5 protocol](C50_A5_PART5_USED_VOCAB_HEAD_PROTOCOL_20261017.md) | PROPOSED |
| 2026-10-17 | **C50-A5 part 5: FAIL on FIT, DEV-B not read.** The holdout extension passes its proof check (regenerated FIT 1-48 and DEV 1-32 equal the committed hashes). On FIT12 (79,709 positions) the used-vocabulary head reaches recall 0.876 at N 8,192 (14.4 MB), 0.983 at 49,152 (83.0 MB) and 0.995 at 98,304 (165.6 MB); no grid N reaches the 0.9995 selection bar, so DEV-B is not scored. The prediction (N 32,768-49,152) is wrong. Diagnostic, after the verdict: SEL (1.34 M tokens) holds 46,764 distinct IDs, and keeping all of them gives 0.982; the missed top-1 are multilingual pieces and code identifiers that never occur in SEL. A frequency-selected head cannot meet the gate on this mix; the route is closed. | [A5 part 5 result](C50_A5_PART5_USED_VOCAB_HEAD_RESULT_20261017.md), `benchmarks/chatbot50/results/c50_a5_used_vocab_fit.json`, `c50_holdout_ext_v1_manifest.json` | MEASURED |
| 2026-10-17 | **C50-A5 part 6 protocol registered (option A, second route).** It targets the measured part-2 failure mode, confident tokens whose head row lies outside the preview subspace. The candidate score adds a residual-norm bonus: preview_v + c x |res_v| x |h_perp| (0.31 MB of float16 norms), and the top K are reranked exactly. Grid: r {512, 1024} q3sim, c in {0..0.4}, K 2,048-32,768. Control: c = 0 must reproduce part 1b to 1e-9. Selection: the fewest bytes with FIT12 recall >= 0.9995. Disclosed: the part-1b point r1024 K 16,384 (0.99967, 106.3 MB) already qualifies. Confirmation once on the unread DEV-B at >= 0.999, captured after the A4 -> A6 chain. Prediction: r512, c 0.05-0.2, K 8,192-16,384, 53-68 MB. | [A5 part 6 protocol](C50_A5_PART6_RESIDUAL_BONUS_PROTOCOL_20261017.md) | PROPOSED |
| 2026-10-17 | **C50-A6 part 2 protocol registered** before any A6 FIT number exists (the A6 run is queued behind A4). The FIT candidate C and R (P0, f16 KV) both go through one runtime (A6 scorer plus a new A6 generator with printed --kv/--fa, sha `e4592d2f`; flash attention on). Panels: DEV items 13-32 per category (60 documents, 130,621 tokens) and the A4 task panels, copied after a hash check. Gates: DEV dBPB upper CI95 <= +0.015, plus the A4 task and rollout rules with R as reference; CONFIRMED iff all pass. A FAIL goes to the owner, never to another arm by itself. About 7-8 h; it needs the owner's authorization and runs after the A4 -> A6 chain. | [A6 part 2 protocol](C50_A6_PART2_DEV_TASKS_PROTOCOL_20261017.md) | PROPOSED |
| 2026-10-17 | **C50-A5 part 6 FIT: selection frozen, DEV-B not yet read.** The control passes (c = 0 reproduces part 1b at all 10 points). The residual-norm bonus helps only at small K: r1024 at K 2,048 goes from 0.97900 to 0.98576 with c = 0.05. It does not lower the cost at the 0.9995 bar, and c >= 0.1 hurts (r512 c = 0.4 at K 32,768: 0.912). Selected: **r1024, c = 0, K = 16,384, 106.3 MB/token, FIT12 recall 0.99967**, the disclosed part-1b point. The prediction (r512, c 0.05-0.2, 53-68 MB) is wrong. The preview, V_r and norm arrays are frozen by sha256 in the record. DEV-B is captured once after the A4 -> A6 chain. Desk: E4B with this head is 21.25 ms (47.1 tok/s), not a rate. | [A5 part 6 protocol](C50_A5_PART6_RESIDUAL_BONUS_PROTOCOL_20261017.md), `benchmarks/chatbot50/results/c50_a5_residual_fit.json` | MEASURED |
| 2026-10-17 | **C50-A1b protocol registered.** Levers added to the engine as default-off, printed options in a separate binary (`c50_engine_a1b.exe`, sha `6cbafb90`): `--mem private` (GGUF image copied into private memory locked in the working set), `--mem large` (large pages) and `--prio high`. Parity before the push: map and private with high priority are byte-identical to the A3 binary on 199 positions. Large pages are unavailable (SeLockMemoryPrivilege not held). Arms B0 (A3 path), B1 (private + high) and B2 (only if the privilege exists), interleaved over 10 rounds of fresh `--bench` processes at context 2048, with the E52 occupancy bar. Gate: max/min of the per-process medians <= 1.06; the first passing arm in order B0, B1, B2 is adopted. Prediction: B0 1.04-1.15, B1 PASS, rates 24-30 tok/s. It needs an empty machine (~30 min) after the chain. | [A1b protocol](C50_A1B_TIMING_STABILITY_PROTOCOL_20261017.md) | PROPOSED |
| 2026-10-17 | **Owner directive: the roadmap is restructured into three phases.** Phase 1 is a scale-invariance pipeline at a measured v0, Phase 2 an organ-by-organ optimization that raises v0, and Phase 3 a set of hardware tiers. A5 is frozen with its state recorded: the part-6 FIT selection and arrays, and DEV-B unread. The A4 → A6 chain continues, which is the first of the directive's two options for running jobs, because it qualifies the exact-head artifact. The DEV-B capture watcher (`after_chain_devb.cmd`) was stopped before it started, after its PID was checked against its command line; nothing was captured. T0 = P0 (§4). A6 part 2 moves to Phase 2, and the rate half of A7 on T0 becomes S0. The 16 October text is kept as `ROADMAP_V1_20261016.md`. Work is paused until the owner's go. | owner directive in session, 2026-10-17; [ROADMAP_V1](ROADMAP_V1_20261016.md) | DECISION |
| 2026-10-17 | **Labels for the record.** The directive's "~38-41 tok/s with the exact head, ~3× llama.cpp" is the desk rate of the A6 candidates (E4B 38.2; P4 41.8 before IQ3_S was repriced), not a measurement. T0 is 34.4 desk tok/s at 2048. The engine's own decode rate has not been measured yet: A1b is the first measurement, with a registered prediction of 24-30 tok/s. llama.cpp measured 14.39 tok/s on T0. | §0, §3; [A1 + A2b rate](C50_A1_A2B_RATE_RESULT_20261017.md) | DESK-MODEL / MEASURED |
| 2026-10-17 | The `--kv q8_0` engine draft was written after the A1b commit and never built or run. It is saved outside the repository (source, diff and patch script) and stashed, and the working tree is back at the committed source. That source rebuilds `c50_engine_a1b.exe` byte-identical apart from the link timestamp, which the PE file stores twice, so the source hash the A1b runner records matches its binary. The draft is the starting point of O2. | `E:/SiliconLLM_Artifacts/campaigns/chatbot50_20261016/drafts/` | DECISION / MEASURED |
| 2026-10-17 | **Plan amendment (owner note on the BF16 router).** S1 measures the expert pool in both router formats, T0's F32 and the lossless BF16, interleaved in one session. The BF16 router is a load-time engine option whose output at m = 1 must be byte-identical to T0's; otherwise the BF16 arm is VOID. A hashed-copy arm H keeps the router at T0's 256 rows and sends each selected expert to one of its copies, so it measures the pool effect alone, with output byte-identical to T0's at every m. Arm E minus arm H gives each format's router cost per byte; the formats are not assumed to share one, because this box streams formats at different bandwidths. The raw ratio v(m) / v(1) is reported, not gated, against a baseline of the same format: desk 0.977 at 2× and 0.934 at 4× with BF16, against 0.955 and 0.878 with F32. Ratios against the F32 T0 are labelled as including the router upgrade. **Correction:** the first 17 October version of §3 said that a BF16 router makes 2× free relative to T0. That mixed the format change (+2.2%) with capacity; it is corrected in place. The lossless half of O4 moves into S1, and rule 9 now requires a same-skeleton baseline. If the known positive passes, S4 and S5 use the BF16 router; T0 stays P0. | owner note in session, 2026-10-17; §3, §5 S1, §6 rule 9, §7 O4 | DECISION / DESK-MODEL |
| 2026-10-17 | **The owner gives the go for Phase 1** ("via Fase 1"), with no correction to the plan. T0 = P0 therefore stands as written in §1 and §4: the go answered the open question of COMMUNICATION item 1, which said that a different T0 had to be asked for. At the go, A4 was still running (since 09:17). Step 1 of §11, the A4 and A6 verdicts, waits for the chain, and step 2 started. | owner in session, 2026-10-17 | DECISION |
| 2026-10-17 | **S0 protocol pushed before any S0 or A1b process.** v0 at 512, 2048 and 4096: 10 interleaved fresh processes per context, with the contexts rotated per round. The memory mode and priority come from the A1b adopted arm; if A1b fails, the complete arm with the smallest max/min is used and v0 is labelled UNSTABLE. The engine gets `--organs` (default off): an organ clock of laps on the main thread, with the routed and shared experts as separate jobs. It also gets per-organ byte counts and the peak working set and commit. The output is byte-identical to A3 in four configurations (`cmp` on 199 positions, the full log-probability rows included). A planted binary computes the head twice: the breakdown is VALIDATED only if the median head ratio is ≥ 1.7 and the other organs stay inside [0.90, 1.12]. A mechanics check on the busy box showed why the second half is needed: head ×2.16, the rest ×1.44. A1b addendum 1 (tooling only) freezes the A1b source as `c50_engine_a1b.c` and pins its binary, because S0 extends `c50_engine.c`. The session is ~1.5-1.75 h of empty machine (ESTIMATE). | [C50_S0_V0_PROTOCOL](C50_S0_V0_PROTOCOL_20261017.md); [A1b addendum 1](C50_A1B_TIMING_STABILITY_PROTOCOL_20261017.md#addendum-1-17-october-2026-before-any-a1b-process) | PROTOCOL / MEASURED (parity) |
| 2026-10-17 | **S1 tools, before the S1 protocol** (which waits for A1b's dispersion and adopted memory mode). Engine: the expert count E is read from the header (multiple of 8, ≥ 32 per group, ≤ 4,096); `--router bf16` converts the F32 router at load time, exits 5 if any value is not exactly BF16, and expands it back to F32 in the F32 kernel's FMA order; `--hash-copies` (arm H) reads only T0's 256 router rows and sends each pick to one of its copies by splitmix64 of (layer, position, expert). The bench line now reports the selection spread. On T0 (E = 256) all five combinations are byte-identical to A3, including the log-probability rows: default, BF16, H, BF16 + H, and BF16 + H + `--organs` + private memory. This previews the BF16 known positive, which S1 runs on the A3 positions. `c50_s1_expand.py` grows each pool by byte copies (slot g·E/8 + 32k + j = copy k of T0 expert 32g + j). The router rows of copies k ≥ 1 get a seeded multiplicative perturbation rounded to BF16, and every slot is re-checked against its source. On a synthetic GGUF it passes independent checks at E = 256, 264 and 536, and its E = 256 output is byte-identical to the source file. **Constraint found for the S1 protocol:** E: is a USB disk and C: (NVMe) has 139 GB free, so an m ≈ 7 artifact (~70 GB) cannot be re-read per process under `--mem private`. The protocol must choose between page-cache residency (`--mem map`) and NVMe staging for the large arms. | `c50_engine.c`, `c50_s1_expand.py` | TOOLING / MEASURED (parity) |
| 2026-10-17 | **The owner leaves the machine free and authorizes the downloads.** Two consequences. (1) The A1b + S0 session is launched by me once the box is clean: after the A4 → A6 chain, with A4's verdict first, as the S0 protocol requires. It is never run during the chain or during a download. (2) The S2 download of Ling-flash-2.0 starts now, during the chain, so that it ends long before the session. It is pinned at revision `96f9b7a73e531b0ca05f8839ea9f271948ad8ea4` (MIT, not gated, 22 safetensors, 205.79 GB from the API), with the same `hf download --revision --local-dir` path as A2, and it will be verified by `a2_source_manifest.py`. **Measured before the download:** the pinned `tokenizer.json` differs from Ling-mini's in bytes but is equal as canonical JSON (vocabulary 156,891, merges, 262 added tokens, pre-tokenizer and normalizer). `tokenizer_config.json` (chat template included) and `special_tokens_map.json` are byte-identical. S2's "same source-tokenizer IDs" therefore holds. Ling-flash config: 32 layers, hidden 4,096, 256 experts, top-8, expert FFN 1,024, 32/4 heads, vocab 157,184, 1 dense layer, 8 groups, top-4 groups, 1 shared expert. Its GGUF conversion is a long CPU job and waits for the S2 protocol and an empty slot after S0. | owner in session, 2026-10-17; HF API at the pinned revision | DECISION / SOURCE-DERIVED |
| 2026-10-17 | **S2 protocol pushed before any S2 arm is built.** The arms are M0 (Ling-mini Q4_K_M, no imatrix) and F0 (Ling-flash-2.0 `96f9b7a` Q4_K_M, the same recipe), with T0 reported beside. No imatrix is a measured constraint: the 205.8 GB BF16 neither fits the RAM nor reads fast from E:, so the gap map compares two artifacts made one way (rule 9), and T0 − M0 is the imatrix effect. Panels: NQ-open 600 and TriviaQA 600 (closed-book short answers, SQuAD-normalized whole-word alias match), MMLU 570 (10 per subject), and A4's GSM8K 100, ARC 600, 72 documents and chat 40. The knowledge datasets are pinned by revision and sha256. Rule: D = RER_K − RER_R (K = nq + tqa, R = gsm8k + arc) with an item bootstrap. KNOWLEDGE, REASONING or MIXED by the CI95 of D; S3-S5 proceed iff the outcome is not REASONING and the CI95 of RER_K is above 0. Controls before the push: planted synthetic cases pass (knowledge-only, reasoning-only, equal, no gap), a null control (T0 against itself) gives exact zeros on the real pipeline, and 15/15 known-answer checks pass. Disclosure: the smoke printed P0's own accuracies on A4's panels before the A4 verdict; they enter no decision. Cost ~18 h of CPU (ESTIMATE). It runs only with the owner's explicit go and never beside a rate cell. | [C50_S2_GAP_MAP_PROTOCOL](C50_S2_GAP_MAP_PROTOCOL_20261017.md) | PROTOCOL |
| 2026-10-17 | **Owner answers: "ok S2", no product threshold, hardware inventory.** S2 is authorized (rule 7) and runs after the A1b + S0 session, in the protocol's order, never beside a rate cell. No product threshold on the raw ratio v(m)/v(1) is set, as §5 already states. The H0 inventory is recorded in §8: two 8 GB laptops that cannot hold T0 resident, a 16 GB AVX2 laptop, and a 16 GB ARM64 Surface whose tier needs an ARM64 kernel path. Rentals are available if needed. The owner offered to free space on C:. Not needed: F0's Q4_K_M (~63 GB) is staged on the NVMe C: for the panels, S1's large arms are staged there later, one at a time, within the 139 GB free, and the 205.8 GB BF16 intermediate stays on E:. The owner keeps at least 100 GB free on C: until S1 ends. The runner's F0 path and pin are set at build time. | owner in session, 2026-10-17 | DECISION |
| 2026-10-17 | **Ling-flash-2.0 download complete and verified.** It was paused while A4 cold-loaded its BF16 model from E: and resumed at 13:17, once that model was resident; it finished at 14:35. `a2_source_manifest.py` at the pinned revision `96f9b7a`: PASS, 32 files, 0 failures (sha256 for the LFS files, git blob ids for the others), 205,792,399,972 bytes. The manifest (`sources/Ling-flash-2.0_96f9b7a.manifest.json`, sha256 `9c1b579a…`) is kept on E: beside the snapshot. No downloaded code was executed. | `a2_source_manifest.py`; `sources/verify_ling_flash.log` | MEASURED |
| 2026-10-17 | **C50-A4: NOT PASSED, and the planted positive passes too.** All 21 panels COMPLETE (09:17-21:57). P0 against BF16: G1 dBPB −0.0063, CI95 [−0.0097, −0.0032], PASS; G2-G5 and G7 PASS. G6 chat is INCONCLUSIVE: BF16 itself scores 27/40, under the 0.75 ability floor. The registered prediction (+0.005..+0.020) is MISSED. The planted Q2_K (PC) fails nothing: G1 −0.0076, upper CI +0.0017. Registered consequence: a P0 PASS would be uninformative, so a design review follows (rule 3). Diagnostic 1 (post hoc, recorded outputs only): PC's whole G1 gain comes from the 1.1% of positions where BF16's target log-probability is below −15; outside them PC is worse by +103 bits. Top-1 agreement with BF16 is 0.93 for P0 and 0.85 for PC. So G1 on FIT text cannot see a Q2_K-sized loss. The "if A4 fails" branch (Q5_K_M) does not apply: no gate measured a P0 loss, and both defects would block any candidate. T0 = P0 stays the working reference, unqualified. Next: an A4-bis protocol (fresh documents, a per-position fidelity gate derived from an existing floor, PC must fail it first, chat floor fixed by rule), which needs the owner's go. S0 waits for it; A1b and the S2 builds do not depend on it. | [A4 result](C50_A4_QUALITY_RESULT_20261017.md); `results/c50_a4_*` | MEASURED / DECISION |
| 2026-10-18 | **C50-A6 FIT selection (part 1), as registered: the FIT candidate is E4B.** Against R = P0 on the 72 A4 documents: K8 +0.0003, E4 +0.0011 and E4B +0.0023 (upper CI +0.0046) qualify under the +0.015 bound; E3 +0.0121 and E3B +0.0125 do not. E4B is the cheapest at a desk 26.17 ms per token at 2048 (R 29.10; DESK-MODEL). Top-1 agreement with R: K8 0.957 (a KV-format change alone), E4 and E4B 0.929, E3 0.898. Caveat from A4 diagnostic 1: the same document statistic did not see a Q2_K loss against BF16, so this ranking is a FIT selection only. A6 part 2 (DEV confirmation, task panels) stays in Phase 2. | `results/c50_a6_verdict.json` | MEASURED |
| 2026-10-18 | **C50-A1b: PASS, arm B0 adopted** (`--mem map --prio normal`; every later rate cell uses it). Attempt 1 VOID and kept: pre-flight 5.54% over the 4.39% bar, from the Visual Studio Installer's background download, which was not killed. Attempt 2 (03:37-03:56, pre-flight 0.91%): B0 max/min 1.0522, B1 1.0498 (gate ≤ 1.06), 10 processes each, 0 replaced; B2 not run (no lock-pages privilege). Per-process medians at context 2048 on P0: B0 36.99..38.92 tok/s (mean 38.32), B1 36.27..38.08 (mean 37.26). These are the first clean-box engine rates, not v0. Predictions: stability HIT on both arms; rate 24-30 tok/s MISSED, upward; within 3% HIT. | [A1b result](C50_A1B_RESULT_20261018.md) | MEASURED |
| 2026-10-18 | **A4 design review, step 1 (rule 3); no new model output read.** Fresh FIT ranks 249-272 were built (72 documents, 161,506 tokens, proof check against ranks 1-248 OK). The fidelity gate proposed on 17 October (top-1 disagreement ≤ 2 × floor_top1 = 0.0820) is **withdrawn before registration**: on A4's own outputs it would fail P0 (0.0735, upper CI 0.0825), whose top-1 accuracy equals BF16's (0.6723 against 0.6726). It measures difference, not degradation. The likely cause of G1's blindness is the text: outside the tail, Q2_K moves BPB by only +0.0002, and A2b's numerics-only pairs also gave negative point dBPB. On memorised FIT text, likelihood does not see quantization. Direction: A4-bis on text written after Ling-mini-2.0's release, with the roadmap dBPB gate unchanged and the planted Q2_K required to fail it first. The tools (`c50_a4_run.py --panels/--cell`, `c50_holdout_ext.py --parts fresh`, `c50_a4bis_verdict.py`) are committed but not registered. | [A4 result, design review](C50_A4_QUALITY_RESULT_20261017.md#design-review-step-1-18-october-no-new-model-output-was-read) | DECISION |
| 2026-10-18 | **S2 builds.** M0 (Ling-mini-2.0 `a810f64`, `llama-quantize` Q4_K_M from the A2 BF16, no imatrix) built 03:59-04:18, sha256 `1f545368…`, pinned in `c50_s2_run.py`. F0 convert: attempt 1 failed at import under `python -I` (the converter imports its own `conversion` package from the script directory); attempt 2 under `python -s -E` (the A2 recipe; remote code refused) ran 04:19-08:35, exit 0, BF16 GGUF 205,851,495,776 B on E:. F0 quantize (Q4_K_M, no imatrix, 6 threads, output on C:) was **stopped by the session host's memory-pressure reaper at 09:06**, at tensor 151 of 446; not a command failure, and the host's rule forbids an unasked restart. The partial output (20.25 GB) was removed. The relaunch waits for the owner's word (COMMUNICATION item 1). Order: S2 runs before S0, because S0 waits for A4-bis and S2 does not depend on T0's qualification; one long job at a time. | `E:/…/s2_builds.status`, `gguf/convert_flash_bf16.log`, `gguf/quantize_flash_q4_k_m_noimatrix.log` | PROGRESS |
| 2026-10-18 | **F0 quantize attempt 2 and the A4-bis corpus build, both stopped by the session host's memory-pressure reaper at 10:57.** The quantize was relaunched at 10:27 on the owner's word ("puoi rilanciare la quantizzazione"). It stopped at tensor 161 of 446; attempt 1 stopped at 151. Both stops came ~30 min into a quantize, and no cause is claimed. The partial output was removed. The corpus build (`c50_postrelease_corpus.py`, network only, started 09:18) was in its Wikipedia phase and wrote nothing. The builder now caches every API response under its corpus directory, so a rerun resumes; its selection rules are unchanged. Next: the owner runs the quantize from a terminal outside the session (COMMUNICATION item 1). | `E:/…/s2_builds.status`, `E:/…/a4bis_post_build.log` | PROGRESS |
| 2026-10-18 | **S1 protocol pushed, before any S1 process.** Gate: the §1 equivalence test per router format, ρ_f(m) = v_E,f(m) / v_pred,f(m), with v_pred priced from the m1 arm and the m1 router lap per byte only, CI95 inside [0.97, 1.03] at m2 and m4 (τ = 3%). 20 processes per gated arm, because A1b's B0 CV of 2.05% gives a ±2.1% half-width on a ratio of two 10-process means. Arms: E and H in both formats at m1, m2 (E = 536), m4 (1,088) and m6 (1,600, reported); arm M (memory-bank lookups from 1-48 GB tables) reported. 230 processes, ~4.5-5 h (ESTIMATE). σ = 0.05, seed 20261016. **Correction to §5:** the largest resident pool is m ≈ 6 (58.6 GB), not ≈ 7; 2,048 experts (74.8 GB) do not fit beside the 8 GB the OS uses. The rebuilt `c50_engine_s1.exe` (`79e2e4f4…`) is byte-identical to the A3 binary on `par1.tsv` in four configurations (default, BF16 router, hashed copies, both). The report's logic was checked on synthetic runs: a router priced exactly gives PASS; a planted 10% unpriced pool effect gives FAIL in both formats. Staging needs ≥ 160 GB free on C: (owner, later). | [S1 protocol](C50_S1_CAPACITY_INVARIANCE_PROTOCOL_20261018.md) | DECISION / MEASURED (parity) |
| 2026-10-18 | **H2 (Phase 3, which ROADMAP §8 lets start before Phase 2), step 1: the thread count becomes a runtime parameter.** `c50_engine.c` gains `--threads N` (1-32, default 6 = the 3600X; the first N physical cores, printed as threads= and cores=). A first version split attention's positions over N threads; at N = 4 it moved the outputs by numerics alone: top-1 disagreement 5.0%, mean KL 0.0245 nats, dBPB +0.0005 on `par1.tsv`, comparable to A2b's build-change floor (4.10%, 0.0149). It was replaced before any commit: attention now always uses A3's six position chunks, spread over the N threads and combined in chunk order. **Measured: the outputs are byte-identical to the A3 binary at N = 1, 2, 4 and 6**, and with `--organs` and with the BF16 router plus hashed copies (position records and the 199 × 157,184 log-probability rows, `cmp`). So a machine with fewer cores runs the same bits as R0, and Phase 3 needs no new fidelity floor for the thread count. Selftest PASS; `--bench` runs at N = 4 (a mechanics check, no rate). The pinned cell binaries (A3, A1b, S0, S1) are not rebuilt. Above six threads the extra threads idle in attention. | `c50_engine.c` | TOOLING / MEASURED (parity) |
| 2026-10-18 | **S1 addendum 1 (before any S1 process): arm M uses the existing `c50_s1_bank.c` (`7a89b83`, 17 October); the `c50_s1_membank.c` written with the protocol is withdrawn as a duplicate.** The protocol was written without searching for an arm-M tool that already existed. The existing one is also the closer model of the engine: k rows per layer over 19 layers, one barrier per layer. Its grid is fixed now: 1-48 GB tables, k ∈ {2, 8, 32} per layer, 3 processes each. The binary was rebuilt from the unchanged source (`b076ddae…`). | [S1 protocol, addendum 1](C50_S1_CAPACITY_INVARIANCE_PROTOCOL_20261018.md#addendum-1-18-october-2026-before-any-s1-process-tooling-only) | CORRECTION / TOOLING |
| 2026-10-18 | **H2 step 2: the A1 roofline takes `--threads N` too** (`c50_a1_roofline.c`; 1-32, default 6, the first N physical cores, printed as `threads=` and `cpus=`), so H1's kernel and layer cells can run on the owner's laptops. The pattern is the engine's: the matvec, expert and SwiGLU splits follow N, and the attention cell keeps six position chunks spread over the N threads. **Checked:** selftest PASS (same lines as the committed source rebuilt); at the default the CONFIG line is byte-identical to the committed source's; `--quick` runs at N = 6 and N = 2 emit the same 14 cells with the same non-timing fields. The roofline prints no output checksum, so unlike step 1 there is no bit-parity check here; its timings from these smokes are not rates (back-to-back, `--quick`). The A1 result keeps its own recorded source hashes. `c50_s1_bank.c` is not changed: S1 pins its binary (`b076ddae`), and H1 can build its own copy when it needs one. Build note: the source compiles with the llvm-mingw clang (`gcc` on PATH); WinLibs GCC stops at the STRAT-01 header's `#error` (clang target attributes required), so the header comment's 'WinLibs GCC' does not build the current source. The committed source rebuilt with clang does not reproduce the hash of `bin/c50_a1_roofline.exe` (0d4c0a0f vs 833aca83); which build produced that binary is not recorded here, and A1's result is not re-read. | `c50_a1_roofline.c` | TOOLING |
| 2026-10-18 | **H2 step 3: topology detection, the first item of every machine's probe pack** (`c50_topology.py`, new). Searched first: the existing readers (`meth308`, `meth359`, the `pick_cores` of the C tools) read physical cores only, and the Python ones assert six cores; none reads L3 domains or NUMA nodes. The tool reads `GetLogicalProcessorInformationEx` through ctypes (pure Python, so it runs natively on the ARM64 Surface, which has no engine build yet) and writes physical cores with SMT and efficiency class, L3 domains (CCX/CCD) with their cores, NUMA nodes, packages, the per-level cache sizes, RAM and the core order that `--threads N` takes. It changes no binary. **Known positive, measured on R0:** 6 cores, 12 logical processors, two 16 MiB L3 domains of cores {0,1,2} and {3,4,5}, one NUMA node, one package: `--expect-r0` PASS, and the first-logical order 0,2,4,6,8,10 equals the engine's printed `cores=`. The check discriminates: four planted mutations (one 32 MiB L3, 4 cores, 2 NUMA nodes, no SMT) each FAIL with the named field. So on R0, `--threads 3` keeps every thread on one CCX and `--threads 4` crosses into the second, a fact the run-time thread map (H2's next item) has to account for. Output: `results/c50_h2_topology_r0.json`. | `c50_topology.py` | TOOLING / MEASURED (R0 topology) |
| 2026-10-18 | **Decision: H2's run-time thread map is deferred; `--threads N` keeps taking the first N cores.** A map that spreads threads over L3 domains changes anything only on a machine with more than one domain and fewer threads than cores. Of the listed machines, the multi-domain ones are R0 (two 16 MiB domains, measured in step 3; its default is 6 threads, all cores) and the Surface (whose cluster layout is unmeasured, and which has no engine build). The three x64 laptops are expected to have one L3 each (unmeasured until `c50_topology.py` runs on them). So no tier the engine can run today would use the map, and choosing a policy would need timings on R0 that do not answer a product question. Reopened when a laptop's topology output shows more than one L3 domain, or when the ARM64 engine path exists. H2's memory mode is already a run-time parameter (`--mem`, A1b). Remaining Phase 1 work is unchanged and waits on the owner: the F0 quantize (S2) and the A4-bis corpus build, both killed by the host reaper and not restarted without the owner's request. | - | DECISION |
| 2026-10-18 | **F0 built by the owner from a terminal outside the session, pinned; S2 launched.** Attempt 3 of the Q4_K_M no-imatrix quantize (COMMUNICATION command A) finished at 13:44: 446 of 446 tensors, 196,309 MiB (16.01 BPW) to 59,527 MiB (4.85 BPW), 5,962 s. The log's only 'error' line is PowerShell 5.1 wrapping the tool's first stderr line (`version: ...`) as `NativeCommandError`, not a tool failure. F0 = `C:/SiliconLLM_stage/s2/ling-flash-2.0-96f9b7a-q4_k_m-noimatrix.gguf`, 62,425,484,640 B, sha256 `5348c73f...`, pinned in `c50_s2_run.py` before any F0 panel. **Decision: S2 starts while the owner's A4-bis corpus build (command B) is still running.** The 'one long job at a time' rule exists for RAM and for rate timings. The build is network-only (40 MB working set, 70 GB RAM free), and S2 is a quality cell that measures no rate; its only time-dependent rule is the 3 h projected-time stop per panel. A4-bis and S0 then wait for S2 to finish, as row 793's order (S2 before S0) already set. If the session host's memory-pressure reaper stops the runner (F0 maps 63 GB of the 80 GB RAM), it is not restarted by me: the runner resumes per panel and the owner gets the command. | `E:/.../gguf/quantize_flash_q4_k_m_noimatrix_3.log`, `c50_s2_run.py` | PROGRESS / DECISION |
| 2026-10-18 | **A4-bis protocol pushed, before any arm scores the post-release documents.** The corpus (`c50_postrelease_corpus.py` at `29b28ba`) was finished by the owner from a terminal outside the session at 16:21, resuming from its cache after the reaper stop. 72 documents, 24 per category, created 2026-03-01..2026-08-31 (Ling-mini-2.0's weights last changed 2025-09-08): 24 `.py` and 24 `.md` files from distinct MIT/Apache-2.0 GitHub repositories (54 searches, 1,611 candidates) and 24 new English Wikipedia articles; 151,811 tokens. Inputs manifest `a102771a…`; the item record is `results/c50_postrelease_v1_manifest.json`. **Apparatus smoke, a null control:** on 3 documents (one per category), two independent P0 processes write byte-identical `docs.pos.bin`; the verdict gives dBPB +0.00000, CI95 (0, 0), PLANT DOES NOT FIRE. Seen in the smoke: P0's BPB on those 3 documents, 0.7165; no BF16 or plant score on the new text has been seen. The gate (G1 unchanged, the plant must show a measured loss, PC2 conditional) is as drafted in the A4 design review. Budget ~2.6 h (ESTIMATE, scaled from A4's document panel); over 1 h, so it waits for the owner's go (rule 7), and it starts after S2's panels end. | [A4-bis protocol](C50_A4BIS_POSTRELEASE_PROTOCOL_20261018.md) | DECISION / MEASURED (null) |
| 2026-10-18 | **Owner: rule 7 amended, and "ok A4-bis".** The owner's words: the authorization is not needed when the machine need not be empty (for benchmarks). Rule 7 (§6) now asks for the owner's go only for runs over one hour that read a rate or a timing; quality cells, builds and corpora are launched without asking, one long job at a time. Registered protocols that cite the old rule 7 are not edited; for the cells still to run, the new rule applies. S0 and S1 read rates, so they still need an empty machine and the owner's go. A4-bis (no timing) is in any case given its go: it starts when S2's runner exits. | owner in session, 2026-10-18 | DECISION |
| 2026-10-18 | **S2: T0 and M0 complete; the runner stopped by the session host's memory-pressure reaper at the start of the F0 arm; the A4-bis chain stopped too.** Session 1 (14:10) recorded all 7 panels of T0 (3 run, 4 reused from A4's P0 with the same executable and inputs) and all 7 of M0, each COMPLETE with its CONFIG confirmed. The host stopped the runner, the log watcher and the A4-bis waiting chain around 18:00, as F0 (63 GB) was being loaded; no F0 panel was recorded. No cause beyond the host's notice is claimed. The chain's inner shell survived the stop and, seeing S2's process gone, started A4-bis's runner at 18:01:57 outside the host's tracking. It was stopped at 18:02, after its command lines were verified and before it recorded anything (it was still hashing the BF16): letting it run would have bypassed the host's stop. Not restarted (host rule). Next: the owner runs, from a terminal outside the session, S2's F0 arm and then A4-bis step 1 (BF16, PC) in one command (COMMUNICATION item 1); both runners resume from their records. | `results/c50_s2_run.json`, `E:/.../s2/s2_run_1.log`, `E:/.../a4bis_post/a4bis_run_1.log` | PROGRESS |
| 2026-10-18 | **S2 F0, first session (owner's terminal, 18:05-18:43): chat COMPLETE, six panels STOPPED by the projected-time guard; S2 addendum 1 registered before the second.** The tools project a process from its first item, and on F0 that item carries the cold start of a 63 GB model: F0's first ARC and MMLU documents took 34-42 times M0's on identical positions, against ~4.4 times the active parameters; on chat, the one complete F0 panel, F0's tool wall is 5.86 times M0's (4,087 against 1,898 generated tokens, ~2.7 times per token). For F0 only, the projection is off and the runner bounds each process at 6 h of wall time; T0 and M0 keep the 3 h guard. Scores cannot move (same tools, settings and inputs; deterministic decoding); STOPPED records are kept under `superseded`. Logic checked on simulated processes. Estimated F0 panels 0.4-4.2 h, ~10-16 h in all (ESTIMATE). Runs after A4-bis. A4-bis step 1 (BF16, then PC) started at 18:44 in the same owner command. | [S2 protocol, addendum 1](C50_S2_GAP_MAP_PROTOCOL_20261017.md), `results/c50_s2_run.json` | MEASURED (guard) / DECISION |
| 2026-10-18 | **A4-bis step 1: PLANT DOES NOT FIRE (PC, Q2_K with the FIT imatrix).** BF16 and PC scored the 72 post-release documents (owner's command C, 18:44-20:27; both COMPLETE, CONFIG confirmed; BF16 hash = the A2 record, PC = `278738d5`). G1 on PC: BPB 0.7800 -> 0.7819, **dBPB +0.0019, CI95 (-0.0045, +0.0083)**; the plant needs upper > +0.020 and lower > 0. By category: code +0.014, technical +0.009, prose -0.018. **The registered prediction fails** (PC +0.015..+0.080, firing). Reported only: top-1 disagreement with BF16 15.5% (CI95 14.3-16.7%), top-1 accuracy 63.03% against BF16's 63.46%, mean abs dlog p 0.447 nats; on the 1,605 positions where BF16 gives the target log p < -15, PC gains 3,149 bits, and outside them it loses 4,243 (+0.0072 BPB), the A4 pattern again. BF16's BPB here: code 0.492, technical 0.923, prose 0.925 (FIT: 0.419, 0.600, 1.306). P0 is not run. Next, as registered (step 3): build PC2 (Q2_K from the same BF16, no imatrix), pin it, score it, plant stage on PC2. | `results/c50_a4bis_run.json`, `results/c50_a4bis_verdict_plant_pc.json` | MEASURED |
| 2026-10-18 | **A4-bis step 3: PC2 built and pinned before it scores.** Q2_K from the same BF16 with no imatrix, the pinned llama-quantize (5b335f4, AVX2 build, 6 threads), 20:29-20:34, exit 0: 6,094,840,704 B, sha256 7f6375e9..., pinned in c50_a4_run.py. Same format as PC: the same type mix (79 tensors to q2_K-class types, 59 q3, 20 q4, 1 q6) and 5,806.30 MiB (3.00 BPW); PC loaded 176 imatrix entries, PC2 none. Next: PC2 scores the 72 documents, then the plant stage on PC2. | E:/.../gguf/quantize_q2_k_noimatrix.log | PROGRESS |
| 2026-10-18 | **A4-bis step 3: PLANT FIRES on PC2 (Q2_K, no imatrix).** PC2 scored the 72 documents (20:36-21:21, COMPLETE, CONFIG confirmed, hash = pin). G1: BPB 0.7800 -> 0.8152, **dBPB +0.0351, CI95 (+0.0274, +0.0437)**: upper > +0.020 and lower > 0. By category: code +0.035, technical +0.055, prose +0.015. Reported only: top-1 disagreement 17.7% (CI95 16.4-19.0%), top-1 accuracy 62.29% against BF16's 63.46%; on BF16's 1,605 tail positions PC2 gains 2,695 bits, outside them it loses 23,410. So the unchanged G1 sees a measured loss on post-release text when the importance weighting is removed, while PC, the same format with the FIT imatrix, stayed inside it. Next, as registered: P0 scores the documents, then the final verdict with `--plant pc2 --cand p0`. | `results/c50_a4bis_verdict_plant_pc2.json` | MEASURED |
| 2026-10-18 | **C50-A4-bis: A4 PASS. T0 = P0 is qualified.** P0 scored the 72 post-release documents (21:26-22:05, COMPLETE, hash = pin). Final verdict with the plant PC2, which fires: P0 against BF16, **dBPB -0.0035, CI95 (-0.0063, -0.0007)**, inside +0.020 and the preferred +0.010; by category code -0.002, technical -0.006, prose -0.003. Reported: top-1 disagreement 7.7% (PC 15.5%, PC2 17.7%), top-1 accuracy 63.46% (= BF16). Predictions: PC firing WRONG, P0 in +0.002..+0.015 WRONG (it is below BF16), P0 passes RIGHT, PC disagreement > P0 RIGHT. A4 closes; S0 follows (an empty machine and the owner's go, ~1 h since A1b has passed). | [A4-bis result](C50_A4BIS_POSTRELEASE_RESULT_20261018.md) | MEASURED / GATE |
