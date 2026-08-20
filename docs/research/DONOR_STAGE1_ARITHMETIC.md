# Donor Adaptation — Stage-1 Arithmetic Pass

**Mandate:** `docs/prompts/master_prompts/DONOR_MODEL_ADAPTATION.md` §9 stage 1 — *"Redo §5 properly for
a shortlist of real donors: footprint inventory, streaming budget, KV budget, head cost, per-SKU
eligibility. All desk, all free. Gate: a ranked shortlist with the arithmetic shown; any donor whose
arithmetic fails is eliminated here, before any compute."*

**Status:** DONE. Desk only. No weights downloaded, no GPU, no engine change, no conversion.
**Tool:** `benchmarks/donor_adaptation/donor_inventory.py`.
**Author role:** Builder. Nothing here is committed or pushed; the Media Manager owns Git.

---

## 0. The headline, stated plainly

**Priced on rates that rest on nothing contested, not one of the 18 reachable donors passes the sealed
`≥10 tok/s` decode gate — at any SKU, at any context, at the most permissive precision map.**
The best number in the whole set is **9.76 tok/s** (`Qwen/Qwen2.5-1.5B`, SKU-A, 32K native attention,
4-bit KV). The second best is 5.98.

If the two figures currently under adversarial audit are granted — the 17.0 GB/s kernel-pure ternary
rate reaching the integrated engine, and the ~8.4 µs/expert dispatch overhead being removed — then
exactly **two** donors clear the gate at SKU-A/32K (`Qwen2.5-1.5B` at 12.11, `OLMoE-1B-7B-0924` at
11.01) and exactly **one** clears it at SKU-B/128K (`Qwen2.5-1.5B` at 10.14).

That is the result. It is a **negative result at stage 1, and it lands before any compute was spent**,
which is exactly what this stage exists to do. Three things follow, and each contradicts §5:

1. **The binding wall is throughput, not footprint.** §5 says *"the footprint wall, which is where the
   real trouble is."* The arithmetic says the opposite: at the all-ternary map **14 of 18 donors fit
   inside SKU-A's 16 GB** with KV and a declared 1 GB margin. Only four are eliminated on RAM. Every
   other elimination is on speed.
2. **§5's streaming table over-predicts tok/s by 2×–10×**, because 42 GB/s is not a rate this engine
   has ever achieved on a weight path. §5 predicts ~28 tok/s for a "30B total / 3B active" donor;
   `Qwen3-30B-A3B` is exactly that shape and computes to **2.72–10.70 tok/s** — the *top* of that
   bracket already assumes the contested 17.0 GB/s.
3. **The P61/D9 precision map does not survive scale-up to a donor.** Keeping projections, head and
   embeddings at fp32 costs 11 MB/token at the 8.3M sandbox. On `granite-4.0-h-small` the same map
   costs **15.92 GB/token** — 503 ms of pure fp32 streaming, a 2.0 tok/s ceiling before a single MLP
   byte moves. Any donor route must ternarize the projections, and P61 measured that costs
   +0.018–0.022 BPB at 8.3M. **That trade was never priced at donor scale and it is now on the critical
   path.**

§5's head arithmetic, by contrast, is **corroborated**: it predicts ~1.24 GB and a ~30 ms lower bound
for a D=2048 / V=152K head; the tool computes 1.16 GiB and 34.6–36.6 ms at the measured rate. §5 was
right about the head and optimistic about everything else.

---

## 1. Method, and what each number is made of

### 1.1 Provenance rule

Every figure is computed from a field **present in that donor's own `config.json`**, fetched from the
Hub and hashed (T9). No model card was consulted for any number. Where a needed key is absent the row
is marked PARTIAL and the missing key is named; the tool never substitutes a default for a load-bearing
field, and refuses by named exception when one is missing (§3, control C3).

The parameter formula is **cross-checked against the repo's own safetensors parameter total** — a second
artefact, independent of my arithmetic. That check is the guard against the project's characteristic
failure (the plausible artefact). It agrees to **< 0.01%** on 15 of 18 donors, which is what makes the
remaining three disagreements informative rather than noise.

### 1.2 Rate constants — measured, from `docs/PHASE64_BUDGET.md`

The mandate's §5 uses a flat 42 GB/s. **That number is superseded here** and is not used anywhere in
the tool. What is used:

| organ class | rate | source |
|---|---|---|
| fp32 projections / head (GEMV) | `r(size)` = 187 / 185 / 134.4 / 60.5 / 55.7 / 45.5 / 45.3 / 36.5 GB/s at 4 / 8 / 16 / 24 / 32 / 48 / 64 / 96 MB, t6 | §1b(a), measured |
| … above 96 MB | 34–36 GB/s asymptote — **EXTRAPOLATION, flagged in every row that uses it** | §1b(a) |
| routed ternary experts, engine-integrated | **4.2 GB/s** (gather + dispatch already inside) | §1, measured |
| dense ternary MLP, engine-integrated | **11.4 GB/s** = 2 359 296 B / 207 µs at t6 | §2 + §1 decomposition, derived below |
| ternary kernel-pure ceiling | **17.0 GB/s** t6 | §1b(b), measured, **but not reached by the integrated engine** |
| per-expert dispatch overhead | **8.4 µs** | §1b(b), decomposed — **under audit** |
| KV-cache sequential read | 40–44 GB/s aggregate DRAM | §1, measured |
| norms + glue | 7 µs at L=6, scaled linearly in L | §2 — **extrapolation** |

**The 11.4 GB/s dense-MLP rate is a correction to my own brief.** The brief instructed pricing all
ternary MLP/expert bytes at [4.2 .. 17.0]. But 4.2 GB/s is the *routed* path, and it is 4.2 precisely
*because* it carries an index gather and per-expert dispatch. §2's own decomposition gives the dense
LUT-MLP at the 8.3M anchor moving `3·256·1024·0.5 B · 6 layers = 2 359 296 B` in `207 µs` at t6 =
**11.4 GB/s** — the same number §3 calls "the LUT ceiling 11.4 → 17.0". Charging a *dense* donor MLP
the routed-path rate would be pricing a gather that does not exist. The measured-only column therefore
puts dense MLP on 11.4 and routed experts on 4.2. Both are measured, engine-integrated, and neither
assumes the overhead fix. This raises the dense donors by ~1.6× and changes no verdict.

### 1.3 The three time columns

| column | ternary MLP/experts | dispatch term | projections/head | what it rests on |
|---|---|---|---|---|
| **PESS** | 4.2 GB/s | + 8.4 µs × calls | `r(size)` streamed floor | deliberately conservative; **double-counts**, because 4.2 already contains the overhead |
| **OPT** | 17.0 GB/s | + 8.4 µs × calls | `r(size)` curve | **contested** — assumes the kernel-pure rate reaches the integrated engine |
| **MEASURED-ONLY** | 11.4 dense / 4.2 routed | none | `r(size)` streamed floor | **the only column resting on nothing under audit** |

Read the MEASURED-ONLY column as the verdict and the OPT column as the prize for winning the audit.

### 1.4 Footprint formula (mandate S2)

```
resident = Σ_matrices [ out_rows · ceil(in_cols/32)·32 · B_per_weight ]      (payload + 32-elem padding)
         + Σ_ternary_matrices [ out_rows · 4 B ]                             (per-row fp32 scales)
         + KV cache at the SKU context
         + logits (V · 4 B)
         + 1.0 GB declared OS/allocator + activations/scratch margin
```

Two precision maps, both reported: **(a)** everything ternary at 0.5 B/w, applied literally (norms and
biases included — they are ppm of the total); **(b)** the **P61/D9 map**: MLP and experts ternary at
0.5 B/w, projections + head + embeddings + norms at fp32 4 B/w. `1 GB = 1024³ B` throughout for
footprint; rates are decimal `1e9 B/s` as the budget states them.

### 1.5 KV formula

Per attention layer, per context token: `2 · n_kv_heads · head_dim` elements for GQA/MHA, or
`kv_lora_rank + qk_rope_head_dim` for MLA (DeepSeek-V2). **Only attention layers are counted**, from
the layer-type map read out of the config (T2). Per-layer `sliding_window` is honoured where the config
carries a per-layer `layer_types` map (this matters for `gpt-oss-20b`, whose alternating sliding layers
cap at 128 tokens and cut its 128K KV from a naive 6.0 GB to a real 3.0 GB). Linear-attention layers
(`qwen3_next`) carry an O(1) recurrent state and are **not** charged a growing cache. SSM layers are
charged nothing. Where no layer-type map exists in the config, the tool returns UNKNOWN rather than
assuming a ratio — no donor in this set hit that path.

Decode reads the entire KV cache once per generated token, so KV is a **throughput** term as well as a
footprint term, and at 128K it is frequently the dominant one. §5 omits it entirely.

---

## 2. The per-donor tables

<!-- BEGIN GENERATED TABLES -->
### T1 — shape, licence and the parameter cross-check

| donor | family | D | L | V | tied | params computed | params advertised (safetensors) | delta | cross-check | licence |
|---|---|---|---|---|---|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B` | qwen2 | 1536 | 28 | 151936 | yes | 1.544B | 1.544B | +0.00% | AGREE | apache-2.0 |
| `Qwen/Qwen3-1.7B` | qwen3 | 2048 | 28 | 151936 | yes | 1.721B | 2.032B | -15.32% | AGREE_TIED_DUPLICATED | apache-2.0 |
| `HuggingFaceTB/SmolLM2-1.7B` | llama | 2048 | 24 | 49152 | yes | 1.711B | 1.711B | +0.00% | AGREE | apache-2.0 |
| `microsoft/Phi-3-mini-4k-instruct` | phi3 | 3072 | 32 | 32064 | no | 3.821B | 3.821B | +0.00% | AGREE | mit |
| `allenai/OLMo-2-1124-7B` | olmo2 | 4096 | 32 | 100352 | no | 7.298B | 7.299B | -0.00% | AGREE | apache-2.0 |
| `mistralai/Mistral-7B-v0.3` | mistral | 4096 | 32 | 32768 | no | 7.248B | 7.248B | +0.00% | AGREE | apache-2.0 |
| `Qwen/Qwen3-8B` | qwen3 | 4096 | 36 | 151936 | no | 8.191B | 8.191B | -0.00% | AGREE | apache-2.0 |
| `Qwen/Qwen3-30B-A3B` | qwen3_moe | 2048 | 48 | 151936 | no | 30.532B | 30.532B | -0.00% | AGREE | apache-2.0 |
| `mistralai/Mixtral-8x7B-v0.1` | mixtral | 4096 | 32 | 32000 | no | 46.703B | 46.703B | +0.00% | AGREE | apache-2.0 |
| `deepseek-ai/DeepSeek-V2-Lite` | deepseek_v2 | 2048 | 27 | 102400 | no | 15.706B | 15.706B | -0.00% | AGREE | other |
| `allenai/OLMoE-1B-7B-0924` | olmoe | 2048 | 16 | 50304 | no | 6.919B | 6.919B | -0.00% | AGREE | apache-2.0 |
| `openai/gpt-oss-20b` | gpt_oss | 2880 | 24 | 201088 | no | 20.908B | 21.512B | -2.81% | PACKED_STORAGE | apache-2.0 |
| `Zyphra/Zamba2-2.7B` | zamba2 | 2560 | 54 | 32000 | no | 7.031B | 2.662B | +164.11% | DISAGREE | apache-2.0 |
| `ibm-granite/granite-4.0-h-small` | granitemoehybrid | 4096 | 40 | 100352 | yes | 32.207B | 32.207B | -0.00% | AGREE | apache-2.0 |
| `nvidia/Nemotron-H-8B-Base-8K` | nemotron_h | 4096 | 52 | 131072 | no | 8.101B | 8.101B | -0.00% | AGREE | other |
| `tiiuae/Falcon-H1-7B-Base` | falcon_h1 | 3072 | 44 | 130048 | no | 7.585B | 7.586B | -0.00% | AGREE | other |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | qwen3_next | 2048 | 48 | 151936 | no | 79.574B | 81.325B | -2.15% | DISAGREE | apache-2.0 |
| `state-spaces/mamba2-2.7b` | mamba_ssm | 2560 | 64 | 50288 | yes | 2.702B | — | — | NO_METADATA | apache-2.0 |
| `meta-llama/Llama-3.2-1B` | — | | | | | | | | **UNAVAILABLE** | — |
| `ai21labs/AI21-Jamba-Mini-1.6` | — | | | | | | | | **UNAVAILABLE** | — |

### T2 — operator map (which layers carry attention), read from the config

| donor | attn layers | SSM layers | linear-attn layers | source key | attention fraction |
|---|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B` | 28 | 0 | 0 | `no layer-type key present and model_type is a dense-transformer family` | 100% |
| `Qwen/Qwen3-1.7B` | 28 | 0 | 0 | `no layer-type key present and model_type is a dense-transformer family` | 100% |
| `HuggingFaceTB/SmolLM2-1.7B` | 24 | 0 | 0 | `no layer-type key present and model_type is a dense-transformer family` | 100% |
| `microsoft/Phi-3-mini-4k-instruct` | 32 | 0 | 0 | `no layer-type key present and model_type is a dense-transformer family` | 100% |
| `allenai/OLMo-2-1124-7B` | 32 | 0 | 0 | `no layer-type key present and model_type is a dense-transformer family` | 100% |
| `mistralai/Mistral-7B-v0.3` | 32 | 0 | 0 | `no layer-type key present and model_type is a dense-transformer family` | 100% |
| `Qwen/Qwen3-8B` | 36 | 0 | 0 | `no layer-type key present and model_type is a dense-transformer family` | 100% |
| `Qwen/Qwen3-30B-A3B` | 48 | 0 | 0 | `no layer-type key present and model_type is a dense-transformer family` | 100% |
| `mistralai/Mixtral-8x7B-v0.1` | 32 | 0 | 0 | `no layer-type key present and model_type is a dense-transformer family` | 100% |
| `deepseek-ai/DeepSeek-V2-Lite` | 27 | 0 | 0 | `no layer-type key present and model_type is a dense-transformer family` | 100% |
| `allenai/OLMoE-1B-7B-0924` | 16 | 0 | 0 | `no layer-type key present and model_type is a dense-transformer family` | 100% |
| `openai/gpt-oss-20b` | 24 | 0 | 0 | `layer_types` | 100% |
| `Zyphra/Zamba2-2.7B` | 9 | 45 | 0 | `layers_block_type` | 17% |
| `ibm-granite/granite-4.0-h-small` | 4 | 36 | 0 | `layer_types` | 10% |
| `nvidia/Nemotron-H-8B-Base-8K` | 4 | 24 | 0 | `hybrid_override_pattern` | 8% |
| `tiiuae/Falcon-H1-7B-Base` | 44 | 44 | 0 | `attn_layer_indices=null -> parallel hybrid, attention on every layer (INFERRED)` | 100% |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | 12 | 0 | 36 | `full_attention_interval` | 25% |
| `state-spaces/mamba2-2.7b` | 0 | 0 | 0 | `attn_layer_idx` | 0% |

### T3 — resident weight footprint, both precision maps

| donor | (a) all-ternary 0.5 B/w | of which scales | of which 32-pad | (b) P61/D9 map | SKU-A 16GB @32K, 4-bit KV, map (a) | SKU-B 64GB @128K, 4-bit KV, map (a) |
|---|---|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B` | 741.8 MB | 3.6 MB | 2.1 MB | 1.98 GB | 1.94 GB FIT | 2.60 GB FIT |
| `Qwen/Qwen3-1.7B` | 825.4 MB | 3.2 MB | 1.7 MB | 2.97 GB | 2.68 GB FIT | 5.31 GB FIT |
| `HuggingFaceTB/SmolLM2-1.7B` | 820.5 MB | 3.0 MB | 1.5 MB | 2.44 GB | 3.30 GB FIT | 7.80 GB FIT |
| `microsoft/Phi-3-mini-4k-instruct` | 1.79 GB | 4.9 MB | 3.0 MB | 6.36 GB | 5.79 GB FIT | 14.79 GB FIT |
| `allenai/OLMo-2-1124-7B` | 3.41 GB | 7.0 MB | 3.9 MB | 13.08 GB | 8.41 GB FIT | 20.41 GB FIT |
| `mistralai/Mistral-7B-v0.3` | 3.39 GB | 6.5 MB | 3.9 MB | 8.63 GB | 5.39 GB FIT | 8.39 GB FIT |
| `Qwen/Qwen3-8B` | 3.83 GB | 7.6 MB | 4.4 MB | 12.80 GB | 5.95 GB FIT | 9.33 GB FIT |
| `Qwen/Qwen3-30B-A3B` | 14.31 GB | 87.3 MB | 2.9 MB | 19.32 GB | 16.06 GB **OVER** | 18.31 GB FIT |
| `mistralai/Mixtral-8x7B-v0.1` | 21.79 GB | 34.5 MB | 3.9 MB | 27.01 GB | 23.79 GB **OVER** | 26.79 GB FIT |
| `deepseek-ai/DeepSeek-V2-Lite` | 7.35 GB | 34.0 MB | 1.7 MB | 9.93 GB | 8.59 GB FIT | 9.30 GB FIT |
| `allenai/OLMoE-1B-7B-0924` | 3.24 GB | 17.1 MB | 1023.0 KB | 4.79 GB | 5.24 GB FIT | 8.24 GB FIT |
| `openai/gpt-oss-20b` | 9.77 GB | 28.6 MB | 3.9 MB | 15.62 GB | 10.96 GB FIT | 11.52 GB FIT |
| `Zyphra/Zamba2-2.7B` | 3.30 GB | 10.7 MB | 10.8 MB | 12.36 GB | 5.70 GB FIT | 9.92 GB FIT |
| `ibm-granite/granite-4.0-h-small` | 15.08 GB | 70.0 MB | 13.5 MB | 28.99 GB | 16.20 GB **OVER** | 16.58 GB FIT |
| `nvidia/Nemotron-H-8B-Base-8K` | 3.79 GB | 8.9 MB | 12.7 MB | 16.40 GB | 4.92 GB FIT | 5.29 GB FIT |
| `tiiuae/Falcon-H1-7B-Base` | 3.55 GB | 10.3 MB | 8.2 MB | 12.02 GB | 4.89 GB FIT | 5.93 GB FIT |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | 37.35 GB | 294.0 MB | 6.9 MB | 44.22 GB | 38.54 GB **OVER** | 39.10 GB FIT |
| `state-spaces/mamba2-2.7b` | 1.28 GB | 6.7 MB | 12.1 MB | 10.07 GB | 2.28 GB FIT | 2.28 GB FIT |

### T4 — KV cache, attention layers only

| donor | KV elems/tok/layer | kind | 4K fp16 | 4K 4-bit | 32K fp16 | 32K 4-bit | 128K fp16 | 128K 4-bit |
|---|---|---|---|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B` | 512 | gqa | 112.0 MB | 28.0 MB | 896.0 MB | 224.0 MB | 3.50 GB | 896.0 MB |
| `Qwen/Qwen3-1.7B` | 2048 | gqa | 448.0 MB | 112.0 MB | 3.50 GB | 896.0 MB | 14.00 GB | 3.50 GB |
| `HuggingFaceTB/SmolLM2-1.7B` | 4096 | gqa | 768.0 MB | 192.0 MB | 6.00 GB | 1.50 GB | 24.00 GB | 6.00 GB |
| `microsoft/Phi-3-mini-4k-instruct` | 6144 | gqa | 1.50 GB | 384.0 MB | 12.00 GB | 3.00 GB | 48.00 GB | 12.00 GB |
| `allenai/OLMo-2-1124-7B` | 8192 | gqa | 2.00 GB | 512.0 MB | 16.00 GB | 4.00 GB | 64.00 GB | 16.00 GB |
| `mistralai/Mistral-7B-v0.3` | 2048 | gqa | 512.0 MB | 128.0 MB | 4.00 GB | 1.00 GB | 16.00 GB | 4.00 GB |
| `Qwen/Qwen3-8B` | 2048 | gqa | 576.0 MB | 144.0 MB | 4.50 GB | 1.12 GB | 18.00 GB | 4.50 GB |
| `Qwen/Qwen3-30B-A3B` | 1024 | gqa | 384.0 MB | 96.0 MB | 3.00 GB | 768.0 MB | 12.00 GB | 3.00 GB |
| `mistralai/Mixtral-8x7B-v0.1` | 2048 | gqa | 512.0 MB | 128.0 MB | 4.00 GB | 1.00 GB | 16.00 GB | 4.00 GB |
| `deepseek-ai/DeepSeek-V2-Lite` | 576 | mla | 121.5 MB | 30.4 MB | 972.0 MB | 243.0 MB | 3.80 GB | 972.0 MB |
| `allenai/OLMoE-1B-7B-0924` | 4096 | gqa | 512.0 MB | 128.0 MB | 4.00 GB | 1.00 GB | 16.00 GB | 4.00 GB |
| `openai/gpt-oss-20b` | 1024 | gqa | 99.0 MB | 24.8 MB | 771.0 MB | 192.8 MB | 3.00 GB | 768.8 MB |
| `Zyphra/Zamba2-2.7B` | 10240 | gqa | 720.0 MB | 180.0 MB | 5.62 GB | 1.41 GB | 22.50 GB | 5.62 GB |
| `ibm-granite/granite-4.0-h-small` | 2048 | gqa | 64.0 MB | 16.0 MB | 512.0 MB | 128.0 MB | 2.00 GB | 512.0 MB |
| `nvidia/Nemotron-H-8B-Base-8K` | 2048 | gqa | 64.0 MB | 16.0 MB | 512.0 MB | 128.0 MB | 2.00 GB | 512.0 MB |
| `tiiuae/Falcon-H1-7B-Base` | 512 | gqa | 176.0 MB | 44.0 MB | 1.38 GB | 352.0 MB | 5.50 GB | 1.38 GB |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | 1024 | gqa | 96.0 MB | 24.0 MB | 768.0 MB | 192.0 MB | 3.00 GB | 768.0 MB |
| `state-spaces/mamba2-2.7b` | None | none | 0.0 KB | 0.0 KB | 0.0 KB | 0.0 KB | 0.0 KB | 0.0 KB |

### T5 — per-token byte traffic and the three time columns (weights only, no KV)

| donor | active params/tok | proj+head fp32 bytes/tok | r(proj) GB/s | ternary MLP bytes/tok | ternary expert bytes/tok | expert calls/tok | tok/s PESS | tok/s OPT | tok/s **MEASURED-ONLY** |
|---|---|---|---|---|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B` | 1544M | 1.44 GB | 34–36* | 553.3 MB | 0.0 KB | 0 | 5.44 | 12.95 | **10.36** |
| `Qwen/Qwen3-1.7B` | 1721M | 2.47 GB | 34–36* | 505.5 MB | 0.0 KB | 0 | 4.89 | 9.53 | **8.02** |
| `HuggingFaceTB/SmolLM2-1.7B` | 1711M | 1.88 GB | 34–36* | 577.7 MB | 0.0 KB | 0 | 4.91 | 10.92 | **8.90** |
| `microsoft/Phi-3-mini-4k-instruct` | 3723M | 4.87 GB | 34–36* | 1.13 GB | 0.0 KB | 0 | 2.26 | 4.62 | **3.85** |
| `allenai/OLMo-2-1124-7B` | 6887M | 9.53 GB | 34–36* | 2.02 GB | 0.0 KB | 0 | 1.22 | 2.43 | **2.04** |
| `mistralai/Mistral-7B-v0.3` | 7114M | 5.50 GB | 34–36* | 2.63 GB | 0.0 KB | 0 | 1.18 | 3.03 | **2.37** |
| `Qwen/Qwen3-8B` | 7568M | 7.94 GB | 34–36* | 2.54 GB | 0.0 KB | 0 | 1.11 | 2.52 | **2.04** |
| `Qwen/Qwen3-30B-A3B` | 3042M | 4.58 GB | 34–36* | 0.0 KB | 869.2 MB | 384 | 2.74 | 5.17 | **2.76** |
| `mistralai/Mixtral-8x7B-v0.1` | 12749M | 5.49 GB | 34–36* | 0.0 KB | 5.26 GB | 64 | 0.66 | 2.01 | **0.66** |
| `deepseek-ai/DeepSeek-V2-Lite` | 2451M | 2.18 GB | 34–36* | 32.2 MB | 861.7 MB | 182 | 3.41 | 8.22 | **3.49** |
| `allenai/OLMoE-1B-7B-0924` | 1179M | 1.39 GB | 34–36* | 0.0 KB | 386.0 MB | 128 | 7.07 | 15.06 | **7.13** |
| `openai/gpt-oss-20b` | 3607M | 4.54 GB | 34–36* | 0.0 KB | 1.12 GB | 96 | 2.33 | 4.84 | **2.33** |
| `Zyphra/Zamba2-2.7B` | 6949M | 10.07 GB | 34–36* | 1.98 GB | 0.0 KB | 0 | 1.21 | 2.35 | **1.98** |
| `ibm-granite/granite-4.0-h-small` | 8803M | 15.92 GB | 34–36* | 0.0 KB | 2.12 GB | 440 | 0.95 | 1.63 | **0.96** |
| `nvidia/Nemotron-H-8B-Base-8K` | 7564M | 12.43 GB | 34–36* | 1.97 GB | 0.0 KB | 0 | 1.12 | 2.02 | **1.73** |
| `tiiuae/Falcon-H1-7B-Base` | 7186M | 8.21 GB | 34–36* | 2.32 GB | 0.0 KB | 0 | 1.17 | 2.55 | **2.09** |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | 3463M | 6.71 GB | 34–36* | 0.0 KB | 798.2 MB | 528 | 2.41 | 3.94 | **2.43** |
| `state-spaces/mamba2-2.7b` | 2702M | 10.07 GB | 34–36* | 0.0 KB | 0.0 KB | 0 | 3.14 | 3.33 | **3.14** |

`*` = above the last measured point of the §1b curve (96 MB): the 34–36 GB/s asymptote, an **extrapolation**.


### T6 — tok/s including KV-cache read traffic, vs the sealed ≥10 tok/s gate

| donor | 4K pess..opt (meas-only) | 32K pess..opt (meas-only) | 128K pess..opt (meas-only) | gate @32K | gate @128K |
|---|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B` | 5.42..12.84 (10.28) | 5.27..12.11 (9.76) | 4.82..10.14 (8.33) | **FAIL** | **FAIL** |
| `Qwen/Qwen3-1.7B` | 4.83..9.29 (7.84) | 4.39..7.92 (6.75) | 3.35..5.25 (4.58) | **FAIL** | **FAIL** |
| `HuggingFaceTB/SmolLM2-1.7B` | 4.80..10.40 (8.52) | 4.10..7.80 (6.55) | 2.74..4.20 (3.66) | **FAIL** | **FAIL** |
| `microsoft/Phi-3-mini-4k-instruct` | 2.21..4.43 (3.70) | 1.91..3.45 (2.94) | 1.31..1.96 (1.72) | **FAIL** | **FAIL** |
| `allenai/OLMo-2-1124-7B` | 1.20..2.36 (1.98) | 1.08..1.96 (1.67) | 0.80..1.25 (1.09) | **FAIL** | **FAIL** |
| `mistralai/Mistral-7B-v0.3` | 1.18..3.00 (2.35) | 1.15..2.82 (2.23) | 1.05..2.34 (1.89) | **FAIL** | **FAIL** |
| `Qwen/Qwen3-8B` | 1.11..2.50 (2.03) | 1.08..2.36 (1.92) | 0.98..1.97 (1.64) | **FAIL** | **FAIL** |
| `Qwen/Qwen3-30B-A3B` | 2.72..5.11 (2.75) | 2.60..4.72 (2.62) | 2.24..3.75 (2.26) | **FAIL** | **FAIL** |
| `mistralai/Mixtral-8x7B-v0.1` | 0.66..2.00 (0.66) | 0.65..1.92 (0.65) | 0.62..1.68 (0.62) | **FAIL** | **FAIL** |
| `deepseek-ai/DeepSeek-V2-Lite` | 3.40..8.17 (3.48) | 3.33..7.85 (3.41) | 3.13..6.90 (3.20) | **FAIL** | **FAIL** |
| `allenai/OLMoE-1B-7B-0924` | 6.91..14.40 (6.96) | 5.94..11.01 (5.98) | 4.02..6.10 (4.04) | **FAIL** | **FAIL** |
| `openai/gpt-oss-20b` | 2.33..4.82 (2.33) | 2.30..4.73 (2.31) | 2.22..4.44 (2.23) | **FAIL** | **FAIL** |
| `Zyphra/Zamba2-2.7B` | 1.21..2.33 (1.96) | 1.16..2.17 (1.84) | 1.02..1.78 (1.52) | **FAIL** | **FAIL** |
| `ibm-granite/granite-4.0-h-small` | 0.95..1.63 (0.96) | 0.95..1.62 (0.95) | 0.94..1.60 (0.95) | **FAIL** | **FAIL** |
| `nvidia/Nemotron-H-8B-Base-8K` | 1.11..2.02 (1.73) | 1.11..2.01 (1.72) | 1.10..1.97 (1.69) | **FAIL** | **FAIL** |
| `tiiuae/Falcon-H1-7B-Base` | 1.17..2.55 (2.09) | 1.16..2.50 (2.05) | 1.12..2.35 (1.94) | **FAIL** | **FAIL** |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | 2.40..3.93 (2.43) | 2.38..3.87 (2.40) | 2.29..3.67 (2.32) | **FAIL** | **FAIL** |
| `state-spaces/mamba2-2.7b` | 3.14..3.33 (3.14) | 3.14..3.33 (3.14) | 3.14..3.33 (3.14) | **FAIL** | **FAIL** |

KV priced at the measured aggregate DRAM stream [40–44 GB/s]; 4-bit KV assumed, which is the most favourable of the two KV precisions asked for.


### T7 — head alone (mandate §5's named first-class problem)

| donor | V | head bytes fp32 | ms/token fp32 | head bytes ternary | ms/token ternary (LUT bracket) | head share of the fp32 per-token stream |
|---|---|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B` | 151936 | 890.2 MB | 25.9–27.5* | 111.9 MB | 6.9–27.9 | 60% |
| `Qwen/Qwen3-1.7B` | 151936 | 1.16 GB | 34.6–36.6* | 149.0 MB | 9.2–37.2 | 47% |
| `HuggingFaceTB/SmolLM2-1.7B` | 49152 | 384.0 MB | 11.2–11.8* | 48.2 MB | 3.0–12.0 | 20% |
| `microsoft/Phi-3-mini-4k-instruct` | 32064 | 375.8 MB | 10.9–11.6* | 47.1 MB | 2.9–11.8 | 8% |
| `allenai/OLMo-2-1124-7B` | 100352 | 1.53 GB | 45.7–48.4* | 196.4 MB | 12.1–49.0 | 16% |
| `mistralai/Mistral-7B-v0.3` | 32768 | 512.0 MB | 14.9–15.8* | 64.1 MB | 4.0–16.0 | 9% |
| `Qwen/Qwen3-8B` | 151936 | 2.32 GB | 69.1–73.2* | 297.3 MB | 18.3–74.2 | 29% |
| `Qwen/Qwen3-30B-A3B` | 151936 | 1.16 GB | 34.6–36.6* | 149.0 MB | 9.2–37.2 | 25% |
| `mistralai/Mixtral-8x7B-v0.1` | 32000 | 500.0 MB | 14.6–15.4* | 62.6 MB | 3.9–15.6 | 9% |
| `deepseek-ai/DeepSeek-V2-Lite` | 102400 | 800.0 MB | 23.3–24.7* | 100.4 MB | 6.2–25.1 | 36% |
| `allenai/OLMoE-1B-7B-0924` | 50304 | 393.0 MB | 11.4–12.1* | 49.3 MB | 3.0–12.3 | 28% |
| `openai/gpt-oss-20b` | 201088 | 2.16 GB | 64.3–68.1* | 276.9 MB | 17.1–69.1 | 48% |
| `Zyphra/Zamba2-2.7B` | 32000 | 312.5 MB | 9.1–9.6* | 39.2 MB | 2.4–9.8 | 3% |
| `ibm-granite/granite-4.0-h-small` | 100352 | 1.53 GB | 45.7–48.4* | 196.4 MB | 12.1–49.0 | 10% |
| `nvidia/Nemotron-H-8B-Base-8K` | 131072 | 2.00 GB | 59.7–63.2* | 256.5 MB | 15.8–64.0 | 16% |
| `tiiuae/Falcon-H1-7B-Base` | 130048 | 1.49 GB | 44.4–47.0* | 191.0 MB | 11.8–47.7 | 18% |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | 151936 | 1.16 GB | 34.6–36.6* | 149.0 MB | 9.2–37.2 | 17% |
| `state-spaces/mamba2-2.7b` | 50288 | 491.1 MB | 14.3–15.1* | 61.6 MB | 3.8–15.4 | 5% |

### T8 — MoE granularity vs the ρ-safe 48 KB chunk

| donor | experts | top-k | expert width | expert chunk (ternary+scales) | ≥48 KB? |
|---|---|---|---|---|---|
| `Qwen/Qwen3-30B-A3B` | 128 | 8 | 768 | 2318 KB | yes |
| `mistralai/Mixtral-8x7B-v0.1` | 8 | 2 | 14336 | 86144 KB | yes |
| `deepseek-ai/DeepSeek-V2-Lite` | 64 | 6 | 1408 | 4243 KB | yes |
| `allenai/OLMoE-1B-7B-0924` | 64 | 8 | 1024 | 3088 KB | yes |
| `openai/gpt-oss-20b` | 32 | 4 | 2880 | 12184 KB | yes |
| `ibm-granite/granite-4.0-h-small` | 72 | 10 | 768 | 4630 KB | yes |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | 512 | 10 | 512 | 1548 KB | yes |

### T9 — reproducibility manifest

| donor | resolved revision SHA | config.json sha256 | bytes | status |
|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B` | `8faed761d45a263340a0528343f099c05c9a4323` | `0e8c8aa86468aba09c9d32157ff4bc23…` | 684 | OK |
| `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | `1ddb5b89ebc90dcb417a45c213d81857…` | 726 | OK |
| `HuggingFaceTB/SmolLM2-1.7B` | `effd688a12921b4cc83e3312b6feb579f70f9c71` | `33397f6af8090f7290f27ce6fb1cd23a…` | 635 | OK |
| `microsoft/Phi-3-mini-4k-instruct` | `f39ac1d28e925b323eae81227eaba4464caced4e` | `072d4df63228ef806a6b2b2f02a93f1d…` | 967 | OK |
| `allenai/OLMo-2-1124-7B` | `7df9a82518afdecae4e8c026b27adccc8c1f0032` | `f1e210de0ba704c579768c154bf57e36…` | 623 | OK |
| `mistralai/Mistral-7B-v0.3` | `caa1feb0e54d415e2df31207e5f4e273e33509b1` | `affafc6478ec0fd07a32f0ca57aa2fc5…` | 601 | OK |
| `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | `f7c4eadfbbf522470667b797a3c89be2…` | 728 | OK |
| `meta-llama/Llama-3.2-1B` | — | — | — | **UNAVAILABLE** — GATED: 401 Client Error. (Request ID: Root=1-6a86ddaa-4855c3b137f4eab75158ba36;67ffb929-ef |
| `Qwen/Qwen3-30B-A3B` | `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` | `2850ddb3bf7aecad20b611e2d44f3077…` | 963 | OK |
| `mistralai/Mixtral-8x7B-v0.1` | `fc7ac94680e38d7348cfa806e51218e6273104b0` | `9d56d04b36d0fd12ff54ae4c5bac769c…` | 720 | OK |
| `deepseek-ai/DeepSeek-V2-Lite` | `604d5664dddd88a0433dbae533b7fe9472482de0` | `f346286b0f1c8b044252fd54cb4fa78b…` | 1522 | OK |
| `allenai/OLMoE-1B-7B-0924` | `6d84c48581ece794365f2b8e9cfb043c68ade9c5` | `3643aa880d2f1c9b418156269ae791c7…` | 759 | OK |
| `openai/gpt-oss-20b` | `6cee5e81ee83917806bbde320786a8fb61efebee` | `3a2a26ded679375b7928ddeca59764df…` | 1806 | OK |
| `ai21labs/AI21-Jamba-Mini-1.6` | — | — | — | **UNAVAILABLE** — GATED: 401 Client Error. (Request ID: Root=1-6a86ddac-27f03ef621376c0713e7c5ed;dd59388d-35 |
| `Zyphra/Zamba2-2.7B` | `31afeeac4c66b4851a54290ba57c995a68c87861` | `0497ca3001366ed8eaef1030c84f856f…` | 2020 | OK |
| `ibm-granite/granite-4.0-h-small` | `b8c0982bab7fde4eb48110f5a069527c008fab39` | `8616e9f0b30e6fac9696f7c1e1dbd08f…` | 1799 | OK |
| `nvidia/Nemotron-H-8B-Base-8K` | `94ea861e008c2dfced3e8e1302094024077aa04e` | `81e822f85d6471312bcc0ccd34cc6278…` | 1504 | OK |
| `tiiuae/Falcon-H1-7B-Base` | `c9a4cbb95c01b1ede39f69eda083d03d8903b8f0` | `a0ed9a14b6a71ed4701b72549eb07e17…` | 1637 | OK |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | `9c7f2fbe84465e40164a94cc16cd30b6999b0cc7` | `2d483c7cabad7c8704478ed4038fa7e7…` | 1154 | OK |
| `state-spaces/mamba2-2.7b` | `99b226cc377d131cccc610ed4346db564f381f1e` | `0955f133a3eeb902ed11e1a65d0be293…` | 331 | OK |
<!-- END GENERATED TABLES -->

---

## 3. The planted control — the instrument shown to FIRE

Project law §6.3: an instrument is not trusted until it screams on a known-positive, and the corruption
must be the *minimal significant* one, not a catastrophe. Verbatim output of
`python donor_inventory.py control`:

```
<!-- BEGIN CONTROL LOG -->
==============================================================================
PLANTED CONTROL -- the tool must FIRE on a known-positive before its nulls count
==============================================================================

[C1] known-positive: the project's own 8.3M sandbox engine
     target = 2400 KB/token, counted in-engine, docs/SIZING.md
     predicted streamed ternary bytes/token = 2457600 B = 2400.0 KB
     independent count                      = 2457600 B = 2400.0 KB
     error = +0.0000%
     C1 PASS
     decomposition: 6 layers x top-8 x 51200 B/expert (50.0 KB: 49152 B codes + 2048 B per-row scales)

[C2] minimal significant perturbation: num_experts_per_tok 8 -> 7 (one field, one step)
     predicted = 2150400 B = 2100.0 KB ; expected 2150400 B = 2100.0 KB
     moved by -307200 B (-12.50%), expected -307200 B (-12.50%)
     C2 PASS -- right direction, right magnitude

[C3] refusal: a config missing a field the tool needs must raise a NAMED error
     removed 'num_hidden_layers' -> MissingConfigField: num_hidden_layers (required for: n_layers)   OK
     removed 'hidden_size' -> MissingConfigField: hidden_size (required for: hidden_size)   OK
     removed 'vocab_size' -> MissingConfigField: vocab_size (required for: vocab)   OK
     unperturbed config -> no refusal   OK (guard exercised in both directions)

ALL CONTROLS PASS
<!-- END CONTROL LOG -->
```

What each leg establishes:

- **C1 (known-positive, must fire correctly).** The tool is fed a synthetic config reproducing the
  project's *own* 8.3M sandbox engine (V=1024, D=256, L=6, E=32, hid_e=128, top-k=8, one SWA layer,
  window 128 — `benchmarks/phase60/engine.c:51-75`), expressed in the `qwen3_moe` schema so it traverses
  **the same code path a donor traverses**. Its streamed-ternary prediction lands on
  **2 457 600 B/token = 2400.0 KB/token**, matching the independently in-engine-counted figure in
  `docs/SIZING.md` to **0.0000%**. The decomposition is exact and non-coincidental:
  `3·256·128 = 98 304` ternary weights × 0.5 B = 49 152 B of codes, plus `(128+128+256) = 512` output
  rows × 4 B = 2 048 B of per-row scales = **51 200 B = 50.0 KB per expert**, × top-8 × 6 layers =
  2400 KB. The scales term is what turns SIZING's "≈48 KB" expert into the 2400 (not 2304) KB/token the
  engine actually counts — the formula reproduces the *counted* number, including the part a naive
  codes-only formula would miss.
- **C2 (minimal significant perturbation).** One field, one step: `num_experts_per_tok` 8 → 7. Output
  moves to 2 150 400 B, i.e. −307 200 B, **−12.50%**, exactly `7/8`. Right direction, right magnitude,
  no over- or under-shoot.
- **C3 (refusal, both directions).** Deleting each of `num_hidden_layers`, `hidden_size`, `vocab_size`
  raises `MissingConfigField` naming the key and the figure that needed it — no silent zero, no
  substituted default. The unperturbed config raises nothing, so the guard is exercised in **both**
  directions as law §4 requires.

**Additional live firing, not planted by me.** The parameter cross-check fired on three real donors and
each firing was diagnostic rather than cosmetic — see §4. A guard that never fires has not been tested;
this one fired four times in one run.

---

## 4. The three cross-check disagreements, and what they mean

The formula agrees with the repo's own safetensors total to < 0.01% on 15 of 18 donors. The three
disagreements are reported loudly because a mismatch means **the formula is wrong for that family**, and
I did not tune the tool to make them go away.

**`Qwen/Qwen3-1.7B`, −15.32% → resolved, formula correct.** The gap is `V·D = 311.2M`, exactly one
embedding matrix. Hypothesis stated and tested by the tool, not by me: the repo physically stores
`lm_head.weight` as a duplicate of the tied embedding, so its safetensors total counts 2.032B where the
logical model has 1.721B. `computed + one embedding = 2.0317B`, **−0.000%**. This is a *storage* fact,
not a parameter fact; it does not change RAM (one copy is loaded) or time. Labelled
`AGREE_TIED_DUPLICATED`. Note `Qwen2.5-1.5B`, also tied, does **not** duplicate — so this could not have
been a family rule, only a per-repo test.

**`openai/gpt-oss-20b`, −2.81% → the headline total is not comparable.** The config carries
`quantization_config: {quant_method: "mxfp4"}`, and the repo's safetensors totals count **packed storage
elements**: `{BF16: 1 804 459 584, U8: 19 707 494 400}`. Counting uint8 blocks as parameters is not a
parameter count. The tool falls back to the secondary check the config itself makes possible — the
`modules_to_not_convert` list names `self_attn`, `router`, `embed_tokens`, `lm_head` as staying float —
and my non-expert organs total 1.7978B against a 1.8045B float subtotal, **−0.37%** (the residue is
per-expert biases and attention sinks, which I do not model). Labelled `PACKED_STORAGE`. Independently,
this donor is an instance of §8.A's open question 5: it is **already 4-bit**, so a ternarization
transform would be operating on information that has already been destroyed once.

**`Zyphra/Zamba2-2.7B`, +164.11% → my formula is wrong for this family, and the row is unusable.**
The config's `layers_block_type` map is readable and `num_mem_blocks: 2` is present, but Zamba2 reuses
**two shared transformer blocks** across all 54 layers with per-layer LoRA adapters (`adapter_rank: 128`,
`use_shared_mlp_adapter`), a topology my per-layer counter cannot express from these keys. I counted
7.031B against an advertised 2.662B. **Every derived figure in that row inherits the error and none of
them should be quoted.** Zamba2 is therefore eliminated from this pass *for want of a valid model*, not
by a number. It is the one candidate that would need a second look if the shortlist were reopened —
though note the direction of the error means its true footprint is ~2.6× smaller than shown, while its
attention fraction (9/54 = 17%, already inside S3's "minority") is read directly and is not affected.

**`Qwen/Qwen3-Next-80B-A3B-Instruct`, −2.15% → unattributed, and I am leaving it unattributed.**
1.751B of parameters that my organ formula does not account for, all in BF16, so no storage explanation
applies. I attributed at most ~0.10B of it (the Qwen3-Next attention `q_proj` carries an output gate,
doubling its width) and could not close the rest without reading tensor shapes — which would mean
touching the weight files, which is outside this brief. **Direction matters and is favourable to the
verdict:** I *under*-count, so this row's RAM and per-token bytes are optimistic, and the donor is
eliminated anyway (39.10 GB > SKU-A; 2.40 tok/s measured-only). Flagged `FORMULA_DISAGREEMENT`.

**`state-spaces/mamba2-2.7b`, PARTIAL.** The repo exposes no safetensors parameter metadata at all, so
there is **no cross-check for this row** — and its `ssm_cfg` is `{"layer": "Mamba2"}`, carrying none of
`d_state`, `headdim`, `ngroups`. The tool used Mamba-2 reference values (128 / 64 / 1) and marked the
row PARTIAL with the missing fields named. Its numbers are the least trustworthy in the table and it is
eliminated on speed by a wide margin regardless (3.14 tok/s measured-only, and it has *no* KV cache at
all, so that number cannot improve with context handling).

**Two donors were never reachable.** `meta-llama/Llama-3.2-1B` and `ai21labs/AI21-Jamba-Mini-1.6` both
return **401 Gated**. Per the brief, recorded as UNAVAILABLE with the reason; no authentication was
attempted and no gate was worked around. The Jamba loss is the more costly of the two: it was the
set's only Transformer/Mamba MoE hybrid with a published `attn_layer_period`, and §8.A question 4 calls
hybrid donors "the highest-leverage question on this axis". If the Owner wants that question answered,
gated access is the cheapest way to get it.

---

## 5. Elimination ledger — every donor, with the number that eliminated it

Pre-registered elimination rule, applied uniformly: **a donor is eliminated at stage 1 if it misses a
sealed gate under the *most favourable* arithmetic available** — all-ternary footprint map, 4-bit KV,
and the OPT time column (17.0 GB/s + dispatch fix). If even the best case fails, no measurement can
save it. Gates: RAM ≤ 16 GiB at SKU-A / ≤ 64 GiB at SKU-B, and ≥ 10 tok/s decode.

| donor | SKU-A @32K best-case tok/s | SKU-A RAM | SKU-B @128K best-case tok/s | eliminated by |
|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B` | **12.11** | 1.94 GB | **10.14** | — **survives both, best case only** |
| `allenai/OLMoE-1B-7B-0924` | **11.01** | 5.24 GB | 6.10 | **SKU-B only**: 6.10 tok/s at 128K (KV 4.00 GB, MHA) |
| `deepseek-ai/DeepSeek-V2-Lite` | 7.85 | 8.59 GB | 6.90 | tok/s 7.85 < 10 |
| `Qwen/Qwen3-1.7B` | 7.92 | 2.68 GB | 5.25 | tok/s 7.92 < 10 |
| `HuggingFaceTB/SmolLM2-1.7B` | 7.80 | 3.30 GB | 4.20 | tok/s 7.80 < 10; MHA KV = 192 KB/token |
| `openai/gpt-oss-20b` | 4.73 | 10.96 GB | 4.44 | tok/s 4.73 < 10; also already-4-bit |
| `Qwen/Qwen3-30B-A3B` | 4.72 | **16.06 GB > 16 GiB** | 3.75 | **both**: RAM at SKU-A *and* tok/s |
| `Qwen/Qwen3-Next-80B-A3B` | 3.87 | **38.54 GB** | 3.67 | **both**; row also `FORMULA_DISAGREEMENT` |
| `microsoft/Phi-3-mini-4k-instruct` | 3.45 | 5.79 GB | 1.96 | tok/s 3.45 < 10 |
| `state-spaces/mamba2-2.7b` | 3.33 | 2.28 GB | 3.33 | tok/s 3.33 < 10; row PARTIAL |
| `mistralai/Mistral-7B-v0.3` | 2.82 | 5.39 GB | 2.34 | tok/s 2.82 < 10 |
| `tiiuae/Falcon-H1-7B-Base` | 2.50 | 4.89 GB | 2.35 | tok/s 2.50 < 10 |
| `Qwen/Qwen3-8B` | 2.36 | 5.95 GB | 1.97 | tok/s 2.36 < 10 |
| `Zyphra/Zamba2-2.7B` | (2.17) | (5.70 GB) | (1.78) | **`FORMULA_DISAGREEMENT` +164%** — no valid number |
| `nvidia/Nemotron-H-8B-Base-8K` | 2.01 | 4.92 GB | 1.97 | tok/s 2.01 < 10 |
| `allenai/OLMo-2-1124-7B` | 1.96 | 8.41 GB | 1.25 | tok/s 1.96 < 10 |
| `mistralai/Mixtral-8x7B-v0.1` | 1.92 | **23.79 GB** | 1.68 | **both** |
| `ibm-granite/granite-4.0-h-small` | 1.62 | **16.20 GB > 16 GiB** | 1.60 | **both** |
| `meta-llama/Llama-3.2-1B` | — | — | — | **UNAVAILABLE (401 gated)** |
| `ai21labs/AI21-Jamba-Mini-1.6` | — | — | — | **UNAVAILABLE (401 gated)** |

**On the MEASURED-ONLY column, this table has no survivors.** `Qwen2.5-1.5B` reaches 9.76 at SKU-A/32K
and 8.33 at SKU-B/128K; `OLMoE` reaches 5.98 and 4.04.

### 5.1 The ranked shortlist

**SKU-A (16 GB, 32K native attention window, 4-bit KV):**

1. **`Qwen/Qwen2.5-1.5B`** — 1.94 GB resident; 12.11 opt / **9.76 measured-only**. Arithmetic: proj+head
   fp32 1.44 GB/token ÷ 34 GB/s = 45.6 ms; ternary MLP 553 MB/token ÷ 11.4 GB/s = 50.9 ms; KV 224 MB ÷
   40 GB/s = 5.9 ms; glue 33 µs → ~102 ms → 9.76 tok/s. Apache-2.0. GQA 2 KV heads = 28 KB/token of
   context, the second-lightest KV in the set.
2. **`allenai/OLMoE-1B-7B-0924`** — 5.24 GB; 11.01 opt / **5.98 measured-only**. Only 1.18B active per
   token out of 6.92B total — the best active-to-total ratio here, and the strongest evidence for §8.A's
   "sparsity is worth more than small total size". Killed at 128K by full MHA (16 KV heads, 128 KB/token
   → 4.00 GB at 128K even at 4 bits).
3. `deepseek-ai/DeepSeek-V2-Lite` — 8.59 GB; 7.85 / 3.49. **MLA is the standout KV result**: 30.4 KB per
   context token across 27 layers, vs SmolLM2's 192 KB with 4× fewer parameters. Licence is `other` —
   redistribution needs a read (§8.A q6).

**SKU-B (64 GB, 128K native attention):** only `Qwen2.5-1.5B` (10.14 opt / 8.33 measured-only) clears
even in the best case. Everything else is bounded by the ternary stream, not by RAM: **all 18 reachable donors fit SKU-B's 64 GiB** under the all-ternary map, the largest being `Qwen3-Next-80B` at 41.35 GB.

---

## 6. Findings that go back to the Architect

Per §9's stopping rule — *"when a measurement contradicts a claim in this document, that is a finding,
and it goes back to the Architect immediately."*

**F1 — §5's streaming table over-predicts by 2×–10×, and the mechanism is the 42 GB/s.** §5's own
caveat anticipates this; the magnitude is now quantified. Row by row, §5's number vs the all-ternary
computed bracket: 30B/3B-active **28 → 2.72–10.70** (`Qwen3-30B-A3B`, active 3.04B); 120B/5B-active
**17 → 2.37–9.31** (`Qwen3-Next-80B`, active 3.46B); 8B dense **10.5 → 1.11–4.48** (`Qwen3-8B`). The
top of every bracket already grants the contested 17.0 GB/s. §5 should be replaced with a pointer to
this document.

**F2 — the constraint inversion of PHASE64_BUDGET §4 does NOT carry over to donors.** Phase 64 found
speed and footprint both slack by 12–300× and named training the binding wall. That inversion was a
property of *this architecture's frugality* — MBs of active bytes per token. A donor brings GBs. On
donors the ordering re-inverts: **throughput binds first, footprint second, and training does not enter
at all** (the whole point is not to train). The two documents do not contradict each other, but the
budget's conclusion must not be carried across, and someone will try to.

**F3 — the P61/D9 map is unaffordable at donor scale and P61 is back on the critical path.** fp32
projections+head cost 1.44–15.92 GB/token across this set. `granite-4.0-h-small` spends **503 ms/token**
on fp32 organs alone — a 2.0 tok/s ceiling before the first MLP byte. Ternarizing them is therefore not
optional for a donor route, but P61 measured that at +0.018–0.022 BPB at 8.3M (≈ 4σ_seed) and rejected
it. **Two things are unpriced and both are needed:** (i) whether that penalty shrinks with scale (a
literature trend, not a measurement this project owns); (ii) what rate ternary *projections* actually
run at — the LUT rate is measured for expert-shaped and dense-MLP-shaped matrices, never for projection
shapes, so the all-ternary time column in these tables is explicitly labelled UNMEASURED. A cheap
CPU-only microbench of the LUT kernel at donor projection dimensions would convert the single largest
unmeasured term in this document into a measured one, at zero GPU cost.

**F4 — decode KV traffic is a first-class throughput term and §5 omits it.** At 128K fp16, SmolLM2-1.7B
carries a **24.00 GB** cache — larger than SKU-A's entire budget — and reading it once per token costs
**644 ms** at the measured 40 GB/s aggregate stream, dropping the donor from 8.90 tok/s (no KV) to
**1.32**. Even at 4-bit KV it only recovers to 3.66. This makes **GQA group count a hard eligibility
criterion, not a preference** (§8.A lists it as "architectural family matters"). The ranking by KV cost
per context token is stark: granite/Nemotron-H **16 KB** (4 attention layers of 40 / 52) < Qwen3-Next
24 KB < Qwen2.5-1.5B 28 KB < DeepSeek-V2-Lite (MLA) 30 KB < gpt-oss 48 KB < Falcon-H1 44 KB ≪ SmolLM2
**192 KB**. The hybrids and MLA win this axis by an order of magnitude — and then lose on active bytes.

**F5 — the hybrids already satisfy S3, and it does not save them.** §8.A question 4 asks whether a
hybrid donor "short-cuts the central tension by arriving half-converted". Read straight from the
configs: `granite-4.0-h-small` **4/40 = 10%** attention (`layer_types`), `Nemotron-H-8B` **4/52 = 8%**
(`hybrid_override_pattern`), `Zamba2-2.7B` **9/54 = 17%** (`layers_block_type`), `Qwen3-Next-80B` **12/48
= 25%** full attention with the other 36 on gated DeltaNet (`full_attention_interval`). All four are
already at or below S3's "minority" **without any conversion at all**, and their KV budgets are the best
in the set. They are eliminated purely on active bytes per token: granite 8.80B active → 0.96 tok/s,
Nemotron-H 7.56B → 1.73. **The conclusion is sharp and useful: attention→SSM conversion is not the
bottleneck the mandate assumes it is** — donors that arrive pre-converted still fail, on a different
term. That is direct arithmetic evidence for §10's outcome 3 ("the SSM backbone is not load-bearing for
CPU speed; the ternary-sparse-cache-resident properties are"), obtained at zero cost.
`Falcon-H1-7B` is the counterexample that proves the read: it is a *parallel* hybrid with attention in
every layer, and its KV (44 KB/token) sits with the transformers, not with the hybrids.

**F6 — MoE granularity is a non-issue for real donors, which closes §8.A question 3 in the negative.**
Every MoE donor's expert is far above the ρ-safe 48 KB bulk-chunk threshold: OLMoE 3 088 KB,
DeepSeek-V2-Lite 4 243 KB, granite 4 630 KB, Qwen3-Next 1 548 KB, Qwen3-30B-A3B 2 318 KB, gpt-oss
12 184 KB, Mixtral 86 144 KB. None is at risk of the 14× ρ-law penalty. Real donors are 10–1800× coarser
than the sandbox's 48 KB expert, so the granularity worry imported from probe-4 does not transfer.
The *other* half of that question does bite: shared always-active experts (DeepSeek-V2-Lite n=2,
Qwen3-Next, granite) add unavoidable per-token bytes and are counted here as such.

**F7 — the expert-dispatch term is arithmetically irrelevant at donor scale, whatever the audit
concludes.** The 8.4 µs/expert figure is under adversarial audit; at these expert counts it does not
matter. `Qwen3-30B-A3B` makes 384 expert calls/token = **3.2 ms** against a 362 ms total (0.9%);
`Qwen3-Next` 528 calls = 4.4 ms of 415 ms (1.1%). The audit's outcome moves no verdict in this document.
**What does move verdicts is the other contested figure**, the 17.0 GB/s reachability: it is worth
2.0–4.0× on every row and is the entire difference between "one donor passes" and "none do".

---

## 7. Assumptions, and what would invalidate this

Ordered by how much a verdict would move if the assumption is wrong.

1. **Ternary projections are priced at the expert LUT bracket in the all-ternary column, and that rate
   has never been measured for projection shapes.** Marked UNMEASURED in every row that uses it. If
   ternary projections run nearer the fp32 GEMV curve than the LUT rate, the all-ternary column improves
   substantially and more donors approach the gate. *This is the single most valuable cheap measurement
   available and it needs no GPU.*
2. **The proj/head rate above 96 MB is an extrapolation.** Every donor's per-token projection bytes
   exceed 96 MB by 15×–170×, so **every** donor row uses the 34–36 GB/s asymptote, flagged `*` in T5/T7.
   The §1b caveat that streamed experts pollute L3 and push real rates toward the streaming tail applies
   here with more force than at sandbox scale. If the true asymptote is materially below 34 GB/s the
   whole table gets worse; the curve's own shape says it is not materially above.
3. **Organ times are summed, not overlapped.** No prefetch, no compute/stream overlap, no interleaving
   of KV read with weight stream. Real engines overlap some of this. **Direction: optimistic to
   overlap**, so the tok/s figures here are, in that one respect, pessimistic.
4. **The per-position compute floor is not modelled beyond a linearly-scaled 7 µs glue term.** §2.2
   warns explicitly that a design fixing bandwidth and ignoring the compute floor has fixed nothing.
   At the 8.3M anchor the floor is 1.14 ms/token; scaled by any plausible law it stays far below the
   100–1000 ms byte terms computed here, so byte-boundedness is safe — but the scan-recurrence and SWA
   organs a *converted* donor would acquire are not priced at all. **Direction: optimistic.**
5. **The 1.0 GB OS/allocator + scratch margin is declared, not measured.** Only `Qwen3-30B-A3B`
   (16.06 GB) and `granite-4.0-h-small` (16.20 GB) sit close enough to 16 GiB for the margin to decide
   their SKU-A verdict. Both fail on speed by 2–6× anyway, so no verdict turns on it.
6. **Gated MLP is assumed (3 matrices) except where `mlp_hidden_act: relu2` says otherwise
   (Nemotron-H, 2 matrices).** A wrong call here is a 33% error in the largest organ, which is precisely
   why the cross-check exists: it would show as a ~20–30% disagreement. It shows 0.00% on 15 donors,
   so the call is right on those.
7. **`qwen2` QKV biases are a declared family assumption** (the config carries no `attention_bias` key
   and Qwen2 has the biases). Worth 4×10⁻⁵ of the total; the cross-check cannot resolve it either way
   and no figure depends on it. Flagged in the tool's notes rather than hidden.
8. **`falcon_h1` attention-on-every-layer is an INFERENCE** from `attn_layer_indices: null` read as "no
   restriction", not a key that says so. It is corroborated by the −0.00% parameter cross-check, which
   would not close if the attention count were wrong. Recorded as an inference anyway.
9. **`state-spaces/mamba2-2.7b` uses Mamba-2 reference defaults for `d_state`/`headdim`/`ngroups`**
   because `ssm_cfg` carries none of them, and the repo publishes no parameter metadata, so **that row
   has no cross-check at all.** It is the least trustworthy row in the document.
10. **`Zamba2-2.7B`'s numbers are invalid** (+164%) and are shown only in parentheses.
    **`Qwen3-Next-80B`'s numbers under-count by 2.15%**, unattributed — favourable to its elimination.
11. **1 GB = 1024³ B for footprints; rates are decimal 1e9 B/s** as `PHASE64_BUDGET.md` states them.
    Mixing these would move footprints by 7%; nothing in the ledger turns on 7%.
12. **All rates are the 3600X reference floor at t6 with decode hygiene locked**, per project law 11
    (design for the class, not the exemplar). A machine with more memory channels moves every row.

**What would invalidate the headline** (`no donor passes on measured rates`): a measured integrated-engine
ternary rate materially above 11.4 GB/s dense / 4.2 GB/s routed; or a measured ternary-projection rate
near the fp32 GEMV curve (assumption 1); or a donor materially smaller than 1.5B that still satisfies
S4's general-purpose retention gates. Nothing else in the list is large enough.

---

## 8. Reproducibility manifest

```
tool     benchmarks/donor_adaptation/donor_inventory.py
fetch    python donor_inventory.py fetch       # config.json only, no weight file is ever requested
analyze  python donor_inventory.py analyze     # -> inventory.json, tables.md, stdout report
control  python donor_inventory.py control     # -> control.log, the planted-control log in section 3
doc      python donor_inventory.py doc         # regenerates T1-T9, the control log and the env
                                               # table below, in place; prose is never touched
```

Full regeneration from a clean checkout: `fetch` -> `control` -> `analyze` -> `doc`. Every number in
this document is produced by that chain; none is transcribed by hand.

<!-- BEGIN ENV -->
| item | value |
|---|---|
| generated (UTC) | 2026-08-20T11:17:00Z |
| python | 3.12.10 |
| huggingface_hub | 0.36.2 |
| platform | Windows-11-10.0.26200-SP0 |
| `donor_inventory.py` sha256 | `86618c15e3758b5fa960e59959580513ae74e3d9b9a50389bb3009fa9a0c3173` |
| `donor_inventory.py` bytes | 60457 |
| rate-constant source | docs/PHASE64_BUDGET.md sec.1 / sec.1b (measured, 3600X reference) |
| r(size) curve MB | [4.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0] |
| r(size) curve GB/s | [187.0, 185.0, 134.4, 60.5, 55.7, 45.5, 45.3, 36.5] |
| r asymptote GB/s (extrapolated) | [34.0, 36.0] |
| ternary routed-expert GB/s (engine-integrated) | 4.2 |
| ternary dense-MLP GB/s (engine-integrated) | 11.398 |
| ternary kernel-pure ceiling GB/s | 17.0 |
| expert dispatch overhead | 8.4 us/call |
| DRAM aggregate GB/s (KV) | [40.0, 44.0] |
| declared OS/allocator margin | 1.0 GiB |
<!-- END ENV -->

Per-donor resolved revision SHAs and `config.json` content hashes are in **T9** above (project law 9:
a tag is not a specification, a content hash is). Cached configs live in
`benchmarks/donor_adaptation/configs/`, with `_manifest.json` carrying the fetch record — resolved SHA,
sha256, byte count, licence string, safetensors totals by dtype, and the UNAVAILABLE reason where
applicable. `inventory.json` carries every intermediate figure, the per-figure provenance map naming
which config key produced each number, and the rate constants used.

---

## 9. What this pass does NOT establish

- **Nothing about quality.** No BPB, no retention, no PTQ fidelity. §9 stage −1 (the PTQ ternary kill
  gate) is untouched and remains the cheapest thing that could kill the route.
- **Nothing measured.** Every tok/s here is an arithmetic model built on rates measured at 8.3M and one
  synthetic-weight microbench grid. §5's own warning — *"a speedup measured in a microbenchmark and
  reported as a system speedup"* is a named project failure — applies to this document in the opposite
  direction, and the honest reading is: **these are desk numbers whose only job is to eliminate, and
  they eliminate almost everything.**
- **Nothing about the conversion itself.** Costs are for running the donor's *shape* on the engine's
  rate classes. What attention→SSM conversion buys is visible as the KV column going to zero, and what
  it costs in quality is entirely unpriced.
- **No claim that the two gated repos would have failed.** They were not measured; they were not
  reachable.
