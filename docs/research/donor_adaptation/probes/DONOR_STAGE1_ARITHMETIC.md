# Donor Adaptation — Stage-1 Arithmetic Pass

**Mandate:** `docs/prompts/master_prompts/DONOR_MODEL_ADAPTATION.md` §9 stage 1 — *"Redo §5 properly for
a shortlist of real donors: footprint inventory, streaming budget, KV budget, head cost, per-SKU
eligibility. All desk, all free. Gate: a ranked shortlist with the arithmetic shown; any donor whose
arithmetic fails is eliminated here, before any compute."*

**Status:** DONE — **rev C, 2026-08-21**. Five-term time model (audit C12), measured MLP constant (`DONOR_PROJ_RATE.md` §10), verdict published as a 2×2 + design curve (audit C11). Desk only. No weights downloaded, no GPU, no engine change, no conversion.
**Tool:** `benchmarks/donor_adaptation/donor_inventory.py`.
**Author role:** Builder. Nothing here is committed or pushed; the Media Manager owns Git.

---

## 0. The headline, stated plainly

> **REVISED 2026-08-21 (rev C).** Rev A said *"not one of the 18 donors passes"* — withdrawn, it stood
> on an unmeasured projection rate. Rev B replaced that with a single PASS — **also withdrawn**, because
> `CONTROLLER_PROJRATE_AUDIT.md` found the time model was **missing two of the five components**
> `PHASE64_BUDGET.md` §2 lists (scan-recurrence and attention), and the margin was only ~2 ms. Rev C adds
> both terms, measures the last open constant, and reports the verdict as a **2×2 and a design curve**
> instead of a headline number. §0.4 records what changed.

**There is no single headline number, and presenting one was the defect.** The verdict moves with three
things: the KV precision, the ternary-MLP constant, and how many attention layers a conversion retains.

### 0.1 The 2×2 the headline was hiding (audit C11)

`Qwen/Qwen2.5-1.5B`, SKU-A / 32K, **unconverted** (attention on all 28 layers, priced windowed),
proj held at the measured donor stream in every cell:

| MLP constant | KV = 4-bit *(does not exist in this engine)* | KV = fp16 *(what is built)* |
|---|---|---|
| **RETIRED 11.398** (integrated `D=256`, and in fact the **dReLU row-skip** path) | 9.40 [8.32..9.96] **FAIL** | 8.12 [7.26..8.59] **FAIL** |
| **MEASURED 21.25** (integrated, donor `D`) | **12.09 [10.36..13.01] PASS** | 10.05 [8.76..10.77] **STRADDLE** |
| *kernel-pure 27.64 (bound only, unreachable by an engine)* | *13.08 [11.08..14.18] PASS* | *10.73 [9.27..11.56] STRADDLE* |

**The rev-B PASS lived in one cell.** The audit was right: three constants each flip this verdict, and
the one measured first (projection) was the smallest lever. Reproduce with
`python donor_inventory.py grid`.

### 0.2 The design curve (audit C12) — what a conversion actually has to buy

For a converted donor the number of **retained attention layers** is a design variable, not a constant.
Every layer not retained becomes an SSM layer: it pays scan-recurrence instead of attention **and stops
holding a KV cache**. `python donor_inventory.py curve`:

| retained attention | pricing | 4-bit KV | fp16 KV |
|---|---|---|---|
| **windowed (SWA-128)** | the only attention cost this engine has ever measured | **PASS at every `n_att`, 0→28** | **PASS for `n_att` ≤ 18 of 28**; STRADDLE from 19 |
| **full attention @32K** | first-principles compute desk model, *not measured* | PASS for `n_att` ≤ 2 of 28; FAIL from 3 | PASS for `n_att` ≤ 2; FAIL from 3 |

**The crossing points are the deliverable.** For `Qwen2.5-1.5B` at 32K:
- **Retained attention must be WINDOWED.** At most **2 of 28** layers may keep *full* attention; a third
  drops it under the gate on either KV precision.
- If retained attention is windowed, the binding constraint is KV precision, not layer count: at 4-bit
  every configuration passes; at fp16 the budget is **18 of 28** layers.
- The sealed constraint is *"attention on a minority of layers"*, i.e. `n_att ≤ 14`. **At `n_att = 14`,
  windowed, the donor passes on both KV precisions** — 13.30 [11.77..14.11] at 4-bit, 11.97
  [10.67..12.68] at fp16. That is the headline configuration, and it is a *converted* one.

**Do not price full attention at SWA rates.** §2's 65 µs anchor is a window-128 cost; the two pricings
above differ by ~25× at 32K. Scaling the SWA form linearly by `ctx/128` (a 256× extrapolation) is
retained in the tool as `full_swaScaled` and is an **upper bound only** — it is ~25× more pessimistic
than the compute model and would say a *single* retained full-attention layer kills the donor.

### 0.3 The ranked result (converted: `n_att = L/2`, windowed, 4-bit KV, each donor at its own native ctx)

| donor | RAM | nat ctx | central | interval | verdict |
|---|---|---|---|---|---|
| **`Qwen/Qwen2.5-1.5B`** | 1.94 GB | 131072 | **13.30** | [11.77..14.11] | **PASS** |
| `HuggingFaceTB/SmolLM2-1.7B` | 3.30 GB | 8192 | 11.01 | [9.76..11.64] | **STRADDLE** |
| `allenai/OLMoE-1B-7B-0924` | 5.24 GB | 4096 | 9.99 | [6.79..17.37] | FAIL (KV-only; 10.15 KV-free) |
| `Qwen/Qwen3-1.7B` | 2.68 GB | 40960 | 9.02 | [8.01..9.51] | FAIL (KV-only; 10.03 KV-free) |
| *(14 others)* | — | — | 5.10 → 1.04 | — | FAIL by 2–13× |

**Unconverted**, with full attention on every layer at 32K, `Qwen2.5-1.5B` is **4.70 tok/s — FAIL**.
The conversion is not a detail; it is the entire difference between failing and passing.

### 0.4 What changed from rev B

1. **Two missing components added** (audit C12 BLOCK). The model charged proj + LUT + KV + glue;
   §2 lists five. **scan-recurrence** (`46 µs · Dn·N·L / (512·96·6)`) and **attention** are now charged,
   in §2's own parametric forms — verified against the live engine, which measures scan-recur 36.5 µs
   and SWA-attn 48.5 µs at `D=256/L=6` against §2's 46 and 65.
2. **The ternary-MLP bracket was measured and closed** — `docs/research/DONOR_PROJ_RATE.md` §10.
   **27.31 ± 0.46 ms/token = 21.25 ± 0.36 GB/s** at donor dims, integrated. The old `[11.40 .. 27.64]`
   spanned two quantities *neither of which was the answer*.
3. **`DENSE_LUT_GBS = 11.398` was doubly misattributed.** It is engine-integrated at `D=256` **and**
   derives from the published 207 µs, which this pass identified as the `--skip on --exp fast` (E3.5)
   **dReLU row-skip** configuration. A SiLU-gated donor cannot use that sparsity. Wrong `D`, wrong path.
4. **Higher central rates adopted from the audit**, which re-derived them independently and got more
   than this Builder measured: fullstack **39.87 ± 0.09** (Builder: 37.74), asymptote **39.18**
   (Builder: 38.84). The interval now spans *both* labs rather than one lab's error bars.
5. **KV now scales with retained attention layers.** Converting a layer to SSM removes its KV entirely —
   previously the KV term was charged as if every original attention layer survived conversion.

### 0.5 What did NOT change

The three structural findings of rev A survive and are strengthened:
1. **The binding wall is throughput, not footprint** — 14 of 18 donors fit SKU-A's 16 GB.
2. **§5's flat 42 GB/s over-predicts** — the measured streamed rate is 39.87 GB/s and the whole weight
   path (proj 41.07 ms + MLP 27.31 ms) dominates everything else.
3. **The P61/D9 map does not survive scale-up** — `granite-4.0-h-small` costs 15.92 GB/token of fp32.
   Ternarizing the projections remains on the critical path, unpriced at donor scale.

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

### 1.2 Rate constants — measured

The mandate's §5 uses a flat 42 GB/s. **That number is superseded** and is used nowhere in the tool.

**Two different projection rates were measured and they go in two different slots.** Conflating them is
the defect this revision exists to remove.

| slot | rate | source |
|---|---|---|
| **whole per-token proj+head stream** | **37.74 ± 0.18 GB/s** | `DONOR_PROJ_RATE.md` §5.2 — Qwen2.5-1.5B's 28×(q,k,v,o)+head walked **in layer order**, 1,550,057,472 B/token (byte accounting exact vs the F1 hand-derivation) = 41.07 ms/token |
| **a single organ priced by its own size** | **38.84 ± 0.68 GB/s**, interval widened down to 36.0 | `DONOR_PROJ_RATE.md` §4 — the size-swept asymptote, 36 measurements, **flat from 64 MB to 1024 MB**; §5.1 measured 8 real donor organs spanning 36.0–39.0 |

The first is used for the per-token time term because **it is the same object the term models** — an
assembled stream, not a lookup — and it already carries the cost of interleaving the two skinny K/V
organs between the square ones, which is why it sits ~3% under the second.

| organ class | rate | source |
|---|---|---|
| fp32 projections / head | **37.74 ± 0.18 GB/s**, measured end-to-end at donor scale | `DONOR_PROJ_RATE.md` §5.2 |
| dense ternary MLP | **[11.40 .. 27.64] GB/s** — *asymmetric on purpose, see below* | §1 (low) / `DONOR_PROJ_RATE.md` §8.4 (high) |
| routed ternary experts | **[4.2 .. 26.4] GB/s** | §1 (low) / `DONOR_PROJ_RATE.md` §8.2 (high) |
| per-expert dispatch overhead | 8.4 µs, added **only** at the kernel-pure end | §1b(b) |
| KV-cache sequential read | 40–44 GB/s, central 42 | §1, measured |
| norms + glue | 7 µs at L=6 scaled linearly — **declared, not measured**, carried with a ×2 interval | §2 |

**The retired constant.** `PROJ_STREAMED_FLOOR = 34.0` is gone. It was never measured; it was the bottom
edge of an asserted bracket, and standing on that edge is what produced "zero donors pass". It is 11%
below the measured donor stream.

**The published §1b curve is no longer the pricing curve.** Re-measurement (`DONOR_PROJ_RATE.md` §2)
found the t1 row reproduces at all 8 points and the 96 MB anchor reproduces (+4.9%), but the 16–32 MB
cache-transition points **move by up to 4.6× on OpenMP thread placement alone**, and `engine.c
--gemv-sweep` — the routine that produced them — does not reproduce its own mid-region either. Those
points are placement-conditioned, not machine facts. **No donor row depends on them** (all donor organs
sit past 48 MB, where the curve is placement-insensitive to 2%), but `PHASE64_BUDGET.md` §1b should
carry the caveat.

**The ternary bracket is asymmetric and the caveat must stay visible.** Its low end (11.40 / 4.2 GB/s) is
**engine-integrated at `D=256`** — it carries dequant, dReLU, per-row scale multiply and combine. Its
high end (27.64 / 26.4 GB/s) is **kernel-pure at donor shapes**. *These are not the same quantity*, and
this tool **cannot** say where between them an integrated donor lands. Closing that bracket needs an
engine-integrated measurement at donor dimensions and is the single largest remaining source of width in
every interval in this document.

**Both old endpoints were `D=256` artefacts.** `DONOR_PROJ_RATE.md` §8 re-measured the path at donor
dimensions and the mechanism changed: holding the kernel fixed and growing the working set 1.1 MB →
512 MB drops the rate **3.6× (96.4 → 27.3 GB/s)**, so the path is **bandwidth-bound at donor working
sets, not compute-bound**. Controller #2's "compute-bound by ~16×" was measured at `D=256`, where the
kernel's own ceiling is only 29 GB/s; at `D=1536` that ceiling is 97 GB/s. **The claim does not transfer
across `D` and must be scoped to the sandbox.**

### 1.3 The time columns

**The single-corner "MEASURED-ONLY" column is WITHDRAWN** (audit F1). It was built as
`t_proj_pess + t_lut_meas + glue + t_kv_pess` — the pessimistic edge of every bracket simultaneously —
and reported as a point verdict. A corner is not a conservative estimate; it is a different quantity.

Every term now carries `(slow, fast, central)`:

| term | slow edge | fast edge | central |
|---|---|---|---|
| proj + head | 37.38 GB/s | 38.10 GB/s | **37.74 (measured)** |
| dense ternary MLP | 11.40 (integrated, `D=256`) | 27.64 (kernel-pure, donor `D`) | midpoint **in time** |
| routed experts | 4.2 (+dispatch inside) | 26.4 (+8.4 µs/call added) | midpoint **in time** |
| KV | 40 GB/s | 44 GB/s | 42 |
| glue | 2× declared | ½ declared | declared |

The reported total is the **sum of centrals**; the reported interval is the **sum of slows / sum of
fasts** — a fully-correlated worst case, deliberately *wider* than a root-sum-square. The terms share
one memory system, so RSS would understate.

Central time is the midpoint **of the time interval**, not of the rate interval — the neutral choice
when terms are about to be summed, and it does not silently favour the fast end.

**Gate reading rule:** `PASS` requires the **lower bound** to clear 10 tok/s. A central estimate on the
right side of a gate whose interval straddles it is reported as **STRADDLE**, never as a pass.

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
| `Qwen/Qwen2.5-1.5B` | 1544M | 1.44 GB | 39.87 | 553.3 MB | 0.0 KB | 0 | 2.75 | 8.23 | **4.83** |
| `Qwen/Qwen3-1.7B` | 1721M | 2.47 GB | 39.87 | 505.5 MB | 0.0 KB | 0 | 2.05 | 6.04 | **3.58** |
| `HuggingFaceTB/SmolLM2-1.7B` | 1711M | 1.88 GB | 39.87 | 577.7 MB | 0.0 KB | 0 | 2.39 | 7.03 | **4.16** |
| `microsoft/Phi-3-mini-4k-instruct` | 3723M | 4.87 GB | 39.87 | 1.13 GB | 0.0 KB | 0 | 1.15 | 3.18 | **1.96** |
| `allenai/OLMo-2-1124-7B` | 6887M | 9.53 GB | 39.87 | 2.02 GB | 0.0 KB | 0 | 0.78 | 1.90 | **1.27** |
| `mistralai/Mistral-7B-v0.3` | 7114M | 5.50 GB | 39.87 | 2.63 GB | 0.0 KB | 0 | 0.84 | 2.23 | **1.41** |
| `Qwen/Qwen3-8B` | 7568M | 7.94 GB | 39.87 | 2.54 GB | 0.0 KB | 0 | 0.73 | 1.89 | **1.21** |
| `Qwen/Qwen3-30B-A3B` | 3042M | 4.58 GB | 39.87 | 0.0 KB | 869.2 MB | 384 | 0.98 | 3.45 | **1.75** |
| `mistralai/Mixtral-8x7B-v0.1` | 12749M | 5.49 GB | 39.87 | 0.0 KB | 5.26 GB | 64 | 0.42 | 1.87 | **0.74** |
| `deepseek-ai/DeepSeek-V2-Lite` | 2451M | 2.18 GB | 39.87 | 32.2 MB | 861.7 MB | 182 | 1.52 | 5.95 | **2.73** |
| `allenai/OLMoE-1B-7B-0924` | 1179M | 1.39 GB | 39.87 | 0.0 KB | 386.0 MB | 128 | 2.78 | 10.34 | **4.97** |
| `openai/gpt-oss-20b` | 3607M | 4.54 GB | 39.87 | 0.0 KB | 1.12 GB | 96 | 1.13 | 3.87 | **1.94** |
| `Zyphra/Zamba2-2.7B` | 6949M | 10.07 GB | 39.87 | 1.98 GB | 0.0 KB | 0 | 1.79 | 2.51 | **2.22** |
| `ibm-granite/granite-4.0-h-small` | 8803M | 15.92 GB | 39.87 | 0.0 KB | 2.12 GB | 440 | 0.89 | 1.85 | **1.25** |
| `nvidia/Nemotron-H-8B-Base-8K` | 7564M | 12.43 GB | 39.87 | 1.97 GB | 0.0 KB | 0 | 1.71 | 2.20 | **2.02** |
| `tiiuae/Falcon-H1-7B-Base` | 7186M | 8.21 GB | 39.87 | 2.32 GB | 0.0 KB | 0 | 0.78 | 1.96 | **1.28** |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | 3463M | 6.71 GB | 39.87 | 0.0 KB | 798.2 MB | 528 | 1.77 | 4.01 | **2.62** |
| `state-spaces/mamba2-2.7b` | 2702M | 10.07 GB | 39.87 | 0.0 KB | 0.0 KB | 0 | 3.34 | 3.67 | **3.62** |

Projection rate is no longer a curve lookup. Every donor is priced on the **measured donor stream, 37.74 ± 0.18 GB/s** (`DONOR_PROJ_RATE.md` §5.2) — the whole per-token proj+head traffic timed in layer order, which is the same object this term models. The retired 34.0 floor was never measured and is 11% below it.


### T6 — tok/s including KV-cache read traffic, vs the sealed ≥10 tok/s gate

Reported as **central [interval]**. The single-corner "MEASURED-ONLY" column is **withdrawn** (audit F1): it took the pessimistic edge of every bracket at once and that is what produced "zero donors pass". `nat` = the donor's own `max_position_embeddings`; a context beyond it is marked `>nat` and is **not natively servable** without RoPE extension or the recall tier.

| donor | nat ctx | 32K central [lo..hi] | 128K central [lo..hi] | UNCONVERTED | **CONVERTED** central [lo..hi] | gate | KV-free |
|---|---|---|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B` | 131072 | 4.70 [2.71..7.88] | 4.36 [2.58..7.00] | 4.70 | 13.30 [11.77..14.11] | **PASS** | 13.82 |
| `Qwen/Qwen3-1.7B` | 40960 | 3.31 [1.95..5.35] | 2.71 [1.72..3.98] `>nat` | 3.31 | 9.02 [8.01..9.51] | FAIL (KV-only) | 10.03 |
| `HuggingFaceTB/SmolLM2-1.7B` | 8192 | 3.59 [2.18..5.59] `>nat` | 2.54 [1.72..3.46] `>nat` | 7.76 | 11.01 [9.76..11.64] | **STRADDLE** | 11.62 |
| `microsoft/Phi-3-mini-4k-instruct` | 4096 | 1.70 [1.05..2.58] `>nat` | 1.22 [0.84..1.65] `>nat` | 4.20 | 4.83 [4.31..5.07] | **FAIL** | 4.95 |
| `allenai/OLMo-2-1124-7B` | 4096 | 1.12 [0.72..1.60] `>nat` | 0.84 [0.59..1.09] `>nat` | 2.35 | 2.61 [2.36..2.71] | **FAIL** | 2.65 |
| `mistralai/Mistral-7B-v0.3` | 32768 | 1.36 [0.82..2.12] | 1.23 [0.77..1.83] `>nat` | 1.36 | 3.20 [2.89..3.36] | **FAIL** | 3.34 |
| `Qwen/Qwen3-8B` | 40960 | 1.17 [0.72..1.79] | 1.06 [0.67..1.56] `>nat` | 1.17 | 2.65 [2.39..2.77] | **FAIL** | 2.75 |
| `Qwen/Qwen3-30B-A3B` | 40960 | 1.69 [0.96..3.25] | 1.54 [0.91..2.76] `>nat` | 1.69 | 3.64 [2.59..5.66] | **FAIL** | 3.78 |
| `mistralai/Mixtral-8x7B-v0.1` | 32768 | 0.72 [0.41..1.79] | 0.69 [0.40..1.58] `>nat` | 0.72 | 1.04 [0.64..2.61] | **FAIL** | 1.06 |
| `deepseek-ai/DeepSeek-V2-Lite` | 163840 | 2.68 [1.51..5.75] | 2.56 [1.47..5.23] | 2.68 | 5.10 [3.36..9.77] | **FAIL** | 5.17 |
| `allenai/OLMoE-1B-7B-0924` | 4096 | 4.41 [2.58..8.26] `>nat` | 3.29 [2.14..5.15] `>nat` | 9.05 | 9.99 [6.79..17.37] | FAIL (KV-only) | 10.15 |
| `openai/gpt-oss-20b` | 131072 | 1.93 [1.12..3.80] | 1.87 [1.10..3.61] | 1.93 | 3.33 [2.28..5.71] | **FAIL** | 3.36 |
| `Zyphra/Zamba2-2.7B` | 4096 | 2.06 [1.68..2.31] `>nat` | 1.68 [1.41..1.87] `>nat` | 2.57 | 2.47 [2.24..2.57] | **FAIL** | 2.56 |
| `ibm-granite/granite-4.0-h-small` | 131072 | 1.24 [0.89..1.84] | 1.23 [0.88..1.81] | 1.24 | 1.28 [0.94..1.84] | **FAIL** | 1.30 |
| `nvidia/Nemotron-H-8B-Base-8K` | 8192 | 2.01 [1.70..2.19] `>nat` | 1.97 [1.67..2.15] `>nat` | 2.20 | 2.13 [1.90..2.22] | **FAIL** | 2.15 |
| `tiiuae/Falcon-H1-7B-Base` | 262144 | 1.27 [0.78..1.93] | 1.22 [0.76..1.84] | 1.27 | 2.76 [2.50..2.88] | **FAIL** | 2.79 |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | 262144 | 2.59 [1.76..3.94] | 2.50 [1.71..3.73] | 2.59 | 3.10 [2.32..4.31] | **FAIL** | 3.20 |
| `state-spaces/mamba2-2.7b` | ? | 3.62 [3.34..3.67] | 3.62 [3.34..3.67] | 3.62 | 3.40 [2.98..3.55] | **FAIL** | 3.40 |

KV priced at the measured aggregate DRAM stream [40–44 GB/s], central 42; 4-bit KV assumed, which **does not exist in this engine** (audit F8) — see the fp16 column in the arithmetic doc.

**`gate @SKU-A` reads PASS only when the LOWER BOUND clears 10 tok/s.** A central estimate on the right side of a gate whose interval straddles it is reported as straddling, not as a pass.

**KV-free central** isolates the owner's directive: a donor may **not** be eliminated solely because full-native KV does not fit or stream — that is what the recall tier absorbs. Any donor failing the gate on KV but clearing it KV-free is a *recall-tier* question, not an elimination.


### T7 — head alone (mandate §5's named first-class problem)

| donor | V | head bytes fp32 | ms/token fp32 | head bytes ternary | ms/token ternary (LUT bracket) | head share of the fp32 per-token stream |
|---|---|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B` | 151936 | 890.2 MB | 23.0–25.9 | 111.9 MB | 6.9–27.9 | 60% |
| `Qwen/Qwen3-1.7B` | 151936 | 1.16 GB | 30.7–34.6 | 149.0 MB | 9.2–37.2 | 47% |
| `HuggingFaceTB/SmolLM2-1.7B` | 49152 | 384.0 MB | 9.9–11.2 | 48.2 MB | 3.0–12.0 | 20% |
| `microsoft/Phi-3-mini-4k-instruct` | 32064 | 375.8 MB | 9.7–10.9 | 47.1 MB | 2.9–11.8 | 8% |
| `allenai/OLMo-2-1124-7B` | 100352 | 1.53 GB | 40.6–45.7 | 196.4 MB | 12.1–49.0 | 16% |
| `mistralai/Mistral-7B-v0.3` | 32768 | 512.0 MB | 13.2–14.9 | 64.1 MB | 4.0–16.0 | 9% |
| `Qwen/Qwen3-8B` | 151936 | 2.32 GB | 61.4–69.1 | 297.3 MB | 18.3–74.2 | 29% |
| `Qwen/Qwen3-30B-A3B` | 151936 | 1.16 GB | 30.7–34.6 | 149.0 MB | 9.2–37.2 | 25% |
| `mistralai/Mixtral-8x7B-v0.1` | 32000 | 500.0 MB | 12.9–14.6 | 62.6 MB | 3.9–15.6 | 9% |
| `deepseek-ai/DeepSeek-V2-Lite` | 102400 | 800.0 MB | 20.7–23.3 | 100.4 MB | 6.2–25.1 | 36% |
| `allenai/OLMoE-1B-7B-0924` | 50304 | 393.0 MB | 10.2–11.4 | 49.3 MB | 3.0–12.3 | 28% |
| `openai/gpt-oss-20b` | 201088 | 2.16 GB | 57.1–64.3 | 276.9 MB | 17.1–69.1 | 48% |
| `Zyphra/Zamba2-2.7B` | 32000 | 312.5 MB | 8.1–9.1 | 39.2 MB | 2.4–9.8 | 3% |
| `ibm-granite/granite-4.0-h-small` | 100352 | 1.53 GB | 40.6–45.7 | 196.4 MB | 12.1–49.0 | 10% |
| `nvidia/Nemotron-H-8B-Base-8K` | 131072 | 2.00 GB | 53.0–59.7 | 256.5 MB | 15.8–64.0 | 16% |
| `tiiuae/Falcon-H1-7B-Base` | 130048 | 1.49 GB | 39.4–44.4 | 191.0 MB | 11.8–47.7 | 18% |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | 151936 | 1.16 GB | 30.7–34.6 | 149.0 MB | 9.2–37.2 | 17% |
| `state-spaces/mamba2-2.7b` | 50288 | 491.1 MB | 12.7–14.3 | 61.6 MB | 3.8–15.4 | 5% |

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

**Revised elimination rule (rev B).** A donor is eliminated only if it misses a sealed gate **with its
interval**, at its **own native context**, and **not solely because of KV**. Three changes from rev A:

- Verdicts are read off the **interval**, not a corner. `PASS` needs the lower bound ≥ 10.
- Each donor is priced at `min(32768, max_position_embeddings)` — its **own** capacity (audit F7).
- **Owner directive: a donor may not be eliminated solely because full-native KV does not fit or
  stream.** That is what the recall tier absorbs. The `KV-free` column isolates exactly this case.

| donor | nat ctx | SKU-A eff | tok/s central | interval | KV-free | eliminated by |
|---|---|---|---|---|---|---|
| **`Qwen/Qwen2.5-1.5B`** | 131072 | 32768 | **12.10** | [10.17..14.92] | 12.98 | — **PASSES** |
| `allenai/OLMoE-1B-7B-0924` | **4096** | 4096 | 10.08 | [7.16..17.04] | 10.42 | — **STRADDLE, undecided** |
| `HuggingFaceTB/SmolLM2-1.7B` | **8192** | 8192 | 9.95 | [8.54..11.91] | **11.00** | **KV only** — see below |
| `Qwen/Qwen3-1.7B` | 40960 | 32768 | 7.96 | [7.09..9.07] | 9.69 | tok/s; KV-free still < 10 |
| `deepseek-ai/DeepSeek-V2-Lite` | 163840 | 32768 | 5.11 | [3.48..9.60] | 5.28 | tok/s (MLA KV is already cheap) |
| `microsoft/Phi-3-mini-4k-instruct` | **4096** | 4096 | 4.48 | [3.90..5.26] | 4.68 | tok/s |
| `Qwen/Qwen3-30B-A3B` | 40960 | 32768 | 3.61 | [2.71..5.40] | 3.88 | **both**: 16.06 GB > 16 GiB, and tok/s |
| `state-spaces/mamba2-2.7b` | ? | 32768 | 3.49 | [3.46..3.52] | 3.49 | tok/s; row PARTIAL |
| `openai/gpt-oss-20b` | 131072 | 32768 | 3.34 | [2.38..5.60] | 3.39 | tok/s; also already-4-bit |
| `Qwen/Qwen3-Next-80B-A3B` | 262144 | 32768 | 3.19 | [2.52..4.35] | 3.24 | **both**; row `FORMULA_DISAGREEMENT` |
| `mistralai/Mistral-7B-v0.3` | 32768 | 32768 | 2.80 | [2.31..3.55] | 3.02 | tok/s |
| `tiiuae/Falcon-H1-7B-Base` | 262144 | 32768 | 2.52 | [2.15..3.03] | 2.58 | tok/s |
| `allenai/OLMo-2-1124-7B` | **4096** | 4096 | 2.39 | [2.09..2.78] | 2.47 | tok/s |
| `Zyphra/Zamba2-2.7B` | **4096** | 4096 | (2.36) | [2.08..2.74] | 2.39 | **`FORMULA_DISAGREEMENT` +164%** — no valid number |
| `Qwen/Qwen3-8B` | 40960 | 32768 | 2.36 | [2.01..2.86] | 2.53 | tok/s |
| `nvidia/Nemotron-H-8B-Base-8K` | **8192** | 8192 | 2.06 | [1.84..2.34] | 2.06 | tok/s |
| `ibm-granite/granite-4.0-h-small` | 131072 | 32768 | 1.30 | [1.00..1.85] | 1.30 | **both**: 16.20 GB > 16 GiB, and tok/s |
| `mistralai/Mixtral-8x7B-v0.1` | 32768 | 32768 | 1.04 | [0.65..2.54] | 1.07 | **both**: 23.79 GB, and tok/s |
| `meta-llama/Llama-3.2-1B` | — | — | — | — | — | **UNAVAILABLE (401 gated)** |
| `ai21labs/AI21-Jamba-Mini-1.6` | — | — | — | — | — | **UNAVAILABLE (401 gated)** |

### 5.0 Eliminations that CHANGED

- **`Qwen/Qwen2.5-1.5B`: FAIL → PASS.** Sole cause: the measured projection rate. 9.77 → 12.10.
- **`allenai/OLMoE-1B-7B-0924`: FAIL → STRADDLE.** Two causes, both corrections rather than concessions.
  It is a **4096-position model** that rev A scored at 32K and 128K (audit F7) — contexts it cannot
  natively serve. At its own 4K its KV term collapses. Its interval is the widest in the set
  (7.16–17.04) because it is the donor most exposed to the open ternary bracket. **Undecided, and
  decidable by one engine-integrated LUT measurement at donor `D`.**
- **`HuggingFaceTB/SmolLM2-1.7B`: eliminated → KV-ONLY elimination, re-opened.** An **8192**-position
  model rev A scored at 32K. At its own 8K it reaches 9.95 [8.54..11.91], and **KV-free it reaches
  11.00**. Per the owner's directive it may **not** be eliminated on full-native KV alone. But its
  KV-free *lower bound* is 9.34, still under 10 — so it re-enters as **undecided**, not as a pass. Its
  MHA (no GQA) KV of 192 KB/context-token is the heaviest per parameter in the set.
- **`Phi-3-mini`, `OLMo-2-7B`, `Zamba2-2.7B` (4K) and `Nemotron-H-8B` (8K)**: all re-priced at their own
  native contexts. Every one improves (e.g. Phi-3 3.45 → 4.48) and **every one still fails by 2–4×.**
  The F7 defect was real but changed no verdict for these four.

**No elimination was reversed by the rate correction alone except the top two.** The 15 donors below
`Qwen3-1.7B` fail by 2–12×; no reading of any remaining bracket reaches the gate.

### 5.1 The ranked shortlist

**SKU-A (16 GB, native attention window capped at each donor's own capacity, 4-bit KV):**

1. **`Qwen/Qwen2.5-1.5B` — the only pass.** 1.94 GB resident; **12.10 tok/s [10.17..14.92]**.
   Arithmetic: proj+head fp32 1.550 GB/token ÷ 37.74 GB/s = **41.07 ms** (measured end-to-end);
   ternary MLP 580 MB/token (codes + per-row scales) ÷ [11.40..27.64] = 20.99–50.91 ms; KV 224 MB ÷ 42 = 5.6 ms; glue 33 µs.
   Apache-2.0. GQA with 2 KV heads = 28 KB/context-token, the second-lightest KV in the set. **Passes
   at the fully-pessimistic corner (10.17), which is what makes it a pass and not a hope.**
   *Caveat: at fp16 KV — the only precision built — it straddles at 10.06 [8.62..12.04] (§0.1).*
2. **`allenai/OLMoE-1B-7B-0924` — undecided.** 5.24 GB; 10.08 [7.16..17.04] at its native 4K. Only
   1.18B active of 6.92B — the best active-to-total ratio here and the strongest evidence for §8.A's
   "sparsity is worth more than small total size". **Its verdict turns entirely on the open ternary
   bracket**, and a 4K native window is a hard limit that the recall tier would have to carry
   completely.
3. **`HuggingFaceTB/SmolLM2-1.7B` — undecided, re-opened on the KV directive.** 3.30 GB; 9.95
   [8.54..11.91] at its native 8K, 11.00 KV-free. Not eliminated (its only failure is KV), not passing
   (KV-free lower bound 9.34 < 10).
4. `deepseek-ai/DeepSeek-V2-Lite` — 8.59 GB; 5.11 [3.48..9.60]. Eliminated on speed, but **MLA remains
   the standout KV result**: 30.4 KB per context token across 27 layers, vs SmolLM2's 192 KB with 4×
   fewer parameters. Licence `other` — redistribution needs a read (§8.A q6).

**SKU-B (64 GB, 128K native attention, no recall path permitted):** `Qwen2.5-1.5B` reaches **10.06
[8.62..12.04] — STRADDLE**, everything else fails by 2×+. Rev A's conclusion that *no donor passes
SKU-B* is **weakened but not overturned**: the best donor now straddles rather than clearly failing.
All 18 reachable donors fit SKU-B's 64 GiB under the all-ternary map, the largest being
`Qwen3-Next-80B` at 39.10 GB — **RAM is not what eliminates anyone at SKU-B.**

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
2.0–4.0× on every row. **Rev B update:** that figure has now been re-measured at donor dimensions and the picture changed — the LUT path is *bandwidth*-bound there, with a kernel ceiling of 97 GB/s at `D=1536` against 29 at `D=256`, so the sandbox reading does not transfer. The live bracket is [11.40 .. 27.64] GB/s, integrated-vs-kernel-pure, and it is now the difference between **one donor passing and two** — not between one and none, because the measured projection rate carries `Qwen2.5-1.5B` over the gate at the pessimistic end of it.

---

## 7. Assumptions, and what would invalidate this

Ordered by how much a verdict would move if the assumption is wrong.

1. **Ternary projections are priced at the expert LUT bracket in the all-ternary column, and that rate
   has never been measured for projection shapes.** Marked UNMEASURED in every row that uses it. If
   ternary projections run nearer the fp32 GEMV curve than the LUT rate, the all-ternary column improves
   substantially and more donors approach the gate. *This is the single most valuable cheap measurement
   available and it needs no GPU.*
2. **~~The proj/head rate above 96 MB is an extrapolation.~~ RESOLVED — it is now MEASURED.**
   `docs/research/DONOR_PROJ_RATE.md` swept the curve to **1024 MB** and measured the asymptote at
   **38.84 ± 0.68 GB/s** (36 measurements, flat across 16× of size, no downward trend and no further
   cliff past 96 MB), then measured the *actual* Qwen2.5-1.5B per-token stream end-to-end at
   **37.74 ± 0.18 GB/s**. Every donor row now rests on a measured point, not an extrapolated one.
   The retired 34.0 floor was **11% low**, and correcting it is what flipped the headline. The §1b
   worry that streamed experts pollute L3 and push real rates toward the tail is answered: the tail
   *is* the measured rate, and it is placement-insensitive to 2% at every donor-relevant size.
   *Residual:* one machine (3600X), and the ternary path's kernel-pure/engine-integrated bracket is
   still open — see item 2b.

2b. **The ternary MLP bracket is the largest remaining source of uncertainty, and it is now the only
   one that matters.** Its low end (11.40 GB/s dense / 4.2 routed) is engine-integrated at `D=256`;
   its high end (27.64 / 26.4) is kernel-pure at donor `D`. **Different quantities.** For
   `Qwen2.5-1.5B` this bracket alone spans 20.99–50.91 ms of a ~68–98 ms token, i.e. essentially the
   entire reported interval. Closing it needs an engine-integrated LUT measurement at donor
   dimensions — **the single most valuable cheap measurement now available, no GPU required.**
   It decides `OLMoE` (10.08 [7.16..17.04]) outright.

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
| `donor_inventory.py` sha256 | `2951bce18adf909378de13e52f03a7e4f4d4323aa11c2ab255a8bdb834ad5c1f` |
| `donor_inventory.py` bytes | 76043 |
| rate-constant source | docs/research/DONOR_PROJ_RATE.md (measured 2026-08-20, 3600X) + docs/PHASE64_BUDGET.md sec.1 (DRAM aggregate) |
| proj donor-stream GB/s (MEASURED, used for the per-token term) | [37.38, **37.74**, 38.10] |
| proj single-organ asymptote GB/s (MEASURED, flat 64->1024 MB) | [36.0, **38.84**, 40.20] |
| RETIRED constant (never measured, 11% low) | PROJ_STREAMED_FLOOR = 34.0 |
| ternary routed-expert GB/s [integrated D=256 .. kernel-pure donor D] | [4.2, 26.4] |
| ternary dense-MLP GB/s [integrated D=256 .. kernel-pure donor D] | [11.398, 27.64] |
| ternary kernel-pure ceiling GB/s (D=256 sandbox shape) | 17.0 (reproduced at 18.6-18.9) |
| expert dispatch overhead | 8.4 us/call, added ONLY at the kernel-pure end (was double-counted) |
| DRAM aggregate GB/s (KV) | [40.0, 42.0, 44.0] |
| bracket policy | central estimate + fully-correlated interval; the single-corner MEASURED-ONLY column is WITHDRAWN |
| gate reading | PASS requires the LOWER BOUND >= 10 tok/s; central-only is reported as STRADDLE |
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
- **Still a desk model, but no longer an unmeasured one.** Rev A's rates came from the 8.3M sandbox and
  one synthetic microbench grid. Rev B's projection term is measured **at donor scale, on the donor's
  own byte count, end-to-end** (`DONOR_PROJ_RATE.md` §5.2) — that is the term that decides the verdict,
  and it is no longer an extrapolation. What remains modelled rather than measured: the ternary MLP
  bracket (integrated vs kernel-pure, §7 item 2b), **zero overlap between organs** (real engines overlap
  some of this — direction: pessimistic), the linearly-scaled glue term, and 4-bit KV, which **does not
  exist in this engine**. §5's warning that *"a speedup measured in a microbenchmark and reported as a
  system speedup"* is a named project failure still applies to every rate here that is kernel-pure.
- **One machine.** All rates are the 3600X reference floor. Per `feedback_portability_no_hardfit` they
  are runtime parameters of a hardware class, not constants of the architecture.
- **Not independently audited.** Rev B is a Builder artefact. Rev A's audit
  (`CONTROLLER_STAGE1_AUDIT.md`) is what prompted this revision; rev B has not itself had a Controller
  pass, and one is in progress on the underlying measurement.
- **Nothing about the conversion itself.** Costs are for running the donor's *shape* on the engine's
  rate classes. What attention→SSM conversion buys is visible as the KV column going to zero, and what
  it costs in quality is entirely unpriced.
- **No claim that the two gated repos would have failed.** They were not measured; they were not
  reachable.
