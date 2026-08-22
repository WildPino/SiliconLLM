# Controller-2 Independent Replication — §7.1 two-key audit of `CONTROLLER_AUDIT_MANDATE.md`

**Status: DRAFT, read-only on all code. 2026-08-20. Branch `codex/research/donor-adaptation`.**

Every number below is re-derived from the primary sources named in the brief. Where I reach the first
Controller's number I say CONFIRM; where I reach a different one I show both. Two of the four answers
rest on evidence that neither the mandate nor the first Controller uses.

**Unit convention, declared once.** An engine expert is 48 **KiB** = 49152 B — forced by the code:
`3·D·h/2 = 3·256·128/2 = 49152` at `D=256, HID_E=128`. Rates are **decimal GB/s** (1e9 B/s), which is
the convention `run_expert_rate` itself uses (`(double)touches*EB/1e9/dt`, `engine.c:838`). KiB bytes
with 1e9 divisors is the only self-consistent reading of the 64.1b bench.

---

## Q3 (highest priority) — the "~8.4 µs/expert dispatch overhead" decomposition

### Q3.1 Replication of the first Controller's arithmetic

Anchors: `PHASE64_BUDGET.md` §1 (MoE 573 µs t1 → 543 µs t6) and §1b (kernel-pure 7.45 → 17.0 GB/s,
2.88 µs/expert), 48 touches/token.

| quantity | t1 | t6 |
|---|---|---|
| engine-integrated MoE, µs/token | 573 | 543 |
| ÷ 48 touches = µs/expert | 11.9375 | 11.3125 |
| kernel-pure µs/expert (49152 B ÷ rate) | 49152/7.45e9 = **6.598** | 49152/17.0e9 = **2.891** |
| **residual "overhead"** | **5.340 µs** | **8.422 µs** |

**CONFIRM, exactly.** The residual is 5.34 µs at t1 and 8.42 µs at t6 — it *grows 57.7 %* as threads go
1→6. A single fixed per-expert serial cost cannot do that. Both cross-checks replicate:

- Hold 8.42 µs constant and predict t1: `48 × (6.598 + 8.422) = 721.0 µs` vs measured 573 →
  **+25.8 % over-prediction.** CONFIRM (Controller-1 said 26 %).
- Hold 5.34 µs constant and predict t6: `48 × (2.891 + 5.340) = 395.1 µs` vs measured 543 →
  **−27.2 % under-prediction.** Controller-1 did not state this direction; it is the same defect seen
  from the other end, and it is the more damaging one because the budget document's forward-looking
  bracket is a t6 bracket.

**Unit slip: CONFIRM, and it is smaller than implied.** 2304 KiB = 2,359,296 B; ÷ 543 µs =
**4.345 GB/s**. The document's 4.2 is reproduced only by reading "2304 KB" as 2,304,000 B
(→ 4.243 GB/s). The slip is real but worth **2.4 %** — it is a hygiene finding, not a load-bearing
error, and it should not be presented as one. The t1 integrated rate, never stated in the budget doc,
is 2,359,296 / 573 µs = **4.117 GB/s**.

So §1b's decomposition is **arithmetically unsound as written**: "~8.4 µs/expert" is a t6-only number
presented as a structural constant, and it does not survive its own t1 datum. **I agree with the first
Controller on all of the above.**

### Q3.2 Where I disagree — the conclusion does not follow, and a better datum exists

Controller-1 reads the failed decomposition as falsifying "engineering lever, not a bandwidth wall",
and falls back to 4.34 GB/s as the design rate. **That inference is wrong, and the evidence against it
sits in the same table both parties were reading.**

**The dense LUT-MLP path streams the identical byte volume through the identical kernel.**
With `MLP_HID = 1024`, `D = 256`, `L = 6` (`engine.c:64-72`): per layer `gate` = `TUP·Mpg = 128·1024 =
131072 B`, `up` = 131072 B, `down` = `TDN·Mpd = 512·256 = 131072 B` → 393216 B/layer × 6 =
**2,359,296 B/token — byte-for-byte the same 2304 KiB the MoE path streams.** Same
`_mm256_shuffle_epi8` kernel, same `acc_add_i8x32`, same `bc_tm` packing.

| path (same 2,359,296 B/token, same kernel) | t1 µs | t1 GB/s | t6 µs | t6 GB/s |
|---|---|---|---|---|
| LUT-MLP **dense** (integrated engine) | 313 | **7.538** | 207 | **11.40** |
| LUT-MoE (integrated engine) | 573 | 4.117 | 543 | 4.345 |
| expert pool (kernel-pure bench, 512 MB DRAM) | 316.7 | 7.45 | 138.8 | 17.0 |

Three consequences, all of which change the answer:

1. **At t1 the integrated dense path (7.538 GB/s) equals the kernel-pure rate (7.45 GB/s) to within
   1.2 %.** The integrated engine carries essentially *zero* overhead over the raw kernel when the call
   structure is coarse. The 5.34 µs residual is therefore **specific to the MoE call structure**, not
   to "being integrated". Controller-1's fallback to 4.34 as *the* integrated rate generalises from a
   single pathological call pattern.
2. **The dense path's entire weight set is 2.25 MiB — L3-resident, re-read every token.** It has no
   DRAM traffic and no gather, and still tops out at 11.40 GB/s at t6, against the 185 GB/s the same
   machine sustains on resident fp32 GEMV (§1b(a)). **The LUT path is compute/kernel-bound by roughly
   16×, not bandwidth-bound.** This is the most important number in this audit and neither the mandate
   nor Controller-1 uses it.
3. Therefore **17.0 GB/s is the LUT kernel's own compute asymptote at t6**, measured with nothing else
   running: no scale-multiply, no dReLU, no router, no int8 re-quantization, no LUT rebuild. It is not
   an integrated-engine target. The best *integrated* rate this engine has demonstrated on this kernel
   is 11.40 GB/s — and that is an **upper bound** (see caveat 5 in the closing section).

**Verdict on Q3's real question — "is 17.0 GB/s defensible?": SOMETHING IN BETWEEN, and the defensible
number is ~11.4 GB/s, neither 17.0 nor 4.2.**

- 17.0 GB/s — **unsupported as a design rate.** Synthetic, weight-free, one call per expert, no glue.
- 4.2 / 4.345 GB/s — **not a floor either.** It is an artefact of `D=256, HID_E=128` (Q3.3).
- 11.4 GB/s — measured, integrated, same kernel, same bytes, on this engine, today.

**Recommendation:** `PHASE64_BUDGET.md` §2's LUT bracket **[4.2 – 17.0] should become [4.35 – 11.4]**,
and `DONOR_MODEL_ADAPTATION.md` §2.2 should stop quoting "~8.4 µs per expert" as a constant. The
*qualitative* claim — *engineering lever, not bandwidth wall* — **survives and is strengthened**,
because the mechanism is now identified rather than asserted. **I partially agree with Controller-1
(its arithmetic) and disagree with its conclusion (that the lever is illusory).**

### Q3.3 Per-CALL vs per-BYTE — can the data separate them?

**From the two published points alone: NO.** There is one expert size (48 KiB) and two thread counts.
`t = c + b·bytes` has two unknowns, and the two data points differ in *thread count*, not size.
Fitting across the two *different* harnesses instead (bench: 49152 B in 2.891 µs; dense: 131072 B in
207/18 = 11.5 µs) yields `b = 1.052e-4 µs/B` and `c = −2.29 µs` — a **negative** per-call constant,
i.e. the points are not from one regime. The fit is inadmissible and must not be quoted.

**But an indirect separation IS available, and it is decisive.** Dense and MoE stream *identical bytes
through an identical kernel* and differ only in call structure. A per-byte cost would be paid by both.

| | t1 | t6 |
|---|---|---|
| MoE − dense, µs/token (identical bytes) | 573 − 313 = 260 | 543 − 207 = 336 |
| ÷ 48 experts | 5.42 µs/expert | 7.00 µs/expert |
| ÷ 126 extra kernel calls (144 vs 18) | 2.06 µs/call | 2.67 µs/call |

**The excess is overwhelmingly per-CALL / per-call-structure, not per-byte.** Reading the code tells us
what those calls cost, and every identifiable term scales as O(1/D) or better against byte volume:

- `mlp_moe` (`engine.c:288-313`) issues **three** `matvec_lut_rows` per selected expert →
  8 top-k × 6 layers × 3 = **144 OpenMP fork/joins per token**, versus 18 for dense and 48 for the
  bench. `OMP_PFOR` sits *inside* the matvec (`engine.c:39, 132`), so every call forks and joins.
- The parallel chunk counts are pathological: gate/up run `M = HID_E = 128` → `128/32 = 4` chunks
  across 6 threads (2 threads idle, 1.5× waste); down runs `M = D = 256` → 8 chunks across 6 threads
  (1.33× waste). **This is precisely why the residual grows with thread count** — fork/join tax plus
  load imbalance, both monotonically increasing in thread count. Controller-1 correctly identified the
  growth as inconsistent with a fixed serial cost; it did not identify the cause, which is what makes
  the cost removable.
- Per expert, strictly serial: `quant_i8(he, HID_E)` = O(h) and `build_lut_t3(g_hq, TDE, ·)` = `16·h/2
  = 8h` scalar ops, plus three scale loops. Cost per byte = `8h / (1.5·D·h) = 5.33/D` — **independent
  of expert width, falling as 1/D.**
- **Layout — the term nobody has named.** `egate_cd` / `eup_cd` are one `GH×D` tile-major block per
  layer with `MPAD_GU = E·HID_E = 4096`. A per-expert row window therefore reads **128 chunks of 128 B
  at a 4096 B stride** — two cache lines per 4 KiB page, for **two thirds of every expert's bytes**.
  Only `eWd_cd` (per-expert block, `MPAD_D = 256`) is contiguous. This is a ρ-law violation hiding
  inside the "dispatch overhead" bucket, and unlike the fork/join tax it *is* per-byte — but its
  contiguous run is `h` bytes wide, so it too amortizes as expert width grows.

**Implication for donors — the finding that matters most here.** A donor expert at `D = 2048, h = 768`
is `3·D·h/2 = 2.36 MB`, **48× the engine's 48 KiB**. Calls per active byte fall ~6×; the serial
LUT-build/quant cost per byte falls 8× (1/D); the strided run widens from 128 B to 768 B.
**The 4.345 GB/s figure is an artefact of `D=256, h=128` and must not be carried into any donor
budget.** The right donor number is the contiguous integrated rate, ~11.4 GB/s — pending measurement.

**The measurement that would settle it** (cheap, CPU-only, no GPU, no weights; the harness exists).
`run_expert_rate` (`engine.c:818-841`) hard-codes `M=384, Mpad=384, T=128`. Sweep
`EB ∈ {48, 96, 192, 384, 768, 1536}` KiB at fixed pool size and fixed thread count; separately sweep
`D ∈ {256, 1024, 2048}` at fixed expert bytes. Fit `t = c + b·bytes` **per thread count** — the
thread-dependence of `c` is the actual finding, so t1 and t6 must both be run. **Until that exists,
[4.35 – 11.4] is a bracket, not a point.**

---

## Q1 — "SKU-B is provably empty"

### Q1.1 The case to work: large-total / small-active

The brief names the shape and a real donor has it (Qwen3-Next-80B-A3B class: ~80B total, ~3B active,
`D ≈ 2048`, `L ≈ 48`, `V ≈ 152K`). Decomposing the 3B active: projections (q/k/v/o or linear-attn
equivalents, GQA-4×128) ≈ `10.5M/layer × 48 = 503M`; output head `V·D = 311M`;
**experts ≈ 3B − 0.50B − 0.31B ≈ 2.19B**. Budget for 10 tok/s = **100 ms/token**.

Rates: fp32 GEMV at the §1b **fully-streamed floor 34–40 GB/s** (donor proj blocks are ~42 MB/layer,
squarely in the DRAM tail of the r(size) curve — I use 37); LUT path at 4.345 / 11.4 / 17.0.

| organ | active params | path in `engine.c` | fp32 bytes | ternary (0.5 B/w) | t fp32 | t ternary |
|---|---|---|---|---|---|---|
| projections | 0.503B | `matvec`/`dotf` fp32 (D9 seals these fp32) | 2.012 GB | 252 MB | **54.4 ms** @37 | **22.1 ms** @11.4 |
| experts | 2.19B | `matvec_lut_rows` | — | 1.095 GB | — | **252 / 96.1 / 64.4 ms** @4.345 / 11.4 / 17.0 |
| head | 0.311B | `matvec` fp32 | 1.244 GB | 156 MB | **33.6 ms** @37 | **13.7 ms** @11.4 |
| KV, 128K native, 12 attn layers, GQA-4, bf16 | — | stream | 3.22 GB | — | **87.0 ms** @37 | — |
| KV, 128K native, 2 attn layers, MQA, bf16 | — | stream | 134 MB | — | **3.6 ms** | — |
| per-position floor (scan + norms + glue, scaled) | — | — | — | — | ~10 ms | ~10 ms |

| scenario | total | tok/s |
|---|---|---|
| D9 as sealed (proj + head fp32), experts @4.345, 12 attn layers @128K | 437 ms | **2.3** |
| D9 as sealed, experts @11.4, 2 MQA layers + recall | 198 ms | **5.1** |
| **all-ternary** (D9 broken), experts @11.4, 2 MQA layers + recall | 146 ms | **6.9** |
| all-ternary, experts @17.0 (kernel asymptote), 2 MQA layers + recall | 114 ms | **8.8** |

RAM inventory, all-ternary at 0.5 B/weight: experts `78.9B × 0.5 = 39.5 GB` + projections 0.25 +
embed/head 0.31 + per-row scales (~38.5M rows × 4 B) 0.15 + KV 0.13 + scratch/logits ≈ **~41 GB**.
**Above 16 GB (SKU-A cannot hold it) and below 64 GB (SKU-B can).** SKU-B is genuinely *required* by
this donor — exactly the case a claim of "provably empty" must exclude.

### Q1.2 Verdict: **PARTIAL REFUTE. I disagree with the first Controller.**

"Provably empty" is unearned. This donor **misses 10 tok/s by 1.45× at the defensible rate (11.4) and
by 1.14× at the kernel asymptote (17.0)**. That is a near-miss inside the error bars of an unmeasured
quantity, not a proof. Controller-1's verdict is *directionally* right — no released donor clears the
gate today — but its modal claim is stronger than its evidence.

**Exact condition under which SKU-B is non-empty.** With all weights ternary at `B` bytes/weight and
the expert path at `R` GB/s:

```
(1) SKU-B trigger:  total_params × B > 16 GB          → at B = 0.5: total > 32B params
(2) speed gate:     active_params × B ≤ (100 ms − t_KV − t_floor) × R
(3) precision:      EVERY fp32 organ ternarized — D9 must be broken
(4) context:        128K native from ≤2 MQA-equivalent retained layers + the recall tier
```

With `t_KV + t_floor = 14 ms`, condition (2) evaluates to:

| R (GB/s) | max active ternary bytes/token | **max active params** |
|---|---|---|
| 4.345 (measured today, 48 KiB experts) | 0.374 GB | **0.75B** |
| **11.4 (defensible, contiguous integrated)** | 0.981 GB | **1.96B** |
| 17.0 (kernel asymptote) | 1.46 GB | **2.93B** |

**SKU-B is non-empty iff a donor exists with total > 32B AND active < ~1.96B params/token — a sparsity
ratio above ~17:1 in a model above 32B — with all projections and the head ternarized and 128K served
by recall rather than retained KV.**

**The binding quantity is absolute active params, not the sparsity ratio,** and that is the trap in
§8.A's framing. The SKU-B trigger is set by *total*; the speed gate is set by *active*; a donor must
be extreme on both axes simultaneously. Released models above 32B sit at 3–5B active (80B/A3B ≈ 3B;
120B-class ≈ 5.1B). None is below 1.96B. **So SKU-B is empty in practice at 11.4 GB/s, and becomes
non-empty at ~17 GB/s only if condition (3) is also paid.** A conditional predicate with the condition
named is what the mandate should carry, not a bare verdict in either direction.

**Condition (3) deserves its own alarm.** Under D9 as sealed, the fp32 organs alone
(`0.503B + 0.311B = 0.814B × 4 B = 3.26 GB`) cost **88 ms at 37 GB/s — 88 % of the entire 10 tok/s
budget, before a single expert byte is read.** D9 is sealed on a P61 measurement (+0.018–0.022 BPB)
taken on **SSM projections at 8.3M params**. It has never been measured on donor attention projections
at `D = 2048`. **Whether D9 holds at donor scale is the highest-leverage open question in the donor
program**, and neither the mandate nor Controller-1 flags it as the gate-breaker it is.

---

## Q4 — the "free 2×" nibble-packing claim

### Q4.1 What the code does — Controller-1's reading is CORRECT

`bc_tm` (`engine.c:160-162`): `codes[t*Mpad + m] = (int8_t)((w0+1)*3 + (w1+1))` with `w ∈ {−1,0,1}` →
**base-3, g = 2, 9 states (0…8), one `int8_t` per weight pair = 0.5 B/weight.** `build_lut_t3`
(`engine.c:158-159`) builds a 16-entry table per pair index `t`, with entries `c ≥ 9` zero-filled. The
kernel does `_mm256_shuffle_epi8(tbl, idx)` with `idx` = the loaded code bytes (`engine.c:115, 125,
134`). **CONFIRM: 9 states carry 4 bits of information and occupy 8 bits solely because `pshufb`
addresses byte lanes.** The 16-entry LUT with 7 dead slots is the fingerprint of that choice.

### Q4.2 The cost, quantified — and it lands on the wrong side

The clean nibble layout packs **two `t` indices per byte** (not two `m`), so the two halves feed two
different LUT tables into the *same* accumulator and no lane reordering is needed anywhere in the hot
loop. Per 32-byte load:

| | current | nibble-packed |
|---|---|---|
| codes per 32 B load | 32 (64 weights) | 64 (128 weights) |
| `vmovdqu` codes | 1 | 1 |
| unpack (`vpand`, `vpsrlw`, `vpand`) | 0 | **3** |
| LUT broadcast (`vmovdqu` + `vpbroadcasti128`) | 1 | 2 |
| `vpshufb` | 1 | 2 |
| `acc_add_i8x32` (1 `vextracti128` + 2 `vpsrldq` + 4 `vpmovsxbd` + 4 `vpaddd`) | 11 | 22 |
| **total ops per 64 codes** | **30** (= 2 × 15) | **33** |

**+10 % compute for −50 % bytes.** On a bandwidth-bound path that is a ~1.8× win. On a **compute-bound**
path it is a ~10 % **loss**.

**Q3.2 established that this path is compute-bound.** The dense LUT-MLP is fully L3-resident (2.25 MiB)
with zero gather and still runs at 11.40 GB/s where the same silicon does 185 GB/s on resident fp32
GEMV. Even the DRAM-streamed bench tops out at 17.0 GB/s against a 40–44 GB/s aggregate ceiling —
**2.4× of bandwidth headroom the kernel cannot consume.** Halving code bytes does not make the kernel
faster, because its throughput is denominated in *weights per second*, not bytes per second, and
nibble-packing adds operations per weight.

### Q4.3 Verdict: **(b) real but not free — and its value is on the footprint axis, not the speed axis**

- **Streamed bytes: genuinely halved.** 0.5 → 0.25 B/weight. CONFIRM.
- **Time: approximately unchanged, plausibly ~10 % worse** on the measured evidence that the LUT path
  is compute-bound. **REFUTE the word "free."**
- **Controller-1's derived conclusion is right on footprint and wrong on speed.** On footprint,
  0.5 → 0.25 is 2× and 0.5 → 0.2 (trit-pack) is 2.5×, so trit-pack does add only **1.25×** over
  nibble-packing — that arithmetic CONFIRMS. But §8.G's framing of trit-pack as "a 2.4–2.5× reduction
  in **streamed bytes**", read as a throughput claim, is what is actually wrong, and Controller-1
  inherits that framing rather than challenging it.
- **Where nibble-packing genuinely pays: the SKU table.** It moves S2's second column and Q1's
  condition (1): at 0.25 B/weight a 64B-param donor fits under 16 GB and a 128B donor under 32 GB. For
  Q1's 80B/A3B case it takes resident weights from ~41 GB to ~21 GB. That is real, it is large, and it
  is the reason to build it — **not** tok/s.
- **The actual speed lever is elsewhere and is bigger than both packing and dispatch.**
  `acc_add_i8x32` (`engine.c:106-110`) spends **11 of 15 ops** widening int8 → int32 on *every*
  shuffle. Accumulating in int16 with periodic widening (or `vpmaddubsw`-style pairing) attacks the
  term that actually caps the kernel at 11–17 GB/s. It is bit-exactness-gated work with an existing
  harness (`--kselftest`, `engine.c:844+`). **Recommend costing that before either packing variant.**

**Measurement required before acting on any of this:** implement nibble unpack behind a flag, run
`--kselftest` for bit-exactness against `ref_t3`, then `run_expert_rate` at both packings across the
Q3.3 size sweep. If the nibble rate in *codes/s* falls by ≤10 % while bytes halve, the packing is a
footprint win at negligible speed cost — which is the outcome I predict.

---

## Q2 — 30B sparse-MoE vs 8B dense: the Principal vs Controller-1

### Q2.1 Which organ goes down which path in `engine.c`

Two paths exist and the assignment is not a matter of opinion:

- **fp32 GEMV** — `matvec` / `dotf` (`engine.c:100`): `in_proj`, `x_proj`, `dt_proj`, `out_proj`,
  `qkv`, `o`, `router_w`, `head`. Rate = the §1b proj curve; fully-streamed floor **34–40 GB/s**.
- **Ternary LUT** — `matvec_lut_full` / `_rows` / `_tileskip`: **`gate`, `up`, `down` — the whole MLP —
  and the MoE experts.** Rate = **4.345 / 11.4 / 17.0 GB/s.**

**The MLP is on the LUT path in both model families.** `load_weights` reads `gate_f/up_f/down_f` for
E1M1 and `egate/eup/eWd` for E4M1, and both are packed by the *same* `bc_tm` into the *same* kernel
(`engine.c:210-227`). **The Principal is structurally correct: a dense donor's MLP cannot be charged to
the 34–40 GB/s projection path.**

### Q2.2 Organ by organ, both donors, 10 tok/s = 100 ms

**8B dense** (Llama-3-8B shape: `D=4096, L=32, h=14336` SwiGLU, GQA 8×128, `V=128256`, untied):
MLP `3·D·h·L = 3·4096·14336·32 = 5.637B` (**70 % of parameters — the Principal's "~2/3" CONFIRMS**);
attention `(D² + 2·D·D/4 + D²)·L = 41.9M × 32 = 1.342B`; head read per token `V·D = 525M`. Σ = 8.03B ✓

**30B sparse MoE** (`D=2048, L=48, E=128, h=768, top-8`, GQA 4×128, `V=152K`): active experts
`8 · 3·2048·768 · 48 = 1.812B`; attention `10.5M × 48 = 503M`; head `311M`; router `12.6M`.

| organ | 8B dense | path | rate | **t** | 30B MoE | path | rate | **t** |
|---|---|---|---|---|---|---|---|---|
| MLP / experts | 5.637B → 2.819 GB | **LUT** | 11.4 | **247.3 ms** | 1.812B → 0.906 GB | **LUT** | 11.4 | **79.5 ms** |
| attention proj | 1.342B → 5.368 GB fp32 | fp32 GEMV | 37 | **145.1 ms** | 0.503B → 2.012 GB fp32 | fp32 GEMV | 37 | **54.4 ms** |
| head | 525M → 2.101 GB fp32 | fp32 GEMV | 37 | **56.8 ms** | 311M → 1.244 GB fp32 | fp32 GEMV | 37 | **33.6 ms** |
| router | — | — | — | — | 12.6M → 50 MB fp32 | fp32 GEMV | 37 | **1.4 ms** |
| KV @8K window, bf16 | 1.074 GB | stream | 37 | **29.0 ms** | 0.805 GB | stream | 37 | **21.8 ms** |
| **total** | | | | **478.2 ms → 2.09 tok/s** | | | | **190.7 ms → 5.24 tok/s** |

Robustness sweep. The ranking does not flip under **any** consistent rate assignment, nor under the
assignment maximally hostile to the MoE (MoE experts charged the small-expert 4.345 GB/s while the
dense MLP is given the favourable 11.4):

| LUT rate assumption | 8B dense | 30B MoE | MoE advantage |
|---|---|---|---|
| both @ 4.345 | 879.6 ms (1.14 tok/s) | 319.7 ms (3.13 tok/s) | **2.75×** |
| both @ 11.4 | 478.2 ms (2.09) | 190.7 ms (5.24) | **2.51×** |
| both @ 17.0 | 396.7 ms (2.52) | 164.5 ms (6.08) | **2.41×** |
| **dense @ 11.4, MoE @ 4.345 (hostile)** | 478.2 ms (2.09) | 319.7 ms (3.13) | **1.50×** |

### Q2.3 Verdict: **the Principal is right. §8.A stands. Controller-1 is REFUTED.**

Controller-1 compared the MoE's *expert* organ against the dense model's *projection* organ — two
different organs on two different code paths, at rates that differ by 8×. Charged correctly, the 8B
dense donor carries **3.1× more active LUT bytes** than the 30B MoE (2.819 GB vs 0.906 GB) **and 2.7×
more fp32 projection bytes** (5.368 GB vs 2.012 GB), because attention scales as `D²·L` and the dense
model's `D` is twice as large. §8.A's "activation sparsity is worth more than small total size" is not
merely preserved — it is *understated*: the MoE wins on **both** paths simultaneously, by a factor
stable across the entire rate bracket.

One caveat I owe the record, which favours the dense donor slightly and does not change the ranking:
the 2.12× dReLU sparsity multiplier is a **learned** property (`SCALEUP_ARCHITECTURE.md` §1) that does
not transfer to a donor without training, so I charged neither donor for it. Recovered by healing, it
would apply to the dense MLP (247 → 117 ms) and to the MoE experts (79.5 → 37.5 ms) roughly
proportionally: 478 → 348 ms vs 191 → 149 ms. Still **2.3×** for the MoE.

**Neither donor reaches 10 tok/s.** This comparison decides *ranking*, not *eligibility*, and §8.A only
ever claimed ranking. Both parties should register the stronger implication: at these numbers the
**1B–100B dense band is comprehensively dead**, and the only live question is which sparse donors
survive condition (2) of Q1.2.

---

## What all three of us may still be getting wrong

1. **The KV cache at 128K is absent from every budget in this program, and it is the largest single
   term.** S2 seals **128K native context for SKU-B**. Full GQA-4 attention over 48 layers at 128K in
   bf16 reads `4·128·2·48·131072·2 B = 12.9 GB per token` — **348 ms at 37 GB/s, 3.5× the entire
   10 tok/s budget, from KV alone.** §5's arithmetic table omits it; §2.2's constants omit it;
   Controller-1 omits it; I found it only by working Q1. S3's "minority of layers" concession does not
   save it — 12 of 48 layers still costs 87 ms. **The 128K contract and the 10 tok/s gate are jointly
   satisfiable only if retained attention collapses to ~2 MQA-equivalent layers and the recall tier
   carries the remainder** — which makes §3.4's recall tier, *the one component never integrated into
   the C engine*, the load-bearing dependency of the entire donor program. It is currently scheduled as
   a single 64.3 smoke probe. This should be escalated ahead of everything in Q3/Q4.
2. **Everyone is reasoning about donor performance from constants measured at `D=256, h=128, L=6,
   V=1024`** — a model ~1000× smaller than the smallest donor, with a vocabulary ~150× smaller. §8.G
   OPEN-1 asks whether the LUT layout survives donor dimensions; the honest answer is that **no rate in
   this program has been measured above 8.3M parameters.** My 11.4 GB/s is as extrapolated as the
   mandate's 17.0; I claim only that it extrapolates from an *integrated* measurement rather than a
   synthetic one. Both are placeholders for `bench_64_1b.sh` re-run at donor shapes.
3. **The engine's dimensions are compile-time `#define`s** (`engine.c:55-74`) and `load_weights` hard-
   fails on any mismatch (`engine.c:190-193`). §8.J names this. Every tok/s number any of the three of
   us has produced describes a binary that **cannot load a donor at all.** The runtime-dimension
   rewrite is a prerequisite to measuring anything, and it is unscheduled.
4. **We are all quoting quality deltas measured at 8.3M as though they were architecture constants.**
   +0.028 BPB for ternary, +0.0006 for dReLU, +0.018–0.022 for projection ternarization, σ_seed = 0.005.
   Q1's condition (3) requires breaking D9 — and the only evidence about the cost of that is a
   single-seed measurement on a 6-layer SSM at `D=256`. The literature trend says the ternary gap
   shrinks with scale; the project's own law (`feedback_verify_public_claims`) says a trend is not a
   measurement we own. **The donor program's central speed argument rests on a quality assumption
   nobody has tested.**
5. **The dense-vs-MoE identity I leaned on may be softer than I have stated.** If the 313/207 µs dense
   figures were taken with `--skip on`, the dense path streamed *fewer* than 2304 KiB and 11.4 GB/s is
   an over-estimate — which pushes the defensible bracket *down*, makes Q1's verdict harsher, and makes
   Q4's compute-bound conclusion *stronger*. I could not determine the flag state from the documents
   and I am flagging it rather than taking the convenient reading. **Re-running `--mlp lut --skip off`
   with a byte counter is a five-minute check that should precede any use of the [4.35 – 11.4]
   bracket.**
6. **The whole eligibility framework prices weights and ignores activations.** At `D=2048, L=48` with
   per-call int8 activation quantization, the per-token scratch, logits (`V=152K` fp32 = 0.6 MB),
   router probabilities, `he[HID_E]` buffers and LUT-rebuild traffic are individually small — but S2
   demands an inventory of *every* O(size) allocation on the path
   (`feedback_scale_law_every_file`), and no such inventory exists for any donor shape. I did not build
   one either. It should be a named deliverable before any donor is promoted.
