# Donor Adaptation — what has been tried, what it cost, where to look

**The goal:** run somebody else's pretrained LLM on our architecture (`engine.c`), target **~10B at
50 tok/s** (good) / **100 tok/s** (excellent).
**Last updated: 2026-09-08 (**E19 `CARVE-DOES-NOT-RANK`: the structural escape E18 left open is closed, on BOTH halves.** (`probes/E19_DOES_THE_CARVE_RANK.md`, brief `3173edc` pushed before part B ran.) **Half one, arithmetic, published in the brief BEFORE the run: FFN-only carving cannot reach 50 tok/s at any depth, including deleting the FFN outright.** On the Coder-7B `attn+head` alone are **1.367 G** active weights against a 50 tok/s budget of **0.982-1.060 G**, so a zero-FFN 7 B still runs at **35.9-38.8 tok/s** -- and EVERY carve this programme has built (D0, D0c, Probe-4) is FFN-only. Projected to the goal's size on each donor's MEASURED attn+head share, a 10 B carries a **1.93-2.51 G uncarvable floor = 1.82-2.37x the whole budget** (19.6-27.4 tok/s with its FFN at zero). **Half two, measured: no carve ranks.** Seven arms, **oracle** router (it reads the true activation mass -- a CEILING, so a null here is strong): `base` and `FULL` hold **160/160**, then **`V52` 12, `S1` 7, `A0` 5, `N0` 6, `D10` 3 -- ALL AT-FLOOR.** The cell that matters is **`V52`: keeping 52% of the FFN -- the depth this donor's OWN budget requires -- costs +0.141846 BPB, sits 3.160 BELOW the chance line, and agrees with its own donor on 12 of 160 tokens, EXACTLY the constant-`\n` floor.** Controls at **zero error**: `FULL` token-identical to `base` (BPB diff 0.000e+00) and **`G-C2` reproduces D0c's four published BPBs to 0.000e+00 -- sixteen decimals**, which is what licenses a restated hook. Unlike E18 the ordering is NOT scrambled -- **r(BPB, agreement) = -0.8562**, the expected sign over 1.896726 BPB -- and **that sharpens it: the degradation is well-behaved and its FIRST usable point is already at zero information.** **And co-activation buys BPB but NOT ranking**: A0-N0 = -0.720513 BPB, 5/160 vs 6/160 (floor noise) -- so D0/D0c's decisions rest on a gap that carries no ranking signal. **Three called directions, three misses, for the FOURTH experiment running**; prediction 3 was argued from a TRUE mechanism (a carve leaves survivors bit-exact) and was wrong by 68 tokens -- **un tipo DIVERSO di danno non e un tipo MINORE di danno.** **Nessun timing preso, 6.79 tok/s resta ESATTO.** **Il fallimento non e una proprieta della ternarizzazione ma della MODIFICA POST-HOC di un denso preallenato in quanto tale.** Resta solo: allenare DENTRO il formato; sul ramo donor, solo lo HEALING e non provato.)**

This is the map. Every row names the artefact that holds the detail; nothing here is a claim that
is not written up somewhere with its controls and its pre-registration.

---

## 0. Where the goal actually stands

| | status |
|---|---|
| **A pretrained donor executes on our runtime** | ✅ **YES** — Qwen2.5-0.5B, parity vs PyTorch `rel l2 2.8e-06`, top-1 `1.0000`; and since E1 the engine **scores the same BPB as PyTorch to `1.5e-05`** on both donors, so the quality numbers below are statements about the deliverable, not about a simulation |
| **at a size the goal is about** | ⚠ **7.07 B of REAL trained weights, since E7** (`probes/E7_REAL_LARGE_DONOR.md`) — 67% of the 10 B target, exported, loaded from a **30.46 GB** artifact (15× the largest this runtime had ever taken), parity `1.07e-05` / top-1 `1.0000`, and **160/160 greedy tokens identical to PyTorch**. Nothing bigger exists offline: `Qwen3-8B` and the A3B files are **1 MB config stubs** |
| **and generates its own text** | ✅ **YES, since E6** — `--generate`, greedy, real fp32 weights: **160/160 tokens identical to PyTorch's greedy trajectory** over five frozen prompts (`probes/E6_GENERATION.md`). Everything before E6 was **teacher-forced** — the next token always came from the corpus, and `--bench` feeds the engine a counter on purpose. **Not true of the ternary build**: it emits tokens, not language |
| **At the target speed** | ❌ **E19 removes the structural escape: FFN-only carving cannot reach 50 tok/s at ANY depth.** On the Coder-7B `attn+head` alone are **1.367 G** active weights against a 50 tok/s budget of **0.982-1.060 G**, so **a 7 B with its FFN deleted outright still runs at 35.9-38.8 tok/s** -- and every carve this programme has built (D0, D0c, Probe-4) is FFN-only. Projected on each donor's MEASURED attn+head share, a 10 B carries a **1.93-2.51 G uncarvable floor = 1.82-2.37x the entire budget** (19.6-27.4 tok/s with its FFN at zero). **A carve that reached the target would have to cut ATTENTION and the OUTPUT HEAD too** -- never attempted, and prima facie hostile to E17's finding that the head is where ranking lives. No timing taken. Previously: ❌ **E18 gives the shortfall a BUDGET instead of a ratio: 50 tok/s permits 0.98-1.06 G active ternary weights per token = **9.8-10.6% of a 10 B model** (100 tok/s: 4.9-5.3%), derived over measured quantities only -- bytes/weight RECOMPUTED from the artifact at exactly 0.500000 -- and the 7 B donor activates **7.07 G**. On the organ ladder the **FASTEST rung is the one that converts EVERYTHING, at 7.21 tok/s (weight path), 6.9x short**, while every rung that keeps an organ exact is slower still (an fp32 weight costs **5.87x** a ternary one here). **Quality improves down that table and speed improves up it: no point on the conversion ladder is both.** So the target is not reachable by converting a dense donor at ANY conversion quality -- it needs a model TRAINED INTO the format that activates ~10% of itself per token. No timing was taken and no speed number moves.** Previously: ❌ **6.79 tok/s exact on the real 7.072 B — 7.4× short of 50 — AND (E15) that rate belongs to an artifact that DOES NOT PREDICT: `qwen25-coder7b_p.bin` scores 5.299200 BPB against a chance line of 4.070106, i.e. **1.229 BPB worse than guessing uniformly**, while the same weights in fp32 score 0.674027. The speed figure itself is untouched — a ternary weight costs the same bandwidth whatever scale multiplies it — but **it is not a rate for a working 7 B**, and closing that is now a QUALITY problem (E15 owed item B2: `--rule R3` at 7 B, never built), not an engine one.** Previously: **6.79 tok/s exact — with a 1.358× lever available on a LOSSY path (E13), which would make it 9.21 and 5.4× short if its quality cost proves acceptable.** **That cost is now MEASURED (E14) and the lever is still not released.** At 1.5 B, int8 activations move BPB by `-0.016961` (whole-vector) — which §4's one-sided band reads as `ACTIVATION-CHEAP` — while changing the **top-1 token on 54% of positions** (greedy 45.6% vs the fp32-activation arm). And the composition is **crossed**: E13's 1.358× was measured on `--lutblk` alone = the whole-vector arm, the one with the larger quality move, while the quality-cheap group-32 arm has **no speed number at all**. So the exact figure is the one quoted. **E13 also closes the engine side**: the weight path now runs at **97% of this machine's demonstrated streaming rate** (33.76 vs 34.75 GB/s in the same sweep), so the 1.46× E10 had left is taken and **there is no third kernel to write**. Three gated changes on 2026-09-07 worth **1.52× exact / 2.065× lossy**: **E8** (the weight `matvec` ran at FMA latency, not bandwidth — 1.349×) and **E9** (the SwiGLU glue ran on one thread — 1.056×, bit-exact). Weight path **16.7 → 24.0 GB/s**, **48.0 G-weights/s**. Ledger §19.3's engine budget was 1.7–2.5×, 1.52× is taken, and **E10 measured the weight path's share of the remainder at ~1.03× — that kernel is on its own ceiling at every footprint**; the rest of the gap is not available to engine work at this shape. Previously: ❌ **and now measured on TRAINED weights too: E7 reads 4.460 tok/s @300 on a real 7.07 B — 11.2× short of 50.** The synthetic proxy was checked in the same breath and held (32.8 vs **31.5** G-weights/s, −4.0%, inside ±5%), so the figure below is not an artefact of untrained weights. Previously: **measured at the target shape, not extrapolated: ~3.1–3.2 tok/s** (`T10`, 10.6 B active) — **~16× short of 50**, and it is the **weights** that are short. E3 found the *non-weight* term `f` broken (9.0× its reservation at 300 context, 22.3× at 800) and concluded a 10 B could not pass **38.3 tok/s at 800 context even with a free weight path**. **E4 overturned that**: the `Q·K` reduction was latency-bound, and 40 lines of AVX2 took `f` from 24.678 → **12.735 ms**, the ceiling `1000/f` from 40.5 → **78.5 tok/s**, and the 50 tok/s active-weight budget at 800 context from **0 → 259 M** — at **ΔBPB 3e-06**. **50 tok/s at 800 context is a weight-side problem again** |
| **At usable quality** | ❌ **NO**, but the number keeps moving: FFN conversion **+3.309 → +1.260 BPB** (T2), still 252 σ_seed. **E12 adds the reference that was missing: chance is `log2(151936)/4.229452` = 4.069819 BPB, and T2's `+3.309` — the rule `qwen_export.py` ships by DEFAULT — lands a model +0.007 ABOVE it, i.e. at chance; `+1.260` (R5) lands 2.042 BELOW it. So the 62% T2 removed is the difference between a model at chance and a model that predicts, and `--rule R3` has never been run across scale.** The **whole runnable model** cost **+2.708111** (T2b) and is now **+2.465779** — E2 confirmed the RMSNorm fold *through the engine* at T3's exact `−0.220001` and it is **adopted as the exporter default** |
| **The binding constraint** | **still quality — but it is the RULE, not the format** (T2, `RULE-HELPS`) |

> ⚠ **Until E1 (`33f0add`) the runtime could not load a model over 2 GB at all** — 32-bit
> `ftell`, silently reported as `bad magic` against an intact file. The largest artifact it had
> ever been given was 1.84 GB. **A 10B ternary packed model is ~5 GB**, so the target was not
> slow, it was unloadable, and no speed probe could have found it.

**The one-line state:** the road exists end to end — safetensors → export → runtime →
**generated tokens** — and since **E6** that last clause is finally backed rather than asserted:
the fp32 build reproduces PyTorch's greedy continuation **token for token**. The **ternary** build
generates too, and what it generates is not language — which is what T2b's **+2.466 BPB** looks
like when you read it instead of quoting it. T2 has now shown the damage at the far end was
**62% a bad map into the format**, not the format itself, and removed that much of it on CPU with
no gradients. What remains is a real quality gap and a 20× speed gap.

**The number that prices everything else, re-derived by E3 at the shape it is about:** a 10B donor
needs **≤318 M active weights/token — 3.0% of a 10 B — for 50 tok/s at 300 tokens of context**, and
**at 800 tokens of context there is no budget at all**, because `f` alone exceeds the whole 20 ms.
100 tok/s is **out of reach at any weight cost** at a 10 B shape: `f` = 10.576 ms > 10.

The old figure, **522 M**, is superseded twice over. It was also **arithmetically wrong in its own
terms**: `27.7 G-w/s` is `493,961,216 / 17.833 ms` = weights / **wall**, so `f` is already inside the
denominator, and `× (20 − 1.17)` charges it a second time; self-consistently it is 554 M.

**The head is no longer the floor.** It is 20.5% of the token at 0.5 B and **1.1% at a 10 B shape**;
the attention projections are 17.9%, sixteen times more. The tokenizer claim itself survives and is
now end-to-end rather than a weight count — `M7` and `Q8`, same width, **4.637× the vocabulary and
4.629× the head time, 0.2%** (§7).

## 1. The runtime (this is the deliverable)

| what | where |
|---|---|
| Export a donor to a flat binary (fp32 / ternary / packed; optional ternary head) | `benchmarks/donor_adaptation/engine/qwen_export.py` |
| The runtime: RMSNorm, GQA + RoPE + KV cache, SwiGLU, ternary matvec, packed `pshufb` path | `benchmarks/donor_adaptation/engine/donor_engine.c` |
| Parity gate vs PyTorch on identical weights | `benchmarks/donor_adaptation/engine/parity_gate.py` |
| Report: profile, the two optimisations, honest position vs the goal | `probes/R1_DONOR_RUNTIME.md` |

**Measured, Qwen2.5-0.5B, 3600X, `--bench 300`, idle, median of 3:** 12.45 tok/s (t1) →
**56.1 tok/s (t6, packed)**. Trajectory: 23.5 (fp32 head) → 38.0 (ternary head) → 48.3 (packed)
→ 50.9 (same binary, re-measured 2026-09-05) → **56.1 (rope hoisted out of the head/layer loops,
bit-identical, `SPEED_LEDGER.md` §12)**. `--fuse --lut --lut-group 32` adds ~3.9% on top, but the
LUT half of that is not numerically free.

> **Two rates have been withdrawn here for two different reasons, and both were load-bearing.**
> R1's "40–46" and §10's 36.1 were taken on a **contended machine**. §11.4's per-organ table was
> taken with **the wrong work inside the timer** — both `rope()` calls sat in the `qkv` bucket, so
> it reported qkv at 4.1 GB/s (real: 14.2) and produced a "32 µs per call" anomaly that does not
> exist. **A tok/s figure is also only comparable at the same `--bench` length**: attention is
> `O(position)`, worth 1.7 ms/token between 300 and 800.

## 2. The speed side — what is priced and what is measured

| probe | question | answer | where |
|---|---|---|---|
| **Ledger** | what must a donor cost per token to hit 50/100 tok/s? | **≤522 M active weights/token** for 50 tok/s, ≤245 M for 100 — from the *measured* **27.7 G-weights/s**, not a byte-rate ÷ bits-per-weight | `SPEED_LEDGER.md` §12.2 |
| **P2** | is the expert path bandwidth- or compute-bound? | **MIXED — ~60% arithmetic.** A denser pack buys ≤1.32× on the FFN, not 2.5× | `probes/P2_EXPERT_PATH_DECOMPOSITION.md` |
| **P3** | what do donor SHAPES cost on the engine's kernels? | 0.5B 81 / 1.5B 30.7 / 3B 15.1 tok/s (matvec only). The ledger was 18% conservative because it priced dense FFNs at a *gather* rate | `probes/P3_DONOR_SHAPE_ON_ENGINE.md` |
| **R1** | what does a real runtime cost? | **56.1 tok/s** at 0.5B (§12). The packing bought nothing, exactly as P2 predicted; the LUT buys 2.2% and is not free; **the one big win was a profiling artefact — `rope()` was 9.6% of every token** | `probes/R1_DONOR_RUNTIME.md`, `SPEED_LEDGER.md` §12 |
| **E3** | what does the engine actually do at the target shape, end to end? | **`RESERVATION-BREAKS`.** 6 shapes 0.5–10.6 B x 2 context lengths, synthetic weights gated against real artifacts at 0.000% (1.5 B) and 0.357% (0.5 B). `T10` = **3.090 tok/s**; rate **rises** to 32.8 G-w/s; **`f` is 9.0× the reservation at 300 context and 22.3× at 800** | `probes/E3_ENGINE_AT_TARGET_SCALE.md` |
| **E4** | is the attention loop latency-bound, as E3 read it, or bandwidth-bound? | **`LATENCY-CONFIRMED`.** The `Q·K` dot loop was **6.5× below its own memory limit**: 14.647 → **2.242 ms**, **5.4 → 35.1 GB/s of unique K bytes**. ILP alone 2.31× (same bytes, less time — bandwidth falsified on its own), SIMD alone 4.28×, both **6.53×**. **`f` 24.678 → 12.735 ms, ceiling 40.5 → 78.5 tok/s**, ΔBPB **3.03e-06**. The floor is now `R` = softmax + `A·V`, **81.4%** of the organ | `probes/E4_ATTENTION_ACCUMULATORS.md` |
| **E5** | what is `R`, the 81.4% of the attention organ E4 could not see inside? | **`OVERHEAD-DOMINATED`.** `R` = softmax `S` **25.3%** + `A·V` `Y` **20.6%** + `P` **54.0%** at `T10` @800 — **more than half of `R` is neither loop**. Nine 3× predictions passed, worst −1.15%; `X` = 2.253 ms reproduces E4's 2.242 by a method sharing no arm. The split **inverts with scale**: `S05` is SOFTMAX-dominated at 47.7% | `probes/E5_DECOMPOSE_R.md` |
| **E6** | does a real donor actually generate text here, or only score it? | **`GENERATION-CONFIRMED`.** New `--generate` mode, greedy. fp32 Qwen2.5-0.5B: **160/160 tokens identical to PyTorch**, five prompts, no divergence. Planted control fires — ternary 0.5 B **1.9%**, ternary 1.5 B **6.2%**, neither writes language. G-P: the prefill logits are **byte-identical** to `--logits`, so this is E1's gated forward pass. **And `1000/f` is a ceiling, not a rate: measured is 3.09–3.14 tok/s at `T10`** | `probes/E6_GENERATION.md` |
| **E7** | does any of this hold on a real donor at a size the goal is about? | **`REAL-WEIGHTS-CONFIRMED`.** Qwen2.5-Coder-7B, **7.07 B active/token**: 30.46 GB consumed exactly, parity **1.070e-05** / top-1 **1.0000**, **160/160** greedy tokens identical to PyTorch. Speed **4.460 tok/s** @300 (3 reps, 0.45% spread) against a **pre-registered 4.0–5.4** — **11.2× short of 50**. And the synthetic proxy checked at last: 32.8 vs **31.5 G-weights/s**, −4.0%, inside ±5% | `probes/E7_REAL_LARGE_DONOR.md` |
| **E7 §8–§9** (extension, pre-registered `8252031` / `9103c26`) | is there a fixed cost inside the timed region, how does `f` grow past 800, and which kernel is even running? | **Fixed cost REFUTED** (1600 median **4.230** against a 4.580 threshold fixed first; the 300→800 inversion was **2.7%**, inside ±5% — noise given a mechanism it did not have). **`f` is LINEAR to 1600**, 0.0319 ms per token of context, two intervals agreeing to 0.2% over a 5.3× range — **E4's owed item 3 CLOSED**. And `donor_engine.c:116` defaults `g_attn` to **`ATTN_SERIAL`**, E4's second-slowest arm: every E7 and E3 figure is a `serial` reading. **No ledger number needs correcting** (E4 published `avx4` vs `serial` at `T10` as **+3.9%**, inside ±5%) — but attention is 2.2% of the token, so this is a correctness note, not a speed result | `probes/E7_REAL_LARGE_DONOR.md` §8–§9 |
| **E7 §10** (pre-registered `9103c26` / `db53f63`) | does E4's `avx4` win show up as a RATE on a real donor? | **Not answerable with the un-profiled instrument.** @300 the **null was called in advance** and landed (4.450 vs 4.460); @1600 the baseline's own spread (6.1%) exceeded the effect (5.7%); the interleaved re-run is **VOID** (paired median **0.973** against a pre-registered 1.00–1.09, within-arm dispersion **11.5%**). **The win does reach the wall** — profiled organs sum to the wall to **0.03%**, the non-attention remainder is arm-invariant, and the arithmetic is consistent to **0.4% of the token** — worth ~6–7% at 1600, **published as no rate at all**. **New general defect: `--bench` has no contention witness** | `probes/E7_REAL_LARGE_DONOR.md` §10 |
| **E7 §11** (pre-registered `55bd8c6`, re-registered `02baefa`) | can `--bench` tell whether the machine was quiet? | **`WITNESS-CONFIRMED`.** `ffn~` on the `BENCH` line, layer 0, on by default. **Failed its first gate** — 0.9877 against 0.990, a 1.2% cost against a 0.008% prediction — rebuilt from 48 timestamps/token to 2, then **0.9996** (predicted 0.995–1.000), `--logits` **byte-identical**, and the **planted control fires** at **+34.3%** under load. A filtered median that would have acquitted the failed build is recorded and **refused as circular**. Validates nothing retroactively | `probes/E7_REAL_LARGE_DONOR.md` §11 |
| **E8** (pre-registered `406b02b`, pushed before the run) | is the weight path bandwidth-bound, or a dependency chain? | **`CHAIN-CONFIRMED`.** One accumulator per `matvec` → the loop ran at FMA **latency**. Four accumulators: **Coder-7B 4.73 → 6.37 tok/s, 1.3491 paired over 10 pairs (1.8% spread)**; 16.7 → 22.5 GB/s; qkv/o/ffn/head all move 1.41–1.49× together. Parity 3.3e-06 and 160/160 greedy. **Negative control on the fp32 kernel predicted 0.99–1.03, measured 1.0029**, and showed 33.3 GB/s through the same kernel. Predicted 1.5–1.7×, got 1.349 — the next binder is the kernel again. **Now the default**; `--mvacc 1` reproduces every earlier rate byte for byte. FFN decomposed: `gate+up`/`down` = 1.990 against a structural 2.000 while chopped 10.6× differently, so per-row cost was never it | `probes/E8_MATVEC_DEPENDENCY_CHAIN.md` |
| **E9** (pre-registered `82dbb54`, pushed before the run) | the glue between the FFN matvecs moves no weights — what does it cost, and is it threaded? | **`GLUE-CONFIRMED`.** It was **not** threaded: 530,432 scalar `expf` per token, **11.346 ms, 7.5% of the token, zero weight traffic**. An elementwise map with no reduction → **bit-exact**, gated on **sha256** rather than parity (`b94b56d0…548765`, identical). **6.45 → 6.79 tok/s, paired median 1.0557 (10 pairs, 3.8% spread), inside the 1.055–1.066 predicted before the run**; glue 11.346 → **2.938 ms**, `sum/ffn` 0.9999. `rmsnorm` deliberately left serial — 0.330 ms over 56 calls cannot pay 2.7 µs per region. The `expf` itself is untouched and is the next item: it would beat this, but it forfeits the sha256 gate | `probes/E9_GLUE_PARALLEL.md` |
| **E10** (pre-registered `a0366d0`, pushed before the bench existed) | is the packed kernel core-bound or stream-bound at 22.5 GB/s? — E8 §9 item 2 | **`CORE-BOUND`.** Same `matvec` source, footprint swept 4 MB → 2 GB with `n_in` fixed: **49–53 G-w/s at every cell, all seven within ±4% of the arm median**, while fp32 moves **4.4×** over the same range. **Planted control read first and PASSED (4.40 vs ≥2.0)**, reproducing probe-3's 16 MB cliff unprompted; G-K2 tied the bench to the engine's 26.1 GB/s within 2.3%. **The engine's FFN organs are AT the ceiling** (51.66 / 52.22 G-w/s). **Withdraws §19.4's 37 and 42 GB/s rows for the packed path** — the assumption §19.5 flagged as weakest — and cuts the weight path's remaining engine budget to **~1.03×**. Not FMA-bound either (0.29/cycle/core of 2): the side is named, **the mechanism is not, deliberately**. G-K3 was aimed at the wrong statistic and it is recorded | `probes/E10_PACKED_KERNEL_BINDER.md` |
| **E11** (pre-registered `00d4446`, pushed before the arm was written) | does the `--lut` kernel have a higher per-weight ceiling than packed? — E10 §6 item 1 | **`NO-LIFT`.** At the 512 MB verdict cell named in advance: **21.90 vs 55.95 G-w/s, ratio 0.391** — the LUT path is **2.5× slower per weight** at donor footprints. **Corrects §12's «1.05×» and §2.1's «3.9%»**, both taken against a pre-E8 single-accumulator packed arm. **But inside L3 it is the fastest kernel measured here — 91 G-w/s, 1.8× packed — and falls 4.2× across 16 MB**, because tile-major puts consecutive reads 299 KB apart at scale: the property that makes it fast is the property that kills the stream (exploratory, outside the bands). Bears on `SCALEUP_ARCHITECTURE`'s ≤16 MB keystone. Run 1 VOID by a concurrent worker; G-L2 then failed by my own mis-specification | `probes/E11_LUT_KERNEL_CEILING.md` |
| **E13** (pre-registered `89fd63e`, pushed before the arm existed) | is the LUT collapse a LAYOUT artifact? — E11 §3's untested mechanism, and the 1.46× E10 left unread in its own table | **`LAYOUT-CONFIRMED`.** `build_tm` stored `tm[t*Mpad + o]`, so consecutive 32-byte reads sat **18.5 KB** apart at `gate|up` and **3.5 KB** at `down` — a different page nearly every read, 32 bytes of every 64-byte line, for the **same total bytes**. `--lutblk` stores each 32-row tile contiguously: same bytes permuted, `t` order and accumulate tree untouched → **bit-identical, gated on sha256** (`7a9b3e04…f48e44`, 311 MB, both arms). **G-M1 = 67.51 G-w/s at E11's pre-named 512 MB cell against ≥65; predicted 65–77 and landed inside.** Blocking recovers **3.0–3.5×** at the largest cells, the 4.2× cliff becomes **1.53×**, and **103.05 G-w/s at 12 MB is the fastest weight kernel measured here**. At 512 MB it moves 33.76 GB/s vs the fp32 arm's 34.75 = **97%** → **the 1.46× is taken; no third kernel to write.** End-to-end **1.358×** (9.21 vs 6.78 tok/s) — but on the **LOSSY** int8-activation path whose 1.40e-01 cost was never carried to BPB, so **6.79 exact remains the quoted rate**. Confirms E11's `NO-LIFT` end-to-end (`--lut` = 0.557×); overturns only its untested §3, which holds. Known-positive reproduced to 0.15%; a 3-rep pass with 180% spread discarded as warm-up | `probes/E13_BLOCKED_TILE_MAJOR.md` |
| **E12** (pre-registered `11a9d83`, runner `87ba957`, both pushed before the sweep; §9 amendment + Z-dispersion `ae403c2`) | does the ternarization cost SHRINK with donor scale? T2's rule is measured at ONE shape and it is the assumption under the whole MoE road | **`CHANCE-LINE` — the question could not be ASKED at these settings. Supersedes an earlier `CONTROL-FAILED`, WITHDRAWN.** dBPB 3.715658 / 4.738239 / 5.010249 at 0.5/1.5/3 B, each against its OWN fp32 baseline. **`log2(151936)/4.229452` = 4.069819 BPB is chance, and EVERY ternarized arm at EVERY cell is above it** (+0.518/+1.436/+1.665) while every fp32 baseline is far below. Above that line BPB measures how confidently WRONG a model is — so `r = 1.3484` is not reported, and the deterministic `F` > `FA` inversion at 3 B needs no instrument bug. **`I - base = +0.000e+00` exactly at all three cells**; the instrument was sound. The withdrawn verdict rested on arm `Z`, **which T2 §4(a) had already retired and t1's own docstring and `ARMS` table label mis-specified**; measured dispersion **spread(3B) = 1.695938 -> `Z-UNSTABLE`, predicted and landed**, larger than the 1.490 gap it was read off, and the outcome FLIPS with the seed at 0.5 B (2/5). **Owed first: FINISH the R3 sweep at 3 B** — `R0` is only the flag DEFAULT; every standing engine artifact ships `--rule R3`, which E1 measures ABOVE chance at 0.5 B and BELOW it at 1.5 B, so the shipped rule crosses the line at an unlocated shape | `probes/E12_TERNARY_COST_VS_SCALE.md` |
| **E14** (pre-registered `537b496`, pushed before the run; G-N3 runner `ba0b15c`, pushed before it ran) | what does the int8 activation quantization that gatekeeps E13's 1.358x lever COST in quality? | **`CHEAP-BUT-NOT-NEUTRAL`.** At the 1.5 B verdict cell, **0.625503 BELOW the chance line** where BPB means what it usually means, int8 activations **LOWER** BPB: `dBPB(A1) = -0.016961`, `dBPB(A2) = -0.005991`. Section 4's bands are **one-sided**, drawn for a cost, so a negative delta reads `ACTIVATION-CHEAP` **by construction**. **G-N3 says it is not free: the top-1 token changes on 54% of positions** (greedy agreement 45.6% whole-vector, diverging at token 0; 64.4% group-32, token 1). **The label is true of the band and false of the model, and no registered gate could tell the difference** -- section 6 forbade promoting G-N3 after the fact, so the gap IS the result. **The lever is not released, and the composition is CROSSED**: E13's 1.358x was measured on `--lutblk` = arm A3 = A1 (G-N0 `+0.000e+00`, and token-for-token identical under G-N3), the arm with the LARGER quality move, while the quality-cheap A2 has **no speed number at all**. E13's bit-identity now holds **three independent ways**. Protocol defect recorded: no `--seqlen` was passed, so every arm scored as ONE 12,288-token sequence; arm-to-arm gates unaffected, a 512 re-run is owed | `probes/E14_ACTIVATION_INT8_COST.md` |
| **E19** (part A + brief `3173edc`, pushed before part B ran; runner `bdd87a4`) | does a CARVE rank -- the one structural lever that can reach E18's ~10% activation budget -- and how deep would it have to cut? D0/D0c scored every carve arm in BPB ONLY, the same omission E18 caught in T2b | **`CARVE-DOES-NOT-RANK`.** **Part A, before the run: FFN-only carving cannot reach 50 tok/s at ANY depth** -- on the Coder-7B `attn+head` alone are **1.367 G** vs a budget of **0.982-1.060 G**, so a **zero-FFN** 7 B still sits at **35.9-38.8 tok/s**; a 10 B at the same measured share has a **1.82-2.37x** uncarvable floor. **Part B: `base`/`FULL` 160/160, then `V52` 12, `S1` 7, `A0` 5, `N0` 6, `D10` 3 -- every carved arm AT-FLOOR**, under an **oracle** router (a ceiling). **`V52` keeps 52% of the FFN for +0.141846 BPB, 3.160 BELOW chance, and scores EXACTLY the constant-`\n` floor of 12/160.** Controls at zero error: `FULL` token-identical, and **G-C2 reproduces D0c's four BPBs to 0.000e+00**. **r(BPB, agreement) = -0.8562** -- right sign, and all five points still at/below the floor. **Co-activation's 0.720513 BPB advantage buys NO ranking** (5 vs 6). Three called directions, **three misses**. **No timing; 6.79 tok/s exact.** |
| **E18** (part B pre-registered `2ac74f5`, pushed before any arm ran; part C `b02a942`) | what does the ladder from «nothing converted» to «everything converted» look like, and is any rung of it fast enough? -- E17 section 8's owed floor, plus the first ranking read of the five organ arms T2b had only ever scored in BPB | **`CLIFF-NOT-SLOPE`.** The floor is **not zero**: the best CONSTANT predictor scores **11/160** (0.5 B) and **12/160** (1.5 B), both `'\n'` -- **every ternary arm ever built here is at or below it**, and E17's best was exactly `12/160`. On the ladder, `base` and the identity arm hold **160/160**, then **`H` 9, `A` 4, `F` 5, `FA` 12, `FAH` 10, all AT-FLOOR**: **ternarizing ANY single organ destroys ranking**, including the head alone at **+0.338989 BPB**, `2.963` BELOW the chance line, `9/160`. **`G-L2` reproduced the engine EXACTLY (12 and 10) before any empty rung was read.** `r(BPB, agreement) = **+0.4989**` across a `2.377667` BPB span -- the wrong sign. **Speed axis, derived (no timing, `6.79 tok/s` untouched): 50 tok/s = 0.98-1.06 G active ternary weights = 9.8-10.6% of a 10 B model, and the fastest rung -- the all-ternary one -- is 6.9x short.** Three called directions, **three misses**; the brief's registered ALTERNATIVE carried the verdict. **Scope: no healing/QAT -- the one untested branch.** |
| **E17** (pre-registered `cf33251`, pushed before any arm ran) | is the ternary HEAD what fails to rank? -- E16's arms were all `--head-ternary`, and on the untied Coder-7B that is a standalone 545 M-parameter output projection | **`HEAD-IS-NOT-THE-MECHANISM`.** Removing it buys **2 tokens of 160** at 1.5 B and **0** at 0.5 B, both still diverging at token 0 -- far below the `20/160` bar derived in advance, so the registered stage-2 export **does not run**. Nothing exported; every artifact already on disk. All four controls fired first, including a NEW third known-positive (1.5 B fp32 `160/160`, which also closes E1 s4.4 in the generate path) and `G-H0d`, which proves E6's and E16's binaries are token-identical and is the only reason the population table is licensed. **The head is not inert but IRRELEVANT** -- it rewrites 49% / 82.5% of the output while changing correct tokens by 2 / 0. **Population: 3 known-positives at exactly `160/160` across three scales and two families; 7 ternary arms across three scales, two rules, two folds, both head settings -- best `12/160`, all diverging at token 0. BPB spans `1.823493` across them while ranking spans 0-12/160**: inside the ternary regime BPB has resolution and ranking has none. **New debt: no floor for agreement by frequency coincidence, so `12/160` is not yet known to beat zero information.** No speed number moves | `probes/E17_DOES_THE_HEAD_RANK.md` |
| **E16** (pre-registered `be65920`, runner `2247923`, control `34e7ba3`, all pushed before the arms ran) | is the RULE what E15 found broken, and is there a ternary 7 B that predicts? -- E7 section 12 owed item 3 | **`SCORE-CROSSES-RANK-DOES-NOT`.** Three arms, one axis each. **`R0` -> `R3` with the fold and all else FIXED is worth `G-R2 = -1.281967`**, carrying the 7 B from +1.229094 above the chance line to **4.017233, -0.052874 BELOW** = 59x the band, so G-R1 reads **`RULE-FIXES-IT`**. **But every ternary arm agrees with the fp32 donor on 0 of 160 greedy tokens, diverging at token 0** -- B2 is 0.155 nats/token better than uniform and cannot rank the donor's next token once. **The label is true of the band and false of the model, and the brief registered ONLY scoring metrics** -- caught by E14's day-old law (pair scoring with ranking); the ranking run reports and decides nothing (E14 s6), and validates itself by reproducing E7's `0/160`. **G-R0 fires at 3.20e-07** against E1. **R3 is NON-MONOTONE in scale** (+0.461 / -0.594 / **+0.098** at 0.5/1.5/7 B): its damage grows with scale as `R0`'s does, one octave later. **The fold is load-bearing** (`G-R4 = -0.151093`) and is the only thing E15's artifact had right; its credit shrinks -0.529 / -0.220 / -0.151. Design gate passed before any BPB was read. **No speed number moves.** Predictions: G-R0 inside, G-R1 direction right and band missed by 0.018, **G-R3 direction WRONG** -- a trend extrapolated from two points, which the brief admitted could not establish curvature and leaned on anyway | `probes/E16_R3_AT_7B.md` |
| **E15** (pre-registered `7c4243f`, amended `c92ffd4`, both pushed before any arm ran) | does the packed Coder-7B that EVERY donor tok/s is quoted on actually predict? -- E7 section 5's "separate experiment nobody has run" | **`DOES-NOT-PREDICT`.** Full 24x512 heldout slice, `--seqlen 512`, chance **4.070106**, band **0.000896**. **`qwen25-coder7b_p.bin` reads 5.299200075 = +1.229094 ABOVE chance -- 1,372x the band, worse than guessing uniformly over the vocabulary.** **The planted control FIRES and fires hard**: the fp32 arm reads **0.674026555, -3.396080 below** the line, so **the engine is sound, the donor is sound, and it is the CONVERSION that is broken.** The 7 B is in fact **the best model this programme has ever run** and the fp32 sweep is monotone in scale on the identical slice (0.871795 / 0.767595 / 0.724450 / **0.674027**) -- a code-specialised donor winning on a prose-majority corpus, and the first evidence that bigger donors keep paying. **E7's two headline halves are about DIFFERENT artifacts**: the fp32 arm is the `160/160`, the packed arm is E7's planted control at `0/160` diverging at token 0. **No speed number moves** -- a ternary weight costs the same bandwidth whatever scale multiplies it -- **6.79 tok/s exact stands; what E15 removes is the right to call it a rate for a WORKING 7 B.** Predictions scored: the control's 0.55-0.85 **contained** 0.674027, while the amended 3.4-5.2 **missed** 5.2992 that the withdrawn 4.6-6.5 would have caught -- the amendment composed measured deltas across THREE configurations, which the brief had itself named as the move E12 forbids. **Owed and load-bearing: B2 -- re-export with `--rule R3` keeping `--fold layers`, so the rule is the only variable.** R3 reads 3.475707 at 1.5 B, **0.594 BELOW** the line | `probes/E15_DOES_THE_7B_PREDICT.md` |
| **P64 matrix** | is the runtime's rate set by OpenMP region count or thread wake-ups? | **neither.** A region costs **2.5–3.2 µs**, measured two ways; `OMP_WAIT_POLICY=active` does nothing. After the rope hoist all four organs sit in a **1.3× band** | `SPEED_LEDGER.md` §12.4, `engine/bench_matrix.py` |

**Terms nobody had attacked before the ledger, now priced:** the **output head** (was 40.4% of every
token in fp32 — fixed; and E3 shows it is **not** the floor at 10 B, §7), the **KV cache** (still
fp32, untouched — and E3 says this is now the binding term), the **attention projections** (17.9% of
a 10 B token, `qkv` + `o_proj`).

> **E4 answered the question E3 left open, and moved the section on again.** The attention loop was
> latency-bound and is not any more: with `R` (softmax + `A·V`) measured and subtracted, the `Q·K`
> loop went **6.53×** and now runs at **35.1 GB/s of unique K bytes** — this machine's DRAM. Two
> consequences. First, **the int8 KV cache named below is retired at ~1.16×** before being built:
> it cuts the dot loop's bytes 4× but the dot loop is only 18.6% of the organ now. Second, **the
> binding term is `R`**, 81.4% of the organ and untouched by anything measured so far. The GQA
> re-reads do **not** reach DRAM — 140 GB/s of touched bytes is not achievable from it — which E3
> explicitly declined to claim and E4 gets for free from the arithmetic.

> **E3 changed which term this section is about.** The weight path is *faster* than the ledger says
> at every shape above 0.5 B. What does not scale is `f`. E3 §4.6 measures the attention loop at
> **one FMA per ~4 cycles per thread across twelve points, ±7%**, invariant to KV size (16× range),
> GQA factor and vocabulary — the signature of a serial FP reduction, not a bandwidth wall:
> `d += qh[i]*kt[i]` compiled without `-ffast-math` cannot be reassociated or vectorised, while
> every matvec is hand-written AVX2 with an 8-wide accumulator. **The single highest-value engine
> experiment now open is giving that loop independent accumulators.**

### 2.1 The LUT kernel — `donor_engine.c --lut`, the lever P2 and R1 named

| what | status |
|---|---|
| tile-major layout, `pshufb` tables, grouped scales | **built.** A `--lut` model and a `--quant packed` model hold **bit-identical weights** — the copy is a pure transpose |
| kernel correctness | **bit-exact** vs a scalar-integer reference; two planted controls fire (`--selftest-lut` cases B and C) |
| numeric cost of the int8 activations it requires | **measured.** rel l2 `1.40e-01` per-vector → **`3.10e-02` at G=32 channels per scale** |
| why it costs that | activation crest factor `amax/rms` is 8.3 avg / 69.6 max, so a 63-step grid leaves ~4–8 usable levels. `--lut-diag` measures it on `x` alone, so the kernel is not implicated |
| **rate** | ⚠ **measured: 1.022×** after the rope hoist (`SPEED_LEDGER.md` §12.4). It helps the two big-matrix organs (ffn −0.40 ms, head −0.33) and **hurts the two small ones** (qkv +0.31, o_proj +0.06); `--fuse` first removes that penalty and the pair is worth 1.039×. **Not the 1.83× the ledger assumed** |
| BPB through the runtime | ❌ not run |

Two predictions were written before their sweeps and **both were wrong in magnitude**: clipping the
grid to `k·rms` (predicted an interior optimum; it is 3× worse at every `k`) and per-group scales
(predicted <0.02 at G=128; it is 0.052). Recorded in commits `95b7fd3` and `e02285c`.

## 3. The quality side — every structural result, and the conversion that undercuts them all

| probe | what was tried | outcome | where |
|---|---|---|---|
| **D0 / D0c** | carve the FFN into co-activation experts | **+1.09 BPB at E=32; +0.706 at the finest legal E.** Granularity helps, but the random null helps *more* → the mechanism does not strengthen (PARTIAL) | `probes/D0_COACTIVATION.md`, `probes/D0C_GRANULARITY.md` |
| **D1** | magnitude / structured pruning | see report | `results/d1_pruning.json` |
| **D2 / D3** | basis rotation, low-rank | see reports | `results/d2_basis.json` |
| **D4** | solve for thin replacement weights (Hessian) | recovery 0.483 honest vs 0.859 leaked → in-sample optimism; budget never swept | `probes/D4_RECONSTRUCTION.md` |
| **S1** | which bar predicts BPB under sparsity | \|h\| is the bar; A≡D at every digit | `benchmarks/donor_adaptation/s1/` |
| **T1** | **ternarize the donor — the engine's own rule** | **+4.738 BPB = 948 σ_seed. CONVERSION-FAILS** (measurement stands; verdict superseded in scope by T2) | `probes/T1_DONOR_TERNARIZATION.md` |
| **T2** | was that the FORMAT or one naive RULE? | **`RULE-HELPS`. It was the RULE.** FFN +3.309 → **+1.260**, 62% removed with no training. **BitLinear158 is statistically indistinguishable from RANDOM SIGNS** (−0.064 ± 0.126) | `probes/T2_TERNARIZATION_RULE.md` |
| **T2b** | does the winning rule survive outside the FFN? | **`UNIFORM`, by 1.0% of its bar.** The runnable model (197 tensors, R3) costs **+2.708**, `1.584×` the FFN alone vs a 1.60 bar — ci95 `[1.514, 1.649]`, the bar is INSIDE it. Head ternarization **not** withdrawn: the head costs `+0.339` alone and **`−0.009 ± 0.020` on top of a ternary FFN+attention**. Per weight **attention is 4.98× the FFN** | `probes/T2B_ORGAN_COVERAGE.md` |
| **T3** | rotate the residual basis (QuaRot/SpinQuant) before ternarizing | **`VOID` as written, `NULL` once the brief's own gate constant is corrected.** Rotation does not fail to help, it **hurts**: `+0.634 ± 0.039` (dense orth) and `+1.138 ± 0.072` (Hadamard) vs a fold-matched control. Kurtosis is not a predictor and cannot be made one. **Keeper: the RMSNorm fold alone is `−0.220 ± 0.053` free** | `probes/T3_ROTATION.md` |
| **E1** | does the model the ENGINE executes score the BPB PyTorch says it does? | **`LOOP-CLOSED`** on the 0.5B; every term of the same rule met on the 1.5B (labels there are `INCOMPLETE` because the brief split the arms across two runs). Largest disagreement over 5 arms and 2 donors **`+1.53e-05` BPB** = `0.003 sigma_seed`, and the delta **does not grow with the arm**. Gate A bit-identical on up to **1,543,569,408 codes**. T2b's `+2.708111` replicated **bit-identically** by a second runner through a file round-trip. **Found on the way: the engine could not load a model over 2 GB** | `probes/E1_BPB_THROUGH_ENGINE.md` |
| **E2** | does T3's RMSNorm fold survive into the artifact the engine runs? | **`FOLD-CONFIRMED`, and adopted.** `NL - TQ = -0.220001`, paired SE `0.052861`, ci95 `[-0.324988, -0.119593]`, 44 sigma_seed on the 1.5B -- **T3's point AND its dispersion**, now through the exporter, in the file format, on weights Gate A proves identical. The **runnable** model goes `+2.708111` -> **`+2.465779`**. Gate F passes on the deciding donor with 4.2e+04x of margin. **`--fold all` is NOT adopted**: folding the final gain into a *ternary* head COSTS `+0.176983` (1.5B) / `+0.629949` (0.5B). **Every sign replicates across donors; no magnitude does, and one comparison inverts** | `probes/E2_RMSNORM_FOLD.md` |

**T2's decomposition, paired between arms** (`probes/T2_TERNARIZATION_RULE.md` §4):

| what | Δ BPB | significant |
|---|---|---|
| searching the per-row scale at all | −0.686 | yes |
| **weighting that search by activation RMS** | **−0.914** | yes |
| GPTQ error compensation **on a well-placed grid** | −0.449 | yes |
| GPTQ error compensation **on the naive grid** | +0.223 | **no** |
| TWN instead of BitLinear158 | −0.225 | **no** |

> ⚠ **Every structural result above was measured on fp32 weights the engine cannot consume**, and
> none was ever composed with the conversion. "Does carving a *ternarized* donor cost the same as
> carving an fp32 one?" has never been asked — and it is now a cheaper question than it was, since
> the conversion it would have to compose with costs +1.260 rather than +3.309.

## 4. Open, in priority order

0. **Decompose `P`** — new, and it inherits first place from the item **E5 just closed**.

   **E5 is CLOSED, `OVERHEAD-DOMINATED`** (`probes/E5_DECOMPOSE_R.md`, brief §14). At `T10` @800,
   `R` = softmax `S` **25.3%** + `A·V` `Y` **20.6%** + `P` **54.0%** — **more than half of `R` is
   neither loop.** Nine 3× predictions the components were not fitted to all passed, worst error
   −1.15%; `X` = 2.253 ms independently reproduces E4's 2.242 by a method sharing no arm with it.
   Taking E5's shares onto E4's own `R` = 9.816: **`S` = 2.485, `Y` = 2.020, `P` = 5.301 ms.**
   The one candidate for `P` that has been priced — the OpenMP fork, run 5's `fork2` arm — is
   **1.4% of it**. So `P` is not the fork, and what it *is* has never been measured: the
   32-heads-over-6-threads imbalance, per-head address arithmetic, `out[]` init, the `mx` reduction.
   **That is item 0.** Two further things E5 hands over: the shape *inverts with scale* (`S05` @800
   is **SOFTMAX-dominated at 47.7%**, `T10` is OVERHEAD-dominated), so a softmax fix measured small
   arrives at target scale worth half what it looked; and `S` came in **48% above** its
   pre-registered band, with `expf` compiling to a *double-precision* `exp` call plus `vzeroupper`
   as the untested candidate.

   ~~**Decompose `R`** — inherited first place from the item E4 closed.~~ **Done.**
   **Pre-registered and running as E5**: `briefs/BRIEF_E5_DECOMPOSE_R.md` (`c1bdd70`, pushed before
   the arms), arms `--attnr {sm2,sm3,av2,av3,fork2}` at `00f4538`, G2 passed **bit-identical on all
   six arms**. The brief adds a third term this row did not name — `P`, the OpenMP region and
   everything that is neither loop — because its own arithmetic for the two loops reaches 3.0–5.5 ms
   against a measured `R` of 9.816, and pre-registers that 4–7 ms of the organ is in neither. `R` = the
   softmax pass + the `A·V` loop = **9.816 ms/token** at `T10` @800, **81.4% of the attention organ**
   after `avx4`, and the organ is ~97% of `f`. Nothing has ever measured what is inside it. The
   method is already built and already validated: E4's planted control split the organ by running the
   `Q·K` loop 2× and 3× **bit-identically** and predicting the third point to −0.56%. Do the same to
   each half — an arm that runs the softmax loop twice, an arm that runs `A·V` twice, both
   value-preserving — and `R` splits the same way. Cheap, engine-only, no format change. Candidate
   levers once it is split: the softmax `sum` is a serial reduction over `pos` with a scalar `expf`
   per element (`L*NH*pos` = 616 k calls/token at `T10` @800), and `A·V` reads V at the same byte
   rate the `Q·K` loop reads K.
   **~~The attention accumulator~~ — CLOSED by E4, `LATENCY-CONFIRMED`** (`probes/E4_ATTENTION_ACCUMULATORS.md`).
   The dot loop went 6.53× and `f` halved. **The int8 KV cache that E3 named as the alternative lever
   is retired at ~1.16×** — it was the right lever for the loop E4 already fixed.

0-bis. **A real 10 B donor.** Opened by E6, **half-closed by E7** (`probes/E7_REAL_LARGE_DONOR.md`, `REAL-WEIGHTS-CONFIRMED`): Qwen2.5-Coder-7B, **7.07 B active/token = 67% of target**, passes every gate — 30.46 GB loaded exactly, parity 1.07e-05, **160/160** greedy tokens against PyTorch — and decodes at **4.460 tok/s** (3 reps, spread 0.45%), **11.2× short of 50**. The prediction written before the export, 4.66, was off by −4.3%. **What remains is the last 33% of the size, and it needs a download** (~28 GB for a real 10–14 B); nothing bigger is on this disk. The original statement of the item, still worth reading:

   ~~**A real 10 B donor.** Opened by E6.~~ Every measurement at target scale in E3, E4 and E5 was taken on `T10`, a **synthetic shape file** — correct dimensions, no trained weights. E6 showed a real donor generating text and could only do so at **0.5 B and 1.5 B**, because those are the only real weights on disk. So "10 B at 50 tok/s" currently has a measured *speed* and no measured *model*, and nothing has ever run end to end at the size the goal is about. Exporting one is a conversion job, not a research question, which is exactly why it keeps not happening.

1. **D4b** — the calibration budget. Promoted from bookkeeping: T2's two best arms are both
   calibration-driven, so every one of their numbers is a **floor** — and after E2 that now
   includes the fold's own `−0.220001`, which was measured at 32 calibration sequences like
   everything else.
2. **Healing** (QAT / layer-wise distillation) — still on the critical path per T2 §7. **Its
   starting point moved**: the whole runnable model is now **`+2.465779`** (E2 arm `NLH`), not
   `+2.708111`. T2b §6 says where to aim it: **attention, 4.98× the FFN's damage per weight at 10%
   of the weights**. Note that T2's GPTQ result — at ternary width the per-layer error is the wrong
   objective on a badly-placed grid — constrains what "layer-wise" is allowed to mean here.
3. **Why the same fold buys on the layers and costs on the head** (E2 §5.4). `q,k,v,gate,up` gain
   `−0.220`; `lm_head` loses `+0.177`. E2 killed the obvious explanation — zero fraction does not
   predict it and does not even hold its sign — and put nothing in its place. The per-row structure
   of R3's threshold search against the per-column structure of a gain fold has not been measured.
   Cheap, exporter-only, and it is the one place the fold left value on the table.
4. **S1's scale arm** — blocked on the fp16 NaN (`eager` attention overflows QK^T; diagnosed, §5).
   Every sparsity result this programme owns is measured at one size.
5. An already-MoE donor, and **a donor with a small vocabulary** (§7). E3 sharpens both: a small
   vocabulary buys much less than §7 used to imply (Mistral-7B is still at **141% of budget at 800
   context** with the smallest vocabulary on the disk), and **nine of eighteen donors — every MoE
   and every hybrid — have no measured `f` at all**, so their screen rows are withdrawn, not
   restated.
6. ~~**`f` beyond 800 tokens of context.** It grows with position and nothing measured bounds it.~~
   **CLOSED by E7 §8.2** (`probes/E7_REAL_LARGE_DONOR.md`): **linear, 0.0319 ms per token of actual
   context, two intervals agreeing to 0.2% over a 5.3× range in context, no knee to 1600.**
   Measured on the real Coder-7B, with contended cells excluded by the `ffn`-invariance witness —
   six of nine survived, and the three discarded would have manufactured a knee between 300 and
   800. `1000/f` may now be extrapolated linearly to at least 1600 tokens.

**Closed since the last revision.** The `--fuse` × `OMP_WAIT_POLICY` matrix ran (`SPEED_LEDGER.md`
§12.4 — both hypotheses die; `--fuse` not adopted). T3 ran and closes the residual-stream rotation
line together with brief §5's two follow-ons (online Hadamard, activation-side rotation).
**E1 ran** (`probes/E1_BPB_THROUGH_ENGINE.md`): the engine scores what PyTorch scores to `1.5e-05`,
T2b's `+2.708111` is replicated bit-identically through a file round-trip, and the 2 GB load ceiling
that would have blocked the target model outright was found and fixed.
**E2 ran and closes the standing item 1** (`probes/E2_RMSNORM_FOLD.md`): the fold survives to the
artifact at T3's exact value **and its exact dispersion**, Gate F passes on the deciding donor with
4.2e+04× of margin, and it is **adopted** — `qwen_export.py --fold` defaults to `layers` as of
`49b6654`, with `--fold none` pinned in E1's runner in the same commit so its published numbers keep
reproducing (both directions checked by sha256 against artifacts already on disk). `--fold all` is
measured and **rejected**: into a ternary head the final gain costs BPB on both donors.

## 5. Bugs found in our own instruments (all fixed, all with controls added)

| bug | how it presented | fix |
|---|---|---|
| **A floor prediction charged the wrong bytes and the measurement beat it** | E4's brief predicted the `Q·K` loop could not go below **5–8 ms** because "315 MB at 50 GB/s is 6.3 ms". It measured **2.242 ms**. The 315 MB is **touched** bytes (GQA re-reads); only the **78.7 MB unique** reach DRAM, at 35.1 GB/s | the two conventions were already law here — a ceiling is a denominator, and this is where the wrong one hid. E4 §4.3 reports the prediction as wrong beside the number that beat it |
| **A gate compared across sessions and could not be answered** | E4's G4 required `serial` to reproduce E3's published tok/s; it missed by +3.4% to +7.0%. My first diagnosis blamed an address-arithmetic hoist — priced, that is **±2% with inconsistent sign**. Rebuilding **E3's own binary** and running it in the same session showed it misses **E3's own table** by −1.6% to −4.2% | the gate is `MALFORMED`, not failed. Any reference from another session must be **re-measured in the same sweep**; E4's run 2 re-timed all seven arms under one binary, which is the only reason the probe survived |
| **The A1.2 gate reported PASS over an all-NaN run** | `max(0.0, nan) == 0.0` in Python swallowed all 51 comparisons | hard-fail on any non-finite, minimum comparison count, and **7 planted self-test cases run before any model loads** — the old code fails 3 of them |
| **`l1_keep_count` turned a NaN into a plausible measurement** | `clamp_(1, F)` mapped a NaN row to "keep 1 neuron" → achieved sparsity `8959/8960` | raises `FloatingPointError` on non-finite input |
| **fp16 NaN blamed on the GPU** | my own diagnostic did not pass `attn_implementation` and tested SDPA, not the `eager` path the probe uses | reproduced on CPU with `--attn eager`; cause is HF eager computing QK^T in fp16 (**274,672 vs the 65,504 limit**) before dividing by √head_dim |
| **T1's planted control was mis-specified** | required random signs ≫ ternarization; they are only 1.21× apart, so it returned VOID on sound numbers | identity-substitution control added (bit-exact). **T2 §4 then showed the premise itself was false**: random signs are not ≫ the treatment, they are indistinguishable from it |
| **`--calib-seqs` defaulted to 8 while T2 measures at 32** | exporting `--rule R3` would have built a model on a quarter of the calibration budget that produced the number, and the BPB gap would have read as the runtime disagreeing with PyTorch | default → 32, stderr warning otherwise, and the sidecar records the budget and the organ lists |
| **the LUT diagnostic reported whole-vector crest while groups were active** | it kept calling the derived figure "effective levels" when the grid was per-group, i.e. a plausible number describing the wrong thing | crest computed in-group, both labels corrected, header states which scale is in force |
| **both `rope()` calls sat inside the `qkv` timer** | the per-organ table read qkv at **4.1 GB/s** against the head's 17.6 — a 4× anomaly that does not exist — and two experiments were built to chase it. It also hid that rope was **9.6% of every token** | `T_ROPE` is its own bucket; rope hoisted out of the head and layer loops (**+10.3%, bit-identical**); §11.4 marked superseded and its two derived claims withdrawn in §12.1 |
| **the bench harness called an idle machine CONTENDED, twice** | once because it compared an 800-token run to a 300-token reference (attention is `O(position)`), once because min-max over 6 rounds is set by a single bad round | reference must match `--bench` length; the gate is now the IQR; the harness prints which of its two blocks is readable |
| **a pre-registration contradicted itself and the gate fired on the contradiction** | T3 returned `VOID`: brief §3 fixed the organ set at FFN+attention (196 tensors), brief §4 pinned the replication constant to T2's **FFN-only** Δ (`+1.709372`, 84 tensors). The runner obeyed both halves and arm Q missed by 1.007 | the constant was checkable and was checked: arm Q reproduces T2b's arm FA **bit-identically** from a different runner. **A replication constant must name the arm, the organ set and the file it came from**, so a mismatch with the arms section is visible on the page |
| **`--quant ternary` had been dead since `--rule` landed** | `w_tern` squeezed a scale that `quantize()` already returns as a `[out]` vector → `IndexError`. Nothing caught it because every artifact this programme built used `--quant packed`, which does not go through that line | the second squeeze removed; verified empirically before the fix, not assumed |
| **two exports of the same command produced different files** | R3's calibration forward changes its reduction order with torch's thread count: 6 vs 1 threads moved **102,123** `act_rms` elements (worst `1.9e-06`), so the sidecar recorded a sha256 it could not reproduce | `--threads`, recorded in the sidecar. **The thread count is part of the artifact's identity, not a speed knob** |
| **the exporter was not loading the model the probes measured** | it omitted `attn_implementation`, so HF gave it **sdpa** while `common.load_model` — and therefore T1, T2, T2b, T3 — uses **eager**. E1's Gate A fired at `2.980e-08`, exactly one ulp. Measured: `act_rms` differs on **142,977** elements (worst rel `8.2e-06`), moving **132,844** stored scales by up to **6 ulp**. All **357,826,560 codes were identical throughout** — the artifact's *identity* was wrong, not its content | `eager` pinned in the exporter and recorded in the sidecar. Gate A then passed **as pre-registered**: 0 scales differing, 0.00 ulp. Same law as the fp16 row, in a second place: **reproduce the configuration, not just the model** |
| **the best quality claim had no artifact** | `qwen05b_packed.bin`'s sidecar has no `rule` field — it predates `--rule`, so every speed number was taken on **R0/BitLinear158** (zero fraction `0.327`), the worst of T2's five rules. **No R3 model had ever been exported.** The format is identical so the speed numbers stand | E1 exports R3 (zero fraction `0.4922`, consistent with T2b's `0.4714` at full budget). Found by reading the sidecar of the file on disk rather than the command that was supposed to have written it |
| **the engine could not load a model over 2 GB, and blamed the file** | E1's 1.5B fp32 arm died with `FATAL: bad magic -- not a QWENDON1 file` on a 6,174,857,268-byte export whose magic was intact. `long` is 32 bits on Windows even on x64, so `fseek(SEEK_END)` FAILS above 2 GB and `ftell` reports 0 → a zero-byte blob was allocated, zero bytes were read (**which equals the zero requested, so the `short read` guard passed**), and `memcmp` compared the magic against an empty buffer. The largest file the engine had ever been given was 1.84 GB, just under the ceiling. **A 10B ternary packed model is ~5 GB: the target was unloadable, and no speed probe could have found it** | 64-bit offsets chosen by platform (`_fseeki64`/`ftello`), a real error when the seek fails, 1 GB chunked reads. **Planted control before the rebuilt binary produced any number**: it re-scored the measured 0.5B TQ artifact at `NATS_TOTAL 162120.4241599279`, bit-identical. **Known-positive**: the 5.75 GB file then loaded and scored |
| **a sweep's `untied` field recorded the wrong thing, and arm state leaked** | `t2b_organs.json` says `"untied": false` on arm FAH, an arm that ternarizes a *tied* head — which would mean the embedding was ternarized too. It was not: E1's arm TQH reproduces FAH **bit-identically**, and two different models cannot. Arm `I` reports `untied: true` and every later arm `false`, because T2b's `restore()` restores weights but never re-ties | no number changed and none is withdrawn — arm `I` returns the base BPB exactly. Recorded because it was caught by a **replication**, not by the sweep: **the field means "did this arm untie", not "is the head untied here"**, and state crossed arm boundaries |
| **a result file was named after the model alone** | E1's pre-registered 1.5B fp32 **subset** run was about to overwrite the 55-minute TQ/TQH JSON written under the same name | a subset run now carries its arms and sequence count in the filename; only the canonical run keeps the bare name |
| **a gate that was never evaluated reported that it had failed** | E2's run 2 carries only the five ternary arms, so Gate F -- which is built from `XF`/`XA` against `F32` -- would have found no arms to compare, left `gate_f` empty, and printed `VOID (Gate F failed: the fold is not exact where it must be)`. The run owning the **label** would have reported the fold inexact on the strength of nobody having looked, and the brief itself had pre-registered the split that causes it (§3.2) without foreseeing what the runner would print under it. The same brief also predicted run 1 would return `INCOMPLETE`; run 1 holds all eight arms and returns a real label | `gate_F_measured` separated from `gate_F_ok`; `VOID` is now reachable only from a gate that was measured and failed, and a run missing the gate prints the fold term plus `GATE-F-NOT-MEASURED-HERE` naming the run that owes it. Brief §3.3 written before run 2, changing no threshold or arm. **All three states planted and shown to fire** (`3aa1bef`), the failing one via a copy with `GATE_F_FP32_TOL = -1.0` -- without it the patch would only have been shown not to say `VOID`, and a guard that never fires looks identical to one that fires correctly |
| **a budget formula charged the same milliseconds twice** | `SPEED_LEDGER` §12.2 computes `Delivered: 493,961,216 / 17.833 ms = 27.7 G-weights/s`, where **17.833 ms is the wall** — so the 1.17 ms of non-weight work is already inside the denominator. It then prices the budget as `27.7 × (20 − 1.17) = 522 M`, subtracting it a second time. Self-consistently the figure is **554 M** (`r_wall × 20`) or **558 M** (`r_w × (20 − f)`), which agree to 0.7% | the error was ~6% and in the **safe** direction, which is why nothing caught it for two weeks. `e3_budget_by_shape.py` prints both conventions side by side and never mixes them. **A rate and a reservation must be read off the same denominator**, and the way to check is that the two self-consistent forms agree |
| **a planted control was planted outside the range the instrument is used in** | E3's Gate V1 compared an **all-zero-code** model to a no-zero one and fired at 15× the IQR, voiding the run. The kernel was innocent: `qkv_proj`, `o_proj` and `head` — pure packed matvecs — were flat to **0.005 ms**, and the whole 2.080 ms sat in `ffn`, the only organ holding a transcendental. An all-zero model feeds `expf` exactly `0.0f`, every libm's early-out; 2.080 ms / 116,736 calls = **17.8 ns per call**. A zero fraction of 1.0 is not something any exporter can produce, and across the range a real export occupies (0.47 vs 0.00) the timing is flat to 0.08 tok/s | the `VOID` was honoured and no arm was generated under it. The replacement gate compares synthetic against a **real exported artifact of the same shape**, fixed at a scale where no synthetic file existed yet — and passed at **0.000%** (1.5 B). The failed gate is reported next to the pass, not deleted. The accident is that it also **measured** what it was meant to assume: the matvec's value-independence is no longer a reading of the source |
| **a speed gate contained a hard constant, and its own ground truth could not pass it** | E3 brief §8.4 fixed Gate V2′ at *"57.790, IQR 0.190"*, measured on a real artifact twenty minutes earlier. The synthetic file gave 56.180 and failed. Re-run **interleaved A/B/A/B**, the *same real file* gave **56.095** — it had moved **2.9%** — while synthetic and real differed by **0.200 tok/s, 0.357%**, with one pair in which both dipped together | **fourth time this ledger has been bitten by the same family**: §11 a rate under contention, §12 a timer bracketing the wrong work, E3 §2.3 an anchor on the wrong head, now a constant that did not survive twenty minutes. The rule is now explicit: **a speed gate may not contain a constant; it must name a file to be measured concurrently, interleaved** |
| **a brief asked for one configuration in prose and the code read another from a config file** | E3 §3 asked for *"a ternary head (the runnable configuration E2 settled)"*. `synth_export.py`'s `SHAPES` carried each donor's own `tied` flag, and **a tied model runs its head as the fp32 embedding** — 544.6 MB/token, 13.8 ms, **52% of the token**. Six arms were not generated on the strength of a configuration nobody intends to ship. `SPEED_LEDGER` §12.2 had said its anchor was untied and packed without naming it: `136,134,656 weights, 68.1 MB/token` is **0.5 bytes per weight** | found because the synthetic file agreed with the **real** artifact of the same configuration to 0.36% while both missed the ledger by 35% — a disagreement that could only be about *which* configuration. `--head {ternary,donor}` added, default `ternary`. **A configuration named in prose must be named in a flag**, and a table's units are a claim about which configuration produced it |

## 6. Working rules this programme has paid for

- **A negative is only as strong as the sweep behind the setting it was measured at.** (D0c, D4b, T2)
- **A control must itself be shown to fire.** A gate never seen to trip is decoration. (§5, row 1)
- **Profile before optimising.** The head was 40% of every token and nobody had looked. (R1)
- **When reproducing a failure, reproduce the CONFIGURATION, not just the model.** A different
  library default invalidated a conclusion. (§5, row 3)
- **Byte-rate ÷ bytes-per-weight is only valid where the path is bandwidth-bound.** (P2 → R1)
- **A Δ against a baseline cannot support a claim about one arm versus another.** The arms are
  correlated across sequences; the contrast has to be bootstrapped paired. Doing it moved two of
  T2's apparent results — TWN and GPTQ-on-the-naive-grid — from "worse/better" to *not
  distinguishable*. (T2 §4)
- **When an instrument returns an impossible ordering, test the instrument before the finding.**
  GPTQ scoring below its own starting point is not physically possible; two controls showed the
  code was right and the objective was wrong. (T2 §5)
- **A profiler bucket is a claim about what is inside it.** `rope()` inside the `qkv` timer both
  hid a 9.6% cost and manufactured a 4× anomaly that two experiments were then aimed at. Before
  deriving a rate from a bucket, name every operation the bracket contains. (§12.1)
- **A throughput number carries its bench length.** Attention is `O(position)`, so a longer bench
  has a genuinely lower tok/s — 1.7 ms/token between 300 and 800 on this donor. Two rates taken at
  different lengths are not comparable and their difference is not contention. (§12.3)
- **Quantization damage does not add across organs.** The head costs `+0.339` alone and
  `−0.009 ± 0.020` on top of a ternary FFN+attention; the three single-organ arms sum to `+3.184`
  where the combination measures `+2.708`. A per-organ cost is only a cost *in the company it was
  measured in*. (T2b §4)
- **A smoke run establishes that the apparatus runs, not the sign of the effect.** T3's 4-of-28-layer
  smoke passed every control (X at `6.4e-07`, planted null clean) and flipped the sign of all three
  contrasts, returning the opposite label to the run. **A truncated model is not a small model.**
  (T3 §4.5)
- **A gate constant is a claim, and it is checkable against the arm that produced it.** T3's
  `VOID` was a mis-specified constant, not a broken harness — provable because an independent
  runner had already measured the identical arm to the last bit. Report the label the rule
  returns, then show the check; do not rewrite the rule. (T3 §1.3)
- **A pre-registered threshold needs its own interval before the label is read as settled.** T2b
  passed its 1.60 bar at 1.584 — but the ratio's ci95 is `[1.514, 1.649]` and a third of the
  bootstrap lands on the other side. The label stands; the confidence in it does not. (T2b §5)

- **A ceiling nothing has reached is a ceiling nobody has tested.** The runtime could not load a
  model above 2 GB; the largest one ever handed to it was 1.84 GB, so the limit had never fired,
  and the error it produced accused the artifact instead of the reader. The target model is ~5 GB.
  **Before scaling a number up, check that the apparatus can hold the object the number is about.**
  (E1 §4.4)
- **An error message is a hypothesis, not a diagnosis.** `bad magic` was reported about a file
  whose magic was intact, by code that had allocated a zero-byte buffer and passed its own
  short-read guard because zero bytes read equals zero bytes requested. **A guard that compares a
  quantity to itself checks nothing.** (E1 §4.4)
- **A pre-registration that splits a decision across runs must say which run owns the label.**
  E1's 1.5B arms were split across two invocations by the brief, so neither could evaluate the
  decision function and both returned `INCOMPLETE` on numbers that met every term. Second time in
  three probes that the brief, not the code, produced the mechanical label. (E1 §6.2, T3 §1.3)
- **A small donor establishes the sign, not the magnitude — and not always the ordering.** Across
  E2's two donors every sign that mattered replicated and **no magnitude did**: the fold's benefit
  differed 1.5×, the head fold's cost 3.6×, the benefit a ternary head adds 8.9×, and one whole
  comparison **inverted** (`--fold all` is worse than no fold on the 0.5B, better on the 1.5B).
  Same shape as E1's "the head is free is a 1.5B result". **Give the deciding donor the label in
  writing before the small one produces a number.** (E2 §5.0, §3.2 of its brief)
- **A replication that shares the estimator's seed reproduces its dispersion too, and that is not
  extra evidence.** E2 recovered T3's fold as `−0.220001 ± 0.052861` — point *and* paired SE at the
  printed precision. Forced, not corroborating: same seed 7, same 24 sequences, same byte weights,
  and per-sequence nats agreeing to `1e-07`. **Count it once.** (E2 §4)
- **`ci95 excludes 0` is a statement about accumulation when the effect is at the ulp.** The same
  contrast, same arms, same code, **excluded** zero on one donor (`+2.037e-08`) and **contained**
  it on the other (`+3.827e-08`). Exclusion is not magnitude; a rule must pair it with a threshold.
  (E2 §5.2)
- **A default that changes behaviour is only safe when both sides are pinned to something already
  on disk.** Adopting the fold flipped `--fold` to `layers`, which would have silently re-exported
  every caller that omits the flag — including E1, whose reference builder still defaults to
  unfolded. The flip and the pin went in one commit, and both directions were checked by **sha256
  against artifacts that had actually been measured**. (E2, `49b6654`)
- **A planted control must be planted inside the range the instrument will actually be used in.**
  An all-zero model is not a model any exporter can produce; the gate that used one measured `expf`
  instead of the kernel it was defending, and voided a run for it. (E3 §2.2)
- **A speed gate may not contain a constant.** It must name a file to be measured **interleaved**
  with the thing under test. E3's reference artifact moved 2.9% in twenty minutes and could not pass
  the gate its own measurement had defined. Fourth instance of the same family in this ledger.
  (E3 §2.5)
- **A within-run IQR is not a reproducibility interval.** This machine's within-sweep
  dispersion is **0.000–0.015 tok/s** and its **between-sweep dispersion is 5–10%** — E3's own
  binary misses E3's own published table by −1.6% to −4.2%, and the same code read **3.240 and
  3.030 two hours apart** while its IQR stayed at 0.005. A gate that compares a measurement to
  a number from another session is measuring the calendar. **Re-measure the reference in the
  same sweep.** Third instance: E3 §2.5's reference moved 2.9% in twenty minutes, and the
  ledger's original 18.6 G-w/s was contended by 35%. (E4 §2.4)
- **A threshold placed inside the instrument's dispersion cannot decide anything.** E4 §5 drew
  `LATENCY-CONFIRMED` at ≤0.50× and the arm came in at **0.503×**. The remedy is not a new
  threshold but a better measurement: **interleaved A/B/A/B/A/B**, ratio per adjacent pair.
  The baseline arm moved **10.55%** across the three pairs; the ratio moved **2.98%**. That one
  comparison is the whole case for interleaving, measured. (E4 §2.5)
- **A planted control is worth more as a decomposition than as a gate.** E4's `serial2` runs
  the dot loop twice **bit-identically**, which does not just prove the timer is attached — it
  *solves* the organ into the loop under test and the rest. `serial3` then hits the third point
  to **−0.56%**, so the split is measured rather than fitted. Every mechanism claim in E4 rests
  on that, and the same trick is now the method for the next probe. (E4 §2.2–§2.3)
- **A rate and a reservation must be read off the same denominator.** `27.7 G-w/s` is weights/**wall**,
  so pricing `27.7 × (20 − f)` charges `f` twice. The check that catches it is cheap: the two
  self-consistent forms — `r_wall × 20` and `r_w × (20 − f)` — must agree, and they do (554 vs 558 M).
  **Committed a second time the same day, in the probe that wrote this law:** E3 first published the
  gap to 50 tok/s as `530 / r_w = 15.7×` when `530 G-w/s` is a *wall* quantity — it is `530 / r_wall`
  = **16.18×**, which is just `50 / 3.090`. *Where a rate-free form of the quantity exists, quote it:
  it has no denominator to swap.* (E3 §4.3, §6.7)
- **A configuration named in prose must be named in a flag.** E3's brief said "ternary head" and the
  code read `tied` out of a donor config; the two differ by 52% of the token. (E3 §2.3)
- **Extrapolating a fit requires publishing its residuals first.** `f` is fitted per donor shape only
  because the fit reproduces the twelve measured points to 7.7% worst and **0.0–1.2% at the shapes
  that decide**; the residual table is the licence, and it is printed above the extrapolation.
  (E3, `speed/e3_budget_by_shape.py`)
- **A shape can be measured without weights, but only if the instrument is gated against a real
  artifact.** E3's synthetic files match real exports **byte-for-byte in size** and to **0.000% in
  time** at 1.5 B — which is what buys the right to measure shapes nobody has 20 GB to download.
  (E3 §2.1, §2.4)

## 7. The head, the tokenizer, and the thing nobody priced

> ### ⚠ SUPERSEDED AGAIN BY E4 (`probes/E4_ATTENTION_ACCUMULATORS.md`) — the screen re-opens
>
> E3's re-run below priced every donor against an `f` measured on an engine whose `Q·K` reduction was
> latency-bound. E4 halved `f`, so the fit's attention coefficient `A` was wrong by **2.04×**
> (`3.09445e-07` → `1.51461e-07` ms per FMA-unit). **Scaling the old table would have been a guess**,
> so all six shapes were re-measured under `--attn avx4` **in one sweep** (`results/e4/shapes_avx4.json`)
> and both `f` and `r_w` are taken from it — E4 §2.4 forbids a rate from one sweep meeting an `f` from
> another.
>
> **Residuals first, because they are the licence** (`speed/e3_budget_by_shape.py --arms ...`):
> **0.3–0.8% at the 7–10 B shapes that decide**, −0.2% to +7.8% at 1.5–3 B, and **−23% at `S05`** —
> the two-term model no longer fits the small end, because with attention halved the per-call
> overheads dominate there. **No donor in this screen is below 1.5 B**, so that residual does not
> price anything; it is stated because the fit is now unfit for the shape it used to fit.
>
> | donor | head weights | **budget @300** | **head as % @300** | was (E3) | **budget @800** | **head as % @800** | was (E3) |
> |---|---|---|---|---|---|---|---|
> | Qwen3-8B | 622 M | **564 M** | **110%** | 152% | **369 M** | **169%** | 2763% |
> | OLMo-2-7B | 411 M | 578 M | 71% | 94% | 404 M | **102%** | 432% |
> | Mistral-7B-v0.3 | 134 M | 578 M | **23%** | 31% | 404 M | **33%** | **141%** |
> | Phi-3-mini | 99 M | 575 M | 17% | 21% | 452 M | 22% | 44% |
> | Qwen3-1.7B | 311 M | 592 M | 53% | 59% | 522 M | 60% | 81% |
> | Qwen2.5-1.5B | 233 M | 604 M | 39% | 42% | 552 M | 42% | 53% |
> | SmolLM2-1.7B | 101 M | 599 M | 17% | — | 539 M | 19% | 24% |
>
> **What changed.** E3's headline at 800 context was that the screen *stopped being about the head*:
> every 7–8 B dense donor was over the line regardless of tokenizer, and **Mistral-7B — the smallest
> vocabulary on disk — was at 141%**. It is now at **33%**. `f` had eaten the budget; it does not any
> more.
>
> **What did not change.** **Qwen3-8B is still the one that breaks**, at 110% @300 and 169% @800, and
> it breaks for the reason it always did: a 151,936-token vocabulary on a 4096-wide model is a 622 M
> head, and the head is charged in full every token. OLMo-2-7B is marginal at 800 (102%). **The head
> is set by the tokenizer, not the model** — the one claim that has survived every re-pricing.
>
> **The nine unpriceable donors are still unpriceable** (MoE and hybrid/SSM), for the same reason and
> with the same withdrawal.
>
> ### ⚠ SUPERSEDED BY E3 (`probes/E3_ENGINE_AT_TARGET_SCALE.md`)
>
> The table below screens twelve donors against **522 M**. That constant is wrong twice: it charges
> the fixed cost twice (§0), and — far larger — it uses a **1.17 ms** non-weight reservation that E3
> measured at **10.576 ms** at a 10 B shape. The screen is re-run below from measured `f`, per shape.
> `donor_speed_budget.py`, which produced the old table, says of itself that *"every attention row
> below is priced by ANALOGY to the proj-GEMV path, and that analogy is the single largest source of
> error here."* E3 measured it, so the analogy is retired.
>
> **The re-run.** `speed/e3_budget_by_shape.py` fits `f = A·L·NH·HD·pos + B·L·D` on E3's twelve
> measured points — worst residual **7.7%**, and **0.0–1.2% at the 7–10 B shapes that decide** — then
> applies it per donor. `budget = r_w × (20 ms − f)`, `r_w` measured per size class.
>
> | donor | D | V | head | fitted `f` @300 | **budget** | **head as % of budget** | was (§7) |
> |---|---|---|---|---|---|---|---|
> | Qwen3-8B | 4096 | 151,936 | 622 M | 7.928 | 409 M | **152%** | 112% |
> | OLMo-2-7B | 4096 | 100,352 | 411 M | 7.047 | 438 M | **94%** | — |
> | Mistral-7B-v0.3 | 4096 | 32,768 | 134 M | 7.047 | 438 M | **31%** | 24% |
> | Phi-3-mini | 3072 | 32,064 | 99 M | 5.286 | 468 M | **21%** | 18% |
> | Qwen3-1.7B | 2048 | 151,936 | 311 M | 3.083 | 528 M | **59%** | — |
> | Qwen2.5-1.5B | 1536 | 151,936 | 233 M | 2.312 | 552 M | **42%** | — |
> | SmolLM2-1.7B | 2048 | 49,152 | 101 M | 2.643 | 542 M | **19%** | — |
>
> **At 800 tokens of context the screen stops being about the head at all:**
>
> | donor | budget @800 | head as % |
> |---|---|---|
> | Qwen3-8B | 23 M | **2763%** |
> | OLMo-2-7B | 95 M | **432%** |
> | **Mistral-7B-v0.3** | 95 M | **141%** |
> | Phi-3-mini | 226 M | 44% |
>
> **Mistral-7B has the smallest vocabulary on the disk and still cannot reach 50 tok/s at 800
> context** — not because of its head, but because `f` has eaten the budget. Every 7–8 B dense donor
> is over the line at 800 context regardless of tokenizer.
>
> **Nine of the eighteen donors on disk are not priced by this fit at all** — MoE (`Qwen3-30B-A3B`,
> `Qwen3-Next-80B`, `OLMoE`, `DeepSeek-V2-Lite`, `Mixtral`, `gpt-oss-20b`, `granite-4.0-h`) and
> hybrid/SSM (`Zamba2`, `Nemotron-H`, `Falcon-H1`, `mamba2`) do not have uniform dense attention in
> every layer, so `f` at their shape has not been measured. **`gpt-oss-20b` at 105% and
> `Nemotron-H-8B` at 97% in the old table rested entirely on the analogy `donor_speed_budget.py`
> flagged as its own largest error, and are withdrawn rather than restated.**
>
> What survives from the old table unchanged: **the head's size is set by the tokenizer, not the
> model** — and E3 turned that from a weight count into an end-to-end measurement, `M7` vs `Q8`,
> 4.637× the vocabulary for 4.629× the head time (0.2%).

### 7.1 The original screen, kept for the record

`donor_speed_budget.py` prices the output head against the measured **27.7 G-weights/s** of the
packed kernel after the rope hoist (`SPEED_LEDGER.md` §12.2). The head is a dense GEMV of `D × V`
touched on **every** token, and **no** MoE, carve, sparsity or reconstruction result in this
programme touches it.

| donor | D | V | head | % of the 522 M budget for 10B @ 50 tok/s |
|---|---|---|---|---|
| Qwen3-8B | 4096 | 151,936 | 622 M | **112%** |
| openai/gpt-oss-20b | 2880 | 201,088 | 579 M | **105%** |
| Qwen2.5-Coder-7B | 3584 | 152,064 | 545 M | 98% |
| nvidia/Nemotron-H-8B | 4096 | 131,072 | 537 M | 97% |
| mistralai/Mistral-7B-v0.3 | 4096 | 32,768 | 134 M | **24%** |
| microsoft/Phi-3-mini | 3072 | 32,064 | 98 M | 18% |

**Two of twelve donors on this disk have an output head larger than the entire 50 tok/s budget,
and two more are within 3% of it.** Qwen3-8B cannot reach 50 tok/s on this machine even if every
other weight in it were free. It was four of twelve at the 495 M budget: **the entire 10.3% rope
win bought exactly two rows crossing back under the line.**

**Qwen3-8B and Mistral-7B are the same width.** Their heads differ by 4.6× entirely because of
vocabulary size. Every donor-adaptation probe this programme owns was measured on Qwen, which has
the worst head-to-body ratio on the disk. **Vocabulary is a speed variable, it is chosen rather
than earned, and it has never been treated as one.**

Two measured facts sharpen it: the head was 40.4% of every token in fp32 before R1 ternarized it
(`probes/R1_DONOR_RUNTIME.md` §3), and it is the one organ that pays **nothing** for int8
activations on the LUT path — 0.13997 → 0.14084 when excluded — because its input is the final
RMSNorm output (`donor_engine.c --lut-no-head`, commit `95b7fd3`).
