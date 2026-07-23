# INVENTORE_07 — The Teardown: the inference engine, disassembled for speed

**Status: Inventor pass 2, engine-scoped, desk + one $0 CPU probe (2026-07-19). No engine code touched.**

> **Scope correction (owner, 2026-07-19).** The product is the **C inference engine and its tok/s on a
> commodity CPU**. The weights are a *fixed input*; the training plan is out of scope for this pass. The
> first cut of this document took apart the training pipeline — wrong object. This version takes apart the
> **runtime**: the per-token execution pipeline, the memory layout, the two-pool streaming, the block
> chassis, the recall lookup, the decode loop — and asks, for every piece and every pair, whether a fusion,
> a parallelization, a decomposition, or an order inversion makes the *engine* faster. All moves operate on
> the weights the training produces; none proposes a new training run.

Grammar at every edge: **FUSE / PARALLELIZE / SPLIT / REORDER / ELIMINATE**. Every move names its
falsification and its cost, and is measured against the *measured* engine profile — not a guess.

Sources: `ENGINE_PLAN.md` (E1→Phase 63, all gates + laws), `SCALEUP_ARCHITECTURE.md` (§1–3 the bandwidth
equation and the two walls), the Phase 60/61/63 results. Companion: INVENTORE_00–06 (organ-level pass 1).

---

## 1. The engine, as measured

**Per-token profile (full consolidated engine, single-thread, 8.3M sandbox — ENGINE_PLAN E3.5):**

| component | share | nature | status of every direct attack |
|---|---:|---|---|
| **SSM-projection GEMVs** (in/x/dt/out_proj, **fp32**) | **52.7%** | **memory-bound** | ternary → REJECTED (P61, +0.018/0.022 BPB); vectorize → REJECTED (memory-bound, −4% in-engine); threads → 2.45× |
| LUT ternary MoE-MLP | 23% | gather/kernel-bound (~4.2 GB/s eff., thread-flat ×1.06) | dispatch overhead ~8.4 µs/expert (engine-v2 queued) |
| scan-exp | 13.7% | compute | fast-exp poly = 25.9× already banked |
| SWA | 7% | mixed | — |
| head (V≤4K) | 3% | small | tied embeddings; hot while V small |

**Cumulative banked:** 176 → **848 tok/s dense** / 701 MoE (single-thread, 4.8×), → **2029 / 1519** at 6
threads (bit-identical). Every step parity-gated; total inference-time quality cost +0.00004 BPB.

**The two walls (§2 of the blueprint):** **W1 bytes** — active per-token slice ≤ **16 MB L3** (the
keystone; ~24–32M active ternary params) → in-cache compute-bound, spill → DRAM-bound and the kernel speed
stops mattering. **W2 the per-position compute floor** — C ≈ 1.14 ms/token at 8.3M, dominated by the fp32
proj-GEMV; ternary does not shrink it. Traffic T ≈ 0.71 ms → **C/T = 1.6**.

**The one sentence the profile screams:** *the engine is 52.7% a block of fp32 projection GEMVs that resist
every attack tried — and every attack tried assumed the extremes (1.58-bit ternary, or full fp32).* That is
where the teardown goes first.

---

## 2. The moves

### G1 — SPLIT the precision map: the hole between ternary and fp32 ★ (the 52.7% bottleneck, **measured**)

P61 asked "can the projections be **ternary** (1.58-bit)?" → no, +0.018/0.022 BPB, "precision-hungry control
organs." The engine therefore stores them **fp32 = 4 bytes/weight**. But the precision map has exactly **two
points**: 1.58-bit (dead) and 32-bit (baseline). **fp16 (2 B), bf16 (2 B), int8-per-row (1 B) were never
measured** — and they are *pure inference-time weight casts on the fixed trained weights*, no retraining.

Because the GEMV is **weight-bandwidth-bound** (matrix streamed once, vector resident), time ∝ stored bytes:
fp16 ≈ **2×**, int8 ≈ **4×** on that component. Two independent payoffs — the second is the deeper one:
- **(a) W2 speed:** the 52.7% component gets ~2–4× faster → Amdahl ⇒ ~**1.35× (fp16)** / ~**1.6× (int8)**
  end-to-end, *if quality holds*.
- **(b) W1 keystone:** the fp32 projections are **~8× denser per weight** than the 0.5-B/weight ternary MLP —
  they are the heavy tenant of the resident ≤16 MB budget. Halving/quartering them **directly expands how much
  model stays under the L3 cliff** = stays in the compute-bound regime where all the other levers pay. This is
  a lever on the project's *core constraint*, not a side optimization.

**Falsification (RUN, $0 CPU):** `benchmarks/in_research/t1_proj_precision.py` — cast only the four SSM
projections of `sp58_base` to each precision, one variable, measure val-BPB delta + top-1 vs the fp32
reference, against the engine's existing quality bar (E2: dBPB ≤ +0.005).

> **RESULT (2026-07-19, smoke 4K/seq256 — full 100K run pending in `t1_proj_precision_out.txt`):**
> | mode | B/w | proj-GEMV ideal | dBPB | top-1 vs fp32 | verdict |
> |---|---:|---:|---:|---:|---|
> | fp32 | 4.0 | 1.00× | ref | 100.00% | reference |
> | **fp16** | 2.0 | ~2.00× | **+0.0000** | 100.00% | **PASS** |
> | bf16 | 2.0 | ~2.00× | +0.0001 | 99.97% | PASS |
> | **int8 per-row** | 1.0 | ~4.00× | **+0.0000** | 99.90% | **PASS** |
> | int8 per-tensor | 1.0 | ~4.00× | −0.0000 | 99.58% | PASS |
>
> **The map has a cliff, not a slope: int8 is essentially free (+0.0000, top-1 99.9%), ternary is dead
> (+0.02).** The organs need ~8 bits of *dynamic range*, not 32 bits of mantissa, and emphatically not 1.58.
> P61's "precision-hungry" verdict was true only at the 1.58-bit extreme it tested — between int8 and fp32
> there is nothing to pay. *(The full 100K/seq512 run is the number of record; the smoke already resolves the
> sign and the cliff.)*

**Reading if it holds at 100K:** the biggest, most attack-resistant component of the engine has a **free ~2–4×
byte reduction on the table**, discovered by testing the *middle* of an axis everyone had only probed at its
ends. This is the crown of pass 2. The engine change is an export-format + kernel variant (int8 weights,
per-row scale — the machinery already exists on the *activation* side, E2); it does **not** touch the model.
**Non-move flag:** this does NOT reopen P61 (ternary stays dead) — it fills the interval P61 never sampled.

### G2 — PARALLELIZE: stream while you compute (two-pool overlap)  `[the scale-up regime]`

At 8.3M everything is L3-resident so this is mute; at scale-up the streamed expert pool is DRAM-bound and the
engine today is **C + T** sequential per token. Two overlap moves push it toward **max(C, T)**:
- **(a) resident double-buffer (no prediction needed):** the resident weights (proj / router / shared parts)
  have a **100%-deterministic** per-token access pattern — it is the *same every token*. Prefetch layer ℓ+1's
  projection matrix while computing ℓ. Pure latency-hiding on the memory-bound bottleneck; correctness-trivial
  (prefetch, not reorder).
- **(b) router-ahead (needs one number):** a provisional router on the *pre-mixer* activation issues the top-8
  expert stream, verified by the exact post-mixer router; the overlap window = the whole mixer. Cost: the
  in-place, same-token router predictability (Phase 58's 86–92% is unit-level, not expert-level).

**Falsification (b):** `$0` CPU probe on `moe_gran.pt` — top-8 overlap between `router(pre-mixer x)` and
`router(post-mixer h)`. High overlap → the window opens for free; low → (b) drops, (a) stands regardless.
**Status:** (a) is a pure engine-v2 lever; (b) is probe-first.

### G3 — REORDER + SPLIT: `in_proj` by consumer  `[inside the Mamba block]`

`in_proj` emits `[x, z]`: the `x` path feeds conv→x_proj→scan *now*; the `z` gate is consumed only **after**
the scan. Today they are one GEMV computed up front. **Split** `in_proj` into `in_proj_x` (needed now) and
`in_proj_z` (needed late), and **reorder** so `in_proj_z` streams/computes overlapped with the scan recurrence
(the 13.7% scan-exp + scan-other window). Exact (a linear split, identical output), engine-only, hides a slice
of the dominant proj block under compute that is already running. **Cost:** layout + loop-order change; the
`z` half is ~1/2 of `in_proj`. **Falsification:** bit-identity gate (it must be exact) + component timing.

### G4 — FUSE: the top-8 experts into one padded kernel  `[MoE MLP, credit the queue]`

Measured (64.0): the MoE-MLP is **gather/kernel-bound** (~4.2 GB/s effective, thread-flat ×1.06) — the ~8.4
µs/expert **dispatch overhead** around the kernel, *not* a bandwidth wall (kernel-pure rate is 7.45→17.0 GB/s).
Eight small separate matvecs pay eight dispatches. **FUSE:** gather the selected experts' ternary blocks into
one contiguous padded GEMV, one dispatch. Already in the engine-v2 queue; named here for completeness because
the teardown must credit the fusions already found. **Status:** queued, not new.

### G5 — the block chassis un-stranding condition (cross-piece: **G1 feeds it**)  `[execution model]`

The block-verify chassis is built, lossless, and **stranded**: scoped out at 8.3M because C/T = 1.6 vs the
win-condition C/T ≲ 0.15–0.20. C is dominated by the **fp32 proj-GEMV (52.7%)**. **G1 shrinks C directly** —
int8 projections cut the compute floor's biggest term ~4×. So G1 is not only an AR speedup: it **moves the
stranded chassis toward its win condition**, a concrete interaction between two pieces the pass-1 view missed.
This is a *scale-up trajectory* note (constraint-inversion says v1 speed is slack), stated as a falsifiable
target: re-measure C/T after G1 lands; if C/T crosses ~0.2 on a shared-streamed-dominated point, `--block K`
earns its keep. **Not a v1 claim.**

### G6 — FUSE the lookup primitive: recall ∪ router ∪ SWA  `[engine-code sense, 10B door]`

Three runtime paths — recall (state-query → IVF-Hadamard → ADC shortlist → top-16), router (state → top-8
experts), SWA (query → window-128 keys) — are the **same operator**: gather-topk-weighted-sum against a key
set, at three ranges. One kernel, three configs → one SIMD optimization surface (the queued 4-bit-ADC SIMD
generalizes to all three), one code path to keep bit-exact. A *simplification + shared perf surface*, not a
speed claim. **Status:** design note extending INVENTORE_04; no v1 action.

### Inherited-from-training dividend (recorded, not proposed)

The engine inherits whatever the training produces. Of pass-1's findings, the adopted **r26 x_proj** (if it
survives the rung-1 A/B) is the one with the biggest *engine* payoff: x_proj drops to 17.6% of its bytes, and
x_proj lives **inside the 52.7% proj block** → fewer bytes/FLOPs in the bottleneck. Noted as an inherited
dividend, correctly attributed to the engine this time; **not a new training proposal** (out of scope).

---

## 3. Ranked shortlist (engine, on fixed weights)

| # | move | operator | cost | why first |
|---|------|----------|------|-----------|
| **1** | **G1 precision-map hole (fp16/int8 projections)** | SPLIT | **$0 CPU probe (RUN); engine = export+kernel variant** | the 52.7% bottleneck, ~2–4× on the table, **measured near-free**; hits both W1 keystone and W2 floor |
| 2 | G2a resident double-buffer | PARALLELIZE | engine-v2, no probe | deterministic prefetch hides latency on the memory-bound bottleneck at scale-up |
| 3 | G3 in_proj split-by-consumer | REORDER+SPLIT | engine, exact | overlaps a slice of the proj block under the scan window, bit-identical |
| 4 | G2b router-ahead | PARALLELIZE | $0 CPU probe on moe_gran.pt | one number opens/closes a scale-up overlap window |
| 5 | G5 chassis un-stranding (via G1) | (cross) | re-measure after G1 | names the falsifiable condition that revives a built-but-stranded asset |
| 6 | G4 expert fusion; G6 lookup primitive | FUSE | queued / design note | already-found / v2 doors |

## 4. Non-moves (bounded, honest)

- **Ternary projections** — measured dead (P61). G1 is the *surviving* precision lever, not a reopening.
- **Block-verify speed claim at v1** — constraint-inversion: v1 speed is slack; G5 is the scale-up door only.
- **GEMV micro-vectorization of the fp32 projections** — measured −4% (memory-bound, ρ-law). G1 attacks the
  *bytes*, which is the actual bound; the FLOPs were never the problem.
- **Anything requiring retraining** — out of scope this pass (training excluded). The r26 dividend is noted as
  *inherited*, not proposed.

## 5. $0 probe queue

- **P1 (G1) — RUNNING:** projection precision sweep fp32/fp16/bf16/int8 on `sp58_base` →
  `t1_proj_precision_out.txt`. Smoke already resolves the cliff; full 100K/seq512 is the number of record.
- **P2 (G2b):** router pre- vs post-mixer top-8 overlap on `moe_gran.pt` — CPU, minutes.

---

**Closing note.** Pass 1 hunted organs (found x_proj). Pass 2 hunted the **runtime** and found that its single
biggest, most attack-resistant component — the 52.7% fp32 projection block — was only ever probed at the ends
of the precision axis. Sampling the middle shows a **cliff, not a slope**: int8 is free, ternary is dead. That
one measurement, if it holds at 100K, is a free ~2–4× on the dominant component and more model under the
keystone — the largest speed lever the engine has seen since fast-exp, and it costs an export-format change,
not a training run.
