# Phase 64.1 — Budget Document (sizing vs tok/s, the two walls as functions)

**Status: DONE 2026-07-06 (Architect); brackets refined 2026-07-12 with the 64.1b registered microbenches (HEAD f4a53cf) — verdict unchanged and strengthened. Inputs: the 64.0 registered tables (HEAD be5f448) + 64.1b (§1b). Feeds: the 64.2 design decisions. Canonical gate plan: `SCALEUP_ARCHITECTURE.md` §8.**

## 1. Measured anchors (3600X reference, per-protocol, 64.0)

- **DRAM cold-stream:** single-thread 21-26 GB/s; **aggregate ceiling ≈ 40-44 GB/s, saturated at 3 threads** (t6 adds nothing; unpinned t6 regresses). The old "~1.4-1.5×" derived figure is closed: measured ~1.6-1.8×.
- **Compute-floor decomposition (µs/tok, t1→t6):** proj-GEMV 662→272 (×2.43); LUT-MLP dense 313→207 (×1.51), MoE 573→543 (**×1.06 — thread-flat**); scan-recur 169→46 (×3.66); SWA 92→65; head 40→8; norms+glue ~7.
- **Two derived rates (load-bearing):** (i) proj-GEMV at t6 = 11 MB / 272 µs ≈ **40 GB/s = the aggregate bandwidth ceiling** — the fp32 projections already run at aggregate BW when threaded, so an L3 spill costs little extra at t6; (ii) MoE LUT effective rate = 2304 KB / 543 µs ≈ **4.2 GB/s** — far below any bandwidth ceiling: the expert path is gather/kernel-bound, not stream-bound (the thread-flatness is real; the "capped at DRAM ceiling" reading is not — the cap is lower).

### 1b. Refinement anchors (64.1b registered microbenches, HEAD f4a53cf, 2026-07-12)

Synthetic-weight kernels (real engine kernels, weight-free harness `benchmarks/phase64/bench_64_1b.sh`), grid sealed in the brief, close/cores.

**(a) proj-GEMV rate vs matrix size r(size)** (fp32, in=512, x L1-resident):

| size MB | 4 | 8 | 16 | 24 | 32 | 48 | 64 | 96 |
|---|---|---|---|---|---|---|---|---|
| t1 GB/s | 30.5 | 31.3 | 27.2 | 27.8 | 24.9 | 24.5 | 24.9 | 23.3 |
| t6 GB/s | 187.0 | 185.0 | 134.4 | 60.5 | 55.7 | 45.5 | 45.3 | 36.5 |

t1 breaks at the single-CCX L3 (16 MB) and sits flat ~23-25 GB/s in DRAM (64.0's 21-26 confirmed). t6 row-partition sees the **aggregate** L3 (2×16 MB): ~185 GB/s while resident, break visible from 24 MB, then a **declining slope** toward an asymptote ≈ 34-36 GB/s — the 45s at 48-64 MB are partial-L3-hit, pre-asymptotic, so the curve *is* the model: look up r at the candidate's proj size. The spill-is-a-slope shape assumption of §2 holds. **Caveat:** cache-region rates are upper bounds for the integrated engine — streamed experts pollute L3, pushing real rates toward the streaming tail; the fully-streamed floor is [34-40 GB/s].

**(b) expert-pool i.i.d. rate** (512 MB pool = 10922 × 48 KB experts, 48 touches/token): kernel-pure **7.45 GB/s t1 → 17.0 GB/s t6 (×2.29)** = 2.88 µs/expert. Against the engine-integrated 4.2 GB/s (543 µs/token, 64.0b), the ~3.9× gap decomposes as **~8.4 µs/expert of overhead around the kernel** (index gather / dispatch / dequant / dReLU / combine) — an engineering-addressable cost, not a bandwidth wall. Corollaries: (i) the expert path is NOT inherently thread-flat — the engine's ×1.06 was overhead-dominated, the kernel itself does ×2.29; (ii) i.i.d. gather at 48 KB granularity pays only ~2.5× vs the sequential aggregate ceiling — granularity-bounded, exactly as probe-4 priced it.

## 2. The model (per component, bracketed; every assumption declared)

For a candidate (D, N, L, n_swa, V, E experts × h, top-k), Dn = 2D, 4-bit ternary codes (0.5 B/w), gated experts (3·D·h params):

| component | formula | bracket source |
|---|---|---|
| proj-GEMV | bytes = 11 MB · (D/256)² · (L/6); t = bytes / r(bytes), r from the §1b curve — fully-streamed floor [34-40 GB/s], up to ~135-185 GB/s if resident in aggregate L3 (pollution-capped) | 64.1b measured r(size); grid priced at the streamed floor |
| LUT-MLP | active bytes = top-k · 3Dh · L · 0.5 B; t = bytes / [4.2-17.0 GB/s] | 4.2 = engine-integrated today (overhead-bound, 64.0b); 17.0 = kernel-pure t6 ceiling (64.1b); the ~8.4 µs/expert overhead is the addressable gap |
| scan-recur | t = 46 µs · (Dn·N·L)/(512·96·6) | elementwise, near-linear thread scaling (×3.66) |
| SWA | t = 65 µs · (D/256) · n_swa | window locked at 128 |
| head | bytes = 4·D·V; t = bytes / [40-100 GB/s] | small; hot at these sizes |

Assumptions: t6 operating point; decode hygiene locked; recall-tier query cost (~18 µs two-stage, Phase 55) is noise at this scale; n-gram table = RAM latency-class (not budgeted here). **Sanity: the anchor config reproduces dense-t6 = predicted 980-1707 tok/s vs measured 1652 — inside the bracket.**

## 3. The candidate grid

| cand | dims | total params | expert pool | resident fp32 | active tern | **predicted tok/s (t6)** |
|---|---|---|---|---|---|---|
| S1 | D256 N96 L8 V2048, E128×h128 top8 | **105M** | 0.05 GB | 16.8 MB | 3.1 MB | **739-1309** |
| S2 | D256 N96 L8 V2048, E256×h128 top8 | **206M** | 0.10 GB | 16.8 MB | 3.1 MB | **739-1309** |
| M1 | D320 N128 L10 V4096, E256×h160 top8 | **402M** | 0.20 GB | 33.9 MB | 6.1 MB | **380-686** |
| M2 | D384 N128 L12 V4096, E256×h192 top8 | 694M | 0.34 GB | 55.8 MB | 10.6 MB | 223-399 |
| L1 | D512 N128 L12 V8192, E384×h256 top8 | 1.8B | 0.91 GB | 105 MB | 18.9 MB | 126-231 |

S1↔S2: doubling total capacity (E) at fixed active is **speed-free** (same row) — the granular-MoE thesis in numbers.

**64.1b re-pricing note:** the grid above was priced at the fully-streamed proj floor and the pre-64.1b LUT bracket; every 64.1b measurement moves an edge neutrally-to-upward (proj floor 35→34 is noise; S1/S2 proj at 16.8 MB could run resident at ≫40 GB/s if residency survives expert pollution; LUT ceiling 11.4→17.0). The grid therefore stands as a **conservative floor** — not recomputed, to keep the model's error bars honest rather than stacking best cases.

## 4. The walls, checked — and the CONSTRAINT INVERSION (the 64.1 verdict)

- **P1 speed (floor 10, band 20-50): SLACK by 12-130×.** Even the illustrative 1.8B design predicts ≥126 tok/s. *Speed ceases to be a design constraint in the trainable region.* This is the bandwidth-equation multipliers compounding as designed: active-per-token is MBs, not hundreds of MBs.
- **P2 footprint (≤16 GB): SLACK by ~20-300×.** Pools are 0.05-0.91 GB; 16 GB would hold ~32B ternary params — absurdly beyond anything trainable on P3.
- **W1 resident (proj+head fp32 vs L3):** shapes D·L mildly — S-class fits one CCX (16.8 MB), M1 (34 MB) slightly exceeds aggregate 32 MB → partial spill, priced at ~40 GB/s = already inside the bracket (the t6 proj insight; 64.1b measures 45-56 GB/s at 32-48 MB — better than priced). Not a hard wall anymore; a slope, now measured point by point (§1b).
- **P3 training (~100-500M practical on 2×T4): THE BINDING WALL.** S1/S2 comfortable; M1 at the edge; M2+ out.

**Verdict: the constraints INVERT at this architecture's frugality. Speed and footprint — the walls the whole project was built to attack — are solved with room to spare in the trainable region. The binding constraints are (1) the training budget (P3), (2) DATA (licensed code tokens at scale), and (3) quality-per-param design. Phase 64.2 therefore optimizes QUALITY under P3, not tok/s under P1.**

## 5. Consequences for the 64.2 decision list (postures, criteria to be pre-registered there)

- **Block-verify: OUT for the first scale-up model** (speed slack removes its purpose; C/T remains unfavorable; routed pools don't amortize). Chassis stays banked.
- **Threads: IN** (measured; also the spill-tolerance of proj depends on aggregate BW).
- **MoE config: the capacity lever** — E as large as training allows at fixed active (S2-pattern); granularity per probe-4.
- **Recall tier: promoted to load-bearing** — capacity *without* params is exactly what the P3 wall rewards; the 128K-context product needs it; it is the one component never integrated (the real 64.3 candidate).
- **Depth-reuse (K2): same logic** — depth without resident bytes; sandbox probe if the spec wants L-equivalent >10.
- **Finding-7 predictability/SKIP: DEFERRED** — it saves streamed bandwidth, which is no longer scarce for v1. Re-enters at L1-class sizes or constrained silicon.
- **Vocab: 2048-4096 per-domain code BPE** — head stays hot at these sizes; probe-5 not needed for v1.

## 6. Open unknowns feeding 64.3/64.4

1. **CPU-cheap (Builder, no GPU): DONE — 64.1b registered (HEAD f4a53cf), folded into §1b/§2.** (a) spill slope + aggregate-L3 row-partition hypothesis CONFIRMED (t6 ~185 GB/s resident, declining slope to ~34-36 streamed); (b) the LUT bracket decomposed: kernel-pure 7.45-17.0 GB/s vs engine-integrated 4.2 + ~8.4 µs/expert overhead → a design lever for the 64.2 MoE config (granularity stays quality-driven per probe-4 and the inversion verdict; the overhead is engineering, attackable in the engine without touching the architecture).
2. **Quality-side (GPU, the real question):** what BPB does an S1/S2-class student deliver on code under QAD? → this is the 64.4 pilot, not a sandbox probe.
3. **Data:** licensed code tokens at 10-30B scale (The Stack-class subsets, license filters) + teacher shortlist → the Researcher brief (64.4).
