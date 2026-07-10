# Phase 64.1 — Budget Document (sizing vs tok/s, the two walls as functions)

**Status: DONE 2026-07-06 (Architect). Inputs: the 64.0 registered tables (HEAD be5f448). Feeds: the 64.2 design decisions. Canonical gate plan: `SCALEUP_ARCHITECTURE.md` §8.**

## 1. Measured anchors (3600X reference, per-protocol, 64.0)

- **DRAM cold-stream:** single-thread 21-26 GB/s; **aggregate ceiling ≈ 40-44 GB/s, saturated at 3 threads** (t6 adds nothing; unpinned t6 regresses). The old "~1.4-1.5×" derived figure is closed: measured ~1.6-1.8×.
- **Compute-floor decomposition (µs/tok, t1→t6):** proj-GEMV 662→272 (×2.43); LUT-MLP dense 313→207 (×1.51), MoE 573→543 (**×1.06 — thread-flat**); scan-recur 169→46 (×3.66); SWA 92→65; head 40→8; norms+glue ~7.
- **Two derived rates (load-bearing):** (i) proj-GEMV at t6 = 11 MB / 272 µs ≈ **40 GB/s = the aggregate bandwidth ceiling** — the fp32 projections already run at aggregate BW when threaded, so an L3 spill costs little extra at t6; (ii) MoE LUT effective rate = 2304 KB / 543 µs ≈ **4.2 GB/s** — far below any bandwidth ceiling: the expert path is gather/kernel-bound, not stream-bound (the thread-flatness is real; the "capped at DRAM ceiling" reading is not — the cap is lower).

## 2. The model (per component, bracketed; every assumption declared)

For a candidate (D, N, L, n_swa, V, E experts × h, top-k), Dn = 2D, 4-bit ternary codes (0.5 B/w), gated experts (3·D·h params):

| component | formula | bracket source |
|---|---|---|
| proj-GEMV | bytes = 11 MB · (D/256)² · (L/6); t = bytes / [35-44 GB/s] | measured: proj runs at aggregate BW at t6; spill-tolerant |
| LUT-MLP | active bytes = top-k · 3Dh · L · 0.5 B; t = bytes / [4.2-11.4 GB/s] | measured MoE-t6 (worst) .. dense-t6 (best) effective rates |
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

## 4. The walls, checked — and the CONSTRAINT INVERSION (the 64.1 verdict)

- **P1 speed (floor 10, band 20-50): SLACK by 12-130×.** Even the illustrative 1.8B design predicts ≥126 tok/s. *Speed ceases to be a design constraint in the trainable region.* This is the bandwidth-equation multipliers compounding as designed: active-per-token is MBs, not hundreds of MBs.
- **P2 footprint (≤16 GB): SLACK by ~20-300×.** Pools are 0.05-0.91 GB; 16 GB would hold ~32B ternary params — absurdly beyond anything trainable on P3.
- **W1 resident (proj+head fp32 vs L3):** shapes D·L mildly — S-class fits one CCX (16.8 MB), M1 (34 MB) slightly exceeds aggregate 32 MB → partial spill, priced at ~40 GB/s = already inside the bracket (the t6 proj insight). Not a hard wall anymore; a slope.
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

1. **CPU-cheap (Builder, no GPU):** (a) proj-GEMV size-sweep with synthetic weights (validates the spill slope + the aggregate-L3 row-partition hypothesis); (b) expert-pool effective rate on a true-DRAM pool ≫ L3 (tightens the [4.2-11.4] bracket — currently the widest in the model).
2. **Quality-side (GPU, the real question):** what BPB does an S1/S2-class student deliver on code under QAD? → this is the 64.4 pilot, not a sandbox probe.
3. **Data:** licensed code tokens at 10-30B scale (The Stack-class subsets, license filters) + teacher shortlist → the Researcher brief (64.4).
