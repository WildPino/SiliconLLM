# Adjudication — `joint_ternary_block_dossier.pdf`

**Artefact under review:** `docs/research/Joint_TernaryBlock_LayerWise_Reconstruction_for_100B_Donor_Models/joint_ternary_block_dossier.pdf` (27 pp.)
**Adjudicated:** 2026-08-21 · repo revision `e410243`
**Mandate:** *"leggitelo (non dare tutto per giusto quello che trovi lì dentro)"*
**Arithmetic:** every numeric claim below was recomputed; the script is `benchmarks/donor_adaptation/adjudicate_dossier.py`.

---

## 0. Verdict in one paragraph

**The frame is PROMOTED. The numbers are REJECTED.**

The dossier correctly identifies and formalises the one genuinely novel object in this programme: a *simultaneous* projection onto ternary ∩ contiguous-block ∩ probe-predictable, solved against the layer-wise reconstruction objective, rather than the sequential prune-then-quantise pipeline the literature actually runs. Its §2–§3 (the manifold definition, the ADMM splitting, the row-separable Cholesky solve, the closed-form per-block scale) are a usable specification and I would build from them.

Everything downstream of §4 is unsafe. **Part B's error model is internally inconsistent and its central inequality has the sparsity variable inverted.** **Part C's drift theorem places its assumption on the wrong operator and its headline arithmetic is off by 100×.** **Part D's cost model is built on a model that is 9.4B parameters, not 100B, on one weight matrix per layer instead of seven, and on a datatype the T4 does not have.**

Nothing in this document may enter a decision as a quantity. The algorithm may enter as a design.

> ### ⚠ RETRACTION (same day, before this document was acted on)
>
> **v1 of this adjudication claimed the "~14 GPU-hours" headline was "~30× short, at ~413 h". That claim
> is WITHDRAWN. It was wrong, and it was wrong in the direction that favoured my own conclusion.**
>
> I derived 413 h by calibrating on *the dossier's own implied throughput* — its 1.7e12-FLOP
> back-substitution in 26 s, i.e. **65 GFLOP/s, which is 0.10% of the T4's fp16 peak and 0.80% of its
> fp32 peak.** I took an absurd number out of the document I was auditing and propagated it as if it were
> a measurement. That is the identical failure I accuse the dossier of in §6.
>
> Recosted against three independent anchors (§2.3b), the true figure is **~15–40 h**. The dossier's 14 h
> sits at the optimistic edge and **is not refuted**. What survives is that it gets there by two errors
> that cancel: it understates the *work* by ~173× (9.4B geometry, one matrix per layer) and understates
> the *throughput* by ~100×. **A right answer from two wrong inputs is still not a costable model** — but
> it is a far weaker charge than the one I made, and the affordability of the programme is no longer in
> question on this axis.

---

## 1. Register

| # | Claim | Verdict |
|---|---|---|
| C1 | Joint manifold `M = T_ternary ∩ T_block ∩ T_probe`; Problem (2) as the layer-wise objective | **PROMOTED** |
| C2 | ADMM splitting with three variables; row-separable `W'`-update via one Cholesky of `H+ρ_Σ I` per layer | **PROMOTED** |
| C3 | Lemma 3.4 — closed-form entry-wise rounding at ±S_k/2 under diagonal `H_kk`, with Remark 3.5 conceding the general case is NP-hard | **PROMOTED** (honest, correctly caveated) |
| C4 | Hessian-weighted block energy `E_k` + greedy top-K as the block-selection rule | **PROMOTED as a heuristic** — no optimality claim is made and none should be read in |
| C5 | Theorem 3.3 ergodic convergence under RSC + Kurdyka–Łojasiewicz | **NEEDS-VERIFICATION** — structure is standard; Assumption 3.1 (RSC of `H`) is contradicted by the dossier's own §7.1 |
| C6 | Restatement of the D2 falsification (weight-space rotations dead) | **PROMOTED** — matches `DENSITY_PROBES.md` |
| C7 | ρ-law: ≥48 KB contiguous, ≥21.3 neurons, `B_block ≥ 22` | **PROMOTED as our constant, REJECTED as a donor constant** — see §3 |
| C8 | Theorem 4.4 unstructured error bound `E ≤ (1−s)/λ_r · Tr(WHWᵀ)` | **REJECTED — inverted in s** |
| C9 | Theorem 4.5 2:4 gives "a factor of 2 inflation" | **REJECTED — its own algebra gives 4×** |
| C10 | Theorem 4.6 + Table 1 (1.6×–3.8× inflation, s-dependent) | **REJECTED — and now MEASURED-FALSIFIED (§2.1b), in magnitude *and* direction.** Not derivable from Thm 4.6, and the real curve runs the other way. |
| C11 | Table 1 column "C-engine skippable fraction ≈ 0.001 for unstructured at every s" | **REJECTED — a measured point of ours, flattened into a constant.** ⚠ And my own gloss that "unstructured buys this engine nothing" is now **partly withdrawn**: that holds at D=1536, not at donor width (§3). |
| C12 | Theorem 5.1 naive drift grows linearly in L | **REJECTED — assumption on the wrong operator; bound does not follow** |
| C13 | Theorem 5.2 healed drift `≤ 0.45`, "~0.3 perplexity-point degradation" | **REJECTED — 100× arithmetic error; and the PPL bridge is underived** |
| C14 | Tier 3 re-ternarisation at `B_block = 11`, "which the C engine supports via a special-cased 11-neuron `pshufb` mask" | **REJECTED — fabricated capability of our own artefact, and ρ-illegal** |
| C15 | `T_probe` constraint / probe matrix `A` | **REJECTED as specified — the engine has no probe** |
| C16 | `M = N = 4096` "typical for a 100B donor", L=80 | **REJECTED — that geometry is 9.4B** |
| C17 | Per-layer ~30 s, full reconstruction ~14 GPU-h on one T4 | **NOT REFUTED — my rejection is retracted.** Three independent anchors give ~15–40 h (§2.3b). The *derivation* is still rejected: right answer, two cancelling errors. |
| C18 | "TF32" on the T4 (§6.1, §6.2, §6.6) | **REJECTED — T4 is Turing; TF32 is Ampere+. Contradicts a fact this project has already banked.** |
| C19 | "Memory traffic is 67 MB … well within the T4's L2 cache" | **REJECTED — T4 L2 is 4 MB** |
| C20 | Calibration forward pass of a 100B donor, ~30 min on one T4 | **REJECTED — ≥7.2 h at 100% MFU, and 200 GB of weights vs 16 GB VRAM is unmodelled** |
| C21 | §6.9 "~670 MB … a 150× compression from the FP16 donor" | **REJECTED — the dossier's own two numbers give 4×** |
| C22 | Per-layer activation storage | **REJECTED — §5.6 says ~2 GB/layer, §6.1 says 33.6 MB/layer. 60× contradiction inside one document.** |
| C23 | `K_active = K/2 = 93` (§6.4) under a stated `s = 0.80` (Table 3) | **REJECTED — s=0.80 gives K_active = 37; §6.4 silently assumes s=0.5** |
| C24 | Reference [10]: "Z. Dong et al., *MOHAWK: Quantization of Mamba-like Models via Hadamard Transforms*, arXiv:2407.03579" | **REJECTED — FABRICATED CITATION.** Verified: that arXiv ID is *"A connection between Lipschitz and Kazhdan constants for groups of homeomorphisms of the real line"*, I. Vergara, pure group theory. Real MOHAWK is Bick, Li, Xing, Kolter, Gu, *"Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models"*, arXiv:2408.10189 (verified: names MOHAWK, Phi-Mamba, 3B tokens). |
| C25 | Reference [9] Lattimore & Szepesvári *Bandit Algorithms* as the source of the `log B/B` "standard coarsening penalty" | **REJECTED — misattribution.** Combinatorial-bandit log factors are regret bounds. The relevant literature is block-thresholding in wavelet denoising. |
| C26 | "~13 GPU-hours" quoted in §1 and reconciled in §5.4 as external corroboration of the 14 h figure | **REJECTED as circular** — 13 h is *my own* unverified estimate from this session, marked `[DA VERIFICARE]`. The dossier cites it back as independent support. |
| C27 | "1000× cheaper than end-to-end" | **NEEDS-VERIFICATION** — not derived anywhere in the document. My own figures were ~1536× *supervision density* and ~320× *budget ratio*; neither is this. |

---

## 2. The three defects that would have cost us real GPU time

### 2.1 Part B's error model is inverted and does not close

Theorem 4.4 defines `s` as the fraction of entries **zeroed**, then bounds

> `E_unstruct(s) ≤ (1−s)/λ_r · Tr(WHWᵀ)`

At `s = 0` (nothing pruned) this returns the **full** energy — maximum error for an untouched matrix. At `s = 1` (everything pruned) it returns **zero** error. The text draws the conclusion out loud: *"doubling the sparsity (halving 1−s) halves the reconstruction error."* That is the statement that pruning more improves fidelity. The variable is inverted throughout §4.

It does not self-repair downstream:

- **Theorem 4.5 contradicts Theorem 4.4.** Unstructured at s=0.5 gives `0.5/λ_r`; 2:4 gives `2/λ_r`. That ratio is **4×**. The theorem's own text, and Table 1, say 2×.
- **Table 1 is not derivable from Theorem 4.6.** Formula (23)'s multiplicative factor is `(1 + log B/B)(1 + η/(1−η))`, which at `B=22, η=0.2` evaluates to **1.43×** (1.50× if the log is base 2) — and, decisively, it **contains no `s` at all**. Table 1's column runs 1.6 → 1.6 → 2.4 → 3.8 across s = 0.50 → 0.95. That s-dependence cannot come from (23), and the `ΔQ(S_k)` additive term is not s-dependent in Lemma 3.4 either. The headline table is asserted, not derived.
- **`log B/B` is not a coarsening penalty.** It is 0 at `B=1` (correct: unstructured), rises to a maximum at `B = e`, and returns to 0 as `B → ∞`. It claims infinitely coarse blocks are free.

**Consequence.** §4.5's "Optimal Operating Point" recommends `s = 0.80` by minimising a quality–speedup product built entirely out of these numbers. That recommendation has no support.

### 2.1b Measured — probe D1 (`results/d1_pruning.json`, 73 records, four controls re-fired)

The algebra above said Table 1 could not be derived. The measurement says it is also wrong. Coarsening
penalty = Δ(block) / Δ(unstructured) at equal sparsity, at our engine-legal block of 64 units:

| s | **measured** | dossier Table 1 |
|---|---|---|
| 0.25 | **17.4× – 826×** | — |
| 0.50 | **7.9× – 135×** | 1.6× |
| 0.75 | 2.4× – 10.3× | — |
| 0.90 | **0.9× – 3.1×** | 2.4× |

**The measured penalty falls monotonically with sparsity; Table 1 rises.** The sign of the trend is
wrong. This is the empirical counterpart of the algebraic defect: formula (23)'s factor contains no `s`,
and a constant can only match a falling curve at one point — which is exactly what happens, at s = 0.90,
where the measured 0.9–3.1× brackets the claimed 2.4×. **Right at one point by coincidence, wrong
everywhere else.**

**But read the absolute numbers before concluding anything in our favour.** Every block-structured point
at every organ and every level is catastrophic against σ_seed = 0.005 — the mildest is +0.132 BPB (26σ),
the worst +4.8. And critically, **this does not refute the solver**: these are one-shot *magnitude* masks
with **no reconstruction**, which is the floor a Hessian-weighted solve exists to beat. D1 measures the
baseline, not the proposal. Two caveats bound it further: the sweep runs at our `D = 1536` where the legal
block is 64 units, versus 12 at donor width (D0b), so it is an **upper bound** on donor-scale cost; and
two points (`k_proj` and `v_proj` at 90%) were **struck** because with only 4 blocks per layer the mask
quantised to 100% — they were bit-identical to the 100% planted control. The probe caught this only
because it records the sparsity it *achieved* alongside the one it was *asked for*.

**We still do not know the right operating sparsity.** What we now know is that it cannot be read off
this dossier.

### 2.2 Part C's drift theorem is stated on the wrong operator

Equation (26) is `δ_{ℓ+1} = (I + J_ℓ) δ_ℓ + Δ_ℓ X_ℓ`. The assumption imposed is `‖J_ℓ‖ ≤ 1 − β`. But the operator propagating the drift is `I + J_ℓ`, whose norm is then bounded by `2 − β` — giving growth up to `(2−β)^L`, **exponential**, not linear. Bound (27) is the sum of a geometric series with ratio `1−β`, which requires `‖I + J_ℓ‖ ≤ 1 − β`, i.e. a *contracting* residual stream — which is not what a residual stream is. Separately, even taking (27) at face value, `(1−(1−β)^L)/β ≤ 1/β` is **bounded in L**; "linear in L" holds only in the limit `β → 0`, which the document does not take.

Theorem 5.2 then reports `√L·σ_probe + αL²σ²_probe + O(...) = 0.09 + 0.32 + 0.04 = 0.45`. With its own stated constants (`σ_probe = 0.01, α = 0.005, L = 80`), the middle term is `0.005 × 6400 × 0.0001 = 0.0032`, **not 0.32 — a factor of 100.** The corrected sum is 0.13.

This matters operationally: **28% of the proposed 90 h/week budget (Tiers 2 and 3, 25 h/week) is allocated to suppressing a growth law that has not been established.** Before spending it we would need to measure whether naive sequential layer-wise reconstruction actually drifts. That is a cheap experiment and it is now the right one to run.

Note also that the final `‖δ_L‖ → "~0.3 perplexity points"` step has no derivation. This is the same class of unbacked cross-domain bridge that was already struck out of the stage-2a criterion.

### 2.3 Part D is costed on a 9.4B model, on a datatype the T4 does not have

§3.5 fixes `M = N = 4096` as *"typical for SwiGLU intermediate dimension in a 100B donor"* and §5.4 fixes `L = 80`. That geometry is:

```
per layer: attn 4·4096² = 67.1M  +  SwiGLU 3·4096² = 50.3M  =  117.4M
× 80 layers                                                  =   9.40 B
```

**9.4B, not 100B — short by 10.6×.** A real 100B-class donor needs `d_model ≈ 8192, ffn ≈ 28672` (Llama-3-70B geometry gives 77.8B at L=80; `d=9216, ffn=32768, L=88` gives 109.6B).

Worse, Table 3 costs **one matrix per layer**. A transformer layer has seven (q, k, v, o, gate, up, down), each with its own `(M, N)` and each requiring its own Hessian, Cholesky and ADMM loop — and `down_proj` has `N = 28672`, so its Cholesky alone is `28672³/3 = 7.9e12` FLOPs, 340× the dossier's per-layer Cholesky.

Recomputing the **work** at real geometry: **1.21e15 FLOPs per layer against the dossier's 7.0e12 — a factor of 173**, from the 10.6× parameter shortfall compounded with the 7-matrices-not-1 error.

### 2.3b What that actually costs — and the retraction

**This is where v1 of this document went wrong.** I converted 173× more work into wall-clock using the
dossier's *own* implied throughput (1.7e12 FLOP back-substitution in 26 s ⇒ **65 GFLOP/s**) and reported
413 h. But 65 GFLOP/s is **0.10% of the T4's fp16 peak and 0.80% of its fp32 peak** — a rate no real
cuBLAS `trsm`/`gemm` path produces. I lifted an implausible constant out of the artefact under audit and
propagated it. **Retracted.**

Recosted honestly. Total work for the full joint solve is 9.72e16 FLOPs (7 matrices × 80 layers × 50
iterations), and three anchors bracket it:

| Anchor | Basis | Full 100B reconstruction on one T4 |
|---|---|---|
| Direct FLOP model | 9.72e16 FLOPs at 1–8 TFLOP/s sustained | **3–27 h** |
| SparseGPT wall-clock | ~4 h for 175B on one A100, single pass `[A, §4]`; scaled to 100B, A100→T4 ×2.4–4.8, ×2.5 for 50 ADMM iterations | **14–27 h** |
| ADMM-Q measured overhead | 117.73 min vs GPTQ's 33.97 min at 32B = **3.47× a single-pass method** `[T]` | **19–38 h** |

**Consensus: ~15–40 h. The dossier's 14 h is at the optimistic edge and is not refuted.**

Two corrections to my own reasoning that produced this. First, an ADMM iteration is **not** a SparseGPT
pass: once the Cholesky `L` is cached (which Algorithm 1 line 3 correctly does, "paid once"), a row-wise
triangular back-substitution is `O(M·N²)` — the *same* order as multiplying by a cached inverse. The
published ADMM-pruning work (arXiv:2401.02938) confirms this by choosing 20 outer iterations explicitly
*"to provide a similar pruning overhead as SparseGPT"* `[T]`. So `T_outer = 50` costs roughly **2.5×** a
single pass, not 50×. Second, my "the 50× is not accounted for anywhere in Part D" was simply false —
Table 3 step 6 does multiply by `T_outer = 50`.

What survives, and it is still a real defect: **the dossier reaches a defensible number by two errors that
cancel.** It understates the work by ~173× and the throughput by ~100×. A cost model that is right by
cancellation cannot be used to plan — change any one input (a wider donor, fewer iterations, a faster GPU)
and it stops being right. The number must be re-derived, not inherited. But **affordability is no longer
the objection to this programme**, and I should not have said it was.

On top of this:

- **The T4 has no TF32.** T4 is Turing (SM 7.5); TF32 is Ampere (SM 8.0+). §6.1 runs "the donor in FP16 with TF32 attention"; §6.2 completes a Cholesky "in ~0.5 s on the T4 with TF32"; §6.6 lists "TF32 8 TFLOPS" in the T4's spec line. **This project has already banked `T4 = fp16 (Turing)` as a measured operational fact.** A dossier that contradicts our own banked hardware constants on the hardware we are about to spend 90 h/week on is not costable.
- **"67 MB … well within the T4's L2 cache."** The T4's L2 is 4 MB.
- **The calibration pass.** 4096 sequences × 2048 tokens = 8.4M positions × 2·100e9 FLOPs/token = 1.68e18 FLOPs → **7.2 h at 100% MFU** on a 65-TFLOPS T4, ~24 h at a realistic 30%. The dossier says 30 minutes. And a 100B donor in fp16 is 200 GB against 16 GB of VRAM — the offload streaming that implies is not modelled at all.
- **`B_tokens = 4096`** is specified as *sequences*, but `X ∈ R^{N×4096}` has 4096 *columns*. Read literally the document discards 2047 of every 2048 token positions, and then §6.8 reassures us that `B_tokens = 4096 ≥ N = 4096` makes rank deficiency rare — at exact equality, with a generically atrocious condition number. SparseGPT uses 128 × 2048 = 262k positions.
- **§6.9's "150× compression"** is contradicted by its own arithmetic three lines earlier: 670 MB of ternary against 2.68 GB of fp16 is **4.0×** (20× if only active blocks at s=0.8 are stored). The 0.5 bytes/weight figure it uses is, to its credit, *correct* for our engine — `engine.c` packs two trits per byte via `(w0+1)*3+(w1+1)`.
- **§5.6 vs §6.1** give per-layer activation storage as ~2 GB and 33.6 MB. Both cannot be true.

---

## 3. Where the dossier transplanted our constants without re-deriving them

`B_block ≥ 22` is *our* number, measured at *our* `D = 1536`. Its derivation is layout-dependent: bytes per neuron = 3 organs × D × 0.5 bytes = `1.5·D`, so neurons per 48 KB = `32768/D`.

**This is now MEASURED, not analytic** — probe D0b, `results/d0b_rho_floor_donor.json`, revision `e410243`,
all controls fired (reproduces the D0 artefact at D=1536; a by-hand D=1024 case; planted runs of 4 at the
donor's `d_ffn=28672` measured back as exactly 4.0; an i.i.d. mask returning 1.1110 against an analytic
1.1111).

| D | bytes/neuron (interleaved) | neurons per 48 KB | per-organ |
|---|---|---|---|
| 1536 (our engine) | 2304 | **21.33** ← the D0 value that was transplanted | 64 |
| 2048 | 3072 | 16.00 | 48 |
| 4096 | 6144 | 8.00 | 24 |
| 5120 | 7680 | 6.40 | 19.2 |
| 8192 (100B donor) | 12288 | **4.00** | **12** |

**Verdict: the correct constant at donor width is 4.00, not 22 — a factor of 5.33.** Every quantity in
Part B evaluated at `B = 22` moves.

Two things the probe settled that I had left open:

- **The 3-organ interleave DOES survive donor asymmetry.** I had flagged this as doubtful because gate/up
  are `ffn × d` while down is `d × ffn`. It survives for a reason worth writing down: **`d_ffn` cancels out
  of the formula entirely.** The contiguous unit is `d_model`-long for all three organs — gate/up's row
  length and down's column length are both `d_model` — so the FFN expansion ratio (3.5× here) never enters.
  All seven donor matrices land on the same 4096 bytes/unit, 12 units per 48 KB. The one caveat is
  pre-existing and scale-independent: `down_proj`'s "column" is contiguous only if it is stored transposed,
  which is already an assumption of our layout at D=1536 and is not something donor scale introduces.
- **⚠ It also corrects something *I* asserted.** In C11 I wrote that the dossier's skippable column was
  wrong but that "the *conclusion* — unstructured sparsity buys this engine nothing — survives and is
  independently ours." **That conclusion is a D=1536 statement, not a universal one.** Measured on a
  synthetic i.i.d. mask at 90% sparsity, at *donor* granularity: block-skippable = **0.2824** at the
  per-organ floor (bs=12) and **0.6563** at the interleaved floor (bs=4) — against ~0.001 at our engine's
  block-64. The reason is the same arithmetic as above: at donor width each neuron costs 5.3× more bytes,
  so 48 KB spans far *fewer* neurons, and the contiguity requirement becomes correspondingly easy to meet
  by accident. **At donor scale, unstructured sparsity is not obviously worthless.** That reopens a
  question this project had treated as closed. Flagged honestly by the probe as a synthetic i.i.d. mask,
  **not** a real donor co-activation pattern — it isolates the block-size effect on the instrument's own
  accounting and nothing more. What a real donor's pattern does is D1's job.

Two further transplants:

- **Table 1's "skippable ≈ 0.001".** That is our D0 i.i.d. control value at one density, reused as a constant across s = 0.50…0.95. For i.i.d. weights at block-64 the true values are 5.4e-20, 6.3e-07, 1.2e-03, 3.8e-02 — the number happens to be right only near s = 0.90. The *conclusion* (unstructured sparsity buys this engine nothing) survives and is independently ours; the column does not.
- **`T_probe`, the probe-alignment constraint.** The engine has **no probe mechanism**. Our 86–92% in-place predictability is a research-side ridge probe from Phase 58 that was never built into `engine.c`. §7.4 concedes the L-BFGS surrogate may not match "the true probe-alignment objective of the C engine (a deterministic `pshufb` mask)" — but there is no such objective to match, because there is no probe. §5.6 goes further and asserts the engine "supports [B_block=11] via a special-cased 11-neuron `pshufb` mask". It does not. And 11 neurons × 2304 B = 25.3 KB, **below the 48 KB ρ-floor** — the document waves this through with "still ρ-safe … when co-loaded with the probe mask", an assertion against a mechanism that does not exist, overriding a law we measured.

---

## 4. What I am taking from it

1. **The joint formulation (§2–§3) is adopted as the design.** It is the right object and it is the part the literature does not have. Build from these sections.
2. **Two of its failure modes are real and worth keeping:** §7.1 rank-deficient Hessian (detect via `rank(H)/N < 0.5`) and §7.2 pathological inter-block coupling (`η > 0.4`, re-partition by spectral clustering of H's leading eigenvectors). Both are cheap to instrument.
3. ~~`T_outer` is the cost driver and the first thing to attack.~~ **Downgraded with the retraction.** With `L` cached, 50 outer iterations cost ~2.5× a single pass, not 50×. Cutting `T_outer` to 5 saves perhaps 8 h out of 15–40, not 370 out of 413. Worth measuring, no longer the top priority.
4. **The central premise is CONFIRMED by the literature and it is the most valuable thing here.** See §4b: no published work solves ternary quantization and contiguous-block structured sparsity *jointly* on a layer-wise objective. The gap the dossier claims to fill is real.
5. **Nothing quantitative.** No number from this dossier is quoted forward.

## 4b. Independent literature verification (returned after v1 of this document)

Every figure below was read in the cited paper's own table `[T]` or its own prose `[A]`; `[X]` means the
paper does not state it and no number was substituted.

- **The joint gap is REAL.** The literature does joint prune+quantize at INT precision with non-block
  sparsity (SparseGPT §3.5; Optimal Brain Restoration, arXiv:2509.11177 at W4A4KV4 + 50%), and it does
  ternary-only with no structured sparsity (PT²-LLM, PTQTP, CAT-Q), and it does ADMM for one *or* the
  other alone (ADMM-Q: W4/W3/W2, quantization only, 300 iterations; admm-pruning arXiv:2401.02938:
  pruning only, up to LLaMA-2-70B). **No work combines ternary with contiguous-block sparsity as
  simultaneous constraints.** Every adjacent cell of the grid is populated except this one — that is a
  gap, not a search miss. **C1 is thereby strengthened, not merely allowed.**
- **SparseGPT** `[A, §4]`: *"under 4.5 hours"* for OPT-175B and BLOOM-176B combined, *"approximately 4
  hours"* for the 175B alone, on **one A100-80GB**, calibrating on **128 segments of 2048 tokens** from C4.
  Single sequential column pass, no outer-iteration loop. There is **no runtime table** in the paper —
  both figures are prose only.
- **PT²-LLM** (arXiv:2510.03267), Table 1: LLaMA-2-7B FP16 **5.47** → PT²-LLM **11.56** = **2.113×**;
  LLaMA-3-8B FP16 **6.14** → **32.19** = **5.244×**. `[T]` **Our banked 2.11× / 5.24× are CONFIRMED.**
- **PTQTP** (arXiv:2509.16989) is a *different, real* paper, Table 9: LLaMA-2-7B **5.47 → 6.32 = 1.155×**;
  LLaMA-2-70B **3.32 → 3.95 = 1.190×**; LLaMA-3-8B **6.14 → 8.51 = 1.386×**. `[T]`
  **⚠ Do not read this as "ternary PTQ is nearly lossless".** PTQTP uses **dual ternary trit-planes** —
  two planes plus scales, carrying materially more than 1.58 bits per weight despite the label. PTQTP and
  PT²-LLM share identical FP16 baselines (5.47, 6.14) yet disagree 1.15× vs 2.11×; they are **not
  apples-to-apples** and neither may be cited as "the" ternary-PTQ cost without this caveat.
- **MOHAWK** (arXiv:2408.10189), Table 1: Phi-1.5-1.3B **64.9** vs Phi-Mamba-1.5B **62.6** `[T]` —
  **our 62.6/64.9 confirmed**, on 3.0B tokens (80M stage 1 + 160M stage 2 + 2.76B stage 3) `[A]`.
  **GPU-hours: `[X]` — the paper never states the distillation's hardware cost.** Any budget figure for
  MOHAWK-style healing in our documents is therefore ours, not theirs, and must be tagged as such.
- **T4 / TF32**: confirmed absent from NVIDIA's own T4 product page, which lists FP32, FP16/FP32 mixed,
  INT8 and INT4 and no TF32; TF32 arrives with Ampere's third-generation Tensor Cores `[A, primary]`.
  **C18 stands.** T4 L2 = 4 MB is consistent secondary sourcing only, **not** primary-table-read — but
  the directional conclusion in C19 holds under any plausible value, since 67 MB fits in neither 4 MB nor
  any L2 the T4 could have.

**Process note worth keeping.** The researcher's *first* pass on PT²-LLM returned FP16 5.09 → 9.11 =
"1.79×" and reported our banked figure as wrong. That pair is the **L-13B row**, mis-paired with the
LLaMA-2-7B label. It was caught only by forcing a verbatim transcription of the whole table. This is the
exact baseline-relabelling trap that produced the original fabrication, reproduced on the same paper by a
different agent on a different day. **The defence that worked was the same one: read the raw table, not a
summary of it.**

## 5. What must be re-derived before any GPU hour is spent

| # | Item | Why |
|---|---|---|
| R1 | Cost model at **real** donor geometry, **per matrix**, with a *measured* T4 throughput for `trsm`/`gemm` at those shapes | Downgraded from blocking to hygiene by the retraction: the answer is ~15–40 h either way. Needed because the dossier's number is right only by cancellation |
| R2 | Does naive sequential layer-wise reconstruction actually drift? | Tiers 2+3 cost 25 h/week to fix a growth law that is not established |
| ~~R3~~ | ~~The ρ-floor at donor width (D0b)~~ | **CLOSED 2026-08-21: 4.00 at D=8192, ratio 5.33×, interleave survives.** Replaced by: does a *real* donor mask, not an i.i.d. one, reach the 0.28–0.66 skippable that donor granularity now permits? |
| ~~R4~~ | ~~Quality-vs-sparsity measured, not bounded (D1)~~ | **CLOSED 2026-08-21 for the naive baseline** (§2.1b): Table 1 falsified in magnitude and direction. Replaced by: what does the *solver* recover against this floor, and how much milder is the penalty at donor granularity? |
| R5 | Feasibility of the calibration forward pass on available hardware | 100B fp16 = 200 GB vs 16 GB VRAM is not addressed |
| R6 | Whether `T_probe` is a constraint we can even impose | The engine has no probe; either build one or drop the third splitting variable |

---

## 6. Note for the author of the dossier

**First, my own.** v1 of this document rejected the 14-hour headline as "~30× short" on the strength of a
throughput constant I lifted out of the dossier itself — 65 GFLOP/s, a thousandth of the hardware's
capability — without once asking whether that constant was plausible. I then used the result as the
lead charge in the verdict paragraph. The error pointed in the direction that made my critique stronger,
which is precisely when this project's rules say to slow down. The retraction is at the top of the
document rather than buried here, and C17 is marked NOT REFUTED.

With that said: the failure pattern in the dossier is the one this project has a named law for —
**the plausible artefact.** Every defect above is *shaped correctly*. The theorems have the right form, the tables have the right columns, the citations have the right author-count and arXiv-number-shape, and the constants are recognisably ours. The document reads as more rigorous than the memo it was derived from — and it is less so, because it converted marked estimates into asserted facts. Specifically, it took a number I had explicitly flagged `[DA VERIFICARE]` (13 GPU-h) and cited it back as external corroboration for its own independently-derived 14 h; the two figures agree because they are the same guess.

Three requests for any revision:

1. **Tag every number with its provenance** — derived here / measured in artefact X / read in the table of paper Y / estimated. A number with no tag cannot be used.
2. **Do not restate our measured constants without re-deriving them at the new scale.** `B_block ≥ 22`, `skippable ≈ 0.001` and `frac99 ≈ 0.64` are all measurements at specific dimensions in a specific layout.
3. **Do not assert capabilities of `engine.c`.** The probe and the 11-neuron `pshufb` mask do not exist. The engine source is in the repo and is the authority.

## 7. Verification log

- arXiv:2407.03579 fetched directly: *"A connection between Lipschitz and Kazhdan constants for groups of homeomorphisms of the real line"*, I. Vergara. **Not MOHAWK, not quantization, not machine learning.**
- arXiv:2408.10189 fetched directly: Bick, Li, Xing, Kolter, Gu, *"Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models"*; abstract names MOHAWK, Phi-Mamba, and 3B tokens. This is the real reference for the §5 healing analogy.
- All arithmetic in §2–§3 recomputed; scripts in `benchmarks/donor_adaptation/adjudicate_dossier.py` and `benchmarks/donor_adaptation/recost_solver.py` (the latter is the retraction check).
- Literature verification returned and is folded into §4b; it forced the §0 retraction.
- Probe **D0b returned and §3 is now MEASURED**, all controls fired (`results/d0b_rho_floor_donor.json`). It settled the interleave question affirmatively and forced a partial withdrawal of my own C11 gloss.
- **D1 returned**, 44 new sweep records + 4 controls re-fired in-session; §2.1b is measured. Two points struck for mask quantisation, found by cross-checking `zero_frac` against the requested `level`. Full write-up in `DENSITY_PROBES.md` §3.
- References [1] SparseGPT, [2] Wanda, [3] SmoothQuant, [4] SpinQuant, [5] Boyd ADMM, [11] Hassibi & Stork OBS all check out on author/venue/ID. Only [10] is fabricated and only [9] is misattributed.
