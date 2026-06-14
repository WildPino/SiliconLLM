# SiliconLLM Handoff

Last updated: 2026-06-14 — **Phase 48 CLOSED. 48.A: armB nonlinear-lift substrate solved the five-phase structural instability (word-clean closed-loop, frontier ~2.25 → ~2.18-2.22, char-flood byte residual). 48.B/C/D then closed the STATIC per-step axis with a proof per arm (static product = generic quadratic; random bilinear dynamics = noise; error-tilt = no-op because RFF is rotation-invariant). armB sits exactly at the boundary of what a static read of this substrate can give. Pivot: error is a no-op on features but everything for DYNAMICS → next axis is FORCE/RLS (learned generation attractor, no backprop). Phase 47 closed 2026-06-12.**

## Goal

Build toward a real language model by "pulling an LLM out of the silicon, not imposing an LLM on the silicon."

Core rule: keep the readout simple, falsifiable, and structurally honest. Gains should come from substrate geometry, memory, local plasticity, boundary/homeostatic dynamics, training signals, and robustness properties. Avoid Transformer-like modules, word/bigram counters in policy, generation hacks, and opaque inference-time machinery.

## Promotion Gate V2 (current, frozen)

A candidate must pass **all** components, **worst-case over 32 samples**, at **both T=0.65 and T=0.55**, then survive the **replica protocol** (4 independent full-gate replicas, rule 4/4), then **human reading** of dumped samples (mandatory, permanent component — the user reads them).

Word-level bars (unchanged from V1):

| component | bar |
|---|---|
| BPB (teacher-forced, avg of 3 val windows) | <= 2.2543 |
| topBi (most repeated word bigram) | <= 8 |
| altLp (word alternation A-B-A-B) | <= 2 |
| nameWst | <= 20 |
| runWst (identical consecutive words) | <= 5 |
| selfBPB (closed-loop) | in [0.8, 2.0] |

Byte-level guards (NEW in V2, added 2026-06-12 after a Goodhart discovery — see below). Bars are **calibrated on the corpus**, not chosen: ~1000 contiguous 2 KB windows of TinyStories, bar = max observed + margin (integers +2, fractions +0.03). Current calibrated bars (also in `docs/gatev2_bars.json`, regenerable by `phase47i.ps1` step 1a):

| guard | definition | corpus max | bar |
|---|---|---|---|
| wsRun | longest run of whitespace bytes (space/tab/CR/LF) | 4 | <= 6 |
| chRun | longest run of identical non-whitespace byte | 6 | <= 8 |
| wsFrac | whitespace fraction of sample | 0.2202 | <= 0.2502 |
| nonPrint | bytes outside {9,10,13,32..126} | 133 (unicode in corpus) | <= 135 |

Post-hoc legitimacy rule: after a Goodhart discovery the only legitimate post-hoc gate movement is **tightening** — it can only reject, never promote. Never loosen a bar after seeing results. Inference-side filters/penalties to "clean" samples are cheating and forbidden.

Replica protocol (process upgrade, permanent): a single full-gate pass is only screening. Promotion requires 4 replicas-of-32 (2 with fresh rngs on standard seeds, 2 with original rngs on held-out prompt grids displaced from all train/val windows), each evaluated worst-case at both temperatures. 4/4 PASS = confirmed; one replica failing one component by one unit = marginal (user decides); >=2 failing = lucky tail, not promotable.

## Background (Phases 42-46, compressed)

- C2.A (Phase 43 champion, linear readout, 26 Oja cells): 2.2593 BPB, `weights/phase43c2_C2A.bin`. Failed word-gate (topBi ~19 @T0.55) but — discovered in 47.I — it is the only **byte-clean** model of the project.
- Phase 44: L2 boundary memory. D1 family (`weights/phase44f_F0.bin`) 2.252x BPB near-stable; delta-write champion 2.2212 BPB but generatively collapsed. Conclusion: volatile inference memory creates attractor pressure.
- Phase 45: delta write usefulness is real (+26k bits) but **per-event inseparable** (all signal correlations ~0). Closed. Synthesis in `docs/PHASE44-45_SYNTHESIS.md`.
- Phase 46: L3 phrase memory over D1 is compression-positive (B3 2.2509) but generation-fragile. "Volatile memory at inference" axis closed for good.

## Phase 47 — Static Nonlinear Readout + On-Policy Training (CLOSED 2026-06-12)

### The arc, sub-phase by sub-phase

- **47.0 (feasibility)**: distillation from unstable teachers (delta, L3-B3) into stable [SEE|L2_D1] features. Discovery: **the lever is readout nonlinearity, not the teacher.** A 1-hidden-layer MLP on the same stable features reaches 2.0947 val (vs linear 2.2410); KD-from-delta adds nothing. Delta/L3 had been a tortuous way of injecting nonlinearity already present in SEE features. Linear-readout invariant revised to: static, small, stateless, falsifiable nonlinear readout is admissible.
- **47.A0/A0b (sanity + anchor repair)**: ladder valid (H32 2.2360 / H64 2.1627 / H128 2.0807), randlabel/shuftime controls clean, ablation shows the nonlinear gain lives in **SEE**, not L2. `frozenD1` anchor (D1 weights pushed through the exact probe path == BASE D1 at 4th digit) became the permanent pipeline-integrity check. Closed-loop: all H fail topBi + off-distribution overconfidence.
- **47.B (regularization)**: label smoothing/RMS-penalty/WD/dropout matrix. RMS normalizes scale but topBi stays → **not a calibration problem**. Diagnosis: exposure mismatch.
- **47.C (proxy robustness)**: one-step corruption/D1-burst training improves dirty validation (valC) but does not transfer to closed-loop → the wall is **multi-step autoregressive drift**. Clue C3burst: topBi almost fixed but selfBPB 0.67 (degenerate).
- **47.D (real rollouts — the structural breach)**: DAgger-style training on bursts sampled from the decoder ITSELF (K bytes every 256, target always true byte, +16B recovery, mix 80/20, lamC 0.02 stop-grad). **First configs ever under topBi<=8 at both temps with sane selfBPB** (D16_h32 topBi 5/6). Dose-response: K=8 no re-entry, K=16 sweet spot, K=32 degenerates. Proxy-vs-true at parity: D1-bursts degenerate, own-bursts fix → the lever is on-policy.
- **47.E (capacity)**: recipe at H64. First one-temperature full PASS ever (D16r3_h64 @T0.65, BPB 2.1901) but T0.55 wall (topBi 13). Budget exonerated via D0 controls; rollout cost ~+0.044 capacity-independent.
- **47.F (temperature coverage)**: hypothesis falsified cleanly (Tmix bursts: bi55 13=13 at 50% coverage). Rounds non-monotone at H64 (r3 optimum). H64 closed. Fallback D16r5_h32: topBi 5/5, best near-pass.
- **47.G (last mile H32)**: plain continuation r6-r9. **P_r7 = first dual-temperature full-gate PASS in project history** (`weights/phase47g_P_h32_r7.bin`: BPB 2.2497, topBi 6/7, altLp 2/2). Anneal (mix 10%) rejected: BPB flies, structure lost — late 20% rollout dose is load-bearing. Checkpoint promoted, not recipe.
- **47.H (extended validation)**: replica protocol. P_r7 **not promotable by pre-registered rule** (2/4 replicas fail, both only altLp@0.55 = worst-of-32 on a ~3% per-sample event → 47.G PASS was a tail on that metric). But the structural fix REPLICATED: topBi 6-8, name/run/selfBPB green in 8/8 evaluations incl. held-out prompts. H48 tail closed (capacity ceiling confirmed; stable regime exists only at H32 on this substrate).
- **HUMAN READ → GOODHART DISCOVERY (the decisive event)**: the user read the samples. Three byte-level pathologies invisible to every word-level metric: **whitespace floods** (runs of 10-50+ spaces — separators, zero words, zero signal), the **"wasteland"** (diffuse far-field attractor: template fragments like "..nhat sh"/"..nke go" with surface variation — no single bigram dominates, topBi silent), **char floods** ("Jaaaa...", 30-50 chars = one weird word, runWst silent). Temporal pattern: second half of samples, after ~500-1000 bytes of self-generation — a far-field region K=16 anchored bursts never visit. DAgger cured the near-field; optimization pushed the residue where the proxies don't look. Verdict: **P_r7 definitively not promotable — by Goodhart, not statistics.** Response: Gate V2 tightening (above).
- **47.I (far-field, one shot, hard cap)**:
  - Step 1 re-score under byte-guards: P_r7 fails 27-32/32 samples in every replica (wsRun worst 106-233 vs bar 6 — the wasteland was almost everywhere, not in a tail); **C2.A is byte-clean**; D1 has a slight whitespace drift (wsFrac +0.008-0.011) born with L2 memory. Byte degeneration grows along the readout/training stack. **No model of the project passes the full gate v2.**
  - Step 2a premise probe (no training): K=128 anchored bursts DO enter the wasteland (38.2%/36.9% of burst tails vs K16 control ~9-11%; wsrMax 64 = whole tails of spaces) → coverage hypothesis live.
  - Step 2b far-field training: prefix r1-r5 bit-identical to 47.G (MD5), rounds 6-9 with burst mix K16 87.5% / K128 12.5%. **Result (human-read confirmed at I_r9, T0.55): the whitespace wasteland is genuinely gone** — text stays structured to the last byte, consecutive spaces 3-5, wsRun at corpus level (6 = bar). Word metrics best ever (bi 4-6). Residuals: **char flood persists** (chRun 20-26 vs bar 8 — its channel was visited by only 2.9% of burst entries) and BPB 2.2621 (far-field cost +0.018 vs P at same round, 0.008 over bar). Quota adjustment NOT spent (it targets neither residual); Phase 47 closed per hard cap.

### The coverage theory (validated 3/3, with predictive power)

**What the rollout visits, the decoder learns to exit; what it does not visit, persists.** Channel by channel:

| degeneration channel | training coverage | outcome |
|---|---|---|
| word-level loops (topBi/run/name) | K16 bursts, near-field (47.D-G) | cured and replicated |
| whitespace wasteland | K128 bursts entering it (15% of entries) | cured at corpus level at I_r9 |
| char flood ("aaaa...") | almost never visited (2.9% of entries) | persists |

Corollaries: far-field training did not undo the near-field (best word metrics ever at I_r9); coverage is about **which** states are visited, not how many; designing visitation is a phase-level question, not a knob.

### The validated stack (the actual deliverable)

1. **Static nonlinear readout discovery (47.0)**: stable SEE features contain large nonlinear predictive structure; a small stateless MLP extracts it (H32: 2.236 TF; the gain lives in SEE, not L2).
2. **DAgger on-policy training (47.D)**: cures attractors channel-by-channel according to rollout coverage. Recipe: burst K16 every 256B from the current decoder @T0.65, target always true byte, 16B recovery, mix 80/20 clean/rollout, lamC=0.02 consistency with stop-grad on the clean branch, no label smoothing, per-round checkpoints, ~7 rounds at H32.
3. **Mature evaluation harness**: gate v2 (word + corpus-calibrated byte guards), two temperatures, frozenD1 anchor, MD5 repro pre-checks, prefix-property verification for continuations (seed-formula determinism; on-disk checkpoints lack Adam state — true branching needs in-memory copy), mini→full→replica protocol→**mandatory human reading**. Each level caught what the previous one missed; the last level is the user.
4. **Coverage theory** (above), with predictive power.

### The honest map (where everything stands)

- **I_r9** (`weights/phase47i_I_h32_r9.bin`): the frontier — word + whitespace clean, char-flood residual, BPB 2.2621 (0.008 over bar).
- **P_r7** (`weights/phase47g_P_h32_r7.bin`): historic word-only dual-temp pass; byte-dirty (wasteland).
- **C2.A** (`weights/phase43c2_C2A.bin`): only byte-clean model; word-dirty (topBi 19); linear.
- **D1** (`weights/phase44f_F0.bin`): near-stable linear reference; slight whitespace drift.
- Nobody passes gate v2 in full. Byte degeneration grows along the readout/training stack.
- **Language is not in the gate, it is in the substrate**: confirmed twice by eye — at ~2.25 BPB the "good parts" are flowing word-salad (correct function words, mangled content words, zero narrative coherence). That is what 2.25 bits/byte can say. Readable language lives around ~1.2-1.5 BPB. Even a full gate-v2 pass at ~2.25 BPB will read as structured word-salad: promotion measures **stability**; language is bought only by lowering BPB, i.e. substrate work.

## What Worked (cumulative)

- Multi-timescale L1; feature clamp/homeostasis; small local Oja plasticity (C2.A).
- Deterministic RNG, explicit seeds, MD5 repro pre-checks, fixed aggregation order, single-trainer guard.
- frozenD1 anchor as permanent pipeline-integrity check.
- Static nonlinear readout on stable SEE/D1 features (47.0): the compression discovery of the project.
- DAgger on-policy rollout training (47.D): the stability discovery of the project. K=16 sweet spot near-field; K=128 reaches far-field; lamC=0.02 free calibration insurance.
- Per-round checkpoints + seed-formula prefix property (bit-identical continuations, MD5-verified).
- Three-level evaluation (gate → replicas → human read): every level caught what the one below missed.
- Calibrating gate bars on the corpus instead of choosing them.
- Premise probes before spending training (47.I step 2a: minutes, not hours).

## What Did Not Work — do not repeat without a genuinely new hypothesis

All pre-47 items stand (pooling variants, scalar gains, generation hacks, L2 write controls of every kind, L3 schedule tuning, per-event write filtering — see git history and `docs/PHASE44-45_SYNTHESIS.md`). Phase 47 additions:

- Teacher distillation from delta/B3: the teacher is not the lever (47.0).
- Calibration regularization (label smoothing, RMS penalty, WD) as a stability fix: not the cause (47.B). Label smoothing is toxic to BPB/language.
- One-step corruption / proxy-burst exposure training: improves dirty validation, does not transfer to closed-loop (47.C).
- K>16 near-field bursts (degenerate), rollout mix 30% (blandness without fix), anneal of late rollout dose (structure lost — the dose is load-bearing) (47.D/G).
- Temperature-coverage bursts for the H64 T0.55 wall: falsified cleanly (47.F).
- H64/H48 capacity for stability on this substrate: capacity ceiling, stable regime only at H32 (47.E/F/H).
- Trusting word-level metrics without byte-level guards and human reading: Goodhart (47.H→I).
- Loosening any bar post-hoc, inference-side sample-cleaning filters, promoting "with reserve": forbidden on principle.
- PowerShell: `$R` and `$r` are the same variable (case-insensitive) — caused a silent infinite loop; smoke-execute new harness sections, keep `-SkipTrain` flags.

## Phase 48 — Substrate Feature Classes (armB / RFF kernel) — CLOSED

The Phase 47 close named the substrate as the bottleneck. Phase 48 mapped it, found the first lever that moves it (armB), and then proved the static per-step axis is exhausted — pivoting to generation dynamics (FORCE/RLS).

### 48.0 — mapping (Q0) + three TF-only probes

- **Q0 (read `silicon_entropy.c` + `silicon_v0.c`)**: L0 is 64-D = `[M4 32 (sum of binary codes = PURELY LINEAR) | wave 32 (the only nonlinearity, from saturating wave dynamics)]`. The temporal reservoir (all three L1 bands) reads only `l0_out[0:43]` = 32 linear M4 dims + just **11 of 32** wave dims; the other **21 nonlinear wave dims are never integrated over time** (snapshot only). Integration is pure linear EMA; Oja's inference feature is linear in L0. **The recurrent substrate is a linear (PCA-like) machine over a mostly-linear 43-D input, discarding its own nonlinearity.**
- **TAPS (Sonda 1) — FAIL, informative**: H32 on `[SEE(t) | SEE(t-8) | SEE(t-32) | SEE(t-128) | L2]` (832-D) vs no-tap. Re-exposing past states *worsened* compression (controls clean: shuffle-tap pure noise, only t-8 carried a sliver). The reservoir already integrates recent history into SEE(t) → **closes the entire "memory/history at the readout" family** (and retro-explains 44-46: the info was already in the state; the problem was extracting it, which the MLP does). The lever is what the substrate *computes*, not how much of it is exposed.
- **EXPAND / armB (Sonda 2) — PASS, big and clean**: a fixed random nonlinear lift `z = Ω·L0norm` (Ω Gaussian), `cos(z)`, temporally integrated as new reservoir dims (two EMAs 0.90/0.99), appended to the frozen base. **+0.040/+0.042/+0.042 BPB on all three held-out windows** (armB 2.2023/2.1839/2.1980 vs notap 2.2427/2.2257/2.2395). Controls perfect: the **linear** arm (integrate `z`, no `cos`) *worsened* 0.17-0.21 (raw linear projections = noise, like TAPS); the **shuffle** guard was inert (the gain is temporal integration of the nonlinearity, not capacity). The lone `cos()` isolated = 0.21 BPB. **First time the project moved what the substrate KNOWS, not how well the readout reads it.**

### What armB actually is (the kernel reading)

`cos(Ω·L0)` with Gaussian `Ω` is **Random Fourier Features** (Rahimi-Recht): it approximates a Gaussian kernel. Unplanned, armB turned the linear reservoir into a **kernel machine** — nonlinear similarity in an effectively infinite-dimensional space via a finite random projection. That is why the lone `cos` was worth 0.21 BPB: byte-to-byte prediction is kernel-shaped and the linear reservoir could not see it. **Consequence for scaling: raising `D_EXP` only approximates the *same* kernel better (error ~1/√D, diminishing).** The real lever is different/better **feature classes**, not more of the same.

### 48.A — armB under the frozen 47 harness (CLOSED)

armB-expanded substrate (512-D feature `[D1 base 256 | armB B-bands 256]`) + the 47.I-final DAgger recipe (H32, 9 rounds, K16 + K128 whitespace far-field), generator rebuilds the lift from the `0x53454548` header. Mandatory **closed-loop determinism pre-check** (new substrate path = new generation path) PASSED (MD5 byte-reproducible). Anchor exact.

**Result: the five-phase structural instability is solved.** armB is **word-clean closed-loop** (the attractor-collapse / word-salad wall that 44-47 could not pass) and the **stable frontier dropped from ~2.25 to ~2.18-2.22** (self-BPB ~1.8; budget for the rollout cost lands comfortably under bar where P_r7 was on the edge). Read by eye (samples in `results/phase48a/human_close/`): real TinyStories phrasing emerges ("Once upon a time, there was a", "she was so excited to", "he had fun", "played", "friend") — **measurably less word-salad than the 2.25 era.** The **residual is the char-flood byte channel** (`chRun` fails: "...aaaaaaaa", "huhhhh") = the **fidelity ceiling**, exactly the coverage prediction (the K128 bursts fall into whitespace, never char-flood).

**48.A-fix** (the one allowed adjustment, branch 2): targeted char-flood coverage — rounds 6-9 add char-flood-seeded bursts (force a short repeated-char run into the substrate, then roll out, target true byte) beside the K128 whitespace bursts. Gated by a **mandatory no-training premise** (does the seeded decoder actually stay flooded? smoke stayPct 15.6% → reachable). Built, smoke-validated (premise + determinism pass), separate `phase48afix_*` checkpoints. armB closed regardless per hard cap; **char-flood accepted as the documented byte residual / watch-metric.**

### The deliverable and the bet

armB is the **frozen baseline** that Phase 48.B stacks on: the structural problem of five phases is solved, the frontier is lowered, and the path forward is **adding feature classes**, not scaling parameters. The bet: if BPB marches 2.18 → 2.10 → 2.0 → toward ~1.5 by adding CPU-native, mantra-pure nonlinearity classes, the thesis is proven — **"pulling an LLM out of silicon scales in feature richness, not parameter count."** That is the claim that, if it holds, eventually justifies scale/GPU (a charter prerequisite).

### 48.B / 48.C / 48.D — the static per-step axis, closed with a proof

armB taught us: **the silicon understands similarities (kernel) but not relations.** A relation is a *product* (A·B = "A in the context of B"); the additive EMA reservoir cannot represent it. Three probes asked "make the per-step read of the substrate smarter" from three angles. Each fell — and, unusually, each fell for a stateable mathematical reason. Same EXPAND methodology throughout (base = frozen armB, classes appended, H32 clean readout, 3 held-out windows, mandatory controls, anchor exact, TF-only).

| Probe | Move | Result | Reason |
|---|---|---|---|
| **48.B** | relation as a static feature | flat (~0.01, none clears 0.015×3) | static product = generic quadratic — `BILIN_sp` (shuffled pairing) kept ~85% of the gain, so it's a second moment, not gating |
| **48.C** | relation as random dynamics | **damage** (DYN < LIN_dyn, gate verified live) | no theorem for a random bilinear form → the multiplication injects noise, not signal |
| **48.D** | error as feature selection | flat (MODOJA-K = PARITY = armB) | RFF is **rotation-invariant**: tilting the kernel's feature directions just resamples the *same* kernel |

**48.D detail (MODOJA-K, the first time the error signal touched the substrate).** The fixed `Ω` inside `cos(γ·Ω·L0norm)` became an error-tilted `P`: per-row Oja, learning-rate modulated by frozen-trigram surprise (mean 1), `P` learned on train then frozen (zero val leak), deterministic. Controls at *identical mean-lr*: PARITY (`m≡1`, unsupervised tilt) and SHUF-MOD (modulator time-permuted). One real failure caught and fixed: at η₀=1e-4 plain per-row Oja collapses `P` to ~1 PC (catastrophic — 128 rows → ~2 effective features destroys the kernel); a cheap `--pdiag` mode tuned η₀ at the true 1M-step count → **η₀=1e-7 gives a genuine ~13° tilt with the kernel still diverse (eff_rank 33/64)**; renormalizing each `P_d` to its original `‖Ω_d‖` (not to 1) kept init ≡ armB with no bandwidth confound. So `P` *did* move and the error *did* shape it — beautifully, **MODOJA-K collapses faster than PARITY at every η₀** (error-focusing concentrates on a more dominant PC). The error touched the substrate differentially; it was simply a **no-op on BPB**: `MODOJA-K − armB = +0.0009/+0.0027/+0.0012` (≪0.015), `MODOJA-K − PARITY ≈ 0`. **You cannot improve a kernel machine by rotating its features** (the Gaussian spectral density is rotation-invariant — every draw is an unbiased estimator of the same kernel). To make the error count you'd have to change the kernel's *shape* (anisotropic bandwidth, another family) — still inside the per-step axis just exhausted from three sides.

**The static axis is closed, and the qualitative agrees:** armB's closed-loop is structured word-salad with short cyclic loops and no long-range structure. Quant (BPB plateau), qual (loops), and theorem (RFF) all converge — **the gap to language is not in the per-step snapshot, it is in the generation trajectory.**

### The pivot: from representation to dynamics

Note the symmetry the phase exposed: **the error is a no-op on features (rotation-invariant) but is everything for dynamics (stability is the only thing that matters in an autoregressive loop).** Four probes asked the substrate to *encode* better; none asked the *generation* to learn its own dynamics. The degenerate loops and char-flood that inflate armB's byte roughness and block the full gate are **autoregressive instability, not per-step errors.**

The no-backprop, CPU-native, mantra-pure way to attack that is **FORCE / RLS (Sussillo-Abbott)**: give the readout recurrent feedback and train it online with recursive least squares, so generation acquires a *learned attractor* while the substrate stays a fixed reservoir (no backprop-through-time). It differs from DAgger (which trains a static, memoryless readout to coverage): FORCE gives generation its own recurrent memory and feedback, aimed directly at what blocks armB from the full gate. **MODOJA / scaling-the-lift / unit-choice remain queued** behind this; armB stays the frozen baseline.

Iron law 44-47 carried in: TF is not generative, no closed-loop read before the gate, no celebration before gate v2 + replicas + human reading. Worst-of-32 on ~3% events is coin-flip-like; any gate re-specification is a user decision BEFORE seeing results.

## Constraints for the Next Agent

- Readout stays H32 static stateless MLP; Transformer-like modules are not admissible.
- No new volatile inference memory; no word/bigram counters in policy; no generation-side hacks.
- Gate v2 bars move only by tightening; recalibrate byte bars only if the corpus changes (method in `phase47i.ps1` step 1a; bars in `docs/gatev2_bars.json`).
- Human reading is a mandatory gate component; dump samples for every full-PASS.
- Deterministic pre-checks before large sweeps; single trainer at a time (RAM); smoke-execute new harness code paths.
- Commit only on explicit user order, at phase closure.

## Key Artifacts

| artifact | role |
|---|---|
| `weights/phase48a_I_h32_r7.bin` / `_r8.bin` | armB closed-loop: word-clean, frontier ~2.18-2.22, char-flood residual (self-BPB ~1.8) |
| `weights/phase48_0exp_armB_h32.bin` | armB TF probe (0x53454548, +0.04 vs notap) — frozen-baseline lift definition for 48.B |
| `benchmarks/phase38-42/phase48a_armb.c` / `phase48a_generator.c` / `phase48a.ps1` | armB DAgger trainer / lift-rebuilding generator / harness (determinism pre-check + gate v2 + replicas) |
| `benchmarks/phase38-42/phase48_0_expand.c` | EXPAND probe (armA linear control / armB cos / shuffle guard) — the 48.B/C/D template |
| `benchmarks/phase38-42/phase48_b.c` / `.ps1` | 48.B static probe (BILIN/WAVE32/MULTIBW + sp/leak guards) — static-map ceiling, no promotion |
| `benchmarks/phase38-42/phase48_c.c` / `.ps1` | 48.C multiplicative-dynamics reservoir (DYN/LIN_dyn/DYN_st/DYN_sp, gate-liveness) — random bilinear = damage |
| `benchmarks/phase38-42/phase48_d.c` / `.ps1` | 48.D MODOJA-K error-modulated Oja kernel tilt (+PARITY/SHUF-MOD, `--pdiag` η₀ tuning, P_DIVERSITY) — RFF-flat |
| `weights/phase48c_DYN_h32.bin` / `phase48d_MK_h32.bin` | 48.C/48.D TF probe checkpoints (0x53454549 / 0x5345454A) — not promoted |
| `weights/phase47i_I_h32_r9.bin` | Phase 47 frontier: word+whitespace clean, char-flood residual, 2.2621 |
| `weights/phase47g_P_h32_r7.bin` | historic word-only dual-temp pass (byte-dirty) |
| `weights/phase43c2_C2A.bin` | only byte-clean model (linear, word-dirty) |
| `weights/phase44f_F0.bin` | D1 stable feature substrate used by all 47 readouts |
| `docs/gatev2_bars.json` | calibrated gate v2 byte bars |
| `benchmarks/phase38-42/phase47i.ps1` | gate v2 reference implementation (calibration step 1a, full pipeline) |
| `benchmarks/phase38-42/phase47_generator.c` | closed-loop generator (D1+MLP, deterministic) |
| `results/phase47*/` (gitignored, local) | full run outputs incl. human-read samples |
