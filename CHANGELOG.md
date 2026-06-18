# CHANGELOG — Silicon Entropy Engine

## V1.0.2 (2026-06-17) — Phase 49 Closed + Commit

**Phase 49: Generation Dynamics (FORCE/RLS output-feedback)**

**49.0 — feasibility (built + smoke-validated; run is the user's)**
- FIXED echo-state feedback reservoir appended to frozen armB feature
- `h_t = (1-α)·h_{t-1} + α·tanh(W_in·φ_armB(s_t) + W_fb[prev_byte] + W_rec·h_{t-1})`
- All weights fixed; only H32 readout learns via 47.I DAgger trainer
- NO-FB control (`--nofb`, W_fb=0) is the discriminant (linear)
- Save magic `0x53454550` carries feedback config for generator rebuild

**49.1 — stabilized output-feedback (ERR + ADAPT)** *(built, core-validated, committed)*
- **Diagnosis**: reinjecting full output amplifies momentum; fix is to reinject only the surprise
- **ERR** (negative auto-corrective): `feedback = W_fb·(onehot(b) − p)`, p = current model's softmax → auto-corrective
- **ADAPT** (spike-frequency adaptation): keep full output, subtract `G_adapt·c_t` where `c_t=(1−β)c+β·h_{t−1}` → persistent activity self-inhibits
- SEE-state snapshotting for determinism: snapshot at window start once, then `memcpy`-restore (bit-identical)
- Save magic `0x53454551` carries ERR/ADAPT config + SEE state snapshot

**49.1b — human read on stabilized checkpoints** *(DONE, decisive negative)*
- NO training. Generates long samples (~3500B) from topBi-cleanest checkpoints (ADAPT A_r8/r9, ERR E_r7/r8) and armB-DAgger reference
- Computes structure+diversity advisory (quoteParity/parenImbal/sentLen vs corpus + n-gram uniqueness)
- Dumps samples to `results/phase49_1b/human/{arm}_{rN}_T{temp}_s{seed}.txt`
- **Architect read 5 samples = DECISIVE NEGATIVE**: all arms (ADAPT/ERR/armB) read in the same word-salad class — real surface words + name-tokens + template anchors + locally-grammatical 3-6-word runs, drowned in heavy degeneration; no sustained multi-clause syntax, no maintained referents
- **Verdict**: branch 3 — stability/frontier improved (TF −0.035 below NO-FB) but still word-salad → gap is REPRESENTATIONAL, not dynamical

**Key findings:**
- Feedback is load-bearing: FORCE beats NO-FB on cyclic loops, TF gain ~−0.024/round
- Positive output-feedback amplifies attractors → monotone runaway (chRun/wsRun r6→r9 ~11→37)
- ERR/ADAPT break the monotone climb (ERR 13/8/17/18, ADAPT 11/17/12/19 vs POS 13/19/26/37) — runaway broken
- Damping incomplete: residual flood floor ~18-19 vs NO-FB ~9 (β0.97/G_adapt0.50)
- Best arm ADAPT beats POS in TF avgVal (−0.011 BPB gain over POS)

---

## V1.0.1 (2026-06-14) — Phase 48 Closed + Commit

**Phase 48: Substrate Feature Classes**

**48.A — armB nonlinear lift under frozen 47 harness** *(CLOSED)*
- **armB**: fixed random nonlinear lift `z = Ω·L0norm` (Gaussian Ω), `cos(z)`, temporally integrated as new reservoir dims (two EMAs 0.90/0.99), appended to frozen base
- First move on what the substrate *knows*, not how well readout reads it: **+0.04 BPB** on all three held-out windows (armB 2.2023/2.1839/2.1980 vs notap 2.2427/2.2257/2.2395)
- Controls perfect: linear arm (integrate `z`, no cos) *worsened* 0.17-0.21; shuffle guard inert
- **Result**: the five-phase structural instability is solved. Word-clean closed-loop, stable frontier ~2.18-2.22 BPB (self-BPB ~1.8), budget for rollout cost lands comfortably under bar where P_r7 was on the edge
- Read by eye: real TinyStories phrasing emerges ("Once upon a time...", "she was so excited to...") — measurably less word-salad than 2.25 era
- **Residual**: char-flood byte channel (`chRun` fails: "...aaaaaaaa", "huhhhh") = the fidelity ceiling, exactly coverage prediction (K128 bursts fall into whitespace, never char-flood)

**48.A-fix — targeted char-flood coverage**
- Rounds 6-9 add char-flood-seeded bursts (force short repeated-char run, then roll out) beside K128 whitespace bursts
- Gated by mandatory no-training premise (smoke stayPct 15.6% → reachable)
- armB closed per hard cap; char-flood accepted as documented byte residual / watch-metric

**48.B / 48.C / 48.D — static per-step axis, closed with a proof**

armB taught us: **the silicon understands similarities (kernel) but not relations.** A relation is a product (A·B = "A in context of B"); additive EMA reservoir cannot represent it.

| Probe | Move | Result | Reason |
|---|---|---|---|
| **48.B** | relation as static feature | flat (~0.01, none clears 0.015×3) | static product = generic quadratic — `BILIN_sp` (shuffled pairing) kept ~85% of gain, so it's second moment, not gating |
| **48.C** | relation as random dynamics | damage (DYN < LIN_dyn, gate verified live) | no theorem for random bilinear form → multiplication injects noise, not signal |
| **48.D** | error as feature selection | flat (MODOJA-K = PARITY = armB) | RFF is rotation-invariant: tilting kernel's feature directions just resamples same kernel |

**48.D detail**: fixed Ω inside `cos(γ·Ω·L0norm)` became error-tilted P; per-row Oja, lr modulated by frozen-trigram surprise; P learned then frozen (zero val leak); η₀=1e-7 gives genuine ~13° tilt with kernel still diverse (eff_rank 33/64). Error touched substrate differentially but was no-op on BPB: `MODOJA-K − armB = +0.0009/+0.0027/+0.0012` (≪0.015). **You cannot improve a kernel machine by rotating its features** (Gaussian spectral density is rotation-invariant).

---

## V1.0.1 (2026-06-12) — Phase 47 Closed + Commit

**Phase 47: Static Nonlinear Readout + On-Policy Training**

**47.I — far-field training, one shot, hard cap**
- Step 1 re-score P_r7 under byte-guards: **P_r7 fails 27-32/32 samples** in every replica (wsRun worst 106-233 vs bar 6 — the wasteland was almost everywhere)
- **C2.A is byte-clean**
- D1 has slight whitespace drift (wsFrac +0.008-0.011) born with L2 memory
- **No model of the project passes full gate v2**
- Step 2a premise probe (no training): K=128 anchored bursts DO enter wasteland (38.2%/36.9% of burst tails vs K16 control ~9-11%) → coverage hypothesis live
- Step 2b far-field training: prefix r1-r5 bit-identical to 47.G (MD5), rounds 6-9 with burst mix K16 87.5% / K128 12.5%
- **Result (I_r9, T0.55)**: whitespace wasteland genuinely gone — text stays structured to last byte, consecutive spaces 3-5, wsRun at corpus level (6 = bar). Word metrics best ever (bi 4-6). Residuals: char flood persists (chRun 20-26 vs bar 8), BPB 2.2621 (far-field cost +0.018)

**Coverage theory (validated 3/3, predictive):**
| degeneration channel | training coverage | outcome |
|---|---|---|
| word-level loops (topBi/run/name) | K16 bursts, near-field | cured and replicated |
| whitespace wasteland | K128 bursts entering it (~15% entries) | cured at corpus level at I_r9 |
| char flood ("aaaa...") | almost never visited (~2.9% entries) | persists |

**47.H — extended validation**
- Replica protocol: P_r7 **not promotable by pre-registered rule** (2/4 replicas fail, both only altLp@0.55 = worst-of-32 on ~3% per-sample event → 47.G PASS was a tail)
- But structural fix REPLICATED: topBi 6-8, name/run/selfBPB green in 8/8 evaluations incl. held-out prompts
- H48 tail closed (capacity ceiling confirmed; stable regime exists only at H32 on this substrate)

**47.G — last mile H32**
- Plain continuation r6-r9. **P_r7 = first dual-temperature full-gate PASS in project history**
- `weights/phase47g_P_h32_r7.bin`: BPB 2.2497, topBi 6/7, altLp 2/2
- Anneal (mix 10%) rejected: BPB flies, structure lost — late 20% rollout dose is load-bearing
- Checkpoint promoted, not recipe

**47.F — temperature coverage**
- Hypothesis falsified cleanly (Tmix bursts: bi55 13=13 at 50% coverage)
- Rounds non-monotone at H64 (r3 optimum). H64 closed.
- Fallback D16r5_h32: topBi 5/5, best near-pass

**47.E — capacity**
- Recipe at H64. First one-temperature full PASS ever (D16r3_h64 @T0.65, BPB 2.1901) but T0.55 wall (topBi 13)
- Budget exonerated via D0 controls; rollout cost ~+0.044 capacity-independent

**47.D — real rollouts (the structural breach)**
- DAgger-style training on bursts sampled from decoder ITSELF (K bytes every 256B, target always true byte, +16B recovery, mix 80/20)
- **First configs ever under topBi<=8 at both temps with sane selfBPB** (D16_h32 topBi 5/6)
- Dose-response: K=8 no re-entry, K=16 sweet spot, K=32 degenerates

**47.C — proxy robustness**
- One-step corruption/D1-burst training improves dirty validation (valC) but does not transfer to closed-loop → wall is **multi-step autoregressive drift**

**47.B — regularization**
- Label smoothing/RMS-penalty/WD/dropout matrix. RMS normalizes scale but topBi stays → **not a calibration problem**
- Diagnosis: exposure mismatch

**47.A0/A0b — sanity + anchor repair**
- Ladder valid (H32 2.2360 / H64 2.1627 / H128 2.0807), randlabel/shuftime controls clean
- Ablation shows nonlinear gain lives in **SEE**, not L2
- `frozenD1` anchor (D1 weights through exact probe path == BASE D1 at 4th digit) became permanent pipeline-integrity check

**47.0 — feasibility**
- Distillation from unstable teachers (delta, L3-B3) into stable [SEE|L2_D1] features
- Discovery: **the lever is readout nonlinearity, not the teacher.** A 1-hidden-layer MLP on stable features reaches 2.0947 val (vs linear 2.2410); KD-from-delta adds nothing

---

## V1.0 (2026-05-26) — Release Seal

**Phase 36: Documentation & Release Hardening**
- README updated: build command corrected (`regime_prior.c` added), `-ffast-math` documented as forbidden, reproducibility guarantee and known limits expanded
- CHANGELOG updated: Phases 31–35 added

**Phase 35: Reproducibility & Portability Audit**
- Encode determinism confirmed: double-encode produces byte-identical archives
- Format identity fixtures committed: `data/fixtures/` contains 3 small `.see` archives with `manifest.json`
- Header rejection confirmed: 5 corruption cases (bad magic, truncation, header_size mismatch, weights hash mismatch, empty file)
- Compiler variant analysis: `-O2` = `-O3` output (MATCH on Clang 21 / x86_64 Windows). `-ffast-math` differs.

**Phase 31: External Compressor Atlas**
- SEE benchmarked against zlib-9, bz2-9, lzma, zstd-22, brotli-11 on 14 corpora
- SEE ties classical compressors only on shuffled/random data (+0.003 BPB vs brotli-11)
- Gaps: prose +0.7 BPB, markdown +2.0 BPB, C code +1.2 BPB vs brotli-11

---

## V0.9.x series (Phases 28–30)

See git history for detailed changelog entries on Phases 28-35.
