# SiliconLLM Handoff

Last updated: 2026-06-18 — **Phase 51 OPEN — Associative memory direction (capacity gate first)**

The new arc. Phase 50 solved byte-fidelity (recognizable English, BPB 1.69) and isolated the residual to **selective memory**: the generator loses the thread because a frozen reservoir of N units cannot retain what it emitted 200 tokens ago. This is not a tuning gap — it is a **theorem**. Echo-state / fading-memory theory bounds the linear memory capacity of any fixed reservoir at **MC ≤ N**, with exponentially decaying fidelity. Adding cells, timescales, or Oja plasticity (the Phase 48 substrate-scaling proposal) raises N but never escapes the bound. The compressive (blurred-summary) memory of a reservoir is the wrong *kind* of memory for language.

The reframe. Language needs an **associative, content-addressable** memory beside the reservoir — a key→value store, the same object a Transformer calls a KV-cache. The unifications worth stating once: **Hopfield networks ≡ attention** (modern Hopfield retrieval = softmax attention), **fast-weights ≡ linear attention** (outer-product write, matrix-vector read), and **VSA/HRR** (Holographic Reduced Representations) realize the same key-value algebra with fixed random atoms and an invertible *binding* operator. The silicon-native form of binding is **permutation** — specifically **cyclic-shift**, the cheapest invertible, decorrelating operator on a vector.

Why this is charter-pure. The substrate stays **frozen** (no backprop, no learned attention, no BPTT). Keys are **fixed/random**; the write is a parameter-free Hebbian/binding superposition; **only the readout learns to read it by similarity**. This moves selection from *write-time* (expensive content selection — what killed 48.C's per-event write gate) to *read-time* (similarity = the free operation of the Phase 48 dichotomy). It is the relax-charter fork (a) from the Phase 50 close, executed in its least-charter-violating form.

**51.0 is the capacity gate (this step).** Before wiring anything to the readout, measure whether the organ can even *hold* the information: a standalone, model-independent VSA capacity proof — codebook of 1024 fixed random hypervectors (one per BPE-1024 token), write `M = Σ_k decay^k · cyclic-shift^k(atom(token_{t-k}))` over **real** TinyStories token streams, read by inverse-shift + cosine/Hamming cleanup, and chart **accuracy(k)** vs distance over a grid of dimension D ∈ {256…8192} × decay, for two algebras (bipolar/cosine vs binary-XOR/Hamming). Headline = k*(D,decay), the recall horizon, compared to the VSA capacity bound ~D/(2·ln vocab). Controls: T=1 ≈ 100% (binding sanity), key-shuffle → chance (proves positional binding carries the info), raw-cosine vs post-cleanup, real-repetition interference. **Green → next phase wires the readout onto the store; red → wall quantified, decide O(1)-superpose vs O(t)-keep-keys.** No model, no training, no substrate change.

Last updated: 2026-06-17 — **Phase 50 CLOSED (milestone)** — unit-choice cartography → **first recognizable English**. BPE-1024 (silicon-chosen by bits/byte) wired into a token readout over the unchanged byte-driven armB substrate: **byte-fidelity solved** (all byte-guards pass), BPB ≈ 1.69, recognizable (bad) English. The lone residual = **topBi** excess, and 50.B's premise falsified every coverage (decoder escapes seeded repeats at the natural rate) while corpus calibration confirmed the bar is honest (real TinyStories topBi p90=8 = the full-gate bar) → topBi = function-word fallback when the model loses the thread = **selective memory = the charter-question**. Every cheaper mantra-pure lever is spent. Fork (open): (a) relax-charter reservoir-native selective memory ("attention from the reservoir", no backprop into substrate); (b) declare 50.A the mantra-pure deliverable and publish; (c) consolidate (done). Full arc map in the Phase 50 section below.

Last updated: 2026-06-17 — **Phase 49 CLOSED + COMMITTED (V1.0.2)** — first PARTIAL win of generation-dynamics (goal-2), NOT a step to language. Phase 48+49 arc crystallized:

1. armB (RFF kernel) solved the structural instability: word-clean closed-loop, frontier ~2.18-2.22 BPB, char-flood residual
2. Output-feedback (ERR/ADAPT variants) breaks monotone runaway but flood floor ~18-19 vs NO-FB ~9 (incomplete damping)
3. Human read of stabilized checkpoints (49.1b): all arms (ADAPT/ERR/armB) in same word-salad class — no sustained syntax, no maintained referents
4. **Verdict**: gap is REPRESENTATIONAL (blurred summary memory vs selective content-addressable retrieval), not dynamical

The mantra-pure axis CEILINGS at "a stable ~2.19-BPB byte generator that emits structured word-salad". The charter question: (i) unit-choice cartography = last mantra-pure lever (Architect recommends FIRST, byte-level floods dissolve at word/token granularity), or (ii) relax charter for selective-memory primitive.

See `CHANGELOG.md` V1.0.2 and `docs/PHASE44-45_SYNTHESIS.md` for technical depth on each phase.

Last updated: 2026-06-16 — **Phase 49 CLOSED + COMMITTED (V1.0.2)** — Phase 48+49 arc crystallized, human read decisive negative on structure. 49.1b: ADAPT/ERR read in same word-salad class as armB at ~2.19 BPB.

Last updated: 2026-06-14 — **Phase 48 CLOSED (V1.0.1)**. 48.A: armB nonlinear-lift substrate solved the five-phase structural instability (word-clean closed-loop, frontier ~2.25 → ~2.18-2.22, char-flood byte residual). 48.B/C/D then closed the STATIC per-step axis with a proof per arm (static product = generic quadratic; random bilinear dynamics = noise; error-tilt = no-op because RFF is rotation-invariant). armB sits exactly at the boundary of what a static read of this substrate can give. Pivot: error is a no-op on features but everything for DYNAMICS → next axis is FORCE/RLS (learned generation attractor, no backprop). Phase 47 closed 2026-06-12.

## Goal

Build toward a real language model by "pulling an LLM out of the silicon, not imposing an LLM on the silicon."

Core rule: keep the readout simple, falsifiable, and structurally honest. Gains should come from substrate geometry, memory, local plasticity, boundary/homeostatic dynamics, training signals, and robustness properties. Avoid Transformer-like modules, word/bigram counters in policy, generation hacks, and opaque inference-time machinery.

## Promotion Gate V2 (current, frozen)

A candidate must pass **all** components, **worst-case over 32 samples**, at **both T=0.65 and T=0.55**, then survive the **replica protocol** (4 independent full-gate replicas, rule 4/4), then **human reading** of dumped samples (mandatory, permanent component — the user reads them).

Word-level bars (unchanged from V1):

| component | bar |
|---|---|
| BPB (teacher-forced, avg of 3 val windows) | <= 2.2543 |
| topBi (most repeated word bigram) | <= 8 (= corpus p90, confirmed corpus-truth in Phase 50; see `token_word_bars`) |
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

## Phase 50 — Unit-Choice Cartography → first recognizable English (CLOSED 2026-06-17, milestone)

The Phase 48+49 wall: a fixed reservoir's memory is a blurred non-selective summary, and the residual degeneration was at the BYTE level (char-flood, ws-flood, broken morphology). The mantra-pure lever left was **unit-choice**: let the silicon emit a larger unit so the byte-degeneration channels become structurally impossible — chosen by the silicon, not imported from LLM convention.

### 50.0 — the MAP (cheap, no substrate, no training)

`phase50_0_map.c`: a fair interpolated-KN n-gram (order 1/2) in **bits-per-BYTE** (unit-invariant, OOV byte-fallback = lossless) over train/held-out, for every candidate unit — byte / word / BPE-512/1024/4096 / 2-byte-stride / hashed-3gram. Result: **BPE-1024 wins** — bpb-o2 ≈ 0.94 vs byte 3.31, repeat-mass (P(next==cur)) collapses byte 0.025 → BPE ~0.0001 (the byte flood dissolves at subword granularity), ~2.5 bytes/unit, feasible vocab, lossless. The silicon-native non-linguistic units (bytepair, hashed-3gram) were beaten or lossy. Mantra honoured: the unit was selected by bits/byte, not by convention.

### 50.A — wire BPE-1024 into the readout (the breakthrough)

Substrate armB stays **byte-driven and INVARIATO**; only the readout changes: at each token boundary it reads φ_armB and predicts the next token (1024-way softmax, H32, + frozen token-bigram prior = the analog of the byte-trigram), trained with the 47.I DAgger recipe lifted to tokens (emit a sampled token → feed its bytes on-policy → target = true next corpus token). Generation: predict token → sample → emit its bytes → substrate advances. Checkpoint `0x53454554` embeds merges + bigram so the generator reconstructs the tokenizer (`phase50_a_token.c` / `phase50_a_generator.c` / `phase50_a.ps1`; codec `bpe_codec.h`).

**Result = breakthrough.** Recognizable English, **byte-fidelity solved** (all byte-guards pass — TOK chRun 2 / wsRun 4 vs armB 6 / 9), self-BPB ~1.27, BPB ≈ 1.69. Fifty phases took the project from a byte compressor to a generator that writes (bad but real) English. The only full-gate fail is **topBi** (word/token repeat — "the the", "to the to the").

### 50.B / 50.B-diag — classify the topBi residual (no training)

Is topBi a coverable token-flood (like the cured char-flood), a bar artifact, or the charter-question? Two cheap measures decided it:
- **Premise (no-train, `phase50_b_token.c --premise`, on r5 AND deployed r9):** seed the decoder into each repeat mode and measure persistence vs a natural control. **single** seeded 0.039 ≈ natural 0.043 (ratio 0.91); **alt** ("the the") seeded 0.30 vs natural 0.23 (ratio 1.3 — mostly NATURAL function-word frequency, weak lift); **dup** ("named named") negligible both ways. **No mode is a coverable attractor** — the decoder escapes seeded repeats at the natural rate; the token-bigram prior already suppresses same-token loops. Branch 3.
- **Corpus calibration (`phase50_consolidate.ps1`, 400 held-out 2 KB gate-windows of REAL TinyStories):** topBi p50=5, **p90=8**, p97=10, max=18. Real human text itself exceeds the mini-gate's topBi≤7; the full-gate bar of 8 = corpus p90 exactly (already corpus-truth — no loosening needed). `docs/gatev2_bars.json` now carries a documented `token_word_bars` section.

**Re-gate FOR THE RECORD (not a promotion, corrected bars):** byte-guards PASS at T0.65 (wsRun 5 / chRun 3); topBi worst-of-32 = **18 (T0.65) / 20 (T0.55)** — i.e. the generator's worst-of-32 reaches corpus-MAX-level repetition (18) about 1-in-32, where real text reaches it ~1-in-400. A real **tail** residual, above the fair worst-of-32 reference (corpus p97=10), driven by function-word fallback when the model loses the thread. (`results/phase50_a/regate_record.txt`.)

### The wall, isolated (arc 48→49→50)

- **Static representation** maxed at armB (Phase 48, RFF kernel + three proofs).
- **Feedback memory** gives stability, not structure (Phase 49 — runaway broken, still word-salad).
- **Unit-choice** dissolves the BYTE degeneration (Phase 50 — first recognizable English, byte-fidelity solved).
- **The single remaining residual = topBi excess = the model falls back on frequent patterns when it loses the thread = SELECTIVE MEMORY = the charter-question.** The premise falsified every coverage; corpus calibration confirmed the bar is honest; every cheaper mantra-pure lever is now spent.

### The fork (open, for when work resumes)

(a) **Relax the charter** with a reservoir-native selective memory — fixed/random-key key-value where only the readout retrieves ("attention from the reservoir", no backprop into the substrate) — the one untried lever that targets selective content-addressable recall directly. (b) **Declare 50.A the mantra-pure deliverable** and close/publish (a fixed reservoir + minimal static readout + silicon-chosen unit writes recognizable English). (c) **Consolidate** (done here). The charter-question is the user's — it is their philosophy.

### Key artifacts (Phase 50)

| artifact | role |
|---|---|
| `benchmarks/phase38-42/phase50_0_map.c` / `phase50_0.ps1` | unit-choice cartography (bits/byte n-gram map); `--save-bpe` dumps the BPE-1024 merges (`weights/bpe1024.bin`) |
| `benchmarks/phase38-42/bpe_codec.h` | shared BPE encode/decode (lossless, round-trip verified) |
| `benchmarks/phase38-42/phase50_a_token.c` / `phase50_a_generator.c` / `phase50_a.ps1` | token trainer (DAgger, 0x53454554) / closed-loop token generator / gate harness |
| `weights/phase50a_tok_I_h32_r1..r9.bin` | 50.A token readout checkpoints (r9 deployed): recognizable English, byte-fidelity solved, topBi residual |
| `benchmarks/phase38-42/phase50_b_token.c` / `phase50_b_diag.ps1` | re-posed premise (single/alt/dup) + corpus calibration — topBi classified as non-coverable (charter-question) |
| `benchmarks/phase38-42/phase50_consolidate.ps1` | corpus calibration → recalibrated `token_word_bars` + re-gate record |
| `docs/gatev2_bars.json` | byte guards (unchanged) + `token_word_bars` (corpus-truth p90) + honest note |
| `results/phase50_a/regate_record.txt` | re-gate of 50.A r9 on corrected bars, for the record |

## Phase 49 — Generation Dynamics (FORCE / output-feedback) — 49.1 OPEN

Phase 48 proved the per-step REPRESENTATION axis is exhausted (armB at the boundary; B/C/D negative with a reason each) and that the gap to language is in the generation TRAJECTORY (structured word-salad with short cyclic loops). The symmetry it exposed: **the error is a no-op on features (RFF rotation-invariant) but is everything for dynamics (stability is the only thing that matters in an autoregressive loop).** Four probes asked the substrate to *encode* better; none asked the *generation* to learn its own dynamics. Phase 49 pivots there.

### 49.0 — FORCE-flavored output-feedback feasibility (built + smoke-validated; run is the user's)

A FIXED echo-state feedback reservoir is appended to the frozen armB feature:
`h_t = (1-α)·h_{t-1} + α·tanh(W_in·φ_armB(s_t) + W_fb[prev_byte] + W_rec·h_{t-1} + b)`, all of W_in/W_fb/W_rec fixed random, W_rec scaled to spectral radius ρ<1 (echo-state, verified empirically by ECHO_STATE at startup). The readout reads `[φ_armB 512 | h 256]`.
- **Output-feedback** = `W_fb[prev_byte]` = a fixed random embedding of the previously emitted/observed byte. In teacher-forcing it's the true byte (so h is model-independent and the base windows extract once); in a DAgger **rollout burst it's the model's OWN sample** (on-policy) — so h carries the model's trajectory and the readout learns to correct from its own feedback state. This is the FORCE-essential bit.
- **Stop-gradient / no BPTT**: h_t is precomputed during extraction and STORED in the feature row, so the existing 47.I DAgger trainer works UNCHANGED on a wider (768-D) row — no gradient flows into the reservoir, no backprop into the substrate. Only the H32 readout learns.
- **NO-FB control** (`--nofb`, W_fb=0): the discriminant (LIN_dyn/PARITY analogue). If FORCE ≈ NO-FB, the feedback is inert (the reservoir dynamics are too weak to carry structure).
- **Gate**: echo-state (both variants must forget init) → anchor → determinism MD5 (new feedback path) → MINI gate v2 over FORCE/NO-FB r6..r9 + the **FORCE-vs-NO-FB discriminant** on the degeneracy channels (topBi/altLp/chRun/selfBPB) → FULL gate v2 + replicas with refs C2.A/D1/P_r7/armB I_r7/I_r8 + a long-range **structure advisory** (quote/paren balance, sentence length) + human reading. New save magic `0x53454550` carries the feedback config so the generator rebuilds the loop.

Pre-registered tree: (1) feedback load-bearing (FORCE > NO-FB) + structure improves → escalate to **49.A = true FORCE** (linear readout + online RLS, native on-policy convergence); (2) FORCE ≈ NO-FB → output-feedback-as-state doesn't help this reservoir → enrich the recurrence (the silicon's own wave dynamics closed in loop) or unit-choice cartography; (3) stability improves but still word-salad → partial win (goal-2, lower stable frontier), gap is representational not dynamical. Honest caveat: FORCE historically generates patterns, not rich language — the structure metric distinguishes stable texture from grammar. TF ≠ generative; a PASS is validated only by gate v2 + replicas + human reading.

**49.0 result**: feedback is **load-bearing** (FORCE beat NO-FB on cyclic loops, TF gain ~−0.024/round) but **momentum-pathological** — positive output-feedback amplifies attractors, so the floods deepen monotonically with the DAgger rounds (chRun/wsRun r6→r9 ~11→29; coverage can't tame them). The wall: memoryless = stable but structureless; momentum-memory = runaway; selective memory = charter-forbidden (learned gate = backprop).

### 49.1 — stabilized / negative feedback (ERR + ADAPT) — built + core-validated

Diagnosis: feeding `W_fb·onehot(b)` reinjects the whole output, including the predictable part `W_fb·p` = the self-reinforcing momentum. Fix-of-principle (predictive coding / control): reinject only the **surprise**. Same fixed reservoir as 49.0; only the feedback content changes (a clean discriminant):
- **ERR** (headline): `feedback = W_fb·(onehot(b) − p)`, p = softmax of the readout at the previous step → negative/auto-corrective feedback. Model-dependent (h re-extracted with the current model's p each round; p detached → no BPTT).
- **ADAPT**: keep `W_fb·onehot(b)` but subtract a spike-frequency adaptation current `G_adapt·c_t`, `c_t=(1−β)c+β·h_{t−1}` → persistent activity self-inhibits, breaking the flood fixed points. Model-independent; present identically train+inference (reservoir dynamics, not an inference hack, not a volatile write).
- POS/NO-FB weights reused from 49.0 as references. **2-D pre-registered criterion**: (1) anti-runaway PRIMARY — ERR/ADAPT chRun/wsRun do NOT rise with rounds (~= NO-FB) unlike POS, read via a per-round **flood scan**; (2) keep-the-signal — TF avgVal stays below NO-FB; (3) WIN — keep the gain AND match NO-FB on floods → mini → full gate v2 + structure + replicas + human. Tree: (1) runaway tamed + signal kept → **first STABILIZED memory** → 49.A (RLS-FORCE); (2) tamed but signal lost (ERR≈NO-FB on TF) → signal and runaway inseparable; (3) neither tames it → output-feedback intrinsically unstable here → axis closed → charter fork.
- **Perf fix (essential)**: ERR re-extracts every round, and the SEE warmup re-walks the prefix (up to ~53M `see_observe`) per extraction → ~1.3B wasted observes. `SiliconEntropyState` is POD → snapshot the SEE state at each window start once and `memcpy`-restore (bit-identical). New magic `0x53454551`. Both ERR/ADAPT generators validated byte-deterministic (MD5 MATCH) with sane, flood-free output.

**49.1 RESULT (real run 2026-06-16; 2-D criterion PASSED → branch 1, then demoted to branch 3 by the human read)**: clean experiment — ESP OK (ERR 3.2e-8 / ADAPT 2.4e-8), anchor OK, determinism MATCH. (1) **keep-the-signal PASS NET** (the earlier len-4000 smoke fear was an undertraining artifact): TF avgVal r9 **ADAPT 2.1896 < ERR 2.1955 < POS 2.2004 < NO-FB 2.2243** — both stabilized arms keep the gain (−0.035 / −0.029 vs NO-FB) and **ADAPT beats even POS** (−0.011): the adaptation current is DUAL-PURPOSE — it damps the runaway AND is itself a predictive feature ("I've been repeating for a while"). ADAPT = winning arm. (2) **anti-runaway PASS PARTIAL** — flood scan @0.55, worst chRun per round: POS r6-9 = 13/19/26/**37** (monotone runaway confirmed at real scale), ERR = 13/8/17/18, ADAPT = 11/17/12/19 (**no monotone climb → runaway BROKEN**) but floor ~18-19 vs NO-FB ~9 (damping at β0.97/G_adapt0.50 incomplete; residual ~2× NO-FB). wsRun r9: ERR **10 (below NO-FB 15!)**, ADAPT 24, POS 37. (3) Mini gate v2: topBi CLEAN like armB (E_r7@0.65 bi2, E_r8 bi3, A_r8 4/4, A_r9 3/3 — bar 8) = healthy cyclic structure, **but wsRun/chRun over bar (8-24 / 7-14) = the SAME diffuse byte roughness as armB (48.A fidelity ceiling) → none qualify to full**; comparable to armB-DAgger on the gate but with TF −0.035 below it.

### 49.1b — structure + human read on the stabilized checkpoints (DONE → decisive negative)

NO training. `phase49_1b.ps1` generates long samples (~3500 byte, both temps, 4 seeds) from the topBi-cleanest stabilized checkpoints (ADAPT A_r8/A_r9, ERR E_r7/E_r8) and the armB-DAgger reference (`phase48afix_I_h32_r7/r8.bin`, magic-verified for `phase48a_generator`), computes a structure+diversity advisory (quoteParity / parenImbal / sentLen-vs-corpus + n-gram uniqueness uniBi/uniTri/uniC4 worst-of-seeds + longest word-run + selfBPB), and dumps every sample to `results/phase49_1b/human/{arm}_{rN}_T{temp}_s{seed}.txt`. **Architect read 5 samples (ADAPT r8 T0.65/0.55, ADAPT r9 T0.55 best-case selfBPB 1.26, ERR r7 T0.65, armB r8 T0.65) = DECISIVE NEGATIVE.** All arms (ADAPT/ERR/armB) read in the SAME word-salad class: real surface words + name-tokens (Lily/Anna/Sam/Max/Jack) + template anchors ("Once up...", "He said", "Suddenly") + locally-grammatical 3-6-word runs, drowned in heavy degeneration with checkpoint-specific flood signatures (ADAPT="tdtdt", ERR=".dtkh", armB="t t t t"); **no sustained multi-clause syntax, no maintained referents, no narrative coherence; at 3500B the bounded floods RE-EMERGE.** Metrics agree (zero separation): uniTri ~0.99-1.0, uniC4 0.40-0.58, maxRun 2-4, sentLen 12-23 vs corpus 8.3 (run-on: sentences never terminate) — all arms alike.

**Verdict — 49.1 = pre-registered branch 3**: stability/frontier improved (TF −0.035, runaway broken, dual-purpose adaptation) but still word-salad → the gap is REPRESENTATIONAL, not dynamical. Goal-2 partial REAL (first output-feedback that doesn't explode + lower frontier) committed as a contribution, NOT a step toward language.

**Arc 48+49 wall, crystallized**: (1) static per-step representation = maxed at armB (3 probes + RFF theorem). (2) dynamics/feedback-memory: no-memory = stable word-salad; momentum = runaway; stabilized = runaway-broken + BPB-down but STILL word-salad. (3) Conclusion: giving memory to a FIXED reservoir (even stabilized) does not produce language, because the memory a fixed reservoir forms is a non-selective BLURRED SUMMARY, not content-addressable retrieval. Language needs SELECTIVE / content-addressable memory (attention / learned gate) = exactly the charter line (no-attention, no-backprop; the random version failed in 48.C). The mantra-pure axis CEILINGS at "a stable ~2.19-BPB byte generator that emits structured word-salad" — the mantra-vs-goal tension is now established with data from five angles. **FORK (user, charter-level)**: (i) unit-choice cartography = the last mantra-pure, charter-safe lever (byte-level floods dissolve at word/token granularity; shorter effective sequence lets the reservoir's blurred memory cover more linguistic context; + LLM prerequisite) — Architect recommends FIRST; (ii) relax the charter for a minimal selective-memory primitive (where language actually lives); (iii) RLS-FORCE = down-weighted (49.1 showed the stabilized loop doesn't buy structure → refining the readout on the same fixed reservoir adds no selective memory). The charter question is the user's (it is their philosophy).

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
| `benchmarks/phase38-42/phase49_0_force.c` / `phase49_0_generator.c` / `phase49_0.ps1` | 49.0 FORCE feedback trainer (`--nofb` control) / feedback-loop generator (0x53454550) / dual-train+echo-state+discriminant+gate harness |
| `benchmarks/phase38-42/phase49_1_force.c` / `phase49_1_generator.c` / `phase49_1.ps1` | 49.1 stabilized feedback trainer (`--ftype err\|adapt\|pos`, SEE-state snapshot) / generator (0x53454551, online p / adaptation c) / flood-scan + discriminant + gate harness (reuses 49.0 POS/NO-FB) |
| `benchmarks/phase38-42/phase49_1b.ps1` | 49.1b structure+diversity+human-read tool (NO training): long samples from A_r8/r9, E_r7/r8 vs armB ref → quoteParity/parenImbal/sentLen + n-gram uniqueness advisory + dumps to `results/phase49_1b/human/` |
| `weights/phase49{force,nofb,err,adapt}_I_h32_r1..r9.bin` | 49.0 POS/NO-FB + 49.1 ERR/ADAPT readout checkpoints (DAgger rounds 1-9); ERR/ADAPT = runaway tamed, predictive gain lost (not promoted) |
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
