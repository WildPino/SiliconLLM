# Rung-1 (S0) — Pre-Registration

**Sealed by the Architect 2026-07-19. Pushed before any rung-1 gate-bearing run executes** (push-before-run; the GitHub received-time is the witness). Nothing below is amended after numbers exist. Costs may re-price; **gates never move**.

Upstream context, all already pushed: `PHASE64_DECISIONS.md` (D1-D9), `PHASE64_TRAINING_PLAN.md` §12 (Inventor returns) and §13 (MVE gates read + re-priced ladder), `PHASE64_RUNG1_BRIEF.md` §8-§10 (WS1/WS2/WS5/WS6 returns).

## 1. What rung-1 is for

S0 (~30 M total / ~11 M active, E32 × h128 top-8) is the first rung of the ladder on **real code data**. It answers three things: does the recipe hold at rung scale on-domain; do the two candidate recipe improvements (vocab, structured x_proj) survive more steps and real data; and does the MoE dial actually deliver capacity (the D2 probe opened by the MVE's D→E finding).

Quality at S0 is **gated** (unlike the MVE, where it was recorded): the per-rung property gates of `PHASE64_DECISIONS.md` §10 apply.

## 2. Recipe and declared deviations from the MVE

Frozen spec v1 (`SCALEUP_ARCHITECTURE.md` §9) with these **declared, arm-symmetric** deviations, each already adjudicated on measured evidence:

| deviation | source | note |
|---|---|---|
| **CE-primary** (KD not in the main run) | §13, D3 FAIL | span-KD returns as a screening arm, §4.3 |
| **α-QAT ON** at stage-D entry, ramp **≤10% of stage-D steps** | §13 gate (iii) fired; brief §8 | end-state bit-identical to hot-swap QAT |
| **`--sparse-moe` ON** | brief §8 | warrant: grad-equivalence 1.1e-06, determinism bit-identical, resume 29-interruption |
| **effective batch 16 via micro-batch 8/rank + accum** | brief §8 | *comparability choice*, not a workaround — keeps the step grid identical to MVE runs 1-3 |
| **fp32 AdamW** (not 8-bit) | brief §10 | measured winner at this scale (−6% throughput for 8-bit); re-opens at S2 |
| **flat LR + `--warmup 200`** | run-3 declared deviation | carried forward, arm-symmetric |
| **stage-E context extension to seq = 2048**, micro-batch reduced to hold B·L constant | curriculum §4 "progressive context", made explicit by the §6.2 distance rule | the SSM scan is **linear in L**, so at constant B·L the tokens per step and the activation memory are unchanged; the open cost is throughput (less batch parallelism against a sequential scan) — **measured at the WS4 smoke and reported, not assumed** |

**`--compile-scan` is OFF — decided, not conditional: gate (b) failed reproducibly on the local screen** (§6.4). **Data**: pipeline of brief §3 with **P62 decontamination J=0.5 mandatory**; the manifest hash is recorded in §7 at launch and verified at every training start.

## 3. Units, baselines, significance

- Canonical metric: **BPB per-domain on the pinned code-val** (P62 CPython byte-slice), never blended. Byte-normalized, so it is comparable across vocabularies.
- **σ_seed = 0.005** (R1 calibration). A single-seed delta **≤ σ_seed is not a claim** — it triggers the pre-declared tie-breaker, never a narrative.
- All A/B arms are **seed-paired** (identical seed across arms of a comparison).
- Eval cap identical at every stage entry and exit (the MVE fix: transition deltas must compare like with like).

## 4. The screening block — stage C, one variable at a time

Three comparisons run **sequentially on stage C only**, each arm at a declared **15% of the stage-C token budget**, each arm seed-paired. These are **screenings, not claims**: they select the configuration for the main run and are labeled as such wherever quoted.

**The whole screening block runs on the logit-covered slice** (§4.3's data specification, generalized here), which buys two things beyond the confound it was introduced to close. First, **chained controls**: each screening's control arm *is* the previous screening's winner, so the block costs **four arms, not six**. Second, the stored logits are per *teacher* token and therefore vocabulary-independent — the same logit set serves both vocab arms, which is why they were stored that way in the first place.

| arm | what it adds | control |
|---|---|---|
| 1 — V=2048 | vocab candidate | paired with arm 2 |
| 2 — V=4096 | vocab candidate | paired with arm 1 |
| 3 — x_proj r=26 on the vocab winner | §4.2 | the winning arm above |
| 4 — span-KD on both winners | §4.3 | the winning arm above |

Four arms × 15% of stage C = **60% of stage C ≈ 33% of one rung's tokens.**

Decision rule, identical for all three and fixed here: **adopt the lower code-val BPB if the margin > σ_seed; if ≤ σ_seed, adopt the pre-declared tie-breaker below.** Every tie-breaker points in the cheaper direction — the conservative one.

**4.1 Vocab (D3's open half).** V=2048 vs V=4096, per-domain BPE.
Tie-breaker: **V=4096** (records a 17% shorter student sequence — compute saved in training and bytes/s at inference).

**4.2 Structured x_proj (§12).** r=26 vs dense, on the vocab winner.
Tie-breaker: **r=26** (17.6% of x_proj bytes; ~1.76 MB fp32 resident freed across the 5 Mamba layers, on the dominant proj-GEMV class).
Pre-registered reading: the sandbox measured r=26 *beating* dense by −0.0095 at 4k steps. **A null result here is the expected outcome of a regularization benefit fading with scale, not a failure** — it is recorded as such, and r=26 is still adopted by the tie-breaker on byte grounds.

**4.3 KD challenger, on-domain (the reopened D3).** span-KD vs CE, on both winners, with teacher logits from Qwen2.5-Coder-1.5B produced on the 3060 (§8 prerequisite).
Tie-breaker: **CE** (no logit cost).

**Data-coverage specification (sealed here, because it is a second variable if left implicit).** The KD arm samples only inside the resident-logit window; the CE arm would otherwise sample the whole corpus — so a naive pairing would measure *KD-on-subset vs CE-on-full-corpus*, penalizing the KD arm for seeing fewer unique tokens, which has nothing to do with KD. Resolution: **the screening block trains on the logit-covered slice, at full coverage (every position carries teacher signal in the KD arm)**. Rationale: this screening exists to settle a question we deliberately reopened, so it is sized for **maximum statistical power and maximum favorability to KD** — if span-KD cannot beat CE on-domain at full coverage, it will not do so at partial coverage, and D3 closes with maximum confidence. Both arms see byte-identical data: one variable.

**The covered slice is a uniform random sample** of the decontaminated training corpus, drawn with **seed = 20260719, fixed here** (the sealing date, so an auditor can verify the seed predates the corpus it draws from) and with the resulting file-list hash written into the manifest. The seed is fixed in this document rather than chosen at sampling time precisely so that "which tokens carry teacher signal" carries no residual degree of freedom, however innocent. This closes — conservatively and for now — the open observation that the "highest-value code" selection criterion for a KD subset is undefined: *which* tokens carry teacher signal is itself a design variable, and it is not stacked onto this comparison. Value-based selection is a separate, later experiment.

**Mixed-batch machinery is still built** (a declared fraction of each batch drawn from the KD window with KD loss, the remainder pure CE from the full corpus): that is the *rung-2 deployment* form specified in `PHASE64_TRAINING_PLAN.md` §2 (KD-on-subset + CE-on-rest), and it is the mechanism that ships if this screening says KD wins. The screening itself runs at full coverage; the deployment fraction is a rung-2 sizing question, not this one.

**Consequence for the 3060 calendar:** logits cover the *screening block* (all four arms), not a full KD curriculum — **0.05-0.10 B teacher tokens ≈ 6-12 h of 3060** rather than 24-36 h, independently recomputed by the Builder from the plan budgets. **Caveat carried to launch:** the teacher/student token ratio used (0.479) is measured on TinyStories with a V=1024 student; on code, against Qwen's code-efficient tokenizer and a V=2048/4096 student, it will differ. **Re-measure the ratio on the real corpus before committing 3060 time** — the figure above is the best available today, not the final one.

**This screening does not flip the main run.** Its role is pre-registered as evidence for the *rung-2* default: if span-KD wins by > 2σ_seed on-domain, the teacher-domain confound of the MVE is confirmed and KD-primary is re-opened at the rung-2 boundary (where a recipe change belongs, per D1). If CE wins again, **D3 closes for good** and the sequence-level fallback is retired with it.

## 5. The main run

Full curriculum C→D→E→F on the screening winners, CE-primary. This is the reference arm and the source of the rung-1 quality claim. `--save-stage-ckpt` on: the stage-D artifact (format `mve-stage-2`: model + gstep + kdc.pos + GradScaler, optimizer deliberately absent because it is rebuilt per stage) is the shared branch point for §6.

## 6. Branch A/Bs from the shared stage-D checkpoint

Each costs only E+F (~25% of the curriculum) and forks byte-identically (verified: brief §10).

**Mandatory branch-point assertion, applied to every branch and every resume.** The fork's val BPB at the branch point must equal the parent's val BPB at the save point, automatically asserted — not eyeballed. Rationale: three separate pieces of state that cross a stage boundary have now been found missing from a checkpoint (`kdc.pos`, `GradScaler`, and the QAT α — the last being a plain Python attribute, so absent from `state_dict` by construction, which silently left a branched arm **non-ternary**). The failure mode is common to the whole class: the restored model is wrong in a way that still produces a plausible loss. A BPB discontinuity at the branch point is the general detector for all of it, and it is cheap. **Any branch whose entry BPB does not match its parent's exit BPB is void, and the run does not proceed on it.**

**6.1 Upcycle ε-identity (the D→E investigation).** Baseline upcycle vs ε-identity upcycle — at insertion the router selects one replica per slice at weight 1/8 which, with `down` already magnitude-matched ×k, reconstructs the dense output **exactly**. Implemented as a **ramp** (router logits (1−α)·constructed-identity + α·learned, Switch aux ramped from 0 in step), never a hard identity.

Three outcomes, pre-registered before the arm exists, all informative:

| | transition shock | end-of-E vs baseline arm | replica-divergence telemetry |
|---|---|---|---|
| **H1** — E repays its own insertion cost | ≈ 0.000 | **better by order +0.07** | diverges |
| **H2** — cost is intrinsic to symmetry-breaking | ≈ 0.000 | converges to baseline | diverges |
| **H3** — dead replicas | ≈ 0.000 | not better | **flat** ← the detector |

H1 → the insertion is the target and the fix is adopted at the rung-2 boundary. H2 → the E *budget* is the target, not the insertion. H3 → the ramp schedule is wrong; re-tune before any adoption.

**6.2 No-recall control** (D4 clause 2, second read, now on data that has a real recall task).
**Pre-registered demotion, sealed here before any number exists: if the recall diagnostic on code shows a benefit ≤ σ_seed against this control, recall demotes to declared-v2** and the ladder gates are re-scoped accordingly. This is the clause armed at the MVE and carried forward; it is not re-interpreted.

**The instrument, specified here because the clause is only as good as what reads it.**

*Gate instrument — code-native probes.* Identifiers defined once and used later, drawn from **held-out, decontaminated** code; the measurement is the probability mass on the correct identifier at the use point. **The gate reads the difference between the recall arm and this no-recall control on the same probe set** — a within-probe paired control (matched position and local context) is kept as a variance reducer, not as the gate. **Probe distance must exceed the attention window (win = 128)**, or the single SWA layer carries the reference and the probe says nothing about the recall tier; report a **distance curve** rather than a point estimate — the tier should help more as distance grows, and the shape is worth more than the number.

**Distance-scale rule, corrected before the gate is ever read (2026-07-19).** The gated distances are **bounded by the trained context length**, with the longest gated distance ≤ half of it. Reason, measured not argued: on the run-3 finals the calibration fires at +1.23 nats / 50% top-1 at d=8 — the instrument sees a copy when there is one — while d≥512 reads exactly zero on **both** the recall and no-recall arms, because those checkpoints were trained at seq=512 and d=2048 is 4× beyond it. **The state has no reach there for reasons that have nothing to do with the recall tier**, so a fixed 128/512/2048 scale would decay to null by construction and fire the pre-registered demotion on an extrapolation artifact. This is an *instrument correction made before any rung-1 number exists, justified by a positive control* — not a reinterpretation after a null. Distances beyond the trained context are still measured and reported, **record-only, never gated**.

**Consequently, stage E's context extension is declared, not left implicit:** the curriculum already specifies "progressive context" at stage E (`PHASE64_TRAINING_PLAN.md` §4); rung-1 extends to **seq = 2048**, with micro-batch reduced to keep B·L — and therefore scan activation memory and tokens per optimizer step — constant. Gated scale then: **128 / 512 / 1024**, plus d=8 as the standing calibration sanity, plus 2048+ record-only. Without the extension the recall tier would be judged in a regime where the state still reaches, which is not the question the tier exists to answer. Rationale for code-native as the gate: this is associative recall in the form the product actually needs, and it needs no separate synthetic corpus to decontaminate.

*Calibration — synthetic MQAR, run once, NOT gated.* Its role is the **positive control for the detector**: an instrument never shown to fire is an assumption, not a measurement (the same discipline applied to the fork-check, which was validated by deliberately re-introducing the α bug). It establishes that the apparatus can resolve a recall benefit that exists, and it fixes the noise floor.

**Pre-registered reading of the 2×2, fixed now so no cell is interpreted after the fact:**

| synthetic | code | reading |
|---|---|---|
| + | + | recall **stays in v1** |
| + | − | mechanism works, code does not need it → **demote to declared-v2**, cleanly diagnosed |
| − | − | the tier is not being learned or used at all → **demote**, but flagged as a *training* failure (InfoNCE / λnce), and the architectural question is recorded as **untested, not answered** |
| − | + | incoherent → apparatus fault; investigate before the gate is read at all |

This specifies what the gate reads; it does not soften it. A null from a **calibrated** instrument demotes recall exactly as written.

**6.3 Dense-paired baseline** (gate 1 of DECISIONS §10) — **definition sealed: ACTIVE-parameter-matched**, realized as the pre-upcycle QAT-dense checkpoint continued from this same branch point at equal step budget (probe-4 pattern, as `PHASE64_TRAINING_PLAN.md` §4 specifies). The dense checkpoint is ~11.0 M against the MoE's ~11.2 M active, so the pairing is natural and costs one E+F-length branch rather than a separate training run.

Why active and not total: the product thesis is **active ≠ total** — we pay inference cost in active parameters and buy quality with total ones. Active-matched is therefore the claim that matters: *at equal running cost, the MoE beats the dense model*. Total-parameter-matched (a 30 M dense model) answers a different and more academic question, would require a full separate run, and is not our claim.

**Honesty constraint attached to this gate, binding on how the result is ever quoted:** active-matching is favorable to the MoE by construction (2.7× the parameters at equal active cost). The claim is stated as *"at equal active cost, +X BPB"*, never as an unqualified "MoE beats dense", and the total-parameter ratio is reported alongside it every time. **No total-matched control is bought at S0**: the ladder itself is that experiment, spread across rungs — S1 (E128, 105 M total at the same active) is what demonstrates quality scaling with total parameters at fixed running cost, and it is already gated by the D2 reading rule in §7.

**6.4 Compile gate** (measured in the first session, folded in per P61 — a compute-bound microbench does not compose onto a memory-bound engine): (a) two fresh processes, same seed, compiled → bit-identical; (b) resume across a session boundary numerically continuous (`mode="default"` pinned, autotuning not used — already closed in code); (c) compiled-vs-eager loss overlay within the AdamW-8bit tolerance; (d) net-positive over a 12 h session **including** warmup, counting one recompile per stage transition (four). **If (a) or (b) fails, compile is out of gate-bearing runs entirely** — the A≡C bit-identity diagnostic outranks throughput.

**RESOLVED 2026-07-19 — the rule fired: compile is OUT of rung-1's gate-bearing runs; every arm runs eager.** The local screen (WSL, torch 2.6.0+cu124, Triton present) returned (a) PASS 2.0931/2.0931, (c) PASS 0.0001 overlay, and **(b) FAIL**: against an eager control on the identical protocol that resumes *exactly* (2.0932 → 2.0932), the compiled arm gave 2.0885 / 2.0915 / 2.0915 across three repeats — not merely different from uninterrupted, but **mutually inconsistent, with two repeats agreeing despite different session counts**. The result depends on something that is not the training state (hypothesized: Inductor guard specialization against the first tensors a process sees, and/or compilation-cache state — mechanism is hypothesis, the measurement is not). `mode="default"` was already pinned and autotuning already excluded, so the pin does not close the hole.

**Why a screen may fire this rule when a passing screen could not certify:** the asymmetry is real and is banked as a principle — *a screen can falsify but not certify*. The hypothesized mechanism is a property of Inductor, not of a GPU model, so the failure is expected to transfer; and compile is opt-in, so the burden of proof sits with it. We do not spend Kaggle session-hours re-confirming a reproducible failure.

**Cost of the decision, stated plainly: it is the largest optimization measured so far** — 1488 vs 825 tok/s (+80%) at smoke config with a warm cache. It is declined anyway, because the bit-identity diagnostic is load-bearing apparatus and that trade was already adjudicated. **Parked with a concrete re-entry lead** rather than abandoned: if the failure is first-tensor guard specialization, then compiling against a *canonical warm-up batch at process start* — identical for a fresh process and a resumed one — should remove the dependence; that hypothesis is testable locally at zero cost. Second recorded datum for whoever picks it up: the compilation cache persists (3.5 min first run, 0.7 min second), so on Kaggle a cache living in `/kaggle/working` under Persistence would be paid once rather than per session, which changes criterion (d)'s arithmetic. Re-entry is a rung-boundary item, only if calendar bites.

## 7. Gates read at rung-1 (DECISIONS §10 property gates + this rung's additions)

1. **Quality vs matched dense** (§6.3) — the rung gate.
2. **Sparsity bands** 92/79 reproduced.
3. **Router health + i.i.d.-union sanity** (locality at scale = finding, not failure).
4. **Recall** — code-native probes (gate) + synthetic MQAR (calibration), with §6.2's demotion clause and its 2×2 reading.
5. **QAT-gap** recorded vs scale.
6. **ε-identity continuity at every switch** (project law): each transition shock reported; a violation is a finding to explain, not a number to bury.
7. **D2 reading rule** (opened by the MVE, sealed here): the **replica-collapse ablation is a standing instrument at every rung**. S0's value anchors the series. At S1/E128, if the differentiation gain is not materially larger than S0/E32's, **D2 is re-opened at that boundary** — E as the sole capacity dial would not be delivering. This is a reading rule, not a gate change.
8. **Throughput** measured on 2×T4 → the rung-2 cost table (the 3860 floor is checked here, per-protocol).

HumanEval stays **record-only** (no gate — anti-Goodhart). Decode hygiene locked.

## 8. Launch prerequisites (all must be true before the owner launches)

- [ ] HF Read-type token with Stack-v2 terms accepted; AWS S3 credentials (owner; env only, never committed)
- [ ] WS3 pipeline complete, **P62 decontamination J=0.5 executed**, manifest + hash recorded here and verified at training start — manifest hash: `__________` (filled at launch)
- [ ] WS4 apparatus smoked and STOPped
- [ ] Challenger teacher logits produced for the screening block — sized to it only (~6-12 h of 3060, off-Kaggle), covered slice drawn with seed 20260719, file-list hash in the manifest, teacher/student token ratio re-measured on the real corpus first
- [ ] This document pushed, and the MM's commit visible on origin

Token budgets, per-arm session counts, and the data manifest hash are the only fields filled at launch. Everything above them is sealed now.

## 9. What rung-1 costs, stated openly

Arm inventory: **screening 4 arms × 15% of stage C ≈ 33% of a rung** + **main run 100%** + **three E+F branches ≈ 75%** ≈ **2.1 rung-equivalents**, i.e. roughly **+108% instrumentation overhead** against the +50-80% assumed in `PHASE64_TRAINING_PLAN.md` §5.

This is declared rather than trimmed, and the science is not cut to fit a bracket written earlier — **costs re-price, gates do not**. Two reasons it is the right call anyway: rung-1 is where the recipe itself is validated and three open questions are settled, so heavy instrumentation is exactly what it is for; and **the overhead is front-loaded and does not repeat** — the screening block decides vocabulary and x_proj once for the whole ladder, so S1 and S2 carry only their per-rung property gates and the standing replica-collapse ablation. S0 re-prices to roughly **70-150 session-hours**; against a 600-1000 h ladder this stays inside the $0 path.
