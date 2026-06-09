# Phase 44–45 Synthesis — L2 Boundary Memory: what closed, what's left

**Status:** filone 45.A–D CHIUSO. Consolidation pause before choosing the next axis.
Nothing here proposes Phase 46 — it records what is proven and frames three candidate
directions for a later decision.

Baseline reference throughout: **C2.A** (`weights/phase43c2_C2A.bin`, magic `0x53454539`),
the SEE-only Phase-43 champion, val BPB **2.2593**. Promotion gate (carried since Phase 43):
`BPB ≤ 2.2543 AND topBi ≤ 8 AND altLp ≤ 2 AND nameWst ≤ 20 AND runWst ≤ 5`, word-gate at
both T=0.65 and T=0.55.

---

## 1. What we proved, 44.A → 45.D

**The core finding (44):** a 64-D boundary-memory feature block (L2), updated by a gated EMA
at entropy-high boundaries and concatenated to the 192-D SEE features before the readout,
**improves compression** — but **no configuration yields a stable closed-loop generator**.

- **44.A–C** — L2 boundary memory + homeostasis brakes (nb_decay, cooldown) + delta-mix
  write established the two regimes that dominate everything after: **delta** (mix1, scale0.5)
  and **D1** (entropy-high gate, mix0.5, scale0.5, alpha0.99).
- **44.D** — readout/scale tuning. Compression champion **D3 delta** (`phase44d_D3_delta_s50.bin`)
  ≈ **2.2212** but generation-unstable. **D1 / D1cap20** ≈ **2.252×**, near-stable but still
  fails the T=0.55 word-gate.
- **44.E / 44.I** — L2 logit-contribution caps and endogenous trust-fatigue. No candidate;
  caps that tame the loop also erase the gain.
- **44.G** — inconclusive.
- **44.H** — causal forensics: L2 is a **co-cause** of the loop (zeroing/freezing it after a
  warning breaks the attractor), but a logit-side gate is not enough — L2 acts as a
  **persistent contextual pressure**, not a per-bigram push.
- **Residual limit:** topBi ≈ 10–11 at T=0.55. Low-temperature sampling does not save it
  (structural, not a sampling artifact).

**45 — substrate-side write dynamics (act on the L2 WRITE, not the readout):**

- **45.0** — measured the L2 scale event-conditioned. The discriminant between stable and
  unstable is **relative move** `‖ΔL2‖/‖L2‖`: D1 ≈ 1e-4 (state frozen, ‖L2‖~128k, a fixed DC
  bias the readout co-adapted to) vs delta ≈ 1.31 (state volatile, ‖L2‖~13, reorients >100%
  per write). Raw ‖L2‖ is *inverted* vs stability → norm is not the pressure.
- **45.A — amplitude (rel_move cap): NEGATIVE.** Every cap pushed BPB to ~2.256 (worse than
  D1), L2/SEE ratio collapsed 4.8%→1.6%. The useful delta signal lives in the strong
  reorientation; limiting amplitude kills it.
- **45.B — write geometry (blend / orthogonalize / energy-normalize): NEGATIVE.** No simple
  geometry stabilizes. **Key reveal:** delta has `cos(write, L2_old) ≈ −0.70` — the useful
  write is **anti-aligned** to the accumulated state. It is a **correction / reversal**, not
  accumulation; making it parallel (blend) or removing the axial part (ortho) destroys the
  gain. In closed loop that anti-L2 reversal becomes oscillatory.
- **45.C — hard timing / reversal gates + deterministic thinning: NEGATIVE.** Gating the
  write on `cos`/`rel_move`, or thinning 1-in-N, **switches L2 off**: every config collapses
  to the SEE baseline (2.2593, L2 ratio ~0%); the margin gate (C3) even diverges to 7.49.
  The `cos≈−0.70` anti-alignment is a property of the *generated* closed loop, **not** of the
  teacher-forced training trajectory, so gating on it drops the write instead of preserving
  it. *(Note: this disproves these specific hard gates, not soft timing in general.)*
- **45.D — per-event causal attribution (diagnostic, no training): the decisive answer.**
  On the co-adapted C0_delta readout, teacher-forced over 206,606 write events, the marginal
  counterfactual `dloss = loss(next byte | L2_old) − loss(next byte | L2_new)`:
  - **Net +26,325 bits** (57.9% helpful / 42.1% harmful) — the delta writes are genuinely
    **compression-positive** (consistent with C0 2.2212 < SEE 2.2593). The signal is real.
  - **But inseparable:** `r(dloss, signal) ≈ 0` for **every** internal signal
    (cos −0.017, rel_move −0.011, write_norm −0.004, surprise −0.032, entropy +0.010). The
    best single-threshold "kept net" equals keep-all (~26.3k) and never approaches the oracle
    (61.1k). Helpful and harmful writes are statistically indistinguishable on every internal
    axis.

This retro-explains 45.A/B/C: they all gated on signals (`rel_move`, `cos`) that are
**uncorrelated with whether a write helps** → they removed writes blindly and erased the net.

---

## 2. Best artifacts

| role | artifact | BPB | status |
|---|---|---|---|
| SEE-only baseline | `phase43c2_C2A.bin` (C2.A) | 2.2593 | stable reference, Phase-43 champion |
| **compression champion** | delta / D3 / 45B0 / 45C0 (`phase44d_D3_delta_s50.bin`, `phase45c_C0_delta.bin`) | **≈ 2.2212** | **unstable** — fails word-gate T=0.55 |
| **near-stable generator family** | **D1 / D1cap20** (`phase44f_F0`, D1cap20) | **≈ 2.252×** | best stability so far, **still fails T=0.55** (topBi ~10–11) |

The compression champion and the stable-ish generator are **different regimes**, and no
intervention found so far moves one toward the other without losing its defining property.

---

## 3. Negative results — do NOT repeat

- **Scalar gain** on the L2 contribution (single multiplier) — pre-44 / 43 eta sweeps.
- **Routing / regime-prior mixtures** (Phase 32–34) — partial gains, never cleared the gate,
  not merged.
- **Generation hacks** — low-temperature, argmax tricks; the loop is structural, sampling
  does not fix it.
- **L2 logit caps + endogenous fatigue** (44.E / 44.I) — taming the loop erases the gain.
- **rel_move amplitude caps** (45.A).
- **Write geometry** — blend / orthogonalize / energy-normalize (45.B).
- **Hard timing / reversal gates + deterministic thinning** (45.C).
- **Per-event filtering on internal signals** (45.D) — proven inseparable; cos, rel_move,
  write_norm, surprise, entropy all carry **zero** information about write usefulness.

---

## 4. Updated hypothesis

> Delta L2 is a **teacher-forced compression signal**: globally / on-average useful
> (+26k bits, the BPB gain is real) but **not locally controllable** — the helpful writes
> cannot be isolated from the harmful ones by any internal per-event signal.

Therefore the next jump toward an LLM is **no longer in the per-event L2 write**. Any
mechanism that decides, write-by-write, *whether / how / when* to write L2 from internal
signals is excluded by 45.D. The lever must move elsewhere: the SEE substrate, the memory
*level*, or delta's *role* (training-time vs inference-time).

---

## 5. Three candidate directions (NOT yet implemented — pick one)

**A. SEE-side closed-loop stability on the D1/mix/H0 family.**
Drop delta as the unstable lever; take the near-stable D1 regime (2.252×) and attack the
residual T=0.55 loop from the SEE substrate / closed-loop dynamics rather than the L2 write.
Goal: make D1 clear the word-gate without touching delta. Lowest-risk continuation of the
existing stable artifact.

**B. Change the memory *level*: a slower phrase/episode L3 above D1.**
The failure is a *volatile* per-event delta-write. Instead add a **slower, higher-level**
memory (phrase/episode timescale, L3) on top of D1 — a memory that integrates over longer
spans and changes rarely, rather than reorienting every boundary. Different hypothesis:
the generative-useful memory lives at a coarser timescale than the byte-boundary write.

**C. Use delta as auxiliary / teacher during substrate training only.**
Keep delta entirely out of inference state. Use the delta signal (which we now know is
compression-positive) as an **auxiliary target / teacher** while training the SEE substrate
or readout, so the substrate *internalizes* what delta captures, with no volatile L2 state at
generation time. Converts a non-controllable inference mechanism into a training signal.

---

*Decision pending: read this, choose the axis. No Phase 46 until then.*
