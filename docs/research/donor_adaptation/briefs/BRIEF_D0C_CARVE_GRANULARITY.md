# BRIEF D0c — Was D0's negative a statement about carving, or about ONE carve granularity?

**Status: PRE-REGISTERED. Written and pushed before the run, per standing practice.**
**Author: the Adapter / Principal. Date: 2026-09-03.**
**Depends on: `probes/D0_COACTIVATION.md` Parts I-III (commit `7c49a1b`), `results/d0b_rho_floor_donor.json`.**

---

## 1. The question

D0 Part III closed FFN co-activation carving as a negative at 1.5B on this number:

> carving all 28 layers at **E = 32, k = 8** (25% active) with an **oracle** router costs
> **+1.09062 BPB = 218 σ_seed**.

That measurement was made at **one** expert granularity, and it is the **coarsest one D0 itself
swept**: E = 32 means **280 neurons per expert**.

D0's own fidelity table — `d0_coactivation.analyse.json` → `fidelity[L].rows`, partition
`coactivation_balanced`, `relerr` (lower is better), **at matched achieved activation fraction** —
says granularity matters, and says it monotonically:

| act. fraction | layer | E=32 (280 neurons) | E=64 (140) | E=128 (70) | E=32 → E=128 |
|---|---|---|---|---|---|
| 0.06250 | L7 | 0.8684 | 0.8396 | **0.8190** | −0.0494 |
| 0.06250 | L14 | 0.9033 | 0.8899 | **0.8663** | −0.0370 |
| 0.06250 | L21 | 0.8152 | 0.7452 | **0.7051** | **−0.1101** |
| 0.06250 | L27 | 0.4899 | 0.4538 | **0.3839** | **−0.1060** |
| 0.03125 | L7 | 0.9172 | 0.8879 | **0.8684** | −0.0488 |
| 0.03125 | L21 | 0.8784 | 0.8243 | **0.7776** | **−0.1008** |
| 0.03125 | L27 | 0.5767 | 0.5573 | **0.4944** | −0.0823 |

**Every one of the 14 cells where two or more `E` values exist at a matched activation fraction
favours the finer carve, except L1 at 0.0625 and 0.125 where the three values sit within 0.006 of
each other.** In the deep layers the gain from E = 32 → E = 128 is **0.08 to 0.11 relerr at constant
budget.** That is not a small effect and it was never carried through to BPB.

> **The question this brief pre-registers:**
> **Is +1.09 BPB a property of carving this donor, or a property of carving it at 280 neurons per
> expert?**

## 2. Why this is not a fishing expedition — the legality argument that makes it sharp

D0 §8 tested engine legality with the **per-organ** floor: at `d_model = 1536` and 0.5 B/weight, one
48 KiB block holds `49152 / (1536 × 0.5) = 64` neurons per organ. Under that floor:

| E | neurons/expert | legal per-organ (floor 64)? |
|---|---|---|
| 32 | 280 | yes — **this is where BPB was measured** |
| 64 | 140 | yes |
| 128 | 70 | yes — at the floor |
| 256 | 35 | no |

**E = 128 is legal under the very floor D0 used, and its BPB was never measured.** This brief's
primary arm therefore requires **no new layout assumption of any kind**. It is a gap in the existing
sweep, not a new proposal.

Separately, and *not* load-bearing for the primary arm: the project records a **second** ρ-floor,
the **interleaved** one — `d0_layout.json` / `d0b_rho_floor_donor.json`:
`neurons_per_48KB_interleaved = 21.333` at `d_model = 1536`, because an interleaved neuron carries
2304 B rather than 768 B. Under that floor E = 256 (35 neurons) and even E = 384 (≈23 neurons) would
be legal. **Whether the engine adopts an interleaved layout is a design decision this brief does not
make and does not assume.** E = 256 is registered below as a clearly-marked secondary arm,
conditional on that decision, and its result may not be reported as if the layout existed.

**The coincidence worth naming:** D0 Part II found the co-activation structure strongest at block
**12-21 neurons** and collapsing by block 64-140. The per-organ floor at this donor's width is **64
neurons** — i.e. the engine's granularity floor at `d_model = 1536` sits almost exactly on top of the
cliff where D0 measured the structure dying. `d0b_rho_floor_donor.json` shows that floor falls to
**12 neurons per organ** at `d_model = 8192`. **This donor may be the worst width in the range for
this mechanism.** That is a hypothesis, it is stated here so it cannot be retrofitted later, and this
brief does not test it — S1's scale arm is where width is under test.

## 3. Arms — all at MATCHED ACHIEVED ACTIVATION FRACTION 0.25, oracle router, all 28 layers

Identical to the Part III carve in every other respect: same donor (Qwen2.5-1.5B, revision
`8faed761d45a263340a0528343f099c05c9a4323`), same eval slice (`heldout`, 24 × 512, seed 1234,
`ids_sha256 = a1a48dc9...`, 12,264 predicted tokens, 51,870 scored bytes), same `cluster_seed = 7`,
same B3 repair applied, per-sequence nats stored so every SE is **paired**.

| arm | E | k | neurons/expert | legality | status |
|---|---|---|---|---|---|
| **A0** | 32 | 8 | 280 | per-organ | **replication** of the Part III number — must reproduce +1.09062 |
| **A1** | 64 | 16 | 140 | per-organ | primary |
| **A2** | 128 | 32 | 70 | per-organ, at the floor | **primary — this is the arm the brief exists for** |
| N0 | 32 | 8 | 280 | — | random-partition null at E=32 (replication: must reproduce +1.81114) |
| N1 | 64 | 16 | 140 | — | random-partition null at E=64 |
| N2 | 128 | 32 | 70 | — | random-partition null at E=128 |
| *S1* | *256* | *64* | *35* | *interleaved only* | *SECONDARY, conditional — see §2* |
| *S1n* | *256* | *64* | *35* | — | *random-partition null at E=256* |

### 3.1 ⚠ THE CONTROL THAT MAKES THIS READABLE — non-negotiable

**Every co-activation arm must be paired with a random-partition null at the SAME E.**

A finer partition can track any active set better, for reasons that have nothing to do with
co-activation structure — smaller blocks waste less of their budget on inactive neurons whatever the
ordering. **If the null improves with finer E by as much as the treatment does, then granularity is
not evidence for the mechanism and D0's negative stands unchanged.** The quantity that decides this
brief is therefore **not** `BPB(E)` but:

> **the co-activation-minus-null gap, as a function of E, at matched achieved activation.**

At E = 32 that gap is already measured: **−0.72051 BPB** (`d0_carved_bpb_paired.json` →
`paired_coact_vs_null.all_layers`). Running the nulls is half the cost of this brief and it is the
half that makes it a measurement rather than an anecdote.

### 3.2 Achieved, never requested

Report **achieved** activation fraction per arm. `E = 128, k = 32` gives a nominal 0.25; if the
achieved value differs, the comparison is against achieved, and any arm whose achieved fraction
differs from A0's by more than 0.002 is reported as unmatched rather than compared.

## 4. Pre-registered decision rule — fixed before any result

Let `Δ(E)` = BPB(co-activation carve at E) − BPB(baseline), and `G(E)` = BPB(co-activation at E) −
BPB(random null at E). Baseline = **0.7675950**. σ_seed = **0.005**.

| outcome | condition | what it means |
|---|---|---|
| **GRANULARITY-BOUND** | `Δ(128) < Δ(32) − 0.20` **and** `G(128) < G(32) − 0.10` | the negative is granularity-local; D0's headline must be restated as an E=32 statement and the axis reopens at finer E and at larger width |
| **PARTIAL** | `Δ(128) < Δ(32) − 0.20` **but** `G` does not improve | finer carving helps, but not because of co-activation; the mechanism claim does not strengthen and the engine gets a cheaper knob, not a new lever |
| **GRANULARITY-INVARIANT** | `|Δ(128) − Δ(32)| ≤ 0.20` | D0's negative is about carving, not about granularity. **It hardens.** Report it as such and close this axis at 1.5B |
| **WORSE** | `Δ(128) > Δ(32) + 0.20` | finer is worse; say so, and the relerr→BPB link is broken and must itself be investigated |
| **INCONCLUSIVE** | any arm's paired SE makes its band span the nearest threshold | report as inconclusive; do not pick the nearer label |

**0.20 BPB is 40 σ_seed and is deliberately coarse:** `Δ(32) = 1.09062`, and a granularity effect
worth reopening a closed axis must be a large fraction of that, not a detectable sliver. The
threshold is set here, before the run, precisely so it cannot be tuned to the answer.

**No extrapolation.** If E = 256 is not run, the trend across E = 32/64/128 may **not** be extended
to it in prose, in a table, or in a figure. This programme has twice been caught transplanting a
quantity validated at one setting into another, and both times it was a monotone trend that looked
safe to extend.

## 5. What this brief does NOT claim and does NOT test

- It does **not** claim the co-activation carve is viable. `Δ(32) = 218 σ_seed`; even a large
  granularity effect leaves a number that has to be argued for on its own.
- It does **not** test width. Every arm is `d_model = 1536`. The §2 coincidence between the 64-neuron
  floor and the 64-140 collapse is a **hypothesis for S1's scale arm**, not a result of this brief.
- It does **not** revisit the router. The router stays an oracle, so every Δ here remains a ceiling
  no real router can reach. **The two biases now run in opposite directions and both must be stated
  together in any summary:** the oracle router flatters the carve, the coarse granularity punishes it.
  Part III stated only the first. That is a defect in Part III's framing and it is recorded here.
- It does **not** touch D4's reconstruction route, which is a different mechanism entirely.

## 6. Cost

The Part III run measured 5 arms in ~34 minutes wall-clock on CPU (`d0_carved_bpb_paired.log`: 355 s
to 455 s per arm, 12,264 tokens each). This brief adds **6 arms** (A1, A2, N1, N2 + the two
replications A0, N0) ≈ **40 minutes**, plus partitioning. The conditional secondary arms add ~15
minutes. **It is the cheapest open question in the programme and it bears directly on a conclusion
already written down as closed.**

## 7. Reporting

Extend `probes/D0_COACTIVATION.md` with **Part IV**, or a standalone `probes/D0C_GRANULARITY.md` if
Part III is under audit at the time. Report: the achieved activation per arm, `Δ(E)` and `G(E)` with
paired SEs, the outcome label from §4's table verbatim, and a restatement of Part III §15's verdict
in the light of it — including, if §4 lands on GRANULARITY-INVARIANT, an explicit note that the
negative is now **stronger** than Part III stated, because it survived a granularity control.
