# D0c — Carve granularity: was +1.09 BPB a statement about carving, or about ONE granularity?

**Outcome: PARTIAL** (pre-registered label, §4 of the brief, read verbatim).
**Date: 2026-09-04.  Donor: Qwen2.5-1.5B, revision `8faed761d45a263340a0528343f099c05c9a4323`.**
**Brief: `briefs/BRIEF_D0C_CARVE_GRANULARITY.md` @ `4a98c89` — pushed before any measurement existed.**
**Raw: `benchmarks/donor_adaptation/density/results/d0c_granularity.json`; per-sequence nats in
`results/d0c_arms/*.npy`; label caches in `results/d0c_labels/`.**
**Driver: `benchmarks/donor_adaptation/density/d0c_granularity.py`.**

> **This report has not been audited.** It was written by the same figure that specified the brief and
> read the numbers. The Controller audit is owed and is listed in §9.

---

## 1. What was measured

D0 Part III closed FFN co-activation carving at 1.5B on a single number: carving all 28 layers at
`E = 32, k = 8` (25% of experts active) with an **oracle** router costs **+1.09062 BPB**. That is
`218.1 σ_seed` at `σ_seed = 0.005`.

`E = 32` means **280 neurons per expert**, and it is the **coarsest** of the three granularities D0's
own fidelity sweep contains. The brief asked whether the headline was a property of carving this
donor or a property of carving it at 280 neurons per expert, and pre-registered the decision rule
before the run.

Nine arms, all at **matched achieved activation fraction 0.25**, oracle router, all 28 layers,
identical eval slice, `cluster_seed = 7`, B3 seeding repair applied:

| arm | E | k | neurons/expert | kind | role |
|---|---|---|---|---|---|
| baseline | — | — | — | dense | reference |
| A0 / N0 | 32 | 8 | 280 | coact / null | **replication of Part III** |
| A1 / N1 | 64 | 16 | 140 | coact / null | primary |
| A2 / N2 | 128 | 32 | 70 | coact / null | **primary — the arm the brief exists for** |
| S1 / S1n | 256 | 64 | 35 | coact / null | **secondary, conditional — see §6** |

## 2. Gates passed before any result was read

**2.1 Replication gate — bit-exact, hard-stop on failure (`exit 3`).** The driver recomputes Part
III's four standing numbers and refuses to continue unless each is within `1e-6`:

| quantity | Part III | D0c | abs diff |
|---|---|---|---|
| baseline BPB | 0.7675949641196624 | 0.7675949641196624 | **0.0** |
| Δ(A0) | 1.090622885724487 | 1.090622885724487 | **0.0** |
| Δ(N0) | 1.8111363812608539 | 1.8111363812608539 | **0.0** |
| G(32) | −0.7205134955363668 | −0.7205134955363668 | **0.0** |

Not "within tolerance" — **identical to the last stored digit**, in a separate process, on a separate
day, at the same `torch.set_num_threads(6)`. The forward path, the partitioner and the statistics
half all reproduce.

**2.2 Partition replication.** `results/d0c_labels/labels_E32.npz` vs Part III's
`results/d0_carved_labels_E32.npz`: coact labels identical on **all 28 layers**, null labels identical
on **all 28 layers**.

**2.3 Expert sizes exact at every E.** `d_ffn = 8960` is divisible by 32/64/128/256, so no remainder
branch is taken: 280/140/70/35 neurons per expert, `coact_all_experts_exact` and
`null_all_experts_exact` both true at every E.

**2.4 Achieved, never requested (brief §3.2).** Achieved activation fraction is **exactly 0.250000**
on all eight carved arms; `abs_diff_vs_A0 = 0.0` for every one. No arm is unmatched, so every
comparison below is like-for-like at the same budget.

**2.5 Eval slice.** `heldout`, 24 × 512, seed 1234,
`ids_sha256 = a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65`, 12,264 predicted
tokens, 51,870 scored bytes, `n_rejected = 0`. Same slice as every other probe in this programme.

## 3. The result

All SEs are **paired** sequence bootstraps over the 24 sequences. `Δ(E)` = carve − baseline.
`G(E)` = coact − null at the same E; **G negative means the co-activation ordering beats a random
ordering.**

| E | neurons/expert | **Δ(E)** | ±SE | **Δ_null(E)** | ±SE | **G(E)** | ±SE | Δ in σ_seed |
|---|---|---|---|---|---|---|---|---|
| 32 | 280 | **+1.09062** | 0.06981 | +1.81114 | 0.14000 | **−0.72051** | 0.11475 | 218.1 |
| 64 | 140 | **+0.82507** | 0.05792 | +1.37690 | 0.07264 | **−0.55183** | 0.04958 | 165.0 |
| 128 | 70 | **+0.70611** | 0.05556 | +1.12937 | 0.06278 | **−0.42326** | 0.03014 | 141.2 |
| *256* | *35* | *+0.61627* | *0.05071* | *+0.87706* | *0.05348* | *−0.26078* | *0.01728* | *123.3* |

*E = 256 is italicised throughout because it is the secondary arm and is legal only under a layout the
project has not adopted. See §6.*

### 3.1 The two pre-registered statistics

Both read off `decision` in the JSON, computed on the primary arms only:

| quantity | value | paired SE | ci95 | threshold (brief §4) | verdict |
|---|---|---|---|---|---|
| **Δ(128) − Δ(32)** | **−0.38452** | 0.04498 | [−0.47267, −0.29636] | < −0.20 | **crossed** — the whole band clears it |
| **G(128) − G(32)** | **+0.29725** | 0.11034 | [+0.08100, +0.51351] | < −0.10 to count as improvement | **not crossed, and the sign is wrong** |

`band_straddles_threshold = false`: neither band spans its threshold, so the INCONCLUSIVE row of §4
does not apply and the label is read directly.

### 3.2 The outcome, verbatim from the brief

> **PARTIAL** — `Δ(128) < Δ(32) − 0.20` **but** `G` does not improve.
> *"finer carving helps, but not because of co-activation; the mechanism claim does not strengthen and
> the engine gets a cheaper knob, not a new lever"*

## 4. Reading it

**4.1 The brief's premise was correct: +1.09062 was granularity-local.** Going from 280 to 70 neurons
per expert removes **0.38452 BPB** at an identical activation budget — `76.9 σ_seed`, and the ci95
lower edge alone (0.29636) is `59.3 σ_seed`. Part III's headline must therefore be restated as an
**E = 32 statement**, not as the cost of carving this donor. "Carving costs +1.09 BPB" is not correct
as written and should be cited as "+1.09062 BPB at 280 neurons per expert".

**4.2 The mechanism claim does not strengthen — it weakens.** The null falls **further** than the
treatment over the same range: `Δ_null` drops **0.68177** (1.81114 → 1.12937) where `Δ` drops
**0.38452**. Consequently `G` moves *toward zero*: −0.72051 → −0.42326, a change of **+0.29725** whose
ci95 excludes zero. Finer partitions track the active set better regardless of how neurons are ordered
— exactly the confound §3.1 of the brief was written to catch — and they do it *more* for a random
ordering than for the co-activation ordering.

**This is the reason the null control was made non-negotiable.** On the treatment arms alone, the same
run reads as "finer granularity substantially rehabilitates the co-activation carve". It does not. The
nulls cost half the run and they are the half that made it a measurement.

**4.3 A descriptive observation, offered with no error bar and no pre-registration.** Expressed as a
*fraction* of the null's own damage, `−G/Δ_null` is 0.3978 / 0.4008 / 0.3748 across E = 32/64/128 —
roughly flat over the legal range, falling to 0.2973 at E = 256. So the ordering keeps saving about
two fifths of what a random partition loses; the absolute gap shrinks because the quantity it is a
fraction of shrinks. **No SE was computed for this ratio and no threshold was pre-registered for it.
It describes four measured rows; it is not a finding, and the decision rule is on absolute `G`, which
is what §3.2 reports.**

**4.4 Nothing here makes the carve viable.** The finest **legal** granularity in this sweep still
costs **+0.70611 BPB = 141.2 σ_seed**, with `frac_tokens_worse = 0.8332` — the carve is worse on 83%
of the 12,264 individual tokens, so this is not a tail effect. And the router is an **oracle**: it
selects experts by retained activation energy computed from the true activations, so **every Δ in this
report is a floor no real router can reach**. The two biases run in opposite directions and both
belong in any one-line summary of D0: *the oracle router flatters the carve, the coarse granularity
punished it.* Part III stated only the first; the brief recorded that defect in advance, and this
report is where both are stated together.

## 5. What changed, and what did not

| claim | before D0c | after D0c |
|---|---|---|
| "the co-activation carve costs +1.09062 BPB" | headline of Part III | **restated: that is the E=32 cost.** At the finest legal E it is +0.70611 |
| "granularity is worth measuring" | untested at BPB | **confirmed: 0.38452 BPB over E=32→128** |
| "co-activation ordering beats random" | true at E=32 (−0.72051) | **still true at every E, but shrinking:** −0.72051 → −0.42326 → *−0.26078* |
| "finer granularity is evidence FOR the co-activation mechanism" | plausible, untested | **falsified.** The null improves more (0.68177 vs 0.38452) |
| "the carve is viable at 1.5B" | no (218 σ_seed) | **still no (141 σ_seed at the finest legal E)** |

D0's negative at 1.5B **stands**. It is smaller than Part III said and its mechanism story is weaker
than Part III said, and those two corrections point in opposite directions.

## 6. The secondary arm, and why it is fenced

`E = 256` is **35 neurons per expert**. D0 §8's per-organ legality floor at `d_model = 1536` is
`49152 / (1536 × 0.5) = 64` neurons — so **E = 256 is illegal under the floor D0 used**, and E = 128
(70 neurons) is the finest granularity that clears it. E = 256 is legal only under the **interleaved**
floor of 21.333 neurons (`d0b_rho_floor_donor.json`), which requires a layout decision the project has
not made. Per brief §2 its result **may not be reported as if that layout existed**, and it is reported
here as: *at E = 256, Δ = +0.61627 and G = −0.26078, conditional on an interleaved layout that does not
currently exist.*

Its value is diagnostic rather than practical: it extends the same monotone pattern one step further —
`Δ` still falling, `G` still moving toward zero, and faster in relative terms (`−G/Δ_null` drops from
0.3748 to 0.2973). **Per brief §4's last paragraph, no trend in this report may be extrapolated to any
E that was not run.** E = 384 and beyond are not discussed.

## 7. The programme-level pattern this is the first instance of

D0c and D4b are the two cases now on record where a **closed negative turns out to have been measured
at one extreme of a free parameter that was never swept** — here, the coarsest of three granularities
the probe's own fidelity table already contained. The measured cost of that omission is **0.38452 BPB
on the headline number**, and the correction did not change the verdict.

> **A negative is only as strong as the sweep behind the setting it was measured at.**

This is a working rule, not a result. It cost ~50 minutes of CPU to check here, and the check was
specified before the numbers existed.

## 8. Reproduce it

```powershell
$dir = "D:\_THINGS\Progetti\SiliconLLM\benchmarks\donor_adaptation\density"
$env:D_THREADS="6"; $env:D0C_SECONDARY="0"     # primary arms; "1" adds S1/S1n
python $dir\d0c_granularity.py
```

Resumable: any `results/d0c_arms/<arm>.npy` already on disk is reloaded and only the missing arms are
computed. The replication gate runs after `baseline`/`A0`/`N0` and exits 3 on failure. `D_THREADS=6`
is **part of the harness**, not a performance setting: fp32 matmul reduction order depends on thread
count, and the bit-exactness in §2.1 is only a fair test at the same count. Peak RSS 9.05 GB.
Wall clock 131–159 s per arm, 9 arms.

## 9. Owed

- **Controller audit of this report.** Written by the figure that held the hypothesis; not independent.
- Part III's headline sentence needs the E=32 qualifier applied inside `D0_COACTIVATION.md` itself —
  that file is under audit and must not be edited from here.
- Width is untested. Every arm is `d_model = 1536`. The brief's §2 hypothesis — that the 64-neuron
  per-organ floor at this width sits almost exactly on the 64–140-neuron block where D0 Part II
  measured the co-activation structure collapsing, making this donor possibly the **worst** width in
  the range for this mechanism — remains a hypothesis. **S1's scale arm is where it is tested**, and
  that arm is now unblocked and queued on Kaggle.
