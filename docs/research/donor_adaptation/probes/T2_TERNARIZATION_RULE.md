# T2 — Was +4.74 BPB the ternary FORMAT, or one naive RULE for reaching it?

**Outcome: `RULE-HELPS` — the pre-registered label, unchanged by anything found afterwards.**
**Date: 2026-09-04. Donor: Qwen2.5-1.5B rev `8faed761…`. CPU only, no gradients, 2888 s = 48.1 min.**
**Brief: `briefs/BRIEF_T2_TERNARIZATION_RULE.md` @ `1286faf` (pushed before the run).**
**Code: `benchmarks/donor_adaptation/ternary/t2_rules.py`. Data: `density/results/t2_rules.json`.**

> **Not audited.** Written by the figure that ran it.

---

## 1. The answer in one line

**It was the rule.** T1's `+3.309` on the FFN organs was not a property of the ternary format; it
was a property of `scale = mean|w|` and round-to-nearest. A better rule removes **62%** of it with
no training, no gradients and no format change, and every arm below still lands in `{-1,0,+1}` with
one fp32 scale per output row — the format `donor_engine.c` already consumes, unmodified.

## 2. Gates, before any result is read

| gate | required | measured | |
|---|---|---|---|
| eval slice | `ids_sha256 = a1a48dc9…`, 24×512, seed 1234 | identical, 51,870 scored bytes | ✅ |
| baseline | `0.7675950` | **`0.767594958`** | ✅ |
| **arm I** (identity substitution through the same code path) | Δ exactly 0 | **`+0.000000000`** | ✅ |
| **arm R0** (replicate T1's rule) | `+3.309099` | **`+3.309099`** | ✅ |
| calibration disjointness | `calib` and `heldout` different corpus halves | asserted in-run, sha256s recorded | ✅ |

Arm I is the instrument control this brief exists to have. T1's control was mis-specified — it
demanded random signs be ≫ the treatment, which conflates an instrument property with a scientific
claim, and returned VOID on sound numbers. **§4 below shows why that mis-specification mattered
more than it looked: random signs are not ≫ the treatment. They are indistinguishable from it.**

## 3. Every arm

Δ = BPB(arm) − BPB(base), paired sequence bootstrap over 24 sequences, byte-weighted.
`σ_seed = 0.005`.

| arm | rule | BPB | Δ | SE | σ_seed | zero frac |
|---|---|---|---|---|---|---|
| base | none | 0.767594958 | — | — | — | — |
| **I** | identity | 0.767594958 | **+0.000000** | 0.000000 | 0 | 0.0000 |
| **R0** | BitLinear158, `α = mean\|w\|`, RTN | 4.076694081 | **+3.309099** | 0.152596 | 662 | 0.3166 |
| R1 | TWN, `Δ = 0.7·mean\|w\|` | 3.851979052 | +3.084384 | 0.090542 | 617 | 0.4307 |
| R2 | per-row `Δ/α` search on L2 | 3.390467023 | +2.622872 | 0.104125 | 525 | 0.4831 |
| **R3** | **same search, weighted by activation RMS** | 2.476967449 | **+1.709372** | 0.068968 | 342 | 0.4642 |
| Z | R0's codes, **random signs** | 4.140276456 | +3.372681 | 0.079844 | 675 | 0.3166 |
| R4 | GPTQ error compensation, R0's scale | 4.299819250 | +3.532224 | 0.108934 | 706 | 0.2848 |
| **R5** ⚠ | **R3's scale + R4's compensation** | 2.027495180 | **+1.259900** | 0.067779 | 252 | 0.4346 |

⚠ **R5 is POST-HOC and gates nothing.** The brief's §4 fixes `best` as the minimum over R1–R4, so
`best = R3 = +1.709372` and the label is computed from that. R5 is reported because it is the arm
that explains the others, not because it changes the verdict — and it would not change it if it
did count: `+1.259900 > 0.50`, still `RULE-HELPS`.

## 4. What actually moved the number — paired contrasts BETWEEN arms

The Δs in §3 are each measured against `base`. They cannot support a claim about one rule versus
another, because the arms are correlated across sequences. These are the paired, byte-weighted
bootstraps of the **differences between arms**, on the same 24 sequences:

| contrast | what it isolates | Δ | SE | ci95 | significant |
|---|---|---|---|---|---|
| R2 − R0 | searching the scale at all | **−0.686** | 0.099 | [−0.884, −0.496] | **yes** |
| R3 − R2 | **weighting that search by activation** | **−0.914** | 0.056 | [−1.027, −0.806] | **yes** |
| R3 − R0 | the scale, in total | **−1.600** | 0.101 | [−1.814, −1.411] | **yes** |
| R5 − R3 | error compensation **on a good grid** | **−0.449** | 0.080 | [−0.611, −0.293] | **yes** |
| R4 − R0 | error compensation **on the naive grid** | +0.223 | 0.134 | [−0.039, +0.483] | **no** |
| R1 − R0 | TWN against BitLinear158 | −0.225 | 0.132 | [−0.498, +0.023] | **no** |
| **R0 − Z** | **BitLinear158 against RANDOM SIGNS** | **−0.064** | 0.126 | [−0.302, +0.205] | **no** |

Four things fall out, and each names the row it comes from.

**(a) The engine's rule is indistinguishable from assigning the signs at random.** `R0 − Z` is
`−0.064 ± 0.126`, ci95 straddling zero. Having decided which weights are non-zero, BitLinear158's
choice of *which sign* carries no measurable information about the donor on this eval slice. This
is the mechanism behind T1's catastrophe, and it is why arm Z was the wrong control: the brief
assumed Z would be far worse than the treatment, and it is not worse at all.

**(b) The scale is the whole story, and the activation weighting is the largest single piece.**
Searching `Δ/α` per row on plain L2 buys `−0.686`. Weighting that identical search by how strongly
each input is actually driven buys a further **`−0.914`** — more than the search itself, at the
tightest SE in the table (0.056).

**(c) Error compensation is conditional on the grid it feeds errors into.** On R3's grid it
recovers `−0.449` and is significant. On R0's grid it does nothing measurable (`+0.223`, ci95
straddling zero). GPTQ is *not* broken — see §5 — it simply has nowhere to push error when the
grid is in the wrong place.

**(d) TWN is not a different answer.** `R1 − R0 = −0.225 ± 0.132` does not clear zero. A fixed
`0.7·mean|w|` threshold is a different constant, not a different idea.

### 4.1 The 2×2

|  | scale = `mean\|w\|` | scale = activation-weighted search |
|---|---|---|
| **no compensation** | R0 `+3.309` | R3 `+1.709` |
| **GPTQ compensation** | R4 `+3.532` | R5 `+1.260` |

Reading down: compensation is worth `+0.223` (n.s.) on the left and `−0.449` (sig) on the right.
Reading across: the scale is worth `−1.600` on the top row and `−2.272` on the bottom.
**The scale dominates, and it also decides whether compensation helps or does nothing.**

## 5. R4 was checked before it was believed

R4 came in at `+3.532`: worse than the round-to-nearest it starts from, and numerically worse than
random signs. A correct GPTQ cannot lose to its own starting point *on its own objective*, so the
instrument was tested before the number was written down.
(`benchmarks/donor_adaptation/ternary/t2_r4_controls.py`, run `9232252`.)

| control | requirement | result |
|---|---|---|
| **A** | with `H = c·I` the inverse-Hessian coupling vanishes, so GPTQ must collapse onto R0 exactly | identical codes **and** identical scales, at two different `c` |
| **B** | on a correlated `H` it must lower `‖(W − Wq)X‖²` versus R0 | `1.176e6 → 8.301e5`, **−29%**, changing 41% of the codes |

Control B is the load-bearing one: A alone would be passed by a GPTQ that does nothing at all.

**So GPTQ is correct, and the finding is real: at ternary width the layer-wise reconstruction error
is the wrong objective when the grid is badly placed.** What that 29% costs per weight:

| | R0 | R4 |
|---|---|---|
| zero fraction | 0.3090 | 0.2131 |
| mean \|residual\| / row scale | 0.4466 | **0.8047** |
| fraction of weights > 0.5 grid step from target | 0.2304 | **0.6116** |

It nearly doubles the per-weight residual to buy cancellation between correlated inputs. That
cancellation is only worth something where the inference input distribution matches the calibration
set; measured on a disjoint eval slice it does not transfer, and the doubled residual is what
survives. The full run's own zero fractions move the same way (§3: `0.3166 → 0.2848`).

**R5 is the discriminator, and it reversed the prediction I wrote before seeing it.** I predicted
that if compensation were the culprit, R5 would land *worse* than R3. It lands `−0.449` *better*.
The culprit is not compensation; it is the grid. Recorded because it was written first.

## 6. What this does to T1

`T1_DONOR_TERNARIZATION.md` returned **CONVERSION-FAILS** on `+4.738 BPB` for the full conversion
and `+3.309` on the FFN organs. Both numbers replicate exactly. What changes is their reading:

- **T1's measurement stands.** R0 reproduces `+3.309099` to six decimals against a value published
  before this run.
- **T1's verdict was correctly scoped and is now superseded in scope.** Its §5 registered, in
  advance, that it tested exactly one rule. It did, and that rule turns out to be the problem.
- **The conclusion "the ternary format destroys a donor" is withdrawn.** The format is unchanged in
  every arm here; only the map into it differs, and the map is worth 2.05 BPB.
- **What does NOT change:** `+1.260` is still 252 σ_seed on a 0.7676 baseline. This is not a usable
  model. It is a diagnosis that moved from *format* to *rule*, which is what the brief asked.

## 7. The pre-registered label, verbatim

> **RULE-HELPS** — `0.50 < best ≤ Δ(R0) − 1.00`, i.e. `0.50 < 1.709372 ≤ 2.309099`.
> *"a better rule recovers a large part but not enough. Healing is still needed but starts from a
> much better place; report the recovered fraction."*

**Recovered fraction: `1.599727 / 3.309099` = 48.3% (pre-registered arm R3); `2.049199 / 3.309099`
= 61.9% including the post-hoc R5.**

## 8. Owed, and what this brief explicitly did not test

- **A Controller audit.** Owed on this probe as on D0c, T1, P2, P3, R1.
- **Organ coverage.** Every number here is the **FFN only** — 84 tensors, 28 layers × `{gate, up,
  down}`. A model that runs also converts `q/k/v/o` and the head, and their cost is unmeasured.
  Pre-registered as `briefs/BRIEF_T2B_ORGAN_COVERAGE.md`; the exporter sidecar now records
  `rule_applied_to` against `t2_measured_organs` so the seam cannot be forgotten.
- **The calibration budget.** R3 and R5 use 32×512 = 16,384 tokens, D4's registered operating
  point. `BRIEF_D4B_CALIBRATION_BUDGET.md` was written to sweep it and is **still unrun**, so every
  R3-family number here is a **floor, not a ceiling**. `qwen_export.py --calib-seqs` now defaults
  to 32 to match, after it was found defaulting to 8 — which would have exported a model built on
  a quarter of the calibration that produced these numbers.
- **Activation quantization.** Untouched here. It is the `--lut` path's separate cost, measured in
  `donor_engine.c`, and it is **not** composed with any number above.
- **Another donor or size.** Everything is Qwen2.5-1.5B.
- **Measurement through the runtime.** These are PyTorch numbers about a model `donor_engine.c`
  executes. `--bpb` closes that loop and has not been run at scale.

## 9. What follows

1. **T2b** — the organ sweep, pre-registered, cheap, and it decides what the runnable model costs.
2. **Export with the winning rule and measure through the runtime**, which is the first time any of
   this programme's quality numbers and its engine will have met.
3. **Healing** (QAT / layer-wise distillation) is still on the critical path per §7 — but it now
   starts from `+1.260` rather than `+3.309`, and the brief for it should be written against the
   *new* starting point.
4. **D4b** is no longer optional bookkeeping: two of the best arms here are calibration-driven and
   their budget has never been swept.

---

## 10. Appended 2026-09-11 after E20 — this probe's rules, read in RANKING for the first time

T2 measured six rules in **BPB only**; the ranking instrument did not exist yet. E20 applied this
probe's own definitions — imported from `t2_rules`, not restated — to `lm_head` on the 1.5 B and
read greedy agreement against the donor's own continuations
(`probes/E20_RULE_OR_FORMAT.md`, ledger §33). Floor `12/160` (E18 part A):

| rule | BPB on the head | greedy | teacher-forced top-1 | mean rank of donor token |
|---|---|---|---|---|
| `R0` | `1.319900` | 17/160 | 107/160 | `16.92` |
| `R1` | `1.280712` | 13/160 | 114/160 | `12.85` |
| `R2` | `1.288465` | **41/160** | 116/160 | `13.96` |
| **`R3`** (this probe's winner) | `1.106584` | 9/160 | 110/160 | `2.98` |
| `R4` | `1.031616` | 15/160 | 115/160 | `2.40` |
| **`R5`** (this probe's post-hoc best) | `0.940203` | 9/160 | 112/160 | `2.71` |

**§4's ordering is confirmed in BPB and does not survive into ranking.** `R5` beats `R3` by
`−0.166380` on the head against this probe's `−0.449472` on the FFN — same sign, and §4's
decomposition is not in question. But `R3` and `R5` score **identically at 9/160**, and the best
ranking in the table belongs to `R2`, the *unweighted* search, which §4 shows is `−0.914` worse
in BPB than weighting by activation RMS.

**E20 part B supplies the mechanism, and it vindicates this probe's objective while limiting it.**
With the donor's context held fixed, the data-aware rules put the donor's token at mean rank
`2.40`-`2.98` versus `12.85`-`16.92` for the weight-space rules — **the activation weighting does
exactly what §4 says it does, it preserves the logit geometry.** What it does not preserve
preferentially is the top-1/top-2 boundary, which is where argmax lives.

**Two things this changes for this probe.** §3's `⚠ R5 is POST-HOC and gates nothing` was correct
discipline and stands. Its untracked consequence is now recorded: **`R5` is not implemented in
`qwen_export.quantize`**, which dispatches `R0`/`R1`/`R2`/`R3` only, so **the best rule this probe
measured has never been in a shipped artefact**. And E20's brief mis-stated the shipped rule as
`R0`, which its replication gate caught — `t2_rules.r3_actsearch` **does** use calibration
activations, on the head included.
