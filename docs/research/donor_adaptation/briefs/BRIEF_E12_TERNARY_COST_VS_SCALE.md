# E12 — does the ternarization cost shrink with donor scale?

**Pre-registered. Pushed before the runner exists.** The oldest open item in this programme, and
after E10 the only one that can still move the goal.

---

## 1. Why this, and why now

E10 closed the engine's weight path: the packed kernel is on its ceiling at every footprint, the
engine's FFN organs are already there, and ledger §19.4's 37 and 42 GB/s budget rows are withdrawn.
**The remaining 7.4× is a property of the model.** The only route to a 10 B at 50 tok/s that this
programme has ever had is **fewer active weights per token** — sparsity/MoE — at **ternary
precision**.

That route rests on a quality number, and **that number has only ever been measured at one shape.**
`probes/T2_TERNARIZATION_RULE.md:4` reads *"Donor: Qwen2.5-1.5B"*. The whole R0→R5 rule
investigation, the +3.309 → **+1.260 BPB** FFN result, and T2b's whole-model **+2.466** are one
point on a curve nobody has drawn.

**If the cost is flat or grows with scale, the strategy's quality premise is false and we would be
building a 10 B whose ternary damage is at least as bad as the 1.5 B's.** If it shrinks, the road is
supported by evidence rather than by hope. Either way it is the cheapest decisive thing left,
**and it is deterministic — it can be measured on a contended machine**, which the speed work
cannot.

## 2. What is measured

For each donor **D ∈ {Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B}**, all present offline and pinned to
the commit hashes already in the local cache (1.5 B's is the **same** hash `8faed761…` this
programme has used throughout, so the middle point is the historical one, not a re-derivation):

**ΔBPB(D) = BPB(D ternarized) − BPB(D fp32)**, on the same held-out corpus, same sequence count,
same organ set, and the **same conversion oracle** — `t1_ternarize.ternarize()`, which is the single
definition the exporter and the parity gate both call.

**ΔBPB is the comparable quantity; absolute BPB is not.** A 3 B is a better model than a 0.5 B and
will score lower on both arms; the question is only what the format costs each of them.

## 3. Controls

**Planted control (read first).** The **1.5 B cell must reproduce the published number at that
shape.** If a sweep cannot re-derive the point it already has, nothing else in it counts. This is
G-L2's discipline applied deliberately rather than after a void — E11's run 1 was caught exactly
this way, by a number carried in from outside the run.

**One variable.** Corpus, sequence count, rule, organ set and dtype are identical in every cell; the
donor is the only thing that changes. Each donor gets **its own fp32 baseline**, computed in the
same run — `t1_ternarize.py`'s `BASELINE_STANDING` constant is a **1.5 B** number and **must not**
be reused for the other two.

**No anchoring.** The runner prints ΔBPB per cell without a verdict; the bands below decide
mechanically.

## 4. Bands, fixed here

Let `r = ΔBPB(3B) / ΔBPB(0.5B)`.

| r | verdict |
|---|---|
| **≤ 0.75** | **COST-SHRINKS** — the format gets cheaper as the donor grows; the MoE road's premise is supported and the 1.5 B number is pessimistic |
| 0.75 – 1.25 | **COST-FLAT** — the 1.5 B number generalises; **no scale relief**, and a 10 B inherits the damage |
| **≥ 1.25** | **COST-GROWS** — the premise is false and the road needs re-planning before anything is built on it |

The middle cell is not decorative: **COST-SHRINKS or COST-GROWS also requires 1.5 B to sit between
0.5 B and 3 B in the same direction.** If it does not, the verdict is **NON-MONOTONE** and three
points were not enough — reported as such, not smoothed into a trend.

## 5. What this expressly does NOT license

**Three points across 6× do not extrapolate to 10 B**, which is 3.3× beyond the largest cell. E12
can establish a **direction** and nothing more. This programme has already been burned by a
two-point fit that a third point refused (E7 §8.2), and the rule written down then applies here:
**below the band is not a phenomenon.** A `COST-SHRINKS` verdict would license *building the 10 B
experiment*, not *claiming the 10 B number*.

**It does not test activation quantization** (`--lut`'s separate 1.40e-01 / 3.10e-02 cost), and it
does not test MoE routing. It tests the format's damage to a dense donor's weights, which is the
input every later step assumes.

## 6. Honest ceiling

No outcome of E12 makes anything faster. Coder-7B is **7.4× short of 50 tok/s** and stays there.
What E12 decides is whether the one remaining strategy is standing on a measurement or on a
single-shape assumption — which, after E10 removed the engine's cover, is the question that
determines what gets built next.

---

# VERDICT (added after the run): `CONTROL-FAILED` at 3 B

Full write-up: `probes/E12_TERNARY_COST_VS_SCALE.md`. Ledger §26. Sweep 2161 s, 24x512, all layers.

| donor | BPB fp32 | BPB ternary | **dBPB** | planted `Z` | real `F` | **Z-F** |
|---|---|---|---|---|---|---|
| 0.5 B | 0.871795 | 4.587453 | **3.715658** | 3.693528 | 3.674503 | **+0.019** |
| 1.5 B | 0.767595 | 5.505834 | **4.738239** | 4.001257 | 3.309099 | **+0.692** |
| 3 B | 0.724450 | 5.734699 | **5.010249** | 3.714379 | 5.203945 | **-1.490 FAIL** |

**The pre-registered question is NOT answered.** `r = 1.3484` reads `COST-GROWS` mechanically, but
its numerator is the cell whose planted control failed, and §4's rule is that a control must fire
before the cell counts. **It is not reported as a result.**

**What IS established**: at 3 B the mis-specified rule (`random_sign`, same organs) produces a
**better** model than the real rule, by 1.49 BPB — and independently, `F` (FFN only) is worse than
`FA` (FFN+ATTN), so converting *more* organs did *less* damage. **Two monotonicity violations at
the same cell.** A rule beaten by random signs is not an expensive rule, it is a broken one.

**Not an instrument bug**: `I - base = +0.000e+00` **exactly at all three cells**, over 168/196/252
substituted tensors, and the substitution counts are structurally correct (7/layer FA, 3/layer F, at
24/28/36 layers). The BPB numbers are trustworthy; the control's *premise* is what failed.

**§1's framing holds, and harder than written.** T2's rule is measured at one shape and **the one
cell where the known-positive fires convincingly is that same 1.5 B**. It fires at +0.019 (~3.8
sigma_seed) at 0.5 B and fails at 3 B. **T2 is not retracted — it holds where it was measured — but
it is now measured NOT to generalise.**

**New, and it goes on the list**: the smoke showed `Z-F = +0.147` at 0.5 B where the full sweep
shows **+0.019**. **A control that passes on a 2-sequence 4-layer smoke is not thereby a control**,
and its full-scale margin is not predictable from the smoke.

**§5's design fix earned its keep.** The `F` arm was added after the first smoke because `Z` was
being compared against `FA` across different organ sets. **Without it the 3 B failure would have
been confounded with organ coverage and unreadable.**

---

# §9 — AMENDMENT, pre-registered before the diagnostic runs

**The verdict block above is suspended.** It was written before I checked E12's arm `Z` against the
prior measurement of the same estimand, and that check does not pass.

## 9.1 Two facts that were available before E12 ran, and that I did not carry in

**(a) This programme had already retired arm `Z` as a control.** `probes/T2_TERNARIZATION_RULE.md`
§4(a) measures `R0 - Z = -0.064 +/- 0.126`, ci95 `[-0.302, +0.205]`, **not significant**, and states
in terms: *"it is why arm Z was the wrong control: the brief assumed Z would be far worse than the
treatment, and it is not worse at all."* The same amendment, dated 2026-09-04, is written into the
docstring of `t1_ternarize.ternarize` — **the function E12 imports** — which records that arm `Z`
"turned out to test a SCIENTIFIC claim ('signs carry information') rather than an INSTRUMENT
property", and that the control which replaced it is arm `I`. **`t1_ternarize.ARMS` even labels its
own `Z` row `"PLANTED CONTROL (mis-specified, see report)"`.** E12's brief re-adopted it anyway.

**Arm `I` is the control T2 installed in its place, and `I - base = +0.000e+00` exactly at all
three cells.** The instrument control that this programme actually sanctions **passed everywhere**.

**(b) E12's `Z` does not reproduce T2's `Z` at the shared cell.** At 1.5 B, on the same eval slice
(`ids_sha256 a1a48dc9...`), same donor revision, same organs, same rule:

| source | arm `F` / `R0` | arm `Z` |
|---|---|---|
| T2 (`t2_rules.py`) | **+3.309099** | **+3.372681** |
| E12 (`e12_scale.py`) | **+3.309099** | **+4.001257** |

**`F` reproduces to six decimals — so the harness, the slice and the rule are identical.** The only
thing that differs is which per-tensor seed lands on which tensor: `t2_rules.py` seeds
`1000 + stats["n"]`, `t1_ternarize.apply_arm` seeds `1000 + rng`, **off by one**. That off-by-one
moves arm `Z` by **0.628 BPB = 126 sigma_seed**.

## 9.2 What is measured, and the bands

`ternary/e12_zvar.py`. Arm `Z`, **K = 5 draws** per cell, `seed_base` in `{1000, 2000, 3000, 4000,
5000}`, everything else held. `s` = max - min of `dBPB_Z` at a cell.

| gate | test | band |
|---|---|---|
| **G-Z0** — replication | `seed_base = 1000` vs `e12_scale.json`'s `dBPB_Z` | **must match to the last digit** at every cell. `apply_arm` gained a `seed_base` argument defaulting to 1000; if the default path moved, nothing below counts. |
| **G-Z1** — the verdict | `s(3B)` against the observed `|Z - F|` = **1.490** | **`s >= 1.490` `Z-UNSTABLE`** · `0.10 <= s < 1.490` `Z-NOISY` · `s < 0.10` `Z-STABLE` |

**`Z-UNSTABLE` means the 3 B "control failure" is inside the comparator's own dispersion and carries
no information about the rule** — the `CONTROL-FAILED` verdict is withdrawn, arm `I` stands as the
instrument control, and E12's measurements are read on their merits. **`Z-STABLE` means arm `Z` is a
real quantity, the T2-vs-E12 gap in §9.1(b) needs its own explanation, and the 3 B inversion stands
as a finding.**

## 9.3 Prediction, on the record

**`Z-UNSTABLE`.** Derived from a **measured** quantity — the **0.628 BPB** gap between two draws of
the identical estimand in §9.1(b) — over a **structural** factor: 3 B has 36 layers to 1.5 B's 28
and `d_ffn` 11008 to 8960, so it has **more** independently-seeded tensors, not fewer. This is the
same shape of derivation that produced the only two predictions this programme has landed (E9, E13),
and unlike E14 §5 it is **not** a magnitude guess.

**What it does not decide.** `Z-UNSTABLE` does **not** rescue the ternary road and does **not**
license `COST-GROWS` as a verdict on its own — see the rewritten probe. It decides only whether the
3 B cell was disqualified for a real reason.
