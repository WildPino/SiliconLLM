# E60 — the rung that was never built: one byte per weight

**Brief**: `briefs/BRIEF_E60_THE_RUNG_THAT_WAS_NEVER_BUILT.md`, pushed at `9a9fc29` before the
runner existed, amended twice (both before any export ran). **Apparatus**: `f1d2fcb`
(`quant=5`, rule `R8`, runner), pushed before the run. **Runner**:
`engine/e60_one_byte.py`, 20/20 self-tests. **Result**: `engine/results/e60_one_byte.json`.
**Binary**: `donor_engine_e60.exe` — every arm, both scales, one build.

---

## 0. Verdict

**`OUTCOME_LABEL: THE-CLIFF-IS-BELOW-ONE-BYTE`**

The engine's precision ladder had two rungs, 4 B/weight and 0.5 B/weight, and the programme had
spent fifty-nine experiments concluding that the first is faithful and slow and the second is fast
and destroyed. **Both halves of that are true and neither is about quantization: they are about a
rung nobody built.** At one byte per weight, on the same donors, through a kernel that already
existed and needed no line written:

| | | `05b` | | | | `15b` | | |
|---|---|---|---|---|---|---|---|---|
| arm | B/w | tok/s (95% CI) | dBPB vs fp32 | greedy | | tok/s (95% CI) | dBPB vs fp32 | greedy |
| `F32` | 4.0 | 19.08 (18.89–19.16) | — | **160/160** | | 6.28 (6.27–6.30) | — | **160/160** |
| **`I8`** | **1.0** | **63.87 (61.76–64.01)** | **+0.000066** | 145/160 | | **21.39 (21.35–21.50)** | **+0.001252** | 129/160 |
| `PACKED` | 0.5 | 86.23 (85.31–87.07) | +3.659423 | 3/160 | | 29.32 (29.19–30.10) | +2.708100 | 10/160 |
| `T1` | 1.0 | 61.89 (61.56–62.41) | *= PACKED* | 3/160 | | — | | |

**`G-E60d`: ABOVE at 0.5 B, BELOW at 1.5 B.** `05b_i8h` is the first trained artifact this
programme has put above the 50 tok/s bar whose modelling cost is measurable in hundredths of a
seed sigma. **×3.35 and ×3.41 over the faithful arm**, at **1.3% and 25% of `σ_seed = 0.005`**.

**`G-E60e`: `DEGRADED`** — registered at ≥150/160 and read 145 and 129. Not re-specified; see §5.

**The second finding outranks the first**, because it corrects arithmetic used everywhere else in
this programme: **the byte ladder is not a bandwidth ladder.** See §3.

---

## 1. The two planted controls, which are why the nulls count

Neither was decoration. Both had to fire before any treatment number was read, and the runner
exits on either failure.

**`G-E60a` — the C change is inert. FIRES 4/4.** `quant=5` is `quant=1`'s payload byte for byte
and exists only so an int8-valued file and a ternary-valued file do not print the same `CONFIG`
line. `donor_engine_e60.exe` against the frozen `donor_engine_e53.exe`, same prompts, greedy:

| | `F32` | `PACKED` |
|---|---|---|
| `05b` | **184/184 identical** | **184/184 identical** |
| `15b` | **184/184 identical** | **184/184 identical** |

**`G-E60b` — the fidelity instrument reproduces E57's known positives. FIRES 4/4.** `F32` at
**160/160** both scales; `PACKED` at **3/160** and **10/160**. Exact, not within tolerance.

**And the BPB instrument replicated itself unasked.** `05b F32` read **0.871810466** against E1's
stored **0.871810461** — `|d| = 4.47e-09`, a different binary and a different runner.

**Why HuggingFace is the right reference here and was the wrong one in E59.** Yesterday E59 nearly
published *"the kernel costs nothing"* because it scored a treatment against a control already
sitting at 3/160 — a counter on its floor cannot show damage. Here the control reads **160/160**:
the counter has all 160 points of range in the direction that fails. The floor rule
(`feedback_gate_is_not_a_progress_meter`, second face, registered 2026-09-13) is satisfied **by
construction**, and was checked against before the run, not after.

## 2. The quality axis — the cliff is not where the programme put it

`dBPB` against the fp32 arm on the standard slice (24×512, 12,264 predicted tokens, 51,870 scored
bytes, chance `4.069819`), with `σ_seed = 0.005`:

| scale | `I8` (1 B/w) | `PACKED` (0.5 B/w) | ratio | `I8` in σ_seed |
|---|---|---|---|---|
| `05b` | **+0.000066418** | +3.659423268 | **55,096×** | **0.013 σ** |
| `15b` | **+0.001252465** | +2.708100319 | **2,162×** | **0.250 σ** |

**The first 4× byte cut is free and the next 2× is catastrophic.** Not a gradient — the second
halving costs between two thousand and fifty-five thousand times what the first quartering did,
and lands the model **above chance** at 0.5 B (4.531 vs 4.070) while 1 B leaves it indistinguishable
from exact.

This re-reads E18's `CLIFF-NOT-SLOPE` rather than contradicting it. E18 measured the cliff along
the *organ* axis at fixed precision and found no partial-conversion operating point: `lm_head`
alone → 9/160, attention alone → 4/160, FFN alone → 5/160, every one inside the floor. **E60
measures the same cliff along the precision axis and finds it is a cliff there too — just located
between 1 B and 0.5 B, not between 4 B and 1 B.** E18's ladder had no rung there to stand on.

**Two points, one direction, stated as a segment and not a law:** the `I8` cost grows **19×** from
0.5 B to 1.5 B (6.6e-05 → 1.25e-03). That runs *against* the usual expectation that larger models
quantize more gracefully. Two points are a segment. If it extrapolates it gets worse exactly where
the goal needs it, and a third scale is the cheapest way to find out.

### 2.1 The SCORE and its RANK partner disagree by four orders of magnitude

`G-E60f` registered the greedy column as the RANK partner for the `dBPB` SCORE before the run, per
E14 §3. They disagree, and the disagreement is the information:

| | `dBPB` says | greedy says |
|---|---|---|
| `05b I8` | +0.000066 = **indistinguishable** | 145/160 = **9.4% of tokens flip** |
| `15b I8` | +0.001252 = **indistinguishable** | 129/160 = **19.4% of tokens flip** |

Both are correctly measured. At `dBPB` 6.6e-05 the distribution is essentially untouched, so the
flips are **near-ties whose argmax order an infinitesimal perturbation tips** — a different failure
mode from ternary, which does not tip ties, it destroys the distribution. The divergence profile
says the same thing: `05b I8` diverges at tokens **11, never, 23, 21, never** and `15b I8` at
**25, 2, never, never, never**, against `PACKED`, which diverges at **token 0** on 4 of 5 and 5 of
5 prompts.

This is E50's law (*"a BPB average does not settle an argmax"*) with the sign reversed. Normally
the average **hides** damage the argmax reveals. Here the argmax **manufactures** damage the
average says is not there, because greedy agreement over 32 positions is an absorbing process: one
tipped tie at position 2 costs the remaining 30 positions. **A 160-token greedy count is not a
fidelity measure on an arm this close to exact — it is a tie-breaking measure.**

## 3. The speed axis — the byte ladder is NOT a bandwidth ladder

The finding that travels furthest. Moved bytes per token computed from the shape (`tqh` unties, so
the fp32 embedding is **gathered**, never streamed, per E57 addendum D's convention); per-row
scales charged explicitly:

| size | arm | B/w | moved MB/tok | tok/s | **achieved GB/s** | **efficiency vs the fp32 arm** |
|---|---|---|---|---|---|---|
| `05b` | `F32` | 4.0 | 1976.1 | 19.08 | **37.70** | 100% |
| `05b` | **`I8`** | **1.0** | **495.8** | **63.87** | **31.67** | **84.0%** |
| `05b` | `PACKED` | 0.5 | 249.1 | 86.23 | **21.48** | **57.0%** |
| `15b` | `F32` | 4.0 | 6174.9 | 6.28 | **38.78** | 100% |
| `15b` | **`I8`** | **1.0** | **1546.8** | **21.39** | **33.09** | **85.3%** |
| `15b` | `PACKED` | 0.5 | 775.6 | 29.32 | **22.74** | **58.6%** |

**Achieved bandwidth falls monotonically as bytes per weight fall.** Fewer bytes per weight makes
the kernel worse at using the memory system, and the loss is large: the packed path reaches barely
half the DRAM rate the fp32 path reaches on the same box, in the same sweep, on the same weights.

The identity closes at every cell to within **0.4%**:

> **rate ratio = byte ratio × bandwidth-efficiency ratio**

| | byte ratio | × efficiency | = predicted | measured |
|---|---|---|---|---|
| `05b` `I8`/`F32` | 4 | 0.840 | **3.359** | **3.347** |
| `05b` `PACKED`/`F32` | 8 | 0.570 | **4.557** | **4.519** |
| `15b` `I8`/`F32` | 4 | 0.853 | **3.413** | **3.406** |
| `15b` `PACKED`/`F32` | 8 | 0.586 | **4.691** | **4.669** |

**This corrects arithmetic used throughout the programme.** Every projection that divided a byte
count by a single bandwidth number assumed the denominator is a property of the box. It is a
property of **the box and the kernel and the format together**. E58 §8 registered *"a bound is a
number with a scope"* as a caution after the ×1.75 withdrawal; E60 replaces the caution with a
measured curve. The three denominators on this box are **37.7–38.8 (fp32), 31.7–33.1 (int8),
21.5–22.7 (packed)** GB/s, and no projection may use one for another's format.

It also prices the unpaid debt. The `quant==1` branch has **a single accumulator chain** — E13 §8
named it owed on 2026-09-07 and it is still owed. At 84–85% of the achievable rate, closing it is
worth up to **×1.19**, i.e. `05b` at ~**76 tok/s** and `15b` at ~**25**. The brief said that if the
rate landed near the low end this debt stops being a footnote. It landed at 84%, so it does.

### 3.1 `G-E60c` : UNRESOLVABLE — and the prediction it broke was mine

`I8` beats `T1` by **1.98 tok/s** (63.87 vs 61.89) against a measured dispersion half-width of
**1.13**. Same bytes (495,785,472 exactly, both files), same layout, same instruction stream,
**different values**. The gate printed its third outcome rather than forcing a verdict.

Brief §3 argued *"speed is a function of bytes, not of values on this branch — the same instruction
stream runs whatever is in the bytes"*, and prediction 3 said `WITHIN`. **That argument is not
supported by the measurement.** The mechanism I would look at first is that `T1` is a **broken**
model (3/160, BPB above chance) and broken models can drive activations into subnormals, which
Zen 2 penalises — i.e. value-dependence enters through the float unit, not the load unit, exactly
where "same instruction stream" stops being an argument. This is a diagnosis, not a result; it is
owed, and it does not touch `G-E60d`, whose interval clears 50 either way.

### 3.2 Conduct

Nine reps, arms **interleaved inside one sweep**, warm-up discarded per arm, `--threads 6`,
`--bench 160`, one binary. **Contamination reported, not cleaned:** mean foreign occupancy ran
**3.70–9.72%** against `OCC_BAR = 4.39`, and the `05b I8` cells took the worst of it (up to
**10.92%**). The load is on the **treatment**, so `G-E60d`'s reading is **conservative**; E52's
coefficient (0.262% of rate per point) accounts for roughly half the 3.5% spread on the worst
cell, which is the registered "necessary not sufficient" clause behaving as documented. Every
absolute tok/s carries ±5%; the ratios in §3 do not.

## 4. What this does and does not do to the goal

The goal's sentence has two halves — *"un LLM già addestrato"* and *"un modello grande (es 10B) a
50 token/s"*. E36 settled the second with noise weights. E57 showed no arm satisfies both.

**Desk model, marked as such, on E36's own registered 10 B cell** (`A10B-K3`, 540 of 46,080 FFN
neurons per layer = 1.17% active, **0.9283 G charged weights/token**, **49.96 tok/s**) — whose
weights are **noise**, per E36 §0, using E60's measured per-format bandwidths:

| the same 10 B shape at… | tok/s | % of the bar | quality |
|---|---|---|---|
| fp32, 4 B/weight | **10.2** | 20% | faithful |
| **int8, 1 B/weight** | **34.1 – 35.6** | **68–71%** | **+6.6e-05 … +1.25e-03 BPB** |
| packed ternary, 0.5 B/weight | 49.96 | 100% | 3/160-class, BPB above chance |

**The gap between "faithful" and the bar at 10 B was ×4.9. It is now ×1.4.** That is the honest
statement of what E60 moved, and it is a desk model on a synthetic artifact until somebody runs it.

**What E60 does NOT do**, stated as forcefully as the brief stated it in advance:

- **It does not reach 10 B at 50 tok/s.** A *dense* 10 B at one byte per weight moves 10 GB/token
  and reads ~3.3 tok/s. On the goal's shape int8 is **strictly worse than ternary** — it doubles
  E34's attention+head floor. Every number above depends on the 1.17%-active MoE shape, which is
  itself unbuilt as a trained model (`project_moe_status`: the true MoE with a jointly trained gate
  has never been tried on this branch).
- **It does not make `--lutblk` available.** That kernel requires `quant=2` (`donor_engine.c:1553`).
  E59's ×1.253/×1.333 does not compose with this rung.
- **It is post-hoc.** Nothing here was trained into. The programme's standing finding is that
  post-hoc is fatal at 0.5 B and trainable out (H0 removes 97.9% of the damage). E60 says the
  post-hoc damage at 1 B is already ~0 in BPB, which makes *training* into 1 B pointless and
  training into **0.5 B** the only place healing can still earn anything.
- **`G-E60e` reads `DEGRADED` and stays that way.** See §5.

## 5. The gate I got wrong, and why it is not being re-specified

`G-E60e` required **≥150/160 at both scales AND `dBPB ≤ 0.020` at both**. It read **145** and
**129** on the first clause and **+0.000066** and **+0.001252** on the second — the count clause
missed by 5 and 31 tokens, the BPB clause passed by factors of **301×** and **16×**.

**The gate fires `DEGRADED` and it is left alone.** E40 addendum A's precedent: a gate that fires
is doing its job and is not re-run, re-worded or re-weighted into a pass. My number was 150 and it
was not met.

**But the gate is defective, and that is a separate statement from its verdict.** A conjunction
that weighs a 5-token argmax miss equally against a BPB delta 301× inside its own tolerance is not
measuring one thing. §2.1 shows why: over 32 greedy positions one tipped tie costs up to 31
downstream tokens, so the count has enormous variance on an arm this close to exact — three of five
`15b` prompts never diverge at all and two carry the entire 31-token deficit. **The count is a
tie-breaking measure being read as a fidelity measure.** The replacement (registered here, for
whatever uses it next) is **per-position top-1 agreement under teacher forcing**, which does not
absorb, over the 12,264-position slice instead of 160 free-running positions.

That makes three gates this programme has had to call defective rather than failed — `G-E44b`
(malformed), `G-E55a2` (blind to what it judged), and now `G-E60e` (two clauses measuring different
things). The common shape: **each was written to be strict, and strictness was mistaken for
validity.**

## 6. Predictions, scored — 4.5 of 6

| # | prediction | outcome |
|---|---|---|
| 1 | `G-E60a` FIRES 4/4 | **HELD** — 184/184 on every pair |
| 2 | `G-E60b` FIRES 4/4 | **HELD** — 160/3/160/10, exact |
| 3 | `G-E60c` **WITHIN** | **WRONG** — `UNRESOLVABLE`; 1.98 vs 1.13, and §3.1 says my mechanism argument was the thing that failed |
| 4 | `I8` at `05b` in **55–80**, `G-E60d` **ABOVE** | **HELD** — 63.87, `ABOVE` |
| 5 | `I8` at `15b` in **18–26** | **HELD** — 21.39 |
| 6 | `G-E60e` **FAITHFUL**, 158–160/160, `dBPB ≤ 0.005` | **HALF** — `dBPB` held by 301×/16×; the count read **145/129** against 158–160 |

**Both misses cost me something I had argued for**: #3 refutes the "bytes not values" claim written
into the brief's own §3, and #6 says the arm is less faithful *on argmax* than I predicted while
being far more faithful *on BPB* than I predicted. The registered falsification — *"if `I8` at
`05b` reads below 50 tok/s the claim dies"* — did not trigger: 61.76 at the bottom of the interval.

## 7. Owed

1. **E13 §8's multi-accumulator debt on the `quant==1` branch**, now priced at **up to ×1.19**
   (§3). Promoted from footnote to next experiment.
2. **§3.1's value-dependence**: is the 1.98 tok/s `I8`−`T1` gap subnormals in a broken model, or
   contention? Diagnosable with a denormal counter or an FTZ build; do **not** use `-ffast-math`
   (Phase 35).
3. **A third scale** for §2's 19× segment. Two points are not a trend.
4. **The 10 B cell at 1 B/weight, measured** rather than desk-modelled — `synth_export.py` +
   `quant=5` is the whole apparatus, and E36's harness already exists.
5. **The replacement fidelity metric** (§5): per-position top-1 under teacher forcing, registered
   before the run that uses it.
6. Untouched from before: `G-E55a2`'s replacement interval gate, E56's `OCC_BAR` zero point,
   E42's void control, the true MoE with a trained gate.
