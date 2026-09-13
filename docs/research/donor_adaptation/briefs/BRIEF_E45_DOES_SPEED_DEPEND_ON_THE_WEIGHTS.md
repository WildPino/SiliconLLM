# E45 — does the engine's speed depend on the weight VALUES? The untested bridge under every 10B number

**PRE-REGISTERED. Pushed before any measurement.**

## 0. The one sentence

**Every throughput number this programme has published at 10B scale was measured on SYNTHETIC
weights**, and the assumption that lets those numbers speak about a real trained model — *that
engine speed does not depend on what the weights actually are* — **has never been measured.**
E45 measures it, on a matched pair that already exists.

## 1. Why this is the load-bearing gap

The programme's headline is that the speed target is met: `R128` — 9,999,220,736 real parameters
— reads ≈113–130 tok/s with the FFN on. That artifact (`D:/_ktmp/e40/e40_r128.bin`) holds
**noise**, not a trained model. E36, E39 and E40 all say so in their own briefs: *speed only, the
weights are noise.*

Separately, E37 measured **both** quality and speed on **real** Qwen2.5-1.5B weights through the
same engine (`DENSE-NF` 30.75 tok/s median, `K16` 89.83). So "a trained LLM runs on `engine.c`"
is established — **at 1.5B**.

The two halves are joined by an unstated inference:

> the 10B rate measured on noise is the rate a trained 10B would get.

That inference is **reasonable and untested**. It is reasonable because the ternary kernels are
fixed-work: a packed/LUT kernel does the same operations whatever the trits are. It is untested
because nobody has put two artifacts of **identical shape and different weight values** through
the engine and compared. If it is false, **every 10B rate in this programme is wrong**, including
the one the goal is judged against.

E37 built exactly that pair and never raced them:

| artifact | weights | bytes | mean ternary zero fraction |
|---|---|---|---|
| `e37_carved_nf.bin` | **real** Qwen2.5-1.5B, R3 | 1,714,582,392 | **0.4810** |
| `e37_carved_syn.bin` | **synthetic** | 1,714,582,392 | **0.4698** |

Same shape, same quant, same byte count, **different values** — and the zero fractions differ by
1.1 points, which is the concrete thing a value-sensitive kernel could notice.

## 2. Why speed COULD depend on values, stated so the null is not assumed

Three mechanisms, none of them exotic:

1. **A zero-skipping path.** If any kernel branches on a zero trit or a zero block, a 1.1-point
   difference in zero fraction is a 1.1-point difference in work.
2. **Denormals or special values in the fp32 scales.** Real per-row scales come from a fitted
   search (`r3_actsearch`); synthetic ones are drawn. A denormal scale costs a microcode trap on
   x86, and that is a classic silent 10× on one organ.
3. **Branch prediction and data-dependent control flow** anywhere in the gather/unpack path.

If none applies, the ratio is 1 and the bridge holds. **The point of E45 is that this is checked
rather than asserted.**

## 3. The design — paired, interleaved, and judged on a RATIO

* **Arms:** `NF` (real) and `SYN` (synthetic), the two files above, identical engine flags.
* **Interleaved** `NF, SYN, NF, SYN, …` for ≥5 pairs. Interleaving is the whole design: it makes
  slow drift in box occupancy, DVFS or thermals cancel within a pair instead of loading onto one
  arm. The first pass of any A/B that runs all of A then all of B is measuring the afternoon.
* **The statistic is the per-pair ratio** `NF/SYN`, reported as median and min–max spread.
  The standing convention in this programme is that **ratios do not carry the ±5% that absolute
  rates do**; E40's verdicts were decided on ratios for the same reason. Absolute rates are
  still printed, still with dispersion, and still may not be quoted as headlines — E43's finding
  stands and E44 addendum A did not repeal it.

## 4. The gates, registered now

**`G-E45a` — PLANTED CONTROL, and nothing below counts until it fires.**
The instrument must be shown to **detect a speed difference that is really there**. Known
positive: `e37_dense_nf.bin` against `e37_carved_nf.bin` at the carve's own k, which E37
measured as **30.75 vs 89.83 tok/s** — a ~2.9× gap. If the harness cannot see that, its null on
`NF vs SYN` is worthless. *No null from an instrument that has not fired.*

**`G-E45b` — the ratio is reported with dispersion or not at all.**
≥5 interleaved pairs; median and min–max spread of the ratio. A bare ratio may not appear.

**`G-E45c` — the decision gate, stated so it can fail in both directions.**
Let `r` be the median per-pair ratio `NF/SYN` and `s` the spread of the per-pair ratios.
* **BRIDGE HOLDS** iff `|r − 1|` is **smaller than `s`** — the difference between real and
  synthetic weights is not resolvable against the pairing's own noise.
* **BRIDGE BROKEN** iff `|r − 1| > 2s` **and** `|r − 1| > 0.05`. Then every synthetic-weight
  rate in the programme, including the 10B headline, is biased by a measured amount, and the
  size of that bias is E45's product.
* Anything between is **INCONCLUSIVE** and is reported as such, not rounded to the convenient
  side.

## 5. What E45 may and may not conclude

**MAY:** state whether synthetic-weight rates transfer to real weights **at this shape**, and
quantify the bias if there is one.

**MAY NOT:** mint any new headline rate. **MAY NOT** claim the result composes to 10B by itself
— it is measured at 1.5B, and Phase 61's law (a microbench does not compose to the engine) has a
sibling here: *a bridge verified at one scale is verified at that scale.* What E45 buys is that
the inference stops being unexamined; extending it to 10B would need the same pair built at 10B,
which is cheap to say and a separate job.

**MAY NOT** be used to revive any E43 cell.

## 6. Cost

CPU only, no GPU, no training, no user action. ~10–15 minutes. Both artifacts already exist on
disk. Because the verdict is a **paired interleaved ratio**, it does not require the idle box
that `G-E44b` does — but occupancy is measured and reported next to every cell regardless, and
if the ratio's own spread is too wide the registered answer is INCONCLUSIVE rather than a
number.

## 7. The prediction, written before the run

I expect **BRIDGE HOLDS**: `|r − 1|` under 2%, comfortably inside the pairing noise, because the
ternary kernels are fixed-work and the zero fraction should not reach a branch. I am recording
that because it is the boring outcome, and because if the ratio comes back at 1.10 it would
invalidate the programme's headline claim and I want it on the record that I did not expect it.

---

# ADDENDUM B — THE PAIR DIFFERS IN TWO THINGS, NOT ONE, AND THAT FIXES WHICH `k` DECIDES

**Written and pushed before any E45 measurement.** The brief's §1 table describes the pair as
"same shape, same quant, same byte count, different values". Reading the two sidecars on disk
shows that is **incomplete**, and the omission is mine:

| field | `e37_carved_nf.bin` | `e37_carved_syn.bin` |
|---|---|---|
| `sha256` | `fae666d2…31b7150` | `79b8a0f1…eabb950e` |
| `bytes` | 1,714,582,392 | 1,714,582,392 |
| `mean_ternary_zero_fraction` | 0.4809541542 | 0.4698122995 |
| `carve_router` | `e37_routers_E256.npz` | **`null`** |
| `carve_router_kind` | **`fitted`** | **`synthetic`** |

The two artifacts differ in **the expert weight values** *and* in **the router baked into the
file**. The real arm carries a router fitted on real activations; the synthetic arm carries a
drawn one. That is not a flaw to be apologised for — it is exactly the contrast the bridge
needs, because `e40_r128.bin`, the artifact every 10B rate comes from, is synthetic **in both
respects too**. But it means a difference in rate could come from either factor, and a brief
that says "values" while measuring "values + router" is the kind of sentence this programme has
been burned by before.

## B.1 A fourth mechanism, named before the run

§2 listed three ways speed could depend on values. The router adds a fourth, and it is the most
plausible of the four:

4. **Selection locality.** At `k < E` the router decides *which* expert groups are read. The
   number of groups is fixed, so the *work* is fixed — but the *addresses* are not. A peaked
   router re-reads the same groups and can hold them in the 32 MB L3; a flat one walks the whole
   expert table and pays DRAM. The L3 cliff at 16 MB (probe-3) is a measured feature of this
   box, so this is a live mechanism, not a hypothetical.

## B.2 Which `k` decides, registered now

The mechanism above is **absent at `k = E = 256`**: when every group is selected, the router
changes nothing about which memory is touched, and the only thing left between the two files is
the weight values themselves.

* **`K256` is the PRIMARY arm and the one `G-E45c`'s verdict is read from.** It is the clean
  value-only comparison, and it is also the arm that matches the headline it is defending: the
  10B claim (`R128`, ≈113–130 tok/s) is quoted **with the FFN fully on**.
* **`K16` is the SECONDARY arm**, reported with its own `G-E45c` verdict and its own dispersion,
  and labelled **value + selection**. It may not override the primary. If the two disagree —
  primary HOLDS, secondary BROKEN — that is not a contradiction and must not be written as one:
  it localises the effect in the *router*, not in the values, and the correct report is "the
  bridge holds for values and the selection pattern is worth its own experiment".
* **`K16` is also the arm `G-E45a` plants its control in**, because that is where E37 measured
  the 89.83 tok/s that the control has to resolve against dense's 30.75.

## B.3 `G-E45a`, made falsifiable in numbers

The brief says the control must "see" a ~2.9× gap. Stated so it can fail: with `DENSE-NF`
(no carve flags) and `K16` (`--carve-k 16`) raced interleaved for ≥3 pairs, `G-E45a` **fires iff
the median per-pair ratio `K16/DENSE` is ≥ 2.0 and every individual pair exceeds 1.5.** Below
that the instrument has not been shown to resolve a difference that is really there, and per the
planted-control law nothing else in E45 may be reported as a null.

## B.4 One identity check before anything is timed

The two files must be shown to be **the same size and different content** — equal `bytes`, and
`sha256` values that differ. Racing a file against itself would return `r = 1` with a tiny
spread and look exactly like the registered prediction. That check runs first and aborts on
failure.

Nothing else in the brief changes. §7's prediction stands as written, and it is now a prediction
about **`K256`**.

**No cell of E45 has been measured at the time this addendum is pushed.**

---

# ADDENDUM C — RUN 1 IS IN, AND IT SHOWED THAT `G-E45c` CAN BE PASSED BY BEING NOISY

**Written after run 1 and before run 2. Run 1's numbers are recorded in
`benchmarks/donor_adaptation/engine/results/e45_weight_values.json` (`c2600b3`) and are not
revised here — E36's run-2 rule applies: run 1 is the registered measurement and run 2 may not
promote it.**

## C.1 What run 1 returned

`G-E45a` **FIRES**: `K16` against `DENSE-NF`, 3 interleaved pairs, median **2.9102** (min
2.8276, max 2.9917) against the ≥ 2.0 / every-pair > 1.5 registered in B.3, and against the
2.92× E37 measured. The identity check fires. The instrument resolves a real difference, so its
nulls are admissible.

`G-E45c`, applied verbatim:

| arm | `r` | `s` | \|r−1\| | literal verdict |
|---|---|---|---|---|
| **K256** (primary, value only) | 1.0471 | 0.1831 | 0.0471 | BRIDGE HOLDS |
| K16 (secondary, value+selection) | 1.0367 | **6.8528** | 0.0367 | BRIDGE HOLDS |

## C.2 Why that is not an answer, and the defect is in my gate

`G-E45c`'s pass condition is **`|r − 1| < s`** — *smaller than the pairing's own noise*. A gate
of that shape **is passed by being noisy**, and run 1 passed it that way. `K16`'s spread is
**6.85 in ratio units**, manufactured by a single cell that read **10.34 tok/s** between
neighbours at 73–81 tok/s. Calling that "the difference is not resolvable" is true and useless:
*nothing* is resolvable against a spread of 6.85.

The brief already says the opposite in **§6**: *"if the ratio's own spread is too wide the
registered answer is INCONCLUSIVE rather than a number."* So **two clauses of the same
pre-registration disagree on this data**, and I am not entitled to pick the convenient one after
seeing it. What run 1 establishes is that **the bridge is neither established nor broken.**

This is the mirror of the defect already in the ledger — *a gate with a tolerance tighter than
the dispersion of the axis it watches is mis-specified* (E43, `G-E43A`'s ±5%). Same cause, other
direction: **a gate whose PASS condition is "inside my own dispersion" rewards a worse
measurement.** It goes in the same place.

## C.3 Two things run 1 did settle, both free

**1. Occupancy measured *across* a cell is mostly the engine itself.** Six threads on twelve
logical CPUs is ≈50% system-wide before any contention exists. The 44.9–65.2% readings are the
engine alone; only the 73–90% ones are external load. My reporting conflated self-load with
contention. `e44_interval.py` is **unaffected** — it samples a `watch` window *before* the run,
never across it — but any future cell-level occupancy figure in this programme must be read
against a ≈50% floor, not against zero.

**2. The first load is a USB read, and it now has a number.** Cold: **15.8 s** for 1,709,047,348
bytes ≈ **108 MB/s**, squarely the USB-HDD class addendum A predicted *from the bus alone*.
Warm: **0.5–0.7 s** ≈ 2.7 GB/s, the page cache. Addendum A observed that startup had never been
reported beside a rate. It has now.

## C.4 The observation that is NOT promoted to a gate

At **K256** all five pairs ran the same way — **NF faster than SYN** — and the four
uncontaminated ones read **1.0471 / 1.0487 / 1.0237 / 1.0025**. NF also carries the **higher**
zero fraction (0.4810 vs 0.4698), which is the direction **mechanism 1** (a zero-skipping path)
predicts. At K16 the direction is not consistent (0.9770, 1.0367, 0.9996, 7.8298, 1.0458).

Per **E14 §6** a post-hoc metric may not be promoted to a gate, so this decides nothing. It is
the reason run 2 exists, and run 2 registers the direction test **in advance**.

## C.5 RUN 2, registered now, before its data exists

Run 2 is **not a re-run of `G-E45c` to a different answer** — E40 addendum A's precedent stands.
It is a differently specified measurement, on new data, with a gate that cannot be passed by
noise.

* **Arms and interleaving unchanged:** `NF`/`SYN`, `K256` primary and `K16` secondary.
* **≥ 15 attempted pairs per arm.**
* **A registered exclusion rule, fixed before the data:** a pair is **discarded** if the
  occupancy sampled across *either* of its two cells exceeds **70%**. The bar is set from C.3 —
  the engine alone is ≈50%, and 70% leaves 20 points of slack before a pair is called
  contaminated. Discards are **counted and printed**; the rule is applied by the program, not by
  me, and it never looks at the rates.
* **`G-E45d`, the decision gate for run 2.** Let `n` be the surviving pairs, `r` their median
  ratio, `s` their max − min, and `u` how many have ratio > 1.
  * **INCONCLUSIVE** if `n < 5`, **or if `s > 0.10`** — twice the BROKEN floor. A pairing that
    noisy cannot answer a 5% question, and this is the clause run 1 lacked.
  * **BRIDGE BROKEN** if `s ≤ 0.10` and `|r − 1| > 0.05`.
  * **DIRECTIONAL** if `s ≤ 0.10`, `|r − 1| ≤ 0.05`, and `u = n` or `u = 0` — a consistent
    difference smaller than the BROKEN floor. This is a **result**, not a pass: it says the
    bridge holds to within 5% but that speed is **not** independent of the values.
  * **BRIDGE HOLDS** if `s ≤ 0.10`, `|r − 1| ≤ 0.05`, and the direction is not unanimous.
* **`G-E45a` is re-fired in run 2** on run 2's own cells. An instrument fires for the data it
  produced, not once forever.

If fewer than 5 pairs survive the exclusion on this box, run 2 reports **"could not be measured
here"** and the quiet box moves to `COMMUNICATION.md` beside `G-E44b`, which needs exactly the
same thing. It does **not** get retried until it passes.

## C.6 What is still true and what the goal still needs

The bridge under every 10B rate in this programme is **still untested**. Run 1 narrowed it: any
value effect at this shape is **small** — nothing in the data suggests the 2.9× that a real
zero-skipping path would produce — but "small" is not "measured", and 4.7% sits at 94% of the
distance to the registered BROKEN floor. **§5 is unchanged:** E45 may not mint a headline rate
and may not claim the result composes to 10B.

**No cell of run 2 has been measured at the time this addendum is pushed.**

---

# ADDENDUM D — RUN 2 IS INCONCLUSIVE, AND THE BLOCKER IS THE MEASUREMENT WINDOW, NOT THE BOX

**Written after run 2 (`e264319`) and pushed before any run-3 cell exists.** Run 2's numbers are
recorded in `results/e45_weight_values_run2.json` and are not revised here.

## D.1 What run 2 returned

`G-E45d`'s eight planted controls fire, `G-E45a` re-fires on run 2's own cells (median
`K16/DENSE` **2.8102**, min 2.7121), the identity check fires, and then:

| arm | discarded | n | `r` | `s` | \|r−1\| | above 1 | verdict |
|---|---|---|---|---|---|---|---|
| **K256** primary | 3 of 15 | 12 | 0.9937 | 0.2517 | 0.0063 | 4/12 | **INCONCLUSIVE** |
| K16 secondary | 0 of 15 | 15 | 0.9975 | 0.6047 | 0.0025 | 7/15 | **INCONCLUSIVE** |

**Run 1's flag is dead.** Run 1 had 5 of 5 K256 pairs with NF faster, median 1.0471, in the
direction a zero-skipping path predicts. Run 2 reads **0.9937** — the other side of 1, 4 of 12
above it. It was noise. Registering it instead of promoting it is the only reason that sentence
can be written today rather than retracted later.

## D.2 The occupancy bar is not why it is inconclusive

The exclusion dropped **3** pairs at K256 and **none** at K16, and the spread stayed at 0.25 and
0.60. So the noise is not external contention, and it is not the slow drift that interleaving
cancels. It is **E43's intrinsic per-cell oscillation**, which lands *inside* each cell and
therefore enters the ratio **twice**: NF's own relative spread is 0.2953 at K256 and SYN's
0.2008, on cells whose occupancy never exceeded 65%. A single-shot paired ratio inherits about
√2 of that, which is exactly the 0.25–0.60 observed.

**No quieter box fixes this.** The design is wrong for the noise it faces, and that is mine to
fix rather than the user's.

## D.3 The thing run 2 exposed, which is bigger than E45

Every cell in E45, and **`NTOK = 40` in `e40_levers_exhausted.py` — the file the 10B headline
comes from** — measures a decode window of **40 tokens**. At the R128 rate that is **≈0.35
seconds**. At E45's K256 rate it is 1.6 s.

A third of a second is squarely inside the window where a Zen 2 part is still on **boost
clock**. So the programme's headline 10B rate is not merely noisy: it may be **systematically
fast**, measured before the clock settles, and nothing in this programme has ever measured a
*sustained* rate. `e44_interval.py` already chose `--ntok 300` for that reason, but `G-E44b` has
never run and no curve exists.

**`H-WINDOW`**: *the engine's 9–22% dispersion, and possibly its central value, are artifacts of
a ~0.35–1.6 s measurement window.*

## D.4 RUN 3, registered now, before its data exists

**Arms:** `NF`, at `K256` (primary) and `K16`. **Windows:** `NTOK ∈ {40, 160, 640, 2560}` at
K256; `{40, 2560}` at K16. **≥7 reps per cell.** The windows are **interleaved in round-robin**,
not measured longest-last, so session-long thermal drift cannot load onto one window.

Let `m(n)` be the median rate at window `n` and `cv(n)` its relative spread `(max−min)/mean`.

**`G-E45g` — THE DRIFT CONTROL, and nothing below counts until it fires.** The `NTOK = 40` block
is measured **twice**, once at the start and once at the end of the whole sweep. The two must
agree to within the wider of their two `cv`s. If they do not, the session drifted monotonically,
the window curve is confounded, and run 3 reports **VOID** rather than a curve. *A comparison
across a sweep needs a control that the sweep itself did not move.*

**`G-E45h` — THE LINEARITY CONTROL.** The engine's own reported `dt` must scale with the window:
`dt(2560) / dt(40)` within **±15%** of 64. If it does not, `--bench N` is not measuring what its
name says and no window conclusion may be drawn from it.

**`G-E45i` — THE DISPERSION QUESTION.**
* **SHORT-WINDOW ARTIFACT** iff `cv(2560) ≤ 0.5 · cv(40)`.
* **SCALE-FREE** iff `cv(2560) ≥ 0.8 · cv(40)`. This is the **bigger** outcome: it would mean no
  absolute tok/s on this machine can be given a ±5% interval by measuring longer, that E43's
  oscillation is a property of the part and not of the probe, and that E45's question cannot be
  settled here at any affordable cost.
* Anything between is **PARTIAL** and is reported as the curve, not as a word.

**`G-E45j` — THE CENTRAL-VALUE QUESTION, which is the one that touches the headline.** Let
`w = m(2560) / m(40)`.
* **BOOST-INFLATED** iff `w < 0.95` **and** `|1 − w| > cv(2560)` — the short window reads at
  least 5% fast by more than the long window's own dispersion. Then **every rate this programme
  has published is biased upward by a measured factor**, including the 10B headline, and that
  factor is run 3's product.
* **NO WINDOW EFFECT** iff `|1 − w| ≤ cv(2560)`.
* Otherwise **INCONCLUSIVE**.

**What run 3 may NOT do.** It may not revise run 1 or run 2 (E36's run-2 rule), it may not mint
a headline rate (§5), and a favourable `G-E45i` does **not** by itself answer E45's own question
— it only licenses a run 4 of the NF/SYN pairing at the window `G-E45i` identifies, with
`G-E45d` unchanged.

**Cost:** ~20–25 minutes, CPU only, no user action.

**No cell of run 3 has been measured at the time this addendum is pushed.**

## D.5 — A PATCH TO `G-E45j`, MADE BEFORE ANY RUN-3 CELL EXISTED

Writing the gate's planted controls caught me **reproducing addendum C's own defect one
addendum later**. As registered in D.4, `G-E45j` returns **NO WINDOW EFFECT** whenever
`|1 − w| ≤ cv(2560)` — *inside my own dispersion* — so a long window that is itself noisy
passes the gate by being noisy. The control that exposed it: `m(40) = 113`, `m(2560) = 60`,
`cv(2560) = 0.90` returned **NO WINDOW EFFECT** for a **47% gap**.

**Patched, before a single cell was measured:** if `cv(2560) > 0.10` the verdict is
**INCONCLUSIVE** regardless of `w`. A long window that cannot hold its own rate to 10% cannot
certify anything about a 5% question. Both shapes are now planted controls in
`e45_window.py --selftest` (`W11`, `W12`), which runs at the head of every invocation.

The rest of D.4 is unchanged.

---

# ADDENDUM E — RUN 3 IS VOID, THE WINDOW LEVER DOES NOT EXIST, AND RUN 4 USES THE ONE THAT DOES

**Written after run 3 (`0897e70`) and pushed before any run-4 cell exists.**

## E.1 `G-E45h` refused, and it refused for a mechanism

| arm | `dt(2560)/dt(40)` | expected | rel err | |
|---|---|---|---|---|
| K256 | 96.47 | 64 | 0.507 | *** FAILS *** |
| K16 | 158.91 | 64 | 1.483 | *** FAILS *** |

`G-E45i` and `G-E45j` were never evaluated. `G-E45g` fired first on both arms (K256 gap 0.0512
against bar 0.0737; K16 0.0241 against 0.2113), so the box did **not** drift under the sweep —
the refusal is about the instrument, not the conditions.

**`--bench N` decodes at positions 1…N.** A longer window is therefore *not* the same work per
token: the KV cache grows and attention is charged for every earlier position. **Window length
and context length are the same knob in this engine**, and `H-WINDOW` asked about one while
moving both. The control caught it on the first run that could have produced a headline.

## E.2 So the window lever is unavailable, and D.4's run 3 is withdrawn

There is no way, in `bench` mode, to lengthen the timing window at fixed context. `H-WINDOW`
cannot be tested with this engine as written, and **run 3 is not re-run** — a control that fires
is doing its job (E40 addendum A). What remains of H-WINDOW moves to its own pre-registration,
because it is a different and larger question (see E.4).

## E.3 RUN 4 — the lever that does exist is REPS PER CELL

Run 2 failed because a **single-shot** paired ratio inherits E43's intrinsic per-cell
oscillation twice. The fix is not a longer window and not a quieter box: it is to **take the
median of `m` reps within each cell before forming the ratio**, which averages the oscillation
down while leaving context fixed at the value every other number in this programme uses.

Sizing it from run 3's own quietest block (`K256_40_last`, cv 0.0705 over 7 reps, so
sd is about 2.4%): a ratio of two `m`-rep medians has sd about 2.4% * sqrt(2)/sqrt(m), and over
12 pairs the range is about 3.5 sd. At **`m = 5`** that is `s` about 0.05, inside `G-E45d`'s
0.10 requirement with margin.

* **`NTOK = 40`, unchanged**, so run 4 is comparable with E37, E40 and runs 1-2.
* **12 pairs**, **`m = 5` reps per cell**, interleaved **at the rep level**: `NF SYN NF SYN ...`
  five times, then the pair's two medians form one ratio.
* **`G-E45d` is unchanged and is the verdict**, including the occupancy exclusion at 70% and the
  `s > 0.10` INCONCLUSIVE clause. A pair is excluded if **any** of its ten cells breaches.
* **`G-E45a` re-fires on run 4's own cells**, and the gate self-test runs first.
* If `s` still exceeds 0.10 at `m = 5`, the answer is INCONCLUSIVE and **the pairing is not
  re-run at larger `m` in this experiment** — that would be tuning until it passes. It would
  instead mean the oscillation does not average down, which is a finding about the box and
  belongs to the item in E.4.

**Registered prediction for run 4:** `s` under 0.08 and **BRIDGE HOLDS**, with `|r-1|` under 2%.
Run 1 said +4.7% and run 2 said -0.6%; the two disagree in sign, so I expect a value-independent
engine and I am recording that before the third look.

**Cost:** about 8 minutes, CPU only, no user action.

## E.4 What run 3's raw cells left behind, which is bigger than E45

Reported as a post-hoc fit and **not promoted to a gate** (E14 section 6). Per-token cost against
mean context position, least squares over run 3's cells:

| arm | fit | at position 0 | residuals |
|---|---|---|---|
| K256 | `s/tok = 0.039967 + 1.454e-05 * pos` | 25.02 tok/s | -3.9%, +2.3%, +1.9%, -0.4% |
| K16 | `s/tok = 0.011840 + 1.427e-05 * pos` | 84.46 tok/s | two points, exact by construction |

**The two slopes agree to 1.9% across arms that differ 3.3x in FFN work** — which is what must
happen if the marginal cost is attention, since the carve does not touch it. And the marginal
cost is **9.0-9.2x its own bandwidth floor**: 1.44e-05 s per token per position against 57,344
bytes of fp32 KV, which at E30's measured 36.30 GB/s would cost 1.58e-06 s.

Two consequences, both for a separate brief rather than this one:

1. **Every tok/s this programme has published was measured at context <= 40, mean position 20**,
   including the 10B headline. A rate at a realistic context has never been measured.
2. **If attention costs 9x its bandwidth floor, E34's wall is soft** — and E34 already found
   attention, not the FFN, is where the budget goes.

**No cell of run 4 has been measured at the time this addendum is pushed.**

---

# ADDENDUM F — E45 CLOSES. THE VERDICT OF RECORD, AND WHAT IT DOES NOT BUY

**Written after run 4 (`b9550a2`).** E45 is closed here; nothing below is a new measurement.

## F.1 The four runs

| run | design | K256 (primary) | K16 (secondary) |
|---|---|---|---|
| 1 | 5 single-shot pairs, `G-E45c` | `r` 1.0471, `s` 0.1831 — HOLDS, **voided by C.2** | 1.0367, `s` 6.8528 — voided |
| 2 | 15 pairs + occupancy exclusion, `G-E45d` | 0.9937, `s` 0.2517 — **INCONCLUSIVE** | 0.9975, `s` 0.6047 — INCONCLUSIVE |
| 3 | window sweep | **VOID** — `G-E45h` refused (E.1) | VOID |
| 4 | 12 pairs × 5 reps/cell, `G-E45d` | 0.9919, `s` **0.1003** — **INCONCLUSIVE** | 1.0025, `s` 0.0930 — **BRIDGE HOLDS** |

`G-E45a` fired on runs 1, 2 and 4, each time on that run's own cells (2.9102 / 2.8102 / and
again in run 4), against E37's 2.92×. The instrument resolves a real difference. Its nulls are
admissible.

## F.2 The verdict of record

**The bridge is SUPPORTED and NOT CERTIFIED at the primary arm.**

* **K16 — BRIDGE HOLDS.** The registered gate cleared on the registered design.
* **K256 — INCONCLUSIVE by three parts in ten thousand** (0.1003 against a 0.10 bar), and
  **not re-run**: E.3 wrote that clause in advance for exactly this moment. The two arms sit at
  the same resolution limit and a threshold put one on each side of it. Calling that "K16 passed
  and K256 failed" would be reading a coin flip as a mechanism.
* **Across every measured run**: K256 read +4.7% / −0.6% / −0.8%, K16 +3.7% / −0.25% / +0.25%.
  Run 1's unanimous direction **reversed** in both later runs. Every `|r−1|` since the
  instrument was repaired is **under 1%**, against a harness that demonstrably resolves 2.9×.

So: **any dependence of this engine's speed on weight VALUES, at this shape, is bounded well
under 5% with no reproducible sign.** That is a bound, and it is the honest product of E45.

## F.3 What it does not buy

* **It is 1.5B.** §5 stands: E45 may not compose to 10B. The 10B pair does not exist and
  building it is a separate job — and now a cheaper-looking one, since the bound at 1.5B removes
  most of the reason to fear the answer.
* **It mints no rate.** Every absolute tok/s in this document is a context-20 number, which is
  E46's subject.
* **The one thing that would certify K256** is a box quieter than this one has been all day.
  That is already item 1 of `COMMUNICATION.md`, beside `G-E44b`, and E45 does not add a new
  request for it: the bound above is enough to stop treating the bridge as an unexamined
  assumption, which is what §0 set out to do.

## F.4 The two things E45 cost, kept as lessons

1. **A gate whose PASS condition is "inside my own dispersion" is passed by being noisy.** Found
   in `G-E45c` (C.2), then **reproduced by me one addendum later** in `G-E45j` and caught by its
   own planted control before any data (D.5). It is the mirror of E43's `G-E43A`, whose
   tolerance was *tighter* than the dispersion it watched. Both are the same error: **a gate must
   be specified against the measured dispersion of the axis it watches, in the right direction.**
2. **Sizing a design from the quietest block you have is a selection effect.** Run 4's `m = 5`
   came from run 3's `K256_40_last` (cv 0.0705), the calmest block on hand. Run 4's own cells
   read cv up to 0.16 at K256 and 0.32 at K16, and the primary arm missed its bar by 0.3% as a
   direct result.
