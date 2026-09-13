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
