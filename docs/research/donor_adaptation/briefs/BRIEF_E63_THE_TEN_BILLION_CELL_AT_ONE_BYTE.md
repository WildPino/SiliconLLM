# E63 — the 10 B cell at one byte per weight, measured instead of projected

**Pre-registered. Pushed before any engine change, any runner and any measurement.**

Opened by `probes/E61_...md` §9 item 3 and `probes/E62_...md` §11 item 2, which name the same
blocker, and made urgent by E62: **the 10 B fidelity number can no longer be extrapolated, only
measured**, and the 10 B *speed* number has never been anything but a desk model.

---

## 1. Why this is now the most consequential thing open

The goal is a ~10 B model at 50 tok/s. The programme's answer today is **36.6–37.6 tok/s,
73–75% of the bar** (`SPEED_LEDGER` §60.4) — and every part of that sentence is a projection:

- it is **E36's `A10B-K3` charged weights (0.9283 G/token) divided by a bandwidth measured on a
  different shape**, stated as a desk model in §60.4 and in E61 §3.1;
- the artefact is **synthetic, noise weights** (E36 §0), so it says nothing about quality;
- the arithmetic has **never been run at one byte per weight at all**, because the engine
  refuses: `donor_engine.c:477` *"row-selected matvec is implemented for the plain packed kind
  only"* and `:709` *"matvec_colacc needs a transposed packed matrix"*.

E62 removed the escape route. dBPB does not rank with N (6.64e-05 → 1.25e-03 → 5.79e-04), so no
fit licenses a 10 B claim; §11 item 2 promoted the measured cell to **the only route**. And
§61.6's audit — including its own correction — established that the denominator question cannot
be settled by arithmetic either: the surviving statement is a one-sided bound, and E26's *"a
gathered weight costs more than a streamed one"* governs at shapes where the carve dominates.

**Everything points at one missing number.** E63 builds the engine path that makes it obtainable
and then obtains it.

## 2. Checked, not assumed

Searched by artefact name — `carve`, `int8`, `MK_PACKED_T`, `colacc`, `gathered`, `10B`, `A10B` —
across `docs/` and every `*/results/`. **Every brief and probe the search returned was opened.**

| where | what it says | bearing on E63 |
|---|---|---|
| `probes/E26_...md` | verdict **`A GATHERED WEIGHT COSTS MORE THAN A STREAMED ONE`**: carved charged-throughput at T10 is **38.03–43.12 G active weights/s** against E25's **49.59–50.36** for a rank cut; the shortfall grows as the carve deepens. Introduced `MK_PACKED_T`, `quant==4`, `--carve-k` | The carve is not free, and the penalty is shape-dependent. E63 must report the carved int8 rate **as measured**, never as bytes ÷ a streaming band. |
| `probes/E31_...md` (+ two corrections) | the gathered-byte penalty lives in **`down`**, `PT_BLK = 64` = one cache line per selected neuron; §6.1's lever went `2.01×` → `1.25×` on re-reading the engine | **`down` is exactly the organ E63 must re-lay-out.** At 0.5 B a 64-byte block holds 128 weights; at 1 B it holds 64. This is the open physical question. |
| `probes/E33_...md` | measured that lever: **1.116 normalised at S15, 1.043 at T10 — unresolvable from 1.0 at the goal's own shape** | A third cut of the same desk arithmetic. E63 registers no locality prediction it cannot measure. |
| `probes/E30_...md` + its correction | at T10 the operative packed path reads **4.363 tok/s**, gap **×11.46**; a *perfect* kernel still reads 6.83 | This is the **dense** 10 B shape. It is why E63's cell is the carved one and why a dense int8 10 B (3.40–3.49 tok/s) is not worth running. |
| `probes/E36_...md` | `A10B-K3`: **540 neurons active (1.17%)**, charged **928,251,904** (`G-E36C` FIRES, zero tolerance), **measured 49.96 tok/s** (run 2: 51.50) at packed; prediction 3 **MISSED** — it came in 2.4%/0.8% *slower* than the charged model, not faster | The artefact, the carve, the charge, and the **packed comparator E63's int8 cell is measured against**. Its file is on disk: `D:/_ktmp/e36/e36_a10b.bin`, 5,485,658,936 B. |
| `probes/E60_...md`, `E61_...md` | the one-byte rung exists, needs no new dense kernel, and the three bands are **37.64 / 33.97–34.90 / 20.65 GB/s** at 4 / 1 / 0.5 B per weight, built from **moved** MB/token | The bands E63 may *compare* against and may **not** divide into carved byte counts (§59.2's rule; §61.6 broke it and was corrected). |
| `probes/E62_...md` | dBPB does not rank with N; §11 item 2 makes the measured 10 B cell the only route | The reason E63 is now first in the queue rather than fourth. |
| `probes/E37_...md`, `E38`, `E39`, `E40` | the carve at *trained* weights, post-hoc router; E40 priced how much FFN 50 tok/s buys | E63 is **synthetic-weights only**, deliberately: it is a speed-and-correctness experiment, not a quality one. Quality on a carved trained model is H1's. |

**No int8 carved artefact exists in this repo, and no engine path can load one.** `MK_PACKED_T`
is the only transposed kind; `read_mat` (`donor_engine.c:896`) knows four kinds and none is int8.

## 3. What is built

Two new **self-describing matrix kinds** in the tagged format, so every existing `quant=3`/`quant=4`
file loads byte for byte and no existing branch is edited:

| kind | layout | consumer |
|---|---|---|
| `MK_I8 = 4` | `[out, in]` int8 codes + `out` fp32 scales | `matvec_sel`, **with the row list threaded through** |
| `MK_I8_T = 5` | block-major `[out/blk][in][blk]`, **one** byte per weight along OUT, + `out` fp32 scales | new int8 branch of `matvec_colacc` |

Three code changes, all additive:

1. **`matvec_sel`'s row list for int8.** Today `:477` dies unless `m->packed`. The int8 fallback
   already computes `scale[o] · dot(code + o·n_in, x)` for every row; a row list is one index
   (`o = rows ? rows[t] : t`) and a compacted write. E61's four-accumulator structure is kept
   exactly, so the dense path stays byte-for-byte what it is today.
2. **`matvec_colacc` for `MK_I8_T`.** Simpler than the packed kernel, not harder: no trit decode,
   no even/odd split, no `pshufb` table — `cvtepi8_epi32 → cvtepi32_ps → fmadd`, 64 weights per
   block instead of 128.
3. **`synth_export.py`** writes the two new kinds under a `--quant int8c` (carved int8) selection,
   reusing E36's router, labels, seed and group size **unchanged**, so the int8 artefact differs
   from `e36_a10b.bin` in format and nothing else.

`blk` stays `PT_BLK = 64` **bytes**. That is the deliberate choice and §4's `G-E63e` is where it
is checked: at 0.5 B/weight a block is 128 weights, at 1 B/weight it is 64, so an int8 carve reads
*twice the bytes per selected neuron* — the physical question E31 identified and nobody has
measured.

## 4. Gates — thresholds fixed here, before anything is written

### Part A — correctness, on a dirty box (deterministic)

**`G-E63a` — PLANTED: the new engine is inert on everything that already exists.**
`donor_engine_e63.exe` vs frozen `donor_engine_e61.exe`, `--generate` on every artefact this
programme still uses: `05b f32/tqh/i8h`, `15b i8h`, `e36_a10b.bin` (packed carve, `--carve-k 3`).

| test | pass |
|---|---|
| generated ids **byte-identical** on every artefact | **sha256 equal, zero tolerance** |

**FAIL ⇒ nothing below is read.** Two new enum values must change nothing; if they do, the
loader was not additive.

**`G-E63b` — PLANTED: the fidelity instrument must DISCRIMINATE before it certifies.**
E59's law — *a fidelity gate must be measured against a reference the treatment can still move.*
Per-position top-1 under teacher forcing (E61's instrument, 12,264 positions) between
**int8-dense** and **packed-carved** on the same synthetic weights.

| test | pass |
|---|---|
| agreement **≤ 60%** | **FIRES** — the counter has range in the direction that fails |

A near-100% reading here means the instrument cannot see a format change and **`G-E63c` is void,
not passed.**

**`G-E63c` — the new path computes the right thing.**
`MK_I8` + `MK_I8_T` with **every** group selected, against the plain dense `MK_I8` file on the
same weights. Different summation order (block-major accumulation vs row-major dot), so bit-exact
is not available and is not demanded:

| test | pass |
|---|---|
| per-position top-1 agreement | **≥ 99.90%** of 12,264 |
| max abs logit difference | **≤ 1e-3** |

### Part B — speed, and **only on an idle box** (`COMMUNICATION.md` item 1)

**`G-E63d` — the measurement the programme has been projecting.**
`A10B-K3` at one byte per weight, `--carve-k 3`, `--threads 6`, **≥ 5 interleaved repetitions**
against the packed artefact in the same sweep, paired median + bootstrap CI.

| reading | verdict |
|---|---|
| CI lower bound **≥ 36.6** | **`DESK-MODEL-HELD`** |
| CI spans 36.6 | **`DESK-MODEL-UNRESOLVED`** |
| CI upper bound **< 36.6** | **`DESK-MODEL-OPTIMISTIC`** — §60.4 is corrected downward by the measurement |

**Admissibility, not a result:** foreign occupancy **< `OCC_BAR` = 4.39%** on every cell, or the
sweep is `VOID` and reported as void. E61 left a control owed for exactly this reason and E40
addendum A forbids re-running a fired gate to a pass.

**`G-E63e` — the locality question E31 identified, answered by the pair.**
Both formats measured in the same sweep, so the comparison is paired:

| quantity | what it decides |
|---|---|
| `rate(int8 carve) ÷ rate(packed carve)` vs the **byte ratio** (2.0) and §59.2's band ratio (34.0–34.9 ÷ 20.65 = 1.65–1.69) | whether a 64-weight `down` block pays the penalty E31's curve predicts |

Descriptive — it has no pass line, because no prediction of mine about this curve has survived
contact (E31 twice, E33 once, E61 once).

## 5. Predictions, fixed now

| # | prediction | falsified by |
|---|---|---|
| 1 | `G-E63a` fires on all five artefacts with zero differing bytes | any difference |
| 2 | `G-E63b` reads **< 20%** agreement — packed on noise weights is nowhere near int8 | ≥ 60% |
| 3 | `G-E63c` reads **100.00%** top-1 and max abs logit diff **< 1e-4** | below 99.90%, or > 1e-3 |
| 4 | **`G-E63d` reads `DESK-MODEL-OPTIMISTIC`** — I expect the measured carved int8 cell to come in **32–36 tok/s**, below §60.4's 36.6, because the int8 `down` block halves to 64 weights per cache line and E26's gathered-weight penalty is charged twice over | anything outside 32–36 |
| 5 | `G-E63e`: the int8/packed rate ratio reads **0.60–0.72**, i.e. *worse* than the 1/2.0 = 0.50 the byte ratio alone would give but *better* than parity — the carve's fixed costs (attention, head, router = 822 M of 928 M charged) dilute the format change | outside |
| 6 | The int8 carved artefact is **10.9 ± 0.2 GB** | outside |

**Prediction 4 is the one that costs me something.** §60.4 is this programme's headline and I am
registering, before the measurement, that I expect it to come in **below** its own projection.
If it reads above 36.6 I was needlessly pessimistic about my own best number; if below, the
headline moves down and I will have predicted it.

## 6. What E63 cannot claim, whatever it reads

- **Nothing about quality at 10 B.** `A10B-K3` is **noise weights**. E63 measures a rate and a
  numerical equivalence, never a BPB. The trained-carve question is H1's and is untouched.
- **Nothing that transfers to another shape.** E26 and §61.6's correction both say it: the carve
  penalty is a function of how much of the token is carved, and `A10B-K3` carves 11% of its
  charged weights. A different 10 B shape gets a different answer and must be measured too.
- **No claim from bytes ÷ bandwidth.** §59.2's rule stands and §61.6 is the cautionary case:
  every rate in E63 is measured, and the bands appear only as comparators.
- **No T4 time is asked for or implied.** CPU only.
- **Part B is not attempted on a contended box.** If the idle window has not happened, Part A
  ships alone and Part B stays owed, stated as owed.

## 7. Owed after E63

1. The same cell on a **trained** carved 10 B, once H1 produces one — the quality leg.
2. The 1.5 B extremum from E62 §11 item 1 (why 1.5 B is the worst at one byte and the best at
   half a byte); needs a fourth scale not currently cached.
3. `G-E61c` on an idle box and the 3 B speed cell — both ride the same window as Part B.
4. `C/T` on the donor engine; channel-granularity mixed precision; the `I8`−`T1` value-dependence
   (E61 §6); `G-E55a2`'s replacement interval gate; E56's `OCC_BAR` zero point; E42's void control.

---

## 8. ADDENDUM A — two gates as written cannot be answered (pushed before the run)

E4's precedent: **a gate that cannot be answered is MALFORMED, not failed, and may be
re-specified** — in writing, before the run it governs, with the reason. Both defects were found
while building the apparatus, not while reading a result.

### A.1 `G-E63b` asked for an artefact the format forbids

§4 specified the discrimination control as **int8-dense vs packed-carved on the same synthetic
weights**. There is no int8-dense synthetic artefact and there cannot be one: `--i8` requires
`--carve` because `donor_engine.c` refuses an FFN whose gate/up/down disagree in format (added in
§3 deliberately, so a half-converted layer cannot read as a plausible rate against the wrong byte
count). Writing one would mean weakening the load check the experiment depends on.

**Replacement, same purpose, strictly closer to what is being certified:** on the **same carved
synthetic file**, `--carve-k 3` against `--carve-k E` (every group selected).

| test | pass |
|---|---|
| per-position top-1 agreement between `k=3` and `k=E` | **≤ 60%** → **FIRES** |

This is a better control than the one it replaces, because it exercises **the carve itself** —
the mechanism `G-E63c` certifies — rather than a format change measured elsewhere. A near-100%
reading means the row list is not selecting anything and **`G-E63c` is void, not passed**.

### A.2 `G-E63c` cited a slice that does not exist at the shape it must run on

§4 specified *"per-position top-1 over 12,264 positions"* — the standard slice. That slice's ids
run to 151,936 and **`A10B` has `V = 32768`**, so they are not valid tokens there. The count was
carried over from E61 without checking it against the shape E63 actually measures.

**Replacement:** a deterministic synthetic prompt of **256 ids in range for each shape**, scored
by `--generate`, giving three numbers per pair:

| test | pass |
|---|---|
| generated ids (prompt + 24 new) identical | **100%** |
| prefill argmax agreement over all 256 positions | **≥ 99.90%** |
| max abs logit difference | **≤ 1e-3** |

Run at **both** shapes: `S05 --carve 16` (cheap, exercises every new path) and **`A10B --carve
256`**, the artefact the goal is about.

### A.3 What does not change

`G-E63a` (zero-tolerance inertness on five existing artefacts), `G-E63d` (the measured 10 B rate,
idle box only, `OCC_BAR` as admissibility) and `G-E63e` (the paired locality read) stand exactly
as registered, as do all six predictions. **Prediction 3's numbers now attach to A.2's three
tests**, which is where they were always pointed.
