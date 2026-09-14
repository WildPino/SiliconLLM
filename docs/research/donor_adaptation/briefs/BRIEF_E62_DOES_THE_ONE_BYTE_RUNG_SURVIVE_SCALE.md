# E62 — does the one-byte rung stay faithful as the donor grows?

**Pre-registered. Pushed before any runner exists and before any measurement is taken.**
Opened by `probes/E60_THE_RUNG_THAT_WAS_NEVER_BUILT.md` §7 item 3 (*"a third scale for §2's 19×
segment; two points are not a trend"*) and by `probes/E12_TERNARY_COST_VS_SCALE.md`'s own owed
item (*"what is still missing is the 3 B cell"*).

---

## 1. Why this, and why it is the most consequential thing open

Every headline this programme currently carries rests on one unmeasured extrapolation.

E60 found the only artefact that is both above the 50 tok/s bar and faithful: one byte per
weight, **dBPB +0.000066** at 0.5 B. E61 made it faster and confirmed the byte→rate curve. The
10 B projection that follows — **36.6–37.6 tok/s, 73–75% of the bar** (`SPEED_LEDGER` §60.4) —
assumes that rung is *still faithful at 10 B*. **Nobody has tested that**, and the two points
that exist do not reassure:

| donor | N (total params) | BPB fp32 (engine) | BPB int8 | **dBPB** |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.4941 B | 0.871810465558026 | 0.8718768840274047 | **6.64185e-05** |
| Qwen2.5-1.5B | 1.5437 B | 0.767606372511151 | 0.7688588381536873 | **1.2524656e-03** |

**×18.86 of damage for ×3.12 of parameters.** Fitted as a power law that is `dBPB ∝ N^2.578`.
Extrapolated to 10 B it is **~0.15 BPB = 30 σ_seed** — the rung would not survive to the target
and §60.4's headline would be void there. Extrapolated as *linear in N* it is **~0.008 BPB =
1.6 σ_seed** and the rung is fine. **The two readings of the same two points differ by ~19× at
the target scale.** One measurement at a third scale separates them.

**Qwen2.5-3B is already on this machine** (`models--Qwen--Qwen2.5-3B`, 2 safetensors shards),
shares the tokenizer (`vocab_size 151936`) so the standard slice and its chance line apply
unchanged, and needs **no code**: `--quant int8 --rule R8` has existed since E60 and
`qwen_export.py --model` is generic.

**And it is deterministic.** E61 measured this box at 4.44–19.57% foreign occupancy with nine of
ten cell-groups over `OCC_BAR`. A BPB pass does not care: E61 reproduced `05b F32` to
**|d| = 0.00e+00** across two runs 710 s apart. **E62 is deliberately scoped to the only axis a
dirty box cannot corrupt, and carries no speed claim of any kind.**

## 2. Checked, not assumed

Searched by artefact name — `Qwen2.5-3B`, `3B`, `third scale`, `dBPB` — across `docs/` and every
`*/results/`. **Every brief and probe the search returned was opened.**

| where | what it says | bearing on E62 |
|---|---|---|
| `BRIEF_E12_TERNARY_COST_VS_SCALE.md` §1–2 | asks this exact question **for ternary**, over {0.5 B, 1.5 B, 3 B}, and argues it *"is deterministic — it can be measured on a contended machine, which the speed work cannot"* | The design is E12's, re-used deliberately. E62 changes the format under test from ternary to int8 and the rule from R0 to R8. |
| `probes/E12_TERNARY_COST_VS_SCALE.md` §1 | verdict **`CHANCE-LINE`**: R0 is above chance at all three scales, so the ratio had nothing to measure. Table gives **fp32 BPB 0.871795121 / 0.767594958 / 0.724449797** and the chance line **4.069819** | **The 3 B fp32 reference already exists** — but from the PyTorch path, not the engine. It is E62's cross-path control, not its baseline. |
| same, correction paragraph | *"`R0` is `qwen_export.py`'s flag DEFAULT but is NOT the rule this programme ships… E12 measured a rule the pipeline does not use"*, and *"what is still missing is the **3 B cell**"* for R3 | **E62 pays that owed item for free** by carrying a third arm at the shipped packed rule. |
| `probes/E60_...md` §2, §7 item 3 | the two int8 points above; *"two points, one direction, stated as a segment and not a law"* | The estimand and the two anchor points. E60 refused to call it a trend; E62 is the experiment that may. |
| `probes/E61_...md` §4 | the engine BPB pass reproduces to `|d| = 0.00e+00` on two independent runs | Why `G-E62a`'s tolerance can be tight. |
| `BRIEF_E15_DOES_THE_7B_PREDICT.md` | asks whether the 7 B predicts, on the **ternary/speed** axis | Coder-7B is a different pretraining mix; mixing it into a dBPB-vs-N fit would confound format damage with corpus. **Excluded on purpose**, see §8. |

**No measurement of int8 damage at any scale other than 0.5 B and 1.5 B exists in this repo.**

## 3. What is measured

**Donor: Qwen2.5-3B**, `D=2048, F=11008, L=36, NH=16, NKV=2, HD=128, V=151936`, tied — read from
the local config, not assumed. **N = 3.0860 B total parameters** (2.7748 B non-embedding).

Three exports, with **E1's protocol verbatim**, the same one E60 used:
`--fold none --calib-seqs 32 --head-ternary`.

| arm | flags | ~size |
|---|---|---|
| `F32` | `--quant fp32` | 12.3 GB |
| `I8` | `--quant int8 --rule R8` | 3.1 GB |
| `PACKED` | `--quant packed --rule R3` | 1.6 GB |

BPB on the **standard slice**: 24×512, **51,870 scored bytes, 12,264 predicted positions**,
chance **4.069819**, ids `D:/_ktmp/e1/ids_qwen25-05b_tqh.bin` (tokenizer-identical across the
family). Engine `donor_engine_e61.exe`, `--threads 6`, default `--mvacc 4`.

**Estimand: `dBPB(N) = BPB(int8) − BPB(fp32)`, each donor against its OWN fp32 baseline.**
Absolute BPB is not comparable across scales — a 3 B is a better model — and E12 §2 said so
first. The relative form `dBPB / BPB_fp32` is reported alongside but **decides nothing**: the
gate is on the absolute number, because `σ_seed = 0.005` is an absolute constant.

## 4. Gates — thresholds fixed here, before anything runs

### G-E62a — PLANTED: the instrument reproduces what it already knows

Re-measure `05b F32` and `05b I8` BPB on this binary and slice.

| test | pass |
|---|---|
| both reproduce E60/E61 | **\|d\| ≤ 1.0e-08** |

E61 read `|d| = 0.00e+00` on `05b F32` twice, so this is tight on purpose. **FAIL ⇒ nothing below
is read.**

### G-E62b — the cross-path control at the new scale

The 3 B `F32` engine baseline against E12's PyTorch reference **0.724449797**.

| pass | **\|d\| ≤ 5.0e-05** |
|---|---|

The same comparison reads **1.5e-05** at 0.5 B (E12 0.871795121 vs engine 0.871810466) and
**1.14e-05** at 1.5 B (E12 0.767594958 vs engine 0.767606373). 5e-05 is ~3× the larger of the two.
**FAIL ⇒ the 3 B export is wrong and the cell is void, not adverse.**

### G-E62c — THE HEADLINE: does the cost explode?

`dBPB(3 B, int8)`. Bands fixed now, with what each implies at 10 B under its own fit:

| dBPB(3 B) | verdict | implied at 10 B | in σ_seed |
|---|---|---|---|
| **≤ 0.0025** | **`AT-MOST-LINEAR`** — the rung survives to the target | ≲ 0.008 | ≲ 1.6 |
| 0.0025 – 0.0060 | **`SUPERLINEAR-BUT-BOUNDED`** — reported as a segment, not a law | 0.01 – 0.06 | 2 – 12 |
| **≥ 0.0060** | **`COST-EXPLODES`** — **§60.4's 10 B headline is void at the target scale** | ~0.15 | ~30 |

0.0025 is *exactly* linear-in-N from the 1.5 B point (`1.2525e-03 × 1.999`); 0.0060 sits below
the power-law point (`7.47e-03`) by enough that a near-miss does not read as an explosion. The
bands are set by the **hypotheses**, not by any dispersion, because this axis has none: the same
BPB pass reproduces bit-for-bit.

### G-E62d — the RANK partner (E14 §3), registered before the run

A SCORE needs a RANK partner. Two orderings must hold, or the SCORE is reported without a trend
claim:

1. **dBPB strictly increasing in N** across 0.5 B → 1.5 B → 3 B.
2. Within the 3 B cell: **`BPB(F32) < BPB(I8) << BPB(PACKED)`**.

### G-E62e — E12's owed 3 B cell, at the rule this programme actually ships

`BPB(3 B, packed, R3)` and its position against the chance line **4.069819**. Descriptive, no
verdict attached — it closes `probes/E12_...md`'s own open item and extends the table E12 could
only fill two thirds of. The R3 packed arm crosses the chance line **between 0.5 B and 1.5 B**
(4.531234 above, 3.475707 below); the 3 B cell says whether it keeps falling.

## 5. Predictions, fixed now, each with what falsifies it

| # | prediction | falsified by |
|---|---|---|
| 1 | `G-E62a` reproduces at \|d\| = 0.00e+00, not merely ≤ 1e-08 | any nonzero digit |
| 2 | `G-E62b` reads \|d\| in **1e-05 … 3e-05**, in family with the other two scales | > 5e-05, or < 1e-06 (which would mean the other two disagreements have a cause I have not found) |
| 3 | **`G-E62c` reads 0.0030 – 0.0060 → `SUPERLINEAR-BUT-BOUNDED`.** I expect growth faster than linear but slower than N^2.578, because a power law fitted to two adjacent points almost always overstates, and because the 0.5 B cell's dBPB is so small (6.6e-05) that it is a fragile lower anchor | anything outside |
| 4 | `G-E62d` clause 1 holds; clause 2 holds with `PACKED` **below** chance (like 1.5 B, unlike 0.5 B) | either ordering breaking |
| 5 | `G-E62e`: `BPB(3 B, packed, R3)` reads **3.0 – 3.5**, i.e. still falling but by less than the 0.5 B → 1.5 B step (4.531 → 3.476) | outside, or above the chance line |
| 6 | The 3 B `F32` engine BPB reads **0.7244 ± 0.0001** and is the best fp32 score in the family | not monotone below the 1.5 B's 0.7676 |

**Prediction 3 is the one that can embarrass this brief**, and it is deliberately placed where
neither registered extreme sits: if the reading comes in at ≤0.0025 the power-law fear was
groundless and I over-alarmed; if it comes in ≥0.0060 the programme's 10 B headline dies and I
under-alarmed. Both are worse for me than the middle, which is why the middle is on record.

## 6. What E62 cannot claim, whatever it reads

- **Three points do not establish the 10 B value.** 3 B → 10 B is another ×3.2, and every band in
  §4 states an *implication under its own fit*, not a measurement. A `COST-EXPLODES` reading kills
  the headline; an `AT-MOST-LINEAR` reading does **not** prove the rung survives — it removes the
  strongest reason to think it does not.
- **Nothing about speed.** No `--bench`, no tok/s, no bandwidth. The box is dirty (E61 §5) and this
  experiment is scoped to the axis that does not care. The 3 B speed cell is owed separately and
  belongs to the clean-box run that also repeats `G-E61c`.
- **Nothing about Coder-7B or any other family.** A different pretraining mix would confound
  format damage with corpus; the fit is within Qwen2.5 base only.
- **Nothing about healing.** These are post-hoc conversions. Whether training inside the format
  changes the curve is H2T's question and is T4-gated behind H1.
- **No T4 time is asked for or implied.**

## 7. Owed after E62, whatever it reads

1. `G-E61c` repeated on an idle box, with the 3 B speed cell in the same clean sweep (E61 §9).
2. The 10 B cell at 1 B/weight, measured — which at the carve needs the row-selected `matvec_sel`
   and `matvec_colacc` to accept an int8 kind (`donor_engine.c`: *"row-selected matvec is
   implemented for the plain packed kind only"*). Scoped here because E62 makes its value
   conditional on `G-E62c`.
3. `C/T` on the donor engine; channel-granularity mixed precision; the `I8`−`T1` value-dependence
   (E61 §6, reopened); `G-E55a2`'s replacement interval gate; E56's `OCC_BAR` zero point; E42's
   void control.
