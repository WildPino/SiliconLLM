# BRIEF E64 — what does the CARVE cost on an INT8 FFN?

**Pre-registered. Pushed BEFORE the apparatus exists and before one BPB is measured.**
Nothing in the probe may contradict this file; anything it has to change is a dated addendum,
in the E40/E43/E63 style.

**Scope: QUALITY only. Not one tok/s is measured here, and §7 says why that is deliberate.**

---

## 0. The question, in one sentence

**Every carve cost this programme has ever measured was measured on top of a TERNARY FFN.
Nobody has measured what the carve costs on an INT8 FFN — and E63's retraction (§62.12) makes
that the number that sizes H1's job.**

## 1. Why this is the next measurement

`BRIEF_H1` §1 reads the damage ladder off `results/e37_sparsity_cost.json`, one instrument, the
frozen 51,870-byte slice. Re-expressed against the one baseline no treatment has moved —
fp32 dense **0.767595**, chance **4.069819**, gap **3.302224** (ledger §62.12):

| treatment | ΔBPB | share of the dense→chance gap |
|---|---|---|
| **half a byte/weight** (ternary FFN, all 256 groups on) | **+2.708112** | **82.01%** |
| **carve to `k=3`** (1.17%) on top of it | +0.553692 | **16.77%** |
| **one byte/weight** (int8, E62 at 1.5 B) | +0.0012525 | **0.038%** |

The first and third rows are two points on **one axis with a cliff between them** (E60). The
second row is a **different** axis, and it has only ever been measured **from the bottom of the
cliff**.

**So the composition in the middle of the table is missing**: `int8 FFN + carve`. It is the
format H1 would train if it took §62.12's advice, and its post-hoc cost is unknown.

**What it decides.**

* If `carve-on-int8 ≈ carve-on-ternary (+0.55)`, then the carve is a property of *selection*,
  not of the weights' precision, post-hoc stays dead at 1.17%, and **H1's healing job at one
  byte shrinks from +3.22 BPB to ≈+0.55 — six times smaller.**
* If it is **much smaller**, a fast post-hoc 10 B becomes arguable for the first time since E38.
* If it is **much larger**, §62.12's "train at one byte" advice is wrong and must be withdrawn
  before any T4 time is spent on it.

**All three outcomes change a decision.** That is the test this brief must pass to be worth
running, and it passes it.

**E19 already returned `CARVE-DOES-NOT-RANK` and closed the carve route for this donor, and E64
does not reopen it.** E64 is not asking whether the carve is a good idea — three probes say it
is not, post-hoc. It asks a **mechanism** question whose answer sizes a run that is already in
flight: *is the carve's damage a property of selection, or of the precision of what survives?*
H1 is spending T4 time healing a number, and §62.12 has just shown that 82% of that number is
the format rather than the mask.

## 2. Checked, not assumed

Per `feedback_search_before_claiming_a_gap` — and this brief is written on the same day that
rule was broken three times — every artefact below was searched **by name** and **opened**:

| opened | what it holds | why it does not answer E64 |
|---|---|---|
| `BRIEF_H1` §1 | the damage ladder above; "the carve is 16%" | carve measured **on ternary** |
| `probes/E37` | `S15-K3 = 4.029398`, the carve's +0.553692, exchange rate 0.007162 | ternary FFN |
| `probes/E38` | `SELECTION-IS-DEAD`; oracle at 1.17% = 4.131817, **above** chance 4.069819 | ternary FFN |
| `probes/E19` | **`CARVE-DOES-NOT-RANK`** — and FFN-only carving cannot reach the target anyway; two findings either of which closes the carve route **for this donor** | carve priced on a **ternary** FFN; says nothing about whether its cost depends on precision |
| `probes/E60` | the 1-byte rung; `05b_i8h` +0.000066, `15b_i8h` +0.001252 | **uncarved** |
| `probes/E62` | 3 scales, `THE-COST-DOES-NOT-RANK`, 6.64e-05 → 1.25e-03 → 5.79e-04 | **uncarved** |
| `probes/E63` | carve **and** int8 in one file, 100.0000% top-1 | **synthetic NOISE weights, no BPB exists** |
| `probes/E39`, `E40` | the attention axis: `RANK-BUYS-SPEED`, then `ATTENTION-LEVERS-EXHAUSTED` (`R128` ~113–130 tok/s) | speed only, noise weights |
| `probes/E41` | **`VERDICT-UNRESOLVABLE`** (run 1's verdict did not survive addendum B) | partition axis, and it has no standing verdict to contradict |
| `probes/E42` | **no registered verdict** — the planted null control went VOID | predictability axis; its void control is a standing open item, not an input here |
| `briefs/BRIEF_E43` | the vocabulary axis; **speed half permanently VOID**, units half done | no BPB on the byte axis |

**A grep of `briefs/ probes/` for carve-and-int8 in one expression returns only E63 and this
brief's own parent.** E63 proves the *engine* computes it exactly; **no BPB has ever been taken
through that path**, because E63's artefact is noise.

## 3. The measurement

**Instrument**: the frozen 51,870-byte slice and `e1_bpb_through_engine.py`, the same instrument
E37, E60 and E62 used — chance **4.069819**, and E62 cross-validated it against E12's PyTorch
fp32 to |d| ≤ 1.53e-05.

**Donors**: Qwen2.5-0.5B and Qwen2.5-1.5B at the revisions E62 pinned and checked on disk.
Two scales, because one scale is not a trend (`feedback_rank_partner_survives_refusal`) — and
**E62's own rank break is the standing warning that two are not one either**, so §6 forbids
any exponent.

**Arms**, per donor, all through the same engine binary and the same slice:

| arm | FFN weights | carve | purpose |
|---|---|---|---|
| `fp32` | fp32 | off | the baseline nothing has moved |
| `tern` | ternary (0.5 B) | off | reproduces E37's `K256` row — **replication, not discovery** |
| `i8` | int8 (1 B) | off | reproduces E60/E62 — **replication** |
| `tern-k3` | ternary | `k=3` of `E=256` | reproduces E37's `K3` row — **replication** |
| **`i8-k3`** | **int8** | **`k=3` of `E=256`** | **the cell. Never measured.** |

`k=16` (6.25%) is added at both formats if the apparatus is cheap enough to run it; it is
**descriptive and ungated**, and no verdict may rest on it.

**The router is held fixed across arms** — same `--carve-labels`, same router source, same seed.
The treatment is the FFN's *precision* and nothing else. Per `feedback_control_arm_different_code_path`,
if the int8 arm cannot use the identical router the cell is **VOID**, not adjusted.

## 4. Gates

| gate | requires | kind |
|---|---|---|
| `G-E64a` | **planted control**: `tern`, `i8` and `tern-k3` reproduce E37/E60/E62's published BPB for the same donor and slice to \|d\| ≤ 1e-06 | the instrument must FIRE on three known positives before the new cell counts |
| `G-E64b` | **discrimination**: at `k = E = 256` (all groups on) the carved file is **BIT-IDENTICAL** to the uncarved file at the same precision, both formats | H1's `G-H1a` trick; rules out a mis-wired mask |
| `G-E64c` | the `i8-k3` cell is reported **with** its `i8` baseline, and the carve cost is stated as `BPB(i8-k3) − BPB(i8)` — never against a ternary or fp32 baseline | **the floor rule, written into a gate because I broke it today** (§62.12) |
| `G-E64d` | **ORDINAL, no tolerance**: does `carve-on-int8` cost MORE, LESS or the SAME (within σ_seed) as `carve-on-ternary` at the same `k`, at **both** donors, with the **same sign** at both | the verdict |

**`G-E64d` is ordinal on purpose.** `feedback_gate_vs_measured_dispersion`: I have no measured
dispersion for this cell, so I may not register a tolerance on it. σ_seed for the comparison is
taken from E62's per-donor values as published (0.013 at 0.5 B, 0.250 at 1.5 B) — **not
re-derived here**, and named in the probe.

**Bands** (assigned from run 1 only; E36's run-2 rule is adopted):

| band | condition |
|---|---|
| `CARVE-IS-PRECISION-BLIND` | the carve costs the same at both formats, both donors |
| `CARVE-IS-CHEAPER-ON-INT8` | strictly less at both donors, same sign |
| `CARVE-IS-DEARER-ON-INT8` | strictly more at both donors, same sign |
| `THE-COST-DOES-NOT-RANK` | the two donors disagree on the sign — **the E62 outcome, and it is a real result, not a failure** |

## 5. Predictions, registered before the apparatus exists

1. **`CARVE-IS-PRECISION-BLIND`.** The carve deletes whole neuron groups; what it deletes does
   not depend on how precisely the survivors are stored. I expect `|Δ| < 0.05 BPB` between the
   two formats' carve costs.
2. The `i8-k3` cell lands at **0.77 + 0.55 ≈ 1.30–1.35 BPB** at 1.5 B — i.e. **far below chance
   and far below `tern-k3`'s 4.029398**, because it carries the carve's damage but not
   ternarisation's 82%.
3. **The 10 B consequence I will NOT be able to draw**: even if 2 holds, it says nothing about
   10 B. E62's rank break forbids the extrapolation and §6 forbids me writing one.
4. `G-E64a` fires on all three replications. If it does not, the instrument has drifted and
   **no cell in E64 may be read.**
5. At `k=16` the gap between formats is **smaller** than at `k=3` (less is deleted).

**Scored honestly**, including against the contaminated alternative if any cell is excluded
(`feedback_score_prediction_against_contaminated_fit`).

## 6. What E64 may NOT conclude

1. **No extrapolation to 10 B.** Two donors, and E62 proved this estimand does not rank over
   three. Any 10 B sentence in the probe is a defect.
2. **No rate.** Not one tok/s is measured. `G-E63d` stays `VOID` and OWED and E64 does not touch
   it. E63's measured **−1.7%** rate cost for one byte is on **synthetic** weights at
   `A10B-K3` and may not be composed with any BPB here.
3. **No claim that post-hoc is reopened.** Even the best outcome leaves E38 standing: an oracle
   at 1.17% is above chance. E64 prices one term; it does not revive a closed route.
4. **No advice to change a run already in flight.** H1 is launched. E64 informs H1's *format
   choice if and when it is revisited* — it is not an instruction to stop anything.
5. **Replication is labelled as such.** Three of the five arms are replications of published
   numbers and the probe must call them replications, not findings.

## 7. Why quality-only, and why now

Because **speed is not the blocker** (§62.10: E40 has a genuine 10 B at ~113–130 tok/s with a
live FFN, on noise weights) and **the byte axis's cliff is the biggest single term in the
damage ladder** (§62.12: 82%). The programme spent §34–§50 pricing shapes on the 17% axis. E64
measures the one composition that the cliff makes interesting, and it needs **no idle box, no
GPU and no user time** — which is the other reason it is the right thing to run while the clean
hour and H1 are both outstanding.

## 8. Apparatus, and the one thing that makes it non-trivial

`qwen_export.py --quant` is an **exclusive choice**: `carved` and `int8` cannot both be asked
for. Worse, line 393 guards `(--quant int8) != (--rule R8)` and exits — correctly, because E60
found that mixing them silently either clips 127 levels to 3 or mislabels ternary codes as int8.

**E64 needs a file whose FFN is R8/int8 while its other organs stay R3/ternary — a per-matrix
rule, which this exporter has never had.** The engine side already exists and is verified:
E63 added `MK_I8`/`MK_I8_T` to `donor_engine.c`, and the carved-FFN loader already refuses a
*mixed* FFN, which is the invariant that keeps this honest.

Plan, in order, each pushed before the thing after it runs:

1. port `w_tag_i8` / `w_tag_i8_t` from `synth_export.py` into `qwen_export.py`;
2. add `--ffn-rule` (default: follow `--rule`) so the carved FFN's precision is stated
   explicitly in the file and **printed in the CONFIG line** — `feedback_config_must_appear_in_output`:
   the runner must REFUSE a cell whose `CONFIG` does not confirm what was asked;
3. widen the line-393 guard to a per-matrix check rather than deleting it;
4. Gate V3 (`e1_bpb_through_engine.layout_bytes_v4`) must predict the new layout at **zero
   tolerance** before any BPB is read — it fired for real twice in E63 and is not to be relaxed.

**If step 3 cannot be done without weakening the guard that E60 put there, E64 is ABANDONED and
the brief says so.** A measurement is not worth removing a check that has already caught a
silent corruption.

---

## 9. Addendum A — 2026-09-14, written BEFORE the run, after opening `results/e37_sparsity_cost.json`

Three things in §3–§5 are wrong on contact with E37's actual result file. Per the **E4
precedent** — *a gate that cannot be answered as written is MALFORMED, not failed, and may be
re-specified* — they are re-specified here, **before the export finishes and before one BPB is
read**. Nothing below was informed by an E64 number, because none exists yet.

### A.1 `G-E64b`'s "BIT-IDENTICAL" is impossible, and E37 already knew it

The carved writer applies a **permutation of the `F` axis** (rows of gate/up, columns of down).
The arithmetic is exact but the **summation order is not the same**, so the carved forward at
`k = E` cannot be bit-identical to the uncarved one. E37's own version of this control,
`G-E37A`, is registered at a **tolerance**, and its reading is:

| | value |
|---|---|
| `bpb_dense` | 3.47570637184527 |
| `bpb_carved_kE` | 3.4757066520304316 |
| `bpb_diff` | **2.801851617384443e-07** |
| `bpb_tol` | 1.0e-04 |

**`G-E64b` is re-specified to E37's form**: at `k = E = 256` the carved file must reproduce the
**same-format uncarved** BPB to **\|d\| ≤ 1e-04**. Registering the tolerance E37 used, on the
instrument E37 used, rather than inventing one.

### A.2 The int8 arm has no uncarved counterpart, and the gate says so instead of pretending

For ternary the uncarved reference exists on disk (`e37_dense_nf.bin`). For int8 it does not:
`--ffn-rule` requires `--quant carved`, and a whole-file `--quant int8` artefact would also
change attention, the head and the router — it is **not** "the same file with the carve off".

**So `G-E64b` fires on the TERNARY arm only, and that is stated as a limitation rather than
papered over.** Writing a version that "passes" on the int8 arm would make it a tautology
(comparing the file to itself), and this programme has a standing memory that *a metric that
always passes discriminates nothing*. The control still does its job: it proves the carve
machinery and the engine's runtime `--carve-k` override are correctly wired **on today's
binary**, and both formats go through **the same carve code path** — only the matrix kind
differs.

### A.3 A single `k` is the wrong comparator: the carve ladder is NON-MONOTONE

§3 named `k = 3` as the cell. E37's ladder, on the same instrument and donor, is:

| `k` | 256 | 64 | 32 | 16 | 8 | 4 | **3** | 2 | 1 |
|---|---|---|---|---|---|---|---|---|---|
| BPB | 3.4757 | 3.9274 | **4.0744** | 3.9868 | 3.9967 | 4.0234 | **4.0294** | 4.0149 | 3.9898 |

**It does not rank** — `k=32` is worse than `k=16`, and `k=1` is better than `k=3`. That is
E19's `CARVE-DOES-NOT-RANK` visible in the numbers, and it means **a two-cell comparison at one
`k` could show either sign by picking the `k`.**

**`G-E64d` is therefore re-specified to compare LADDERS, not cells**: the full E37 ladder
`k ∈ {256, 64, 32, 16, 8, 4, 3, 2, 1}` is run at **both** formats from **one file each**, and
the verdict is read off the ladder as a whole:

| band | condition |
|---|---|
| `CARVE-IS-PRECISION-BLIND` | the two ladders agree at **every** `k` within E62's σ_seed for this donor (0.250 at 1.5 B) |
| `CARVE-IS-CHEAPER-ON-INT8` | int8's carve cost is strictly smaller at **every** `k` |
| `CARVE-IS-DEARER-ON-INT8` | strictly larger at **every** `k` |
| `THE-LADDERS-DISAGREE` | the sign changes with `k` — **a real result**, and the one E19 + E37's non-monotonicity make most likely |

This is cheap: one exported file per format, nine engine invocations each, and it supplies the
**RANK partner** E14 §3 requires for the SCORE, which §4 as written did not have.

### A.4 What this addendum does NOT do

It does not touch the **predictions** in §5 — they stand as registered and will be scored as
written, including prediction 1's `|Δ| < 0.05 BPB`, now read across the ladder rather than at
one `k`. It does not touch §6's prohibitions. **`G-E63d` is still `VOID` and OWED**, and E64
still measures no rate.

---

# ADDENDUM B — the re-run, pre-registered 2026-09-14, pushed BEFORE the runner is touched

**Run 1 is VOID and stays void.** This addendum registers a *new* measurement. Run 1's int8
numbers may not be quoted as corroborating anything this run returns (**E36 run-2 rule**).

## B.1 Why run 1 is re-specifiable at all, and why that is not a licence

`G-E64a` failed at `max|d| = 8.838e-04` against `1e-06`. The probe's section 7 and ledger
§63.3 then **measured** the cause: not build noise, but **E50's change of the default attention
kernel** (`61d1c29`, 2026-09-13, `g_attn` from `ATTN_SERIAL` to `ATTN_AVX4`). E37's result was
committed the day before, on `donor_engine_e26.exe`. Two runs, each registered before it ran,
both taken:

| `k` | E37 published | `e26` | `e63 --attn serial` | `e63` default (`avx4`) |
|---|---|---|---|---|
| 256 | 3.4757066520304316 | \|d\| = 0 | \|d\| = 0 | **5.889e-07** |
| 32 | 4.074431016419263 | \|d\| = 0 | \|d\| = 0 | **8.838e-04** |

So `G-E64a` asked *"same binary?"* when the quantity that decides replication is *"same kernel
arm"* — which it never asserted and the engine has printed in its `CONFIG` line since E50.
**That is the E4 precedent: a gate that cannot answer the question it was written for is
MALFORMED, not failed, and may be re-specified.**

**This is explicitly NOT the `E40 addendum A` case** (a gate that fired correctly being re-run
until it passes). Two things separate them, and both must hold or this addendum is illegitimate:

1. **The repair makes the control STRICTER, not looser.** The tolerance goes *down*, from
   `1e-06` to `1e-09`, and a new assertion is added that run 1 did not have.
2. **The repair was forced by a measurement taken before it was written**, not chosen after
   browsing run 1's deltas for a way through. The deltas suggested "widen the tolerance"; the
   measurement said the opposite — there is nothing to widen for.

## B.2 `G-E64a2` — the repaired planted control, in two clauses that must BOTH hold

| clause | requires |
|---|---|
| **a2.1 — the arm is ASSERTED, not assumed** | every engine invocation passes `--attn serial`, and the runner **parses the engine's own `CONFIG` line and REFUSES the cell** unless it reports `attn=serial`. A cell whose `CONFIG` does not confirm what was asked is not read. |
| **a2.2 — E37's ladder reproduces at `1e-09`** | all **nine** of E37's published `k` values, to \|d\| ≤ **1e-09** — three orders of magnitude tighter than run 1's `1e-06`. |

Clause a2.1 is `feedback_config_must_appear_in_output`'s own remedy, applied for the first time
as a **gate** rather than as a log line: *the runner refuses the cell whose `CONFIG` does not
confirm what it asked for.* Run 1 had the `CONFIG` line in its output and nothing read it.

**`1e-09` is a real risk and is registered as one.** I have measured exactly **two** of the nine
cells (`k = 256` and `k = 32`) and both are exact. **The other seven are untested.** If any of
them misses `1e-09`, the kernel arm is **not** the whole story, `G-E64a2` does not fire, and
**E64 stops again** — this addendum does not get a third attempt without a new measured cause.

## B.3 Everything else about the design is unchanged

Same artefacts (the int8 export `D:/_ktmp/e64/e64_carved_i8.bin` is **kept**, verified, and not
rebuilt — its sidecar confirms `ffn_bytes_per_weight=1.0`, `kinds=['MK_I8','MK_I8_T']`), same
ladder `k ∈ {256,64,32,16,8,4,3,2,1}`, same `G-E64b` discrimination control at `1e-04`, same
verdict gate `G-E64d`, same frozen slice.

**One apparatus change, from E65's run 1:** the result is written to a **new** path,
`results/e64_carve_on_int8_run2.json`. Run 1's file is a published record and is not mutated,
and no arm may be resumed from a cache that lives inside the file a control validates against.

## B.4 Predictions — registered, falsifiable, and the mechanism is named

1. **`G-E64a2` fires**: all nine cells at \|d\| ≤ 1e-09, and all `CONFIG` lines report
   `attn=serial`. *If not, E64 stops.*
2. **The carve costs MORE on int8 than the same ladder costs on ternary.** This is the
   programme's own `§62.12` lesson turned into a prediction: half a byte spends **82%** of the
   dense→chance gap before the carve is applied at all, so E37's ternary ladder measures the
   carve **on a base that was already destroyed**, and a base with more left to lose should lose
   more. Concretely: `carve_cost_i8(k) > carve_cost_tern(k)` at **`k = 32` and `k = 16`**, the
   two rungs that bracket E18's 10%-activation budget.
3. **At `k = 32` the int8 carve cost lands between `+0.10` and `+1.20` BPB** over its own
   `k = E` baseline. The band is deliberately wide because prediction 2 asserts only an
   ordering, and I have no measurement of this composition at any `k`.
4. **The int8 `k = E` baseline beats the ternary `k = E` baseline by more than `2.0` BPB**,
   because `§62.12` prices the byte axis at 82% of a 3.302224 gap. *This is close to a planted
   positive — it should be easy — and if it fails, `§62.12` is in trouble, not E64.*
5. **Free-running stays at the floor on every carved int8 arm.** No arm generates.

## B.5 What the re-run may NOT conclude

1. **Nothing about 10 B, and nothing about rate.** No tok/s is measured. **`G-E63d` stays
   `VOID` and OWED**, and it is blocked on an idle machine here, not on this run.
2. **Nothing about a TRAINED carve on int8.** H1 trained a carve on a **ternary** FFN; this is
   post-hoc on int8, and — exactly as in E65 — post-hoc is a **floor**, not a verdict.
3. **Nothing that promotes run 1.** Its numbers stay unpublished whatever this returns.
4. **Nothing about the 7 B.** E15's `DOES-NOT-PREDICT` is a separate object on a different
   donor, and this run does not touch it.

