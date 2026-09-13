# BRIEF E53 — A REAL EXPONENTIAL INSTEAD OF A LIBM CALL

**Registered before any E53 code exists.** Nothing below has been measured on the treatment arm.

## 1. Why this is next, and why it could not have been written a week ago

E51 tried to make `expf` cheap with a compiler flag and failed, but it produced the number that
makes this experiment worth doing and it explained the mechanism that makes it *possible*:

* **`expf` on this toolchain IS `(float)exp((double)x)`** — `cvtss2sd ; callq <exp> ; cvtsd2ss ;
  retq`, read out of the object code. Every softmax element pays a call frame, two conversions,
  and the **entire double-precision `exp` routine**, to produce a float.
* **`G-E51d` priced the attention softmax pass at `1.6890 ms` of a `13.8870 ms` token at mean
  position 640 — `12.16%` — from three reps dispersing `1.8%`.** As a slope that is
  `0.002639 ms/position` against E49's `b(avx4) = 0.008144`, i.e. **32.4% of the whole context
  term**, and the context term is what sets `C50`.
* **E9 already measured the other site**: the SwiGLU glue is `530,432 expf per token` and was
  `7.5% of the token` *at Coder-7B*. That one is **per token, not per position**, so it lands on
  the intercept `a`, not on `b`.

One kernel serves both. `C50 = (20 − a)/b` improves from both directions.

**E51's lesson is the design constraint.** A flag told the compiler it was allowed to do less
work and the work moved into the callee. **Explicit code cannot relocate anything**: if the
polynomial is in the loop, the polynomial is what runs. And `-ffast-math` remains forbidden
(Phase 35) — this brief does not ask for it and does not need it.

## 2. What is built

A single AVX2 kernel, `vexpf8`, evaluating `e^x` on 8 floats:

1. `y = x · log2(e)`; `n = round(y)` (`vroundps`), `r = y − n` on `[−0.5, 0.5]`.
2. `2^r` by a degree-5 minimax polynomial in Estrin/FMA form.
3. `2^n` by constructing the exponent field directly (`n + 127 << 23`), multiplied in.
4. Explicit flush of `x < −103` to `+0` and saturation of `x > 88` to `+inf`, so no branch and no
   denormal path.

Wired to **both** call sites behind **one** runtime flag, `--fexp {libm,poly}`, default `libm`:

* `donor_engine.c:1175/1179` — the attention softmax (`a[t]=expf(a[t]-mx); sum+=a[t];`)
* `donor_engine.c:719` — `silu`, via `donor_engine.c:1049/1237/1242`

**`CONFIG fexp=libm|poly` is printed on every run, in every mode.** That is E50's law
(*a configuration that does not appear in the output is a configuration nobody can audit*), and
it is the reason the slow attention kernel hid for twenty-two experiments.

**One interaction is closed in advance, not discovered later.** The `smrep`/`avrep` repetition
probes exist to make a loop's cost measurable by running it 2× or 3× and folding the discarded
result back as an exact zero — which requires the discarded pass to be **bitwise** equal to the
kept one. That still holds under `poly` (same code, same inputs, deterministic), so the probes
remain valid. They are nevertheless **printed with the `fexp` arm beside them**, because a probe
that reports a cost without naming which kernel produced it is the same defect in a new place.

## 3. Parity cannot be bit-exact, and pretending otherwise would be the failure mode

Two independent reasons, both stated before any measurement:

1. A minimax polynomial is a **different approximation** to `e^x` than the double routine
   rounded to float. Agreement is to a few ulp, never to zero.
2. The attention softmax contains a **reduction**. Eight lanes summed horizontally is a
   different summation order, so `sum` would differ even if every element were bit-identical.
   (`silu` has no reduction, so its site is elementwise-only — a strictly easier case, and the
   two are reported separately.)

Therefore **`G-E53b`'s numeric bound does not license anything on its own.** Phase 60's law
applies at full force: *kernel-bit-exact does not compose to system-correctness* — and here the
kernel is not even bit-exact. **The end-to-end gates are the gates.** In particular the greedy
argmax is **discrete**: a change far below any BPB threshold can still flip a token, and E50
exists because that distinction was nearly missed once already.

## 4. The gates

### `G-E53a` — the kernel is right, and the instrument that says so can fail (planted control)

`vexpf8` is swept against a double-precision reference over `x ∈ [−104, 89]` at 2^20 points plus
every special case (0, −0, the flush and saturation boundaries, ±inf, NaN). Reported: max
relative error and max ulp.

**The control**: the same harness is run against a deliberately degraded kernel (the degree-5
polynomial truncated to degree 3). **It must exceed the bar.** If it does not, the harness cannot
see a bad exponential and nothing it says about the good one counts. **Nothing below is read if
this control fails.**

### `G-E53b` — the numeric bar

Max relative error of `vexpf8` against the double reference, over the range the softmax actually
feeds it (`x = a[t] − mx ≤ 0`) and over silu's range: **≤ 1e-6** (about 8 ulp at float
precision). Above that, the arm is not measured for speed at all.

### `G-E53c` — end-to-end, and this is the one that decides

Both on the standing ids (`D:/_ktmp/e1/ids_qwen25-15b_tq.bin`), `libm` against `poly`:

* **`c1` continuous:** `|ΔNATS_TOTAL| / NATS_TOTAL ≤ 1e-3` over 12,264 positions — the same bar
  `G-E49a` and `G-E51a` used, with the same planted control, which fired at `1.593e-01`. The
  actual value is reported to full precision whatever it is.
* **`c2` discrete:** A1 teacher-forced greedy **160/160**, on `avx4`, with E6's planted controls
  reproducing **3/160** and **10/160**. This is `G-E50a`'s exact shape and it is not optional:
  **`c1` passing and `c2` failing is a real outcome**, and it would mean the poly arm is a
  different model, not a faster one.

If `c2` fails, the speed phase does not run and the finding is recorded as a failure. **A faster
engine that answers differently has not been sped up.**

### `G-E53d` — the speed, under the bar E52 just derived

`e44_interval.py` with `OCC_BAR = 4.39`, ≥5 reps, both arms, the same session, **interleaved**,
at the positions E49 fitted (`n ∈ {40, 160, 320, 640, 1280}` — mean positions 20…640).

* E49 addendum C's dispersion rule applies: **a window whose reps disperse by more than 6% takes
  no verdict and enters no fit.** If fewer than two windows survive, `C50` is **NOT COMPUTABLE**
  and the runner says so instead of extrapolating (`g_e51c_summary`'s behaviour, already built).
* Reported: Δ per window with its dispersion; the refit `a'`, `b'`; the new `C50`; and the
  **attribution between the two sites** from the `smrep` probe (attention) and a matching
  `silurep` probe (FFN), each reported with its own dispersion.
* **No paired-ratio statistic is promoted.** E51 A.4's drift-pairing rule is still unregistered
  and un-controlled; it may appear here as a diagnostic, labelled as such, and may not carry a
  verdict. That is E14 §6.

## 5. Predictions, registered

| quantity | prediction |
|---|---|
| `G-E53a` control | fires; the degree-3 kernel exceeds 1e-6 by at least two orders |
| `G-E53b` max rel err | **1e-7 to 5e-7** — a few ulp, comfortably inside the bar |
| `G-E53c1` | **PARITY HOLDS**; the actual `\|Δ\|/N` lands at **1e-6 to 1e-5**, i.e. 2–3 orders inside the bar |
| `G-E53c2` | **160/160 holds** — but this is the claim that can break, and it is why it is gated |
| fraction of `S` recovered | **50–75%** — the polynomial still costs ~10 flops per element, and `S` also contains the store, the accumulate and the loop |
| `b` | `0.008144` → **0.0062–0.0068 ms/pos** |
| silu site | **3–8% of the intercept**; this is the one number here I am *transferring* from a different model (E9, Coder-7B, 7.5%) and it is therefore to be **measured, not believed** |
| **`C50`** | `1443` → **1750–2000** |
| the short-context headline | mean position 20 moves by **+2 to +5% only** — almost all of this win is context, exactly as E49's was |

**What would make this a failure rather than a disappointment:** `c2` below 160/160, or a
recovered fraction under ~25%, which would mean the call frame was not the cost and the double
routine was being predicted and pipelined better than I think.

## 6. Scope

E53 changes one arithmetic kernel and the two places that call it. It does not touch the
attention kernel, the carve, the quantisation, the tokenizer, the weights, or any gate constant.
It may not be quoted as a new headline rate: the arm is `R128 --carve-k 3` on **synthetic**
weights and the bridge to real weights is E45's, *supported and not certified*.

`donor_engine_e52.exe` — byte-identical to the present `donor_engine.exe`, itself identical to
`donor_engine_e50.exe` — is frozen before the first edit, so every pre-E53 reading stays
reproducible by command.

---

# ADDENDUM A — THE KERNEL PASSED, `G-E53b` AS WRITTEN WAS MALFORMED, AND FOUR THINGS ARE REGISTERED BEFORE THE ENGINE RUNS

**§A.1 and §A.2 report measurements already taken** (the kernel's own gates, which §4
registered before the kernel existed). **§A.3 onward is registered before any engine cell
exists** and is pushed before the runner is started.

## A.1 `G-E53a` FIRES and the kernel is good to 1 ulp

`e53_vexpf8_test.c` includes `vexpf8.h`, the same header `donor_engine.c` includes. There is
one definition of the kernel and this is a measurement of that one.

| | max rel (normal ref) | max ulp | denormal band | zero / inf / NaN |
|---|---|---|---|---|
| `vexpf8` | **1.1920e-07** | **1** | 1.00 denormal ulp | all exact |
| degree-2 control | 5.5718e-05 | 661 | 330 denormal ulp | all exact |

**`G-E53a` FIRES**: the degraded kernel reads 5.57e-05, **56× the 1e-06 bar**. The harness can
see a bad exponential, so its reading of the good one counts.

**Two predictions scored, one right and one wrong.**

* `G-E53b` max rel err: §5 said **1e-7 to 5e-7**; measured **1.1920e-07**, exactly one ulp — the
  smallest value the prediction could have taken. **Right.**
* `G-E53a` control: §5 said the degraded kernel would exceed the bar **"by at least two orders"**;
  measured **56×**, which is 1.75 orders. **WRONG**, and in the direction that matters least but
  is still wrong: I over-estimated how badly a truncated minimax degrades. (The control as built
  keeps three coefficients, so `P(r)` is degree 2 and `e^r` is degree 4 overall, against the good
  kernel's degree 7 — §4 described it as "truncated to degree 3", which is the same operation
  counted the other way round. The change is in the counting, not in the control.)

## A.2 `G-E53b` as written cannot be answered by anything, so it is MALFORMED

§4 says *"max relative error … ≤ 1e-6"* over `x ∈ [−104, 89]`. At the bottom of that interval
the answer is a denormal or nothing at all:

    exp(-104)                        = 6.813557e-46
    the two representable neighbours = +0  and  1.401298e-45
    relative error of +0             = 1.0000
    relative error of 1 denormal ulp = 1.0566

**Every float that could be returned there is at least 100% off.** The reference itself —
`(float)exp((double)x)`, i.e. the function being replaced — returns `+0` and is 100% off. A bar
no implementation can meet, the incumbent included, is not a bar the treatment failed; it is a
bar that was never answerable. **E4's precedent applies: MALFORMED, not failed, and it may be
re-specified.**

I want to be exact about the order of events, because this is the move I keep warning myself
about: **I noticed this because the first sweep failed.** The re-specification is therefore
post-hoc with respect to the kernel measurement, and the only thing that makes it legitimate is
that the replacement is **strictly tighter**:

### `G-E53b2`, registered

> **Max 1 ulp against `(float)exp((double)x)`, everywhere in `[−104, 89]`, normal and denormal
> alike, plus exact identity at zero, at both infinities and at NaN.**

One uniform criterion, no domain split, no threshold of mine anywhere in it. On normals 1 ulp is
`1.19e-07` relative, so **`G-E53b2` implies the registered `1e-6` with eight times the margin**.
The planted control must still fire against it and does, at **661 ulp**.

If I had loosened anything, this paragraph would say so. What was replaced was unanswerable;
what replaced it is eight times stricter.

## A.3 Four defects the self-test found in my own kernel, recorded because each was silent

The kernel failed its own test three times before passing. Not one of these would have shown up
as a crash, and all four produce plausible numbers:

1. **`r = x·log2e − n` loses accuracy at large `|x|`.** Single-precision scaling puts ~6e-8
   relative into `y`, which at `y = 127` is 7.6e-6 *absolute*, and `2^y` turns that back into
   5e-6 relative. Read **30 ulp at x = 88**. Fixed with Cody-Waite reduction. *The softmax never
   visits that end of the range; the FFN does.*
2. **`(n+127)<<23` in one step breaks for `n < −126`** — the entire denormal band.
3. **`p·(s1·s2)` overflows and underflows where `(p·s1)·s2` does not** — returned `+inf` at
   **1883 points where `exp` is perfectly finite**.
4. **Clamping `n` instead of `x` leaves `r` unreduced**: `exp(−inf)` came back as **`−inf`**.

Recorded in the header of `vexpf8.h` next to the lines that fix them.

**§2's description of the kernel is superseded by this list**, and the shipped kernel is what
`vexpf8.h` says it is: Cody-Waite reduction against `ln2` split in two, a Cephes degree-5
minimax for `e^r` (not an `exp2` minimax for `2^r`), `x` clamped to `[−110, 95]` instead of a
flush at −103, and no denormal shortcut — the denormal band is computed, and correctly. §2 was
an outline written before the kernel existed and three of its four steps did not survive
contact with the self-test. That is what the self-test is for; it is recorded rather than
quietly edited into §2.

## A.4 `c0` — a regression control the brief did not ask for, registered now

The brief's design forced a refactor: the softmax loop and the three SwiGLU loops now go through
`sm_exp_pass` and `swiglu`. **A refactor that silently moved the libm arm would make every later
comparison meaningless**, and it would look like a plausible number rather than an error.

> **`c0`: `donor_engine_e53.exe --fexp libm` must reproduce `NATS_TOTAL 124963.9517608703` — the
> published pre-E53 value — to every digit.**

Compared against the *published* number, not one measured beside it in the same session, so the
control cannot drift with me. Anything but bit-identity stops the experiment.

## A.5 `G-E53e` — a drift witness, because E51's speed phase failed without one

E51's speed phase resolved **one window in five**: both arms stepped down together at the same
repetition under sustained load and stayed down. E52 then showed the fix works — rotation plus a
bracketing witness gave a **CLEAN** run at 2.09%.

Registered for `G-E53d`, all three together:

1. **The arm order alternates each repetition** (`libm,poly` on even reps, `poly,libm` on odd),
   so arm and time are not the same axis.
2. **An `L`-free drift witness**: one `libm` cell at `n = 160` before the sweep and one after.
3. **`G-E53e`**: if the two differ by more than **2.3%** — E52's `G-E52b` constant, which is
   E44 run 2's measured zero-load dispersion — the run is **DRIFT-CONTAMINATED**, the cells are
   printed, and **the fit is not read as a difference between arms**.

`REPS = 5`, the brief's "≥ 5 reps", kept as written.

## A.6 What `G-E53d` inherits from E52

`OCC_BAR = 4.39`, foreign occupancy per cell via `GetProcessTimes`, and any cell above the bar
recorded and **not citable**. This is the first experiment to run under the derived bar.

## A.7 The runner

`e53_fexp.py`, **19 of 19 self-tests fire**, including three that plant a *dead control* (a
parity control that does not separate, a greedy control that matches, a one-window fit) and one
that checks the fit recovers a planted slope and intercept exactly.

## A.8 The attribution instrument, changed before it runs

§4 promised the site attribution would come from the `smrep` probe and *"a matching `silurep`
probe"*. **No `silurep` probe was built, and none should be.** The engine already times the
SwiGLU glue directly, as the organ `glue(silu)` (`donor_engine.c:61`, printed at `:1627`). A
repetition probe would be a second and strictly weaker instrument measuring what a direct timer
already measures — weaker because it infers a cost from a difference of two runs rather than
reading it, and E51 showed exactly how that inference dies when the two runs drift apart.

So: **the FFN site is read from `glue(silu)` directly; the attention site keeps E51's `sm2−sm1`
difference**, because there is no direct timer for the softmax pass alone. The two sides are
therefore measured by instruments of different strength, and the weaker one is named as such
wherever its number appears.

Both are **reported with their own dispersion and carry no verdict** — §4 already said
attribution is reported, not gated, and E14 §6 forbids promoting it afterwards.

---

# ADDENDUM B — QUALITY PASSES EVERYWHERE; THE SPEED PHASE IS **NOT CITABLE** AND SAYS SO ON ITS OWN

## B.1 The quality gates, all four

| gate | verdict | number |
|---|---|---|
| `G-E53a` kernel control | **FIRES** | degree-2 kernel 5.5718e-05 / **661 ulp**, 56× the bar |
| `G-E53b2` kernel | **PASS** | **1 ulp everywhere**; 1.1920e-07 relative on normals; denormals 1 ulp; zero, both infinities, NaN and every special case exact |
| `c0` refactor regression | **IDENTICAL** | `NATS_TOTAL 124963.9517608703`, the published pre-E53 value, **every digit** |
| `G-E53c1` continuous parity | **PARITY HOLDS** | poly `124963.9588801368` vs libm `124963.9517608703` = **5.697e-08** relative over 12,264 positions; k=3 control fires at **1.593e-01** |
| `G-E53c2` discrete parity | **PASS** | **A1 160/160**; planted controls **3/160** and **10/160**, reproducing E6 to the token; `G-D` **15 of 15** |

**`G-E53c2` is the one that could have broken.** A different approximation, summed in a
different order, reproduces this engine's greedy output token for token at 1.5B. It was scored
against E6's stored HuggingFace reference and not against my own libm arm, which would have
proved nothing.

`c1` and `c2` exercise **both call sites at once**, so these are joint passes and a failure would
not have localised. §4 registered the bisect in advance for that case; it is not needed.

## B.2 The speed phase: three gates, three refusals

    drift witness opens  libm n=160  115.36 tok/s  foreign  8.4%
    drift witness closes libm n=160  122.10 tok/s  foreign  6.6%

| gate | verdict | |
|---|---|---|
| `G-E53e` drift | **DRIFT-CONTAMINATED** | the bracketing cells differ by **5.68%**, above the registered 2.3% |
| occupancy | **50 of 50 cells above the bar** | foreign ran **4.9% to 23.8%** against `OCC_BAR = 4.39` |
| `G-E53d` | **C50 NOT COMPUTABLE** | **1 of 5** windows survived the 6% dispersion rule |

**Nothing in the speed phase is citable and no rate from it is quoted anywhere.**

**Why the box was loaded, measured after the run rather than guessed.** Sampling per-process CPU
over 8 seconds with no engine running: **5.6% of the box**, of which **Chrome alone is ~4.2%**
(three processes) and Task Manager 0.8%. **The machine as it normally sits cannot meet the 4.39%
bar.** E52 addendum A.6 predicted exactly this and I scored that prediction **WRONG at the
margin** because E52's own `L=0` cells happened to read 4.4%. A.6 was right and my scoring of it
was wrong; that correction is made here and in E52's record.

This is the derived bar doing precisely what it was derived to do, on the first experiment to run
under it. **It is not re-run to a pass** (E40 addendum A). It is re-run on a quiet box, and that
needs the user — the request is in `COMMUNICATION.md`.

## B.3 Two diagnostics, labelled, carrying no verdict

### B.3.1 The attribution — the softmax exponential essentially disappears

| | libm | poly | |
|---|---|---|---|
| attention `S` = `sm2 − sm1` | **1.6530 ms** (range 1.6010–1.7880) | **0.0540 ms** (range 0.0060–0.0960) | **−96.7%** |
| FFN `glue(silu)`, timed directly | 0.0900 ms | 0.0380 ms | −57.8% |
| token at `--bench 1280` | 13.5099 ms | 12.0106 ms | −11.10% |

**A correction to my own runner's reasoning, which printed the wrong caveat.** It said a
difference is not a measurement when an arm's reps disperse by more than the difference, and
pointed at poly's **166.7%** spread on `S`. That is the wrong comparison: 166.7% of 0.054 ms is
**0.09 ms**, and the difference is **1.599 ms** — seventeen times larger. *A percentage of a
near-zero quantity is not a dispersion you can compare to anything.* The two ranges do not come
within an order of magnitude of touching. The runner now prints `S` ranges in milliseconds.

**Unarranged corroboration.** `S` on the libm arm reads **1.6530 ms** at mean position 640 here,
against E51's **1.6890 ms** on the e50 build — **−2.1%**, a different build, a different session
and a different runner. E51's central number reproduces.

**A projection, and it is a projection and not a result.** If the whole of that `S` difference is
slope, `b` would go `0.008144 → 0.005646 ms/pos` and `C50` `1443 → 2082`. It is written here only
because leaving it out would look like hiding it. It rests on a contaminated session and on the
weaker of the two attribution instruments (§A.8), and **the registered gate refused to compute
`C50` at all.**

### B.3.2 The arms do not overlap where the mechanism says they should not

Five reps per arm per window, arm order alternating each repetition:

| n | mean pos | libm reps | poly reps | disjoint? |
|---|---|---|---|---|
| 40 | 20 | 108.0 110.0 127.0 127.4 129.3 | 111.7 121.7 122.9 128.2 131.8 | no |
| 160 | 80 | 116.3 118.3 118.8 119.2 120.4 | 102.2 121.8 122.0 122.6 127.0 | no |
| 320 | 160 | 96.8 97.4 106.9 108.4 108.7 | 111.0 111.5 112.8 115.0 115.3 | **yes**, gap +2.18% |
| 640 | 320 | 90.0 90.7 90.7 90.7 94.1 | 96.7 97.4 97.4 98.3 99.2 | **yes**, gap +2.76% |
| 1280 | 640 | 70.5 72.6 73.4 74.7 75.5 | 84.6 84.7 86.1 86.3 86.5 | **yes**, gap +12.01% |

Separation appears only at long context and grows monotonically with it, and is absent at the
two short windows — which is the signature of a **slope** change and is what §1 said the
mechanism would produce. It is not an ordering artefact: libm goes first in reps 1, 3, 5 and
second in 2, 4, and poly is high in both positions.

**This is not scored, and run 1 is not re-scored by it.** The statistic was noticed after seeing
the cells, and E52's precedent is the one that applies: run 1 stays as the record of gates that
fired, and the successor gate judges a fresh run.

## B.4 `G-E53f`, registered now, before the re-run

E14 §3 requires every SCORE metric to have a RANK partner. `G-E53d` is a SCORE — it compares two
medians and refuses when the reps disperse, because a wandering level can fake a ratio. It has
never had a partner, and B.3.2 is what the missing partner would have looked at.

> **`G-E53f`** — for each window, do the two arms' repetition sets **overlap**? Complete
> separation of `k` against `k` is non-parametric: it does not care how wide either arm is, only
> that they do not meet. Under exchangeability the chance of all `k` of one arm beating all `k`
> of the other, either way round, is `2 / C(2k, k)` — at `k = 5`, **0.0079** per window.
> **Windows separating in BOTH directions is `INCONSISTENT` and yields nothing**: that is the
> level wandering, not an effect.
>
> **It answers *which* arm is faster and never *by how much*.** The how-much stays `G-E53d`'s,
> and `G-E53d` may still refuse it.

Eight planted controls, all firing: a fully separated window, the exact `2/252`, **one**
overlapping repetition killing the separation, the reverse direction, both directions at once
returning `INCONSISTENT`, identical arms returning nothing, too few reps taking no verdict, and a
hair's separation still counting because it is a rank test. The runner is now **27 of 27**.

## B.5 The scorecard so far — 3 right, 3 wrong, and the speed row is still blank

| prediction | measured | |
|---|---|---|
| `G-E53a` control exceeds the bar "by at least two orders" | 56× = 1.75 orders | **WRONG** |
| `G-E53b` max rel err 1e-7 to 5e-7 | 1.1920e-07 | **right** |
| `G-E53c1` PARITY HOLDS | holds | **right** |
| `G-E53c1` lands at 1e-6 to 1e-5 | **5.697e-08** | **WRONG**, eighteen times tighter |
| `G-E53c2` 160/160 holds | 160/160 | **right** |
| fraction of `S` recovered, 50–75% | diagnostic says 96.7% | **not scored** — no citable run |
| `b`, `C50`, the short-context headline | — | **not scored** |

Both misses are magnitude, in opposite directions, and both come from the same habit: guessing
the size of something I had a way to compute. The `G-E53c1` miss in particular — I did not
account for the softmax **normalising**, which divides most of a 1-ulp kernel error straight back
out.

## B.6 What is owed

1. **A quiet box, and a re-run of `d` and `attrib` only.** `c0`, `c1` and `c2` are quality gates,
   deterministic, and are not re-run. Cost: about 50 minutes hands-off.
2. The re-run is judged by `G-E53d` **and** `G-E53f`, both registered, with the occupancy bar and
   the drift witness unchanged.
3. If the box still cannot hold 4.39%, that is a finding about the machine and not about the
   kernel, and it is reported as one.
