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
