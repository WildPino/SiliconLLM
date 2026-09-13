# BRIEF E51 — THE SOFTMAX WAS PAYING FOR `errno`

**Registered before any E51 cell exists.** No speed number here may be quoted until `G-E51a`
(BPB parity) and `G-E51b` (the discrete greedy test) both pass. E50 is the precedent for why both:
a scalar average over 12,264 positions does not compose to a discrete argmax.

## 1. The finding, read out of the compiler and not assumed

The baseline softmax is three lines of `donor_engine.c`:

    float sum=0.0f;
    for(int t=0;t<=pos;t++){ a[t]=expf(a[t]-mx); sum+=a[t]; }

Compiled at the project's own flags (`clang -O3 -mavx2 -mfma -ffp-contract=on -fopenmp`), that
inner loop is **eleven instructions per element**, `.LBB12_145`:

    vmovss   4(%r14,%rsi,4), %xmm0      # load a[t]
    vsubss   %xmm10, %xmm0, %xmm0       # - mx
    vcvtss2sd %xmm0, %xmm0, %xmm0       # float -> DOUBLE
    vzeroupper                          # AVX state flush, PER ELEMENT
    callq    exp                        # libm, DOUBLE precision, PER ELEMENT
    vcvtsd2ss %xmm0, %xmm0, %xmm0       # double -> float
    vmovss   %xmm0, 4(%r14,%rsi,4)      # store a[t]
    vaddss   %xmm0, %xmm11, %xmm11      # sum += ...  (serial dependency chain)
    movslq   (%rbx), %r15               # RELOAD the loop bound FROM MEMORY, per element
    incq %rsi / cmpq %r15, %rsi / jl

Four separate costs, none of them the exponential itself:

1. **`expf` is promoted to double `exp`.** The source says `expf`; the object calls `exp`, with a
   `vcvtss2sd` in and a `vcvtsd2ss` out.
2. **A `vzeroupper` per element**, because the call may clobber the upper YMM state.
3. **The loop bound is reloaded from memory on every iteration.**
4. **The sum is a serial `vaddss` chain** — the same shape E8 found in the weight matvec.

### 1.1 Why, exactly — and it is one cause, not four

**`exp` is specified to set `errno`.** That makes it a memory-writing call, so the compiler may
not treat it as pure: it cannot keep the loop bound in a register across it, cannot unroll, cannot
vectorise, and must assume the AVX state is clobbered. **One promise nobody in this programme has
ever wanted is buying all four costs.**

Recompiling with **`-fno-math-errno`** and nothing else, the same loop becomes:

    vmovss   -12(%r12,%r14,4), %xmm0
    vmovss   -8(%r12,%r14,4), %xmm12
    vsubss   %xmm10, %xmm0, %xmm0
    vzeroupper
    callq    expf                        # SINGLE precision -- both vcvt gone
    vaddss   %xmm0, %xmm11, %xmm11
    ... three more expf, unrolled 4x ...
    addq     $4, %r14
    cmpq     %r14, 336(%rsp)             # bound HOISTED, compared once per FOUR elements
    jne      .LBB12_154

`exp` → **`expf`**, the two conversions gone, the loop **unrolled 4×**, the bound **hoisted**, and
**one `vzeroupper` per four elements** instead of per element. The FFN's SiLU
(`x/(1+expf(-x))`, `.LBB15_3`) has the identical defect and is fixed by the same flag.

## 2. This is NOT `-ffast-math`, and the difference is the whole point

Phase 35 forbids `-ffast-math` and that law is not being bent. `-fno-math-errno` says exactly one
thing: **math functions do not have to set `errno`.** It does not enable reassociation, does not
enable flush-to-zero or denormals-are-zero, does not permit reciprocal or rsqrt substitution, does
not change `-ffp-contract=on`, and does not license any algebraic rewrite. Every arithmetic
operation the engine performs is still performed in the same order with the same rounding.

`-ffast-math` implies `-fno-math-errno` **plus** all of the above. E51 takes the one component
that is a side-effect promise and leaves every component that is a numerics promise.

## 3. What DOES change numerically, stated up front

**`expf(x)` and `(float)exp((double)x)` are different functions** and may differ in the last ulp.
So this change is **not bit-exact** and must not be gated as if it were. Nothing else changes: the
subtraction, the store, the accumulation order and the division are untouched.

That is the same situation E49 was in with `avx4`, and it gets the same treatment — plus E50's
addition, because E50 is the experiment that showed a scalar BPB parity does not settle a discrete
claim.

## 4. The gates

### `G-E51a` — BPB parity, with a planted control that must fire

`--bpb` on S15 (`e37_carved_nf.bin`, real weights, real corpus) at `--carve-k 256`, both builds.

* Bar **`|ΔBPB| < 1e-4`**, i.e. `2e-5` relative on `NATS_TOTAL` (the mapping E49 §4 wrote down).
* **PLANTED CONTROL, and the null does not count without it**: the same harness, unchanged, must
  FIRE on `--carve-k 3` against `--carve-k 256` on the same artifact — E37 measured those
  0.5537 BPB apart, **5,537× the bar**. This is E49 addendum A's control, reused because it is
  the one in this programme proven capable of firing.

### `G-E51b` — the discrete test, because E50 showed parity does not imply it

E6's engine stage, re-run on the new build, scored against E6's **frozen** `results/e6/ref.json`.

* **A1 (fp32 0.5B) must read 160/160.** Anything less and the flag does not ship.
* Controls **A2 at 3/160 and A3 at 10/160**, each within **±2**, or the gate is VOID rather than
  passed — the scorer must be shown to see disagreement.
* Results to `results/e51/`; **E6's own results are not overwritten.**

### `G-E51c` — the speed, judged ONLY where the dispersion permits

**This gate is written under E49 addendum C's rule and is the first to implement it.** Windows
`{40, 160, 640, 1280, 2560}`, 5 reps, both builds interleaved at rep level, `e40_r128.bin
--carve-k 3`.

* **A window is judged only if the observed spread of BOTH arms is at or below the gate's own
  tolerance.** E49 measured 2.3–5.9% at `{640, 1280, 2560}` and 7.9–26.1% at `{40, 160}`.
* **The runner must print `UNRESOLVABLE` instead of a verdict** for any window whose measured
  spread exceeds the tolerance, and must do so from the data of *this* run, not from E49's.
* `b` is refit per build and `C50` recomputed. Absolutes carry the standing ±5%; ratios do not.

### `G-E51d` — attribution: is the win in the softmax, or somewhere else?

A faster engine is not evidence that the softmax got faster. Under `--profile`, with `--attnr sm1`
and `sm2` on both builds, solve S as `(sm2 − sm1)` the way E48 did and compare S between builds.

* **S must fall**, and the drop must account for the bulk of any total improvement.
* **If the total improves while S does not fall, the story in §1 is wrong** even if the number is
  good — the win would then be the FFN's SiLU or something unnamed, and E51 reports that instead.

## 5. My prediction, registered

| quantity | prediction |
|---|---|
| `G-E51a` | **passes**, `ΔBPB` below 1e-5 — a last-ulp difference in one elementwise function |
| `G-E51a` control | fires, ~0.55 BPB |
| `G-E51b` A1 | **160/160 survives.** The fp32 0.5B's top-2 gaps are order 0.1–1.5 nats; a last-ulp move in softmax should not flip an argmax |
| `G-E51c` | `b` falls **10–30%**; `C50` rises from 1443 to **1600–2050** |
| `G-E51c` small windows | **`UNRESOLVABLE` at 40 and 160**, by the rule, not by my choosing |
| `G-E51d` | S falls and accounts for **most** of the total improvement |

The `b` range is wide on purpose. E48 put S at **33.1%** of the `avx4` attention organ, but that
is a SCORE with no RANK partner (`G-E48d` is VOID), the flag removes overhead rather than the
exponential itself, and it helps the FFN's SiLU too — which is a *constant* cost, not a context
cost, so it would move the intercept and not `b`. **A drop in the intercept with `b` unchanged is
a perfectly possible outcome and would still be a win, just a different one.**

## 6. What E51 does not claim

It does not vectorise the softmax — an explicit polynomial `exp2` kernel is a larger,
non-value-preserving change and belongs to its own experiment, which this one is partly meant to
price. It does not touch `-ffast-math`, reassociation or the accumulation order. It does not
revisit `C50` for any shape but R128 `--carve-k 3`. And it does not re-open the headline at short
context: E49 addendum C established that this instrument cannot resolve that window, and E51
inherits that limit rather than pretending otherwise.

---

# ADDENDUM A — THE FLAG IS A NO-OP, AND THE REASON IS IN THE LIBM, NOT IN THE COMPILER

**Run 1, 2026-09-13.** `G-E51a` and `G-E51b` pass. `G-E51c` returns **NO FIT** and the
treatment is **not separable from the baseline at the one window the dispersion rule allows
judging**. `G-E51d` did not run — the runner crashed before it. Every number below is run 1's;
no cell was re-measured.

## A.1 `G-E51a` — PARITY HOLDS, and more exactly than the gate asked

| | `NATS_TOTAL` |
|---|---|
| base (`donor_engine_e50.exe`), `--carve-k 256` | `124963.9517608703` |
| e51 (`donor_engine_e51.exe`), `--carve-k 256` | `124963.9517608703` |
| planted control, `--carve-k 3` | `144871.9226166338` |

* **planted control FIRES**: `1.593e-01` relative against a `1e-03` bar, **159x the bar**. The
  null does not count without this, and it did not have to.
* **pair: `0.000e+00` relative.** Not "inside the bar" — **identical to every digit** over
  12,264 predicted positions.

Brief §3 said `expf(x)` and `(float)exp((double)x)` are different functions, that the change is
therefore not bit-exact, and that it must not be gated as if it were. That was the right stance
to register in advance, and A.2 explains why the outcome beat it: **on this toolchain they are
the same function.** The gate was built to tolerate a difference that does not exist here.

**One honesty note on the units.** The `|dBPB|` figures this programme derives from `NATS_TOTAL`
— E49's `|dBPB| <= 8.47e-07` included — are obtained by multiplying the relative agreement by an
assumed **5.0 BPB** for this artifact. `donor_engine.c:1625` prints `NATS_TOTAL`, `N_PREDICTED`
and `NATS_PER_TOKEN` and **no byte total**, so BPB is not computable from any run this
programme has stored; solving E49's recorded pair backwards returns `BPB = 4.999999`, which is
the constant it was multiplied by. Nothing is withdrawn — the load-bearing quantity is the
*relative* agreement, which is scale-free, and here it is exactly zero — but the BPB form of
those bounds carries an assumption that has never been measured in the same run. **Owed: the
scored byte total belongs in the sidecar of the ids file, so the conversion stops being an
assumption.**

## A.2 THE FINDING — `-fno-math-errno` moved the cost, it did not remove it

The binaries differ and the flag took effect. `objdump -d`, call sites of the exponential:

| build | `callq <exp>` | `callq <expf>` |
|---|---|---|
| `donor_engine_e50.exe` (base) | **8** | 0 |
| `donor_engine_e51.exe` | 6 | **49** (the 4x unroll, at every site) |

So brief §1's reading of the caller was correct: the two `vcvt`s, the per-element
`vzeroupper` and the reloaded loop bound are gone from the loop body, and the loop is unrolled.
Then the callee:

    0000000140013fb0 <expf>:
      subq      $0x28, %rsp
      cvtss2sd  %xmm0, %xmm0       # float -> DOUBLE
      callq     <exp>              # THE SAME DOUBLE ROUTINE
      cvtsd2ss  %xmm0, %xmm0       # double -> float
      addq      $0x28, %rsp
      retq

**`expf` on this toolchain is `(float)exp((double)x)`, written out.** The flag did not remove
the promotion to double; it **relocated it from the caller into the callee**, and added a call
frame on top. Three of the four costs §1.1 attributed to `errno` were never removed, only moved
across a boundary — and the fourth, worth a handful of instructions against a memory-bound
token, is invisible.

This also explains A.1 exactly: if `expf(x)` *is* `(float)exp((double)x)`, the two builds
compute the same values in the same order, so `0.000e+00` is not luck, it is the identity.

**The law, and it is a cousin of two we already pay for.** Phase 61 says a compute-bound
microbench does not compose to a memory-bound engine. The byte-convention law says a ratio whose
numerator and denominator are counted at different boundaries is not a ratio. This is the same
family: *an improvement read on one side of a call can be an accounting move. The cost is not
the caller's assembly; it is the caller's plus the callee's.* Reading one side of the boundary
and publishing a mechanism is how §1.1 came to be written.

## A.3 `G-E51c` — the runner crashed, and the fix is the point

`fit()` raised `degenerate fit -- every window at the same position. STOP.` **One** window
survived the dispersion rule, a straight line needs two, and the gate printed its per-window
table and then **no verdict at all** — which is worse than any verdict it could have printed.
`G-E51d` never ran.

The decision function now handles it, and three new self-tests plant the three cases
(`UNRESOLVABLE THROUGHOUT`, `NO FIT -- ONE RESOLVABLE WINDOW`, and the computable case);
**20 of 20 checks fire.** The verdict below is produced by `e51c_rescore.py` re-reading run 1's
own 50 cells out of `e51_run1.log`. **No cell was re-measured** — the same treatment E50 B.8
gave its own crash.

| n | base | e51 | delta | spread base | spread e51 | verdict |
|---|---|---|---|---|---|---|
| 40 | 124.07 | 125.96 | +1.5% | 10.1% | 12.0% | **UNRESOLVABLE** |
| **160** | **117.89** | **117.67** | **-0.2%** | 5.6% | 3.0% | **JUDGED — NOT separable** |
| 640 | 97.44 | 96.27 | -1.2% | 9.7% | 7.6% | **UNRESOLVABLE** |
| 1280 | 81.67 | 81.54 | -0.2% | 10.4% | 8.7% | **UNRESOLVABLE** |
| 2560 | 55.56 | 55.38 | -0.3% | 12.4% | 11.8% | **UNRESOLVABLE** |

**`G-E51c` : NO FIT — ONE RESOLVABLE WINDOW.** `C50` is not computable and none is published.
At the single window the rule permits judging, the two builds are **not separable**.

## A.4 A diagnostic that is NOT a gate: the dispersion here is a shared drift

The registered gates are decided in A.3 and nothing below may change them — **a metric invented
after seeing the data may not promote a result** (E14 §6). It is recorded because it is about
the instrument, not about the flag.

The two builds were interleaved at rep level, so each repetition is a pair. The **paired** ratio:

| n | e51/base, rep by rep | median |
|---|---|---|
| 40 | +5.7% -10.3% +8.9% +2.4% -0.0% | +2.4% |
| 160 | +0.2% +0.5% -2.0% -0.6% -0.2% | -0.2% |
| 640 | -2.5% +2.4% -1.5% +0.7% -0.1% | -0.1% |
| 1280 | -0.2% -0.2% -1.5% +0.1% +0.4% | -0.2% |
| 2560 | +0.5% -0.5% +0.0% -0.4% +0.2% | +0.0% |

And the **levels** the spread test actually saw, in time order:

| n | base | e51 |
|---|---|---|
| 1280 | 81.67 82.91 **84.33 -> 75.89 75.85** | 81.54 82.78 **83.09 -> 75.96 76.15** |
| 2560 | 60.21 62.03 **-> 55.16 55.56 55.26** | 60.51 61.69 **-> 55.17 55.32 55.38** |

**Both arms step down together, at the same repetition, and stay down.** The long windows — the
ones that load six AVX2 threads hardest and longest — drop ~8-10% part-way through a 35-minute
sustained run and do not recover. The short windows do not. That is the signature of a power or
thermal limit being reached, not of per-measurement noise: it is a **level shift shared by both
arms**, which is exactly what pairing removes and what a per-arm min/max spread cannot see.

**What it suggests, for a future experiment and not for this one:** E43's "9-22% dispersion on
an idle machine" may be substantially a drifting *level* rather than i.i.d. noise; if so, a
paired interleaved design can resolve differences far below that band while an unpaired one
cannot — and the two arms' absolute numbers would still, correctly, refuse to be points.
**This must be registered as its own gate before it is used to judge anything**, with its own
planted control, because promoting it here would be the `G-E49c` mistake with the sign flipped.

## A.5 The scorecard

| §5 prediction | outcome |
|---|---|
| `G-E51a` passes, `dBPB` below 1e-5 | **RIGHT**, and by more than asked: `0.000e+00` relative |
| `G-E51a` control fires, ~0.55 BPB | **fires** — `1.593e-01` relative, 159x its bar. The *magnitude* is **not scored**: converting to BPB needs a byte total no run recorded (A.1) |
| `G-E51b` A1 160/160 survives | **RIGHT**, 160/160, controls 3/160 and 10/160 to the token |
| `G-E51c`: `b` falls 10-30%, `C50` 1443 -> 1600-2050 | **WRONG.** No window shows a fall; the one judgeable window reads -0.2% and is not separable. No `C50` is computable |
| `G-E51c`: `UNRESOLVABLE` at 40 and 160 | **HALF RIGHT, and for the wrong reason.** 40 is unresolvable; 160 is the *only* window that resolved, and 640/1280/2560 — which E49 measured at 2.3-5.9% — did not, because of A.4 |
| `G-E51d`: S falls and accounts for most of the improvement | **NOT RUN** (the crash). There is no improvement to attribute |

**3 right, 1 wrong, 1 half, 1 not run, 1 unscorable.** The wrong one is the headline prediction,
and §1.1's mechanism is the reason it was wrong.

## A.6 What E51 bought, since its treatment is null

It was written partly to **price** the experiment §6 defers — a real vectorised polynomial
`exp2` softmax. It priced it, and upward: the engine is still executing the **entire
double-precision `exp` routine, once per element**, in both builds. Nothing on the E51 axis
removes that; only replacing the routine does. `G-E51d` is what says whether that is worth
doing, and it is now the single owed measurement of this experiment.

It also bought a third independent reproduction of `NATS_TOTAL 124963.9517608703` — E49, E50 and
E51, three sessions, two binaries, every digit — and a fourth reproduction of E6's discrete
controls at 3/160 and 10/160.

## A.7 Owed

1. **`G-E51d`**, which never ran. Its registered logic will return `NO IMPROVEMENT` because the
   total did not improve; the value is `S` itself, which prices the deferred experiment.
2. The **scored byte total** in the ids sidecar, so `|dBPB|` stops resting on an assumed 5.0.
3. The **paired-interleaved dispersion rule** of A.4, registered as its own gate with its own
   planted control, before it judges anything.
